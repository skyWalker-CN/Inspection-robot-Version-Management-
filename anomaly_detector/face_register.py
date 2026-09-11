#!/usr/bin/env python3
"""
Face Registration Tool for Anomaly Detector
============================================
Standalone script to register face data (multiple angles) into the face database
used by the anomaly detector node.

Each person's data is saved as:
  face_database/<name>/
      face_000.jpg       ← saved image for visual verification
      feature_000.npy    ← 512-dim embedding vector
      face_001.jpg
      feature_001.npy
      ...

Supports registering multiple angles (front face, left profile, right profile,
looking up, looking down) so the detector can recognize the person from any angle.

Usage:
    # Register with name on command line
    python face_register.py shuo

    # Register with custom database path
    python face_register.py shuo --database ~/ros2_ws/src/anomaly_detector/face_database

    # Interactive mode (prompts for name)
    python face_register.py

Controls:
    Press [S]  → Save current face capture (feature + image)
    Press [Q]  → Quit and exit

Tips:
    - Stand at different angles and press S each time
    - Recommended: 5-8 captures per person (front, left, right, up, down, slight turns)
    - Make sure the face bounding box is green (GOOD quality) before saving
    - Keep a neutral expression for best recognition accuracy

Database Structure (shared with anomaly_detector_node):
    ~/ros2_ws/src/anomaly_detector/face_database/
        shuo/
            feature_000.npy
            face_000.jpg
            feature_001.npy
            face_001.jpg
            ...
        zhangsan/
            feature_000.npy
            face_000.jpg
            ...
"""

import sys
import os
import argparse
from pathlib import Path

# ============================================================
# Conda / virtual env path setup
# ============================================================
# The script auto-detects the current conda env from CONDA_PREFIX.
# If running outside conda, you can set SITE_PACKAGES manually:
#   export SITE_PACKAGES=/home/jetson/miniforge3/envs/anomaly_detect/lib/python3.10/site-packages

_SITE_PACKAGES = os.environ.get("SITE_PACKAGES", "")
if _SITE_PACKAGES and os.path.isdir(_SITE_PACKAGES):
    sys.path.insert(0, _SITE_PACKAGES)
else:
    # Try conda env auto-detection
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        sp = os.path.join(conda_prefix, "lib", "python3.10", "site-packages")
        if os.path.isdir(sp):
            sys.path.insert(0, sp)
    if not conda_prefix:
        # Fallback: try common Jetson paths
        for sp in [
            "/home/jetson/miniforge3/envs/anomaly_detect/lib/python3.10/site-packages",
            "/home/jetson/miniforge3/envs/insightface/lib/python3.10/site-packages",
        ]:
            if os.path.isdir(sp):
                sys.path.insert(0, sp)
                break

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except ImportError as e:
    print(f"❌ InsightFace 导入失败: {e}")
    print("   请确保在正确的 conda 环境中运行，或设置 SITE_PACKAGES 环境变量")
    sys.exit(1)


# ============================================================
# Helper: find next available file index
# ============================================================

def _next_index(person_dir: Path) -> int:
    """Scan person_dir for existing feature_*.npy files and return the next index."""
    existing = sorted(person_dir.glob("feature_*.npy"))
    if not existing:
        # Also check for legacy feature.npy (will be counted as index 0)
        legacy = person_dir / "feature.npy"
        if legacy.exists():
            return 1
        return 0
    # Extract the numeric part from the last file
    # feature_005.npy → 005 → 6
    last_name = existing[-1].stem  # e.g. "feature_005"
    try:
        last_idx = int(last_name.split("_")[-1])
    except (ValueError, IndexError):
        last_idx = len(existing) - 1
    return last_idx + 1


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Face Registration Tool for Anomaly Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_register.py shuo
  python face_register.py zhangsan --database /path/to/face_database
  python face_register.py                  # interactive mode
        """
    )
    parser.add_argument("name", nargs="?", default=None,
                        help="Person name to register (will prompt if omitted)")
    parser.add_argument("--database", type=str,
                        default=str(Path.home() / "ros2_ws" / "src" /
                                    "anomaly_detector" / "face_database"),
                        help="Path to face database directory")
    parser.add_argument("--pack", type=str, default="buffalo_l",
                        choices=["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"],
                        help="InsightFace model pack (buffalo_l=most accurate, "
                             "buffalo_s=fastest. Must match anomaly detector setting!)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Inference device (cpu or cuda)")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Detection input size in pixels (default: 640)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")
    args = parser.parse_args()

    # ---- Person name ----
    person_name = args.name
    if not person_name:
        person_name = input("请输入注册人员姓名: ").strip()
    if not person_name:
        print("❌ 错误: 姓名不能为空")
        sys.exit(1)
    # Sanitize: replace spaces/slashes with underscores
    person_name = person_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

    # ---- Paths ----
    db_dir = Path(args.database).expanduser().resolve()
    person_dir = db_dir / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    start_idx = _next_index(person_dir)

    # ---- Startup info ----
    print()
    print("=" * 60)
    print("  人脸注册工具  Face Registration Tool")
    print("=" * 60)
    print(f"  姓名 (Name)      : {person_name}")
    print(f"  数据库目录 (DB)  : {db_dir}")
    print(f"  存储路径 (Path)  : {person_dir}")
    print(f"  模型 (Model)     : {args.pack}")
    print(f"  设备 (Device)    : {args.device}")
    print(f"  已有照片 (Exist) : {start_idx} 张")
    print("=" * 60)
    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │  操作说明                                │")
    print("  │  [S]  保存当前人脸  Save current face   │")
    print("  │  [Q]  退出程序      Quit                │")
    print("  └─────────────────────────────────────────┘")
    print()
    print("  💡 建议: 在不同角度各按 S 保存一张")
    print("      正面 · 左转30° · 右转30° · 低头 · 抬头")
    print("      确保检测框显示 GOOD 时再保存")
    print()

    # ---- Initialize InsightFace ----
    print("⏳ 正在加载 InsightFace 模型...", flush=True)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    ctx_id = 0 if args.device == "cuda" else -1

    try:
        app = FaceAnalysis(name=args.pack, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))
    except Exception as e:
        print(f"❌ InsightFace 初始化失败: {e}")
        sys.exit(1)
    print("✅ 模型加载完成!")
    print()

    # ---- Open camera ----
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 (ID={args.camera})")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ 摄像头已打开")
    print()

    # ---- State ----
    saved_count = start_idx   # total saved (including previous sessions)
    session_saved = 0         # saved in this session only
    face_quality = ""
    last_save_flash = 0       # frame counter for save flash effect

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  摄像头读取失败，重试中...")
                continue

            display = frame.copy()
            faces = app.get(frame)
            face_detected = len(faces) > 0

            # ==== Face quality assessment ====
            face_quality = ""
            q_color = (255, 255, 255)
            if face_detected:
                face = faces[0]  # primary face
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                det_score = float(face.det_score)
                face_w = x2 - x1
                face_h = y2 - y1

                # Quality tiers
                if face_w >= 150 and face_h >= 150 and det_score >= 0.8:
                    face_quality = "✅ GOOD — 推荐保存"
                    q_color = (0, 255, 0)
                elif face_w >= 100 and face_h >= 100 and det_score >= 0.6:
                    face_quality = "⚠️  OK — 可以保存"
                    q_color = (0, 255, 255)
                elif face_w >= 60 and face_h >= 60:
                    face_quality = "🔴 TOO SMALL — 太远了"
                    q_color = (0, 0, 255)
                else:
                    face_quality = "🔴 TOO FAR — 请靠近"
                    q_color = (0, 0, 255)

                # Draw all face boxes
                for i, f in enumerate(faces):
                    fb = f.bbox.astype(int)
                    color = (0, 255, 0) if i == 0 else (0, 255, 255)
                    thick = 2 if i == 0 else 1
                    cv2.rectangle(display, (fb[0], fb[1]), (fb[2], fb[3]), color, thick)
                    # Detection score
                    cv2.putText(display, f"{f.det_score:.2f}",
                                (fb[0], fb[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Show quality text below the primary box
                cv2.putText(display, face_quality,
                            (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, 2)

            # ==== HUD overlay ====
            # Semi-transparent top bar
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 85), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

            cv2.putText(display, f"Name: {person_name}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Saved: {saved_count} total | +{session_saved} this session",
                        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Bottom instruction bar
            bbar_h = 35
            cv2.rectangle(display, (0, display.shape[0] - bbar_h),
                          (display.shape[1], display.shape[0]), (30, 30, 30), -1)
            if face_detected:
                cv2.putText(display, "[S] Save  |  [Q] Quit",
                            (12, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No face detected  |  [Q] Quit",
                            (12, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 2)

            # Save flash effect (brief green flash on full frame after save)
            if last_save_flash > 0:
                flash = display.copy()
                flash[:, :] = (0, 80, 0)
                cv2.addWeighted(flash, 0.3 * (last_save_flash / 15), display,
                                1 - 0.3 * (last_save_flash / 15), 0, display)
                last_save_flash -= 1

            cv2.imshow("Face Registration", display)

            key = cv2.waitKey(1) & 0xFF

            # ==== Key handlers ====
            if key == ord('q') or key == 27:  # Q or ESC
                break

            elif key == ord('s'):
                if len(faces) == 0:
                    print("⏭️  跳过: 未检测到人脸")
                    continue

                face = faces[0]
                embedding = face.embedding
                if embedding is None:
                    print("❌ 错误: 无法提取人脸特征向量 (embedding is None)")
                    continue

                # Check face quality
                fb = face.bbox.astype(int)
                fw, fh = fb[2] - fb[0], fb[3] - fb[1]
                if fw < 60 or fh < 60:
                    print("⏭️  跳过: 人脸太小，请靠近摄像头")
                    continue

                # Save feature
                feat_path = person_dir / f"feature_{saved_count:03d}.npy"
                np.save(str(feat_path), embedding.astype(np.float32))

                # Save face image
                img_path = person_dir / f"face_{saved_count:03d}.jpg"
                cv2.imwrite(str(img_path), frame)

                saved_count += 1
                session_saved += 1
                last_save_flash = 15  # ~0.5s flash at 30fps

                print(f"  💾 #{saved_count - 1:03d}  特征: {feat_path.name}  "
                      f"|  图片: {img_path.name}  "
                      f"|  尺寸: {fw}x{fh}px  "
                      f"|  置信度: {face.det_score:.2f}")

    except KeyboardInterrupt:
        print("\n⏸️  用户中断")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        print()
        print("=" * 60)
        print(f"  注册完成!")
        print(f"  姓名: {person_name}")
        print(f"  本会话新增: {session_saved} 张")
        print(f"  总计: {saved_count} 张")
        print(f"  数据路径: {person_dir}")
        print("=" * 60)
        if session_saved > 0:
            print()
            print("  ✅ 现在可以启动异常检测节点使用这些数据了:")
            print(f"     ros2 run anomaly_detector anomaly_detector_node")
        else:
            print()
            print("  ⚠️  本次没有保存新的人脸数据。")


if __name__ == "__main__":
    main()
