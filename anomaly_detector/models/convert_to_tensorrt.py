#!/usr/bin/env python3
"""
TensorRT Engine Conversion Script for Jetson Orin NX
=====================================================

Converts YOLOv8 .pt models to TensorRT .engine files for 5-10x inference speedup.

Usage:
    python convert_to_tensorrt.py --models-dir /path/to/models
    python convert_to_tensorrt.py --models-dir /path/to/models --imgsz 640 --half
    python convert_to_tensorrt.py --check       # Check environment only

Prerequisites:
    - Jetson device with JetPack SDK
    - TensorRT installed (comes with JetPack)
    - ultralytics package installed in conda env
    - YOLO .pt model files in the models directory

Expected files in models-dir:
    - helmet_best.pt   -> helmet_best.engine
    - smoking_best.pt  -> smoking_best.engine

Optional:
    - If you have a combined safety model, name it safety_combined.pt
      and it will be converted to safety_combined.engine
"""

import argparse
import sys
import os
from pathlib import Path

# Add conda env path
sys.path.append('/home/jetson/anaconda3/envs/test/lib/python3.10/site-packages/')


def check_environment():
    """Check available AI acceleration providers."""
    print("\n" + "=" * 60)
    print("  Environment Check")
    print("=" * 60)

    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers: {providers}")

        has_cuda = "CUDAExecutionProvider" in providers
        has_trt = "TensorrtExecutionProvider" in providers

        if has_trt:
            print("  [OK] TensorRT Execution Provider — best performance")
        elif has_cuda:
            print("  [OK] CUDA Execution Provider — good performance")
            print("  [TIP] For TensorRT EP, install onnxruntime-gpu with TensorRT support")
        else:
            print("  [WARN] No GPU provider!")
            print("         Install: pip install onnxruntime-gpu")
    except ImportError:
        print("  [MISS] onnxruntime not installed")

    # Ultralytics
    try:
        import ultralytics
        print(f"  [OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  [MISS] ultralytics not installed")

    # PyTorch & CUDA
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        print(f"       CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"       CUDA device: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"       Compute capability: {cap[0]}.{cap[1]}")
    except ImportError:
        print("  [MISS] PyTorch not installed")

    # TensorRT
    try:
        import tensorrt as trt
        print(f"  [OK] TensorRT {trt.__version__}")
    except ImportError:
        print("  [WARN] TensorRT Python module not found")
        print("         (JetPack includes it, but check: sudo apt install python3-libnvinfer)")

    # Jetson info
    try:
        import subprocess
        result = subprocess.run(
            ["cat", "/etc/nv_tegra_release"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            release = result.stdout.strip().split("\n")[0]
            print(f"  [OK] JetPack: {release}")
    except Exception:
        pass

    # GPU memory
    try:
        import subprocess
        result = subprocess.run(
            ["tegrstats", "--memory"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  [OK] Memory: {result.stdout.strip()[:100]}")
    except Exception:
        try:
            import subprocess
            result = subprocess.run(
                ["free", "-h"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  Memory:\n{result.stdout.strip()}")
        except Exception:
            pass

    print("=" * 60 + "\n")


def convert_yolo_model(pt_path: str, imgsz: int = 640, half: bool = True):
    """Convert a single YOLO .pt model to TensorRT .engine."""
    from ultralytics import YOLO

    pt_path = Path(pt_path)
    engine_path = pt_path.with_suffix(".engine")

    if engine_path.exists():
        print(f"  [SKIP] {engine_path.name} already exists")
        return str(engine_path)

    if not pt_path.exists():
        print(f"  [SKIP] {pt_path.name} not found")
        return None

    print(f"  Converting: {pt_path.name}")
    print(f"    imgsz={imgsz}, half={half}, device=0")
    print(f"    This may take 2-5 minutes...")

    model = YOLO(str(pt_path))

    # Export to TensorRT engine
    engine_path_str = model.export(
        format="engine",
        device=0,
        half=half,
        imgsz=imgsz,
        simplify=True,
    )

    print(f"  [DONE] Created: {Path(engine_path_str).name}")
    return engine_path_str


def convert_onnx_to_trt(onnx_path: str, fp16: bool = True):
    """Convert an ONNX model to TensorRT engine using trtexec.

    This is an alternative to onnxruntime's TensorRTExecutionProvider.
    Use this if you want a standalone .engine file for the fire/smoke model.
    """
    import subprocess

    onnx_path = Path(onnx_path)
    engine_path = onnx_path.with_suffix(".engine")

    if engine_path.exists():
        print(f"  [SKIP] {engine_path.name} already exists")
        return str(engine_path)

    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path.name} not found")
        return None

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--useCudaGraph",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  [DONE] Created: {engine_path.name}")
        return str(engine_path)
    else:
        print(f"  [ERROR] trtexec failed:")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert models to TensorRT engines for Jetson Orin NX")
    parser.add_argument("--models-dir", type=str, required=False,
                        help="Path to models directory")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size for YOLO models (default: 640)")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 (use FP32 — slower but more accurate)")
    parser.add_argument("--check", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--convert-onnx", action="store_true",
                        help="Also convert fire/smoke ONNX model via trtexec")
    args = parser.parse_args()

    if args.check:
        check_environment()
        return

    if not args.models_dir:
        print("Error: --models-dir is required (or use --check)")
        parser.print_help()
        sys.exit(1)

    models_dir = Path(args.models_dir)
    half = not args.no_half

    print("\n" + "=" * 60)
    print("  TensorRT Engine Conversion")
    print(f"  Models dir: {models_dir}")
    print(f"  Image size: {args.imgsz}")
    print(f"  FP16: {half}")
    print("=" * 60 + "\n")

    # Convert YOLO models
    yolo_models = [
        "smoking_best.pt", 
        "helmet_best.pt", 
        "fire_smoke_best.pt", # optional combined model
    ]

    for name in yolo_models:
        pt_path = models_dir / name
        if pt_path.exists():
            convert_yolo_model(str(pt_path), imgsz=args.imgsz, half=half)

    # Optionally convert fire/smoke ONNX via trtexec
    if args.convert_onnx:
        onnx_path = models_dir / "weights.onnx"
        if onnx_path.exists():
            print("\n  Converting fire/smoke ONNX model via trtexec...")
            convert_onnx_to_trt(str(onnx_path), fp16=half)
        else:
            print(f"\n  [SKIP] weights.onnx not found in {models_dir}")

    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("  The optimized node will auto-detect .engine files.")
    print("  If conversion fails, the node falls back to .pt files.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
