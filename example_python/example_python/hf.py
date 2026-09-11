#!/usr/bin/env python3

import cv2
import numpy as np
import math
import argparse
import os
import time


class LaneDetector:
    def __init__(self, roi_ratio=0.1, input_source="camera", video_source=None, center_tolerance=0.08, max_center_shift=50):
        # 车道检测相关
        self.input_source = input_source
        self.video_source = video_source
        self.cap = None

        # 车道检测控制参数
        self.roi_ratio = roi_ratio  # 底部ROI区域比例，默认0.1表示90%-100%
        self.bottom_lane_width = None
        self.mid_lane_width = None
        self.last_known_center_bottom = None
        self.last_known_center_mid = None
        self.frames_without_both_lanes_bottom = 0  # 底部ROI连续丢失两条车道线的帧数
        self.frames_without_both_lanes_mid = 0  # 中间ROI连续丢失两条车道线的帧数
        self.max_frames_without_lanes = 200  # 连续丢失两条车道线的最大帧数阈值

        # 注意：这里的center_tolerance是单边容错，表示中心点可以偏离画面中心的比例
        self.center_tolerance = center_tolerance
        # 中心点最大允许水平位移（像素），用于平滑
        self.max_center_shift = max_center_shift

        # 新增：用于历史平均值的队列（每个ROI中心点）
        self.center_queue_bottom = []  # 存储底部ROI最近的有效中心点 (x, y)
        self.center_queue_mid = []     # 存储中间ROI最近的有效中心点 (x, y)
        self.max_queue_len = 25         # 队列最大长度

        # MODIFIED: 新增融合中心点的历史队列
        self.fused_center_queue = []   # 存储最近的有效融合中心点 (x, y)
        self.max_fused_queue_len = 50    # 融合中心点队列最大长度

        self.current_center_point_bottom = None
        self.current_center_point_mid = None
        self.frame_width = 640
        self.frame_center_x = 0
        self.current_frame = None

        # 车道线距离中心的最小阈值（像素）
        self.min_distance_to_center = 100

        # 调试信息
        self.debug_info_bottom = {}
        self.debug_info_mid = {}

        # 两个ROI区域参数
        # 底部ROI区域：90%-100%（由roi_ratio=0.1控制）
        # 中间ROI区域：40%-60%（固定）
        self.mid_roi_start_ratio = 0.5
        self.mid_roi_end_ratio = 0.6

        # 斜率过滤参数
        self.min_slope = 0.3  # 最小斜率阈值，过滤水平线
        self.max_slope = 3.0  # 最大斜率阈值，过滤垂直线

        # 颜色定义
        self.colors = {
            'bottom_left': (0, 0, 255),  # 红色 - 底部ROI左车道线
            'bottom_right': (255, 0, 0),  # 蓝色 - 底部ROI右车道线
            'mid_left': (0, 165, 255),  # 橙色 - 中间ROI左车道线 (BGR: 0, 165, 255)
            'mid_right': (255, 0, 255),  # 紫色 - 中间ROI右车道线 (BGR: 255, 0, 255)
            'bottom_center': (0, 255, 0),  # 绿色 - 底部ROI中心点
            'mid_center': (255, 255, 0),  # 青色 - 中间ROI中心点
            'center_line': (255, 0, 255),  # 紫色 - 中心线
            'tolerance_line': (200, 200, 200),  # 灰色 - 容错线
            'bottom_roi': (255, 255, 0),  # 青色 - 底部ROI边界
            'mid_roi': (255, 100, 100),  # 粉色 - 中间ROI边界
            'fused_center': (255, 255, 255),  # 白色 - 融合中心点
        }

        # 设置输入源
        self.setup_input_source()

        # 创建窗口
        cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mid ROI Edges', cv2.WINDOW_NORMAL)

        # print(f'车道检测系统已启动，单边容错距离: {center_tolerance * 100:.0f}% (半个画面宽度)')
        # print(f'在{self.frame_width}像素宽的画面中，容错范围为: ±{int(self.frame_width * center_tolerance / 2)}像素')
        # print(f'车道线距离中心最小阈值: {self.min_distance_to_center}像素')
        # print(f'中心点平滑最大位移: {max_center_shift}像素（基于历史平均值）')
        # print(f'历史队列长度: {self.max_queue_len}')
        # print(f'斜率过滤: 水平线(<{self.min_slope})和垂直线(>{self.max_slope})将被忽略')
        # print(f'底部ROI区域: {100 - int(roi_ratio * 100)}%-100%')
        # print(f'中间ROI区域: {int(self.mid_roi_start_ratio * 100)}%-{int(self.mid_roi_end_ratio * 100)}%')
        # print('两个ROI区域独立检测，互不干扰')
        # print('重要规则: 只要检测到的车道线距离中心点过近，都按照没有检测到处理')
        # print('新增融合中心点（白色）：两个ROI中心点的平均值（或历史平均值替代）')

    def setup_input_source(self):
        """设置输入源"""
        if self.input_source == "camera":
            try:
                camera_id = int(self.video_source) if self.video_source else 0
                self.cap = cv2.VideoCapture(camera_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception as e:
                print(f'摄像头初始化错误: {str(e)}')
        elif self.input_source == "video":
            if self.video_source and os.path.exists(self.video_source):
                self.cap = cv2.VideoCapture(self.video_source)

    def process_and_display_frame(self, frame):
        """处理并显示帧"""
        self.current_frame = frame
        result, edges, mid_edges, center_point_bottom, center_point_mid = self.process_frame(frame)

        cv2.imshow('Lane Detection', result)
        cv2.imshow('Edges', edges)
        cv2.imshow('Mid ROI Edges', mid_edges)

        self.current_center_point_bottom = center_point_bottom
        self.current_center_point_mid = center_point_mid

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup_and_shutdown()

    def set_roi_regions(self, frame):
        """设置两个ROI区域：底部ROI和中间ROI"""
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_center_x = width // 2

        # 底部ROI区域：90%-100%（当roi_ratio=0.1时）
        bottom_start_row = int(height * (1 - self.roi_ratio))
        bottom_end_row = height
        bottom_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bottom_mask[bottom_start_row:bottom_end_row, :] = 255

        # 中间ROI区域：40%-60%
        mid_start_row = int(height * self.mid_roi_start_ratio)
        mid_end_row = int(height * self.mid_roi_end_ratio)
        mid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mid_mask[mid_start_row:mid_end_row, :] = 255

        return bottom_mask, bottom_start_row, bottom_end_row, mid_mask, mid_start_row, mid_end_row

    def preprocess_frame(self, frame, bottom_mask, mid_mask):
        """预处理帧，提取两个ROI区域的边缘"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 整个画面的边缘检测
        edges_all = cv2.Canny(blurred, 50, 150)

        # 分别提取两个ROI区域的边缘
        bottom_edges = cv2.bitwise_and(edges_all, edges_all, mask=bottom_mask)
        mid_edges = cv2.bitwise_and(edges_all, edges_all, mask=mid_mask)

        return bottom_edges, mid_edges, gray

    def is_lane_too_close_to_center(self, lane, roi_start_row, roi_end_row):
        """检查车道线是否距离中心点过近"""
        if lane is None:
            return False

        # 获取车道线在ROI区域内的底部端点
        bottom_x, _ = self.get_lane_bottom_point(lane, roi_start_row, roi_end_row)

        # 检查距离是否过近
        distance_to_center = abs(self.frame_center_x - bottom_x)

        return distance_to_center < self.min_distance_to_center

    def filter_lanes_by_distance(self, lanes, roi_start_row, roi_end_row, is_left=True):
        """过滤掉距离中心点过近的车道线"""
        filtered_lanes = []

        for lane in lanes:
            # 获取车道线底部端点
            bottom_x, _ = self.get_lane_bottom_point(lane, roi_start_row, roi_end_row)

            # 检查车道线位置是否正确
            if is_left and bottom_x > self.frame_center_x:
                continue  # 左车道线应该在中心线左侧
            if not is_left and bottom_x < self.frame_center_x:
                continue  # 右车道线应该在中心线右侧

            # 检查距离是否过近
            distance_to_center = abs(self.frame_center_x - bottom_x)
            if distance_to_center < self.min_distance_to_center:
                continue  # 距离中心点过近，忽略

            filtered_lanes.append(lane)

        return filtered_lanes

    def detect_lanes(self, edges, original_frame, start_row, end_row):
        """检测车道线，并过滤掉距离中心点过近的线条"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=25, maxLineGap=12)

        left_lanes = []
        right_lanes = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 确保线条在ROI区域内
                if (y1 < start_row and y2 < start_row) or (y1 > end_row and y2 > end_row):
                    continue

                if x2 - x1 == 0:
                    continue

                # 计算斜率
                slope = (y2 - y1) / (x2 - x1)

                # 计算斜率的绝对值
                slope_abs = abs(slope)

                # 过滤水平线和垂直线
                # 水平线：斜率绝对值小于0.3
                # 垂直线：斜率绝对值大于3.0（即非常陡峭的线）
                if slope_abs < self.min_slope or slope_abs > self.max_slope:
                    continue

                # 根据斜率和位置判断左右车道线
                if slope < 0 and x1 < original_frame.shape[1] * 0.7:
                    left_lanes.append(line[0])
                elif slope > 0 and x1 > original_frame.shape[1] * 0.3:
                    right_lanes.append(line[0])

        # 过滤掉距离中心点过近的车道线
        left_lanes = self.filter_lanes_by_distance(left_lanes, start_row, end_row, is_left=True)
        right_lanes = self.filter_lanes_by_distance(right_lanes, start_row, end_row, is_left=False)

        return left_lanes, right_lanes

    def select_closest_lane(self, lanes, frame_center_x, is_left=True):
        """选择最靠近画面中心的车道线"""
        if not lanes:
            return None

        closest_lane = None
        min_distance = float('inf')

        for lane in lanes:
            x1, y1, x2, y2 = lane
            # 计算线段的中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # 计算到画面中心线的水平距离
            distance = abs(mid_x - frame_center_x)

            if distance < min_distance:
                min_distance = distance
                closest_lane = lane

        return closest_lane

    def get_lane_bottom_point(self, lane, roi_start_row, roi_end_row):
        """获取车道线在ROI区域内的底部端点"""
        if lane is None:
            return None

        x1, y1, x2, y2 = lane

        # 找到在ROI区域内且y值较大的端点
        points = []
        if roi_start_row <= y1 <= roi_end_row:
            points.append((x1, y1))
        if roi_start_row <= y2 <= roi_end_row:
            points.append((x2, y2))

        if not points:
            # 如果两个端点都不在ROI区域内，使用y值较大的端点
            if y1 > y2:
                bottom_x, bottom_y = x1, y1
            else:
                bottom_x, bottom_y = x2, y2
        else:
            # 选择在ROI区域内y值最大的点
            bottom_x, bottom_y = max(points, key=lambda p: p[1])

        return bottom_x, bottom_y

    def calculate_center_for_roi(self, left_lanes, right_lanes, roi_start_row, roi_end_row,
                                 frame_shape, debug_info_key="bottom"):
        """为指定ROI区域计算中心点 - 每个ROI单边只选择一条最靠近画面中心的线"""
        height, width = frame_shape[:2]

        # 关键修改：先筛选出最靠近中心的左右车道线
        left_lane = self.select_closest_lane(left_lanes, self.frame_center_x, is_left=True)
        right_lane = self.select_closest_lane(right_lanes, self.frame_center_x, is_left=False)

        # 更新调试信息，显示选择的车道线信息
        if debug_info_key == "bottom":
            debug_info = self.debug_info_bottom
            lane_width = self.bottom_lane_width
        else:
            debug_info = self.debug_info_mid
            lane_width = self.mid_lane_width

        # 重置调试信息
        debug_info = {
            'left_bottom_x': None,
            'right_bottom_x': None,
            'left_distance': None,
            'right_distance': None,
            'decision': 'unknown',
            'left_too_close': False,
            'right_too_close': False,
            'any_too_close': False,
            'left_lane_selected': left_lane is not None,
            'right_lane_selected': right_lane is not None,
            'left_lanes_count': len(left_lanes),
            'right_lanes_count': len(right_lanes),
        }

        if debug_info_key == "bottom":
            self.debug_info_bottom = debug_info
        else:
            self.debug_info_mid = debug_info

        # 检查车道线是否距离中心点过近
        left_too_close = False
        right_too_close = False

        if left_lane is not None:
            left_bottom_x, _ = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            left_distance_to_center = abs(self.frame_center_x - left_bottom_x)
            debug_info['left_bottom_x'] = left_bottom_x
            debug_info['left_distance'] = left_distance_to_center

            # 检查左车道线是否在中心线左侧
            if left_bottom_x > self.frame_center_x:
                # 左车道线在中心线右侧，位置错误
                debug_info['decision'] = 'left_on_wrong_side'
                left_too_close = True
            elif left_distance_to_center < self.min_distance_to_center:
                # 左车道线距离中心过近
                debug_info['left_too_close'] = True
                left_too_close = True

        if right_lane is not None:
            right_bottom_x, _ = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)
            right_distance_to_center = abs(right_bottom_x - self.frame_center_x)
            debug_info['right_bottom_x'] = right_bottom_x
            debug_info['right_distance'] = right_distance_to_center

            # 检查右车道线是否在中心线右侧
            if right_bottom_x < self.frame_center_x:
                # 右车道线在中心线左侧，位置错误
                debug_info['decision'] = 'right_on_wrong_side'
                right_too_close = True
            elif right_distance_to_center < self.min_distance_to_center:
                # 右车道线距离中心过近
                debug_info['right_too_close'] = True
                right_too_close = True

        # 检查是否有车道线距离中心点过近
        any_too_close = left_too_close or right_too_close
        debug_info['any_too_close'] = any_too_close

        # 核心规则：只要有任何车道线距离中心点过近，就按照没有检测到处理
        if any_too_close:
            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom += 1
            else:
                self.frames_without_both_lanes_mid += 1

            # 确定具体的决策原因
            if left_too_close and right_too_close:
                debug_info['decision'] = 'both_too_close'
            elif left_too_close:
                debug_info['decision'] = 'left_too_close'
            elif right_too_close:
                debug_info['decision'] = 'right_too_close'

            # 不返回中心点
            return None, None, None

        # 检查左右车道线是否都存在
        if left_lane is not None and right_lane is not None:
            # 获取车道线底部端点
            left_bottom_x, left_bottom_y = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            right_bottom_x, right_bottom_y = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)

            # 检查两条车道线是否都在画面中心的同一侧
            if (left_bottom_x < self.frame_center_x and right_bottom_x < self.frame_center_x) or \
                    (left_bottom_x > self.frame_center_x and right_bottom_x > self.frame_center_x):
                # 两条车道线都在同一侧，视为无效检测
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'both_on_same_side'
                return None, left_lane, right_lane

            # 计算车道宽度
            if debug_info_key == "bottom":
                self.bottom_lane_width = abs(right_bottom_x - left_bottom_x)
                lane_width = self.bottom_lane_width
            else:
                self.mid_lane_width = abs(right_bottom_x - left_bottom_x)
                lane_width = self.mid_lane_width

            # 计算中心点
            center_x = (left_bottom_x + right_bottom_x) / 2
            center_y = (left_bottom_y + right_bottom_y) / 2

            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom = 0
            else:
                self.frames_without_both_lanes_mid = 0
            debug_info['decision'] = 'valid_center'
            return (int(center_x), int(center_y)), left_lane, right_lane

        # 只有左车道线检测到的情况
        elif left_lane is not None:
            # 检查左车道线是否在中心线左侧（合理位置）
            left_bottom_x, _ = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            if left_bottom_x > self.frame_center_x:
                # 左车道线在中心线右侧，视为无效
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'left_on_wrong_side'
                return None, left_lane, None

            # 如果有已知的车道宽度，可以估计中心点
            if lane_width is not None:
                # 估计右侧车道线位置
                estimated_right_x = left_bottom_x + lane_width
                center_x = (left_bottom_x + estimated_right_x) / 2
                center_y = (roi_start_row + roi_end_row) / 2  # 使用ROI区域中间作为y坐标

                debug_info['decision'] = 'estimated_from_left'
                return (int(center_x), int(center_y)), left_lane, None
            else:
                # 没有已知的车道宽度，无法估计中心点
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'no_width_for_left'
                return None, left_lane, None

        # 只有右车道线检测到的情况
        elif right_lane is not None:
            # 检查右车道线是否在中心线右侧（合理位置）
            right_bottom_x, _ = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)
            if right_bottom_x < self.frame_center_x:
                # 右车道线在中心线左侧，视为无效
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'right_on_wrong_side'
                return None, None, right_lane

            # 如果有已知的车道宽度，可以估计中心点
            if lane_width is not None:
                # 估计左侧车道线位置
                estimated_left_x = right_bottom_x - lane_width
                center_x = (estimated_left_x + right_bottom_x) / 2
                center_y = (roi_start_row + roi_end_row) / 2  # 使用ROI区域中间作为y坐标

                debug_info['decision'] = 'estimated_from_right'
                return (int(center_x), int(center_y)), None, right_lane
            else:
                # 没有已知的车道宽度，无法估计中心点
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'no_width_for_right'
                return None, None, right_lane

        else:
            # 两条车道线都未检测到
            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom += 1
            else:
                self.frames_without_both_lanes_mid += 1
            debug_info['decision'] = 'no_lanes'
            return None, None, None

    def process_frame(self, frame):
        """核心处理函数：两个ROI区域独立处理，并加入中心点平滑（基于历史平均值）"""
        # 获取两个ROI区域
        bottom_mask, bottom_start_row, bottom_end_row, mid_mask, mid_start_row, mid_end_row = self.set_roi_regions(
            frame)

        # 预处理帧，获取边缘图像
        bottom_edges, mid_edges, gray = self.preprocess_frame(frame, bottom_mask, mid_mask)

        # 独立处理底部ROI区域（90%-100%）
        bottom_left_lanes, bottom_right_lanes = self.detect_lanes(bottom_edges, frame, bottom_start_row, bottom_end_row)
        center_point_bottom, bottom_left_lane, bottom_right_lane = self.calculate_center_for_roi(
            bottom_left_lanes, bottom_right_lanes, bottom_start_row, bottom_end_row,
            frame.shape, debug_info_key="bottom"
        )

        # 独立处理中间ROI区域（40%-60%）
        mid_left_lanes, mid_right_lanes = self.detect_lanes(mid_edges, frame, mid_start_row, mid_end_row)
        center_point_mid, mid_left_lane, mid_right_lane = self.calculate_center_for_roi(
            mid_left_lanes, mid_right_lanes, mid_start_row, mid_end_row,
            frame.shape, debug_info_key="mid"
        )

        # ----- 中心点平滑处理：基于历史平均值 -----
        # 底部 ROI 平滑
        bottom_smoothed = False
        if center_point_bottom is not None:
            # 计算当前队列的平均值（如果有）
            if len(self.center_queue_bottom) > 0:
                avg_x = sum(p[0] for p in self.center_queue_bottom) / len(self.center_queue_bottom)
                avg_y = sum(p[1] for p in self.center_queue_bottom) / len(self.center_queue_bottom)
                avg_point = (int(avg_x), int(avg_y))
            else:
                avg_point = None

            if avg_point is not None:
                diff_x = abs(center_point_bottom[0] - avg_point[0])
                if diff_x > self.max_center_shift:
                    # 变化过大，采用平均值
                    center_point_bottom = avg_point
                    bottom_smoothed = True
                    # 不将当前异常点加入队列，但更新 last_known_center 为平均值（可选）
                else:
                    # 变化正常，将当前点加入队列
                    self.center_queue_bottom.append(center_point_bottom)
                    if len(self.center_queue_bottom) > self.max_queue_len:
                        self.center_queue_bottom.pop(0)
            else:
                # 队列为空，直接使用当前点并加入队列
                self.center_queue_bottom.append(center_point_bottom)
            # 更新 last_known_center_bottom 为最终使用的中心点
            self.last_known_center_bottom = center_point_bottom
        else:
            # 没有检测到中心点，队列保持不变
            pass

        # 中间 ROI 平滑
        mid_smoothed = False
        if center_point_mid is not None:
            if len(self.center_queue_mid) > 0:
                avg_x = sum(p[0] for p in self.center_queue_mid) / len(self.center_queue_mid)
                avg_y = sum(p[1] for p in self.center_queue_mid) / len(self.center_queue_mid)
                avg_point = (int(avg_x), int(avg_y))
            else:
                avg_point = None

            if avg_point is not None:
                diff_x = abs(center_point_mid[0] - avg_point[0])
                if diff_x > self.max_center_shift:
                    center_point_mid = avg_point
                    mid_smoothed = True
                else:
                    self.center_queue_mid.append(center_point_mid)
                    if len(self.center_queue_mid) > self.max_queue_len:
                        self.center_queue_mid.pop(0)
            else:
                self.center_queue_mid.append(center_point_mid)
            self.last_known_center_mid = center_point_mid
        else:
            pass

        # ----- MODIFIED: 计算融合中心点（两个ROI中心点的平均值或历史平均值）-----
        fused_center_point = None
        fused_source = "none"  # 用于显示来源：actual / historical

        if center_point_bottom is not None and center_point_mid is not None:
            # 两个都存在，计算融合中心点
            fused_x = (center_point_bottom[0] + center_point_mid[0]) // 2
            fused_y = (center_point_bottom[1] + center_point_mid[1]) // 2
            fused_center_point = (fused_x, fused_y)
            fused_source = "actual"
            # 加入队列
            self.fused_center_queue.append(fused_center_point)
            if len(self.fused_center_queue) > self.max_fused_queue_len:
                self.fused_center_queue.pop(0)

        elif center_point_bottom is not None or center_point_mid is not None:
            # 只有一个中心点存在，且队列不为空，则使用历史平均值
            if len(self.fused_center_queue) > 0:
                avg_x = sum(p[0] for p in self.fused_center_queue) / len(self.fused_center_queue)
                avg_y = sum(p[1] for p in self.fused_center_queue) / len(self.fused_center_queue)
                fused_center_point = (int(avg_x), int(avg_y))
                fused_source = "historical"
                # 不将推测点加入队列，保持队列为实际有效点
            # 队列为空，则保持 None

        else:
            # 两个中心点都消失，清空历史队列（因为后续可能重新出现有效点）
            #self.fused_center_queue = []
            fused_center_point = None
            fused_source = "none"

        # 绘制结果图像
        result = frame.copy()

        # 绘制两条容错参考线（左右各一条）
        self.draw_reference_lines(result)

        # 绘制两个ROI区域的边界
        # 底部ROI区域边界
        cv2.line(result, (0, bottom_start_row), (result.shape[1], bottom_start_row), self.colors['bottom_roi'], 2)
        cv2.line(result, (0, bottom_end_row), (result.shape[1], bottom_end_row), self.colors['bottom_roi'], 2)

        # 中间ROI区域边界
        cv2.line(result, (0, mid_start_row), (result.shape[1], mid_start_row), self.colors['mid_roi'], 2)
        cv2.line(result, (0, mid_end_row), (result.shape[1], mid_end_row), self.colors['mid_roi'], 2)

        # 标注ROI区域
        #cv2.putText(result, f'Bottom ROI ({100 - int(self.roi_ratio * 100)}%-100%)',
                    #(10, bottom_start_row - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['bottom_roi'], 2)
        #cv2.putText(result, f'Mid ROI ({int(self.mid_roi_start_ratio * 100)}%-{int(self.mid_roi_end_ratio * 100)}%)',
                    #(10, (mid_start_row + mid_end_row) // 2),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['mid_roi'], 2)

        # 只绘制选定的有效车道线（如果不过近且位置正确）
        # 底部ROI车道线
        if bottom_left_lane is not None and not self.debug_info_bottom.get('left_too_close', False):
            x1, y1, x2, y2 = bottom_left_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['bottom_left'], 3)  # 红色 - 底部ROI左车道线

        if bottom_right_lane is not None and not self.debug_info_bottom.get('right_too_close', False):
            x1, y1, x2, y2 = bottom_right_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['bottom_right'], 3)  # 蓝色 - 底部ROI右车道线

        # 中间ROI车道线 - 使用新的颜色
        if mid_left_lane is not None and not self.debug_info_mid.get('left_too_close', False):
            x1, y1, x2, y2 = mid_left_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['mid_left'], 3)  # 橙色 - 中间ROI左车道线

        if mid_right_lane is not None and not self.debug_info_mid.get('right_too_close', False):
            x1, y1, x2, y2 = mid_right_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['mid_right'], 3)  # 紫色 - 中间ROI右车道线

        # 绘制底部ROI中心点
        if center_point_bottom is not None:
            cv2.circle(result, center_point_bottom, 15, self.colors['bottom_center'], -1)  # 绿色 - 底部ROI中心点
            # 显示底部ROI中心点信息，如果平滑则添加标记
            #bottom_center_text = f'Bottom Center: ({center_point_bottom[0]}, {center_point_bottom[1]})'
            #if bottom_smoothed:
                #bottom_center_text += ' (smoothed)'
            #cv2.putText(result, bottom_center_text, (10, result.shape[0] - 140),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['bottom_center'], 2)

        # 绘制中间ROI中心点
        if center_point_mid is not None:
            cv2.circle(result, center_point_mid, 12, self.colors['mid_center'], -1)  # 青色 - 中间ROI中心点
            # 显示中间ROI中心点信息，如果平滑则添加标记
            #mid_center_text = f'Mid Center: ({center_point_mid[0]}, {center_point_mid[1]})'
            #if mid_smoothed:
                #mid_center_text += ' (smoothed)'
            #cv2.putText(result, mid_center_text, (10, result.shape[0] - 110),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['mid_center'], 2)

        # 绘制融合中心点（白色）
        if fused_center_point is not None:
            cv2.circle(result, fused_center_point, 10, self.colors['fused_center'], -1)  # 白色 - 融合中心点
            #fused_center_text = f'Fused Center: ({fused_center_point[0]}, {fused_center_point[1]})'
            #if fused_source == "historical":
                #fused_center_text += ' (historical)'
            #cv2.putText(result, fused_center_text, (10, result.shape[0] - 80),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['fused_center'], 2)

        # 显示两个ROI区域的车道线状态
        bottom_left_detected = bottom_left_lane is not None
        bottom_right_detected = bottom_right_lane is not None
        bottom_left_valid = bottom_left_detected and not self.debug_info_bottom.get('left_too_close', False)
        bottom_right_valid = bottom_right_detected and not self.debug_info_bottom.get('right_too_close', False)

        mid_left_detected = mid_left_lane is not None
        mid_right_detected = mid_right_lane is not None
        mid_left_valid = mid_left_detected and not self.debug_info_mid.get('left_too_close', False)
        mid_right_valid = mid_right_detected and not self.debug_info_mid.get('right_too_close', False)

        bottom_lanes_status = f'Bottom ROI: L={bottom_left_valid}, R={bottom_right_valid}'
        mid_lanes_status = f'Mid ROI: L={mid_left_valid}, R={mid_right_valid}'

        cv2.putText(result, bottom_lanes_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, mid_lanes_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示调试信息 - 底部ROI
        debug_y_bottom = 125
        debug_texts_bottom = []

        # 添加车道线数量信息
        if 'left_lanes_count' in self.debug_info_bottom:
            debug_texts_bottom.append(f'Bottom L Lines: {self.debug_info_bottom["left_lanes_count"]}')
        if 'right_lanes_count' in self.debug_info_bottom:
            debug_texts_bottom.append(f'Bottom R Lines: {self.debug_info_bottom["right_lanes_count"]}')

        if 'left_bottom_x' in self.debug_info_bottom and self.debug_info_bottom['left_bottom_x'] is not None:
            debug_texts_bottom.append(f'Bottom L X: {int(self.debug_info_bottom["left_bottom_x"])}')
        if 'right_bottom_x' in self.debug_info_bottom and self.debug_info_bottom['right_bottom_x'] is not None:
            debug_texts_bottom.append(f'Bottom R X: {int(self.debug_info_bottom["right_bottom_x"])}')

        if 'left_distance' in self.debug_info_bottom and self.debug_info_bottom['left_distance'] is not None:
            left_dist = int(self.debug_info_bottom['left_distance'])
            debug_texts_bottom.append(f'Bottom L Dist: {left_dist}px')
            if self.debug_info_bottom.get('left_too_close', False):
                debug_texts_bottom.append(f'Bottom L TOO CLOSE!')

        if 'right_distance' in self.debug_info_bottom and self.debug_info_bottom['right_distance'] is not None:
            right_dist = int(self.debug_info_bottom['right_distance'])
            debug_texts_bottom.append(f'Bottom R Dist: {right_dist}px')
            if self.debug_info_bottom.get('right_too_close', False):
                debug_texts_bottom.append(f'Bottom R TOO CLOSE!')

        # 显示决策原因 - 底部ROI
        if 'decision' in self.debug_info_bottom:
            decision_map = {
                'valid_center': '有效中心点',
                'both_on_same_side': '同侧无效',
                'both_too_close': '两条线都过近',
                'left_too_close': '左线过近',
                'right_too_close': '右线过近',
                'left_on_wrong_side': '左线位置错误',
                'right_on_wrong_side': '右线位置错误',
                'estimated_from_left': '从左线估计',
                'estimated_from_right': '从右线估计',
                'no_width_for_left': '无宽度左线',
                'no_width_for_right': '无宽度右线',
                'no_lanes': '无线条',
                'unknown': '未知'
            }
            decision_text = decision_map.get(self.debug_info_bottom['decision'], self.debug_info_bottom['decision'])
            debug_texts_bottom.append(f'Bottom 决策: {decision_text}')

        # 在画面上显示底部ROI调试信息
        for i, text in enumerate(debug_texts_bottom):
            y_pos = debug_y_bottom + i * 20
            if y_pos < result.shape[0] - 50:
                pass
                #cv2.putText(result, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示调试信息 - 中间ROI
        debug_y_mid = debug_y_bottom + len(debug_texts_bottom) * 20 + 10
        debug_texts_mid = []

        # 添加车道线数量信息
        if 'left_lanes_count' in self.debug_info_mid:
            debug_texts_mid.append(f'Mid L Lines: {self.debug_info_mid["left_lanes_count"]}')
        if 'right_lanes_count' in self.debug_info_mid:
            debug_texts_mid.append(f'Mid R Lines: {self.debug_info_mid["right_lanes_count"]}')

        if 'left_bottom_x' in self.debug_info_mid and self.debug_info_mid['left_bottom_x'] is not None:
            debug_texts_mid.append(f'Mid L X: {int(self.debug_info_mid["left_bottom_x"])}')
        if 'right_bottom_x' in self.debug_info_mid and self.debug_info_mid['right_bottom_x'] is not None:
            debug_texts_mid.append(f'Mid R X: {int(self.debug_info_mid["right_bottom_x"])}')

        if 'left_distance' in self.debug_info_mid and self.debug_info_mid['left_distance'] is not None:
            left_dist = int(self.debug_info_mid['left_distance'])
            debug_texts_mid.append(f'Mid L Dist: {left_dist}px')
            if self.debug_info_mid.get('left_too_close', False):
                debug_texts_mid.append(f'Mid L TOO CLOSE!')

        if 'right_distance' in self.debug_info_mid and self.debug_info_mid['right_distance'] is not None:
            right_dist = int(self.debug_info_mid['right_distance'])
            debug_texts_mid.append(f'Mid R Dist: {right_dist}px')
            if self.debug_info_mid.get('right_too_close', False):
                debug_texts_mid.append(f'Mid R TOO CLOSE!')

        # 显示决策原因 - 中间ROI
        if 'decision' in self.debug_info_mid:
            decision_map = {
                'valid_center': '有效中心点',
                'both_on_same_side': '同侧无效',
                'both_too_close': '两条线都过近',
                'left_too_close': '左线过近',
                'right_too_close': '右线过近',
                'left_on_wrong_side': '左线位置错误',
                'right_on_wrong_side': '右线位置错误',
                'estimated_from_left': '从左线估计',
                'estimated_from_right': '从右线估计',
                'no_width_for_left': '无宽度左线',
                'no_width_for_right': '无宽度右线',
                'no_lanes': '无线条',
                'unknown': '未知'
            }
            decision_text = decision_map.get(self.debug_info_mid['decision'], self.debug_info_mid['decision'])
            debug_texts_mid.append(f'Mid 决策: {decision_text}')

        # 在画面上显示中间ROI调试信息
        for i, text in enumerate(debug_texts_mid):
            y_pos = debug_y_mid + i * 20
            if y_pos < result.shape[0] - 50:
                pass
                #cv2.putText(result, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 如果两个ROI都检测到了中心点，计算并显示差异
        if center_point_bottom is not None and center_point_mid is not None:
            diff_x = center_point_mid[0] - center_point_bottom[0]
            diff_text = f'Center Diff: {diff_x}px'
            color = (0, 0, 255) if abs(diff_x) > 20 else (0, 255, 0)  # 差异超过20像素显示红色
            # cv2.putText(result, diff_text, (result.shape[1] - 200, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示当前容错设置信息
        tolerance_info = f'Tolerance: ±{self.center_tolerance * 100:.0f}% of half width'
        # cv2.putText(result, tolerance_info, (10, result.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
        #             1)

        # 显示最小距离阈值
        distance_threshold_info = f'Min distance to center: {self.min_distance_to_center}px'
        # cv2.putText(result, distance_threshold_info, (10, result.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (200, 200, 200), 1)

        # 显示ROI比例
        roi_ratio_info = f'Bottom ROI: {100 - int(self.roi_ratio * 100)}%-100%, Mid ROI: 40%-60%'
        # cv2.putText(result, roi_ratio_info, (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        #             (150, 150, 150), 1)

        return result, bottom_edges, mid_edges, center_point_bottom, center_point_mid

    def draw_reference_lines(self, result):
        """绘制参考线 - 修正为绘制两条容错线"""
        # 绘制中心线（紫色）
        cv2.line(result, (self.frame_center_x, 0),
                 (self.frame_center_x, result.shape[0]), self.colors['center_line'], 2)

        # 计算容错距离（像素）
        tolerance_pixels = int(self.frame_width * self.center_tolerance / 2)

        # 绘制左容错线（灰色虚线）
        left_tolerance_x = self.frame_center_x - tolerance_pixels
        cv2.line(result, (left_tolerance_x, 0),
                 (left_tolerance_x, result.shape[0]), self.colors['tolerance_line'], 1, cv2.LINE_AA)

        # 绘制右容错线（灰色虚线）
        right_tolerance_x = self.frame_center_x + tolerance_pixels
        cv2.line(result, (right_tolerance_x, 0),
                 (right_tolerance_x, result.shape[0]), self.colors['tolerance_line'], 1, cv2.LINE_AA)

        # 绘制容错区域（半透明）
        overlay = result.copy()
        cv2.rectangle(overlay, (left_tolerance_x, 0),
                      (right_tolerance_x, result.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, result, 0.9, 0, result)

    def run(self):
        """运行主循环"""
        if self.cap is None or not self.cap.isOpened():
            print("无法打开视频源")
            return

        # print("车道检测系统 - 双ROI独立检测")
        # print("=" * 50)
        # print(f"画面宽度: {self.frame_width}像素")
        # print(f"中心线位置: {self.frame_center_x}像素")
        # print(f"容错范围: ±{int(self.frame_width * self.center_tolerance / 2)}像素")
        # print(f"车道线距离中心最小阈值: {self.min_distance_to_center}像素")
        # print(f"中心点平滑最大位移: {self.max_center_shift}像素（基于历史平均值）")
        # print(f"历史队列长度: {self.max_queue_len}")
        # print(f"斜率过滤: 水平线(斜率绝对值<{self.min_slope})和垂直线(斜率绝对值>{self.max_slope})将被忽略")
        # print(f"底部ROI区域: {100 - int(self.roi_ratio * 100)}%-100%")
        # print(f"中间ROI区域: {int(self.mid_roi_start_ratio * 100)}%-{int(self.mid_roi_end_ratio * 100)}%")
        # print("两个ROI区域独立检测，互不干扰")
        # print(f"重要规则: 只要检测到的车道线距离中心点<{self.min_distance_to_center}像素，都按照没有检测到处理")
        # print("颜色说明:")
        # print("  底部ROI: 左车道线(红色), 右车道线(蓝色), 中心点(绿色)")
        # print("  中间ROI: 左车道线(橙色), 右车道线(紫色), 中心点(青色)")
        # print("  融合中心点: 白色（两个ROI中心点的平均值或历史平均值）")
        # print("只显示最靠近画面中心的单条车道线（不显示其他检测到的线）")
        # print("按 'q' 退出程序")
        # print("按 's' 保存当前帧")
        # print("按 '+' 增加容错距离")
        # print("按 '-' 减少容错距离")
        # print("按 'd' 增加距离阈值")
        # print("按 'a' 减少距离阈值")
        # print("按 'w' 增加最大斜率阈值")
        # print("按 'x' 减少最大斜率阈值")
        # print("按 'e' 增加最小斜率阈值")
        # print("按 'c' 减少最小斜率阈值")

        while True:
            if self.input_source == "camera" or self.input_source == "video":
                ret, frame = self.cap.read()
                if not ret:
                    if self.input_source == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                self.process_and_display_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.current_frame is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"lane_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, self.current_frame)
                    print(f"已保存帧到: {filename}")
            elif key == ord('+'):
                # 增加容错距离
                self.center_tolerance = min(self.center_tolerance + 0.02, 0.5)  # 最大50%
                print(
                    f"增加容错距离: {self.center_tolerance:.2f} (±{int(self.frame_width * self.center_tolerance / 2)}像素)")
            elif key == ord('-'):
                # 减少容错距离
                self.center_tolerance = max(self.center_tolerance - 0.02, 0.01)  # 最小1%
                print(
                    f"减少容错距离: {self.center_tolerance:.2f} (±{int(self.frame_width * self.center_tolerance / 2)}像素)")
            elif key == ord('d'):
                # 增加距离阈值
                self.min_distance_to_center = min(self.min_distance_to_center + 5, 200)  # 最大200像素
                print(f"增加距离阈值: {self.min_distance_to_center}像素")
            elif key == ord('a'):
                # 减少距离阈值
                self.min_distance_to_center = max(self.min_distance_to_center - 5, 10)  # 最小10像素
                print(f"减少距离阈值: {self.min_distance_to_center}像素")
            elif key == ord('w'):
                # 增加最大斜率阈值
                self.max_slope = min(self.max_slope + 0.5, 10.0)  # 最大10.0
                print(f"增加最大斜率阈值: {self.max_slope}")
            elif key == ord('x'):
                # 减少最大斜率阈值
                self.max_slope = max(self.max_slope - 0.5, 1.0)  # 最小1.0
                print(f"减少最大斜率阈值: {self.max_slope}")
            elif key == ord('e'):
                # 增加最小斜率阈值
                self.min_slope = min(self.min_slope + 0.1, 1.0)  # 最大1.0
                print(f"增加最小斜率阈值: {self.min_slope}")
            elif key == ord('c'):
                # 减少最小斜率阈值
                self.min_slope = max(self.min_slope - 0.1, 0.1)  # 最小0.1
                print(f"减少最小斜率阈值: {self.min_slope}")

    def cleanup_and_shutdown(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
        print("程序已退出")


def main():
    parser = argparse.ArgumentParser(description='车道检测系统 - 双ROI独立检测')
    parser.add_argument('--input', type=str, default='video',
                        choices=['camera', 'video'],
                        help='输入源类型')
    parser.add_argument('--source', type=str, default='4.mp4',
                        help='源路径: 摄像头ID或视频文件路径')
    parser.add_argument('--roi', type=float, default=0.1,
                        help='底部ROI区域比例，例如0.1表示90%-100%的区域')
    parser.add_argument('--tolerance', type=float, default=0.08,
                        help='单边容错距离（相对于半个画面宽度的比例），默认0.08表示8%') 
    parser.add_argument('--min-distance', type=int, default=100,
                        help='车道线距离画面中心的最小阈值（像素），默认100像素')
    parser.add_argument('--min-slope', type=float, default=0.3,
                        help='最小斜率阈值，过滤水平线，默认0.3')
    parser.add_argument('--max-slope', type=float, default=1.0,
                        help='最大斜率阈值，过滤垂直线，默认3.0')
    # 中心点平滑最大位移（像素）
    parser.add_argument('--max-shift', type=int, default=25,
                        help='中心点水平方向最大允许位移（像素），超过此值使用历史平均值，默认25像素')

    args = parser.parse_args()

    detector = LaneDetector(
        roi_ratio=args.roi,
        input_source=args.input,
        video_source=args.source,
        center_tolerance=args.tolerance,
        max_center_shift=args.max_shift
    )

    # 设置最小距离阈值
    detector.min_distance_to_center = args.min_distance

    # 设置斜率阈值
    detector.min_slope = args.min_slope
    detector.max_slope = args.max_slope

    try:
        detector.run()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    finally:
        detector.cleanup_and_shutdown()


if __name__ == "__main__":
    main()
