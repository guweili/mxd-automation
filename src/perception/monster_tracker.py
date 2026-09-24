"""怪物多目标 KCF 跟踪器。

YOLO 检测到怪物后，为每个怪物创建 KCF 跟踪器做每帧实时跟踪（毫秒级）。
YOLO 只定期（每 N 帧）重新检测，刷新怪物列表。
两次 YOLO 检测之间，怪物位置由 KCF 跟踪器插值更新，实现 60FPS 实时跟随。
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .yolo_detector import Detection


def _create_kcf():
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        return cv2.legacy.TrackerKCF_create


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """计算两个 (x,y,w,h) 框的 IoU。"""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class MonsterTracker:
    """管理多个怪物的 KCF 跟踪器。

    - update(frame): 每帧调用，用 KCF 更新所有怪物位置，返回当前怪物列表。
    - refresh(detections, frame): YOLO 有新检测结果时调用，用 IoU 匹配
      新旧怪物，更新/新增/删除跟踪器。
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self._trackers: List[dict] = []
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._next_id = 0

    def reset(self) -> None:
        self._trackers = []
        self._next_id = 0

    def update(self, frame: np.ndarray) -> List[Detection]:
        """每帧调用，用 KCF 更新所有怪物位置。"""
        alive: List[dict] = []
        for t in self._trackers:
            try:
                ok, bbox = t["tracker"].update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    if w > 5 and h > 5:
                        t["box"] = (x, y, w, h)
                        t["age"] = 0
                        alive.append(t)
                        continue
            except Exception:
                pass
            t["age"] += 1
            if t["age"] < self._max_age:
                alive.append(t)
        self._trackers = alive
        return [
            Detection(
                cls_name=t["cls"],
                confidence=t["conf"],
                x=t["box"][0], y=t["box"][1],
                w=t["box"][2], h=t["box"][3],
            )
            for t in self._trackers
        ]

    def refresh(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """YOLO 有新检测结果时调用，刷新跟踪器列表。

        用 IoU 匹配新旧怪物：
          - 匹配上的：用新检测框重新初始化 KCF（校正位置/尺寸）
          - 未匹配的新检测：新建 KCF 跟踪器
          - 未匹配的旧跟踪器：保留（KCF 继续跟踪，直到 max_age 过期）
        """
        new_dets = list(detections)
        matched_old = set()
        matched_new = set()

        # 1. 对每个现有跟踪器，找 IoU 最大的新检测
        for ti, t in enumerate(self._trackers):
            best_iou = 0.0
            best_di = -1
            for di, d in enumerate(new_dets):
                if di in matched_new:
                    continue
                if d.cls_name != t["cls"]:
                    continue
                iou = _iou(t["box"], (d.x, d.y, d.w, d.h))
                if iou > best_iou:
                    best_iou = iou
                    best_di = di
            if best_iou >= self._iou_threshold and best_di >= 0:
                matched_old.add(ti)
                matched_new.add(best_di)
                # 用新框重新初始化 KCF
                d = new_dets[best_di]
                box = (d.x, d.y, d.w, d.h)
                try:
                    tracker = _create_kcf()
                    tracker.init(frame, box)
                    t["tracker"] = tracker
                    t["box"] = box
                    t["conf"] = d.confidence
                    t["age"] = 0
                except Exception:
                    pass

        # 2. 未匹配的新检测 → 新建跟踪器
        for di, d in enumerate(new_dets):
            if di in matched_new:
                continue
            box = (d.x, d.y, d.w, d.h)
            try:
                tracker = _create_kcf()
                tracker.init(frame, box)
                self._trackers.append({
                    "id": self._next_id,
                    "cls": d.cls_name,
                    "conf": d.confidence,
                    "box": box,
                    "tracker": tracker,
                    "age": 0,
                })
                self._next_id += 1
            except Exception:
                pass

        # 3. 未匹配的旧跟踪器保留（KCF 继续跟踪），但 age+1
        for ti, t in enumerate(self._trackers):
            if ti not in matched_old:
                t["age"] += 1

        # 清理过期的
        self._trackers = [t for t in self._trackers if t["age"] < self._max_age]

        return [
            Detection(
                cls_name=t["cls"],
                confidence=t["conf"],
                x=t["box"][0], y=t["box"][1],
                w=t["box"][2], h=t["box"][3],
            )
            for t in self._trackers
        ]