"""角色外观模板跟踪器。

使用用户提供的角色全身截图作为模板，通过模板匹配首次定位角色，
之后用 KCF 跟踪器做每帧实时跟踪（毫秒级）。KCF 跟丢时回退模板匹配
重新锁定并重新初始化 KCF。

换时装/换地图后需重新截图并替换模板（路径由 resolve_template_path() 统一解析）。
"""

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np

from ..utils.config_loader import resolve_template_path


def _create_kcf():
    """创建 KCF 跟踪器（兼容不同 OpenCV 版本）。"""
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        return cv2.legacy.TrackerKCF_create


class PlayerTracker:
    """基于 KCF + 模板匹配的角色定位器。

    工作流程:
        1. 加载用户提供的角色全身模板图
        2. 首帧/跟丢时用全图多尺度模板匹配找到角色，初始化 KCF
        3. 之后每帧用 KCF.update() 实时跟踪（毫秒级，比 matchTemplate 快一个数量级）
        4. KCF 跟丢或置信度低时回退模板匹配重新锁定

    返回的是包围框 (x, y, w, h, confidence)，调用方取底部中心即可作为脚底坐标。
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        threshold: float = 0.55,
        scale_range: Optional[List[float]] = None,
        search_margin: int = 120,
        max_miss: int = 8,
        full_threshold: float = 0.75,
    ):
        """Args:
            template_path: 角色全身模板图路径。None 时使用 resolve_template_path()
                      统一解析（exe 旁边 / 项目根目录）。
            threshold: 模板匹配阈值，0~1。像素风游戏角色建议 0.50~0.65。
            scale_range: 多尺度匹配的缩放列表。默认 [0.8, 0.9, 1.0, 1.1, 1.2]。
            search_margin: 局部搜索时，在上一帧框四周扩展的像素边距。
            max_miss: 连续丢失多少帧后放弃局部跟踪、强制全图搜索。
            full_threshold: 全图回退匹配阈值，必须高于局部阈值。
        """
        # 未显式传入时，用统一解析出的默认路径（exe 旁边 / 项目根目录）
        if template_path is None:
            template_path = resolve_template_path()

        # 支持绝对路径和相对路径
        if not os.path.isabs(template_path):
            candidates = []
            try:
                from ..utils.config_loader import APP_DIR, BUNDLE_DIR
                candidates.append(os.path.join(APP_DIR, template_path))
                candidates.append(os.path.join(BUNDLE_DIR, template_path))
            except Exception:
                pass
            candidates.append(
                os.path.join(os.path.dirname(__file__), "../..", template_path)
            )
            candidates.append(template_path)
            for p in candidates:
                if os.path.exists(p):
                    template_path = p
                    break

        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if self.template is None:
            raise FileNotFoundError(
                f"无法加载角色模板图，请确认路径正确: {os.path.abspath(template_path)}"
            )

        self.th, self.tw = self.template.shape[:2]
        self.threshold = threshold
        self.full_threshold = full_threshold
        self.scale_range = scale_range if scale_range is not None else [0.8, 0.9, 1.0, 1.1, 1.2]
        self.search_margin = search_margin
        self.max_miss = max_miss

        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.last_score = 0.0
        self.miss_count = 0

        # KCF 跟踪器：每帧毫秒级更新，替代慢的 matchTemplate 局部搜索
        self._tracker: Optional[cv2.Tracker] = None
        # KCF 不支持尺度变化，每隔 N 帧用模板匹配校准一次尺寸
        self._recalibrate_interval = 30
        self._frame_since_recal = 0

    def set_template(self, template_path: str) -> None:
        """运行时替换模板图（换时装/换地图后调用），并重置跟踪状态。"""
        new_template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if new_template is None:
            raise FileNotFoundError(
                f"无法加载角色模板图: {os.path.abspath(template_path)}"
            )
        self.template = new_template
        self.th, self.tw = new_template.shape[:2]
        self.reset()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空缓存和 KCF 跟踪器，下一帧强制全图搜索。"""
        self.last_box = None
        self.last_score = 0.0
        self.miss_count = 0
        self._tracker = None
        self._frame_since_recal = 0

    def locate(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """在当前帧中定位角色。

        优先用 KCF 跟踪器做毫秒级实时更新；KCF 跟丢或需校准时回退模板匹配。

        Args:
            frame: BGR 图像。
            exclude_bottom: 排除底部像素数（底部 UI 条/角色头像区域）。

        Returns:
            (x, y, w, h, confidence) 若找到，否则 None。
        """
        h, w = frame.shape[:2]

        # ---- 1. KCF 实时跟踪（已有跟踪器且未跟丢）----
        if self._tracker is not None and self.miss_count < self.max_miss:
            ok, bbox = self._tracker.update(frame)
            if ok:
                x, y, bw, bh = [int(v) for v in bbox]
                # KCF 返回的框可能越界或尺寸异常，做基本校验
                if 0 <= x < w and 0 <= y < h and bw > 10 and bh > 10:
                    self.last_box = (x, y, bw, bh)
                    self.last_score = 1.0  # KCF 成功即视为高置信
                    self.miss_count = 0
                    self._frame_since_recal += 1
                    # 定期用模板匹配校准尺寸（KCF 不支持尺度变化）
                    if self._frame_since_recal >= self._recalibrate_interval:
                        self._recalibrate_with_template(frame, exclude_bottom)
                    return (x, y, bw, bh, 1.0)
            # KCF 跟丢
            self.miss_count += 1

        # ---- 2. 模板匹配全图搜索（回退/初始化/校准）----
        box, score = self._match_full(frame, exclude_bottom)
        if box is not None:
            if score < self.full_threshold:
                self.miss_count += 1
                return None
            self.last_box = box
            self.last_score = score
            self.miss_count = 0
            # 初始化/重置 KCF 跟踪器
            self._init_tracker(frame, box)
            return (*box, score)

        # 彻底跟丢
        self.miss_count += 1
        return None

    def _init_tracker(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        """用当前帧和检测框初始化 KCF 跟踪器。"""
        try:
            self._tracker = _create_kcf()
            self._tracker.init(frame, tuple(box))
            self._frame_since_recal = 0
        except Exception:
            self._tracker = None

    def _recalibrate_with_template(self, frame: np.ndarray, exclude_bottom: int) -> None:
        """用模板匹配校准 KCF 跟踪器的框尺寸（KCF 不支持尺度变化）。

        在 last_box 附近做局部模板匹配，如果找到且与 KCF 框差异不大，
        用模板匹配的框重新初始化 KCF（更新尺寸）。
        """
        if self.last_box is None:
            return
        box, score = self._match_local(frame, exclude_bottom)
        if box is not None and score >= self.threshold:
            self.last_box = box
            self._init_tracker(frame, box)

    # ------------------------------------------------------------------
    # 模板匹配（用于初始化/校准/回退）
    # ------------------------------------------------------------------

    def _match_local(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """在上一帧位置附近做局部多尺度匹配（用于尺寸校准）。"""
        x, y, bw, bh = self.last_box  # type: ignore[misc]
        margin = self.search_margin
        fh, fw = frame.shape[:2]

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(fw, x + bw + margin)
        y2 = min(fh, y + bh + margin)

        roi = frame[y1:y2, x1:x2]
        roi_h = roi.shape[0]
        local_exclude = 0
        if exclude_bottom > 0:
            global_bottom = frame.shape[0] - exclude_bottom
            if y2 > global_bottom:
                local_exclude = y2 - global_bottom
        box, score = self._match_multi_scale(roi, local_exclude)
        if box is not None:
            abs_box = (box[0] + x1, box[1] + y1, box[2], box[3])
            return abs_box, score
        return None, 0.0

    def _match_full(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """在全图做多尺度匹配，排除底部 UI 区域内的候选。"""
        return self._match_multi_scale(frame, exclude_bottom)

    def _match_multi_scale(
        self,
        image: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """对给定图像做多尺度模板匹配，返回 (最佳框, 置信度)。"""
        ih, iw = image.shape[:2]
        best_score = -1.0
        best_box: Optional[Tuple[int, int, int, int]] = None

        flipped = cv2.flip(self.template, 1)
        bottom_limit = ih - exclude_bottom if exclude_bottom > 0 else ih

        for scale in self.scale_range:
            tw = int(self.tw * scale)
            th = int(self.th * scale)
            if tw > iw or th > ih:
                continue

            # 正向模板
            resized = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_box = (max_loc[0], max_loc[1], tw, th)

            # 镜像模板（角色反向）
            resized_f = cv2.resize(flipped, (tw, th), interpolation=cv2.INTER_AREA)
            result_f = cv2.matchTemplate(image, resized_f, cv2.TM_CCOEFF_NORMED)
            _, max_val_f, _, max_loc_f = cv2.minMaxLoc(result_f)
            if max_val_f > best_score:
                best_score = max_val_f
                best_box = (max_loc_f[0], max_loc_f[1], tw, th)

        # 底部 UI 排除
        if best_box is not None and exclude_bottom > 0:
            bx, by, bw_, bh_ = best_box
            if by + bh_ >= bottom_limit:
                masked = image.copy()
                masked[bottom_limit:, :] = 0
                best_score = -1.0
                best_box = None
                for scale in self.scale_range:
                    tw = int(self.tw * scale)
                    th = int(self.th * scale)
                    if tw > iw or th > ih:
                        continue
                    resized = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
                    result = cv2.matchTemplate(masked, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_score:
                        best_score = max_val
                        best_box = (max_loc[0], max_loc[1], tw, th)
                    resized_f = cv2.resize(flipped, (tw, th), interpolation=cv2.INTER_AREA)
                    result_f = cv2.matchTemplate(masked, resized_f, cv2.TM_CCOEFF_NORMED)
                    _, max_val_f, _, max_loc_f = cv2.minMaxLoc(result_f)
                    if max_val_f > best_score:
                        best_score = max_val_f
                        best_box = (max_loc_f[0], max_loc_f[1], tw, th)

        if best_box is not None and best_score >= self.threshold:
            return best_box, float(best_score)
        return None, 0.0