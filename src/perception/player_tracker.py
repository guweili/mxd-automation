"""角色外观模板跟踪器。

使用用户提供的角色全身截图作为模板，通过多尺度模板匹配在画面中定位角色。
支持"局部搜索加速"（一旦锁定，下一帧只在附近搜索）和"全图回退"（跟丢后全局重搜），
不依赖脚底阴影或名字 OCR，因此即使脚底被地图遮挡也能工作。

换时装/换地图后需重新截图并替换模板（路径由 resolve_template_path() 统一解析）。
"""

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np

from ..utils.config_loader import resolve_template_path


class PlayerTracker:
    """基于外观模板的角色定位器。

    工作流程:
        1. 加载用户提供的角色全身模板图
        2. 每帧优先在上一帧位置附近做局部多尺度匹配（快）
        3. 局部失败则回退到全图多尺度匹配（慢但稳定）
        4. 连续跟丢 max_miss 帧后清空缓存，下一帧强制全图搜索

    返回的是模板匹配的包围框，调用方取底部中心即可作为脚底坐标。
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
            threshold: 局部搜索匹配阈值，0~1。像素风游戏角色建议 0.50~0.65，
                      太高容易漏，太低容易误检到 NPC/其他玩家。
                      局部搜索位置可信（受上一帧位置约束），用低阈值即可。
            scale_range: 多尺度匹配的缩放列表。默认 [0.8, 0.9, 1.0, 1.1, 1.2]。
            search_margin: 局部搜索时，在上一帧框四周扩展的像素边距。
            max_miss: 连续丢失多少帧后放弃局部跟踪、强制全图搜索。
            full_threshold: 全图回退匹配阈值，必须高于局部阈值。
                      全图匹配无位置约束，低置信度大概率是 NPC/怪物/背景
                      与模板相似；接受会污染 last_box → 自身坐标错误 →
                      追怪方向反。默认 0.75，只有高置信度才允许重新锁定。
        """
        # 未显式传入时，用统一解析出的默认路径（exe 旁边 / 项目根目录）
        if template_path is None:
            template_path = resolve_template_path()

        # 支持绝对路径和相对路径
        if not os.path.isabs(template_path):
            # 查找顺序: exe 旁边(APP_DIR, 用户上传的可写位置) → 打包内(BUNDLE_DIR)
            # → 项目根目录 → 当前工作目录
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
        # 上次匹配到的模板方向："normal" 正向 / "flipped" 镜像。
        # 局部搜索时只搜该方向，减半 matchTemplate 次数；角色转向时
        # 局部搜索会失败，回退全图搜索（两个方向都搜）重新锁定。
        self._last_direction: str = "both"
        # 局部搜索使用的尺度范围：角色大小几乎不变，用 3 个尺度即可，
        # 比全图的 5 个尺度少 40% 计算量。
        self.local_scale_range = [0.95, 1.0, 1.05]

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
        """清空缓存，下一帧强制全图搜索。"""
        self.last_box = None
        self.last_score = 0.0
        self.miss_count = 0
        self._last_direction = "both"  # 重置后局部搜索搜两个方向，避免方向锁死

    def locate(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """在当前帧中定位角色。

        Args:
            frame: BGR 图像。
            exclude_bottom: 排除底部像素数（用于排除底部 UI 条/角色头像）。
                           不是裁剪图像（裁剪会导致 matchTemplate 输出尺寸
                           变小、坐标计算偏移），而是在匹配结果里丢弃
                           匹配框底部落在该区域的候选，改选上方游戏场景
                           内的最佳匹配。

        Returns:
            (x, y, w, h, confidence) 若找到，否则 None。
            (x, y) 为模板左上角，(w, h) 为匹配到的实际尺寸（已含缩放）。
        """
        # ---- 1. 局部搜索（已有历史位置且未连续跟丢太多帧）----
        if self.last_box is not None and self.miss_count < self.max_miss:
            box, score, direction = self._match_local(frame, exclude_bottom)
            if box is not None:
                self.last_box = box
                self.last_score = score
                self.miss_count = 0
                if direction:
                    self._last_direction = direction
                return (*box, score)
            self.miss_count += 1

        # ---- 2. 全图搜索（回退）----
        box, score, direction = self._match_full(frame, exclude_bottom)
        if box is not None:
            # 全图回退必须达到高阈值才接受：无位置约束的匹配大概率是
            # NPC/怪物/背景与模板相似，接受会污染 last_box → 后续局部
            # 搜索在错误位置维持 → 自身坐标错误 → 追怪方向反。
            if score < self.full_threshold:
                self.miss_count += 1
                return None
            self.last_box = box
            self.last_score = score
            self.miss_count = 0
            if direction:
                self._last_direction = direction
            return (*box, score)

        # 彻底跟丢
        self.miss_count += 1
        return None

    # ------------------------------------------------------------------
    # 内部匹配逻辑
    # ------------------------------------------------------------------

    def _match_local(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
        """在上一帧位置附近做局部多尺度匹配。

        局部搜索优化：
          - 只用 3 个尺度（角色大小几乎不变），比全图 5 尺度少 40% 计算
          - 只搜上次匹配到的方向（正向/镜像），减半 matchTemplate 次数
        角色转向时局部搜索会失败，回退全图搜索（两方向都搜）重新锁定。
        """
        x, y, bw, bh = self.last_box  # type: ignore[misc]
        margin = self.search_margin
        fh, fw = frame.shape[:2]

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(fw, x + bw + margin)
        y2 = min(fh, y + bh + margin)

        roi = frame[y1:y2, x1:x2]
        # 局部搜索区域可能因 last_box 偏移（OCR 回填的脚底位置有误差）
        # 延伸到底部 UI，仍需排除底部区域内的候选，避免匹到 UI 头像。
        # 注意：roi 是裁剪后的子图，exclude_bottom 要换算成 roi 坐标系
        # 下的底部排除量。
        roi_h = roi.shape[0]
        # 全局底部 UI 起点 bottom_limit，换算到 roi 内：roi 底边 = y2，
        # 若 y2 超过全局 bottom_limit，则 roi 内的排除量 = y2 - bottom_limit
        local_exclude = 0
        if exclude_bottom > 0:
            global_bottom = frame.shape[0] - exclude_bottom
            if y2 > global_bottom:
                local_exclude = y2 - global_bottom
        # 局部搜索：少尺度 + 单方向，大幅加速
        box, score, direction = self._match_multi_scale(
            roi, local_exclude,
            scales=self.local_scale_range,
            direction=self._last_direction,
        )
        if box is not None:
            # 转回全局坐标
            abs_box = (box[0] + x1, box[1] + y1, box[2], box[3])
            return abs_box, score, direction
        return None, 0.0, ""

    def _match_full(
        self,
        frame: np.ndarray,
        exclude_bottom: int = 0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
        """在全图做多尺度匹配，排除底部 UI 区域内的候选。"""
        return self._match_multi_scale(frame, exclude_bottom)

    def _match_multi_scale(
        self,
        image: np.ndarray,
        exclude_bottom: int = 0,
        scales: Optional[List[float]] = None,
        direction: str = "both",
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, str]:
        """对给定图像做多尺度模板匹配，返回 (最佳框, 置信度, 匹配方向)。

        使用 TM_CCOEFF_NORMED，对整体亮度变化有一定鲁棒性。
        同时匹配模板的镜像（左右翻转）版本：冒险岛角色转向时身体是
        镜像的，只有单一朝向的模板会导致角色转身后匹配失败、坐标
        停留在旧位置不实时变动。镜像匹配让角色朝哪个方向都能锁定。

        Args:
            image:          搜索图像
            exclude_bottom: 排除底部像素数（底部 UI 条/角色头像区域）。
                            匹配框底部 (y+th) 落在该区域内的候选会被丢弃，
                            改选上方游戏场景内的最佳匹配，避免误匹配到
                            UI 里的角色头像（和模板形象几乎相同、分数更高）。
            scales:         本次匹配使用的尺度列表。None 时用 self.scale_range。
                            局部搜索时可传更少的尺度（角色大小几乎不变）加速。
            direction:      匹配方向："normal"（只正向）/"flipped"（只镜像）/
                            "both"（正反都匹配）。局部搜索时传上次匹配到的
                            方向，只搜一个方向可减半计算量。

        Returns:
            (box, score, matched_direction)，未命中时 (None, 0.0, "")
        """
        ih, iw = image.shape[:2]
        best_score = -1.0
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_dir = ""

        # 模板镜像（左右翻转），用于匹配角色反向时的外观
        flipped = cv2.flip(self.template, 1)

        # 底部 UI 区域起点（像素）：匹配框底部 >= 此值则丢弃
        bottom_limit = ih - exclude_bottom if exclude_bottom > 0 else ih

        if scales is None:
            scales = self.scale_range

        for scale in scales:
            tw = int(self.tw * scale)
            th = int(self.th * scale)
            # 跳过比搜索区域还大的尺度
            if tw > iw or th > ih:
                continue

            if direction in ("both", "normal"):
                # 正向模板
                resized = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_box = (max_loc[0], max_loc[1], tw, th)
                    best_dir = "normal"

            if direction in ("both", "flipped"):
                # 镜像模板（角色反向）
                resized_f = cv2.resize(flipped, (tw, th), interpolation=cv2.INTER_AREA)
                result_f = cv2.matchTemplate(image, resized_f, cv2.TM_CCOEFF_NORMED)
                _, max_val_f, _, max_loc_f = cv2.minMaxLoc(result_f)
                if max_val_f > best_score:
                    best_score = max_val_f
                    best_box = (max_loc_f[0], max_loc_f[1], tw, th)
                    best_dir = "flipped"

        # ---- 底部 UI 排除 ----
        # 如果最佳匹配的框底部 (y+th) 落在底部 UI 区域，说明匹到了 UI
        # 里的角色头像。此时在全图中屏蔽该区域、重新匹配，选上方场景
        # 内的最佳匹配（真实角色）。
        if best_box is not None and exclude_bottom > 0:
            bx, by, bw_, bh_ = best_box
            if by + bh_ >= bottom_limit:
                # 把底部 UI 区域涂黑后重新匹配，强制选上方场景内的匹配
                masked = image.copy()
                masked[bottom_limit:, :] = 0
                best_score = -1.0
                best_box = None
                best_dir = ""
                for scale in scales:
                    tw = int(self.tw * scale)
                    th = int(self.th * scale)
                    if tw > iw or th > ih:
                        continue
                    if direction in ("both", "normal"):
                        resized = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
                        result = cv2.matchTemplate(masked, resized, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        if max_val > best_score:
                            best_score = max_val
                            best_box = (max_loc[0], max_loc[1], tw, th)
                            best_dir = "normal"
                    if direction in ("both", "flipped"):
                        resized_f = cv2.resize(flipped, (tw, th), interpolation=cv2.INTER_AREA)
                        result_f = cv2.matchTemplate(masked, resized_f, cv2.TM_CCOEFF_NORMED)
                        _, max_val_f, _, max_loc_f = cv2.minMaxLoc(result_f)
                        if max_val_f > best_score:
                            best_score = max_val_f
                            best_box = (max_loc_f[0], max_loc_f[1], tw, th)
                            best_dir = "flipped"

        if best_box is not None and best_score >= self.threshold:
            return best_box, float(best_score), best_dir
        return None, 0.0, ""