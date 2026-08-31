"""RapidOCR 名字定位器。

通过 OCR 识别画面中的角色名字，定位角色坐标。

================================================================================
原理（冒险岛实际布局）
================================================================================

  角色名字显示在角色脚下。名字中心 ≈ 角色脚底，从名字中心向上
  延伸"人物高度一半"得到角色中心点（角色身体在名字上方）：

    ╔═══════════╗   ← 名字中心 y - 30 = 角色中心点 y（往上）
    ║  我是立立   ║
    ╚═══╤═══════╝
        │  ↑ 名字中心 = 脚底
        │
        │  ↑ - 人物高度一半（character_height // 2）
        │
    ╔═══╧═══╗
    ║ 角色   ║
    ║       ║  ← 人物高度（character_height，约 60px）
    ╚═══════╝

================================================================================
用法
================================================================================

  locator = OCRNameLocator(character_height=60, on_log=print)
  # 返回 (center_x, center_y, foot_x, foot_y) 或 None
  result = locator.locate(frame, "我是立立")
  if result:
      cx, cy, fx, fy = result

================================================================================
注意事项
================================================================================

  - OCR 模型首次加载较慢（约 1-2 秒），后续帧很快
  - 建议配合缓存使用：OCR 帧找到后缓存坐标，非 OCR 帧直接返回缓存
  - 如果名字和其他文字重叠，可能匹配失败
  - character_height 需要根据角色实际高度调整（通常 50~70 像素）
"""
import os
import sys
from typing import Callable, Optional, Tuple

import numpy as np


class OCRNameLocator:
    """基于 RapidOCR 的角色名字定位器。

    冒险岛角色名字显示在脚下，名字中心 ≈ 脚底；
    角色中心点 = 名字中心向上延伸"人物高度一半"。

    每帧调用 locate()。为提升定位更新频率，优先在"上次位置附近的小窗口"
    内 OCR（小图识别快，几十 ms/次），找不到再回退全屏搜索区域。
    同时返回角色中心点和脚底坐标。

    Args:
        character_height: 人物高度像素数，默认 60；角色中心 = 名字中心 - 高度一半
        ocr_interval:     保留参数（不再按帧节流，每帧都尝试定位）
        exact_match:      是否精确匹配名字（True=完全相等，False=包含即可）
        on_log:           日志回调
    """

    def __init__(self, character_height: int = 60,
                 ocr_interval: int = 30,
                 exact_match: bool = False,
                 on_log: Optional[Callable[[str], None]] = None):
        self._character_height = character_height
        self._interval = ocr_interval
        self._exact_match = exact_match
        self._on_log = on_log or (lambda m: None)
        self._engine = None
        self._frame_count = 0
        self._init_ok = False
        self._last_center: Optional[Tuple[int, int]] = None
        self._last_foot: Optional[Tuple[int, int]] = None

        # 帧间跳变过滤：OCR 偶尔会把画面中固定位置的 UI 文字（聊天框、
        # 任务栏、怪物名等）误识别为角色名（匹配是"包含"逻辑），
        # 导致角色坐标瞬间跳到别处，决策层基于错误坐标把同平台的怪
        # 判成跨层 → 一直移动不攻击。
        # 策略：新识别位置与上次有效位置偏移超过阈值 → 直接判定为
        # 误识别，沿用上次位置且不更新缓存。角色正常移动每帧最多几十
        # px，超过上限只能是误识别（游戏无瞬移）；换图后 OCR 找不到
        # 名字会返回 None，由决策层按未定位处理，不会用错误位置。
        # 这里用的是曼哈顿距离(|dx|+|dy|)，所以 300 ≈ 角色在 200×100
        # 范围内都算合理（移动+小跳）。
        self._max_jump = 300  # 单次识别相对上次有效位置的偏移上限(px，曼哈顿)

        # 连续误识别超时重置：如果首帧就识别到了错误位置（比如匹配到
        # UI 固定文字），后续所有帧都会被 _max_jump 过滤掉，位置永远
        # 卡在错误坐标上。策略：连续 N 帧都被跳变过滤 → 接受新位置
        # 并重置缓存，防止首帧错误永久锁死定位。
        self._skip_counter = 0
        self._skip_threshold = 5

        # 位置长期未变强制复位：OCR 首帧匹配到 UI 固定文字（如聊天框
        # 中的角色名）后，后续帧因"离上次最近"策略永远选同一位置，
        # 导致定位锁死在错误坐标。策略：连续 N 次 OCR 位置未变（偏移
        # < 3px），强制改用"置信度最高"选候选，打破死锁。
        self._stale_counter = 0
        self._stale_threshold = 10

        # 局部搜索窗口：优先在上次位置附近的小窗口内 OCR。
        # 小图识别快（几十 ms）→ 定位可每帧高频更新，角色移动时
        # 坐标变动更及时；且窗口外区域（聊天框/UI 同名文本）天然
        # 被排除，误识别更少。窗口内找不到再回退全屏搜索区域。
        self._window_w = 1100
        self._window_h = 600

    # ---- 公开接口 ----

    def locate(self, frame: np.ndarray, name: str,
               min_confidence: float = 0.5,
               search_region: Optional[Tuple[int, int, int, int]] = None,
               exclude_region: Optional[Tuple[int, int, int, int]] = None
               ) -> Optional[Tuple[int, int, int, int]]:
        """在画面中查找指定名字，返回 (中心点x, 中心点y, 脚底x, 脚底y)。

        每帧都执行 OCR，但结果经过两层过滤，避免定位锁死到 UI 固定文字：
          1. exclude_region: 识别候选中心落在该区域(如左下角 UI 面板)
             内 → 直接丢弃。UI 面板上的固定同名文字（"我是立立"）永远
             不会与角色头顶/脚下的真实名字竞争。
          2. 跳变过滤(_max_jump): 与上次有效位置偏移过大(>200px)的候选
             视为误识别丢弃。角色每帧正常移动只有几十 px，不会瞬移；
             连续多帧(_skip_threshold)都无有效候选 → 判定缓存已失效
             （换图/瞬移/首帧选错），清空缓存下一帧全图重新锁定。

        冒险岛角色名字在脚下，所以:
          - 脚底 ≈ 名字中心 y（名字就在脚底位置）
          - 角色中心 = 名字中心 - character_height // 2（向上延伸人物高度一半）

        Args:
            frame:         BGR 截图 (H, W, 3)
            name:          要查找的角色名字（如 "我是立立"）
            min_confidence: 最低置信度阈值
            search_region:  可选搜索区域 (x, y, w, h)，裁剪后 OCR 更快
            exclude_region: 可选排除区域 (x, y, w, h)。候选中心落在该区域
                            内即丢弃（用于排除 UI 面板固定同名文字）

        Returns:
            (center_x, center_y, foot_x, foot_y) 或 None
        """
        name = name.strip()
        if not name:
            return None

        engine = self._get_engine()
        if engine is None:
            return None

        h, w = frame.shape[:2]

        # ---- 1. 优先在"上次位置附近的局部窗口"内搜索 ----
        # 角色每帧移动最多几十px，以上次脚底为中心裁一个小窗口即可覆盖。
        # 小图 OCR 快得多（几十 ms/次）→ 定位可每帧高频更新，角色移动
        # 时坐标变动更及时；且窗口外区域（聊天框/UI 同名文本）被天然排除。
        # 窗口内找不到（换图/瞬移/被遮挡）再回退全屏搜索区域。
        if self._last_foot is not None:
            win_w = min(self._window_w, w)
            win_h = min(self._window_h, h)
            cx, cy = self._last_foot
            x1 = max(0, min(w - win_w, cx - win_w // 2))
            y1 = max(0, min(h - win_h, cy - win_h // 2))
            roi = frame[y1:y1 + win_h, x1:x1 + win_w]
            candidates = self._match_in_roi(
                engine, roi, name, min_confidence, x1, y1, exclude_region)
            if candidates:
                chosen = self._select_candidate(candidates)
                if chosen is not None:
                    return self._commit(chosen)

        # ---- 2. 回退：全屏搜索区域 ----
        roi = frame
        offset_x, offset_y = 0, 0
        if search_region is not None:
            sx, sy, sw, sh = search_region
            x1 = max(0, sx)
            y1 = max(0, sy)
            x2 = min(w, sx + sw)
            y2 = min(h, sy + sh)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        candidates = self._match_in_roi(
            engine, roi, name, min_confidence, offset_x, offset_y, exclude_region)
        if candidates:
            chosen = self._select_candidate(candidates)
            if chosen is not None:
                return self._commit(chosen)

        # 本帧所有候选都被过滤（UI 误识别/跳变异常/名字被遮挡）
        self._skip_counter += 1
        if self._skip_counter >= self._skip_threshold:
            # 连续多帧无有效候选 → 上次位置很可能已失效（换图/瞬移/
            # 首帧选错锁死），清空缓存，下一帧按"首帧"全图重新锁定
            self._skip_counter = 0
            self._last_foot = None
            self._last_center = None
        return None

    def _select_candidate(self, candidates):
        """在候选列表中选择本帧采用的匹配。

        策略：
          - 无上次位置（首帧/缓存已清空）→ 直接取置信度最高的候选
          - 有上次位置 → 只接受"偏移 <= _max_jump"的候选（角色不会
            瞬移，跳变过大必是误识别），剩下的取置信度最高
          - 全部跳变过大 → 返回 None，由调用方计入 _skip_counter，
            连续多帧异常才接受新位置重新锁定，避免单帧误识别污染缓存

        Args:
            candidates: [(中心点, 置信度), ...]

        Returns:
            (中心点, 置信度) 或 None
        """
        last = self._last_foot
        if last is None:
            return max(candidates, key=lambda c: c[1])
        nearby = [
            c for c in candidates
            if abs(c[0][0] - last[0]) + abs(c[0][1] - last[1]) <= self._max_jump
        ]
        if nearby:
            return max(nearby, key=lambda c: c[1])
        return None

    def _commit(self, chosen):
        """把选中的候选写入缓存并返回 (center_x, center_y, foot_x, foot_y)。"""
        (name_cx, name_cy), _ = chosen
        foot = (name_cx, name_cy)
        # 角色中心 = 名字中心向上延伸"人物高度一半"（-30px）
        center = (name_cx, name_cy - self._character_height // 2)
        self._last_center = center
        self._last_foot = foot
        return (*center, *foot)

    def _match_in_roi(self, engine, roi, name, min_confidence,
                      offset_x, offset_y,
                      exclude_region: Optional[Tuple[int, int, int, int]] = None):
        """在裁剪区域 roi 内执行 OCR 并匹配名字。

        返回所有匹配候选（全局坐标），已过滤：
          - 置信度低于 min_confidence
          - 候选中心落在 exclude_region（UI 固定文字区域）内
        不做跳变过滤（由 locate() 统一处理）。

        Args:
            engine:       RapidOCR 引擎
            roi:          裁剪后的图片
            name:         角色名
            min_confidence: 最低置信度
            offset_x, offset_y: roi 左上角在整帧中的偏移
            exclude_region: 排除区域 (x, y, w, h)，候选中心落在其中则丢弃

        Returns:
            [(中心点, 置信度), ...]（全局坐标），可能为空列表
        """
        try:
            result, _ = engine(roi)
        except Exception as e:
            self._on_log(f"[定位] OCR 执行异常: {e}")
            return []
        if result is None:
            return []

        candidates = []
        for box, text, confidence in result:
            if not self._match(name, text):
                continue
            if confidence < min_confidence:
                continue

            # 名字区域中心（全局坐标）
            name_cx = int((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4) + offset_x
            name_cy = int((box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4) + offset_y

            # 排除区域内的候选（如左下角 UI 面板固定名字）→ 丢弃
            if exclude_region is not None:
                ex, ey, ew, eh = exclude_region
                if ex <= name_cx <= ex + ew and ey <= name_cy <= ey + eh:
                    continue

            candidates.append(((name_cx, name_cy), float(confidence)))

        return candidates

    def locate_all(self, frame: np.ndarray,
                   min_confidence: float = 0.5) -> list:
        """返回画面中所有识别到的文字及其坐标。

        Returns:
            [(text, (cx, cy), confidence), ...]
        """
        engine = self._get_engine()
        if engine is None:
            return []

        try:
            result, _ = engine(frame)
        except Exception as e:
            self._on_log(f"[定位] OCR 执行异常: {e}")
            return []
        if result is None:
            return []

        items = []
        for box, text, confidence in result:
            if confidence >= min_confidence:
                cx = int((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4)
                cy = int((box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4)
                items.append((text, (cx, cy), confidence))
        return items

    @property
    def ready(self) -> bool:
        """OCR 引擎是否已就绪。"""
        return self._init_ok

    # ---- 内部 ----

    def _match(self, target: str, text: str) -> bool:
        """匹配名字。

        OCR 对中文名常识别出空格/标点/单字误差（如"彤彤 是我"、
        "彤彤是我。"、"彤彤是我a"），精确比较会漏 → 定位失败。
        先归一化（去掉所有空白）再匹配，提高定位鲁棒性：
        - exact_match: 归一化后完全相等；归一化后长度差 ≤ 2 时也接受
          "包含"（OCR 在名字前后粘了 1~2 个杂字仍可命中），
          长文本（如聊天框"彤彤是我 你在哪"）因长度差大被拒绝，防误配。
        - 包含匹配: 归一化后 target in text。
        """
        text = text.strip()
        if not text or not target:
            return False
        norm = "".join(text.split())
        if self._exact_match:
            if norm == target:
                return True
            if abs(len(norm) - len(target)) <= 2:
                return target in norm
            return False
        return target in norm

    def _get_engine(self):
        """延迟初始化 RapidOCR 引擎。

        PyInstaller 打包后，rapidocr_onnxruntime 的 config.yaml 和 ONNX 模型
        需要作为数据文件包含在 _internal/rapidocr_onnxruntime/ 目录下。
        如果初始化失败，会尝试在 sys._MEIPASS 下查找缺失的文件并给出诊断信息。
        """
        if self._engine is not None:
            return self._engine if self._engine is not False else None

        self._on_log("[定位] 正在初始化 RapidOCR ...")
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._init_ok = True
            self._on_log("[定位] RapidOCR 初始化完成")
            return self._engine
        except Exception as e:
            self._on_log(f"[定位] RapidOCR 初始化失败: {e}")
            self._engine = False

            if getattr(sys, "frozen", False):
                self._diagnose_frozen_ocr()

            return None

    def _diagnose_frozen_ocr(self):
        """PyInstaller 冻结模式下的 OCR 诊断。

        检查 sys._MEIPASS 下 rapidocr_onnxruntime 包的关键数据文件
        是否存在，给出缺失文件列表，帮助定位打包遗漏。
        """
        meipass = getattr(sys, "_MEIPASS", "")
        if not meipass:
            return

        pkg_dir = os.path.join(meipass, "rapidocr_onnxruntime")
        if not os.path.isdir(pkg_dir):
            self._on_log(f"[定位] 诊断: _internal 下未找到 rapidocr_onnxruntime 目录")
            return

        required = [
            "config.yaml",
            os.path.join("models", "ch_PP-OCRv4_det_infer.onnx"),
            os.path.join("models", "ch_PP-OCRv4_rec_infer.onnx"),
            os.path.join("models", "ch_ppocr_mobile_v2.0_cls_infer.onnx"),
        ]
        missing = [f for f in required if not os.path.isfile(os.path.join(pkg_dir, f))]
        if missing:
            self._on_log(f"[定位] 诊断: 缺失数据文件: {missing}")
            self._on_log("[定位] 修复: 重新打包时添加 --collect-data rapidocr_onnxruntime")
        else:
            self._on_log("[定位] 诊断: 数据文件完整，初始化失败可能是其他原因")