"""主程序入口与自动打怪主循环。

================================================================================
架构概览（三层架构）
================================================================================

  感知层 (perception/)         决策层 (decision/)        执行层 (execution/)
  ┌──────────────────┐       ┌──────────────────┐     ┌──────────────────┐
  │ ScreenCapture    │──帧──▶│ Context          │     │ ActionExecutor   │
  │  (截图)          │       │  (数据载体)       │     │  (动作聚合)      │
  │                  │       │                  │     │                  │
  │ YoloDetector     │──框──▶│ DecisionEngine   │──▶│ KeyboardController│
  │  (YOLO检测)      │       │  (反应式决策)     │     │  (按键注入)      │
  │                  │       │                  │     │                  │
  │ detect_bar_ratio │──比─▶│ 同平台追击/爬绳  │     │ MouseController  │
  │  (HP/MP检测)     │       │ 下落/探索/技能   │     │  (鼠标注入)      │
  │                  │       │                  │     │                  │
  │ OCRNameLocator   │──坐─▶│ self_position    │     │                  │
  │  (名字定位)       │       │  (自身坐标)      │     │                  │
  │ PlayerTracker    │──框──▶│                  │     │                  │
  │  (外观模板跟踪)   │       │                  │     │                  │
  └──────────────────┘       └──────────────────┘     └──────────────────┘

================================================================================
数据流（每帧）
================================================================================

  1. ScreenCapture.grab()          → frame (numpy BGR 数组)
  2. YoloDetector.detect(frame)    → [Detection, ...]  (怪物/地板/绳索)
  3. detect_bar_ratio(hp_region)   → hp_ratio (0.0~1.0)
  4. detect_bar_ratio(mp_region)   → mp_ratio (0.0~1.0)
  5. OCRNameLocator.locate(frame)  → self_position (cx, cy) 或 None
  5. PlayerTracker.locate(frame)     → self_position (box, conf) 或 None
  6. Context(monsters, floors, ..., hp_ratio, mp_ratio, self_position)
     自身定位优先级: OCR 名字 → 外观模板
  7. DecisionEngine.decide(ctx)    → 反应式决策（同平台追击/爬绳/下落/探索/技能）
  8. on_frame(frame, ...)          → 预览回调（UI 渲染检测框）

================================================================================
自身定位策略
================================================================================

  1. RapidOCR 文字识别（主方案，每帧实时执行）
     原理: OCR 识别画面中的角色名字 → 名字中心 ≈ 脚底；
           角色中心点 = 名字中心向上延伸"人物高度一半"（character_height//2，约 30px）
     条件: 已配置 self_name；每帧执行，与 YOLO 检测同节奏。
     优点: 精度最高。
     缺点: 名字被地图/UI 面板挡住时失效。

  2. PlayerTracker 外观模板跟踪（次方案，名字被遮挡时启用）
     原理: 用户截取角色全身图作为模板 → cv2.matchTemplate 多尺度匹配 →
           一旦锁定，下一帧只在附近局部搜索（加速）。
     条件: 模板已上传（exe 界面"上传角色全身照"，保存位置由 resolve_template_path()
           统一解析，即 exe 旁边 / 项目根目录的 assets/templates/）；换时装/换地图需重新上传。
     优点: 不依赖名字 OCR，角色下半身被地图挡住也能定位。
     缺点: 依赖外观特征，时装/姿势变化大时可能匹配不上。

  说明: 早期"HP 条偏移推算"用的是 UI 底部固定血条（hp_region），
        不随角色移动，算出的坐标恒定不变（定位 bug），已移除。
        阴影检测方案已废弃（冒险岛角色脚底无固定阴影，无法稳定定位）。

================================================================================
运行方式
================================================================================

  GUI 模式:  python main.py
  CLI 模式:  python -m src.main
"""
import time
import threading
from typing import Callable, Optional, Any, Tuple

import numpy as np

from .perception.screen_capture import ScreenCapture
from .perception.yolo_detector import Detector, create_detector
from .perception.hp_mp_detector import detect_bar_ratio
from .perception.ocr_name_locator import OCRNameLocator
from .perception.player_tracker import PlayerTracker
from .execution.action_executor import ActionExecutor
from .decision.context import Context, DecisionEngine
from .utils.config_loader import Config, resolve_model_path, resolve_template_path


class Automation:
    """自动打怪主循环控制器。

    【职责】
    整合三层架构，在独立线程中循环执行"截图 → 检测 → 决策 → 执行"。

    【线程模型】
    - 主线程: PyQt5 GUI 事件循环
    - 工作线程: Automation._loop() 运行控制循环
    - 通过回调 (on_log, on_frame) 把结果推回主线程

    【生命周期】
    1. 构造 Automation(config, detector, on_log, on_frame)
    2. lock_window(title)    锁定游戏窗口
    3. start()               启动工作线程
    4. stop()                停止工作线程
    5. unlock_window()       释放窗口

    Args:
        config:   全局配置对象（窗口/检测/按键/技能等）
        detector: YOLO 检测器实例，None 时自动创建 MockDetector
        on_log:   日志回调，参数 (message: str)
        on_frame: 预览回调，参数 (frame, detections, hp_ratio, mp_ratio)
    """

    def __init__(self, config: Config, detector: Optional[Detector] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_frame: Optional[Callable[..., None]] = None):
        # ---- 感知层 ----
        self.config = config
        self.capture = ScreenCapture()  # 窗口截图（客户区 BitBlt）

        # 检测器：如果传了就用，否则根据配置自动创建（模型不存在时回退 Mock）
        self.model_path = resolve_model_path(config.model_path)
        self.detector = detector if detector is not None else create_detector(
            self.model_path, config.confidence, on_log or (lambda m: None)
        )

        # ---- 执行层 ----
        # 聚合键盘 + 鼠标控制器；on_log 用于上报按键注入结果（是否已锁定窗口等）
        # 使用 SendInput 驱动层模拟真实全局按键，游戏窗口必须在前台
        self.executor = ActionExecutor(
            on_log=on_log or (lambda m: None),
        )

        # ---- 决策层 ----
        # 反应式决策引擎：基于 YOLO 画面实时决策，不需要地图
        self.engine = DecisionEngine(
            config, self.executor,
            on_log=on_log or (lambda m: None)
        )

        # ---- 回调 ----
        self.on_log = on_log or (lambda m: None)
        self.on_frame = on_frame or (lambda f, d, h, m: None)

        # ---- 线程控制 ----
        self._running = False  # 控制循环是否继续
        self._thread = None    # 工作线程对象
        self._frame_count = 0  # 帧计数器（用于限频日志）
        self._grab_fail_count = 0  # 连续截图失败计数（限频日志用）

        # ---- OCR 名字定位器（延迟初始化，首次使用时才加载模型）----
        # ocr_interval=1: 每帧都执行 OCR（和 YOLO 一样），位置实时更新
        # character_height=60: 人物高度约 60px，角色中心 = 名字中心 - 高度一半（向上）
        self._ocr = OCRNameLocator(
            character_height=60, ocr_interval=1,
            exact_match=True,  # 精确匹配角色名，避免"包含匹配"误识别聊天框/UI文字
            on_log=self.on_log,
        )

        # ---- 外观模板跟踪器（名字被地图/UI 遮挡时的次方案）----
        # 模板路径由 resolve_template_path() 统一解析（exe 旁边 / 项目根目录），
        # 即界面"上传角色全身照"写入的位置，不在代码里写死字符串。
        self._player_template_path = resolve_template_path()
        try:
            self._player = self._create_player_tracker(self._player_template_path)
            self.on_log("[模板] 角色外观模板加载成功")
        except FileNotFoundError:
            self._player = None
            # 兜底策略：没有模板时看是否有名字可定位
            if self.config.self_name:
                self.on_log(
                    "[模板] 未找到角色模板，名称遮挡时将无法定位"
                    "（可在界面点\"上传角色全身照\"补齐）"
                )
            else:
                self.on_log(
                    "[模板] 未找到角色模板，且未配置自身名字，"
                    "自身定位不可用（请填写自身名字或点\"上传角色全身照\"）"
                )

        # ---- 角色位置缓存 ----
        self._cached_center: Optional[Tuple[int, int]] = None   # 中心点（OCR 时才有）
        self._last_foot_pos: Optional[Tuple[int, int]] = None   # 脚底坐标（OCR/模板共用）
        self._locate_method: Optional[str] = None               # 当前定位方式: ocr / template
        self._locate_failed_notified = False                    # 定位失败兜底提示是否已发出（限频用）

        # ---- HP/MP 变化追踪 ----
        self._last_hp_ratio: Optional[float] = None
        self._last_mp_ratio: Optional[float] = None

        # ---- 自动拾取 ----
        self._last_pickup_time = 0.0  # 上次拾取按键的时间戳

    # =========================================================================
    # 窗口管理
    # =========================================================================

    def list_windows(self):
        """枚举所有可见窗口，返回 [(hwnd, title), ...]。

        用于 UI 下拉框选择要锁定的窗口。
        """
        return self.capture.list_windows()

    def lock_window(self, title: str) -> str:
        """按标题锁定游戏窗口。

        锁定后：
        1. ScreenCapture 可以截取该窗口画面
        2. ActionExecutor 的 SendInput 会注入到该窗口

        Returns:
            锁定后的窗口标题（用于确认）
        """
        locked = self.capture.lock(title=title)
        # 把窗口句柄传给执行层，这样按键/鼠标消息会发到游戏中
        self.executor.set_target_window(self.capture.hwnd)
        return locked

    def unlock_window(self):
        """释放窗口锁定。"""
        self.capture.unlock()

    @property
    def window_locked(self):
        return self.capture.locked

    def get_window_rect(self):
        """获取窗口在屏幕中的矩形 (left, top, width, height)。"""
        return self.capture.get_rect() if self.capture.locked else None

    # =========================================================================
    # 检测器管理
    # =========================================================================

    def set_detector(self, detector: Detector):
        """运行时替换检测器（切换模型时用）。"""
        self.detector = detector

    # =========================================================================
    # 角色外观模板管理
    # =========================================================================

    def _create_player_tracker(self, template_path: str) -> PlayerTracker:
        """创建外观模板跟踪器（统一参数，避免多处重复）。"""
        # 截图定位置信度从配置读取（config/user.yaml 的 template_confidence），
        # 局部搜索和全图搜索共用同一个阈值，避免两套阈值导致全图搜索被
        # 卡死。默认 0.55：真实角色因缩放/朝向/光照/部分遮挡，匹配分数
        # 常在 0.55 上下，配合 exclude_bottom 排除底部 UI 头像，够用且
        # 不易误匹配 UI。
        conf = float(getattr(self.config, "template_confidence", 0.55))
        return PlayerTracker(
            template_path=template_path,
            threshold=conf,
            search_margin=200,
            max_miss=8,
            full_threshold=conf,
        )

    def set_player_template(self, path: str) -> None:
        """运行时更换角色外观模板（UI 上传新截图时调用）。

        换时装/换地图后，重新上传角色全身照即可立即生效，无需重启程序。

        Args:
            path: 模板图片的绝对路径（png/jpg/bmp）

        Raises:
            FileNotFoundError: 图片不存在或无法解码
        """
        if self._player is None:
            self._player = self._create_player_tracker(path)
        else:
            self._player.set_template(path)
        self.on_log(f"[模板] 角色外观模板已更新: {path}")

    # =========================================================================
    # 主循环控制
    # =========================================================================

    @property
    def running(self):
        """是否正在运行中。"""
        return self._running

    def start(self):
        """启动自动打怪主循环。

        在独立线程中运行 _loop()，不阻塞 UI 线程。

        Raises:
            RuntimeError: 未锁定窗口时调用
        """
        if self._running:
            return
        if not self.capture.locked:
            raise RuntimeError("请先锁定游戏窗口")

        # UI 可能修改了配置，同步到决策引擎
        self.engine.update_config(self.config)
        self.engine.reset()  # 清空技能轮转索引、冷却记录

        self._running = True
        # daemon=True: 主线程退出时自动结束，不会卡住进程
        self._last_foot_pos = None
        self._cached_center = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.on_log("[启动] 自动打怪已开始")

    def stop(self):
        """停止自动打怪。

        设置 _running = False，_loop() 会在下一次迭代时退出。
        同时释放所有按住的移动键，防止停止后方向键卡住。
        """
        if not self._running:
            return
        self._running = False
        self.engine.release_keys()  # 释放按住的方向键/上键
        if self._player is not None:
            self._player.reset()     # 重置外观跟踪器
        self._last_foot_pos = None   # 重置脚底位置缓存
        self.on_log("[停止] 自动打怪已停止")

    def _loop(self):
        """主循环（在独立线程中运行）。

        每帧执行:
          1. 截图（grab）
          2. YOLO 检测（detect）→ 按类别过滤
          3. HP/MP 检测（颜色数像素）→ 比例
          4. 自身定位（HP条偏移 / OCR 名字识别）→ 坐标
          5. 组装 Context → DecisionEngine.decide()
          6. 预览回调 → UI 渲染

        FPS 控制: 通过 time.sleep() 补齐到 1/fps 秒
        """
        # 每帧的目标间隔时间（秒）
        interval = 1.0 / max(1, self.config.fps)

        while self._running:
            self._frame_count += 1

            # ---- 全局异常兜底 ----
            # 任何一步抛异常（截图/检测/OCR/决策）都不允许静默崩溃线程，
            # 必须记录 traceback 到日志，便于定位问题。
            try:
                self._loop_frame(interval)
            except Exception:
                import traceback
                self.on_log("[错误] 主循环异常(已捕获, 继续运行):")
                for line in traceback.format_exc().splitlines():
                    self.on_log(f"  {line}")
                time.sleep(interval)

    def _loop_frame(self, interval: float):
        """单帧执行：截图→检测→HP/MP→定位→决策→预览。"""
        t0 = time.time()  # 帧开始时间
        try:
            frame = self.capture.grab()
            self._grab_fail_count = 0  # 截图成功，重置失败计数
            # 首次截图记录帧尺寸 + DPI 诊断
            if self._frame_count == 1:
                h, w = frame.shape[:2]
                self.on_log(f"[帧尺寸] {w}x{h}")
                self.on_log(
                    f"[配置] 参考分辨率: "
                    f"{self.config.reference_width}x{self.config.reference_height}"
                )
                # DPI 诊断：对比 GetClientRect 与实际帧尺寸
                try:
                    cw, ch = self.capture.get_client_rect()
                    if cw != w or ch != h:
                        self.on_log(
                            f"[DPI警告] GetClientRect={cw}x{ch} "
                            f"与实际帧 {w}x{h} 不一致，"
                            f"可能存在 DPI 缩放偏移！"
                        )
                    else:
                        self.on_log(
                            f"[DPI] 客户区尺寸匹配 ({cw}x{ch})，坐标应无偏移"
                        )
                except Exception:
                    pass
        except Exception as e:
            # 窗口刚锁定/最小化恢复的过渡期，grab 会短暂失败。限频日志：
            # 只在首次失败、以及每连续 30 次失败时打印，避免刷屏几十条。
            self._grab_fail_count += 1
            if self._grab_fail_count == 1 or self._grab_fail_count % 30 == 0:
                self.on_log(
                    f"[错误] 截图失败(连续{self._grab_fail_count}次): {e}"
                )
            time.sleep(interval)
            return

        # ---- 2. YOLO 检测 ----
        # detect() 返回 [Detection, ...]，每个 Detection 包含:
        #   cls_name, confidence, x, y, w, h, center
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            self.on_log(f"[错误] 检测失败: {e}")
            detections = []

        # 按类别名过滤（配置中可能用逗号分隔多个类别名）
        monster_classes = self._monster_classes()
        monsters = [d for d in detections if d.cls_name in monster_classes]
        floors = [d for d in detections if d.cls_name in self._floor_classes()]
        ropes = [d for d in detections if d.cls_name in self._rope_classes()]

        # 每 30 帧输出一次地图元素概览（YOLO 基于当前截图分析的结果）
        if self._frame_count % 30 == 0:
            parts = []
            if monsters:
                coords = ", ".join(
                    f"({d.center[0]},{d.center[1]})" for d in monsters
                )
                parts.append(f"怪{len(monsters)}只:{coords}")
            if floors:
                parts.append(f"平台{len(floors)}个")
            if ropes:
                parts.append(f"绳索{len(ropes)}条")
            if parts:
                self.on_log("[地图] " + " | ".join(parts))

        # ---- 3. HP 检测 ----
        # scale_region(): 把参考分辨率下的坐标缩放到当前帧的实际像素
        # 这样同一个配置文件兼容不同窗口大小
        hp_region = self.config.scale_region(
            self.config.hp_region, frame.shape[1], frame.shape[0]
        )
        # detect_bar_ratio(): 多方法融合检测（边缘→亮度→颜色）
        hp_ratio = detect_bar_ratio(
            frame, hp_region,
            tuple(self.config.hp_color) if self.config.hp_color else None,
            self.config.hp_tolerance,
        )

        # ---- 4. MP 检测 ----
        mp_region = self.config.scale_region(
            self.config.mp_region, frame.shape[1], frame.shape[0]
        )
        mp_ratio = detect_bar_ratio(
            frame, mp_region,
            tuple(self.config.mp_color) if self.config.mp_color else None,
            self.config.mp_tolerance,
            expand=3,  # MP 条下方是同色蓝色面板，减小扩展避免背景混入
        )

        # ---- 5. 自身定位 ----
        self_pos = self._locate_self(frame)

        # ---- HP/MP 变化检测 ----
        if hp_ratio is not None:
            if self._last_hp_ratio is None or abs(hp_ratio - self._last_hp_ratio) >= 0.05:
                self._last_hp_ratio = hp_ratio
                self.on_log(f"[HP] {hp_ratio:.0%}")
        if mp_ratio is not None:
            if self._last_mp_ratio is None or abs(mp_ratio - self._last_mp_ratio) >= 0.05:
                self._last_mp_ratio = mp_ratio
                self.on_log(f"[MP] {mp_ratio:.0%}")

        # 每 30 帧输出一次状态
        if self._frame_count % 30 == 0:
            # HP/MP 比例
            hp_str = f"{hp_ratio:.0%}" if hp_ratio is not None else "N/A"
            mp_str = f"{mp_ratio:.0%}" if mp_ratio is not None else "N/A"

            # 自身坐标
            center = self._get_last_center()
            if self_pos:
                if center:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"中心:({center[0]},{center[1]}) "
                        f"脚底:({self_pos[0]},{self_pos[1]})"
                    )
                else:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"脚底:({self_pos[0]},{self_pos[1]})"
                    )
            else:
                self.on_log(f"[状态] HP={hp_str} MP={mp_str} 自身未定位")

        # ---- 6. 决策与执行 ----
        # Context 是感知层 → 决策层的数据载体
        ctx = Context(
            monsters=monsters,
            floors=floors,
            ropes=ropes,
            self_position=self_pos,     # 自身脚底坐标 (cx, cy) 或 None
            self_center=self._get_last_center(),  # 角色中心点（距离推算用）
            hp_ratio=hp_ratio,          # 0.0~1.0
            mp_ratio=mp_ratio,          # 0.0~1.0
            detections=detections,      # 全部检测结果（供调试/日志用）
        )
        self.engine.decide(ctx)  # 决策引擎根据上下文执行动作

        # ---- 6.5. 自动拾取（定时按拾取键，每秒N次）----
        self._auto_pickup()

        # ---- 7. 预览回调 ----
        # 把 frame 和检测结果推给 UI 线程渲染
        self.on_frame(frame, detections, hp_ratio, mp_ratio)

        # ---- 8. FPS 控制 ----
        elapsed = time.time() - t0
        if elapsed < interval:
            # 帧太快，sleep 补齐
            time.sleep(interval - elapsed)

    # =========================================================================
    # 类别名解析（从配置的逗号分隔字符串 → 列表）
    # 例如: "monster" → ["monster"]
    #       "monster,boss" → ["monster", "boss"]
    # =========================================================================

    def _auto_pickup(self):
        """自动拾取：按配置的间隔定时触发拾取键。

        仅在 pickup_enabled=True 且已锁定窗口时生效。
        间隔由 config.pickup_interval 控制（默认 0.333s = 每秒3次）。
        """
        enabled = getattr(self.config, "pickup_enabled", False)
        if not enabled:
            return
        if not self.capture.locked:
            return
        now = time.time()
        interval = getattr(self.config, "pickup_interval", 0.333)
        if now - self._last_pickup_time >= interval:
            pickup_key = getattr(self.config, "pickup_key", "z")
            self.executor.press_key(pickup_key, cooldown=0.0)
            self._last_pickup_time = now

    def _monster_classes(self):
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def _floor_classes(self):
        return [c.strip() for c in self.config.floor_classes.split(",") if c.strip()]

    def _rope_classes(self):
        return [c.strip() for c in self.config.rope_classes.split(",") if c.strip()]

    # =========================================================================
    # 自身定位
    # =========================================================================

    def set_self_name(self, name: str):
        """运行时更新自身名字（UI 输入框改动时调用）。"""
        self.config.self_name = name

    def _locate_self(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """定位自身脚底在画面中的坐标。

        策略（优先级）：
          1. RapidOCR 名字定位 —— 主方案。角色名字显示在脚下，OCR 识别
             名字中心即脚底，精度最高、最稳定。涂黑底部 UI 防 OCR 认到
             左下角固定 UI 里的同名文字（等级面板里的"我是立立"）。
          2. 外观模板跟踪（截图全身匹配）—— 辅助方案，名字被地图遮挡时
             启用。角色站在地图底部时脚底 y 可达 640~660（地图底部边缘），
             不能用"y > 0.82h"判定为误匹。

        说明:
          角色在地图底部边缘站立时（名字刚好被地图底部挡住）只能靠截图
          定位。底部 UI 没有"角色头像"等和模板撞脸的元素，所以截图匹
          配不会受 UI 干扰——之前的"误匹到 UI 头像"是误判。

        Returns:
            (cx, cy) 脚底坐标，None 表示本帧所有方法都失败
        """
        h, w = frame.shape[:2]

        # ---- 1. OCR 名字定位（主方案）----
        # 涂黑底部 UI 条（全宽度），避免 OCR 认到左下角固定 UI 里的同名
        # 文字。search_region 也排除底部 18%，但同时允许结果落在更下方
        # ——角色站在地图底部时脚底就在 640~660，OCR 找到名字也要采信。
        search_region = (0, int(h * 0.10), w, int(h * 0.72))
        ocr_frame = frame.copy()
        mask_h = int(h * 0.18)
        ocr_frame[h - mask_h:h, 0:w] = 0

        result = self._ocr.locate(ocr_frame, self.config.self_name,
                                  search_region=search_region)
        if result is not None:
            center_x, center_y, foot_x, foot_y = result
            # 仅做"明显不合理"的检查：脚底超出画面就丢弃
            if foot_y < 0 or foot_y >= h or foot_x < 0 or foot_x >= w:
                self.on_log(
                    f"[定位] OCR 结果 ({foot_x},{foot_y}) 越界，丢弃"
                )
            else:
                self._cached_center = (center_x, center_y)
                self._last_foot_pos = (foot_x, foot_y)
                self._locate_failed_notified = False  # 定位成功，复位失败提示标记
                if self._locate_method != "ocr":
                    self._locate_method = "ocr"
                    self.on_log(f"[定位] 名称识别成功, 切换为名称定位 ({foot_x},{foot_y})")
                # OCR 命中时回填模板位置，让截图下一帧能续上
                if self._player is not None:
                    bw, bh = self._player.tw, self._player.th
                    px = max(0, foot_x - bw // 2)
                    py = max(0, int(foot_y - bh * 0.9))
                    self._player.last_box = (px, py, bw, bh)
                    self._player.miss_count = 0
                return (foot_x, foot_y)

        # ---- 2. 外观模板跟踪（辅助方案：名字被地图遮挡时）----
        # 角色可以合法地站在地图底部边缘（脚底 y 可达 640~660），
        # 不要用 y 阈值或底部排除把这种合法位置过滤掉。
        if self._player is not None:
            box = self._player.locate(frame)
            if box is not None:
                x, y, bw, bh, conf = box
                foot_x = int(x + bw / 2)
                foot_y = int(y + bh)
                # 仅检查"明显不合理"：匹配框出界 或 框太大（>200，疑似整张图）
                if foot_y < 0 or foot_y >= h or foot_x < 0 or foot_x >= w:
                    self._player.reset()
                elif bh > 200:
                    self.on_log(
                        f"[定位] 模板结果 bh={bh} 过大（疑似整张图），丢弃"
                    )
                    self._player.reset()
                else:
                    # 跳变校验：与上次有效位置偏移过大 → 模板误匹配
                    jumped = False
                    if self._last_foot_pos is not None:
                        lfx, lfy = self._last_foot_pos
                        if abs(foot_x - lfx) > 250 or abs(foot_y - lfy) > 150:
                            self._player.reset()
                            jumped = True
                    if not jumped:
                        self._cached_center = (int(x + bw / 2), int(y + bh / 2))
                        self._last_foot_pos = (foot_x, foot_y)
                        self._locate_failed_notified = False  # 定位成功，复位失败提示标记
                        if self._locate_method != "template":
                            self._locate_method = "template"
                            self.on_log(
                                f"[定位] 名称被遮挡, 切换为截图定位 "
                                f"({foot_x},{foot_y}) 置信度 {conf:.2f}"
                            )
                        return (foot_x, foot_y)

        # ---- 3. 兜底：两种方法都失败 ----
        # 首次失败时给一次明确提示（避免每帧刷屏），之后静默返回 None。
        if not self._locate_failed_notified:
            self._locate_failed_notified = True
            if self._player is None and not self.config.self_name:
                self.on_log(
                    "[定位] 无法定位自身：未配置名字且未上传模板，"
                    "请填写自身名字或点\"上传角色全身照\""
                )
            else:
                self.on_log(
                    "[定位] 当前帧定位失败（名字被遮挡/模板未匹配），"
                    "将尝试下一帧重新定位"
                )
        return None

    def _get_last_center(self) -> Optional[Tuple[int, int]]:
        """获取最近一次定位到的角色中心点（供日志用）。"""
        return self._cached_center


def main():
    """CLI 入口（无 GUI）：加载配置并启动主循环，按 Ctrl+C 退出。

    用法: python -m src.main
    """
    from .utils.logger import get_logger
    log = get_logger()

    cfg = load_config()
    if not cfg.window_title:
        log.error("未配置 window_title，请在 config/user.yaml 中设置")
        return

    auto = Automation(cfg, on_log=lambda m: log.info(m))
    try:
        locked = auto.lock_window(cfg.window_title)
        log.info(f"已锁定窗口: {locked}")
    except Exception as e:
        log.error(f"锁定窗口失败: {e}")
        return

    auto.start()
    try:
        while auto.running:
            time.sleep(1)
    except KeyboardInterrupt:
        auto.stop()


if __name__ == "__main__":
    from .utils.config_loader import load_config
    main()