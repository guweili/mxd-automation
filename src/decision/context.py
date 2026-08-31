"""决策上下文与反应式决策引擎。

================================================================================
设计理念
================================================================================

  不再建地图、不做 A* 寻路。每一帧只看 YOLO 检测到的画面内容，
  像人类玩家一样"看到什么就做什么反应"。

  画面里有什么 → 就应该做什么:
    - 看到怪 → 判断同平台还是跨平台，走过去或爬绳跳下去
    - 看到地板 → 知道哪里能站
    - 看到绳索 → 知道哪里能爬
    - 没看到怪 → 往一个方向走探索
    - HP/MP 低 → 加血加蓝

================================================================================
架构
================================================================================

  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────┐
  │  感知层      │ →  │  DecisionEngine     │ →  │ ActionExec   │
  │ YOLO/OCR/HP │    │  (反应式决策 + FSM)  │    │ (方向键/技能) │
  └─────────────┘    └─────────────────────┘    └──────────────┘

================================================================================
决策流程（优先级从高到低）
================================================================================

  1. HP 低于阈值 → 加血键
  2. MP 低于阈值 → 加蓝键
  3. 检测到怪物:
     a. 同平台 → 按住方向键走过去，进入 200px 攻击范围后停止移动原地攻击
     b. 怪在上方 + 有绳索 → 走到绳索正下方，跳跃 + 按住上键爬绳
     c. 怪在下方/跨平台 → 按住方向键移动 + 按需跳跃
     d. 都不满足 → Tab 选怪 + 原地攻击
  4. 没怪 → 探索（按住方向键往一个方向走，遇坑跳）

【移动方式】所有移动（追击/攀爬/探索）都是"按住方向键不松手"，
  进入攻击范围或攻击时才释放方向键，攻击期间完全不移动。

================================================================================
Context 字段说明
================================================================================

  monsters:       YOLO 检测到的怪物列表
  floors:         地板列表
  ropes:          绳索列表
  self_position:  自身脚底坐标 (cx, cy) 或 None
  self_center:    自身角色中心点坐标 (cx, cy) 或 None（OCR 定位时记录）
  hp_ratio:       血量比例 0.0~1.0
  mp_ratio:       蓝量比例 0.0~1.0
  detections:     全部 YOLO 检测结果（含所有类别）
"""
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from ..perception.yolo_detector import Detection
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config
from .fsm import FSM, State
from .distance import (
    estimate_path_distance,
    JUMP_HEIGHT,
    PLATFORM_JUMP_GAP_X,
    SAME_LEVEL_Y_TOLERANCE,
    ROPE_REACH_Y,
    PathEstimate,
)


# =============================================================================
# 反应式决策的阈值常量
# =============================================================================

ATTACK_RANGE_X = 200
"""攻击范围默认值：自身与怪物 X 坐标差小于此值才开始攻击（像素）。

仅长手(远程)使用：实际生效值取 config.attack_range
（exe 界面"攻击距离px"可修改），此常量仅作 __init__ 里的兜底默认值。
短手(近战)固定 MELEE_ATTACK_RANGE_X=50，不读此配置。
"""

MELEE_ATTACK_RANGE_X = 50
"""短手(近战)固定贴脸距离（像素），【不允许配置修改】。

短手与长手攻击距离完全独立：长手读 exe 配置(attack_range)，
短手固定 50px。贴脸判定用"角色到怪近侧身体边缘"的距离，
宽怪站旁边就能打，不穿越怪身体。
"""

ROPE_SEARCH_RANGE_X = 200
"""搜索绳索的水平范围（像素）"""

CLIMB_ALIGN_TOLERANCE = 5
"""攀爬对准容差：人物中心与绳索中心 X 差小于此值（±5px）
视为在同一竖直轴线（绳索正下方），才允许抓绳攀爬（像素）"""

CLIMB_EXIT_FRAMES = 45
"""爬绳结束后横向走出绳索的帧数（约 2 秒 @20fps），期间不重新抓绳"""

ATTACK_STALE_FRAMES = 90
"""锁定同一目标持续攻击的最大帧数（约 4.5 秒 @20fps）。

怪物死亡后 YOLO 仍可能把尸体/消失残影检测为 monster，
位置匹配会一直锁定这个残影，导致角色原地打空气、不换下一只。
超过该帧数目标仍未消失（画面仍检测到）→ 判定为残影/无敌，
立即解除锁定重新选目标。"""

STUCK_FRAMES = 60
"""卡住判定帧数：持续此帧数位置不变则视为卡住"""

EXPLORE_DIRECTION_SWITCH_FRAMES = 180
"""探索方向切换帧数：探索状态下持续此帧数没遇到怪就换方向"""

LOST_DIRECTION_SWITCH_MIN = 1.0
LOST_DIRECTION_SWITCH_MAX = 5.0
"""迷失恢复方向切换时长的随机范围（秒）。

自身定位失败（迷失）时，探索状态走"随机左右走 + 随机跳"的恢复策略。
每次换方向都随机抽一个持续秒数（1~5 秒），且方向随机抽 left/right，
避免角色无脑往一个方向一路跑出屏幕。按秒计时（用 time.time() 时间戳），
与帧率无关，行为稳定可预期。
"""

LOST_JUMP_INTERVAL_MIN = 1.0
LOST_JUMP_INTERVAL_MAX = 3.5
"""迷失恢复跳跃间隔的随机范围（秒）。

未定位时按随机间隔按跳跃键：角色名字在脚下，被地图/UI 遮挡时
OCR 看不到，跳起来能让名字/身体从遮挡层露出来，便于重新定位。
按秒计时；press_key 自带 cooldown 兜底限频。
"""

DISTANCE_LOG_FRAMES = 60
"""距离推算日志输出间隔帧数（避免刷屏）"""

FACE_TURN_X = 20
"""攻击转向判定：怪物中心 x 与角色 x 差超过此值才调整朝向（像素）。
小于此值视为怪物在正下方/重叠，保持当前朝向即可。"""

MELEE_HIT_TOL_X = 10
"""短手贴脸"命中容差"（像素）。

攻击距离 50px 指攻击特效的有效命中距离。角色到怪近侧身体边缘
≤ 攻击距离+此容差 才判定"能打到"、站定攻击；容差覆盖攻击特效
判定框冗余 + 检测抖动（±10px），避免怪在边缘 50~60px 处时
"判定未贴脸→追→怪又贴近→又未贴脸"的抖动。
超过此容差(>60px)攻击特效够不到，必须追击，不停在原地空打。
"""

MELEE_HYSTERESIS_X = 40
"""短手(近战)攻击中"确认怪物走远"的防抖窗口宽度（像素）。

贴脸命中区 = 攻击距离 + MELEE_HIT_TOL_X（例: 50+10=60px）。
攻击中角色到怪近侧边缘距离超过命中区、但 ≤ 攻击距离+此窗口
（60~90px）时，先用防抖帧数 MELEE_LEAVE_FRAMES 吸收瞬时抖动
（攻击位移/怪物被推开 1~3 帧），连续超限才转追击——避免
"怪被推一下就走远 → 追 → 怪弹回 → 追"的来回抖动。
超过此窗口(>90px)说明怪真走远或已切换目标，立即追击，不停在
原地对着够不着的怪空打。
"""

MELEE_LEAVE_FRAMES = 6
"""短手(近战)攻击中连续超出滞回窗口的帧数阈值（约 0.3 秒 @20fps）。

超过滞回窗口后不立即切追击：攻击位移/怪物被推开通常是 1~3 帧的
瞬时抖动，连续超出该帧数才判定"怪物真走远了"转为追击，避免攻击
状态 1 帧就断、攻击断断续续。防抖期间原地继续攻击（不移动）。
"""

MELEE_EDGE_MAX_HALF_W = 50
"""短手贴脸判定中怪半宽的上限（像素）。

近战攻击命中怪身体任意部位即可，贴脸判定用"角色到怪近侧身体边缘"
的距离（= 怪中心距 - 怪半宽）。普通怪 bbox 半宽约 25~70px，直接减
半宽会让超宽怪（boss/大怪）离老远就判定贴脸；封顶 50px 后最大有效
贴脸距离 = 攻击距离 + 50，既消除大怪穿越、又不会离远就空打。
"""

MELEE_FACE_DEADZONE_X = 25
"""短手攻击前转向的"重叠死区"（像素）。

先扭头再攻击：攻击前判定怪在角色哪一侧，背对怪就按方向键转身再
施法，避免朝反方向空打。但转身按方向键会让角色朝怪移动一小步，
若角色已与怪身体重叠/极近（角色到怪近侧边缘 ≤ 死区），转身会穿过
怪身体造成"左右来回顶"——此时保持当前朝向直接攻击（怪就在身前/
身侧，攻击可命中），不转向。
"""

OCCLUSION_HALF_WIDTH_X = 40
"""遮挡判定水平容差（像素）。

短手贴脸攻击时角色站在怪正前方，角色模型+攻击特效会遮住怪物，
YOLO 置信度骤降被 conf 阈值过滤 → 本帧看不到锁定目标。
角色脚底 x 与怪中心 x 的最大距离约等于攻击距离(attack_range)，
再加角色半宽+怪半宽+检测抖动容差(40px)即视为"角色在怪跟前"，
目标本帧消失 → 判定为被角色遮挡而不是怪真消失。
"""

OCCLUSION_MAX_FRAMES = 300
"""目标被遮挡时沿用最后已知位置的帧数上限（约 10 秒 @30fps）。

超过该帧数仍未被 YOLO 重新看到 → 判定怪真消失（被击退/逃出
画面/已死亡），放弃攻击。正常近战 2~4 秒内击杀或怪露出，
但厚血怪/被击退后角色追错位置时遮挡可持续更久；若上限太短
(150=5 秒)会出现"厚血怪被遮 5 秒 → 判定消失 → 转探索/换远处
目标 → 角色离开 → 怪露出 → 重新贴脸 → 又被遮"的反复空转，
表现为在怪物堆里到处跑不攻击。
"""

SELF_POS_STALE_FRAMES = 60
"""自身定位(OCR)连续失败的帧数上限（约 2 秒 @30fps）。

站定攻击中角色位置不变，短时定位失败（技能特效遮挡角色名字/
怪物名与角色名重叠/OCR 抖动）可用最后已知位置(_last_self_pos)
继续攻击，避免"特效挡名字 → 放弃目标转探索 → 角色离开 →
特效消失 → 重新定位"的反复空转；
超过该帧数仍定位不到 → 判定真丢失，避免用过期坐标乱跑。
"""

LOST_RECOVER_TRIGGER_FRAMES = 90
"""自身连续定位失败多少帧后，无条件转入迷失恢复（约 3 秒 @30fps）。

角色走到角落/名字被地图遮挡时，OCR 和截图都可能定位不到。若一直
卡在原地不动，角色会永远定位不到（走不出来）。连续定位失败满 3 秒
就无条件开始随机移动，让角色尽快从角落/遮挡中走出来重新被定位。
"""


@dataclass
class Context:
    """感知层 → 决策层的数据载体（每帧一份）。

    self_position 由 OCR 识别得到（窗口内坐标，脚底）。
    self_center 为角色中心点（名字中心 - 人物高度一半，向上），用于距离推算。
    """
    monsters: List[Detection] = field(default_factory=list)
    floors: List[Detection] = field(default_factory=list)
    ropes: List[Detection] = field(default_factory=list)
    self_position: Optional[Tuple[int, int]] = None
    self_center: Optional[Tuple[int, int]] = None
    hp_ratio: Optional[float] = None
    mp_ratio: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)


class DecisionEngine:
    """反应式决策引擎：根据画面实时内容决定下一步动作。

    【核心理念】
    不做地图、不做全局规划。每一帧只看 YOLO 检测结果，
    模拟人类玩家的反应模式。

    【状态机】
    使用 FSM 管理 7 个状态：
      IDLE → CHASING → ATTACKING  （同平台追击）
      IDLE → CLIMBING              （爬绳追怪）
      IDLE → DROPPING              （跳下追怪）
      任意 → HEALING / RECOVERING  （生存优先）

    Args:
        config:   全局配置
        executor: 动作执行器
        on_log:   日志回调
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self._log = on_log or (lambda m: None)
        self._skill_index = 0

        self._fsm = FSM(on_log=self._log)

        self._target_monster: Optional[Detection] = None
        self._explore_direction = "right"
        self._last_self_pos: Optional[Tuple[int, int]] = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0

        # 移动键按住状态（持续移动/攀爬）
        self._held_key: Optional[str] = None      # 当前按住的键（left/right/up/down）
        self._climbing = False                    # 是否正在沿绳索攀爬
        self._climb_exit_frames = 0               # 脱离绳索后横向走出的剩余帧数
        self._climb_log_count = 0                 # 攀爬日志限频计数
        self._attack_stale_counter = 0            # 锁定同一目标持续攻击的帧数（残影检测）
        self._face_dir: Optional[str] = None      # 记忆的角色朝向（left/right），攻击前据此调整
        self._melee_leave_frames = 0              # 短手攻击中连续超限帧数（防抖计数）
        self._occluded_frames = 0                 # 目标被角色遮挡的连续帧数（虚拟目标维持）
        self._self_pos_stale_frames = 0           # 自身定位连续失败的帧数（最后已知位置时效）

        # 迷失恢复（未定位时随机左右走 + 随机跳，按秒计时）
        self._lost_direction_start = 0.0           # 当前方向开始的时间戳（秒）
        self._lost_switch_seconds = random.uniform(  # 下一次换方向的随机秒数
            LOST_DIRECTION_SWITCH_MIN, LOST_DIRECTION_SWITCH_MAX
        )
        self._lost_last_jump_time = 0.0            # 上次跳跃的时间戳（秒）
        self._lost_jump_seconds = random.uniform(  # 下一次跳跃的随机间隔秒数
            LOST_JUMP_INTERVAL_MIN, LOST_JUMP_INTERVAL_MAX
        )

    def update_config(self, config: Config):
        self.config = config

    def reset(self):
        self._skill_index = 0
        self._target_monster = None
        self._explore_direction = "right"
        self._last_self_pos = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0
        self._attack_stale_counter = 0
        self._face_dir = None
        self._melee_leave_frames = 0
        self._lost_direction_start = 0.0
        self._lost_switch_seconds = random.uniform(
            LOST_DIRECTION_SWITCH_MIN, LOST_DIRECTION_SWITCH_MAX
        )
        self._lost_last_jump_time = 0.0
        self._lost_jump_seconds = random.uniform(
            LOST_JUMP_INTERVAL_MIN, LOST_JUMP_INTERVAL_MAX
        )
        self._occluded_frames = 0
        self._self_pos_stale_frames = 0
        self.release_keys()
        self._fsm.reset()
        self.executor.reset()

    def release_keys(self):
        """释放所有按住的移动键（停止时调用，防止方向键卡住）。"""
        self._release_move()
        self._climbing = False
        self._climb_exit_frames = 0
        self._climb_log_count = 0

    @property
    def state_name(self) -> str:
        """当前状态名（供 UI 显示）。"""
        return self._fsm.state_name

    # =========================================================================
    # 决策主入口
    # =========================================================================

    def decide(self, ctx: Context):
        """每帧调用一次，根据画面内容执行动作。

        Args:
            ctx: 当前帧的感知数据
        """
        self._fsm.tick()

        # 检测卡住 + 自身定位时效统计
        if ctx.self_position:
            self._self_pos_stale_frames = 0
            if self._last_self_pos and self._last_self_pos == ctx.self_position:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            self._last_self_pos = ctx.self_position
        else:
            # OCR 定位失败帧计数：站定攻击中用最后已知位置兜底的时效依据
            self._self_pos_stale_frames += 1

        # ---- 优先级 1: 没血加血 ----
        if ctx.hp_ratio is not None and ctx.hp_ratio < self.config.hp_threshold:
            # 满状态（>=95%）不触发，防止刚加完又按
            if ctx.hp_ratio < 0.95:
                self._fsm.transition(State.HEALING)
                self._release_move()  # 加血时站住不动
                if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                    self._log(
                        f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                        f"按下 {self.config.hp_key}"
                    )
                    return

        # ---- 优先级 2: 没蓝加蓝 ----
        if ctx.mp_ratio is not None and ctx.mp_ratio < self.config.mp_threshold:
            if ctx.mp_ratio < 0.95:
                self._fsm.transition(State.RECOVERING)
                self._release_move()  # 加蓝时站住不动
                if self.executor.press_key(self.config.mp_key, cooldown=1.5):
                    self._log(
                        f"[加蓝] MP={ctx.mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                        f"按下 {self.config.mp_key}"
                    )
                    return

        # ---- 优先级 3: 自身长时间未定位 → 迷失恢复 ----
        # 角色走到角落/名字被地图遮挡时，OCR 和截图都可能定位不到。
        # 此时无论画面里有没有怪物，只要连续定位失败超过阈值，就无条件
        # 转入迷失恢复（左右随机移动 + 随机跳跃），让角色从角落走出来、
        # 名字/身体重新露出被定位，而不是卡在原地不动。
        # 注意：必须放在怪物检测之前，否则"有怪但定位不到"时会走攻击
        # 分支，而攻击分支遇到 self_position=None 只释放移动不移动，
        # 导致角色卡在角落 30 秒不动。
        if ctx.self_position is None \
                and self._self_pos_stale_frames >= LOST_RECOVER_TRIGGER_FRAMES:
            self._fsm.transition(State.IDLE)
            self._target_monster = None
            self._attack_stale_counter = 0
            # 进入迷失恢复前随机抽初始方向 + 初始化计时，避免固定往右走
            self._explore_direction = random.choice(["left", "right"])
            self._lost_direction_start = time.time()
            self._lost_switch_seconds = random.uniform(
                LOST_DIRECTION_SWITCH_MIN, LOST_DIRECTION_SWITCH_MAX
            )
            self._lost_last_jump_time = time.time()
            self._lost_jump_seconds = random.uniform(
                LOST_JUMP_INTERVAL_MIN, LOST_JUMP_INTERVAL_MAX
            )
            self._explore(ctx)
            return

        # ---- 优先级 4: 检测到怪物 ----
        if ctx.monsters:
            self._handle_monsters(ctx)
        elif self._fsm.current == State.ATTACKING and self._target_monster is not None:
            # 攻击中整帧看不到任何怪：短手贴脸时角色+攻击特效可能把怪
            # 完全遮住 → YOLO 整帧漏检。先走遮挡判定维持虚拟目标继续攻击，
            # 避免"怪消失 → 转探索乱走 → 怪露出 → 重选目标"的反复空转。
            occluded = self._occluded_target(ctx)
            if occluded is not None:
                if getattr(self.config, "attack_type", "long") == "short":
                    self._handle_melee(ctx, occluded)
                else:
                    self._attack(ctx, occluded)
                return
            self._fsm.transition(State.IDLE)
            # 画面中已没有怪物：立即解除锁定并清理攀爬等残留状态，
            # 防止"上帧还锁着怪/在爬绳"的状态影响后续探索与重新选怪
            self._target_monster = None
            self._attack_stale_counter = 0
            self._climbing = False
            self._climb_exit_frames = 0
            self._explore(ctx)
        else:
            self._fsm.transition(State.IDLE)
            # 画面中已没有怪物：立即解除锁定并清理攀爬等残留状态，
            # 防止"上帧还锁着怪/在爬绳"的状态影响后续探索与重新选怪
            self._target_monster = None
            self._attack_stale_counter = 0
            self._climbing = False
            self._climb_exit_frames = 0
            self._explore(ctx)

    # =========================================================================
    # 怪物处理
    # =========================================================================

    def _handle_monsters(self, ctx: Context):
        """处理画面中的怪物（就近攻击）。

        【就近原则】每帧重新选择附近最近的怪物，谁在附近打谁：
          1. 怪物消失/离开画面后，下一帧自动选到别的怪，不用等
          2. 【残影防护】持续攻击同一目标超过 ATTACK_STALE_FRAMES 帧
             仍未击杀（画面仍检测到）→ 判定为死尸残影/无敌，
             跳过它重新选目标，避免原地打空气
        """
        target = self._resolve_locked_target(ctx)

        # ---- 无有效目标（same_platform_only 过滤后无同平台怪）----
        # 不走 CRASH 路径，释放移动键并转探索，避免角色卡在上一帧的移动状态
        if target is None:
            # 攻击中目标短暂从画面消失（YOLO 单帧漏检/波动）→ 沿用旧锁定
            # 目标继续攻击。若此时重选别的怪，dx 会瞬间翻转、角色左右乱转。
            # 若怪真被杀/消失，残影检测(ATTACK_STALE_FRAMES)会兜底换目标。
            if self._fsm.current == State.ATTACKING and self._target_monster is not None:
                target = self._target_monster
            else:
                self._target_monster = None
                self._attack_stale_counter = 0
                self._release_move()
                self._fsm.transition(State.IDLE)
                self._explore(ctx)
                return

        # ---- 攻击超时检测（残影防护）----
        # 持续攻击同一只怪（上帧目标 == 本帧最近目标）才累计；
        # 目标被角色遮挡期间（虚拟目标维持中）不计入，避免"攻击超时
        # 判定残影→中断"把遮挡中的正常战斗打断
        if self._occluded_frames == 0 and self._target_monster is not None \
                and self._is_same_monster(self._target_monster, target):
            if self._fsm.current == State.ATTACKING:
                self._attack_stale_counter += 1
            else:
                self._attack_stale_counter = 0
        else:
            self._attack_stale_counter = 0

        # 同一只怪攻击过久仍没死 → 很可能是尸体残影/无敌
        if self._attack_stale_counter >= ATTACK_STALE_FRAMES:
            self._log("[换目标] 持续攻击无效果(疑似残影/已死)，跳过该目标")
            self._attack_stale_counter = 0
            # 排除残影后重新选最近目标
            target = self._pick_best_target(ctx, exclude=self._target_monster)
            if target is None:
                # 画面里只剩打不死的残影 → 不空耗，转探索
                self._target_monster = None
                self._fsm.transition(State.IDLE)
                self._explore(ctx)
                return

        self._target_monster = target

        # ---- 短手（近战）走独立攻击流程，长手走远程流程 ----
        # 近战判定与远程完全不同：必须贴脸才打，未贴脸就径直追上，
        # 攻击中怪物走远立即转为追击，没有远程的"站定攻击+大滞回"。
        if getattr(self.config, "attack_type", "long") == "short":
            self._handle_melee(ctx, target)
            return

        mx, my = target.center

        # ---- 攻击判定（规则1/2/3/4）----
        # 同一平台 = 角色脚底y 与 怪物脚底y(bbox底部) 的垂直差 ≤ 垂直容差
        #   —— 脚底对"是否站在同一条地面线"最准确。
        #      若用"中心点y"对比，高大怪物会被误判为跨层(见 _pick_best_target)。
        # 攻击距离 = 水平方向 |角色x - 怪物中心x| ≤ 攻击距离px（规则2）
        foot = self._effective_self_pos(ctx)
        if foot is None:
            # OCR 定位失败（技能特效遮挡名字/OCR 抖动）且不在站定攻击中
            # → 无法判断距离，不能攻击，转入探索状态，避免盲打空放技能。
            # 站定攻击中定位失败由 _effective_self_pos 用最后已知位置兜底，
            # 不会走到这里。
            self._target_monster = None
            self._attack_stale_counter = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return
        px, py = foot
        monster_foot = (mx, target.y + target.h)

        # 路径推算：人物脚底 / 怪物 bbox 底部，与 YOLO 平台/绳索检测框
        # 的 y 语义一致，只用于"怎么走"。
        est = estimate_path_distance(
            foot, monster_foot, ctx.floors, ctx.ropes,
            same_level_tolerance=getattr(self.config, "attack_range_y", 60),
        )

        # 距离日志（限频输出，避免刷屏）
        self._distance_log_frame_count += 1
        if self._distance_log_frame_count >= DISTANCE_LOG_FRAMES:
            self._distance_log_frame_count = 0
            vy = abs(py - monster_foot[1])
            same_plat = self._same_platform(py, monster_foot[1])
            plan_str = ""
            if est.path_type == "jump" and est.path_floors:
                seq = "→".join(
                    f"({f.center[0]},{f.y})" for f in est.path_floors[:6]
                )
                plan_str = f" 路线平台: {seq}"
            elif est.path_type == "rope" and est.climb_rope is not None:
                r = est.climb_rope
                plan_str = f" 绳索: ({r.center[0]},{r.y}) 长{r.h}"
            self._log(
                f"[距离] 人物脚底({px},{py}) → 怪脚底({monster_foot[0]},{monster_foot[1]}) "
                f"同平台={'是' if same_plat else '否'}"
                f"(垂直差{vy} 容差{getattr(self.config, 'attack_range_y', 60)}) "
                f"路径={est.path_type} 距离={est.distance}px "
                f"(水平={abs(px - mx)})"
                f" [{('短手' if getattr(self.config, 'attack_type', 'long') == 'short' else '长手')}"
                f"有效攻击距={self._get_attack_range()}px]"
                + (f" 绳长={est.rope_length}" if est.path_type == "rope" else "")
                + (f" 跳数={est.jump_count}" if est.path_type == "jump" else "")
                + f"){plan_str}"
            )

        # 同一平台（脚底垂直差 ≤ 容差）：
        #   · 水平差 ≤ 攻击距离 → 站定攻击（不移动、不乱跑）
        #   · 否则 → 朝怪物直线移动逼近，进入攻击距离后再打
        if self._same_platform(py, monster_foot[1]):
            if self._can_attack(ctx, target):
                self._fsm.transition(State.ATTACKING)
                self._attack(ctx, target)
            else:
                self._fsm.transition(State.CHASING)
                self._chase(ctx, target)
            return

        # 不同平台：直接放弃该目标，只打同平台怪物。
        # 不爬绳、不跳跃、不兜底追击——目标在另一层时当前层打不到，
        # 跨层追过去成本高且容易卡地形。释放移动转探索，
        # 等画面里出现同平台怪物再攻击。
        self._target_monster = None
        self._attack_stale_counter = 0
        self._release_move()
        self._fsm.transition(State.IDLE)
        self._explore(ctx)

    # =========================================================================
    # 短手（近战）独立攻击流程
    # =========================================================================

    def _handle_melee(self, ctx: Context, target: Detection):
        """短手（近战）独立攻击判定。

        与长手（远程）完全不同，核心是【贴脸】:
          - 未贴脸（到怪近侧身体边缘距离 > 攻击距离）→ 不攻击，径直走向怪物贴身
          - 贴脸（边缘距离 ≤ 攻击距离 且同平台）→ 停止移动，转向 + 攻击
          - 攻击中怪物走远 → 下一帧判定未贴脸 → 立即转为追击，不停在原地空打
        没有远程那套"站定攻击 + 大滞回窗口"的逻辑。
        攻击距离固定 MELEE_ATTACK_RANGE_X(50px)，不随 exe 配置变化，
        距离判定用"角色到怪近侧身体边缘"，宽怪站旁边就打、不穿越。

        贴脸命中区 = 攻击距离 + MELEE_HIT_TOL_X（50+10=60px），超过
        即攻击特效够不到，立即追击不原地空打。攻击中(ATTACKING)超出
        命中区但 ≤ 攻击距离+MELEE_HYSTERESIS_X（60~90px）时用防抖帧数
        MELEE_LEAVE_FRAMES 吸收瞬时抖动（攻击位移/怪物被推开 1~3 帧），
        连续超限才转追击；超过 90px 说明怪真走远/已换目标，立即追击。
        """
        if target is None:
            self._target_monster = None
            self._attack_stale_counter = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return

        foot = self._effective_self_pos(ctx)
        if foot is None:
            # 无法定位自身 → 默认无法判断贴脸，原地等待下一帧定位，
            # 不切 IDLE 不转探索，避免 OCR 定位"时有时无"导致的高频抖动。
            # 例外：目标仍在最后已知位置的贴脸范围内（近战贴脸怪基本不动，
            # 角色站定没走远）→ 直接用最后位置继续攻击，避免
            # "怪已经连上(贴脸)但 OCR 恰好失败 → 角色干等不攻击"。
            if self._last_self_pos is not None \
                    and self._self_pos_stale_frames <= SELF_POS_STALE_FRAMES \
                    and abs(self._last_self_pos[0]
                            - self._melee_edge_x(self._last_self_pos[0], target)) \
                        <= self._get_attack_range():
                foot = self._last_self_pos
            else:
                self._release_move()
                return

        sx, sy = foot
        # 贴脸距离用"角色到怪近侧身体边缘"，不用怪中心：
        # 宽怪站怪旁边（离怪身体 0~攻击距离）就能攻击，不穿越怪身体。
        dx = abs(sx - self._melee_edge_x(sx, target))
        dy = abs(sy - (target.y + target.h))
        melee_range = self._get_attack_range()
        ry = getattr(self.config, "attack_range_y", 60)

        # 不同平台：放弃该目标（与长手一致，不跨层）
        if dy > ry:
            self._target_monster = None
            self._attack_stale_counter = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return

        # 贴脸命中判定：
        # 角色到怪近侧边缘 ≤ 攻击距离+命中容差 → 站定攻击（特效够得到）。
        # 攻击中超出命中区但 ≤ 攻击距离+滞回窗口 → 防抖帧数吸收瞬时
        # 抖动后仍继续攻击；明显超限(目标切换/怪真走远) → 立即追击，
        # 不停在原地对着够不着的怪空打。
        melee_hit = melee_range + MELEE_HIT_TOL_X
        if dx > melee_hit:
            if self._fsm.current == State.ATTACKING \
                    and dx <= melee_range + MELEE_HYSTERESIS_X:
                # 攻击中超出命中区但还在防抖窗口内：先用防抖帧数吸收
                # 瞬时抖动（攻击位移/怪物被推开 1~3 帧）。期间【原地继续
                # 攻击、绝不移动】——攻击动画期间按方向键会边打边跑、
                # 追过头穿越怪物，表现为"贴着怪左右跑"。连续超出
                # MELEE_LEAVE_FRAMES 帧才判定怪物真走远，转追击。
                self._melee_leave_frames += 1
                if self._melee_leave_frames < MELEE_LEAVE_FRAMES:
                    self._melee_attack(ctx, target)
                    return
            self._melee_leave_frames = 0
            self._fsm.transition(State.CHASING)
            self._melee_chase(ctx, target)
            return

        # 已贴脸命中 → 转向 + 攻击（回到命中区即清零防抖计数）
        self._melee_leave_frames = 0
        self._fsm.transition(State.ATTACKING)
        self._melee_attack(ctx, target)

    def _melee_chase(self, ctx: Context, target: Detection):
        """近战追击：径直走向怪物，进入攻击距离即停手攻击。

        与长手语义一致：角色到怪近侧身体边缘 ≤ 攻击距离(50px) 就开始
        攻击，不需要走到怪物脸上（不追到 攻击距离-5px）。
        """
        if ctx.self_position is None:
            self._release_move()
            return
        sx = ctx.self_position[0]
        # 追击目标点 = 怪近侧身体边缘（走到离怪身体 攻击距离-5px 处停下）
        tx = self._melee_edge_x(sx, target)
        melee_range = self._get_attack_range()

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[近战] 卡住了，尝试跳跃")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 进入攻击距离即停手攻击，不走到怪物脸上。
        # 停点 = 攻击距离(50px)；命中容差(+10px)保证停住后攻击能命中，
        # 不会在边界"差一步打不到 → 蹭一下 → 又超距"来回蹭。
        stop_at = max(5, melee_range)
        if tx > sx + stop_at:
            self._hold_move("right")
        elif tx < sx - stop_at:
            self._hold_move("left")
        else:
            self._release_move()

    def _melee_attack(self, ctx: Context, target: Detection):
        """近战攻击：先判定怪在哪边扭头面向它，再站定释放技能。

        核心：每次攻击前判定怪物相对角色的方位（看怪中心 x 的符号），
        只要怪在角色背后就按方向键转身——先扭头再攻击，避免朝反方向
        空打。转身会让角色朝怪移动一小步，因此两个例外【不转向】:
          - 角色已与怪身体重叠/极近（edge_dx ≤ MELEE_FACE_DEADZONE_X）：
            转身会穿过怪身体左右来回顶，保持原朝向直接攻击（怪就在
            身前/身侧，攻击可命中）。
          - 防抖窗口内(_melee_leave_frames>0)/遮挡虚拟目标期间
            (_occluded_frames>0)：怪刚走远正在确认、或位置是最后的，
            转向=边打边追穿越怪，表现为"贴着怪左右晃动"。
        """
        self._release_move()  # 攻击时站定
        self._stuck_counter = 0  # 站定攻击不算"卡住"（位置不变是正常的）

        # 先判定怪在哪边（怪中心 x 相对角色 x 的符号），背对怪就扭头。
        # 方向看怪中心（符号决定面朝哪侧），距离看怪近侧身体边缘
        # （决定能否安全转身）。转向带 0.6s 冷却，不会高频反复转。
        if target is not None and self._melee_leave_frames == 0 \
                and self._occluded_frames == 0:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                self._cast_skill()
                return
            sx = foot[0]
            center_dx = target.center[0] - sx
            edge_dx = abs(sx - self._melee_edge_x(sx, target))
            need = None
            if center_dx > FACE_TURN_X and edge_dx > MELEE_FACE_DEADZONE_X:
                need = "right"
            elif center_dx < -FACE_TURN_X and edge_dx > MELEE_FACE_DEADZONE_X:
                need = "left"
            if need is not None and need != self._face_dir:
                # need != _face_dir 即"角色背对怪"→ 扭头。edge_dx ≤ 死区
                # 时 need 已为 None（重叠极近不转），不会出现转身穿怪。
                if self.executor.press_key(need, cooldown=0.6):
                    self._face_dir = need
                    self._log(
                        f"[朝向] 怪物在{'右' if need == 'right' else '左'}"
                        f"(中心{center_dx:+d}px/边缘{edge_dx}px)，按{need}转向"
                    )

        self._cast_skill()

    # =========================================================================
    # 按住移动
    # =========================================================================

    def _hold_move(self, direction: str):
        """按住方向键持续移动。

        切换方向时先释放旧键再按住新键，避免两个方向键同时按下。
        移动期间不松手，直到调用 _release_move() 停止。
        """
        if direction not in ("left", "right", "up", "down"):
            return
        if direction in ("left", "right"):
            self._face_dir = direction  # 移动方向即角色朝向
        if self._held_key == direction:
            return
        if self._held_key:
            self.executor.key_up(self._held_key)
        self.executor.key_down(direction)
        self._held_key = direction

    def _release_move(self):
        """释放当前按住的移动键（停止移动/攀爬）。"""
        if self._held_key:
            self.executor.key_up(self._held_key)
            self._held_key = None

    # =========================================================================
    # 目标选择
    # =========================================================================

    def _resolve_locked_target(self, ctx: Context) -> Detection:
        """解析当前应攻击的目标怪物（就近原则 + 攻击期保持锁定）。

        优先沿用已锁定目标：只要同一只怪仍在画面中就继续打它，
        避免 YOLO 帧间检测波动导致目标来回切换、攻击刚触发就中断；
        已锁定目标消失/被击杀/残影超时/离开同平台后，才重新选目标。
        首次选目标时按"同平台最近"原则（_pick_best_target）。

        Returns:
            当前应攻击的怪物；若画面中没有同平台怪，返回 None。
        """
        if self._target_monster is not None:
            for m in ctx.monsters:
                if self._is_same_monster(self._target_monster, m):
                    # 锁定目标仍在画面，但已不在角色同一平台
                    # （跨层/掉下平台）→ 放弃锁定，只打同层怪
                    if ctx.self_position is not None and not self._same_platform(
                            ctx.self_position[1], m.y + m.h):
                        break
                    # 目标重新可见（遮挡解除）→ 清除遮挡计数
                    self._occluded_frames = 0
                    return m
            # 锁定目标本帧不在画面：可能是被角色自身遮挡（贴脸攻击）
            occluded = self._occluded_target(ctx)
            if occluded is not None:
                return occluded
            self._occluded_frames = 0
            self._target_monster = None
        return self._pick_best_target(ctx)

    def _is_same_monster(self, a: Detection, b: Detection) -> bool:
        """按中心点距离判断两个检测是否可能是同一只怪。

        容差与锁定匹配一致（目标 bbox 宽度的 1.5 倍，保底 40px），
        用于攻击超时统计。
        """
        if a is None or b is None:
            return False
        tolerance = max(int(a.w * 1.5), 40)
        d = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
        return d <= tolerance

    def _occluded_target(self, ctx: Context) -> Optional[Detection]:
        """目标从画面消失时，判断是否被角色遮挡并返回可继续攻击的虚拟目标。

        短手贴脸攻击时角色站在怪正前方，角色模型+攻击特效会遮住怪物，
        YOLO 置信度骤降被 conf 阈值过滤 → 本帧"看不到"锁定目标。
        此时怪其实还在原处（贴脸近战怪基本不动），用最后已知位置继续打，
        避免目标消失引发：
          - 重选别的怪 → dx 符号翻转 → 左右乱转
          - 攻击中断 → 转探索 → 角色离开 → 怪重新可见 → 反复贴脸失败

        判定条件（全部满足才视为"被遮挡"，否则按怪真消失处理）：
          1. 短手模式且正处于攻击状态（角色站定，位置稳定）
          2. 角色与目标最后位置在同一平台
          3. 角色与目标最后位置水平距离很近（贴脸距离 + 宽度容差）
          4. 连续遮挡未超上限（OCCLUSION_MAX_FRAMES，超时视为怪真消失）
        """
        if getattr(self.config, "attack_type", "long") != "short":
            return None  # 长手站远程打，不会贴脸遮挡，保持原逻辑
        if self._target_monster is None:
            return None
        if self._fsm.current != State.ATTACKING:
            return None
        # 用"有效位置"而非实时位置：贴脸时角色名字也可能被怪/攻击特效遮挡，
        # OCR 定位失败 → ctx.self_position 为 None。攻击中角色位置不变，
        # 用最后已知位置(_last_self_pos)兜底判遮挡，避免"怪被遮 + 名字被遮"
        # 同时发生时遮挡判定失败 → 清空锁定 → 转探索乱走的死循环。
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return None
        sx, sy = foot
        t = self._target_monster
        if not self._same_platform(sy, t.y + t.h):
            return None  # 已跨层 → 非遮挡
        # 遮挡判定同样用"到怪近侧边缘"的距离：贴脸打怪时角色就在怪
        # 身体旁边（边缘距离≈0），中心距离可能超过攻击距离的怪（宽怪）
        # 也能正确判定"角色在怪跟前"，不会误判非遮挡而清空锁定。
        if abs(sx - self._melee_edge_x(sx, t)) \
                > self._get_attack_range() + OCCLUSION_HALF_WIDTH_X:
            return None  # 角色不在目标跟前 → 非遮挡
        self._occluded_frames += 1
        if self._occluded_frames > OCCLUSION_MAX_FRAMES:
            self._log("[目标] 目标被遮挡超时，判定已消失，放弃攻击")
            return None
        if self._occluded_frames == 1:
            self._log("[目标] 目标被角色遮挡，沿用最后位置继续攻击")
        return t

    def _pick_best_target(self, ctx: Context,
                          exclude: Optional[Detection] = None) -> Detection:
        """选择离角色最近的【同平台】怪物。

        只打同平台的怪：角色脚底（名字中心）与怪物脚底（bbox 底部）
        的垂直差 ≤ 垂直容差（attack_range_y，exe 界面"垂直容差px"）
        才视为可攻击目标，距离只算水平方向：dx = |角色x - 怪物中心x|。
        跨层怪物直接忽略（不爬绳/不跳跃/不追击），画面里没有同平台怪
        时返回 None，转探索等自己走到那一层再打。

        说明：同平台必须用"脚底 vs 脚底"。若用"角色中心y vs 怪物bbox中心y"，
        高大怪物（如 150px 高）的中心点比角色中心高出 40~50px，
        超过 30px 容差 → 同平台的怪被误判为跨层 → 一直爬绳/跳跃/乱跑不攻击。
        无法定位自身时回退为画面中最大的怪物。

        Args:
            ctx:     当前帧感知数据
            exclude: 需要跳过的怪物（如疑似残影），可选
        """
        monsters = ctx.monsters
        if not monsters:
            return None
        if exclude is not None:
            monsters = [m for m in monsters
                        if not self._is_same_monster(exclude, m)]
            if not monsters:
                return None
        player = ctx.self_position
        if player is None:
            return max(monsters, key=lambda d: d.w * d.h)

        # 【只打同平台】仅考虑与角色在同一平台的怪物：
        # 脚底垂直差 ≤ 垂直容差（attack_range_y）。
        # 跨层怪物直接忽略——不爬绳不跳跃，等角色自己走到那层再打。
        best = None
        best_dist = float("inf")
        for m in monsters:
            mfoot = m.y + m.h  # 怪物脚底（bbox 底部）
            if not self._same_platform(player[1], mfoot):
                continue
            dx = abs(player[0] - m.center[0])
            if dx < best_dist:
                best = m
                best_dist = dx
        return best

    # =========================================================================
    # 地板判定
    # =========================================================================

    def _has_floor_under(self, ctx: Context, pos: Tuple[int, int]) -> bool:
        """检查指定位置下方是否有地板。

        判断: 地板检测框的 Y 范围是否覆盖了该位置的 Y 坐标附近。
        """
        px, py = pos
        for f in ctx.floors:
            if f.x <= px <= f.x + f.w:
                if f.y - 10 <= py <= f.y + f.h + 10:
                    return True
        return True  # 没检测到地板时默认认为可以站（宽容处理）

    # =========================================================================
    # 攻击范围判定
    # =========================================================================

    def _effective_self_pos(self, ctx: Context) -> Optional[Tuple[int, int]]:
        """返回当前帧决策用的自身脚底坐标。

        OCR 定位成功 → 实时坐标。
        定位暂时失败（技能特效遮挡角色名字 / 怪物名与角色名重叠 /
        OCR 抖动）但正处于【站定攻击】中 → 用最后已知位置兜底
        （_last_self_pos，带 SELF_POS_STALE_FRAMES 时效）。
        站定攻击中角色位置不变，短时用旧坐标准确且安全，避免
        "特效挡名字 → 放弃目标 → 转探索乱走" 的反复空转。
        其余状态（追击/探索/移动中）定位失败 → 返回 None，
        由各调用方按原有逻辑处理（不盲打/不追错方向）。
        """
        if ctx.self_position is not None:
            return ctx.self_position
        if self._fsm.current == State.ATTACKING and self._last_self_pos is not None \
                and self._self_pos_stale_frames <= SELF_POS_STALE_FRAMES:
            return self._last_self_pos
        return None

    def _get_attack_range(self) -> int:
        """返回当前生效的攻击距离（像素）。

        - 短手(近战): 固定 MELEE_ATTACK_RANGE_X(50)，【不允许配置修改】，
          与长手完全独立，不读 exe 的 attack_range。
        - 长手(远程): 读 config.attack_range（exe 界面"攻击距离px"可改）。
        切换的是攻击判定代码：
        - 长手: 远程站定攻击（_can_attack/_chase/_attack）
        - 短手: 近战贴脸攻击（_handle_melee/_melee_chase/_melee_attack）
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return MELEE_ATTACK_RANGE_X
        return getattr(self.config, "attack_range", ATTACK_RANGE_X)

    def _melee_edge_x(self, sx: int, target: Detection) -> int:
        """短手贴脸判定用的"怪近侧身体边缘 x"。

        近战攻击特效命中怪身体任意部位即可，角色不用走到怪中心。
        贴脸/追击/转向的距离判定统一用"角色到怪近侧边缘"：
        宽怪站在旁边就能打，避免为贴中心而穿进怪身体、穿到另一侧
        又超距折返的"左右来回穿"抖动。
        怪半宽取 min(bbox 半宽, MELEE_EDGE_MAX_HALF_W) 封顶，
        防止超宽怪（boss）离老远就判定贴脸。
        注：此函数只用于"距离"判定；"怪在哪边/面朝哪侧"仍看怪中心。
        """
        half_w = min(target.w / 2.0, MELEE_EDGE_MAX_HALF_W)
        if target.center[0] > sx:
            return target.center[0] - half_w
        return target.center[0] + half_w

    def _in_attack_range(self, sx: int, tx: int) -> bool:
        """判断是否在攻击距离内（规则2：水平距离 ≤ 当前生效攻击距离）。

        长手/短手共用 attack_range，此处仅做水平距离判定。
        """
        return abs(sx - tx) <= self._get_attack_range()

    def _same_platform(self, y1: int, y2: int) -> bool:
        """判断两个纵坐标是否在同一平台（规则1：垂直容差内）。

        入参为"角色脚底y"与"怪物脚底y(bbox底部)"，垂直差 ≤ 配置的
        attack_range_y（exe 界面"垂直容差px"）即视为同一平台；
        攻击必须满足此条件才允许发起。
        """
        return abs(y1 - y2) <= getattr(self.config, "attack_range_y", 60)

    def _can_attack(self, ctx: Context, target: Detection) -> bool:
        """综合攻击判定（规则1/2 + 滞回防抖）。

        仅长手(远程)模式走这里；短手(近战)走独立的 _handle_melee。
        - 规则1: 同一平台 = 角色脚底y 与 怪物脚底y(bbox底部) 垂直差 ≤ 容差
        - 规则2: 攻击距离 = 角色x 与 怪物中心x 水平差 ≤ 当前生效攻击距离
          (长手/短手共用 attack_range)
        攻击中（FSM 处于 ATTACKING）且轻微超限时仍允许攻击，
        避免 OCR/YOLO 帧间几像素抖动导致"攻击刚触发就中断、来回跑"。
        目标已明显离开（超过滞回窗口）才判定不可攻击。
        """
        foot = self._effective_self_pos(ctx)
        if foot is None or target is None:
            return False
        sx, sy = foot
        mx = target.center[0]
        mfoot = target.y + target.h
        dx = abs(sx - mx)
        dy = abs(sy - mfoot)
        ry = getattr(self.config, "attack_range_y", 60)
        attack_range = self._get_attack_range()

        # 硬阈值：同平台 + 在攻击距离内
        if dx <= attack_range and dy <= ry:
            return True
        # 滞回：攻击中轻微超出（怪物中心/bbox 波动、OCR 抖动、角色攻击位移）不中断
        # 滞回窗口: 长手 +60px (近战不走这里)
        if self._fsm.current == State.ATTACKING:
            if dx <= attack_range + 60 and dy <= ry + 40:
                return True
        return False

    # =========================================================================
    # 追击（同平台）
    # =========================================================================

    def _chase(self, ctx: Context, target: Detection):
        """按住方向键持续走向怪物（同一平台内直线接近）。

        仅长手(远程)模式走这里（短手走独立的 _melee_chase 贴脸追击）。
        长手模式: 到达 attack_range 前不松手，进入攻击范围后释放方向键。

        贴近目标后保持朝向（不翻转），交给攻击判定，避免原地乱跑抖动。
        """
        if ctx.self_position is None:
            self._release_move()
            return
        sx = ctx.self_position[0]
        tx = target.center[0]
        attack_range = self._get_attack_range()
        is_melee = getattr(self.config, "attack_type", "long") == "short"

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[追击] 卡住了，尝试跳跃")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # ---- 方向滞回：x 差超过死区才切换方向 ----
        # 短手模式死区更小（近战需要更精确对位），长手模式保持原有逻辑
        if is_melee:
            dead_zone = min(15, max(8, attack_range // 3))
        else:
            dead_zone = min(30, max(15, self.config.attack_range // 4))

        if tx > sx + dead_zone:
            self._hold_move("right")
        elif tx < sx - dead_zone:
            self._hold_move("left")
        else:
            # 已贴近目标：若仍在攻击范围外，则保持原方向小步逼近；
            # 已进入攻击范围则停止，交给攻击判定。
            # 追击停止阈值: 长手在攻击范围+10px 处停下，短手在攻击范围+5px 处停下
            stop_margin = 5 if is_melee else 10
            if abs(sx - tx) <= attack_range + stop_margin:
                self._release_move()
            else:
                # 保持当前朝向逼近（不翻转），避免边缘抖动左右跑
                self._hold_move(self._face_dir or "right")

    # =========================================================================
    # 攻击
    # =========================================================================

    def _attack(self, ctx: Context, target: Detection):
        """在攻击范围内原地释放技能（攻击时不移动）。

        长手模式: 在攻击距离外原地释放远程技能。
        短手模式: 在极近距离释放近战技能，角色贴脸攻击。

        攻击前【每次】都判断怪物在角色左边还是右边，然后执行对应方向键
        转向（怪物在右 → 按右键，怪物在左 → 按左键），确保角色面向怪物
        后技能才能打中。短按（约30ms）只转向不位移；冷却 0.2s 防止攻击
        状态每帧狂按方向键抖动。

        【距离守卫】攻击前统一校验：必须满足两个条件才发动攻击：
        1. 同一平台：垂直差 ≤ attack_range_y
        2. 水平差 < 当前生效攻击距离
        否则直接放弃攻击（不释放技能），避免怪物不在附近时一直空打。
        """
        self._release_move()  # 攻击时保持不动

        # ---- 距离守卫（规则1/2）：脚底判同平台 + 中心x判攻击距离 ----
        # 不满足 → 直接放弃攻击，避免怪物不在附近时一直空打/乱跑。
        # 攻击中带滞回（_can_attack）：轻微抖动/怪物中心波动不中断攻击，
        # 防止"打一下就跑"的横跳。
        if target is not None:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                # OCR 定位失败且不在站定攻击中 → 无法判断距离，放弃攻击
                self._log("[攻击] 无法定位自身位置，放弃攻击")
                return
            if not self._can_attack(ctx, target):
                sx, sy = foot
                mx = target.center[0]
                self._log(
                    f"[攻击] 不同平台或怪物不在攻击范围内"
                    f"(水平={abs(sx - mx)} 垂直={abs(sy - (target.y + target.h))}"
                    f" 容差={getattr(self.config, 'attack_range_y', 60)})，"
                    f"停止攻击"
                )
                return

            # ---- 转向：怪物在右 → 按右键；怪物在左 → 按左键；正下方 → 不按 ----
            sx = foot[0]
            dx = target.center[0] - sx
            need = None
            if dx > FACE_TURN_X:
                need = "right"
            elif dx < -FACE_TURN_X:
                need = "left"
            if need is not None and need != self._face_dir:
                # 朝向不对 → 先转向，然后继续释放技能。
                # 冒险岛转向和攻击可在同一帧完成，不需要等下一帧。
                if self.executor.press_key(need, cooldown=0.3):
                    self._face_dir = need
                    self._log(
                        f"[朝向] 怪物在{'右' if need == 'right' else '左'}"
                        f"({dx:+d}px)，按{need}转向"
                    )

        self._cast_skill()

    def _tab_attack(self, ctx: Context, target: Detection = None):
        """Tab 选怪 + 原地攻击（兜底方案，不移动）。

        与 _attack 相同的距离守卫：能拿到自身位置和目标时，
        水平差/垂直差超限就停止攻击，防止怪物不在附近空打。
        无法定位自身时直接放弃攻击，不盲打。
        """
        self._release_move()

        # ---- 距离守卫（规则1/2）：脚底判同平台 + 中心x判攻击距离 ----
        # 无法定位自身 → 放弃攻击（不盲打）
        if target is not None:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                self._log("[攻击] 无法定位自身位置，放弃攻击")
                return
            if not self._can_attack(ctx, target):
                sx, sy = foot
                mx = target.center[0]
                self._log(
                    f"[攻击] 不同平台或怪物不在攻击范围内"
                    f"(水平={abs(sx - mx)} 垂直={abs(sy - (target.y + target.h))}"
                    f" 容差={getattr(self.config, 'attack_range_y', 60)})，"
                    f"停止攻击"
                )
                return

        self.executor.press_key(self.config.target_key, cooldown=0.8)
        self._cast_skill()

    # =========================================================================
    # 攀爬
    # =========================================================================

    def _rope_reachable(self, rope: Detection, foot_y: int) -> bool:
        """判断人物脚底能否够到绳底端（可跳抓）。

        绳底端 y（rope.y + rope.h）最多高出人物脚底 ROPE_REACH_Y。
        若绳底远高于人物脚底（如挂在半空/画面顶部的高绳），
        人物跳抓不到，爬不了这条绳。
        """
        rope_bottom = rope.y + rope.h
        return (foot_y - rope_bottom) <= ROPE_REACH_Y

    def _try_climb(self, ctx: Context, target: Detection,
                   planned_rope: Optional[Detection] = None) -> bool:
        """攀爬追怪：走到绳索正下方 → 跳跃 + 按住上键爬绳。

        流程:
          1. 脱离绳索后的横向走出阶段（不重新抓绳）
          2. 用路径规划选定的绳索（est.climb_rope，基于当前截图
             YOLO 分析结果）；规划绳不在附近时回退找最近绳索
          3. 【对准判定】人物中心与绳索中心是否在同一竖直轴线
             （X 差 <= 5px）→ 否，先水平移动到绳索正下方
          4. 已对准 → 按跳跃 + 按住上键沿绳索向上爬
          5. 爬到怪物所在高度（人物中心与怪物中心 Y 差 <= 30px）
             或爬到绳顶 → 停止爬绳，横向走出绳索继续追击

        返回 True 表示找到了绳索并执行了动作，False 表示没找到。
        """
        # 用人物中心点（不是脚底），回退脚底
        player = ctx.self_center or ctx.self_position
        if player is None:
            self._release_move()
            return False
        sx, sy = player
        tx = target.center[0]

        # 刚爬完绳，横向走出绳索（此阶段不重新抓绳）
        if self._climb_exit_frames > 0:
            self._climb_exit_frames -= 1
            if tx > sx:
                self._hold_move("right")
            else:
                self._hold_move("left")
            return True

        # 找要爬的绳索：优先用路径规划选定的绳（YOLO 分析出的路线）。
        # 只要水平距离在搜索范围内就视为附近，不再用"绳底够不着"
        # 过滤——YOLO 绳框常只覆盖绳子上段，绳底远高于人物脚底是
        # 检测框不完整造成的，实际地图绳索通常垂到地面，都能爬上。
        foot_y = ctx.self_position[1] if ctx.self_position is not None else sy
        nearest_rope = None
        min_dist = float("inf")
        if planned_rope is not None:
            rxd = abs(planned_rope.center[0] - sx)
            if rxd < ROPE_SEARCH_RANGE_X:
                nearest_rope = planned_rope
                min_dist = rxd
        if nearest_rope is None:
            for r in ctx.ropes:
                rx = r.center[0]
                dist = abs(rx - sx)
                if dist < ROPE_SEARCH_RANGE_X and dist < min_dist:
                    nearest_rope = r
                    min_dist = dist

        if nearest_rope is None:
            self._climbing = False
            self._release_move()
            return False

        rx, ry = nearest_rope.center

        # ---- 阶段 1: 对准判定（人物中心与绳索中心同一竖直轴线）----
        if not self._climbing:
            if abs(rx - sx) > CLIMB_ALIGN_TOLERANCE:
                # 不在绳索正下方 → 水平移动对准
                if rx > sx:
                    self._hold_move("right")
                else:
                    self._hold_move("left")
                return True
            # 人物中心与绳索中心在同一竖直轴线（±5px）→ 跳跃 + 按住上键爬绳
            self._release_move()
            self._climbing = True
            self._log("[攀爬] 对准绳索，跳跃并开始攀爬")
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._hold_move("up")
            return True

        # ---- 阶段 2: 正在爬绳 ----
        ty = target.center[1]

        # 到达条件: 人物中心与怪物中心 Y 差 <= 30px（已爬到怪物所在层）
        if abs(sy - ty) <= SAME_LEVEL_Y_TOLERANCE:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到达怪物所在高度，脱离绳索")
            return True

        # 爬到绳顶（人物中心已接近绳索顶部）仍没到怪物高度 → 停止，横向走出
        if sy <= ry - (nearest_rope.h / 2) + 10:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到绳顶仍追不上，脱离绳索")
            return True

        # 还没到 → 继续按住上键向上爬（日志限频，防刷屏）
        self._hold_move("up")
        self._climb_log_count += 1
        if self._climb_log_count % 15 == 1:
            self._log("[攀爬] 沿绳索向上")
        return True

    # =========================================================================
    # 跨层跳跃追击（无绳索时）
    # =========================================================================

    def _jump_chase(self, ctx: Context, target: Detection,
                    est: Optional[PathEstimate] = None):
        """无绳索时，按平台路径逐层跳跃追击怪物（动态路线规划）。

        核心思路（与 YOLO 检测到的平台/绳索动态规划路线）：
          1. 优先：怪物所在平台高度差 <= 跳跃高度 → 直接跳上
          2. 否则：找到"往怪物方向、人物能跳上去"的下一层平台，
             先横向走到其正下方/附近，再起跳，一层一层往上/靠近
          3. 若 BFS 已给出规划路径(est.path_floors)，优先用它确定
             中间过渡平台；否则动态在 ctx.floors 里找可跳平台
          4. 脚下没地板（边缘/空中）→ 下落

        平台高度按"最上面的 y 坐标"（floor.y，top_y）计算。
        """
        if ctx.self_position is None:
            self._release_move()
            return
        sx, sy = ctx.self_position
        tx = target.center[0]
        # 怪物站立高度用 bbox 底部（脚底），比框中心更接近其所在平台
        ty = target.y + target.h

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[跳跃追击] 卡住了，起跳")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 怪物所在平台的 top_y
        target_top = self._find_floor_top(ctx, tx, ty)
        target_top = target_top if target_top is not None else ty

        # 人物脚下的平台 top_y
        foot_top = self._find_floor_top(ctx, sx, sy)
        on_floor = self._has_floor_under(ctx, ctx.self_position)

        # ---- 情况1: 可以直接跳上目标平台 ----
        if on_floor and target_top is not None and ty < sy - 5:
            # 怪物在上层，高度差在跳跃高度内 → 横向走到其下方后起跳
            if (sy - target_top) <= JUMP_HEIGHT:
                # 对齐用"怪物所在平台"（比怪物 bbox 更可靠），找不到则用怪物
                goal_floor = self._find_floor_object(ctx, (tx, ty))
                align_ref = goal_floor if goal_floor is not None else target
                need_x = self._align_x_for_jump(sx, tx, align_ref, ctx)
                if need_x:
                    return  # 还在水平对准中
                self._log(
                    f"[跳跃追击] 目标平台 top_y={target_top}，"
                    f"高度差 {sy - target_top}px <= {JUMP_HEIGHT}px，起跳"
                )
                self._release_move()
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
                return

        # ---- 情况2: 高度差太大，需逐层跳（动态规划中间平台）----
        if on_floor and target_top is not None and ty < sy - 5:
            next_floor = self._pick_next_floor(ctx, target, est)
            if next_floor is not None:
                nx, ny = next_floor.center
                ntop = next_floor.y
                # 需要走到该平台上方/附近才能跳上去
                align = self._align_x_for_jump(sx, nx, next_floor, ctx,
                                               target_ty=ntop)
                if align:
                    return  # 还在水平对准中间平台
                # 高度差在跳跃高度内 → 起跳
                if (sy - ntop) <= JUMP_HEIGHT:
                    self._log(
                        f"[跳跃追击] 逐层跳: 目标平台top={ntop}，"
                        f"高度差 {sy - ntop}px <= {JUMP_HEIGHT}px，起跳"
                    )
                    self._release_move()
                    self.executor.press_key(self.config.jump_key, cooldown=1.0)
                    return
                # 中间平台也不够 → 继续往怪物方向走（保持按住方向键）
                self._hold_toward(sx, tx)
                return

            # 找不到中间平台 → 继续往怪物方向走
            self._hold_toward(sx, tx)
            return

        # ---- 情况3: 脚下没地板（边缘/空中）→ 下落 ----
        if not on_floor:
            self._log("[跳跃追击] 脚下没地板，下落")
            self.executor.press_key("down", cooldown=0.3)
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            return

        # 其他情况：往怪物方向走
        self._hold_toward(sx, tx)

    def _hold_toward(self, sx: int, tx: int):
        """按住方向键朝目标 x 移动（移动时按住方向键）。"""
        if tx > sx + 10:
            self._hold_move("right")
        elif tx < sx - 10:
            self._hold_move("left")
        else:
            self._release_move()

    def _align_x_for_jump(self, sx: int, tx: int, floor: Detection,
                          ctx: Context, target_ty: Optional[int] = None) -> bool:
        """把人物横向移动到目标平台覆盖范围内，准备起跳。

        Returns:
            True 表示还在水平对准（需要继续移动，本次不应跳）；
            False 表示已经对齐（可以起跳）。
        """
        # 目标平台的横向覆盖范围（向内收缩 10px，避免站边缘起跳）
        f_left = floor.x + 10
        f_right = floor.x + floor.w - 10

        # 已站在平台覆盖范围内
        if f_left <= sx <= f_right:
            self._release_move()
            return False

        # 平台在右边 → 向右走；平台在左边 → 向左走
        if f_right < sx:
            self._hold_move("left")
        else:
            self._hold_move("right")
        return True

    def _pick_next_floor(self, ctx: Context, target: Detection,
                         est: Optional[PathEstimate] = None) -> Optional[Detection]:
        """规划"下一步跳往的中间平台"。

        优先用 BFS 规划路径(path_floors)中的人物脚下平台的下一个平台；
        否则动态在 ctx.floors 中找：位于人物上方、怪物方向侧、
        高度差 <= JUMP_HEIGHT、能被人物跳到的最远/最近平台。
        """
        # 优先用 BFS 规划路径
        if est is not None and est.path_floors:
            foot = self._find_floor_object(ctx, ctx.self_position)
            if foot is not None:
                for f in est.path_floors:
                    if self._is_same_floor(foot, f):
                        # 找到当前平台在序列中的位置，返回下一层
                        idx = est.path_floors.index(f)
                        if idx + 1 < len(est.path_floors):
                            nxt = est.path_floors[idx + 1]
                            # 中间平台必须高于当前，且高度差可跳
                            if nxt.y < foot.y - 5 and (foot.y - nxt.y) <= JUMP_HEIGHT:
                                return nxt
                            continue
            # BFS 路径不适用（人物已偏离起始平台）→ 回退动态搜索

        if ctx.self_position is None:
            return None
        sx, sy = ctx.self_position
        tx = target.center[0]

        # 动态搜索：人物上方、高度差可跳、在怪物方向一侧的平台
        candidates = []
        for f in ctx.floors:
            # 必须在人物上方（更高的平台，y 更小）
            if f.y >= sy - 5:
                continue
            if (sy - f.y) > JUMP_HEIGHT:
                continue
            # 横向位置应靠近人物或位于怪物方向
            if abs(f.center[0] - sx) > PLATFORM_JUMP_GAP_X:
                continue
            candidates.append(f)

        if not candidates:
            return None
        # 选水平距离最近（先到达）的平台
        candidates.sort(key=lambda f: abs(f.center[0] - sx))
        return candidates[0]

    def _is_same_floor(self, a: Detection, b: Detection) -> bool:
        """判断两个平台检测是否为同一平台（按位置重叠）。"""
        if a is None or b is None:
            return False
        ax0, ay0 = a.x, a.y
        ax1, ay1 = a.x + a.w, a.y + a.h
        bx0, by0 = b.x, b.y
        bx1, by1 = b.x + b.w, b.y + b.h
        ox = min(ax1, bx1) - max(ax0, bx0)
        oy = min(ay1, by1) - max(ay0, by0)
        return ox > 5 and oy > 5

    def _find_floor_object(self, ctx: Context, pos) -> Optional[Detection]:
        """找到覆盖人物脚底 (x, y) 的平台对象，找不到返回 None。"""
        if pos is None:
            return None
        x, y = pos
        best = None
        best_key = float("inf")
        for f in ctx.floors:
            if f.x <= x <= f.x + f.w:
                if f.y - 30 <= y <= f.y + f.h + 30:
                    d = abs(f.y - y)
                    if d < best_key:
                        best_key = d
                        best = f
        return best

    def _find_floor_top(self, ctx: Context, x: int, y: int) -> Optional[int]:
        """找到覆盖 (x, y) 的平台，返回其"最上面的 y"（top_y）。

        找不到返回 None。
        """
        best = None
        best_key = float("inf")
        for f in ctx.floors:
            if f.x <= x <= f.x + f.w:
                if f.y - 30 <= y <= f.y + f.h + 30:
                    d = abs(f.y - y)
                    if d < best_key:
                        best_key = d
                        best = f.y
        return best

    # =========================================================================
    # 探索
    # =========================================================================

    def _explore(self, ctx: Context):
        """画面里没怪时，往一个方向走探索。

        行为:
          - 往探索方向走
          - 遇到平台边缘（脚下没地板）就跳
          - 卡住时反向走
          - 长时间没遇到怪就换方向

        自身定位失败（迷失）时走"恢复定位"分支：
          左右来回走 + 定期跳跃。让角色名字/身体从地图遮挡中露出来，
          尽快被 OCR/模板重新定位；避免无脑朝一个方向跑出屏幕，
          越跑越定位不到（老版本一直往右跑就是这个问题）。
        """
        self._explore_frame_count += 1

        # ---- 迷失恢复：自身定位失败，随机左右走 + 随机跳跃（按秒计时）----
        if ctx.self_position is None:
            now = time.time()
            # 卡住 → 随机方向 + 跳跃（少数情况下角色卡在地形里）
            if self._stuck_counter >= STUCK_FRAMES:
                self._log("[探索] 未定位且卡住，跳跃并随机换方向")
                self._explore_direction = random.choice(["left", "right"])
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
                self._stuck_counter = 0
                self._lost_direction_start = now
                self._lost_switch_seconds = random.uniform(
                    LOST_DIRECTION_SWITCH_MIN, LOST_DIRECTION_SWITCH_MAX
                )
                return
            # 随机间隔跳跃：让名字/身体从遮挡层露出来便于重新定位
            if now - self._lost_last_jump_time >= self._lost_jump_seconds:
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
                self._lost_last_jump_time = now
                self._lost_jump_seconds = random.uniform(
                    LOST_JUMP_INTERVAL_MIN, LOST_JUMP_INTERVAL_MAX
                )
            # 按秒随机换方向：当前方向持续够随机秒数后，随机抽新方向（各50%）
            if now - self._lost_direction_start >= self._lost_switch_seconds:
                self._explore_direction = random.choice(["left", "right"])
                self._lost_direction_start = now
                self._lost_switch_seconds = random.uniform(
                    LOST_DIRECTION_SWITCH_MIN, LOST_DIRECTION_SWITCH_MAX
                )
                self._log(
                    f"[探索] 自身未定位，随机方向 → {self._explore_direction} "
                    f"(持续{self._lost_switch_seconds:.1f}秒)"
                )
            self._hold_move(self._explore_direction)
            return

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[探索] 卡住了，跳跃并反向")
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 长时间探索没遇到怪，换方向
        if self._explore_frame_count >= EXPLORE_DIRECTION_SWITCH_FRAMES:
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self._explore_frame_count = 0
            self._log(f"[探索] 换方向 → {self._explore_direction}")

        # 检测脚下是否有地板
        if ctx.self_position and not self._has_floor_under(ctx, ctx.self_position):
            self._log("[探索] 脚下没地板，跳跃")
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            return

        # 按住方向键持续往前走
        self._hold_move(self._explore_direction)

    # =========================================================================
    # 技能释放
    # =========================================================================

    def _cast_skill(self):
        """轮转释放技能。"""
        skills = self.config.skills
        if not skills:
            return
        for _ in range(len(skills)):
            skill = skills[self._skill_index % len(skills)]
            self._skill_index += 1
            if self.executor.press_key(skill["key"], skill["cooldown"]):
                self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                break