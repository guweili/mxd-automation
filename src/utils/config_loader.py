"""配置加载与配置实体。

================================================================================
配置系统设计
================================================================================

  配置分为两层:
    - 默认配置: 内置在代码中，提供合理的出厂默认值
    - 用户配置: 保存在 config/user.yaml (或 user.json)，覆盖默认值

  双层覆盖机制:
    1. 先加载默认配置（_defaults()）
    2. 再读取 user.yaml / user.json，用用户配置覆盖同名字段
    3. 最终 Config 对象包含合并后的值

  这样用户只需要配置自己关心的字段，其余使用默认值即可。

================================================================================
YAML / JSON 双格式支持
================================================================================

  优先读取 config/user.yaml（推荐格式，支持中文注释）。
  若 user.yaml 不存在，回退读取 config/user.json（向后兼容）。
  保存时统一写入 user.yaml。

================================================================================
坐标自适应
================================================================================

  问题: 配置文件中的 HP 区域 / 自身偏移 等坐标是参考 1366×768 分辨率
       记录的，但实际运行时窗口可能不同（如 1920×1080）。

  解决: scale_region() 和 scale_offset() 根据参考分辨率与当前帧的比例，
       自动缩放坐标值。

  公式:
    scale_x = 当前帧宽 / 参考帧宽
    scale_y = 当前帧高 / 参考帧高
    scaled_x = x * scale_x
    scaled_y = y * scale_y

================================================================================
YAML 配置字段说明
================================================================================

  window_title:     游戏窗口标题（用于 FindWindow 锁定）
  reference_width:  参考分辨率宽度（坐标记录时的分辨率）
  reference_height: 参考分辨率高度
  fps:              目标帧率
  confidence:       YOLO 检测置信度阈值 (0.0~1.0)
  model_path:       YOLO 模型文件路径
  monster_classes:  怪物类别名（逗号分隔）
  floor_classes:    地板类别名
  rope_classes:    绳索类别名
  self_name:        自身角色名字（用于 OCR 定位）
  self_offset:      自身 HP 条底部到脚底的偏移像素数
  hp_region:        HP 条参考区域 [x, y, w, h] 或百分比
  hp_color:         HP 条颜色 [R, G, B]
  hp_tolerance:     HP 条颜色容差
  hp_threshold:     HP 加血阈值 (0.0~1.0)
  hp_key:           加血键
  mp_region:        MP 条参考区域
  mp_color:         MP 条颜色
  mp_tolerance:     MP 条颜色容差
  mp_threshold:     MP 加蓝阈值
  mp_key:           加蓝键
  target_key:       选目标键
  jump_key:         跳跃键
  skills:           技能列表 [{name, key, cooldown}, ...]
"""
import os
import sys
from typing import List, Optional, Dict, Any, Tuple

import yaml


# ---- 路径常量 ----
def _bundle_dir() -> str:
    """打包内资源目录（只读）。

    PyInstaller 打包后: sys._MEIPASS
      - onedir 模式  = <程序目录>/_internal
      - onefile 模式 = 运行时临时解压目录（每次启动都重新生成）
    开发环境:          PROJECT_ROOT（项目根目录）
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _app_dir() -> str:
    """应用数据目录（用户可见、可写的目录）。

    模型与配置文件外置于此，方便用户查看/替换，且不会因为
    onefile 解压到临时目录而丢失:
      - PyInstaller 打包后 = exe 所在目录
      - 开发环境          = 项目根目录
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return _bundle_dir()


# 兼容旧名称：BUNDLE_DIR 即原 PROJECT_ROOT
BUNDLE_DIR = _bundle_dir()
PROJECT_ROOT = BUNDLE_DIR
APP_DIR = _app_dir()

# 配置文件：优先 exe 旁边（外置、可持久化），打包内仅作为兜底默认值
CONFIG_DIR = os.path.join(APP_DIR, "config")
DEFAULT_YAML_PATH = os.path.join(CONFIG_DIR, "user.yaml")
DEFAULT_JSON_PATH = os.path.join(CONFIG_DIR, "user.json")


def resolve_model_path(raw_path: str) -> str:
    """解析模型路径。

    - 绝对路径: 直接返回
    - 相对路径: 优先在 APP_DIR（exe 旁边，外置模型）查找；
      不存在时回退 BUNDLE_DIR（打包内，兼容旧版把模型打进包的情况）。
    """
    if not raw_path:
        return raw_path
    if os.path.isabs(raw_path):
        return raw_path
    for base in (APP_DIR, BUNDLE_DIR):
        cand = os.path.normpath(os.path.join(base, raw_path))
        if os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(APP_DIR, raw_path))


# ---- 角色外观模板路径 ----
# 用户通过界面"上传角色全身照"上传的截图统一保存到此处（exe 旁边，可写、持久化）。
# UI 上传、主循环加载、日志提示都应引用同一个路径，避免多处写死不一致。
TEMPLATE_FILENAME = "player_template.png"


def resolve_template_path() -> str:
    """返回角色外观模板的绝对路径。

    始终指向 APP_DIR/assets/templates/player_template.png（exe 旁边，开发环境 = 项目根目录）。
    该文件由界面"上传角色全身照"写入；开发环境也可能直接手动放置同名文件。
    与 resolve_model_path 不同：模板只存一份用户数据，不回退 BUNDLE_DIR。
    """
    return os.path.join(APP_DIR, "assets", "templates", TEMPLATE_FILENAME)


def config_path() -> str:
    """返回当前生效的配置文件路径。

    优先级:
      1. APP_DIR/config/user.yaml    （exe 旁边，用户配置）
      2. APP_DIR/config/user.json
      3. BUNDLE_DIR/config/user.yaml （打包内默认配置）
      4. BUNDLE_DIR/config/user.json
    都不存在时返回 APP_DIR/config/user.yaml（首次启动后保存到这里）。
    """
    for p in (
        os.path.join(APP_DIR, "config", "user.yaml"),
        os.path.join(APP_DIR, "config", "user.json"),
        os.path.join(BUNDLE_DIR, "config", "user.yaml"),
        os.path.join(BUNDLE_DIR, "config", "user.json"),
    ):
        if os.path.isfile(p):
            return p
    return os.path.join(APP_DIR, "config", "user.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置。"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _load_json(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载配置（向后兼容）。"""
    if not os.path.isfile(path):
        return {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_user_config() -> Dict[str, Any]:
    """加载用户配置：优先 exe 旁边（外置）user.yaml，回退打包内默认。"""
    data = _load_yaml(config_path())
    if not data:
        data = _load_json(config_path())
    return data


def _defaults() -> Dict[str, Any]:
    """返回默认配置字典。

    这里定义了所有配置项的默认值。用户 YAML/JSON 中未配置的字段
    会使用这些默认值。

    修改这些默认值会影响所有用户（除非用户在 user.yaml 中覆盖）。

    Returns:
        包含所有默认配置的字典
    """
    return {
        # ---- 窗口 ----
        "window_title": "",
        # ---- 热键 ----
        "start_stop_hotkey": "F6",
        # ---- 分辨率自适应 ----
        # 参考分辨率：坐标录制时的分辨率，所有坐标配置都基于此分辨率
        "reference_width": 1366,
        "reference_height": 768,
        # ---- 性能 ----
        "fps": 13,
        # ---- 执行 ----
        # 键盘注入模式:
        #   使用 SendInput 驱动层模拟真实全局按键，游戏窗口必须在前台。
        #   冒险岛通过 DirectInput 读取全局键盘状态，这是唯一有效的方式。
        "keyboard_mode": "sendinput",
        # ---- 模型 ----
        "confidence": 0.5,
        "model_path": "best.onnx",  # 相对路径，解析到 exe 旁边的模型
        # ---- 类别 ----
        "monster_classes": "monster",
        "floor_classes": "floor",
        "rope_classes": "rope",
        # ---- 自身定位 ----
        "self_name": "",  # 角色脚底名字，用于 OCR 定位
        "self_offset": 85,  # HP 条底部 → 脚底的偏移像素
        "template_confidence": 0.55,  # 截图模板定位置信度阈值（0~1），
                                      # 名字被遮挡时截图兜底定位用。过低易误
                                      # 匹配 UI/NPC，过高导致截图定位不到。
        # ---- HP 检测 ----
        "hp_region": None,  # [x, y, w, h] 参考分辨率下的 HP 条区域
        "hp_color": [51, 204, 51],  # HP 条绿色 RGB
        "hp_tolerance": 30,  # 颜色容差
        "hp_threshold": 0.3,  # 低于 30% 时加血
        "hp_key": "f",  # 加血快捷键
        # ---- MP 检测 ----
        "mp_region": None,
        "mp_color": [51, 153, 255],  # MP 条蓝色 RGB
        "mp_tolerance": 30,
        "mp_threshold": 0.3,
        "mp_key": "g",  # 加蓝快捷键
        # ---- 战斗 ----
        "target_key": "tab",  # 选目标键
        "jump_key": "alt",    # 跳跃键
        "attack_type": "long",  # 攻击类型: "long"(长手) / "short"(短手)
        "attack_range": 200,  # 攻击距离（px）：长手专用（exe 界面可修改）；短手固定 50px 不读此配置
        "attack_range_y": 60, # 攻击垂直容差（px）：垂直差小于此值时怪物就在身边，
                              # 即使路径被推算为 rope/jump 也直接攻击，不绕路
        "skills": [
            {"name": "技能1", "key": "1", "cooldown": 1.0},
            {"name": "技能2", "key": "2", "cooldown": 3.0},
            {"name": "技能3", "key": "3", "cooldown": 8.0},
        ],
        # ---- 拾取 ----
        "pickup_enabled": True,   # 是否启用自动拾取
        "pickup_key": "z",        # 拾取键
        "pickup_interval": 0.333, # 拾取间隔（秒），默认每秒3次
    }


class Config:
    """配置实体类。

    封装双层配置（默认 + 用户），提供属性访问和坐标缩放功能。

    用法:
        cfg = Config.load()  # 从 config/user.yaml 加载
        cfg = Config(overrides={"hp_region": [100, 200, 50, 10]})  # 覆盖某项

    Attributes:
        所有 YAML 配置字段都作为属性直接访问，如 cfg.hp_threshold, cfg.fps 等。
        属性名与 YAML 字段名一致（下划线命名）。
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """构造配置对象。

        Args:
            overrides: 覆盖项字典，键值对会覆盖默认配置
        """
        # 1. 加载默认配置
        data = _defaults()

        # 2. 用用户配置覆盖（YAML 优先，JSON 回退）
        user = _load_user_config()
        data.update(user)

        # 3. 用运行时覆盖项覆盖
        if overrides:
            data.update(overrides)

        self._data = data

    # =========================================================================
    # 工厂方法
    # =========================================================================

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """从指定路径加载配置。

        Args:
            path: 配置文件路径（.yaml 或 .json），None 时自动查找默认路径

        Returns:
            Config 实例
        """
        if path is None:
            path = config_path()

        data = _defaults()
        if path.endswith((".yaml", ".yml")):
            data.update(_load_yaml(path))
        elif path.endswith(".json"):
            data.update(_load_json(path))
        else:
            # 尝试 YAML 再尝试 JSON
            data.update(_load_yaml(path))
            if not data:
                data.update(_load_json(path))

        return cls(overrides=None)

    def save(self, path: Optional[str] = None):
        """保存当前配置到 YAML 文件。

        Args:
            path: 保存路径，None 时使用默认路径 config/user.yaml
        """
        if path is None:
            path = DEFAULT_YAML_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 清理 None 值，使 YAML 更干净
        clean_data = {}
        for k, v in self._data.items():
            if v is not None:
                clean_data[k] = v
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(clean_data, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False, width=120)

    def merge(self, updates: Dict[str, Any]):
        """合并配置项（运行时修改）。

        Args:
            updates: 要更新的键值对
        """
        self._data.update(updates)

    # =========================================================================
    # 属性访问：让配置项像属性一样访问
    # 例如: cfg.hp_threshold 等价于 cfg._data["hp_threshold"]
    # =========================================================================

    def __getattr__(self, name: str):
        """属性访问降级到 _data 字典。

        如果正常属性找不到（如 __dict__ 中没有），
        就在 _data 字典中查找。这样可以让配置项像属性一样访问。

        例如: cfg.hp_threshold → cfg._data["hp_threshold"]
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config 没有 '{name}' 字段")

    def __setattr__(self, name: str, value):
        """属性赋值：保存在 _data 中。

        非私有属性（不以下划线开头）直接写入 _data 字典。
        私有属性（如 _data）正常走标准属性赋值。
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # =========================================================================
    # 坐标自适应缩放
    # =========================================================================

    def scale_region(self, region: Optional[List],
                     frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """将配置区域转换为当前帧像素坐标。

        支持两种格式:
          1. 百分比 [x%, y%, w%, h%]  每个值 0.0~1.0，相对帧宽高
          2. 像素 [x, y, w, h]  整数，基于 reference_width/reference_height 缩放

        Args:
            region:       区域 [x, y, w, h]，None 返回 None
            frame_width:  当前帧宽度（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            区域 (x, y, w, h) 像素坐标，无效时返回 None
        """
        if region is None:
            return None
        if len(region) < 4:
            return None

        # 判断格式：浮点或 <= 1.0 → 百分比，否则 → 像素缩放
        is_percent = any(isinstance(v, float) for v in region) or max(region) <= 1.0

        if is_percent:
            # 百分比：直接乘以帧尺寸
            sx_i = int(region[0] * frame_width)
            sy_i = int(region[1] * frame_height)
            sw_i = int(region[2] * frame_width)
            sh_i = int(region[3] * frame_height)
        else:
            # 像素：按参考分辨率缩放
            sx = frame_width / self.reference_width
            sy = frame_height / self.reference_height
            sx_i = int(region[0] * sx)
            sy_i = int(region[1] * sy)
            sw_i = int(region[2] * sx)
            sh_i = int(region[3] * sy)

        # 边界保护
        if sx_i < 0:
            sw_i += sx_i
            sx_i = 0
        if sy_i < 0:
            sh_i += sy_i
            sy_i = 0
        if sx_i + sw_i > frame_width:
            sw_i = frame_width - sx_i
        if sy_i + sh_i > frame_height:
            sh_i = frame_height - sy_i
        if sw_i <= 0 or sh_i <= 0:
            return None
        return (sx_i, sy_i, sw_i, sh_i)

    def scale_offset(self, offset: int, frame_height: int) -> int:
        """将参考分辨率下的偏移量缩放到当前帧高度。

        公式: scaled_offset = offset * (frame_height / reference_height)

        Args:
            offset:       参考分辨率下的偏移（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            缩放后的偏移量
        """
        return int(offset * frame_height / self.reference_height)


def load_config(path: Optional[str] = None) -> Config:
    """快捷函数：加载配置。

    Args:
        path: 配置文件路径（.yaml / .json），None 自动查找默认路径

    Returns:
        Config 实例
    """
    return Config.load(path)


def save_config(config: Config, path: str):
    """保存配置到指定路径。

    Args:
        config: Config 实例
        path: 保存路径
    """
    config.save(path)


def save_user_config(config: Config):
    """保存配置到默认用户配置文件 (config/user.yaml)。

    Args:
        config: Config 实例
    """
    config.save(DEFAULT_YAML_PATH)