"""MXD 游戏辅助控制台主窗口。

布局：
  顶部: 窗口锁定 (下拉选择 + 刷新 + 锁定)
  左侧: 实时预览 (检测框叠加) + 开始/停止 + 日志
  右侧: 配置面板 (检测 / 血量 / 蓝量 / 战斗 / 热键)

依赖：
  - ``src.main.Automation``：主循环
  - ``src.utils.config_loader``：配置加载
  - ``src.perception``：检测器与区域颜色识别
  - ``ui.preview_label.PreviewLabel``：预览与框选
"""
import os
import sys
import time

# 确保项目根目录在 sys.path 中，使得 `src` / `ui` 可作为顶层包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QCheckBox, QFileDialog, QMessageBox, QSplitter,
    QAbstractItemView, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from src.utils.config_loader import (
    load_config, save_config, save_user_config, config_path, resolve_model_path,
    APP_DIR, BUNDLE_DIR,
)
from src.perception.yolo_detector import create_detector
from src.perception.hp_mp_detector import detect_region_color
from src.main import Automation

from ui.preview_label import PreviewLabel


class MainWindow(QMainWindow):
    # 跨线程信号：自动化线程 → GUI 线程
    log_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(object, object, object, object)  # frame, detections, hp, mp
    hotkey_signal = pyqtSignal()  # F12 触发

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXD 游戏辅助控制台")
        self.resize(1180, 760)

        self.config = load_config()
        # 先解析为绝对路径再创建检测器，避免 CWD 不对导致 os.path.exists 失败
        model_path = resolve_model_path(self.config.model_path) if self.config.model_path else ""
        self.detector = create_detector(
            model_path, self.config.confidence, self._log
        )
        self.automation = Automation(
            self.config, self.detector,
            on_log=self.log_signal.emit,
            on_frame=self.frame_signal.emit,
        )

        self._fps_counter = [0, time.time()]
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_fps)

        self._init_ui()
        self._load_config_to_ui()

        # 攻击距离/近战距离（统一）修改时实时同步到 YAML
        self.distance_spin.valueChanged.connect(self._on_distance_changed)
        self.attack_range_y_spin.valueChanged.connect(self._on_attack_range_y_changed)
        self.attack_type_combo.currentIndexChanged.connect(self._on_attack_type_changed)

        self.log_signal.connect(self._on_log)
        self.frame_signal.connect(self._on_frame)
        self.hotkey_signal.connect(self._toggle_run)

        self._register_hotkey()

    # ---------------- UI 构建 ----------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 顶部: 窗口锁定
        root.addWidget(self._build_window_bar())

        # 主体: 左侧预览+日志 / 右侧配置
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.statusBar().showMessage("就绪。请锁定游戏窗口后点击「开始」。")

    def _build_window_bar(self):
        box = QGroupBox("游戏窗口")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("窗口标题:"))
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.setMinimumWidth(360)
        h.addWidget(self.window_combo, 1)
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self._refresh_windows)
        h.addWidget(self.refresh_btn)
        self.lock_btn = QPushButton("锁定窗口")
        self.lock_btn.clicked.connect(self._lock_window)
        h.addWidget(self.lock_btn)
        self.win_status = QLabel("未锁定")
        self.win_status.setStyleSheet("color: #c0392b; font-weight:bold;")
        h.addWidget(self.win_status)
        return self._wrap(box)

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        # 预览
        self.preview = PreviewLabel()
        self.preview.region_selected.connect(self._on_region_selected)
        v.addWidget(self.preview, 1)

        # FPS
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet("color: #27ae60; font-weight:bold;")
        v.addWidget(self.fps_label)

        # 控制按钮
        ctl = QHBoxLayout()
        self.run_btn = QPushButton("▶ 开始自动打怪")
        self.run_btn.setStyleSheet(
            "padding:10px; font-size:14px; font-weight:bold; "
            "background-color:#27ae60; color:white;"
        )
        self.run_btn.clicked.connect(self._toggle_run)
        ctl.addWidget(self.run_btn)
        self.hp_pick_btn = QPushButton("框选血条区域")
        self.hp_pick_btn.clicked.connect(
            lambda: self.preview.set_select_mode(True, "hp")
        )
        ctl.addWidget(self.hp_pick_btn)
        self.mp_pick_btn = QPushButton("框选蓝条区域")
        self.mp_pick_btn.clicked.connect(
            lambda: self.preview.set_select_mode(True, "mp")
        )
        ctl.addWidget(self.mp_pick_btn)
        v.addLayout(ctl)

        # 日志
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(180)
        self.log_box.setStyleSheet("background-color:#111; color:#ddd;")
        v.addWidget(self.log_box)
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(self._build_detect_group())
        v.addWidget(self._build_hp_group())
        v.addWidget(self._build_mp_group())
        v.addWidget(self._build_combat_group())
        v.addStretch()
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self._save_config)
        v.addWidget(save_btn)
        return panel

    def _build_detect_group(self):
        box = QGroupBox("检测设置 (YOLO)")
        g = QGridLayout(box)
        g.addWidget(QLabel("模型/EXE路径:"), 0, 0)
        self.model_edit = QLineEdit()
        g.addWidget(self.model_edit, 0, 1)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_model)
        g.addWidget(browse, 0, 2)

        g.addWidget(QLabel("置信度:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        self.conf_label = QLabel("0.50")
        self.conf_label.setMinimumWidth(40)
        g.addWidget(self.conf_slider, 1, 1)
        g.addWidget(self.conf_label, 1, 2)

        g.addWidget(QLabel("怪物类别:"), 2, 0)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("逗号分隔, 如 monster,boss")
        g.addWidget(self.classes_edit, 2, 1, 1, 2)

        g.addWidget(QLabel("检测FPS:"), 3, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(8)
        g.addWidget(self.fps_spin, 3, 1, 1, 2)

        g.addWidget(QLabel("自身名字:"), 4, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入角色名, 如 我是立立")
        g.addWidget(self.name_edit, 4, 1, 1, 2)

        g.addWidget(QLabel("角色模板:"), 5, 0)
        template_row = QHBoxLayout()
        self.template_btn = QPushButton("上传角色全身照")
        self.template_btn.clicked.connect(self._browse_template)
        template_row.addWidget(self.template_btn)
        self.template_status = QLabel("未上传")
        self.template_status.setStyleSheet("color:#c0392b;")
        template_row.addWidget(self.template_status, 1)
        g.addLayout(template_row, 5, 1, 1, 2)
        return box

    def _build_hp_group(self):
        box = QGroupBox("血量设置")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("按键:"))
        self.hp_key_edit = QLineEdit()
        self.hp_key_edit.setFixedWidth(80)
        self.hp_key_edit.setPlaceholderText("f / hm")
        h.addWidget(self.hp_key_edit)

        h.addWidget(QLabel("阈值%:"))
        self.hp_thr_spin = QSpinBox()
        self.hp_thr_spin.setRange(0, 100)
        self.hp_thr_spin.setValue(50)
        self.hp_thr_spin.setFixedWidth(55)
        h.addWidget(self.hp_thr_spin)

        h.addWidget(QLabel("颜色:"))
        self.hp_swatch = QLabel()
        self.hp_swatch.setFixedSize(24, 24)
        self.hp_swatch.setStyleSheet("background-color: rgb(255,0,0); border:1px solid #333;")
        h.addWidget(self.hp_swatch)

        h.addWidget(QLabel("区域:"))
        self.hp_region_label = QLabel("未设置")
        self.hp_region_label.setMinimumWidth(60)
        h.addWidget(self.hp_region_label)

        h.addStretch()
        return box

    def _build_mp_group(self):
        box = QGroupBox("蓝量设置")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("按键:"))
        self.mp_key_edit = QLineEdit()
        self.mp_key_edit.setFixedWidth(80)
        self.mp_key_edit.setPlaceholderText("g / pu")
        h.addWidget(self.mp_key_edit)

        h.addWidget(QLabel("阈值%:"))
        self.mp_thr_spin = QSpinBox()
        self.mp_thr_spin.setRange(0, 100)
        self.mp_thr_spin.setValue(30)
        self.mp_thr_spin.setFixedWidth(55)
        h.addWidget(self.mp_thr_spin)

        h.addWidget(QLabel("颜色:"))
        self.mp_swatch = QLabel()
        self.mp_swatch.setFixedSize(24, 24)
        self.mp_swatch.setStyleSheet("background-color: rgb(0,120,255); border:1px solid #333;")
        h.addWidget(self.mp_swatch)

        h.addWidget(QLabel("区域:"))
        self.mp_region_label = QLabel("未设置")
        self.mp_region_label.setMinimumWidth(60)
        h.addWidget(self.mp_region_label)

        h.addStretch()
        return box

    def _build_combat_group(self):
        box = QGroupBox("战斗设置")
        v = QVBoxLayout(box)

        # 攻击类型 + 攻击距离（放到上面）
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("攻击类型:"))
        self.attack_type_combo = QComboBox()
        self.attack_type_combo.addItem("长手 (远程攻击)", "long")
        self.attack_type_combo.addItem("短手 (近战攻击)", "short")
        self.attack_type_combo.setToolTip(
            "长手: 远程职业(弓/弩/法), 在较远距离攻击\n"
            "短手: 近战职业(战/盗), 贴近怪物才能攻击"
        )
        self.attack_type_combo.setFixedWidth(140)
        type_row.addWidget(self.attack_type_combo)

        self.distance_label = QLabel("攻击距离px:")
        type_row.addWidget(self.distance_label)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(10, 800)
        self.distance_spin.setValue(200)
        self.distance_spin.setToolTip("人物与怪物水平差小于此值才触发攻击")
        self.distance_spin.setFixedWidth(70)
        type_row.addWidget(self.distance_spin)

        type_row.addStretch()
        v.addLayout(type_row)

        # 第二行：跳跃键 + 垂直容差
        row = QHBoxLayout()
        row.addWidget(QLabel("跳跃键:"))
        self.jump_key_edit = QLineEdit("alt")
        self.jump_key_edit.setPlaceholderText("alt / space")
        self.jump_key_edit.setFixedWidth(60)
        row.addWidget(self.jump_key_edit)

        row.addWidget(QLabel("垂直容差px:"))
        self.attack_range_y_spin = QSpinBox()
        self.attack_range_y_spin.setRange(10, 300)
        self.attack_range_y_spin.setValue(60)
        self.attack_range_y_spin.setToolTip("垂直差小于此值时即使不同层也直接攻击（不绕路）")
        self.attack_range_y_spin.setFixedWidth(70)
        row.addWidget(self.attack_range_y_spin)

        row.addStretch()
        v.addLayout(row)

        # 拾取设置
        pickup_row = QHBoxLayout()
        self.pickup_checkbox = QCheckBox("自动拾取")
        self.pickup_checkbox.setToolTip("勾选后自动按拾取键捡东西")
        pickup_row.addWidget(self.pickup_checkbox)
        pickup_row.addWidget(QLabel("拾取键:"))
        self.pickup_key_edit = QLineEdit("z")
        self.pickup_key_edit.setPlaceholderText("z")
        self.pickup_key_edit.setFixedWidth(50)
        pickup_row.addWidget(self.pickup_key_edit)
        pickup_row.addWidget(QLabel("间隔(ms):"))
        self.pickup_interval_spin = QSpinBox()
        self.pickup_interval_spin.setRange(100, 2000)
        self.pickup_interval_spin.setValue(333)
        self.pickup_interval_spin.setSuffix("ms")
        self.pickup_interval_spin.setToolTip("拾取间隔（毫秒），333ms=每秒3次")
        self.pickup_interval_spin.setFixedWidth(80)
        pickup_row.addWidget(self.pickup_interval_spin)
        pickup_row.addStretch()
        v.addLayout(pickup_row)

        # 技能表
        v.addWidget(QLabel("技能列表 (轮转释放):"))
        self.skill_table = QTableWidget(0, 3)
        self.skill_table.setHorizontalHeaderLabels(["名称", "按键", "冷却(秒)"])
        self.skill_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.skill_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self.skill_table)

        skill_btns = QHBoxLayout()
        add_btn = QPushButton("+ 添加")
        add_btn.clicked.connect(self._add_skill_row)
        del_btn = QPushButton("- 删除选中")
        del_btn.clicked.connect(self._del_skill_row)
        skill_btns.addWidget(add_btn)
        skill_btns.addWidget(del_btn)
        skill_btns.addStretch()
        v.addLayout(skill_btns)
        return box

    @staticmethod
    def _wrap(widget):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(widget)
        return w

    # ---------------- 配置 ↔ UI ----------------
    def _load_config_to_ui(self):
        c = self.config
        # 显示解析后的完整路径，让用户能直观看到模型实际位置
        # （打包后相对路径 best.onnx 会解析为 _internal\best.onnx）
        self.model_edit.setText(resolve_model_path(c.model_path) if c.model_path else "")
        self.conf_slider.setValue(int(c.confidence * 100))
        self.classes_edit.setText(c.monster_classes)
        self.fps_spin.setValue(c.fps)
        self.name_edit.setText(c.self_name)
        # 角色模板状态：检查 exe 旁边（开发环境=项目根目录）是否已存在
        if os.path.isfile(
            os.path.join(APP_DIR, "assets", "templates", "player_template.png")
        ):
            self.template_status.setText("已加载")
            self.template_status.setStyleSheet("color:#27ae60;")
        else:
            self.template_status.setText("未上传")
            self.template_status.setStyleSheet("color:#c0392b;")
        self.hp_key_edit.setText(c.hp_key)
        self.hp_thr_spin.setValue(int(c.hp_threshold * 100))
        if c.hp_color:
            self._set_swatch(self.hp_swatch, c.hp_color)
        if c.hp_region:
            self.hp_region_label.setText(
                f"x={c.hp_region[0]:.1%} y={c.hp_region[1]:.1%} "
                f"w={c.hp_region[2]:.1%} h={c.hp_region[3]:.1%}"
            )
        # MP
        self.mp_key_edit.setText(c.mp_key)
        self.mp_thr_spin.setValue(int(c.mp_threshold * 100))
        if c.mp_color:
            self._set_swatch(self.mp_swatch, c.mp_color)
        if c.mp_region:
            self.mp_region_label.setText(
                f"x={c.mp_region[0]:.1%} y={c.mp_region[1]:.1%} "
                f"w={c.mp_region[2]:.1%} h={c.mp_region[3]:.1%}"
            )
        self.jump_key_edit.setText(c.jump_key)
        # 攻击类型
        atk_type = getattr(c, "attack_type", "long")
        idx = self.attack_type_combo.findData(atk_type)
        self.attack_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        # 攻击距离：长手读 config.attack_range，短手固定 50（_sync_distance_ui 内处理）
        self.distance_spin.setValue(int(getattr(c, "attack_range", 200)))
        self.attack_range_y_spin.setValue(int(getattr(c, "attack_range_y", 60)))
        self._sync_distance_ui()
        # 拾取
        self.pickup_checkbox.setChecked(getattr(c, "pickup_enabled", True))
        self.pickup_key_edit.setText(getattr(c, "pickup_key", "z"))
        self.pickup_interval_spin.setValue(int(getattr(c, "pickup_interval", 0.333) * 1000))
        # 技能表
        self.skill_table.setRowCount(0)
        for s in c.skills:
            self._add_skill_row(s.get("name", ""), s.get("key", ""), s.get("cooldown", 1.0))

    def _read_ui_to_config(self):
        c = self.config
        c.window_title = self.window_combo.currentText().strip()
        # 若文本框里是 exe 旁边(APP_DIR)或打包内(BUNDLE_DIR)模型的绝对路径，
        # 保存时转回相对路径(best.onnx)，避免文件夹移动后路径失效
        _text = self.model_edit.text().strip()
        if getattr(sys, "frozen", False):
            _norm = os.path.normpath(_text)
            for _base in (APP_DIR, BUNDLE_DIR):
                _base_n = os.path.normpath(_base)
                if _norm == _base_n or _norm.startswith(_base_n + os.sep):
                    _text = os.path.relpath(_norm, _base_n)
                    break
        c.model_path = _text
        c.confidence = self.conf_slider.value() / 100
        c.monster_classes = self.classes_edit.text().strip() or "monster"
        c.fps = self.fps_spin.value()
        c.self_name = self.name_edit.text().strip()
        c.hp_key = self.hp_key_edit.text().strip()
        c.hp_threshold = self.hp_thr_spin.value() / 100
        # hp_color / mp_color 由框选区域时自动写入，此处不覆盖
        # mp
        c.mp_key = self.mp_key_edit.text().strip()
        c.mp_threshold = self.mp_thr_spin.value() / 100
        c.jump_key = self.jump_key_edit.text().strip() or "alt"
        c.attack_type = self.attack_type_combo.currentData()
        if c.attack_type == "long":
            c.attack_range = self.distance_spin.value()  # 长手距离可在界面修改
        c.attack_range_y = self.attack_range_y_spin.value()
        # 拾取
        c.pickup_enabled = self.pickup_checkbox.isChecked()
        c.pickup_key = self.pickup_key_edit.text().strip() or "z"
        c.pickup_interval = self.pickup_interval_spin.value() / 1000.0
        # 技能
        skills = []
        for r in range(self.skill_table.rowCount()):
            name = self.skill_table.item(r, 0).text() if self.skill_table.item(r, 0) else ""
            key = self.skill_table.item(r, 1).text() if self.skill_table.item(r, 1) else ""
            cd_text = self.skill_table.item(r, 2).text() if self.skill_table.item(r, 2) else "1.0"
            try:
                cd = float(cd_text)
            except ValueError:
                cd = 1.0
            if name or key:
                skills.append({"name": name, "key": key, "cooldown": cd})
        c.skills = skills

    def _add_skill_row(self, name="", key="", cd=1.0):
        r = self.skill_table.rowCount()
        self.skill_table.insertRow(r)
        self.skill_table.setItem(r, 0, QTableWidgetItem(str(name)))
        self.skill_table.setItem(r, 1, QTableWidgetItem(str(key)))
        self.skill_table.setItem(r, 2, QTableWidgetItem(str(cd)))

    def _del_skill_row(self):
        rows = {i.row() for i in self.skill_table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.skill_table.removeRow(r)

    def _save_config(self):
        self._read_ui_to_config()
        save_user_config(self.config)
        self._log(f"[配置] 已保存到 {config_path()}")


    def _on_attack_range_y_changed(self, value):
        """垂直容差微调框变化时，实时同步到 config 并保存到 YAML。"""
        self.config.attack_range_y = value
        save_user_config(self.config)

    def _on_attack_type_changed(self, index):
        """攻击类型切换时，更新距离 UI 并同步到 config。"""
        self.config.attack_type = self.attack_type_combo.currentData()
        self._sync_distance_ui()
        save_user_config(self.config)

    def _sync_distance_ui(self):
        """根据攻击类型同步距离输入框状态。

        长手: 距离在界面可调（写入 config.attack_range，exe 可修改）。
        短手: 贴脸距离固定 50px，禁用输入框，不读 config.attack_range。
        """
        if self.config.attack_type == "short":
            self.distance_label.setText("近战距离px: (固定50)")
            self.distance_spin.setEnabled(False)
            self.distance_spin.setValue(50)
            self.distance_spin.setToolTip(
                "短手(近战): 贴脸距离固定 50px，不允许修改；"
                "与长手攻击距离完全独立"
            )
        else:
            self.distance_label.setText("攻击距离px:")
            self.distance_spin.setEnabled(True)
            self.distance_spin.setValue(int(getattr(self.config, "attack_range", 200)))
            self.distance_spin.setToolTip(
                "长手(远程): 与怪物水平差≤此值才触发攻击，攻击中轻微波动不中断"
            )

    def _on_distance_changed(self, value):
        """攻击距离微调框变化时，实时同步到 config 并保存到 YAML。

        长手/短手共用同一个 attack_range。
        """
        self.config.attack_range = value
        save_user_config(self.config)

    # ---------------- 事件处理 ----------------
    def _refresh_windows(self):
        self.window_combo.clear()
        try:
            windows = self.automation.list_windows()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"枚举窗口失败: {e}")
            return
        for hwnd, title in windows:
            self.window_combo.addItem(title)
        if self.config.window_title:
            self.window_combo.setCurrentText(self.config.window_title)

    def _lock_window(self):
        title = self.window_combo.currentText().strip()
        if not title:
            QMessageBox.warning(self, "提示", "请先选择或输入窗口标题")
            return
        try:
            locked = self.automation.lock_window(title)
        except Exception as e:
            self.win_status.setText("锁定失败")
            self.win_status.setStyleSheet("color:#c0392b;font-weight:bold;")
            QMessageBox.warning(self, "锁定失败", str(e))
            return
        self.config.window_title = title
        self.win_status.setText(f"已锁定: {locked}")
        self.win_status.setStyleSheet("color:#27ae60;font-weight:bold;")
        self._log(f"[窗口] 已锁定: {locked}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型或 EXE", "", "模型/EXE (*.pt *.onnx *.exe);;所有文件 (*.*)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_template(self):
        """上传角色全身照作为外观模板（换时装/换地图后重新上传，立即生效）。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择角色全身照", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            import shutil
            # 保存到 exe 旁边（开发环境 = 项目根目录），可写、持久化
            save_dir = os.path.join(APP_DIR, "assets", "templates")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "player_template.png")
            shutil.copyfile(path, save_path)
            # 立即热加载，无需重启
            self.automation.set_player_template(save_path)
            self.template_status.setText("已加载")
            self.template_status.setStyleSheet("color:#27ae60;")
            self._log(f"[模板] 角色全身照已上传: {save_path}")
        except Exception as e:
            self.template_status.setText("加载失败")
            self.template_status.setStyleSheet("color:#c0392b;")
            self._log(f"[模板] 上传失败: {e}")
            QMessageBox.warning(self, "模板上传失败", str(e))

    @staticmethod
    def _set_swatch(swatch, rgb):
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #333;"
        )

    def _on_region_selected(self, target, x, y, w, h):
        """框选区域后自动识别颜色并写入配置（存储为百分比）。"""
        frame = self.automation.capture.grab()
        fh, fw = frame.shape[:2]
        # 转换为百分比存储
        region_pct = [x / fw, y / fh, w / fw, h / fh]
        if target == "mp":
            self.config.mp_region = region_pct
            self.mp_region_label.setText(
                f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                f"w={region_pct[2]:.1%} h={region_pct[3]:.1%}"
            )
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.mp_color = color
                self._set_swatch(self.mp_swatch, color)
                self._log(
                    f"[蓝量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色: RGB{tuple(color)}"
                )
            else:
                self._log(
                    f"[蓝量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色识别失败"
                )
        else:
            self.config.hp_region = region_pct
            self.hp_region_label.setText(
                f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                f"w={region_pct[2]:.1%} h={region_pct[3]:.1%}"
            )
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.hp_color = color
                self._set_swatch(self.hp_swatch, color)
                self._log(
                    f"[血量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色: RGB{tuple(color)}"
                )
            else:
                self._log(
                    f"[血量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色识别失败"
                )

    def _detect_region_color(self, x, y, w, h):
        """从当前窗口截图中识别指定区域的主颜色。"""
        try:
            frame = self.automation.capture.grab()
            return detect_region_color(frame, [x, y, w, h])
        except Exception:
            return None

    def _toggle_run(self):
        if self.automation.running:
            self.automation.stop()
            self.run_btn.setText("▶ 开始自动打怪")
            self.run_btn.setStyleSheet(
                "padding:10px;font-size:14px;font-weight:bold;"
                "background-color:#27ae60;color:white;"
            )
            return

        # 启动前: 读 UI → 存配置 → (必要时)重建检测器 → 锁窗口 → 启动
        self._read_ui_to_config()
        save_user_config(self.config)

        if not self.automation.window_locked:
            if self.config.window_title:
                try:
                    self.automation.lock_window(self.config.window_title)
                    self.win_status.setText(f"已锁定: {self.config.window_title}")
                    self.win_status.setStyleSheet("color:#27ae60;font-weight:bold;")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"锁定窗口失败: {e}")
                    return
            else:
                QMessageBox.warning(self, "提示", "请先锁定游戏窗口")
                return

        # 模型路径变化时重建检测器（用 resolved 路径比较，避免 UI 截断/相对绝对路径差异导致误重建）
        resolved_path = resolve_model_path(self.config.model_path) if self.config.model_path else ""
        detector_path = getattr(self.detector, "_path", "")
        # detector._path 可能是相对路径，也做一次 resolve 再比较
        if detector_path:
            detector_path = resolve_model_path(detector_path)
        if resolved_path and resolved_path != detector_path:
            self.detector = create_detector(
                resolved_path, self.config.confidence, self._log
            )
            self.automation.set_detector(self.detector)

        self.automation.config = self.config
        try:
            self.automation.start()
        except Exception as e:
            QMessageBox.warning(self, "启动失败", str(e))
            return

        self.run_btn.setText("⏹ 停止")
        self.run_btn.setStyleSheet(
            "padding:10px;font-size:14px;font-weight:bold;"
            "background-color:#c0392b;color:white;"
        )
        self._preview_timer.start(1000)

    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_frame(self, frame, detections, hp_ratio, mp_ratio):
        if frame is None:
            return
        h, w = frame.shape[:2]
        self.preview.set_frame_size(w, h)

        disp = frame.copy()
        # 按类别使用不同颜色绘制检测框
        monster_classes = [c.strip() for c in self.config.monster_classes.split(",")]
        floor_classes = [c.strip() for c in self.config.floor_classes.split(",")]
        rope_classes = [c.strip() for c in self.config.rope_classes.split(",")]
        for d in detections:
            if d.cls_name in monster_classes:
                color = (0, 255, 0)          # 绿色: 怪物
            elif d.cls_name in rope_classes:
                color = (0, 165, 255)        # 橙色: 绳索
            elif d.cls_name in floor_classes:
                color = (128, 128, 128)      # 灰色: 地板
            else:
                color = (0, 165, 255)        # 默认橙色
            cv2.rectangle(disp, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            cv2.putText(disp, f"{d.cls_name} {d.confidence:.2f}",
                        (d.x, max(0, d.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 画血条区域
        hp_region = self.config.scale_region(
            self.config.hp_region, frame.shape[1], frame.shape[0]
        )
        if hp_region:
            rx, ry, rw, rh = hp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
        # 画蓝条区域
        mp_region = self.config.scale_region(
            self.config.mp_region, frame.shape[1], frame.shape[0]
        )
        if mp_region:
            rx, ry, rw, rh = mp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 128, 0), 1)
        # 自身位置：直接复用主循环已算好的缓存中心点（避免 UI 线程执行 OCR 卡死）
        center = self.automation._get_last_center()
        if center:
            sx, sy = center
            cv2.circle(disp, (sx, sy), 6, (255, 255, 0), -1)
            cv2.putText(disp, "self", (sx + 8, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # HP / MP 文本
        hp_text = f"HP: {hp_ratio:.0%}" if hp_ratio is not None else "HP: -"
        cv2.putText(disp, hp_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mp_text = f"MP: {mp_ratio:.0%}" if mp_ratio is not None else "MP: -"
        cv2.putText(disp, mp_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.preview.update_frame(QPixmap.fromImage(qimg.copy()))

        self._fps_counter[0] += 1

    def _update_fps(self):
        now = time.time()
        fps = self._fps_counter[0] / max(0.001, now - self._fps_counter[1])
        self._fps_counter = [0, now]
        self.fps_label.setText(f"FPS: {fps:.1f}")
        if not self.automation.running:
            self._preview_timer.stop()

    def _log(self, msg):
        self.log_signal.emit(msg)

    def _register_hotkey(self):
        try:
            import keyboard
            keyboard.add_hotkey(self.config.start_stop_hotkey,
                                self.hotkey_signal.emit)
        except Exception as e:
            self._log(f"[热键] 注册 {self.config.start_stop_hotkey} 失败: {e}")

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        try:
            self.automation.stop()
        except Exception:
            pass
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())