"""MXD 游戏辅助控制台 - 程序入口（GUI 模式）。

================================================================================
启动方式
================================================================================

  GUI 模式（推荐）:
    python main.py
    启动 PyQt5 图形界面，可以锁定窗口、配置参数、查看实时预览。

  CLI 模式（无 GUI）:
    python -m src.main
    纯命令行模式，适合调试或服务器运行。

================================================================================
sys.path 处理
================================================================================

  项目根目录被加入 sys.path，这样 `src` 和 `ui` 可以作为顶层包导入:
    from ui.main_window import MainWindow
    from src.perception.yolo_detector import Detector

  这避免了相对导入的路径问题。
"""
import sys
import os
import ctypes

# ---- DPI 感知必须在 PyQt5 导入/创建 QApplication 之前设置 ----
# PyQt5 在创建 QApplication 时会自动设置 DPI 模式，如果先创建了 QApplication
# 再调用 SetProcessDPIAware，该调用将失效，导致 BitBlt 截图在高 DPI 屏幕上
# 出现右上角偏移 10~30 像素的问题。
#
# 同时禁用 Qt 内部的 DPI 缩放（让 Qt 渲染使用逻辑像素，而我们的 BitBlt
# 使用物理像素，两者不一致会导致 UI 映射坐标到物理像素时出现偏移）。
#
# QT_SCALE_FACTOR=1: 禁用 Qt 的字体/控件缩放
# QT_ENABLE_HIGHDPI_SCALING=0: 禁用 Qt 高 DPI 自适应（PyQt5 5.14+ 默认开启）
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

# 设置 Windows 进程级 DPI 感知
try:
    # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE)-4
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# 把项目根目录加入 sys.path，让 `src` / `ui` 作为顶层包可被导入
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# 禁用 Qt 内部高 DPI 缩放（PyQt5 5.14+），让 Qt 使用物理像素坐标
# 这样屏幕坐标、鼠标坐标、BitBlt 截图坐标全部一致
try:
    QApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, False)
except Exception:
    pass

from ui.main_window import MainWindow


def main():
    """启动 PyQt5 GUI 主窗口。

    流程:
      1. 创建 QApplication（PyQt5 应用实例）
      2. 创建 MainWindow（主窗口，包含所有 UI 控件）
      3. 显示窗口
      4. 进入事件循环（app.exec_()），直到用户关闭窗口
    """
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    # app.exec_() 启动 Qt 事件循环，阻塞直到窗口关闭
    # 返回值是退出码，传给 sys.exit
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()