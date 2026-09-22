"""窗口锁定与高性能截图。

锁定游戏窗口后使用 Win32 ``BitBlt`` API 截取客户区（游戏内容区）：
  - 使用 GetClientRect + GetDC 获取客户区 DC，排除标题栏/边框
  - 不受屏幕遮挡影响（即使被其他窗口盖住也能截到）
  - 不会截到自己的工具窗
  - 适合窗口模式（可见）的游戏

若游戏是全屏独占 DirectX，需改用 ``mss`` 或 ``dxcam`` 等方案（保留接口、
替换实现即可，外部调用方代码无需改动）。

参考：项目硬约束要求截屏优先使用 mss（比 pyautogui 快 20 倍）；当前
BitBlt 客户区实现针对"窗口锁定 + 不受遮挡 + 排除边框"场景。
"""
import ctypes
import ctypes.wintypes
import win32gui
import win32con
import numpy as np
import cv2


# ---- DPI 感知：让截图返回物理像素而非逻辑像素 ----
# 高 DPI 显示器上，如果不设置 DPI 感知，GetClientRect 返回逻辑坐标
# 而 BitBlt 返回物理像素，两者不匹配导致坐标偏移。
#
# 优先使用 Per-Monitor V2（Win10 1703+，兼容 Win11），确保多显示器、
# 不同 DPI 缩放场景下 GetClientRect 都返回物理像素，避免截图比实际窗口小。
# 回退到 SetProcessDPIAware 兼容旧版 Windows。
#
# 注意：此调用可能因 PyQt5 先初始化而失效（GUI 模式下）。
# 真正生效的 DPI 设置在 main.py 中完成，那里在 QApplication 创建之前调用。
try:
    # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE)-4
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class ScreenCapture:
    """窗口截图基类。子类实现 ``grab()``。

    对外保持与原 ``app/window_capture.py`` 的 ``WindowCapture`` 同名方法，
    以便上层代码无感迁移。
    """

    def __init__(self):
        self._hwnd = None
        self._title = None
        self._last_rect = None  # 上次客户区尺寸诊断

    # ---------------- 窗口枚举/锁定 ----------------

    @staticmethod
    def list_windows():
        """列出所有可见且有标题的窗口，返回 [(hwnd, title), ...]。"""
        results = []

        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    results.append((hwnd, title))

        win32gui.EnumWindows(_cb, None)
        return results

    def lock(self, title=None, hwnd=None) -> str:
        """锁定窗口。可按标题或 hwnd。返回锁定后的窗口标题。"""
        if hwnd is None and title is None:
            raise ValueError("需要提供 title 或 hwnd")
        if hwnd is None:
            hwnd = win32gui.FindWindow(None, title)
            if not hwnd:
                raise ValueError(f"找不到窗口: {title}")
        self._hwnd = hwnd
        self._title = win32gui.GetWindowText(hwnd)
        return self._title

    def unlock(self):
        self._hwnd = None
        self._title = None

    def get_rect(self):
        """返回窗口在屏幕中的矩形 (left, top, width, height)。"""
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        return (left, top, right - left, bottom - top)

    def get_client_rect(self):
        """返回客户区尺寸 (width, height)，使用物理像素。

        用于诊断 DPI 缩放是否导致客户区尺寸与截图帧尺寸不一致。
        """
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")
        rect = ctypes.wintypes.RECT()
        user32 = ctypes.windll.user32
        user32.GetClientRect(self._hwnd, ctypes.byref(rect))
        return (rect.right - rect.left, rect.bottom - rect.top)

    # ---------------- 截图 ----------------

    def grab(self) -> np.ndarray:
        """截取锁定窗口客户区（游戏内容区），返回 BGR numpy 数组。

        使用 ``GetDC(hwnd)`` + ``BitBlt`` 截取客户区而非整个窗口，
        避免标题栏/边框导致的坐标偏移。
        """
        if not self._hwnd:
            raise RuntimeError("未锁定窗口")

        hwnd = self._hwnd
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        # 获取客户区尺寸（游戏实际内容区域，不含标题栏/边框）
        rect = ctypes.wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        width = rect.right - rect.left
        height = rect.bottom - rect.top

        if width <= 0 or height <= 0:
            raise RuntimeError("客户区尺寸无效")

        # 客户区 DC（GetDC 而非 GetWindowDC）
        hwnd_dc = user32.GetDC(hwnd)
        if not hwnd_dc:
            raise RuntimeError("获取窗口 DC 失败")

        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        save_bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        gdi32.SelectObject(mfc_dc, save_bitmap)

        # BitBlt 从客户区 DC 拷贝
        result = gdi32.BitBlt(
            mfc_dc, 0, 0, width, height, hwnd_dc, 0, 0, win32con.SRCCOPY
        )

        if not result:
            user32.ReleaseDC(hwnd, hwnd_dc)
            gdi32.DeleteDC(mfc_dc)
            gdi32.DeleteObject(save_bitmap)
            raise RuntimeError("BitBlt 截图失败")

        # 读取位图数据
        bmi = ctypes.create_string_buffer(32)
        gdi32.GetObjectA(save_bitmap, 32, bmi)
        data_size = ((width * 32 + 31) // 32) * 4 * height
        bmp_data = ctypes.create_string_buffer(data_size)
        gdi32.GetBitmapBits(save_bitmap, data_size, bmp_data)

        # 释放资源
        gdi32.DeleteObject(save_bitmap)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        # 转换为 numpy 数组 (BGRA -> BGR)
        arr = np.frombuffer(bmp_data.raw, dtype=np.uint8)
        row_size = ((width * 32 + 31) // 32) * 4
        arr = arr.reshape(height, row_size)[:, :width * 4]
        frame = arr.reshape(height, width, 4)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ---------------- 属性 ----------------

    @property
    def title(self):
        return self._title

    @property
    def locked(self):
        return self._hwnd is not None

    @property
    def hwnd(self):
        """锁定的窗口句柄（供执行层做按键注入用）。"""
        return self._hwnd


# 向后兼容别名（旧代码用 WindowCapture 类名）
WindowCapture = ScreenCapture