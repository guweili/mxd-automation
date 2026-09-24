"""
自动识别窗口并无感截图的程序
支持：按窗口标题、类名、进程名查找窗口，并静默截图保存
"""

import ctypes
import os
import time
from datetime import datetime
from ctypes import wintypes

import psutil
import win32con
import win32gui
import win32process
from PIL import Image


class WindowInfo:
    """窗口信息数据结构"""

    def __init__(self, hwnd, title, class_name, pid, process_name, rect):
        self.hwnd = hwnd
        self.title = title
        self.class_name = class_name
        self.pid = pid
        self.process_name = process_name
        self.rect = rect  # (left, top, right, bottom)

    def __repr__(self):
        return (
            f"WindowInfo(hwnd={self.hwnd}, title='{self.title}', "
            f"class_name='{self.class_name}', process='{self.process_name}', "
            f"rect={self.rect})"
        )

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]


class WindowDetector:
    """窗口检测器：枚举系统中所有可见窗口"""

    def __init__(self):
        self._windows = []

    def refresh(self):
        """刷新窗口列表"""
        self._windows = []
        win32gui.EnumWindows(self._enum_callback, None)


    def _enum_callback(self, hwnd, _):
        """EnumWindows 回调"""
        if not win32gui.IsWindowVisible(hwnd):
            return True
        if win32gui.IsIconic(hwnd):
            return True

        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return True

        class_name = win32gui.GetClassName(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = self._get_process_name(pid)

        try:
            rect = win32gui.GetWindowRect(hwnd)
        except Exception:
            rect = (0, 0, 0, 0)

        self._windows.append(
            WindowInfo(hwnd, title, class_name, pid, process_name, rect)
        )
        return True

    @staticmethod
    def _get_process_name(pid):
        try:
            return psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "unknown"

    def list_all(self):
        """列出所有窗口"""
        self.refresh()
        return self._windows

    def find_by_title(self, keyword, fuzzy=True):
        """按窗口标题查找，支持模糊匹配"""
        self.refresh()
        results = []
        for w in self._windows:
            if fuzzy and keyword.lower() in w.title.lower():
                results.append(w)
            elif not fuzzy and keyword == w.title:
                results.append(w)
        return results

    def find_by_process(self, process_name):
        """按进程名查找"""
        self.refresh()
        results = []
        for w in self._windows:
            if w.process_name.lower() == process_name.lower():
                results.append(w)
        return results

    def find_by_class(self, class_name):
        """按窗口类名查找"""
        self.refresh()
        results = []
        for w in self._windows:
            if w.class_name.lower() == class_name.lower():
                results.append(w)
        return results

    def find_first(self, keyword=None, process_name=None, class_name=None):
        """多条件查找，返回第一个匹配的窗口"""
        self.refresh()
        for w in self._windows:
            if keyword and keyword.lower() not in w.title.lower():
                continue
            if process_name and process_name.lower() != w.process_name.lower():
                continue
            if class_name and class_name.lower() != w.class_name.lower():
                continue
            return w
        return None

    def print_all(self):
        """打印所有窗口信息"""
        self.refresh()
        print(f"{'HWND':<12} {'标题':<40} {'进程':<20} {'尺寸'}")
        print("-" * 90)
        for w in self._windows:
            print(
                f"{w.hwnd:<12} {w.title[:38]:<40} {w.process_name:<20} "
                f"{w.width}x{w.height}"
            )


class StealthScreenshot:
    """无感截图器：截取指定窗口或全屏，不显示任何界面"""

    def __init__(self, save_dir="screenshots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def capture_window(self, hwnd):
        """截取指定窗口（即使被遮挡也能截取）"""
        from ctypes import windll

        user32 = windll.user32

        try:
            rect = win32gui.GetWindowRect(hwnd)
        except Exception:
            return None

        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        if width <= 0 or height <= 0:
            return None

        hwnd_dc = user32.GetWindowDC(hwnd)
        if not hwnd_dc:
            return None

        mfc_dc = windll.gdi32.CreateCompatibleDC(hwnd_dc)
        if not mfc_dc:
            user32.ReleaseDC(hwnd, hwnd_dc)
            return None

        save_bitmap = windll.gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        if not save_bitmap:
            windll.gdi32.DeleteDC(mfc_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)
            return None

        windll.gdi32.SelectObject(mfc_dc, save_bitmap)
        result = windll.gdi32.BitBlt(
            mfc_dc, 0, 0, width, height, hwnd_dc, 0, 0, win32con.SRCCOPY
        )

        if result:
            bmp_info = ctypes.create_string_buffer(32)
            windll.gdi32.GetObjectA(save_bitmap, 32, bmp_info)
            data_size = (
                ((width * 32 + 31) // 32) * 4 * height
            )
            bmp_data = ctypes.create_string_buffer(data_size)
            windll.gdi32.GetBitmapBits(save_bitmap, data_size, bmp_data)

            img = Image.frombuffer(
                "RGB", (width, height), bmp_data.raw, "raw", "BGRX", 0, 1
            )
        else:
            img = None

        windll.gdi32.DeleteObject(save_bitmap)
        windll.gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        return img

    def capture_foreground(self):
        """截取当前前台窗口"""
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            return self.capture_window(hwnd)
        return None

    def capture_fullscreen(self):
        """截取全屏"""
        import ctypes

        user32 = ctypes.windll.user32
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)

        hdc = user32.GetDC(0)
        if not hdc:
            return None

        mfc_dc = ctypes.windll.gdi32.CreateCompatibleDC(hdc)
        if not mfc_dc:
            user32.ReleaseDC(0, hdc)
            return None

        save_bitmap = ctypes.windll.gdi32.CreateCompatibleBitmap(hdc, width, height)
        if not save_bitmap:
            ctypes.windll.gdi32.DeleteDC(mfc_dc)
            user32.ReleaseDC(0, hdc)
            return None

        ctypes.windll.gdi32.SelectObject(mfc_dc, save_bitmap)
        result = ctypes.windll.gdi32.BitBlt(
            mfc_dc, 0, 0, width, height, hdc, 0, 0, win32con.SRCCOPY
        )

        if result:
            bmp_info = ctypes.create_string_buffer(32)
            ctypes.windll.gdi32.GetObjectA(save_bitmap, 32, bmp_info)
            data_size = ((width * 32 + 31) // 32) * 4 * height
            bmp_data = ctypes.create_string_buffer(data_size)
            ctypes.windll.gdi32.GetBitmapBits(save_bitmap, data_size, bmp_data)
            img = Image.frombuffer(
                "RGB", (width, height), bmp_data.raw, "raw", "BGRX", 0, 1
            )
        else:
            img = None

        ctypes.windll.gdi32.DeleteObject(save_bitmap)
        ctypes.windll.gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(0, hdc)

        return img

    def save(self, img, prefix="screenshot"):
        """保存截图到文件"""
        if img is None:
            print("截图失败：图像为空")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        img.save(filepath, "PNG")
        return filepath


class AutoScreenshot:
    """
    自动截图器：组合窗口检测和无感截图，提供便捷接口
    """

    def __init__(self, save_dir="screenshots"):
        self.detector = WindowDetector()
        self.screenshot = StealthScreenshot(save_dir)

    def capture_by_title(self, keyword, fuzzy=True, save_prefix="window"):
        """按窗口标题自动截图"""
        windows = self.detector.find_by_title(keyword, fuzzy=fuzzy)
        results = []
        for w in windows:
            img = self.screenshot.capture_window(w.hwnd)
            if img:
                path = self.screenshot.save(img, prefix=save_prefix)
                results.append((w, path))
        return results

    def capture_by_process(self, process_name, save_prefix="process"):
        """按进程名自动截图"""
        windows = self.detector.find_by_process(process_name)
        results = []
        for w in windows:
            img = self.screenshot.capture_window(w.hwnd)
            if img:
                path = self.screenshot.save(img, prefix=save_prefix)
                results.append((w, path))
        return results

    def capture_by_class(self, class_name, save_prefix="class"):
        """按窗口类名自动截图"""
        windows = self.detector.find_by_class(class_name)
        results = []
        for w in windows:
            img = self.screenshot.capture_window(w.hwnd)
            if img:
                path = self.screenshot.save(img, prefix=save_prefix)
                results.append((w, path))
        return results

    def capture_first(self, keyword=None, process_name=None, class_name=None):
        """多条件查找，截取第一个匹配窗口"""
        w = self.detector.find_first(keyword, process_name, class_name)
        if w is None:
            print("未找到匹配的窗口")
            return None, None
        img = self.screenshot.capture_window(w.hwnd)
        if img:
            path = self.screenshot.save(img)
            return w, path
        return w, None

    def monitor_and_capture(self, keyword, interval=5, max_count=None):
        """
        持续监控指定窗口并按间隔截图
        interval: 截图间隔（秒）
        max_count: 最大截图次数，None 表示无限
        """
        count = 0
        print(f"开始监控窗口（关键词: '{keyword}'，间隔: {interval}秒）")
        print("按 Ctrl+C 停止...")

        try:
            while max_count is None or count < max_count:
                windows = self.detector.find_by_title(keyword)
                if windows:
                    for w in windows:
                        img = self.screenshot.capture_window(w.hwnd)
                        if img:
                            path = self.screenshot.save(img, prefix="monitor")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                  f"已截图: {w.title[:30]} -> {path}")
                            count += 1
                            if max_count and count >= max_count:
                                break
                else:
                    pass  # 窗口未找到，静默等待
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n监控已停止，共截图 {count} 次")

    def list_windows(self):
        """列出所有可见窗口"""
        self.detector.print_all()


def main():
    """自动识别冒险岛窗口，每10秒无感截图一次"""
    save_dir = "screenshots"
    detector = WindowDetector()
    screenshot = StealthScreenshot(save_dir)

    WINDOW_KEYWORDS = ["冒险岛", "MapleStory", "Maple Story"]
    PROCESS_NAMES = ["maplestory.exe", "maple story.exe"]

    print("=" * 50)
    print("  冒险岛 自动无感截图程序")
    print("  截图间隔: 10秒")
    print("=" * 50)

    count = 0
    print("正在搜索冒险岛窗口...")
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            target_hwnd = None
            target_title = ""

            detector.refresh()
            for w in detector._windows:
                title_lower = w.title.lower()
                proc_lower = w.process_name.lower()
                for kw in WINDOW_KEYWORDS:
                    if kw.lower() in title_lower:
                        target_hwnd = w.hwnd
                        target_title = w.title
                        break
                if not target_hwnd:
                    for pn in PROCESS_NAMES:
                        if pn.lower() == proc_lower:
                            target_hwnd = w.hwnd
                            target_title = w.title
                            break
                if target_hwnd:
                    break

            if target_hwnd:
                count += 1
                img = screenshot.capture_window(target_hwnd)
                if img:
                    path = screenshot.save(img, prefix="mxd")
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] 第{count}次截图: {path}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 截图失败，等待重试...")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 未找到冒险岛窗口，继续搜索...")

            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n已停止，共截图 {count} 次")


if __name__ == "__main__":
    main()