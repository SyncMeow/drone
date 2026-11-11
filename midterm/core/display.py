"""display.py - 視窗縮放: 讓顯示的畫面塞得進小螢幕 (只縮顯示, 錄影 / log 仍是原尺寸)

screen_size()            -> (w, h) 螢幕大小 (點); 問不到就用 DEFAULT_SCREEN
fit_scale(w, h)          -> 讓 w x h 的畫面放得進螢幕 (扣掉選單列 / 標題列 / Dock) 的縮放倍率, 最大 1.0
parse_scale(text)        -> "--scale" 參數: "auto" / None -> None (= 自動), 否則 float
shrink(img, scale)       -> 縮小後的畫面 (scale >= 1 時原樣回傳)

螢幕大小用 tkinter 在子行程裡問 (tkinter 是選用的: 沒裝 / 沒有螢幕 / 逾時都會退回預設值;
放在子行程是為了不讓 Tk 和 OpenCV 的視窗系統在同一個行程裡打架, macOS 上會出問題)。
"""
import os
import subprocess
import sys

import cv2

DEFAULT_SCREEN = (1440, 900)       # 問不到螢幕大小時的假設 (13 吋 MacBook Air 的預設解析度)
MARGIN = (40, 130)                 # 扣掉: 左右邊框 / 選單列 + 視窗標題列 + Dock (px)
_cache = []


def screen_size():
    if _cache:
        return _cache[0]
    size = None
    env = os.environ.get("SIM_SCREEN") or os.environ.get("DRONE_SCREEN")    # 例如 "1440x900", 手動指定
    if env:
        try:
            w, h = env.lower().split("x")
            size = (int(w), int(h))
        except ValueError:
            size = None
    if size is None and (sys.platform == "darwin" or os.environ.get("DISPLAY") or os.name == "nt"):
        code = ("import tkinter\nr = tkinter.Tk()\nr.withdraw()\n"
                "print(r.winfo_screenwidth(), r.winfo_screenheight())\nr.destroy()\n")
        try:
            out = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=4.0).stdout.decode().split()
            w, h = int(out[0]), int(out[1])
            if w >= 320 and h >= 240:
                size = (w, h)
        except Exception:
            size = None
    _cache.append(size or DEFAULT_SCREEN)
    return _cache[0]


def fit_scale(w, h, screen=None):
    sw, sh = screen or screen_size()
    return float(min(1.0, (sw - MARGIN[0]) / float(w), (sh - MARGIN[1]) / float(h)))


def parse_scale(text):
    if text is None or str(text).strip().lower() in ("", "auto"):
        return None
    v = float(text)
    if not 0.1 <= v <= 4.0:
        raise ValueError("--scale must be between 0.1 and 4 (or auto)")
    return v


def shrink(img, scale):
    if img is None or scale is None or abs(scale - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
