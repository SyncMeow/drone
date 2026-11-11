"""
sim_world.py - main.py (core/) 的離線模擬器 (不用接無人機)

做法:
  * FakeTello 取代 djitellopy.Tello: 只記錄位置 (x, z)、機頭方向、高度;
    每 DT 秒把 send_rc_control 的值當成 cm/s 與 deg/s 累加 (沒有慣性 / 延遲)
  * 虛擬相機: 依無人機目前的位置與方向, 用 calibrate.xml 的相機參數把 SCENE 裡的 ArUco marker
    投影成 960x720 的畫面, 交給 core/ 原本的偵測流程
  * core.app.run(drone, clock, io) 直接傳入假無人機、假時鐘 (跑得比真實時間快) 和 SimIO
    (wait_key 照劇本按鍵並推進模擬, show 不顯示; on_enter 記錄每個狀態第一次進入的時間)

座標 (俯視): x 向右, z 向前 (起點時機頭朝 +z); 高度 H 向上 (cm); 機頭方向 psi 順時針為正 (deg)

限制: rc 值 = cm/s 是假設; 沒有桌子 / 板子 / 牆等實體 (不會撞, 也不會遮擋);
      marker 之間不會互相遮擋 (後畫的蓋住先畫的); 畫面沒有模糊與光線變化;
      yaw 方向定義與程式的假設相同, 所以抓不到 HEADING_SIGN / SIGN_ALIGN_YAW 反了的問題
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # lab6/
from core import app                                                             # noqa: E402
from core.params import P                                                        # noqa: E402
from core.vision import load_calib                                               # noqa: E402

DT = 0.06                                   # 每次 waitKey 前進的模擬時間 (s)


class Clock(object):
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, x):
        pass


clock = Clock()

K, DIST = load_calib()
W_IMG, H_IMG = 960, 720
_h = P.MARKER_SIZE / 2
OBJ = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], np.float32)
_DICT = cv2.aruco.getPredefinedDictionary(P.ARUCO_DICT)
_IMG = {}
SRC = np.array([[40, 40], [240, 40], [240, 240], [40, 240]], np.float32)


def marker_img(mid):
    if mid not in _IMG:
        if hasattr(cv2.aruco, "generateImageMarker"):          # OpenCV >= 4.7
            m = cv2.aruco.generateImageMarker(_DICT, mid, 200)
        else:                                                 # OpenCV < 4.7
            m = cv2.aruco.drawMarker(_DICT, mid, 200)
        _IMG[mid] = cv2.copyMakeBorder(m, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return _IMG[mid]


# 記錄每個狀態第一次進入的時間 (給劇本與分析用)
STATE_T = {}

# 場景: SCENE[id] = function(t) -> None (看不到) 或 (x, z, mode, 高度, 角度)
#   mode "wall": 固定朝向; 角度 0 = 面朝起點方向 (-z), 90 = 面朝 -x; "face": 永遠面向無人機 (手持)
SCENE = {}
NOISE = 0.0                                 # marker 角點的像素雜訊 (標準差)


class FakeTello(object):
    last = None
    DRIFT = 0.0                             # 機頭自己漂移的速度 (deg/s)
    TAKEOFF_ALT = 80.0

    def __init__(self):
        self.is_flying = False
        self.P = np.array(START, float)
        self.H = 0.0
        self.psi = START_PSI
        self.cmd = (0, 0, 0, 0)
        self.rc = []
        self.trace = []                     # (t, P, psi, cmd, H)
        FakeTello.last = self

    # --- djitellopy 介面 ---
    def connect(self): pass
    def streamon(self): pass
    def streamoff(self): pass
    def get_battery(self): return 80
    def get_yaw(self): return ((self.psi + 180) % 360) - 180
    def get_distance_tof(self): return self.H
    def get_height(self): return self.H
    def rotate_clockwise(self, d): self.psi += d
    def takeoff(self): self.is_flying = True; self.H = self.TAKEOFF_ALT
    def land(self): self.is_flying = False; print("LAND")
    def send_rc_control(self, lr, fb, ud, yaw):
        self.cmd = (lr, fb, ud, yaw)
        self.rc.append(self.cmd)
    def get_frame_read(self):
        drone = self
        class FrameRead(object):
            @property
            def frame(self):
                return drone.render()
        return FrameRead()
    def __getattr__(self, name):            # 其他沒模擬的指令 (例如 keyboard 裡的 get_height) 回傳 0
        return lambda *a, **k: 0

    # --- 物理 ---
    def axes(self):
        p = math.radians(self.psi)
        f = np.array([math.sin(p), math.cos(p)])           # 機頭方向
        r = np.array([math.cos(p), -math.sin(p)])          # 右手邊
        return f, r

    def step(self, dt):
        if not self.is_flying:
            return
        lr, fb, ud, yaw = self.cmd
        f, r = self.axes()
        self.P = self.P + (lr * r + fb * f) * dt
        self.H = max(10.0, self.H + ud * dt)
        self.psi += (yaw + self.DRIFT) * dt
        self.trace.append((clock.t, self.P.copy(), self.psi, self.cmd, self.H))

    def render(self):
        img = np.full((H_IMG, W_IMG), 128, np.uint8)
        f, r = self.axes()
        p = math.radians(self.psi)
        for mid, fn in SCENE.items():
            spec = fn(clock.t)
            if spec is None:
                continue
            mx, mz, mode = spec[:3]
            mh = spec[3] if len(spec) > 3 else 80.0
            mang = spec[4] if len(spec) > 4 else 0.0
            d = np.array([mx, mz]) - self.P
            xc, zc = d @ r, d @ f
            if zc < 20:
                continue
            th = (-p + math.radians(mang)) if mode == "wall" else math.atan2(xc, zc)
            Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
            rv, _ = cv2.Rodrigues(Ry @ np.diag([1.0, -1.0, -1.0]))
            pts, _ = cv2.projectPoints(OBJ, rv, np.array([xc, self.H - mh, zc]), K, DIST)
            pts = pts.reshape(4, 2).astype(np.float32)
            if NOISE:
                pts += np.random.normal(0, NOISE, pts.shape).astype(np.float32)
            if pts[:, 0].min() < 0 or pts[:, 0].max() > W_IMG or pts[:, 1].min() < 0 or pts[:, 1].max() > H_IMG:
                continue
            M = cv2.getPerspectiveTransform(SRC, pts)
            w = cv2.warpPerspective(marker_img(mid), M, (W_IMG, H_IMG), borderValue=0)
            mask = cv2.warpPerspective(np.full((280, 280), 255, np.uint8), M, (W_IMG, H_IMG), borderValue=0)
            img = np.where(mask > 0, w, img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


START, START_PSI = (0.0, 0.0), 0.0


class SimIO(object):
    """core.app.run 的 io: 每次 wait_key 推進模擬 DT 秒並回傳劇本的按鍵; 不開視窗"""

    def __init__(self, script, max_iter):
        self.script = script
        self.max_iter = max_iter
        self.i = 0

    def wait_key(self, ms):
        i = self.i
        self.i += 1
        clock.t += DT
        if FakeTello.last is not None:
            FakeTello.last.step(DT)
        return self.script(i) if i < self.max_iter else ord('q')

    def show(self, image):
        pass

    def close(self):
        pass

    def on_enter(self, state, now):
        STATE_T.setdefault(state, now)


def run(script, max_iter, start=(0.0, 0.0), start_psi=0.0):
    """script(i) -> 第 i 次 waitKey 要回傳的按鍵 (-1 = 沒按); 跑完 max_iter 次後自動按 q"""
    global START, START_PSI
    START, START_PSI = start, start_psi
    STATE_T.clear()
    drone = FakeTello()
    app.run(drone, clock=clock, io=SimIO(script, max_iter), **LOG_OPTS)
    return drone


# log 資料夾 (logs/..._sim) + 錄影 + HUD: 預設關 (劇本跑得快、不產生一堆資料夾); SIM_LOG=1 python run_seg345.py 打開
# 開或關, 終端機輸出都一樣
LOG_OPTS = dict(log=False, record=False, hud=False, tag="sim")
if os.environ.get("SIM_LOG", "") not in ("", "0"):
    LOG_OPTS = dict(log=True, record=True, hud=True, tag="sim")


def states_line():
    return " -> ".join("%s@%.1f" % (k, v - 1000) for k, v in sorted(STATE_T.items(), key=lambda kv: kv[1]))


def pose_at(drone, state):
    t0 = STATE_T.get(state)
    if t0 is None:
        return None
    for (tt, P, psi, c, H) in drone.trace:
        if tt >= t0:
            return P, psi, H
    return None
