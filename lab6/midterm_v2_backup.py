"""
midterm.py - 期中賽道 (EAB06): 第一段 + 第二段

第一段 (marker 1, 2)
    S1_APPROACH : 面向 marker 1 前進, 停在 APPROACH_DIST (不撞到 marker 1)
    S1_RIGHT    : 向右平移避開板子; 移到 marker 2 正前方 (marker 2 的 x ~ 0) 就停
                  (看不到 marker 2 時改用時間 SIDE_TIME 當備案)
    S1_FORWARD  : 直直往前, 停在 marker 2 前 M2_STOP_DIST (太近會往後退)
    S1_LEFT     : 向左平移, 直到看到手持的 marker 0
第二段 (marker 0)
    S2_FOLLOW   : 跟著手持的 marker 0 移動 (轉向對準 + 保持距離), 繞椅子一圈飛過藍線
    S2_DONE     : 看到 marker 3 且 marker 0 已收起來 => 第二段完成, 原地懸停 (第三段之後再接)

每個狀態只看自己的目標 marker id, 同時看到多個 marker 時其他的會被忽略。

左右 / 上下修正:
    不再用 "對準畫面中心像素" 的方式 (增益大 + 最小速度, 會左右快速來回甩)。
    改用 marker 的公分位置 (solvePnP 的 x, y), 先做平滑 (EMA) 再乘小增益, 有死區、速度上限低、
    不強迫最小速度, 動作會慢但穩。

按鍵 (視窗 "drone" 要在最上層, 輸入法要切成英文)
    g   開始自動飛行: 依畫面中最近的 marker id 判斷目前在哪一段, 從那邊開始
        (marker 1 => 第一段開頭, marker 2 => marker 2 前面, marker 0 => 第二段)
        還沒起飛時按 g 會先自動起飛; 看不到 marker 就維持手動
    1 起飛 / 2 降落 / 3 停止
    w s a d z x c v 手動移動 (放開約 KEY_HOLD 秒後自動停止); 5 高度 / 6 電量
    自動飛行中按任何操作鍵 (w s a d x z c v 1 2 3) => 立刻切回手動, 要再按 g 才會回到自動
    q / ESC 結束 (停止並降落); Ctrl+C 也會降落
"""
import math
import os
import time

import cv2
import numpy as np
from djitellopy import Tello

from keyboard_djitellopy import keyboard

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 參數
# ============================================================================
MARKER_SIZE = 15.0                    # marker 邊長 (cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_250

M0_ID, M1_ID, M2_ID, M3_ID = 0, 1, 2, 3

# ---- 第一段 ------------------------------------------------------------------
APPROACH_DIST = 70.0                  # 停在 marker 1 前的距離 (cm); 必須 < 150
SAFE_MIN_DIST = 60.0                  # 離 marker 小於此值時絕不再往前 (cm)
APPROACH_MAX_TIME = 6.0               # 接近 marker 1 最多花這麼久 (s), 超過就直接進入右移
NEAR_LOST_EXTRA = 25.0                # 已靠近 (<= APPROACH_DIST + 此值) 時 marker 1 看不到, 視為抵達

SIDE_SPEED = 30                       # 向右平移速度
SIDE_MODE = "marker2"                 # "marker2": 移到 marker 2 正前方就停; "time": 固定移 SIDE_TIME 秒
SIDE_ALIGN_CM = 5.0                   # marker 2 的 x <= 此值 (cm) 就算已在它正前方
SIDE_MIN_TIME = 1.0                   # 至少右移這麼久 (s), 確保離開板子
SIDE_TIME = 2.0                       # "time" 模式的右移時間; "marker2" 模式下看不到 marker 2 時的備案
SIDE_MAX_TIME = 4.0                   # 右移最長時間 (s), 保險用

FWD_SPEED = 20                        # 還沒看到 marker 2 時往前飛的速度
FWD_MAX_TIME = 9.0                    # 往前飛最多幾秒還沒看到 marker 2 就停 (s)
M2_STOP_DIST = 120.0                  # 停在 marker 2 前的距離 (cm); 圖上寫約 115
M2_TOL = 5.0                          # M2_STOP_DIST ± 此值內算到位 (cm); 比這近會往後退

LEFT_SPEED = 30                       # 向左平移速度
LEFT_MIN_TIME = 0.0                   # 至少左移這麼久才開始找 marker 0 (s); 板子擋不住時可設 0
LEFT_MAX_TIME = 8.0                   # 左移這麼久還沒看到 marker 0 就停下來等 (s)

# ---- 第二段 ------------------------------------------------------------------
FOLLOW_DIST = 100.0                   # 跟 marker 0 保持的距離 (cm)
FOLLOW_DEAD = 10.0                    # 距離死區 (cm)
FOLLOW_FB_MAX = 25                    # 跟隨時前後速度上限
FOLLOW_YAW_KP = 0.8                   # 轉向: marker 0 偏離正前方的角度 (deg) * 此值
FOLLOW_YAW_DEAD = 5.0                 # 轉向死區 (deg)
FOLLOW_YAW_MAX = 30
FOLLOW_YAW_MIN = 8                    # 超出死區時的最小轉向速度 (Tello 對太小的 rc 幾乎不會轉)
HANDOFF_LOST = 0.5                    # marker 0 消失超過此秒數, 且看到 marker 3, 就結束第二段
M3_HANDOFF_DIST = 250.0               # marker 3 要在這個距離內 (cm) 才算 (避免繞椅子時遠遠瞄到就結束)
LAND_AT_DONE = False                  # 第二段結束後是否降落 (False = 懸停, 之後接第三段)

PAUSE_TIME = 0.8                      # 每個動作之間的懸停穩定時間 (s)
ARRIVE_FRAMES = 3                     # 連續幾張 frame 都到位才算抵達

# ---- 平滑的位置修正 (取代原本的畫面中心置中) --------------------------------------
EMA_ALPHA = 0.35                      # 量測值平滑: 越小越平滑但反應越慢
LAT_KP, LAT_DEAD, LAT_MAX = 0.4, 6.0, 15      # 左右: x (cm) -> 速度; 死區 6 cm; 上限 15
UD_KP, UD_DEAD, UD_MAX = 0.3, 15.0, 20        # 上下: y (cm) -> 速度; 死區 15 cm; 上限 20
FB_KP, FB_DEAD, FB_MAX = 0.5, 8.0, 25         # 前後: 距離誤差 (cm) -> 速度; 上限 25
FB_MIN = 10                           # 前後誤差超出死區時的最小速度 (避免停在差一點的地方)

# ---- 航向保持 (直線段不要歪) -------------------------------------------------------
HOLD_HEADING = True
HEADING_KP = 1.0
HEADING_MAX = 20
HEADING_MIN = 8
HEADING_DEAD = 3.0                    # degree
HEADING_SIGN = 1                      # Tello 的 yaw 順時針為正; 若發現越修越歪, 乘 -1
HEADING_ABORT = 45.0                  # 偏離超過此角度就停止修正 (防止符號反了時一直轉)
DEBUG_HEADING = False                 # True: 每秒印一次 yaw / 目標 / 誤差 / 指令

# ---- 手動 -----------------------------------------------------------------------
MANUAL_MAX = (25, 25, 50, 40)         # (左右, 前後, 上下, 旋轉)
KEY_HOLD = 0.2
START_KEY = ord('g')
START_WINDOW = 1.5                    # 按 g 之後最多等幾秒找 marker 來判斷位置 (s)

BATTERY_PERIOD = 3.0
CONTROL_PERIOD = 0.05
WINDOW = "drone"

# 狀態
MANUAL = "MANUAL"
S1_APPROACH, S1_PAUSE1, S1_RIGHT, S1_PAUSE2 = "S1_APPROACH", "S1_PAUSE1", "S1_RIGHT", "S1_PAUSE2"
S1_FORWARD, S1_PAUSE3, S1_LEFT, S1_WAIT0 = "S1_FORWARD", "S1_PAUSE3", "S1_LEFT", "S1_WAIT0"
S2_FOLLOW, S2_DONE = "S2_FOLLOW", "S2_DONE"
PAUSES = {S1_PAUSE1: S1_RIGHT, S1_PAUSE2: S1_FORWARD, S1_PAUSE3: S1_LEFT}

NON_MOVE_KEYS = (ord('1'), ord('2'), ord('3'), ord('5'), ord('6'))
OPERATION_KEYS = tuple(ord(c) for c in "wsadxzcv123")


# ============================================================================
# 相機 / ArUco
# ============================================================================
def load_calib(path=None):
    path = path or os.path.join(HERE, "calibrate.xml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError("cannot read intrinsic/distortion from " + path)
    return K, dist


def make_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    if hasattr(aruco, "ArucoDetector"):                # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda g: detector.detectMarkers(g)
    params = aruco.DetectorParameters_create()         # OpenCV < 4.7
    return lambda g: aruco.detectMarkers(g, dictionary, parameters=params)


def draw_axes(img, K, dist, rvec, tvec, length):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
    else:
        cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)


def put(frame, text, org, color=(0, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect_markers(frame, detect, obj_pts, K, dist):
    """回傳 {id: dict(t=(x, y, z) cm, d=距離 cm)}; x 右正, y 下正, z 前正; 並畫在 frame 上"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect(gray)
    found = {}
    if ids is None:
        return found
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for c, mid in zip(corners, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(4, 2).astype(np.float32), K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t = tvec.flatten()
        mid = int(mid)
        if mid in found and found[mid]["d"] <= float(np.linalg.norm(t)):
            continue                                   # 同 id 出現兩次: 取近的
        found[mid] = dict(t=t, d=float(np.linalg.norm(t)))
        draw_axes(frame, K, dist, rvec, tvec, MARKER_SIZE)
        cx, cy = c[0].mean(axis=0).astype(int)
        put(frame, "id %d  %.0f cm" % (mid, found[mid]["d"]), (cx - 60, cy - 40))
        put(frame, "x %.0f y %.0f z %.0f" % (t[0], t[1], t[2]), (cx - 60, cy - 15))
    return found


# ============================================================================
# 控制小工具
# ============================================================================
def limit(v, max_speed):
    return int(max(-max_speed, min(max_speed, v)))


def p_ctrl(err, kp, dead, cap, min_speed=0):
    """P 控制: 死區內 0; 否則 kp*err, 可設最小速度, 再限速"""
    if abs(err) < dead:
        return 0
    out = kp * err
    if min_speed and abs(out) < min_speed:
        out = min_speed if out > 0 else -min_speed
    return limit(out, cap)


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


class Ema(object):
    """指數移動平均, 用來把 marker 位置量測平滑化 (去掉單張雜訊造成的抖動)"""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.v = None

    def reset(self):
        self.v = None

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.v = x if self.v is None else self.alpha * x + (1 - self.alpha) * self.v
        return self.v


class LimitedDrone(object):
    """給 keyboard() 用: send_rc_control 依 MANUAL_MAX 限速, 其餘轉給 drone"""

    def __init__(self, drone, max_speeds):
        self._drone = drone
        self._max = max_speeds

    def send_rc_control(self, lr, fb, ud, yaw):
        vals = (lr, fb, ud, yaw)
        self._drone.send_rc_control(*[limit(v, m) for v, m in zip(vals, self._max)])

    def __getattr__(self, name):
        return getattr(self._drone, name)


def read_yaw(drone):
    try:
        return float(drone.get_yaw())
    except Exception:
        return None


# ============================================================================
# 賽道狀態機
# ============================================================================
class Course(object):
    def __init__(self, drone):
        self.drone = drone
        self.state = MANUAL
        self.t0 = 0.0
        self.ema = Ema()
        self.ema_id = None
        self.ema_seen = 0.0
        self.arrive = 0
        self.heading_ref = None
        self.heading_ok = False
        self.last_z = None
        self.last_seen = 0.0
        self.m0_lost_since = None
        self.notes = set()                 # 已印過的提示 (避免重複洗版)
        self.want_land = False
        self.last_dbg = 0.0

    # ---------------- 狀態切換 ----------------
    def enter(self, state, now):
        if state != self.state:
            print("[%.1f] %s -> %s" % (now, self.state, state))
        self.state = state
        self.t0 = now
        self.arrive = 0
        self.ema.reset()
        self.ema_id = None
        self.last_z = None
        self.last_seen = now
        self.m0_lost_since = None
        self.notes = set()
        if state == S2_DONE:
            self.set_heading_ref()             # 繞完椅子方向變了: 以現在的方向為準
            print("第二段完成")
            if LAND_AT_DONE:
                self.want_land = True

    def note(self, key, msg):
        if key not in self.notes:
            self.notes.add(key)
            print("  " + msg)

    def set_heading_ref(self):
        self.heading_ref = read_yaw(self.drone)
        self.heading_ok = self.heading_ref is not None
        if not self.heading_ok:
            print("  (讀不到 yaw, 不做航向保持)")

    def start(self, now, markers):
        """依看到的 marker 決定從哪一段開始; 回傳 False = 沒看到可用的 marker"""
        known = dict((i, m) for i, m in markers.items() if i in (M0_ID, M1_ID, M2_ID))
        if not known:
            return False
        mid = min(known, key=lambda i: known[i]["d"])      # 同時看到多個: 以最近的為準
        target = {M1_ID: S1_APPROACH, M2_ID: S1_FORWARD, M0_ID: S2_FOLLOW}[mid]
        print("看到 marker %d (%.0f cm) -> 從 %s 開始" % (mid, known[mid]["d"], target))
        self.set_heading_ref()                             # 直線段的方向 = 按下 g 時的方向
        self.enter(target, now)
        return True

    def abort(self, now, why):
        if self.state != MANUAL:
            print("[%.1f] 切回手動 (%s)" % (now, why))
        self.state = MANUAL

    # ---------------- 量測 ----------------
    def track(self, markers, mid, now):
        """取目標 marker 的平滑位置 (x, y, z); 看不到回傳 None。換目標或中斷超過 0.5 s 會重設平滑"""
        m = markers.get(mid)
        if m is None:
            return None
        if self.ema_id != mid or now - self.ema_seen > 0.5:
            self.ema.reset()
            self.ema_id = mid
        self.ema_seen = now
        self.last_seen = now
        x, y, z = self.ema.update(m["t"])
        self.last_z = z
        return x, y, z

    # ---------------- 航向保持 ----------------
    def heading_cmd(self):
        if not (HOLD_HEADING and self.heading_ok):
            return 0
        yaw = read_yaw(self.drone)
        if yaw is None:
            return 0
        err = wrap180(yaw - self.heading_ref)          # 正 = 偏順時針
        if abs(err) > HEADING_ABORT:
            print("  航向偏離 %.0f 度, 停止航向修正 (檢查 HEADING_SIGN)" % err)
            self.heading_ok = False
            return 0
        cmd = p_ctrl(-HEADING_SIGN * err, HEADING_KP, HEADING_DEAD, HEADING_MAX, HEADING_MIN)
        if DEBUG_HEADING and time.time() - self.last_dbg >= 1.0:
            self.last_dbg = time.time()
            print("  [%s] yaw=%.0f ref=%.0f err=%.0f -> %d" % (self.state, yaw, self.heading_ref, err, cmd))
        return cmd

    @staticmethod
    def lat(x):
        return p_ctrl(x, LAT_KP, LAT_DEAD, LAT_MAX)          # marker 在右 -> 往右

    @staticmethod
    def vert(y):
        return p_ctrl(-y, UD_KP, UD_DEAD, UD_MAX)            # marker 在下 (y>0) -> 往下

    @staticmethod
    def fwd(err, dead=FB_DEAD, cap=FB_MAX):
        return p_ctrl(err, FB_KP, dead, cap, FB_MIN)         # 太遠 -> 往前; 太近 -> 往後

    def arrived(self, cond):
        self.arrive = self.arrive + 1 if cond else 0
        return self.arrive >= ARRIVE_FRAMES

    # ---------------- 每個週期 ----------------
    def step(self, now, markers):
        st = self.state
        el = now - self.t0

        if st in PAUSES:                                   # 動作之間停一下, 保持航向
            if el >= PAUSE_TIME:
                self.enter(PAUSES[st], now)
            return (0, 0, 0, self.heading_cmd())

        # ---------- 第一段: 接近 marker 1 ----------
        if st == S1_APPROACH:
            if el > APPROACH_MAX_TIME:
                print("  接近 marker 1 超過 %.0f 秒, 直接右移" % APPROACH_MAX_TIME)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            last_z = self.last_z
            p = self.track(markers, M1_ID, now)
            if p is None:
                if last_z is not None and last_z <= APPROACH_DIST + NEAR_LOST_EXTRA:
                    print("  靠太近看不到 marker 1 (最後 %.0f cm), 視為抵達" % last_z)
                    self.enter(S1_PAUSE1, now)
                else:
                    self.note("lost1", "看不到 marker 1, 懸停等待")
                return (0, 0, 0, self.heading_cmd())
            x, y, z = p
            if self.arrived(z <= APPROACH_DIST + FB_DEAD):
                print("  到達 marker 1 前 %.0f cm" % z)
                self.enter(S1_PAUSE1, now)
                return (0, 0, 0, 0)
            fb = self.fwd(z - APPROACH_DIST)
            if z < SAFE_MIN_DIST:
                fb = min(fb, 0)
            return (self.lat(x), fb, self.vert(y), self.heading_cmd())

        # ---------- 第一段: 向右平移避開板子 ----------
        if st == S1_RIGHT:
            p = self.track(markers, M2_ID, now)
            done = False
            if SIDE_MODE == "marker2":
                if p is not None:
                    self.note("see2", "右移中看到 marker 2, x = %.0f cm" % p[0])
                    done = el >= SIDE_MIN_TIME and p[0] <= SIDE_ALIGN_CM
                elif el >= SIDE_TIME and "see2" not in self.notes:
                    print("  右移 %.1f 秒沒看到 marker 2, 依時間停止" % SIDE_TIME)
                    done = True
            else:
                done = el >= SIDE_TIME
            if el >= SIDE_MAX_TIME:
                print("  右移達上限 %.1f 秒" % SIDE_MAX_TIME)
                done = True
            if done:
                self.enter(S1_PAUSE2, now)
                return (0, 0, 0, 0)
            return (SIDE_SPEED, 0, 0, self.heading_cmd())

        # ---------- 第一段: 往前到 marker 2 前 ----------
        if st == S1_FORWARD:
            p = self.track(markers, M2_ID, now)
            if p is None:
                if el >= FWD_MAX_TIME:
                    self.note("nofind2", "往前 %.0f 秒仍沒看到 marker 2, 懸停 (可手動調整後再按 g)" % FWD_MAX_TIME)
                    return (0, 0, 0, self.heading_cmd())
                return (0, FWD_SPEED, 0, self.heading_cmd())
            x, y, z = p
            err = z - M2_STOP_DIST
            if self.arrived(abs(err) <= M2_TOL):
                print("  到達 marker 2 前 %.0f cm" % z)
                self.enter(S1_PAUSE3, now)
                return (0, 0, 0, 0)
            # 死區 = 到位範圍, 否則會卡在 '不動但也還沒到位' 的距離
            return (self.lat(x), self.fwd(err, dead=M2_TOL), self.vert(y), self.heading_cmd())

        # ---------- 第一段: 向左平移直到看到 marker 0 ----------
        if st in (S1_LEFT, S1_WAIT0):
            if el >= LEFT_MIN_TIME and M0_ID in markers:
                if self.arrived(True):
                    print("  看到手持的 marker 0 (%.0f cm), 進入第二段" % markers[M0_ID]["d"])
                    self.enter(S2_FOLLOW, now)
                    return (0, 0, 0, 0)
            else:
                self.arrive = 0
            if st == S1_LEFT and el >= LEFT_MAX_TIME:
                print("  左移 %.0f 秒仍沒看到 marker 0, 原地等待" % LEFT_MAX_TIME)
                self.enter(S1_WAIT0, now)
                return (0, 0, 0, 0)
            if st == S1_WAIT0:
                return (0, 0, 0, self.heading_cmd())
            return (-LEFT_SPEED, 0, 0, self.heading_cmd())

        # ---------- 第二段: 跟著手持的 marker 0 ----------
        if st == S2_FOLLOW:
            p = self.track(markers, M0_ID, now)
            if p is not None:
                self.m0_lost_since = None
                x, y, z = p
                bearing = math.degrees(math.atan2(x, z))    # marker 0 偏離正前方的角度, 右為正
                yaw = p_ctrl(bearing, FOLLOW_YAW_KP, FOLLOW_YAW_DEAD, FOLLOW_YAW_MAX, FOLLOW_YAW_MIN)
                fb = p_ctrl(z - FOLLOW_DIST, FB_KP, FOLLOW_DEAD, FOLLOW_FB_MAX, FB_MIN)
                return (0, fb, self.vert(y), yaw)
            # marker 0 看不到: 懸停; 若 marker 3 在附近, 第二段結束
            if self.m0_lost_since is None:
                self.m0_lost_since = now
            m3 = markers.get(M3_ID)
            if m3 is not None and now - self.m0_lost_since >= HANDOFF_LOST:
                if m3["d"] <= M3_HANDOFF_DIST:
                    print("  marker 0 收起來了, 看到 marker 3 (%.0f cm)" % m3["d"])
                    self.enter(S2_DONE, now)
                    return (0, 0, 0, 0)
                self.note("far3", "看到 marker 3 但還太遠 (%.0f cm), 繼續等 marker 0" % m3["d"])
            self.note("lost0", "看不到 marker 0, 懸停等待")
            return (0, 0, 0, 0)

        if st == S2_DONE:
            return (0, 0, 0, self.heading_cmd())

        return (0, 0, 0, 0)


# ============================================================================
def main():
    K, dist = load_calib()
    detect = make_detector()

    half = MARKER_SIZE / 2
    obj_pts = np.array([[-half,  half, 0],
                        [ half,  half, 0],
                        [ half, -half, 0],
                        [-half, -half, 0]], dtype=np.float32)

    drone = Tello()
    drone.connect()
    print("Battery:", drone.get_battery(), "%")
    drone.streamon()
    frame_read = drone.get_frame_read()

    course = Course(drone)
    manual_drone = LimitedDrone(drone, MANUAL_MAX)
    key_expire = 0.0
    start_until = 0.0                      # > 0: 按了 g, 在這個時間前找 marker 判斷位置
    interrupted = False
    battery = None
    last_bat = 0.0
    last_raw = None
    last_ctrl = 0.0

    try:
        while True:
            now = time.time()

            # ---- 鍵盤: 最高優先權 ------------------------------------------
            key = cv2.waitKey(1)
            if key != -1:
                key &= 0xFF
            if key in (27, ord('q')):
                break
            if key == START_KEY:
                if course.state not in (MANUAL, S2_DONE, S1_WAIT0):
                    pass                                   # 自動飛行進行中, 忽略
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                    if not getattr(drone, "is_flying", False):
                        print("還沒起飛: 自動起飛")
                        try:
                            drone.takeoff()
                        except Exception as e:
                            print("takeoff failed:", e)
                    if getattr(drone, "is_flying", False):
                        start_until = time.time() + START_WINDOW
                    now = time.time()
            elif key != -1:
                if key in OPERATION_KEYS:
                    start_until = 0.0
                    if course.state != MANUAL:
                        course.abort(now, "key %c" % key)
                        drone.send_rc_control(0, 0, 0, 0)
                try:
                    keyboard(manual_drone, key)
                except Exception as e:
                    print("command failed (key %d): %s" % (key, e))
                if key in (ord('1'), ord('2'), ord('3')):
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                elif key not in NON_MOVE_KEYS:
                    key_expire = time.time() + KEY_HOLD

            if key_expire and time.time() >= key_expire:
                drone.send_rc_control(0, 0, 0, 0)
                key_expire = 0.0

            # ---- 取影像 (限制控制頻率) ---------------------------------------
            raw = frame_read.frame
            if raw is None or raw is last_raw or now - last_ctrl < CONTROL_PERIOD:
                continue
            last_raw = raw
            last_ctrl = now
            frame = raw.copy()

            if now - last_bat >= BATTERY_PERIOD:
                last_bat = now
                try:
                    battery = int(drone.get_battery())
                except Exception:
                    pass

            markers = detect_markers(frame, detect, obj_pts, K, dist)
            flying = bool(getattr(drone, "is_flying", False))

            # ---- 按了 g: 依看到的 marker 決定從哪一段開始 ------------------------
            if start_until:
                if course.start(now, markers):
                    start_until = 0.0
                elif now > start_until:
                    start_until = 0.0
                    ids = sorted(markers)
                    if ids:
                        print("看到 marker %s, 這些 id 的賽道還沒實作, 維持手動" % ids)
                    else:
                        print("沒看到 marker, 無法判斷在賽道哪一段, 維持手動")
                    course.abort(now, "no marker")

            # ---- 自動飛行 ----------------------------------------------------
            rc = (0, 0, 0, 0)
            if course.state != MANUAL:
                if not flying:
                    course.abort(now, "not flying")
                else:
                    rc = course.step(now, markers)
                    drone.send_rc_control(*rc)
                    if course.want_land:
                        course.want_land = False
                        drone.send_rc_control(0, 0, 0, 0)
                        try:
                            drone.land()
                        except Exception as e:
                            print("land failed:", e)
                        course.abort(now, "landed")

            put(frame, "%s  %s  rc=%s" % (course.state, "FLYING" if flying else "ON GROUND", rc),
                (10, 25), (0, 255, 0))
            if battery is not None:
                put(frame, "Battery: %d%%" % battery, (10, 50),
                    (0, 0, 255) if battery < 20 else (0, 255, 0))
            if course.state == MANUAL:
                put(frame, "press g to start", (10, 75), (0, 200, 255))
            cv2.imshow(WINDOW, frame)

    except KeyboardInterrupt:                          # Ctrl+C (SIGINT): 立刻降落
        interrupted = True
        print("\nSIGINT: 發送降落指令")
    finally:
        try:
            drone.send_rc_control(0, 0, 0, 0)
            if interrupted or getattr(drone, "is_flying", False):
                drone.land()
        except BaseException as e:
            print("land failed:", e)
        drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
