"""
midterm.py - 期中賽道 (多段賽道, 每段以 ArUco marker id 1~5 標記)

流程
    1. 手動階段: 用鍵盤起飛、把無人機擺到起跑位置 (前方約 150 cm 要看得到 marker 1)
    2. 確認好之後按 g 開始自動飛行
    3. 第一段 (目前只做到這裡):
         APPROACH : 面向 marker 1 前進, 停在距離 APPROACH_DIST (< 150 cm, 不要撞到)
         SIDE     : 朝右平移 SIDE_TIME 秒 (距離未知, 實測後調整)
         FORWARD  : 直直往前飛, 直到看見 marker 2, 然後停在距離 M2_STOP_DIST (約 115 cm) 原地懸停
         PAUSE_C/LEFT : 停穩後朝左平移 LEFT_TIME 秒
         LEFT     : 停穩後朝左平移 LEFT_TIME 秒 (速度 LEFT_SPEED)
         DONE     : 懸停. (之後的 marker 3, 4, 5 再往下接)

按鍵 (視窗 "drone" 要在最上層)
    1 起飛 / 2 降落 / 3 停止 (hover) / 5 高度 / 6 電量
    w s 前進 後退 | a d 左移 右移 | z x 下降 上升 | c v 順時針 逆時針
    程式一開始是手動模式; 按 g 切換成自動飛行 (只在手動模式有效, 從第一段重新開始)
    自動飛行中只要按任何操作鍵 (w s a d x z c v 1 2 3) => 立刻切回手動 (鍵盤優先權最高),
    要再按 g 才會重新進入自動飛行; 5, 6 只是查詢, 不會中斷自動飛行
    q 或 ESC 結束 (會先停止並降落)
"""
import math
import os
import time

import cv2
import numpy as np
from djitellopy import Tello

from keyboard_djitellopy import keyboard
from pyimagesearch.pid import PID

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 參數
# ----------------------------------------------------------------------------
MARKER_SIZE = 15.0                    # marker 邊長 (cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_250

# ---- 第一段賽道 -------------------------------------------------------------
M1_ID = 1
M2_ID = 2
APPROACH_DIST = 65.0                  # 面向 marker 1 前進後要停的距離 (cm); 必須 < 150
SAFE_MIN_DIST = 55.0                  # 離 marker 小於此值時絕對不再往前 (cm)
ARRIVE_FRAMES = 4                     # 連續幾張 frame 都到位才算抵達 (避免單張雜訊)
CENTER_PX = 25                        # 看到 marker 後, 要先讓它離畫面中心 < 這個像素數才開始往前
RECENTER_PX = 50                      # 前進途中 marker 偏離畫面中心超過這個像素數, 就停下來重新置中
CENTER_CM = 4.0                       # 置中容許誤差 (cm, 換算到 marker 深度的像素數); 離得近時像素死區會自動放大, 避免來回喬
RECENTER_CM = 10.0                    # 前進途中偏離超過這個距離 (cm) 才停下來重新置中
APPROACH_MAX_TIME = 12.0              # 接近 marker 1 最多花這麼久 (s), 超過就直接進入下一個動作 (右移)
NEAR_LOST_EXTRA = 25.0                # 已經靠得很近 (<= APPROACH_DIST + 此值) 時 marker 看不到了, 視為抵達
CENTER_FRAMES = 3                     # 連續幾張 frame 都置中才算 (避免單張雜訊)
LOST_WARN = 3.0                       # 接近 marker 1 時看不到超過這麼久就提示 (s), 期間懸停

SIDE_SPEED = 30                       # 向右平移速度 (rc 值)
SIDE_TIME = 2.0                       # 向右平移時間 (s)   <-- 距離未知, 實測後調整

FWD_SPEED = 20                        # 直直往前飛的速度 (rc 值)
FWD_MAX_TIME = 9.0                    # 往前飛最多幾秒還沒看到 marker 2 就停下來 (s)   <-- 實測後調整
M2_STOP_DIST = 120.0                  # 停在 marker 2 前面的距離 (cm)
M2_TOL = 4.0                          # 距離 <= M2_STOP_DIST + M2_TOL 就停 (cm)

LEFT_SPEED = 30                      # 停在 marker 2 前之後, 向左平移的速度 (rc 值)
LEFT_TIME = 2.0                       # 向左平移時間 (s)
LAND_AT_DONE = False                  # 全部做完 (DONE) 後是否直接降落; False = 原地懸停 (之後要接 marker 3~5)
PAUSE_TIME = 0.8                      # 每個動作之間的懸停穩定時間 (s)

# ---- 保持航向 (平移 / 往前飛時不要歪掉) ----------------------------------------
HOLD_HEADING = True
# 接近 marker 1 時的轉向怎麼處理:
#   "heading": 保持按下 g 當下的航向 (也就是你手動擺好的方向); 只用左右平移來置中,
#              後面右移 / 往前飛的賽道方向就等於你擺放的方向, 不受 marker 角度估計誤差影響
#   "marker" : 依 marker 法向量轉向去 "正對" marker (marker 沒擺正或角度估不準時, 賽道方向會跟著歪)
APPROACH_YAW = "heading"
HEADING_KP = 1.0
HEADING_MAX = 20                      # 修正旋轉速度上限
HEADING_MIN = 8                       # 超出死區時至少給這麼大的旋轉速度 (Tello 對太小的 rc 值幾乎不會轉)
HEADING_DEAD = 3.0                    # degree
HEADING_SIGN = 1                      # Tello 的 yaw 順時針為正; 若發現越修越歪, 乘 -1
HOLD_HEADING_DONE = True              # 停在 marker 2 前 (DONE) 懸停時也保持航向, 抵銷機身自己慢慢轉的漂移
DEBUG_HEADING = True                  # 每秒印一次 yaw / 目標 / 誤差 / 修正指令, 方便檢查 HEADING_SIGN
HEADING_ABORT = 45.0                  # 航向偏離超過此角度就停止修正 (防止方向反了時一直轉)

# ---- PID / 速度 (沿用 lab6 的設定) ----------------------------------------------
GAIN_LR_FB = (0.7, 0.0001, 0.1)
GAIN_CENTER = (1.6, 0.0001, 0.1)      # 置中 (左右 / 上下) 專用: 增益比較大, 置中比較快; 若來回晃就調小
CENTER_MAX_LR = 35                    # 置中時的左右 / 上下速度上限 (置中時 fb = 0, 可以快一點)
CENTER_MAX_UD = 40
GAIN_UD = (1.0, 0.0001, 0.1)
GAIN_YAW = (1.0, 0.0001, 0.1)

MAX_LR = 25
MAX_FB = 25
MAX_UD = 30
MAX_YAW = 25

DEAD_XY = 8.0                         # cm
DEAD_Y = 15.0                         # cm
DEAD_Z = 8.0                          # cm
DEAD_YAW = 6.0                        # degree
MIN_SPEED = 10                        # 誤差在死區外時的最小速度 (Tello 對太小的 rc 幾乎不會動)

SIGN_LR = 1
SIGN_UD = -1
SIGN_FB = 1
SIGN_YAW = -1

# ---- 手動 -----------------------------------------------------------------------
MANUAL_MAX = (25, 25, 50, 40)         # (左右, 前後, 上下, 旋轉)
KEY_HOLD = 0.2
START_KEY = ord('g')

BATTERY_PERIOD = 3.0
CONTROL_PERIOD = 0.05
WINDOW = "drone"

# 狀態
MANUAL, APPROACH, PAUSE_A, SIDE, PAUSE_B, FORWARD, PAUSE_C, LEFT, DONE = (
    "MANUAL", "APPROACH", "PAUSE_A", "SIDE", "PAUSE_B", "FORWARD", "PAUSE_C", "LEFT", "DONE")
AUTO_STATES = (APPROACH, PAUSE_A, SIDE, PAUSE_B, FORWARD, PAUSE_C, LEFT, DONE)


# ----------------------------------------------------------------------------
# 相機 / ArUco
# ----------------------------------------------------------------------------
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


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def marker_pose(corner, obj_pts, K, dist):
    """回傳 (rvec, tvec(3,), distance, yaw_err_deg) 或 None; yaw_err 定義同 lab6 (0 = marker 正對無人機)"""
    img_pts = corner.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    z_prime = R @ np.array([0.0, 0.0, 1.0])
    angle = math.degrees(math.atan2(z_prime[2], z_prime[0]))
    yaw_err = wrap180(angle + 90.0)
    t = tvec.flatten()
    return rvec, t, float(np.linalg.norm(t)), yaw_err


def detect_markers(frame, detect, obj_pts, K, dist):
    """回傳 {id: dict(corner, rvec, t, d, yaw_err)}, 同時把偵測結果畫在 frame 上"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect(gray)
    found = {}
    if ids is None:
        return found
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for c, mid in zip(corners, ids.flatten()):
        pose = marker_pose(c, obj_pts, K, dist)
        if pose is None:
            continue
        rvec, t, d, yaw_err = pose
        found[int(mid)] = dict(corner=c, rvec=rvec, t=t, d=d, yaw_err=yaw_err)
        draw_axes(frame, K, dist, rvec, t, MARKER_SIZE)
        cx, cy = c[0].mean(axis=0).astype(int)
        put(frame, "id %d  dist: %.1f cm" % (mid, d), (cx - 100, cy - 40))
        put(frame, "x: %.0f  y: %.0f  z: %.0f  ang: %.0f" % (t[0], t[1], t[2], yaw_err),
            (cx - 100, cy - 15))
    return found


# ----------------------------------------------------------------------------
# 控制小工具
# ----------------------------------------------------------------------------
def put(frame, text, org, color=(0, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 

def deadband(v, th):
    return 0.0 if abs(v) < th else v


def limit(v, max_speed):
    return int(max(-max_speed, min(max_speed, v)))


def drive(pid, err, sign, max_speed, first=False):
    """單一軸 PID -> 速度: err == 0 (死區內) 輸出 0 並清積分; 否則至少 MIN_SPEED"""
    if first:
        pid.prevError = err
    out = sign * pid.update(err, sleep=0)
    if err == 0:
        pid.cI = 0
        return 0
    if abs(out) < MIN_SPEED:
        out = MIN_SPEED if out >= 0 else -MIN_SPEED
    return limit(out, max_speed)


class LimitedDrone(object):
    """給 keyboard() 用: send_rc_control 會被 MANUAL_MAX 限速, 其餘轉給 drone"""

    def __init__(self, drone, max_speeds):
        self._drone = drone
        self._max = max_speeds

    def send_rc_control(self, lr, fb, ud, yaw):
        vals = (lr, fb, ud, yaw)
        self._drone.send_rc_control(*[limit(v, m) for v, m in zip(vals, self._max)])

    def __getattr__(self, name):
        return getattr(self._drone, name)


NON_MOVE_KEYS = (ord('1'), ord('2'), ord('3'), ord('5'), ord('6'))
# 操作鍵: 自動飛行中按到其中任何一個, 立刻切回手動 (5, 6 只是查詢, 不會中斷自動飛行)
OPERATION_KEYS = tuple(ord(c) for c in "wsadxzcv123")


def read_yaw(drone):
    try:
        return float(drone.get_yaw())
    except Exception:
        return None


# ----------------------------------------------------------------------------
class Course(object):
    """第一段賽道的狀態機. step() 每個控制週期被呼叫一次, 回傳 (lr, fb, ud, yaw)"""

    def __init__(self, drone, K):
        self.drone = drone
        self.K = K
        self.pids = (PID(*GAIN_CENTER), PID(*GAIN_CENTER), PID(*GAIN_LR_FB), PID(*GAIN_YAW))
        self.state = MANUAL
        self.t0 = 0.0
        self.heading_ref = None
        self.heading_ok = True
        self.arrive = 0
        self.centered = False
        self.center_cnt = 0
        self.z_first = True
        self.last_seen = 0.0
        self.warned_lost = False
        self.m2_seen = False
        self.fresh = True
        self.last_z = None
        self.want_land = False
        self.last_dbg = 0.0

    # -- 狀態切換 --
    def enter(self, state, now):
        if state != self.state:
            print("[%.1f] %s -> %s" % (now, self.state, state))
        self.state = state
        self.t0 = now
        self.arrive = 0
        self.fresh = True
        for p in self.pids:
            p.initialize()
        if state == APPROACH:
            self.last_z = None
            self.centered = False
            self.center_cnt = 0
            self.z_first = True
            self.last_seen = now
            self.warned_lost = False
        if state == SIDE:
            if not (APPROACH_YAW == "heading" and self.heading_ref is not None):
                self.heading_ref = read_yaw(self.drone)     # 否則沿用 g 當下記的航向
                self.heading_ok = self.heading_ref is not None
            if not self.heading_ok:
                print("  (讀不到 yaw, 不做航向修正)")
        if state == FORWARD:
            self.m2_seen = False
        if state == DONE and LAND_AT_DONE:
            self.want_land = True

    def start(self, now, markers):
        """依目前看到的 marker id 決定從賽道哪一段開始; 回傳 False = 沒辦法判斷 (呼叫端要回到手動)"""
        if not markers:
            print("沒看到任何 marker, 無法判斷在賽道哪一段, 維持手動")
            return False
        known = dict((i, m) for i, m in markers.items() if i in (M1_ID, M2_ID))
        if not known:
            print("看到 marker %s, 但這個 id 的賽道還沒實作, 維持手動" % sorted(markers))
            return False
        mid = min(known, key=lambda i: known[i]["d"])      # 同時看到多個: 以最近的為準
        if mid == M1_ID:
            print("看到 marker %d (%.0f cm) -> 從第一段開始: 接近 marker %d" % (mid, known[mid]["d"], M1_ID))
            self.enter(APPROACH, now)
        else:
            print("看到 marker %d (%.0f cm) -> 從 marker %d 前面開始" % (mid, known[mid]["d"], M2_ID))
            self.enter(FORWARD, now)
        self.heading_ref = read_yaw(self.drone)             # 記下手動擺好的航向
        self.heading_ok = self.heading_ref is not None
        if APPROACH_YAW == "heading" and not self.heading_ok:
            print("  (讀不到 yaw, 接近 marker 1 時改用 marker 角度轉向)")
        return True

    def abort(self, now, why):
        if self.state != MANUAL:
            print("[%.1f] abort auto (%s)" % (now, why))
        self.state = MANUAL

    # -- 航向保持 --
    def heading_cmd(self):
        if not (HOLD_HEADING and self.heading_ok and self.heading_ref is not None):
            return 0
        yaw = read_yaw(self.drone)
        if yaw is None:
            return 0
        err = wrap180(yaw - self.heading_ref)         # 正 = 目前比目標偏順時針
        if abs(err) > HEADING_ABORT:
            print("  航向偏離 %.0f 度, 停止航向修正 (檢查 HEADING_SIGN)" % err)
            self.heading_ok = False
            return 0
        if abs(err) < HEADING_DEAD:
            cmd = 0
        else:
            raw = -HEADING_SIGN * HEADING_KP * err
            if abs(raw) < HEADING_MIN:
                raw = HEADING_MIN if raw >= 0 else -HEADING_MIN
            cmd = limit(raw, HEADING_MAX)
        if DEBUG_HEADING:
            t = time.time()
            if t - self.last_dbg >= 1.0:
                self.last_dbg = t
                print("  [%s] yaw=%.0f ref=%.0f err=%.0f -> yaw cmd %d" % (self.state, yaw, self.heading_ref, err, cmd))
        return cmd

    # -- 每個週期 --
    def step(self, now, markers, frame_size):
        st = self.state
        first, self.fresh = self.fresh, False
        x_pid, y_pid, z_pid, yaw_pid = self.pids

        if st == APPROACH:
            if now - self.t0 > APPROACH_MAX_TIME:
                print("  接近 marker %d 超過 %.0f 秒, 直接進入下一個動作" % (M1_ID, APPROACH_MAX_TIME))
                self.enter(PAUSE_A, now)
                return (0, 0, 0, 0)
            m = markers.get(M1_ID)
            if m is None:
                if self.last_z is not None and self.last_z <= APPROACH_DIST + NEAR_LOST_EXTRA:
                    print("  靠太近看不到 marker %d (最後距離 %.0f cm), 視為抵達" % (M1_ID, self.last_z))
                    self.enter(PAUSE_A, now)
                    return (0, 0, 0, 0)
                if now - self.last_seen > LOST_WARN and not self.warned_lost:
                    print("  看不到 marker %d, 先懸停等待" % M1_ID)
                    self.warned_lost = True
                self.arrive = 0
                return (0, 0, 0, 0)
            self.last_seen = now
            self.warned_lost = False
            x, y, z = m["t"]
            self.last_z = z
            fw, fh = frame_size
            cx, cy = m["corner"].reshape(4, 2).mean(axis=0)
            du, dv = cx - fw / 2.0, cy - fh / 2.0      # marker 離畫面中心的像素偏移 (右 / 下 為正)
            # 置中容許誤差: 至少 CENTER_PX 像素, 離得越近 (同樣的公分數 = 越多像素) 就放得越寬
            tol = max(CENTER_PX, CENTER_CM * self.K[0, 0] / z)
            rtol = max(RECENTER_PX, RECENTER_CM * self.K[0, 0] / z)
            ex = 0.0 if abs(du) < tol else du * z / self.K[0, 0]
            ey = 0.0 if abs(dv) < tol else dv * z / self.K[1, 1]
            lr = drive(x_pid, ex, SIGN_LR, CENTER_MAX_LR, first)
            ud = drive(y_pid, ey, SIGN_UD, CENTER_MAX_UD, first)
            if APPROACH_YAW == "heading" and self.heading_ok and self.heading_ref is not None:
                yaw = self.heading_cmd()
            else:
                yaw = drive(yaw_pid, deadband(m["yaw_err"], DEAD_YAW), SIGN_YAW, MAX_YAW, first)

            # 距離到了就算抵達 (不再要求一定要置中): 不會因為一直在喬位置而越壓越近
            if z <= APPROACH_DIST + DEAD_Z:
                self.arrive += 1
                if self.arrive >= ARRIVE_FRAMES:
                    self.enter(PAUSE_A, now)
                    return (0, 0, 0, 0)
                return (lr, 0, ud, yaw)
            self.arrive = 0

            # 還沒到距離: 先置中, 再前進 (沒置中以前 fb = 0, 只做左右 / 上下 / 轉向修正)
            in_center = abs(du) < tol and abs(dv) < tol
            if not self.centered:
                self.center_cnt = self.center_cnt + 1 if in_center else 0
                if self.center_cnt >= CENTER_FRAMES:
                    self.centered = True
                    self.z_first = True
                    print("  marker %d 已置中, 開始前進" % M1_ID)
                return (lr, 0, ud, yaw)
            if abs(du) > rtol or abs(dv) > rtol:
                self.centered = False                  # 前進途中偏掉了: 停下來重新置中
                self.center_cnt = 0
                print("  marker %d 偏離中心, 重新置中" % M1_ID)
                return (lr, 0, ud, yaw)

            fb = drive(z_pid, deadband(z - APPROACH_DIST, DEAD_Z), SIGN_FB, MAX_FB, self.z_first)
            self.z_first = False
            if z < SAFE_MIN_DIST:                      # 太近了: 絕不往前
                fb = min(fb, 0)
            return (lr, fb, ud, yaw)

        if st == PAUSE_A:
            if now - self.t0 >= PAUSE_TIME:
                self.enter(SIDE, now)
            return (0, 0, 0, 0)

        if st == SIDE:
            if now - self.t0 >= SIDE_TIME:
                self.enter(PAUSE_B, now)
                return (0, 0, 0, 0)
            return (SIDE_SPEED, 0, 0, self.heading_cmd())

        if st == PAUSE_B:
            if now - self.t0 >= PAUSE_TIME:
                self.enter(FORWARD, now)
            return (0, 0, 0, 0)

        if st == FORWARD:
            m = markers.get(M2_ID)
            if m is not None:
                z = m["t"][2]
                if not self.m2_seen:
                    self.m2_seen = True
                    print("  看到 marker %d, 距離 %.0f cm" % (M2_ID, z))
                err = z - M2_STOP_DIST
                if abs(err) <= M2_TOL:                # 在 M2_STOP_DIST ± M2_TOL 內: 到位
                    print("  到達 marker %d 前 %.0f cm, 停止" % (M2_ID, z))
                    self.enter(PAUSE_C, now)
                    return (0, 0, 0, 0)
                # err > 0: 太遠 -> 往前; err < 0: 太近 -> 往後, 直到回到臨界距離
                fb = drive(z_pid, err, SIGN_FB, FWD_SPEED, first)
                return (0, fb, 0, self.heading_cmd())
            if now - self.t0 >= FWD_MAX_TIME:
                print("  往前飛 %.1f 秒仍沒看到 marker %d, 停止" % (FWD_MAX_TIME, M2_ID))
                self.enter(DONE, now)
                return (0, 0, 0, 0)
            return (0, FWD_SPEED, 0, self.heading_cmd())

        if st == PAUSE_C:
            if now - self.t0 >= PAUSE_TIME:
                self.enter(LEFT, now)
            return (0, 0, 0, self.heading_cmd())

        if st == LEFT:
            if now - self.t0 >= LEFT_TIME:
                self.enter(DONE, now)
                return (0, 0, 0, 0)
            return (-LEFT_SPEED, 0, 0, self.heading_cmd())

        if st == DONE and HOLD_HEADING_DONE:
            return (0, 0, 0, self.heading_cmd())
        return (0, 0, 0, 0)                            # DONE / MANUAL: 懸停


# ----------------------------------------------------------------------------
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

    course = Course(drone, K)
    manual_drone = LimitedDrone(drone, MANUAL_MAX)
    key_expire = 0.0
    start_req = False
    interrupted = False
    battery = None
    last_bat = 0.0
    last_raw = None
    last_ctrl = 0.0

    try:
        while True:
            now = time.time()
            flying = bool(getattr(drone, "is_flying", False))

            # ---- 鍵盤: 最高優先權 ------------------------------------------
            key = cv2.waitKey(1)
            if key != -1:
                key &= 0xFF
            if key in (27, ord('q')):
                break
            if key == START_KEY:
                if course.state not in (MANUAL, DONE):
                    pass                                   # 自動飛行進行中, 忽略
                elif not flying:
                    print("還沒起飛, 先按 1 起飛")
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                    start_req = True                       # 下一張 frame 偵測到 marker 後再決定從哪一段開始
            elif key != -1:
                if key in OPERATION_KEYS:
                    start_req = False
                if key in OPERATION_KEYS and course.state != MANUAL:
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

            if start_req:
                start_req = False
                if not course.start(now, markers):
                    course.abort(now, "no marker")        # 看不到 marker: 回到手動

            # ---- 自動飛行 ----------------------------------------------------
            rc = (0, 0, 0, 0)
            if course.state != MANUAL:
                if not flying:
                    course.abort(now, "not flying")
                else:
                    rc = course.step(now, markers, (frame.shape[1], frame.shape[0]))
                    drone.send_rc_control(*rc)
                    if course.want_land:                       # 第一段完成: 降落, 回到手動模式
                        course.want_land = False
                        print("第一段完成, 降落")
                        drone.send_rc_control(0, 0, 0, 0)
                        try:
                            drone.land()
                        except Exception as e:
                            print("land failed:", e)
                        course.abort(now, "landed")

            put(frame, "%s  %s  rc=%s" % (course.state, "FLYING" if flying else "ON GROUND", rc),
                (10, 25), (0, 255, 0))
            if course.state == MANUAL:
                put(frame, "press g to start", (10, 75), (0, 200, 255))
            if battery is not None:
                put(frame, "Battery: %d%%" % battery, (10, 50),
                    (0, 0, 255) if battery < 20 else (0, 255, 0))
            cv2.imshow(WINDOW, frame)

    except KeyboardInterrupt:                          # Ctrl+C (SIGINT): 立刻降落
        interrupted = True
        print("\nSIGINT: 發送降落指令")
    finally:
        try:
            drone.send_rc_control(0, 0, 0, 0)
            # SIGINT 時不管 is_flying 旗標 (例如起飛到一半被中斷), 一律送降落
            if interrupted or getattr(drone, "is_flying", False):
                drone.land()
        except BaseException as e:                     # 降落途中又按 Ctrl+C 也不要中斷清理
            print("land failed:", e)
        drone.streamoff()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
