"""
    1 起飛 / 2 降落 / 3 停止 (hover)
    w s 前進 後退 | a d 左移 右移 | z x 下降 上升 | c v 順時針 逆時針
    5 顯示高度 / 6 顯示電量                      
    t 開/關自動追蹤
    q 或 ESC 結束
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
# 參數 (實飛時主要調這裡)
# ----------------------------------------------------------------------------
MARKER_SIZE = 15.0                    # marker 邊長 (cm), 與 lab5 相同
ARUCO_DICT = cv2.aruco.DICT_4X4_250
TARGET_ID = None                      # 只追蹤特定 id; None = 追蹤畫面中第一個 marker

TARGET_DIST = 80.0                   # 想保持的距離 (cm), 投影片範例為 100

# PID 增益 (投影片: kP, kI, kD 可自己調整, 但別設太大), 格式 (kP, kI, kD)
GAIN_LR_FB = (0.7, 0.0001, 0.1)       # 左右 / 前後
GAIN_UD = (1.0, 0.0001, 0.1)          # 上升 / 下降
GAIN_YAW = (1.0, 0.0001, 0.1)         # 旋轉

# 速度上限, 務必限制以防失控 (rc 範圍 -100 ~ 100)
MAX_LR = 40
MAX_FB = 40
MAX_UD = 50
MAX_YAW = 40

# 手動 (鍵盤) 速度上限, 順序 (左右, 前後, 上下, 旋轉); keyboard_djitellopy 裡寫死 40~50, 這裡再壓低
MANUAL_MAX = (25, 25, 50, 40)
# 鍵盤指令只維持這麼久 (s): cv2 只收得到 "按下", 收不到 "放開",
# 所以超過這段時間沒有再收到同一個按鍵 (按住時系統會連續重送) 就自動停止
KEY_HOLD = 0.2

# 死區: 誤差小於此值就當作 0, 避免在目標附近抖動
DEAD_PX = 15                          # pixel, marker 離畫面中心小於這個距離就算置中 (左右 / 上下)
DEAD_Z = 8.0                          # cm, 前後
MIN_SPEED = 10                        # 誤差在死區外時的最小速度; Tello 對太小的 rc 值幾乎不會動, 導致停在偏離中心的位置
DEAD_YAW = 6.0                        # degree

# 方向符號: 如果實飛時發現某個方向反了, 把對應的值乘 -1 即可
SIGN_LR = 1                           # marker 在右邊 -> 往右飛
SIGN_UD = -1                          # 相機 y 軸朝下, marker 在下方 -> 往下飛 (ud 為負)
SIGN_FB = 1                           # marker 太遠 -> 往前飛
SIGN_YAW = -1                         # marker 面朝畫面右側 -> 逆時針轉 (同時往右移)

# 左右 / 上下是依 marker 中心與 "畫面中心" 的像素差來修正, 讓 marker 維持在畫面中央。

BATTERY_PERIOD = 3.0                  # 電量更新間隔 (s)

MANUAL_HOLD = 1.5                     # 按鍵後, 自動追蹤暫停的秒數
CONTROL_PERIOD = 0.05                 # 控制迴圈最短間隔 (s); 小於影像更新週期以免重複算同一張 frame

WINDOW = "drone"


# ----------------------------------------------------------------------------
# 相機校正 / ArUco (沿用 lab5)
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
    """把角度折回 [-180, 180)"""
    return (deg + 180.0) % 360.0 - 180.0


def marker_pose(corner, obj_pts, K, dist):
    """
    回傳 (rvec, tvec(3,), distance, yaw_err_deg), 失敗回傳 None

    distance : 相機到 marker 中心的直線距離 (cm)
    yaw_err  : marker 法向量 Z' 投影到 XZ 平面後, 與 "正對相機" (Z' = (0, 0, -1)) 的夾角
               = 0 代表 marker 正對無人機; 正值代表 marker 面朝畫面右側
    """
    img_pts = corner.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)                 # 旋轉向量 -> 旋轉矩陣
    z_prime = R @ np.array([0.0, 0.0, 1.0])    # marker 的 Z 軸在相機座標下的方向 Z'
    # Z' 投影到 XZ 平面 -> V = (Z'x, Z'z); 夾角用 atan2(z, x)
    angle = math.degrees(math.atan2(z_prime[2], z_prime[0]))
    yaw_err = wrap180(angle + 90.0)            # 正對時 V 指向 -Z (= -90 度), 所以補 90
    t = tvec.flatten()
    return rvec, t, float(np.linalg.norm(t)), yaw_err


def pick_marker(corners, ids):
    """依 TARGET_ID 選一個 marker, 回傳 (corner, id) 或 None"""
    if ids is None:
        return None
    for c, mid in zip(corners, ids.flatten()):
        if TARGET_ID is None or int(mid) == TARGET_ID:
            return c, int(mid)
    return None


# ----------------------------------------------------------------------------
# PID -> 速度指令
# ----------------------------------------------------------------------------
def deadband(v, th):
    return 0.0 if abs(v) < th else v


def limit(v, max_speed):
    """限制最高速度, 並確保送給無人機的是 int"""
    return int(max(-max_speed, min(max_speed, v)))


def drive(pid, err, sign, max_speed, first):
    """單一軸: PID -> 速度。err == 0 (在死區內) 直接輸出 0 並清掉積分; 否則至少給 MIN_SPEED"""
    if first:
        pid.prevError = err
    out = sign * pid.update(err, sleep=0)
    if err == 0:
        pid.cI = 0
        return 0
    if abs(out) < MIN_SPEED:
        out = MIN_SPEED if out >= 0 else -MIN_SPEED
    return limit(out, max_speed)


def compute_rc(pids, x, y, z, yaw_err, first=False):
    """把 marker 的誤差經過 PID 轉成 (left_right, forward_back, up_down, yaw)

    first=True 代表剛抓到 marker 的第一筆資料: 把 prevError 設成目前誤差,
    避免第一次微分項 (誤差變化 / 極小的 dt) 爆衝。
    """
    x_pid, y_pid, z_pid, yaw_pid = pids
    # x, y 已在呼叫端用像素死區處理過 (0 = 已置中)
    ex = x
    ey = y
    ez = deadband(z - TARGET_DIST, DEAD_Z)
    eyaw = deadband(yaw_err, DEAD_YAW)

    lr = drive(x_pid, ex, SIGN_LR, MAX_LR, first)
    ud = drive(y_pid, ey, SIGN_UD, MAX_UD, first)
    fb = drive(z_pid, ez, SIGN_FB, MAX_FB, first)
    yaw = drive(yaw_pid, eyaw, SIGN_YAW, MAX_YAW, first)
    return lr, fb, ud, yaw


class LimitedDrone(object):
    """包住 Tello 給 keyboard() 用: 其餘呼叫照舊轉給 drone, 只有 send_rc_control 會被限速"""

    def __init__(self, drone, max_speeds):
        self._drone = drone
        self._max = max_speeds            # (左右, 前後, 上下, 旋轉)

    def send_rc_control(self, lr, fb, ud, yaw):
        vals = (lr, fb, ud, yaw)
        self._drone.send_rc_control(*[limit(v, m) for v, m in zip(vals, self._max)])

    def __getattr__(self, name):
        return getattr(self._drone, name)


# 這些按鍵不是移動指令 (起飛 / 降落 / 停止 / 查詢), 不需要計時停止
NON_MOVE_KEYS = (ord('1'), ord('2'), ord('3'), ord('5'), ord('6'))


def put(frame, text, org, color=(0, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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

    pids = (PID(*GAIN_LR_FB),             # x   (左右)
            PID(*GAIN_UD),                # y   (上下)
            PID(*GAIN_LR_FB),             # z   (前後)
            PID(*GAIN_YAW))               # yaw (轉向)
    for p in pids:
        p.initialize()

    auto_enabled = True
    tracking = False                      # 上一輪是否有在追蹤 (用來重設 PID)
    manual_until = 0.0
    key_expire = 0.0
    battery = None
    last_bat = 0.0
    manual_drone = LimitedDrone(drone, MANUAL_MAX)
    last_raw = None
    last_ctrl = 0.0
    last_dist = None

    try:
        while True:
            # ---- 鍵盤: 最高優先權 ------------------------------------------
            key = cv2.waitKey(1)
            if key != -1:
                key &= 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('t'):
                auto_enabled = not auto_enabled
                drone.send_rc_control(0, 0, 0, 0)
                print("auto tracking:", "ON" if auto_enabled else "OFF")
            elif key != -1:
                try:
                    keyboard(manual_drone, key)
                except Exception as e:      # 例如起飛失敗: 印出原因, 不要讓整個程式結束 (結束會自動降落)
                    print("command failed (key %d): %s" % (key, e))
                if key in (ord('1'), ord('2'), ord('3')):
                    drone.send_rc_control(0, 0, 0, 0)   # 起飛 / 降落後也歸零, 不殘留方向
                    key_expire = 0.0
                elif key not in NON_MOVE_KEYS:
                    key_expire = time.time() + KEY_HOLD  # 移動鍵: 一小段時間後自動停
                manual_until = time.time() + MANUAL_HOLD

            # 移動鍵超過 KEY_HOLD 沒再收到 -> 視為放開, 停止
            if key_expire and time.time() >= key_expire:
                drone.send_rc_control(0, 0, 0, 0)
                key_expire = 0.0

            # ---- 取影像 (限制控制頻率) ---------------------------------------
            now = time.time()
            raw = frame_read.frame
            if raw is None or raw is last_raw or now - last_ctrl < CONTROL_PERIOD:
                continue
            last_raw = raw
            last_ctrl = now
            frame = raw.copy()

            if now - last_bat >= BATTERY_PERIOD:     # 定時更新電量
                last_bat = now
                try:
                    battery = int(drone.get_battery())
                except Exception:
                    pass

            flying = bool(getattr(drone, "is_flying", False))
            manual_active = now < manual_until
            if manual_active:
                tracking = False          # 手動結束後重新累積 PID

            # ---- 偵測 marker / 估計位姿 -------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)
            picked = pick_marker(corners, ids)
            pose = None
            if picked is not None:
                corner, mid = picked
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                pose = marker_pose(corner, obj_pts, K, dist)

            rc = (0, 0, 0, 0)
            if pose is not None:
                rvec, (x, y, z), d, yaw_err = pose
                last_dist = d
                draw_axes(frame, K, dist, rvec, np.array([x, y, z]), MARKER_SIZE)
                cx, cy = corner[0].mean(axis=0).astype(int)
                # 以 "畫面中心" 為基準的偏移 (cm, 位於 marker 深度): 右 / 下 為正
                fh, fw = frame.shape[:2]
                du = corner[0][:, 0].mean() - fw / 2.0     # 像素偏移 (右 / 下 為正)
                dv = corner[0][:, 1].mean() - fh / 2.0
                x_c = 0.0 if abs(du) < DEAD_PX else du * z / K[0, 0]
                y_c = 0.0 if abs(dv) < DEAD_PX else dv * z / K[1, 1]
                put(frame, "x: %.1f  y: %.1f  z: %.1f" % (x, y, z), (cx - 150, cy - 65))
                put(frame, "id: %d  dist: %.1f cm" % (mid, d), (cx - 150, cy - 40))
                put(frame, "angle: %.1f deg" % yaw_err, (cx - 150, cy - 15))

                if auto_enabled and flying and not manual_active:
                    first = not tracking
                    if first:             # 剛抓到 marker: 重設 PID, 避免舊的積分/微分值
                        for p in pids:
                            p.initialize()
                        tracking = True
                    rc = compute_rc(pids, x_c, y_c, z, yaw_err, first)
            else:
                tracking = False

            # ---- 自動追蹤指令 (手動期間 / 沒起飛 / 關閉自動 時不送) ------------
            if auto_enabled and flying and not manual_active:
                drone.send_rc_control(*rc)      # 沒看到 marker 時 rc = 0 -> 懸停

            # 畫面中心十字 (marker 要維持在這裡)
            fh, fw = frame.shape[:2]
            cv2.drawMarker(frame, (fw // 2, fh // 2), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

            mode = "MANUAL" if manual_active else ("AUTO" if auto_enabled else "AUTO OFF")
            put(frame, "%s  %s  rc=%s" % (mode, "FLYING" if flying else "ON GROUND", rc),
                (10, 25), (0, 255, 0))
            if battery is not None:
                put(frame, "Battery: %d%%" % battery, (10, 50),
                    (0, 0, 255) if battery < 20 else (0, 255, 0))
            cv2.imshow(WINDOW, frame)

    finally:
        # 結束: 先停下來, 再降落
        try:
            drone.send_rc_control(0, 0, 0, 0)
            if getattr(drone, "is_flying", False):
                drone.land()
        except Exception as e:
            print("land failed:", e)
        drone.streamoff()
        cv2.destroyAllWindows()
        if last_dist is not None:
            print("last measured distance to marker: %.1f cm" % last_dist)


if __name__ == '__main__':
    main()
