"""app.py - 主迴圈: 鍵盤 / keepalive / 狀態監控 / 取影像 / 自動控制 / 顯示

run(drone, clock, io): 無人機、時鐘、視窗都由呼叫的人傳入 -> 實機 (main.py) 和模擬器跑的是同一份迴圈
  drone : djitellopy.Tello 或相同介面的假無人機
  clock : 有 time() / sleep() (預設 time 模組)
  io    : 有 wait_key(ms) -> int, show(image), close() (預設 CvIO: cv2.waitKey / cv2.imshow / cv2.destroyAllWindows);
          可以另外有 on_enter(state, now): 每次狀態切換時呼叫
log / record / hud_video / hud: flight log 資料夾、錄影、HUD 視窗 (core/blackbox.py); tag: "real" / "sim"
          (None = 自動: 不是 djitellopy 的無人機就是 sim)。這些功能出錯只會自己關掉, 不會影響控制
"""
import time
import traceback

import cv2

from keyboard_djitellopy import keyboard

from core.blackbox import Blackbox
from core.console import log
from core.control import LimitedDrone, scale_rc
from core.machine import Course
from core.params import P, apply_speed_scale
from core.states import MANUAL, S2_FOLLOW, S3_ALIGN, WAIT_STATES, NON_MOVE_KEYS, OPERATION_KEYS
from core.telemetry import read_battery, read_temp, state_field
from core.vision import detect_markers, load_calib, make_detector, marker_obj_points, put


class CvIO(object):
    """預設的視窗 / 鍵盤: OpenCV"""

    def __init__(self, window=None):
        self.window = P.WINDOW if window is None else window

    def wait_key(self, ms):
        return cv2.waitKey(ms)

    def show(self, image):
        cv2.imshow(self.window, image)

    def close(self):
        cv2.destroyAllWindows()


def _auto_tag(drone):
    return "real" if type(drone).__module__.split(".")[0] == "djitellopy" else "sim"


def run(drone, clock=time, io=None, log=True, record=True, hud_video=True, hud=True, tag=None, log_root=None):
    io = CvIO() if io is None else io
    bb = Blackbox(clock, tag or _auto_tag(drone), log=log, record=record, hud_video=hud_video, hud=hud,
                  log_root=log_root)
    run.blackbox = bb                                   # 給呼叫的人看 log 資料夾 / HUD 耗時
    try:
        _run(drone, clock, io, bb)
    except BaseException as e:
        try:                                            # 例外 (程式錯誤) 的 traceback 也寫進 flight.log
            if not isinstance(e, KeyboardInterrupt):
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
                for line in tb.split("\n"):
                    bb.flog.message(line, "exc")
            bb.event("exception", error=repr(e))
        except Exception:
            pass
        raise
    finally:
        try:
            bb.close(clock.time())
        except Exception:
            pass


def _run(drone, clock, io, bb):
    K, dist = load_calib()
    detect = make_detector()

    obj_pts = marker_obj_points()

    drone.connect()
    print("Battery:", drone.get_battery(), "%")
    drone.streamon()
    frame_read = drone.get_frame_read()

    apply_speed_scale()
    if (P.SPEED_SCALE_MOVE, P.SPEED_SCALE_UPDOWN, P.SPEED_SCALE_TURN) != (1.0, 1.0, 1.0):
        print("Speed scale: move x%.2f, up/down x%.2f, turn x%.2f" % (P.SPEED_SCALE_MOVE, P.SPEED_SCALE_UPDOWN, P.SPEED_SCALE_TURN))
    last_sent = [clock.time()]
    _raw_send = drone.send_rc_control

    def _tracked_send(*a):
        last_sent[0] = clock.time()
        _raw_send(*a)
        bb.sent(a)                                     # log: 最後送出的 rc

    drone.send_rc_control = _tracked_send                  # 記錄最後一次送指令的時間 (給 keepalive 用)
    course = Course(drone, clock, bb.wrap_on_enter(getattr(io, "on_enter", None)))
    bb.attach(course, drone)
    bb.snapshot()
    manual_drone = LimitedDrone(drone, P.MANUAL_MAX)
    key_expire = 0.0
    interrupted = False
    battery = None
    last_bat = 0.0
    temp = None                                       # (templ, temph) °C
    last_mon = last_log = 0.0
    ground_since = None                               # 開始讀到高度 0 的時間 (偵測自己降落)
    self_landed = temp_warned = False
    max_temp, min_bat = None, None
    last_raw = None
    seen_raw = None
    n_new = n_done = 0                                # 畫面張數統計 (算 fps)
    fps_t = clock.time()
    last_ctrl = 0.0

    last_loop = clock.time()
    try:
        while True:
            now = clock.time()
            bb.loop(now)
            if now - last_loop > P.LOOP_WARN:
                print("  [main loop stalled %.1f s: keys were ignored meanwhile]" % (now - last_loop))
                bb.event("warn", what="loop stall", seconds=round(now - last_loop, 2))
            last_loop = now

            # ---- 保持連線: 太久沒送指令就補一個懸停, 避免 Tello 15 秒自動降落 ----
            if getattr(drone, "is_flying", False) and not key_expire and now - last_sent[0] >= P.KEEPALIVE:
                drone.send_rc_control(0, 0, 0, 0)
                bb.keepalive(now)

            # ---- 鍵盤: 最高優先權 ------------------------------------------
            key = io.wait_key(1)
            if key != -1:
                key &= 0xFF
                bb.on_key(key, now)
            if key in (27, ord('q')):
                break
            if key in P.START_KEYS:
                if course.state == S2_FOLLOW and key == ord('j'):
                    drone.send_rc_control(0, 0, 0, 0)      # 第二段跟隨中按 j: 切到第三段
                    course.begin(S3_ALIGN, clock.time())
                elif course.state != MANUAL and course.state not in WAIT_STATES:
                    pass                                   # 自動飛行進行中, 忽略
                else:
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                    if not getattr(drone, "is_flying", False):
                        print("not flying: taking off")
                        try:
                            drone.takeoff()
                            bb.event("takeoff", ok=True)
                        except Exception as e:
                            print("takeoff failed:", e)
                            bb.event("takeoff", ok=False, error=repr(e))
                    now = clock.time()
                    if getattr(drone, "is_flying", False):
                        course.begin(P.START_KEYS[key], now)
                    else:
                        print("takeoff failed, staying in manual")
            elif key != -1:
                if key in OPERATION_KEYS:
                    if course.state != MANUAL:
                        course.abort(now, "key %c" % key)
                        drone.send_rc_control(0, 0, 0, 0)
                try:
                    keyboard(manual_drone, key)
                    if key in (ord('1'), ord('2')):
                        bb.event("takeoff" if key == ord('1') else "land", ok=True, source="key")
                except Exception as e:
                    print("command failed (key %d): %s" % (key, e))
                    bb.event("command_failed", key=key, error=repr(e))
                if key in (ord('1'), ord('2'), ord('3')):
                    drone.send_rc_control(0, 0, 0, 0)
                    key_expire = 0.0
                elif key not in NON_MOVE_KEYS:
                    key_expire = clock.time() + P.KEY_HOLD

            if key_expire and clock.time() >= key_expire:
                drone.send_rc_control(0, 0, 0, 0)
                key_expire = 0.0

            # ---- 狀態監控: 溫度 / 電量 / 自己降落 (影像卡住時也照常跑) ----------
            if now - last_mon >= P.MONITOR_PERIOD:
                last_mon = now
                t = read_temp(drone)
                if t is not None:
                    temp = t
                    max_temp = t[1] if max_temp is None else max(max_temp, t[1])
                b = read_battery(drone)
                if b is not None:
                    battery = b
                    min_bat = b if min_bat is None else min(min_bat, b)
                h = state_field(drone, "h")
                tof = state_field(drone, "tof")
                in_air = bool(getattr(drone, "is_flying", False))
                desc = "battery %s%%  temp %s  height h=%s tof=%s  %s  %s" % (
                    "?" if battery is None else battery,
                    "?" if temp is None else "%.0f~%.0f°C" % temp,
                    "?" if h is None else "%.0f" % h, "?" if tof is None else "%.0f" % tof,
                    course.state, "FLYING" if in_air else "ON GROUND")
                if now - last_log >= P.STATUS_LOG_PERIOD:
                    last_log = now
                    span = max(now - fps_t, 1e-6)
                    desc += "  video %.1f fps (processed %.1f fps)" % (n_new / span, n_done / span)
                    n_new = n_done = 0
                    fps_t = now
                    log("[status] " + desc)
                if temp is not None and temp[1] >= P.TEMP_WARN and not temp_warned:
                    temp_warned = True
                    log("[WARN] temperature high! " + desc)
                    bb.event("warn", what="overheat", templ=temp[0], temph=temp[1])
                elif temp is not None and temp[1] < P.TEMP_WARN - 3:
                    temp_warned = False
                # 程式以為還在飛 (沒送過 land), 但高度一直是 0 → 多半是 Tello 自己降落 (過熱 / 沒電 / 保護)
                if in_air and h is not None and h <= 0 and (tof is None or tof < 20):
                    if ground_since is None:
                        ground_since = now
                    if not self_landed and now - ground_since >= P.SELF_LAND_TIME:
                        self_landed = True
                        log("[!!!] drone may have landed by itself (no land sent): " + desc)
                        bb.event("warn", what="self landing", battery=battery, temp=temp, h=h, tof=tof)
                else:
                    ground_since = None
                    self_landed = False

            # ---- 取影像 (限制控制頻率) ---------------------------------------
            raw = frame_read.frame
            if raw is not None and raw is not seen_raw:    # 統計: 無人機傳來幾張新畫面 (fps)
                seen_raw = raw
                n_new += 1
                bb.new_frame(now)
            if raw is None or raw is last_raw or now - last_ctrl < P.CONTROL_PERIOD:
                # 沒有新畫面: 控制不跑 (和以前一樣); HUD 每 0.1 s 用最後一張畫面更新 (畫面卡住時也看得到狀態)
                canvas = bb.idle(now, bool(getattr(drone, "is_flying", False)), battery, temp)
                if canvas is not None:
                    io.show(canvas)
                continue
            last_raw = raw
            last_ctrl = now
            n_done += 1
            # djitellopy 2.5 回傳的是 RGB, OpenCV 用 BGR: 不轉的話紅藍對調, 畫面看起來像負片
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if P.FRAME_IS_RGB else raw.copy()

            if now - last_bat >= P.BATTERY_PERIOD:
                last_bat = now
                b = read_battery(drone)
                if b is not None:
                    battery = b

            markers = detect_markers(frame, detect, obj_pts, K, dist)
            flying = bool(getattr(drone, "is_flying", False))

            # ---- 自動飛行 ----------------------------------------------------
            rc = (0, 0, 0, 0)
            if course.state != MANUAL:
                if not flying:
                    course.abort(now, "not flying")
                else:
                    rc = scale_rc(*course.step(now, markers))
                    drone.send_rc_control(*rc)
                    if course.want_land:
                        course.want_land = False
                        drone.send_rc_control(0, 0, 0, 0)
                        try:
                            drone.land()
                            bb.event("land", ok=True, source="course")
                        except Exception as e:
                            print("land failed:", e)
                            bb.event("land", ok=False, source="course", error=repr(e))
                        course.abort(now, "landed")

            canvas = bb.cycle(now, raw, frame, markers, flying, battery, temp)    # log 一列 + 錄影 + HUD
            if canvas is not None:
                io.show(canvas)
                continue
            # HUD 關掉 / 出錯時: 原本的簡單疊字畫面
            put(frame, "%s  %s  rc=%s" % (course.state, "FLYING" if flying else "ON GROUND", rc),
                (10, 25), (0, 255, 0))
            if battery is not None:
                put(frame, "Battery: %d%%" % battery, (10, 50),
                    (0, 0, 255) if battery < 20 else (0, 255, 0))
            if temp is not None:
                put(frame, "Temp: %.0f-%.0f C" % temp, (200, 50),
                    (0, 0, 255) if temp[1] >= P.TEMP_WARN else (0, 255, 0))
            el_run = course.elapsed(now)
            if el_run is not None:
                put(frame, "Time: %.1f s" % el_run, (400, 50), (0, 255, 255))
            if course.state == MANUAL:
                put(frame, "g/h/j/k/l: start segment 1-5", (10, 75), (0, 200, 255))
            io.show(frame)

    except KeyboardInterrupt:                          # Ctrl+C (SIGINT): 立刻降落
        interrupted = True
        print("\nSIGINT: sending land")
    finally:
        try:
            drone.send_rc_control(0, 0, 0, 0)
            if interrupted or getattr(drone, "is_flying", False):
                drone.land()
                bb.event("land", ok=True, source="exit")
        except BaseException as e:
            print("land failed:", e)
            bb.event("land", ok=False, source="exit", error=repr(e))
        if getattr(course, "run_t0", None) is not None and not getattr(course, "timer_done", False):
            course.report_time(clock.time())
        log("[end] max temperature %s, min battery %s" % (
            "?" if max_temp is None else "%.0f°C" % max_temp, "?" if min_bat is None else "%d%%" % min_bat))
        drone.streamoff()
        io.close()
