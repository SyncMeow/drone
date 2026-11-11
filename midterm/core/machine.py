"""machine.py - 賽道狀態機 Course: 狀態切換 / 計時 / 量測平滑 / 航向保持 / 對正 / 找 marker

每段的邏輯在 segments.py (step_s1 ~ step_s5); step() 先處理共用的暫停 / 等待狀態再分派過去
"""
import math
import time

from core.console import log
from core.control import Ema, p_ctrl, wrap180
from core.params import P
from core.segments import Segments
from core.states import (MANUAL, S1_APPROACH, S1_LEFT, S1_WAIT0, S2_FOLLOW, S3_ALIGN, S3_UNDER, S3_TURN,
                         S3_WAIT4, S3_ALIGN4, S4_LEFT, S4_WAIT5, S5_ALIGN, S5_TABLE, FINISHED,
                         SPLIT_STATES, PAUSES, SEG1, SEG2, SEG3, SEG4, SEG5)
from core.telemetry import read_alt, read_tof, read_yaw


# ============================================================================
# 賽道狀態機
# ============================================================================
class Course(Segments):
    def __init__(self, drone, clock=time, on_enter=None):
        self.drone = drone
        self.clock = clock                 # time() 的來源 (模擬時是假時鐘)
        self.on_enter = on_enter           # on_enter(state, now): 每次 enter() 先呼叫 (模擬器記錄狀態時間用)
        self.state = MANUAL
        self.t0 = 0.0
        self.ema = Ema()
        self.ema5 = Ema(P.S5_ANG_ALPHA)      # 第五段: marker 5 角度的平均
        self.ema_id = None
        self.ema_seen = 0.0
        self.arrive = 0
        self.heading_ref = None
        self.heading_ok = False
        self.turn_target = None
        self.last = None                   # 最後一次看到目標 marker 的平滑量測 (x, y, z, ang)
        self.last_bearing = None           # 第二段: marker 0 最後出現的方向 (deg)
        self.prev_bearing = None           # 第二段: 上一次的方向 / 時間, 用來估角速度
        self.prev_bt = 0.0
        self.bearing_rate = 0.0            # marker 0 繞著無人機移動的角速度 (deg/s, 右/順時針為正)
        self.last_yaw_cmd = 0
        self.last_seen = 0.0
        self.lost_since = None
        self.alt0 = None                   # 進入狀態時的高度
        self.hold_alt = None               # 鑽桌子時要維持的高度
        self.alt_left = None               # 第四段向左時要維持的高度
        self.notes = set()
        self.want_land = False
        self.last_dbg = 0.0
        self.dbg_track = (None, None)      # 只給 log / HUD 看: (時間, marker id) = 最後一次 track() 在追哪個 marker

    # ---------------- 狀態切換 ----------------
    def enter(self, state, now):
        if self.on_enter is not None:
            self.on_enter(state, now)
        if state != self.state:
            print("[%.1f] %s -> %s" % (now, self.state, state))
        prev = self.state
        self.state = state
        self.t0 = now
        self.arrive = 0
        self.ema.reset()
        self.ema5.reset()
        self.ang5 = None
        self.ang3 = None
        self.ema3a = Ema(P.S3_ANG_ALPHA)                     # S3_ALIGN: marker 3 角度的平均
        self.s5_hist = []                                  # 第五段: 到位後的位置紀錄 (判斷有沒有停穩)
        self.s5_n = 0                                      # 第五段簡單版: 連續到位的張數
        self.ema.alpha = P.FOLLOW_EMA_ALPHA if state == S2_FOLLOW else P.EMA_ALPHA
        self.ema_id = None
        self.last = None
        self.last_bearing = None
        self.prev_bearing = None
        self.bearing_rate = 0.0
        self.last_yaw_cmd = 0
        self.m0_ever = False                               # 第二段: 進來之後有沒有看過 marker 0
        self.m3_since = self.m3_last = None                # 第二段: marker 0 不在 + marker 3 在 的開始時間
        self.ema3 = getattr(self, "ema3", None) or Ema()
        self.m0_back = 0                                   # S3_ALIGN: 連續看到 marker 0 的 frame 數
        self.lat_i, self.lat_t = 0.0, None                 # S3_ALIGN 左右積分 (抵抗一直飄)
        self.last_seen = now
        self.lost_since = None
        self.notes = set()
        self.alt0 = read_alt(self.drone)
        if state == S3_UNDER:
            self.hold_alt = self.alt0
            self.under_time = P.S3_UNDER_TIME
            v = P.S3_UNDER_SPEED * P.SPEED_SCALE_MOVE * P.S3_UNDER_CM_PER_RC
            z0 = getattr(self, "s3_z0", None)
            if z0 is not None and v > 0:
                self.under_time = (z0 + P.S3_PAST_CM) / v
                print("  %.0f cm from marker 3, flying forward %.0f cm (%.1f s)" % (z0, z0 + P.S3_PAST_CM, self.under_time))
            else:
                print("  distance to marker 3 unknown, flying forward for %.1f s" % self.under_time)
        self.s3_z0 = None
        if state == S3_TURN:
            yaw = read_yaw(self.drone)
            base = self.heading_ref if self.heading_ok else yaw
            self.turn_target = None if base is None else wrap180(base + P.S3_TURN_DEG)
        if state in (S3_ALIGN, S3_ALIGN4, S5_ALIGN) and prev != state:
            self.heading_ok = False                    # 對正時用 marker 角度轉向, 不做航向保持
        if state in (S4_LEFT, S5_ALIGN) and prev not in (S4_LEFT, S4_WAIT5):
            self.tof_floor = None                          # 重新記錄地板高度
        if getattr(self, "run_t0", None) is not None and not getattr(self, "timer_done", False):
            for key in SPLIT_STATES:
                if state == key and (key == S3_ALIGN or key not in self.splits):
                    self.splits[key] = now                 # S3_ALIGN 取最後一次 (會從 S3_ALIGN 回 S2)
            if state == FINISHED:
                self.splits[FINISHED] = now
                self.timer_done = True
                self.report_time(now)
        self.s2_hold_until = now + P.S2_START_HOLD if (state == S2_FOLLOW and prev in (S1_LEFT, S1_WAIT0)) else 0.0
        if state == FINISHED:
            print("course finished, landing")
            self.want_land = True

    def elapsed(self, now):
        """按 g 之後經過幾秒 (沒按過 g 回傳 None)"""
        t0 = getattr(self, "run_t0", None)
        if t0 is None:
            return None
        end = self.splits.get(FINISHED) if getattr(self, "timer_done", False) else None
        return (end if end is not None else now) - t0

    def report_time(self, now):
        """印出總時間與各段時間 / 佔比 (按 g 開始計時)"""
        if getattr(self, "run_t0", None) is None:
            return
        total = self.elapsed(now)
        marks = [("g", self.run_t0)] + [(k, self.splits.get(k)) for k in SPLIT_STATES] + \
                [("land", self.splits.get(FINISHED))]
        names = ["seg 1  g -> S2_FOLLOW", "seg 2  S2_FOLLOW -> S3_ALIGN", "seg 3  S3_ALIGN -> S4",
                 "seg 4  S4 -> S5_ALIGN", "seg 5  S5_ALIGN -> land"]
        log("[timer] total %.1f s%s" % (total, "" if FINISHED in self.splits else " (not finished)"))
        for i, name in enumerate(names):
            a, b = marks[i][1], marks[i + 1][1]
            if a is None or b is None:
                log("[timer]   %-28s   -" % name)
            else:
                log("[timer]   %-28s %6.1f s  %5.1f%%" % (name, b - a, 100.0 * (b - a) / max(total, 1e-6)))

    def note(self, key, msg):
        if key not in self.notes:
            self.notes.add(key)
            print("  " + msg)

    def set_heading_ref(self):
        self.heading_ref = read_yaw(self.drone)
        self.heading_ok = self.heading_ref is not None
        if not self.heading_ok:
            print("  (cannot read yaw, heading hold off)")

    def begin(self, state, now):
        """從手動直接進入某一段 (g h j k l)"""
        print("start %s" % state)
        if state == S1_APPROACH:                           # 按 g: 從這裡開始計時
            self.run_t0 = now
            self.splits = {}
            self.timer_done = False
            log("[timer] start")
        self.set_heading_ref()                             # 直線段的方向 = 按下按鍵時的方向
        self.enter(state, now)

    def abort(self, now, why):
        self.dbg_abort = why                            # 只給 log 看: 最後一次切回手動的原因
        if self.state != MANUAL:
            print("[%.1f] back to manual (%s)" % (now, why))
        self.state = MANUAL

    # ---------------- 量測 ----------------
    def track(self, markers, mid, now):
        """目標 marker 的平滑量測 (x, y, z, ang); 看不到回傳 None"""
        self.dbg_track = (now, mid)                     # 只給 log / HUD 看, 不影響控制
        m = markers.get(mid)
        if m is None:
            if self.lost_since is None:
                self.lost_since = now
            return None
        self.lost_since = None
        if self.ema_id != mid or now - self.ema_seen > 0.5:
            self.ema.reset()
            self.ema_id = mid
        self.ema_seen = now
        self.last_seen = now
        x, y, z = m["t"]
        self.last = self.ema.update((x, y, z, m["ang"]))
        return self.last

    # ---------------- 航向 ----------------
    def heading_err(self, target):
        yaw = read_yaw(self.drone)
        return None if yaw is None or target is None else wrap180(yaw - target)

    def heading_cmd(self):
        if not (P.HOLD_HEADING and self.heading_ok):
            return 0
        err = self.heading_err(self.heading_ref)
        if err is None:
            return 0
        if abs(err) > P.HEADING_ABORT:
            print("  heading off by %.0f deg, heading hold stopped (check HEADING_SIGN)" % err)
            self.heading_ok = False
            return 0
        cmd = p_ctrl(-P.HEADING_SIGN * err, P.HEADING_KP, P.HEADING_DEAD, P.HEADING_MAX, P.HEADING_MIN)
        if P.DEBUG_HEADING and self.clock.time() - self.last_dbg >= 1.0:
            self.last_dbg = self.clock.time()
            print("  [%s] err=%.0f -> %d" % (self.state, err, cmd))
        return cmd

    @staticmethod
    def lat(x):
        return p_ctrl(x, P.LAT_KP, P.LAT_DEAD, P.LAT_MAX)          # marker 在右 -> 往右

    @staticmethod
    def vert(y):
        return p_ctrl(-y, P.UD_KP, P.UD_DEAD, P.UD_MAX)            # marker 在下 (y>0) -> 往下

    @staticmethod
    def fwd(err, dead=None, cap=None):
        dead = P.FB_DEAD if dead is None else dead
        cap = P.FB_MAX if cap is None else cap
        return p_ctrl(err, P.FB_KP, dead, cap, P.FB_MIN)         # 太遠 -> 往前; 太近 -> 往後

    def alt_hold(self, target):
        alt = read_alt(self.drone)
        if alt is None or target is None:
            return 0
        return p_ctrl(target - alt, P.ALT_KP, P.ALT_DEAD, P.ALT_MAX)

    def arrived(self, cond):
        self.arrive = self.arrive + 1 if cond else 0
        return self.arrive >= P.ARRIVE_FRAMES

    @staticmethod
    def clear3(y, z):
        """marker 3 下緣比鏡頭高多少 (cm, 真正的垂直方向): 補償鏡頭往下傾 CAM_PITCH_DEG"""
        a = math.radians(P.CAM_PITCH_DEG)
        down = y * math.cos(a) + z * math.sin(a)          # marker 中心在鏡頭 "正下方" 多少 (負 = 在上方)
        return -(down + P.MARKER_SIZE / 2)

    @staticmethod
    def low_enough(alt):
        return P.S3_PASS_ALT is None or alt is None or alt <= P.S3_PASS_ALT

    def seek(self, mid, now):
        """對正中的 marker 跑出畫面: 依最後看到的位置往那個方向慢慢移動找回來; 沒有資料或超時就懸停"""
        if self.last is None or now - self.lost_since > P.SEEK_TIME:
            self.note("lost%d" % mid, "marker %d not visible, hovering" % mid)
            return (0, 0, 0, 0)
        x, y, z = self.last[0], self.last[1], max(self.last[2], 1.0)
        # 只有 marker 消失前靠近畫面邊緣才往那邊找; 在畫面中間消失 = 只是漏抓 (晃動 / 模糊 / Wi-Fi), 原地等
        # (原本只要偏離中心 6 cm 就全速往旁邊 + 往上飛, 實飛第五段每漏抓一次就被甩開一次)
        waited = now - self.lost_since > P.SEEK_WAIT          # 原地等了一陣子還是沒有: 不是漏抓, 真的跑掉了
        edge_x = waited and abs(x) > P.LAT_DEAD or abs(math.degrees(math.atan2(x, z))) > P.SEEK_EDGE_X_DEG
        edge_y = waited and abs(y) > P.LAT_DEAD or abs(math.degrees(math.atan2(y, z))) > P.SEEK_EDGE_Y_DEG
        lr = (P.SEEK_SPEED if x > 0 else -P.SEEK_SPEED) if edge_x else 0
        ud = (-P.SEEK_SPEED if y > 0 else P.SEEK_SPEED) if edge_y else 0   # y < 0: marker 在上方
        if not lr and not ud:
            self.note("seek%d" % mid, "marker %d missed (was mid-frame), waiting in place" % mid)
        else:
            self.note("seek%d" % mid, "marker %d left the frame, searching %s" % (
                mid, "-".join(t for t, on in (("right" if x > 0 else "left", lr), ("down" if y > 0 else "up", ud)) if on)))
        return (lr, 0, ud, 0)

    def align(self, p, dist, tol, use_ud):
        """對正 marker: 左右置中、距離 dist、(可選) 高度置中、轉向正對。回傳 (rc, 是否對正)"""
        x, y, z, ang = p
        if P.ALIGN_YAW:
            yaw = p_ctrl(P.SIGN_ALIGN_YAW * ang, P.ALIGN_YAW_KP, P.ALIGN_YAW_DEAD, P.ALIGN_YAW_MAX, P.ALIGN_YAW_MIN)
        else:
            yaw = 0
        ud = p_ctrl(-y, P.UD_KP, P.ALIGN_UD_DEAD, P.UD_MAX, P.ALIGN_UD_MIN) if use_ud else 0
        rc = (self.lat(x), self.fwd(z - dist, dead=tol), ud, yaw)
        ok = (abs(x) < P.LAT_DEAD + 2 and abs(z - dist) <= tol
              and (not use_ud or abs(y) < P.ALIGN_UD_DEAD + 2)
              and (not P.ALIGN_YAW or abs(ang) < P.ALIGN_YAW_DEAD + 1))
        return rc, ok

    # ---------------- 每個週期 ----------------
    def step(self, now, markers):
        st = self.state
        el = now - self.t0

        if st in (S4_LEFT, S4_WAIT5, S5_ALIGN, S5_TABLE):   # 記錄地板的 TOF (取最大 = 下面是地板時)
            t = read_tof(self.drone)
            if t is not None:
                self.tof_floor = t if getattr(self, "tof_floor", None) is None else max(self.tof_floor, t)

        if st in PAUSES:                                   # 動作之間停一下, 保持航向
            if el >= P.PAUSE_TIME:
                self.enter(PAUSES[st], now)
            return (0, 0, 0, self.heading_cmd())

        if st in (S1_WAIT0, S3_WAIT4, S4_WAIT5):           # 停下來等目標 marker 出現
            want = {S1_WAIT0: (P.M0_ID, S2_FOLLOW), S3_WAIT4: (P.M4_ID, S3_ALIGN4),
                    S4_WAIT5: (P.M5_ID, S5_ALIGN)}[st]
            if self.arrived(want[0] in markers):
                self.enter(want[1], now)
                return (0, 0, 0, 0)
            return (0, 0, 0, self.heading_cmd())

        if st in SEG1:
            return self.step_s1(st, now, el, markers)
        if st in SEG2:
            return self.step_s2(st, now, el, markers)
        if st in SEG3:
            return self.step_s3(st, now, el, markers)
        if st in SEG4:
            return self.step_s4(st, now, el, markers)
        if st in SEG5:
            return self.step_s5(st, now, el, markers)
        return (0, 0, 0, 0)
