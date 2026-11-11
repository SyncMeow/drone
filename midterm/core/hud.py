"""hud.py - operator window: video (960x720) + right panel + bottom message bar, cv2 only, English

Hud().draw(frame, info) -> BGR canvas (a new numpy array each call; safe to hand to a recorder)
`info` is a plain dict built by core/blackbox.py (missing keys are shown as "?").
"""
import time

import cv2
import numpy as np

from core.params import P
from core import states as S

VID_W, VID_H = 960, 720
PANEL_W = 340
BAR_H = 128
CANVAS_W, CANVAS_H = VID_W + PANEL_W, VID_H + BAR_H
MSG_LINES = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (235, 235, 235)
GRAY = (150, 150, 150)
DIM = (90, 90, 90)
GREEN = (80, 210, 80)
RED = (60, 60, 255)
YELLOW = (0, 215, 255)
ORANGE = (0, 150, 255)
CYAN = (230, 200, 60)
BG = (32, 32, 32)
BAR_BG = (18, 18, 18)

BAT_LOW = 20                # % -> red
TOF_LOW = 30                # cm -> red (close to the ground)
STALL_WARN = 0.5            # s without a new video frame -> "VIDEO STALLED"

# marker each state works on (shown when the state did not call track() this cycle)
STATE_TARGET = {
    S.S1_APPROACH: "M1_ID", S.S1_PAUSE1: "M1_ID", S.S1_RIGHT: "M2_ID", S.S1_PAUSE2: "M2_ID",
    S.S1_FORWARD: "M2_ID", S.S1_PAUSE3: "M2_ID", S.S1_LEFT: "M0_ID", S.S1_WAIT0: "M0_ID",
    S.S2_FOLLOW: "M0_ID", S.S3_ALIGN: "M3_ID", S.S3_DOWN: "M3_ID", S.S3_UNDER: "M3_ID",
    S.S3_PAUSE1: "M4_ID", S.S3_TURN: "M4_ID", S.S3_UP: "M4_ID", S.S3_WAIT4: "M4_ID", S.S3_FIND4: "M4_ID",
    S.S3_ALIGN4: "M4_ID", S.S4_CLIMB: "M5_ID", S.S4_LEFT: "M5_ID", S.S4_WAIT5: "M5_ID",
    S.S5_ALIGN: "M5_ID", S.S5_TABLE: "M5_ID",
}

# wanted values (distance, lateral, vertical, angle) per state; None = not controlled there
WANT = {
    S.S1_APPROACH: ("APPROACH_DIST", 0, 0, None),
    S.S1_RIGHT: (None, "SIDE_ALIGN_CM", None, None),
    S.S1_FORWARD: ("M2_STOP_DIST", 0, 0, None),
    S.S2_FOLLOW: ("FOLLOW_DIST", 0, None, None),
    S.S3_ALIGN: ("S3_DIST", 0, None, 0),
    S.S3_ALIGN4: ("M4_DIST", 0, 0, 0),
    S.S5_ALIGN: ("M5_DIST", 0, None, 0),
}


def target_for(state):
    name = STATE_TARGET.get(state)
    return None if name is None else getattr(P, name)


def segment_of(state):
    if state.startswith("S") and len(state) > 1 and state[1].isdigit():
        return int(state[1])
    return 5 if state == S.FINISHED else 0


def wanted(state):
    w = WANT.get(state)
    if w is None:
        return (None, None, None, None)
    return tuple(getattr(P, v) if isinstance(v, str) else v for v in w)


def key_hints(state):
    if state == S.MANUAL:
        return ["g h j k l: start seg 1-5 (takes off)", "1 takeoff  2 land  3 stop",
                "w s a d: move  z x: up/down  c v: turn", "q / ESC: quit (lands)"]
    if state in S.WAIT_STATES:
        return ["waiting: g h j k l restart a segment", "any move key: MANUAL", "q / ESC: quit (lands)"]
    hints = ["any move key (w s a d z x c v 1 2 3):", "  take over -> MANUAL"]
    if state == S.S2_FOLLOW:
        hints.insert(0, "j: go to segment 3 now")
    hints.append("q / ESC: quit (lands)")
    return hints


def _t(img, text, org, scale=0.5, color=WHITE, thick=1):
    text = text.replace(u"\u00b0", "").encode("ascii", "replace").decode("ascii")   # putText 只能畫 ASCII
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _fit(text, scale, thick, width):
    """cut text so it fits `width` pixels"""
    (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    if w <= width:
        return text
    n = max(1, int(len(text) * width / float(w)) - 2)
    return text[:n] + ".."


def _num(v, f="%.0f"):
    return "?" if v is None else f % v


class Hud(object):
    def __init__(self):
        self.ms = 0.0               # draw time EMA (ms)
        self.n = 0

    def draw(self, frame, info):
        t_start = time.perf_counter()
        canvas = np.empty((CANVAS_H, CANVAS_W, 3), np.uint8)
        canvas[:, VID_W:] = BG
        canvas[VID_H:, :] = BAR_BG
        self._video(canvas, frame, info)
        self._panel(canvas, info)
        self._bar(canvas, info)
        ms = (time.perf_counter() - t_start) * 1000.0
        self.ms = ms if self.n == 0 else 0.9 * self.ms + 0.1 * ms
        self.n += 1
        return canvas

    # ---------------- video area ----------------
    def _video(self, c, frame, info):
        v = c[:VID_H, :VID_W]
        if frame is None:
            v[:] = 0
            _t(v, "NO VIDEO", (VID_W // 2 - 110, VID_H // 2), 1.4, RED, 3)
        elif frame.shape[0] == VID_H and frame.shape[1] == VID_W:
            v[:] = frame if frame.ndim == 3 else frame[..., None]
        else:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            v[:] = cv2.resize(frame, (VID_W, VID_H), interpolation=cv2.INTER_NEAREST)
        stall = info.get("since_frame")
        if stall is not None and stall >= STALL_WARN and frame is not None:
            cv2.rectangle(v, (0, 0), (VID_W - 1, VID_H - 1), RED, 6)
            msg = "VIDEO STALLED %.1f s" % stall
            (w, _), _ = cv2.getTextSize(msg, FONT, 1.4, 3)
            x = (VID_W - w) // 2
            cv2.rectangle(v, (x - 12, 40), (x + w + 12, 100), (0, 0, 0), -1)
            _t(v, msg, (x, 88), 1.4, RED, 3)
        if info.get("rec"):
            cv2.circle(v, (VID_W - 70, 26), 9, RED, -1)
            _t(v, "REC", (VID_W - 55, 33), 0.6, RED, 2)

    # ---------------- right panel ----------------
    def _panel(self, c, info):
        x0 = VID_W + 12
        w = PANEL_W - 24
        st = info.get("state", "?")
        auto = st != S.MANUAL
        y = 40
        _t(c, _fit(st, 1.0, 2, w), (x0, y), 1.0, GREEN if auto else ORANGE, 2)

        # AUTO / MANUAL, FLYING / ON GROUND
        y += 14
        cv2.rectangle(c, (x0, y), (x0 + 96, y + 26), GREEN if auto else ORANGE, -1)
        _t(c, "AUTO" if auto else "MANUAL", (x0 + 8, y + 19), 0.55, (0, 0, 0), 2)
        fly = info.get("flying")
        cv2.rectangle(c, (x0 + 104, y), (x0 + 228, y + 26), CYAN if fly else DIM, -1)
        _t(c, "FLYING" if fly else "ON GROUND", (x0 + 112, y + 19), 0.55, (0, 0, 0), 2)
        _t(c, "hud %.1fms" % self.ms, (x0 + 236, y + 18), 0.4, GRAY)

        # segment progress 1..5
        y += 38
        seg = segment_of(st) if auto else info.get("last_seg", 0)
        bw = (w - 4 * 6) // 5
        for i in range(1, 6):
            xa = x0 + (i - 1) * (bw + 6)
            if i == seg:
                col, tc = (GREEN if auto else ORANGE), (0, 0, 0)
            elif i < seg:
                col, tc = (60, 100, 60), WHITE
            else:
                col, tc = (55, 55, 55), GRAY
            cv2.rectangle(c, (xa, y), (xa + bw, y + 22), col, -1)
            _t(c, "%d" % i, (xa + bw // 2 - 5, y + 17), 0.55, tc, 2 if i == seg else 1)

        # target marker
        y += 46
        tgt = info.get("target")
        seen = info.get("seen")
        if tgt is None:
            _t(c, "target: -", (x0, y), 0.6, GRAY, 1)
        else:
            _t(c, "target: marker %d" % tgt, (x0, y), 0.6, WHITE, 1)
            if seen:
                _t(c, "SEEN", (x0 + 220, y), 0.6, GREEN, 2)
            else:
                lost = info.get("lost_for")
                _t(c, "LOST" + ("" if lost is None else " %.1fs" % lost), (x0 + 200, y), 0.55, RED, 2)
        y += 22
        meas = info.get("meas")            # smoothed (x, y, z, ang)
        want = wanted(st)
        _t(c, "         meas    want", (x0, y), 0.45, GRAY)
        rows = (("dist", 2, want[0]), ("lat", 0, want[1]), ("vert", 1, want[2]), ("angle", 3, want[3]))
        for name, k, wv in rows:
            y += 19
            mv = None if meas is None else meas[k]
            col = WHITE if seen else GRAY
            _t(c, name, (x0, y), 0.45, GRAY)
            _t(c, "%7s" % _num(mv, "%+.0f" if k != 2 else "%.0f"), (x0 + 58, y), 0.45, col)
            _t(c, "%6s" % ("-" if wv is None else "%.0f" % wv), (x0 + 128, y), 0.45, YELLOW if wv is not None else DIM)
        arr = info.get("arrive")
        _t(c, "arrive %s/%d" % ("?" if arr is None else arr, P.ARRIVE_FRAMES), (x0 + 210, y - 38), 0.45,
           GREEN if arr else GRAY)
        _t(c, "seen: " + (info.get("markers") or "-"), (x0 + 210, y - 19), 0.45, GRAY)
        _t(c, "(cm / deg)", (x0 + 210, y), 0.4, DIM)

        # rc bars
        y += 16
        rc = info.get("rc") or (0, 0, 0, 0)
        half = (w - 70) // 2
        cx = x0 + 40 + half
        for name, v in zip(("LR", "FB", "UD", "YAW"), rc):
            y += 18
            _t(c, name, (x0, y), 0.45, GRAY)
            cv2.rectangle(c, (x0 + 40, y - 11), (x0 + 40 + 2 * half, y - 1), (55, 55, 55), -1)
            L = int(half * max(-100, min(100, v)) / 100.0)
            if L:
                cv2.rectangle(c, (cx, y - 11), (cx + L, y - 1), CYAN, -1)
            cv2.line(c, (cx, y - 13), (cx, y + 1), WHITE, 1)
            _t(c, "%+d" % v, (x0 + 46 + 2 * half, y), 0.42, WHITE)

        # drone status
        y += 26
        bat = info.get("bat")
        temp = info.get("temp")
        _t(c, "BAT", (x0, y), 0.45, GRAY)
        _t(c, "?" if bat is None else "%d%%" % bat, (x0 + 36, y), 0.55, RED if bat is not None and bat < BAT_LOW else GREEN, 2)
        _t(c, "TEMP", (x0 + 110, y), 0.45, GRAY)
        hot = temp is not None and temp[1] >= P.TEMP_WARN
        _t(c, "?" if temp is None else "%.0f-%.0fC" % temp, (x0 + 156, y), 0.55, RED if hot else GREEN, 2)
        y += 22
        tof, h, yaw = info.get("tof"), info.get("h"), info.get("yaw")
        _t(c, "TOF", (x0, y), 0.45, GRAY)
        _t(c, _num(tof), (x0 + 36, y), 0.5, RED if tof is not None and fly and tof < TOF_LOW else WHITE)
        _t(c, "H", (x0 + 92, y), 0.45, GRAY)
        _t(c, _num(h), (x0 + 108, y), 0.5, WHITE)
        _t(c, "YAW", (x0 + 166, y), 0.45, GRAY)
        _t(c, _num(yaw, "%+.0f"), (x0 + 204, y), 0.5, WHITE)
        y += 22
        fin, fpr, sf = info.get("fps_in"), info.get("fps_proc"), info.get("since_frame")
        stalled = sf is not None and sf >= STALL_WARN
        _t(c, "VIDEO %s fps  ctrl %s" % (_num(fin, "%.1f"), _num(fpr, "%.1f")), (x0, y), 0.45,
           RED if stalled or (fin is not None and fin < 5) else WHITE)
        if stalled:
            y += 22
            _t(c, "VIDEO STALLED %.1f s" % sf, (x0, y), 0.6, RED, 2)
        rec = info.get("rec_text")
        if rec:
            y += 20
            _t(c, _fit(rec, 0.42, 1, w), (x0, y), 0.42, RED if "drop" in rec or "off" in rec else GRAY)

        # timer
        y += 30
        run_t = info.get("run_time")
        _t(c, "RUN %s" % ("-" if run_t is None else "%.1f s" % run_t), (x0, y), 0.75, YELLOW, 2)
        splits = info.get("splits") or []
        for i, sp in enumerate(splits):
            xa = x0 + (i % 3) * 104
            ya = y + 20 + (i // 3) * 18
            _t(c, "s%d %s" % (i + 1, "-" if sp is None else "%.1f" % sp), (xa, ya), 0.45,
               WHITE if sp is not None else DIM)

        # key hints (bottom of the panel)
        hints = key_hints(st)
        yh = VID_H - 8 - 18 * (len(hints) - 1)
        cv2.line(c, (x0, yh - 20), (x0 + w, yh - 20), (70, 70, 70), 1)
        for i, line in enumerate(hints):
            _t(c, _fit(line, 0.45, 1, w), (x0, yh + 18 * i), 0.45, (200, 200, 120))

    # ---------------- bottom message bar ----------------
    def _bar(self, c, info):
        msgs = info.get("messages") or []
        y = VID_H + 22
        cv2.line(c, (0, VID_H), (CANVAS_W, VID_H), (70, 70, 70), 1)
        for level, text in msgs[-MSG_LINES:]:
            col = RED if level == "ERROR" else (ORANGE if level == "WARN" else (200, 200, 200))
            _t(c, _fit(text, 0.5, 1, CANVAS_W - 20), (10, y), 0.5, col, 1)
            y += 19
