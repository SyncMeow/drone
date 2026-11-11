"""blackbox.py - glue between the main loop (app.py) and flight log / video recording / HUD

The main loop calls a few hooks; this module turns them into telemetry rows, events, mp4 frames and the
HUD canvas. Every hook is wrapped by _safe: an exception switches that hook off with one warning and
the control loop carries on. Nothing here sends commands to the drone or changes control values.
"""
import functools
import time

from core.flightlog import FlightLog, _fmt
from core.params import P
from core.states import MANUAL, SPLIT_STATES, FINISHED
from core.telemetry import state_field

RECORD_FPS = 15.0          # mp4 frame rate (frames are repeated / skipped against the clock)
IDLE_REFRESH = 0.1         # s: HUD refresh while no new frame is processed (video stalled)
IDLE_ROW_PERIOD = 0.2      # s: telemetry row while no new frame is processed
FPS_WINDOW = 1.0           # s: window for the fps numbers shown / logged


def _safe(fn):
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        if name in self.dead:
            return None
        try:
            return fn(self, *a, **k)
        except Exception as e:
            self.dead.add(name)
            try:
                print("[WARN] %s disabled after error: %r" % (name, e), flush=True)
            except Exception:
                pass
            return None
    return wrapper


class Blackbox(object):
    def __init__(self, clock, tag="real", log=True, record=True, hud_video=True, hud=True, log_root=None):
        self.dead = set()
        self.clock = clock
        self.tag = tag
        now = clock.time()
        self.flog = FlightLog(clock, tag, log_root, write_files=log)
        if log or hud:
            self.flog.install()
        self.course = None
        self.drone = None
        self.hud = None
        if hud:
            try:
                from core.hud import Hud
                self.hud = Hud()
            except Exception as e:
                print("[WARN] HUD off: %r" % e, flush=True)
        self.rec_raw = self.rec_hud = None
        if record and self.flog.dir is not None:
            from core.recorder import Recorder
            from core.hud import VID_W, VID_H, CANVAS_W, CANVAS_H
            self.rec_raw = Recorder(self.flog.path("fpv_raw.mp4"), (VID_W, VID_H), RECORD_FPS, clock,
                                    name="fpv_raw", maxsize=24)
            if hud_video and self.hud is not None:
                self.rec_hud = Recorder(self.flog.path("fpv_hud.mp4"), (CANVAS_W, CANVAS_H), RECORD_FPS, clock,
                                        name="fpv_hud", maxsize=12)
        # loop / video bookkeeping
        self.loop_t = now
        self.loop_dt = 0.0
        self.last_frame_t = None
        self.win_t = now
        self.win_new = self.win_done = 0
        self.fps_in = self.fps_proc = None
        self.last_row = -1e9
        self.last_show = -1e9
        self.last_img = None            # last annotated frame (for refreshes while the video is stalled)
        self.raw_idx = self.hud_idx = -1
        self.rc = (0, 0, 0, 0)
        self.n_sent = 0
        self.ka_n = -10
        self.key = None
        self.last_state = MANUAL
        self.n_splits = 0
        self.last_seg = 0

    # ---------------- setup ----------------
    @_safe
    def attach(self, course, drone):
        self.course = course
        self.drone = drone
        self.flog.state_fn = lambda: course.state
        self.flog.event("start", tag=self.tag, log_dir=self.flog.dir)

    @_safe
    def snapshot(self):
        speed = {"SPEED_SCALE_MOVE": P.SPEED_SCALE_MOVE, "SPEED_SCALE_UPDOWN": P.SPEED_SCALE_UPDOWN,
                 "SPEED_SCALE_TURN": P.SPEED_SCALE_TURN}
        self.flog.snapshot(P, {"speed_scale": speed, "tag": self.tag, "record_fps": RECORD_FPS,
                               "recording": {"raw": bool(self.rec_raw and self.rec_raw.enabled),
                                             "hud": bool(self.rec_hud and self.rec_hud.enabled)}})

    def wrap_on_enter(self, other):
        """on_enter hook for Course: logs the transition, then calls the io's own hook (if any)"""
        def on_enter(state, now):
            self.on_enter(state, now)
            if other is not None:
                other(state, now)
        return on_enter

    # ---------------- events ----------------
    @_safe
    def on_enter(self, state, now):
        prev = self.course.state if self.course is not None else "?"
        reason = None
        for t, lv, text in reversed(list(self.flog.recent)):
            if t >= self.loop_t and "->" not in text:
                reason = text.strip()
            break
        self.flog.event("state", **{"from": prev, "to": state, "reason": reason, "t_clock": round(now, 3)})
        self.last_state = state
        seg = state[1] if len(state) > 1 and state[0] == "S" and state[1].isdigit() else None
        if seg:
            self.last_seg = int(seg)

    def _poll(self, now):
        """transitions not made through Course.enter (abort -> MANUAL) and new timer splits"""
        c = self.course
        if c is None:
            return
        if c.state != self.last_state:
            self.flog.event("state", **{"from": self.last_state, "to": c.state,
                                        "reason": getattr(c, "dbg_abort", None), "t_clock": round(now, 3)})
            if c.state == MANUAL:
                self.flog.event("manual", reason=getattr(c, "dbg_abort", None))
            self.last_state = c.state
        splits = getattr(c, "splits", None)
        if splits is not None and len(splits) != self.n_splits:
            self.n_splits = len(splits)
            t0 = getattr(c, "run_t0", None)
            self.flog.event("split", splits={k: round(v - t0, 2) for k, v in splits.items()} if t0 else None)

    @_safe
    def event(self, kind, **fields):
        self.flog.event(kind, **fields)

    @_safe
    def loop(self, now):
        self.loop_dt = now - self.loop_t
        self.loop_t = now
        self._poll(now)

    @_safe
    def on_key(self, key, now):
        self.key = key
        self.flog.event("key", key=chr(key) if 32 < key < 127 else key)

    @_safe
    def sent(self, rc):
        self.rc = tuple(int(v) for v in rc)
        self.n_sent += 1

    @_safe
    def keepalive(self, now):
        if self.n_sent != self.ka_n + 1:                   # first keepalive after a real command
            self.flog.event("keepalive", note="no command for %.1f s, sending hover" % P.KEEPALIVE)
        self.ka_n = self.n_sent

    @_safe
    def new_frame(self, now):
        self.last_frame_t = now
        self.win_new += 1
        self._fps(now)

    def _fps(self, now):
        span = now - self.win_t
        if span >= FPS_WINDOW:
            self.fps_in = self.win_new / span
            self.fps_proc = self.win_done / span
            self.win_new = self.win_done = 0
            self.win_t = now

    def since_frame(self, now):
        return None if self.last_frame_t is None else now - self.last_frame_t

    # ---------------- per control cycle ----------------
    @_safe
    def cycle(self, now, raw, frame, markers, flying, battery, temp):
        """called once per processed frame (after control). Returns the canvas to show, or None"""
        self.win_done += 1
        self._fps(now)
        if self.rec_raw is not None:
            self.raw_idx = self.rec_raw.submit(raw, now, rgb=P.FRAME_IS_RGB)
        self.last_img = frame
        canvas = self._draw(now, frame, markers, flying, battery, temp)
        self._row(now, 1, markers, flying, battery, temp)
        return canvas

    @_safe
    def idle(self, now, flying, battery, temp):
        """called on loop iterations without a processed frame: HUD refresh (~10 Hz) + periodic row"""
        canvas = None
        self._fps(now)
        if self.hud is not None and now - self.last_show >= IDLE_REFRESH:
            canvas = self._draw(now, self.last_img, None, flying, battery, temp)
        if now - self.last_row >= IDLE_ROW_PERIOD:
            self._row(now, 0, None, flying, battery, temp)
        return canvas

    def _target(self, now, markers):
        """(target id, raw (x, y, z, ang) or None, smoothed or None)"""
        c = self.course
        t_dbg, mid = getattr(c, "dbg_track", (None, None))
        if t_dbg != now or mid is None:
            from core.hud import target_for
            mid = target_for(c.state) if c.state != MANUAL else None
            tracked = False
        else:
            tracked = True
        raw = None
        if markers is not None and mid is not None and mid in markers:
            m = markers[mid]
            raw = (float(m["t"][0]), float(m["t"][1]), float(m["t"][2]), float(m["ang"]))
        sm = None
        if tracked and raw is not None and c.last is not None:
            sm = tuple(float(v) for v in c.last)
        return mid, raw, sm

    def _row(self, now, ctrl, markers, flying, battery, temp):
        self.last_row = now
        c = self.course
        d = self.drone
        mid, raw, sm = self._target(now, markers) if ctrl else (None, None, None)
        rc = self.rc
        vals = {
            "t": "%.3f" % (now - self.flog.t0), "wall": "%.3f" % time.time(), "state": c.state,
            "flying": int(bool(flying)), "ctrl": ctrl,
            "rc_lr": rc[0], "rc_fb": rc[1], "rc_ud": rc[2], "rc_yaw": rc[3],
            "key": "" if self.key is None else (chr(self.key) if 32 < self.key < 127 else self.key),
            "markers": "" if markers is None else " ".join(str(k) for k in sorted(markers)),
            "target": "" if mid is None else mid, "arrive": c.arrive,
            "yaw": _fmt(state_field(d, "yaw"), "%.0f"), "tof": _fmt(state_field(d, "tof"), "%.0f"),
            "h": _fmt(state_field(d, "h"), "%.0f"), "bat": "" if battery is None else battery,
            "templ": "" if temp is None else "%.0f" % temp[0], "temph": "" if temp is None else "%.0f" % temp[1],
            "loop_dt": "%.4f" % self.loop_dt, "fps_in": _fmt(self.fps_in), "fps_proc": _fmt(self.fps_proc),
            "since_frame": _fmt(self.since_frame(now), "%.3f"),
            "rec_raw_idx": self.raw_idx if ctrl else "", "rec_hud_idx": self.hud_idx,
        }
        if raw is not None:
            vals.update(raw_x="%.1f" % raw[0], raw_y="%.1f" % raw[1], raw_z="%.1f" % raw[2], raw_ang="%.1f" % raw[3])
        if sm is not None:
            vals.update(sm_x="%.1f" % sm[0], sm_y="%.1f" % sm[1], sm_z="%.1f" % sm[2], sm_ang="%.1f" % sm[3])
        self.flog.row(vals)
        self.key = None

    def _draw(self, now, frame, markers, flying, battery, temp):
        if self.hud is None:
            return None
        c = self.course
        d = self.drone
        mid, raw, sm = self._target(now, markers) if markers is not None else (None, None, None)
        if markers is None:                          # stalled refresh: keep the last target info
            info_prev = getattr(self, "_last_info", {})
            mid, seen, meas = info_prev.get("target"), info_prev.get("seen"), info_prev.get("meas")
        else:
            seen = raw is not None
            meas = sm if sm is not None else raw
        lost_for = None
        if not seen and getattr(c, "lost_since", None) is not None:
            lost_for = now - c.lost_since
        t0 = getattr(c, "run_t0", None)
        splits = None
        if t0 is not None:
            sp = getattr(c, "splits", {})
            marks = [t0] + [sp.get(k) for k in SPLIT_STATES] + [sp.get(FINISHED)]
            cur = None if getattr(c, "timer_done", False) else now
            splits = []
            for i in range(5):
                a, b = marks[i], marks[i + 1]
                if a is not None and b is None and all(m is None for m in marks[i + 1:]) and cur is not None:
                    b = cur                                # running segment
                splits.append(None if a is None or b is None else b - a)
        rec_text = None
        recs = [r for r in (self.rec_raw, self.rec_hud) if r is not None]
        if recs:
            parts = []
            for r in recs:
                if not r.enabled:
                    parts.append("%s off" % r.name)
                elif r.dropped:
                    parts.append("%s drop %d" % (r.name, r.dropped))
            rec_text = "REC " + (", ".join(parts) if parts else "fpv_raw + fpv_hud" if len(recs) == 2 else "fpv_raw")
        info = {
            "state": c.state, "flying": flying, "last_seg": self.last_seg,
            "target": mid, "seen": seen, "lost_for": lost_for, "meas": meas,
            "arrive": c.arrive if c.state != MANUAL else None,
            "markers": " ".join(str(k) for k in sorted(markers)) if markers else
                       (getattr(self, "_last_info", {}).get("markers") if markers is None else ""),
            "rc": self.rc, "bat": battery, "temp": temp,
            "tof": state_field(d, "tof"), "h": state_field(d, "h"), "yaw": state_field(d, "yaw"),
            "fps_in": self.fps_in, "fps_proc": self.fps_proc, "since_frame": self.since_frame(now),
            "run_time": c.elapsed(now), "splits": splits,
            "messages": self.flog.messages(now),
            "rec": any(r.enabled for r in recs), "rec_text": rec_text,
        }
        self._last_info = info
        canvas = self.hud.draw(frame, info)
        self.last_show = now
        if self.rec_hud is not None:
            self.hud_idx = self.rec_hud.submit(canvas, now)
        return canvas

    @property
    def hud_ms(self):
        return None if self.hud is None else self.hud.ms

    # ---------------- end ----------------
    def close(self, now):
        for r in (self.rec_raw, self.rec_hud):
            try:
                if r is not None:
                    r.close(now)
                    self.flog.event("recording", file=r.name, frames=r.written, dropped=r.dropped, error=r.error)
            except Exception:
                pass
        try:
            self.flog.event("end", hud_ms=None if self.hud is None else round(self.hud.ms, 2))
        except Exception:
            pass
        try:
            self.flog.close()
        except Exception:
            pass
