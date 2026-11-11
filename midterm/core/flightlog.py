"""flightlog.py - one log directory per run: logs/YYYYmmdd_HHMMSS_real/ (or _sim)

    flight.log     human-readable: every line the program prints (stdout / stderr are teed, the terminal
                   output itself is untouched) + djitellopy's own logging (commands / responses)
    telemetry.csv  one row per control cycle (+ a row every IDLE_ROW_PERIOD s while no frame is processed)
    events.jsonl   state transitions, keys, warnings, take-off / land, timer splits
    params.json    effective parameters (after speed scaling) + git commit + versions

Robustness rule: nothing here may stop the control loop. Every public method catches its own errors;
a file that fails is switched off with one warning and the rest keeps going.
"""
import collections
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time

LAB6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(LAB6, "logs")

WARN_HOLD = 8.0           # HUD: warnings stay in the message bar this long (s)

TELEMETRY_COLUMNS = [
    "t", "wall", "state", "flying", "ctrl",
    "rc_lr", "rc_fb", "rc_ud", "rc_yaw", "key", "markers",
    "target", "raw_x", "raw_y", "raw_z", "raw_ang", "sm_x", "sm_y", "sm_z", "sm_ang", "arrive",
    "yaw", "tof", "h", "bat", "templ", "temph",
    "loop_dt", "fps_in", "fps_proc", "since_frame", "rec_raw_idx", "rec_hud_idx",
]

_WARN_TAGS = ("[WARN]", "[!!!]", "failed", "stalled", "SIGINT", "WARNING", "cannot", "not found",
              "may have landed", "too low", "hit the", "take over", "manually", "hovering", "not visible",
              "back to manual", "lost", "unknown", "without", "no marker", "heading off", "took over")
_ERROR_TAGS = ("Traceback", "Error", "ERROR", "Exception")


def classify(text):
    if any(t in text for t in _ERROR_TAGS):
        return "ERROR"
    if any(t in text for t in _WARN_TAGS):
        return "WARN"
    return "INFO"


def _fmt(v, f="%.1f"):
    if v is None:
        return ""
    try:
        return f % v
    except TypeError:
        return str(v)


class _Tee(object):
    """sys.stdout / sys.stderr replacement: writes through unchanged, and hands every full line to the log"""

    def __init__(self, stream, sink, name):
        self._stream = stream
        self._sink = sink
        self._name = name
        self._buf = ""

    def write(self, s):
        n = self._stream.write(s)
        try:
            self._buf += s
            if "\n" in self._buf:
                parts = self._buf.split("\n")
                self._buf = parts.pop()
                for line in parts:
                    self._sink.message(line, self._name)
        except Exception:
            pass
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _TelloFilter(logging.Filter):
    """adds seconds-since-start and the current state to djitellopy's log records"""

    def __init__(self, flog):
        logging.Filter.__init__(self)
        self.flog = flog

    def filter(self, record):
        try:
            record.trel = "%+9.3fs" % (self.flog.clock.time() - self.flog.t0)
            record.state = self.flog.state_fn()
        except Exception:
            record.trel, record.state = "", "?"
        return True


class FlightLog(object):
    def __init__(self, clock=time, tag="real", root=None, write_files=True):
        self.clock = clock
        self.t0 = clock.time()
        self.lock = threading.RLock()
        self.recent = collections.deque(maxlen=60)       # (clock t, level, text) for the HUD
        self.state_fn = lambda: "-"
        self.dir = None
        self._log = self._csv = self._csv_f = self._ev = None
        self._tees = None
        self._tello_handler = None
        self.warned = set()
        if not write_files:
            return
        try:
            root = root or LOG_ROOT
            name = time.strftime("%Y%m%d_%H%M%S") + "_" + tag
            path = os.path.join(root, name)
            i = 1
            while True:                                    # two runs in the same second: _2, _3, ...
                try:
                    os.makedirs(path)
                    break
                except FileExistsError:
                    i += 1
                    if i > 99:
                        raise
                    path = os.path.join(root, "%s_%d" % (name, i))
            self.dir = path
            self._log = open(os.path.join(path, "flight.log"), "a", buffering=1, encoding="utf-8")
            self._ev = open(os.path.join(path, "events.jsonl"), "a", buffering=1, encoding="utf-8")
            self._csv_f = open(os.path.join(path, "telemetry.csv"), "w", buffering=1, newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_f, lineterminator="\n")
            self._csv.writerow(TELEMETRY_COLUMNS)
        except Exception as e:
            self._warn_once("dir", "[WARN] flight log disabled (%s)" % e)
            self._close_files()

    # ---------------- helpers ----------------
    def path(self, name):
        return None if self.dir is None else os.path.join(self.dir, name)

    def _warn_once(self, key, text):
        if key in self.warned:
            return
        self.warned.add(key)
        try:
            print(text, flush=True)                       # goes through the tee if installed
        except Exception:
            pass

    def _close_files(self):
        for f in (self._log, self._ev, self._csv_f):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._log = self._ev = self._csv_f = self._csv = None

    # ---------------- stdout / stderr / djitellopy ----------------
    def install(self):
        """tee stdout + stderr into flight.log, attach a FileHandler to djitellopy's logger"""
        try:
            if self._tees is None:
                self._tees = (sys.stdout, sys.stderr)
                sys.stdout = _Tee(sys.stdout, self, "out")
                sys.stderr = _Tee(sys.stderr, self, "err")
        except Exception as e:
            self._warn_once("tee", "[WARN] cannot capture terminal output (%s)" % e)
        try:
            if self.dir is not None and self._tello_handler is None:
                h = logging.FileHandler(os.path.join(self.dir, "flight.log"), mode="a", encoding="utf-8")
                h.setFormatter(logging.Formatter(
                    "%(asctime)s.%(msecs)03d %(trel)s %(levelname)-5s %(state)-12s [tello] %(message)s",
                    "%Y-%m-%d %H:%M:%S"))
                h.addFilter(_TelloFilter(self))
                logging.getLogger("djitellopy").addHandler(h)
                self._tello_handler = h
        except Exception as e:
            self._warn_once("tello", "[WARN] djitellopy log not captured (%s)" % e)

    def uninstall(self):
        try:
            if self._tees is not None:
                for t in (sys.stdout, sys.stderr):
                    if isinstance(t, _Tee) and t._buf:
                        self.message(t._buf, t._name)
                        t._buf = ""
                sys.stdout, sys.stderr = self._tees
                self._tees = None
        except Exception:
            pass
        try:
            if self._tello_handler is not None:
                logging.getLogger("djitellopy").removeHandler(self._tello_handler)
                self._tello_handler.close()
                self._tello_handler = None
        except Exception:
            pass

    # ---------------- flight.log ----------------
    def message(self, text, source="out"):
        """one printed line (from the tee) -> ring buffer for the HUD + flight.log"""
        text = text.rstrip("\r")
        if not text.strip():
            return
        level = classify(text)
        if source == "err" and level == "INFO":
            level = "WARN"
        if source == "exc":
            level = "ERROR"
        try:
            now = self.clock.time()
            state = self.state_fn()
        except Exception:
            now, state = 0.0, "?"
        with self.lock:
            self.recent.append((now, level, text.strip()))
            if self._log is not None:
                try:
                    t = time.time()
                    self._log.write("%s.%03d %+9.3fs %-5s %-12s %s\n" % (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)), int((t % 1) * 1000),
                        now - self.t0, level, state, text.strip()))
                except Exception as e:
                    self._log = None
                    self._warn_once("log", "[WARN] flight.log write failed, file off (%s)" % e)

    def messages(self, now, n=6):
        """last n messages for the HUD; warnings / errors from the last WARN_HOLD s stay in view"""
        with self.lock:
            items = list(self.recent)
        if len(items) <= n:
            return [(lv, tx) for _, lv, tx in items]
        keep = set(range(len(items) - n, len(items)))
        held = [i for i, (t, lv, _) in enumerate(items) if lv != "INFO" and now - t <= WARN_HOLD and i not in keep]
        held = held[-(n // 2):]
        # replace the oldest info lines in the window with held warnings
        window = sorted(keep)
        for h in reversed(held):
            for j, i in enumerate(window):
                if items[i][1] == "INFO":
                    window.pop(j)
                    break
            else:
                break
            window.append(h)
        return [(items[i][1], items[i][2]) for i in sorted(window)]

    # ---------------- events.jsonl ----------------
    def event(self, kind, **fields):
        if self._ev is None:
            return
        try:
            now = self.clock.time()
            rec = {"t": round(now - self.t0, 3), "wall": round(time.time(), 3), "ev": kind,
                   "state": self.state_fn()}
            rec.update(fields)
            with self.lock:
                self._ev.write(json.dumps(rec, default=str, ensure_ascii=True) + "\n")
        except Exception as e:
            self._ev = None
            self._warn_once("ev", "[WARN] events.jsonl write failed, file off (%s)" % e)

    # ---------------- telemetry.csv ----------------
    def row(self, values):
        """values: dict keyed by TELEMETRY_COLUMNS (missing -> empty)"""
        if self._csv is None:
            return
        try:
            with self.lock:
                self._csv.writerow([values.get(c, "") for c in TELEMETRY_COLUMNS])
        except Exception as e:
            self._csv = None
            self._warn_once("csv", "[WARN] telemetry.csv write failed, file off (%s)" % e)

    # ---------------- params.json ----------------
    def snapshot(self, params, extra=None):
        p = self.path("params.json")
        if p is None:
            return
        try:
            d = params.as_dict()
            d = dict(d)
            d["START_KEYS"] = {chr(k): v for k, v in d["START_KEYS"].items()}
            d["MANUAL_MAX"] = list(d["MANUAL_MAX"])
            info = {"params": d,
                    "base_times_before_speed_scale": dict(params.__dict__.get("_base_times", {})),
                    "params_file": params.__dict__.get("_path", ""),
                    "git": _git_info(),
                    "platform": platform.platform(), "python": sys.version.split()[0],
                    "versions": _versions(), "argv": sys.argv,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            if extra:
                info.update(extra)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1, sort_keys=False, default=str)
        except Exception as e:
            self._warn_once("params", "[WARN] params.json not written (%s)" % e)

    def close(self):
        self.uninstall()
        with self.lock:
            self._close_files()


def _git_info():
    def git(*args):
        try:
            out = subprocess.run(("git",) + args, cwd=LAB6, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=3)
            return out.stdout.decode("utf-8", "replace").strip() if out.returncode == 0 else None
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": git("status", "--porcelain", "--", ".")}


def _versions():
    out = {}
    for mod in ("cv2", "numpy", "djitellopy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", None)
            if out[mod] is None and mod == "djitellopy":
                try:
                    from importlib import metadata
                    out[mod] = metadata.version("djitellopy")
                except Exception:
                    out[mod] = "?"
        except Exception:
            out[mod] = None
    return out
