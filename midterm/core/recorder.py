"""recorder.py - mp4 recording on a background thread (the control loop only hands over frame references)

Recorder(path, size, fps, clock)
    submit(img, now, rgb=False) -> index of this moment in the video (-1 = dropped: queue full / recorder off)
    close(now)                  -> pads the end to `now`, finalises the file (an unfinalised mp4 is unreadable)

Timing: frame index = int((now - t_first) * fps) with the run's clock, so playback time ~ real time:
the writer repeats the previous image to fill gaps (video stall -> frozen picture of the right length)
and frames arriving faster than fps are skipped. Images of another size are resized for the file only.
"""
import atexit
import threading
import weakref

import cv2

try:
    import queue
except ImportError:                                     # pragma: no cover
    import Queue as queue                               # noqa: N813

MAX_GAP_S = 120.0          # longest gap filled with repeated frames (s)
_LIVE = weakref.WeakSet()


def _close_all():
    for r in list(_LIVE):
        try:
            r.close(None, timeout=5.0)
        except Exception:
            pass


atexit.register(_close_all)


class Recorder(object):
    def __init__(self, path, size, fps, clock, warn=print, maxsize=24, name="video"):
        self.path = path
        self.size = tuple(size)                          # (w, h)
        self.fps = float(fps)
        self.clock = clock
        self.warn = warn
        self.name = name
        self.t_first = None
        self.last_idx = -1
        self.dropped = 0
        self.written = 0
        self.enabled = False
        self.error = None
        self._closed = False
        self._q = queue.Queue(maxsize=maxsize)
        self._warned = set()
        self._writer = None
        try:
            for api in ((cv2.CAP_FFMPEG,) if hasattr(cv2, "CAP_FFMPEG") else ()) + (None,):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w = (cv2.VideoWriter(path, api, fourcc, self.fps, self.size) if api is not None
                     else cv2.VideoWriter(path, fourcc, self.fps, self.size))
                if w.isOpened():
                    self._writer = w
                    break
                w.release()
            if self._writer is None:
                raise RuntimeError("cv2.VideoWriter could not open %s" % path)
            self._thread = threading.Thread(target=self._run, name="rec-" + name)
            self._thread.daemon = True
            self._thread.start()
            self.enabled = True
            _LIVE.add(self)
        except Exception as e:
            self.error = str(e)
            self._warn_once("open", "[WARN] recording %s off: %s" % (name, e))

    def _warn_once(self, key, text):
        if key not in self._warned:
            self._warned.add(key)
            try:
                self.warn(text)
            except Exception:
                pass

    # ---------------- control-loop side ----------------
    def submit(self, img, now, rgb=False):
        """hand a frame to the writer thread (never blocks); returns its video frame index or -1"""
        if not self.enabled or img is None:
            return -1
        try:
            if self.t_first is None:
                self.t_first = now
            idx = int((now - self.t_first) * self.fps)
            if idx <= self.last_idx:
                return self.last_idx                       # faster than fps: this moment is already in the video
            self._q.put_nowait((idx, img, rgb))
            self.last_idx = idx
            return idx
        except queue.Full:
            self.dropped += 1
            self._warn_once("full", "[WARN] recording %s: writer too slow, dropping frames" % self.name)
            return -1
        except Exception as e:
            self.enabled = False
            self._warn_once("submit", "[WARN] recording %s off: %s" % (self.name, e))
            return -1

    def close(self, now=None, timeout=10.0):
        """pad to `now`, stop the thread, release the writer. Safe to call more than once."""
        self.enabled = False
        if self._writer is None:
            return
        try:
            if not self._closed:                           # first call: tell the thread to finish
                self._closed = True
                end = -1
                if now is not None and self.t_first is not None:
                    end = int((now - self.t_first) * self.fps)
                self._q.put((None, end, None), timeout=timeout)
            self._thread.join(timeout)                     # later calls (atexit after Ctrl+C) only wait
            if self._thread.is_alive():
                self._warn_once("join", "[WARN] recording %s: writer did not finish in %.0f s, file may be cut"
                                % (self.name, timeout))
        except Exception as e:
            self._warn_once("close", "[WARN] recording %s close: %s" % (self.name, e))

    # ---------------- writer thread ----------------
    def _prep(self, img, rgb):
        if rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return img

    def _run(self):
        prev = None
        max_gap = int(MAX_GAP_S * self.fps)
        try:
            while True:
                idx, img, rgb = self._q.get()
                if idx is None:                            # close: pad the end with the last image
                    end = img
                    if prev is not None and end is not None and end >= self.written:
                        for _ in range(min(end + 1 - self.written, max_gap)):
                            self._writer.write(prev)
                            self.written += 1
                    break
                frame = self._prep(img, rgb)
                if prev is not None and idx > self.written:
                    gap = idx - self.written
                    for _ in range(min(gap, max_gap)):
                        self._writer.write(prev)
                    if gap > max_gap:
                        self._warn_once("gap", "[WARN] recording %s: gap of %.0f s not filled, video shorter "
                                        "than the flight from here" % (self.name, gap / self.fps))
                    self.written = idx                     # keep counting in clock frames (after a cut gap the
                                                           # telemetry index runs ahead of the file position)
                self._writer.write(frame)
                self.written += 1
                prev = frame
        except Exception as e:
            self.error = str(e)
            self.enabled = False
            self._warn_once("thread", "[WARN] recording %s stopped: %s" % (self.name, e))
            try:                                           # drain so close() / submit() never block
                while True:
                    self._q.get_nowait()
            except Exception:
                pass
        finally:
            try:
                self._writer.release()
            except Exception:
                pass
