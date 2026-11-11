"""
viewer.py - sim window: flight program image (HUD canvas: FPV + panel + message bar) on the left, a right
column with the 3D course view on top and sim status / keys below; plus SimIO, the io object the flight
loop talks to (wait_key / show). The composed image is shrunk for display to fit the screen (--scale);
logs and recordings are not affected.

Sim keys (chosen not to clash with the flight keys w s a d z x c v 1 2 3 5 6 g h j k l q ESC):
    p / space  pause          n  single step (while paused)     [ ]  speed x0.5 / x1 / x2 / x4
    r  reset drone + human    t  top view   y  side view   u  follow view   i  overview
    = / -  zoom in / out      mouse: left drag = orbit, wheel = zoom
    m  human manual on/off    arrow keys = move the human (manual)     b  human hides / shows marker 0
"""
import csv
import math
import os
import sys
import time as _time

import cv2
import numpy as np

from render import Renderer, drone_model, intrinsics_fov, look_at

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # lab6/ (core.display)
from core.display import fit_scale, shrink                                          # noqa: E402

SPEEDS = (0.5, 1.0, 2.0, 4.0)
# arrow keys from cv2.waitKeyEx: GTK (Linux), Qt, macOS (cocoa), Windows
ARROWS = {65361: "left", 65362: "up", 65363: "right", 65364: "down",
          0x1000012: "left", 0x1000013: "up", 0x1000014: "right", 0x1000015: "down",
          63234: "left", 63232: "up", 63235: "right", 63233: "down",
          2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down"}
HELP = ["SIM KEYS (flight keys go to the program)",
        "p / space  pause      n  single step",
        "[ ]  sim speed x0.5 .. x4      r  reset drone + human",
        "t y u i  view: top / side / follow / overview",
        "= -  zoom     mouse: drag = orbit, wheel = zoom",
        "m  human manual (arrow keys move him)",
        "b  human hides / shows marker 0"]
TRUTH_COLUMNS = ["t", "x", "y", "z", "yaw", "pitch", "roll", "vx", "vy", "vz", "collision", "human_x", "human_y",
                 "m0_visible"]


def put(img, text, org, color=(255, 255, 255), scale=0.45, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Viewer(object):
    def __init__(self, drone, panel=(640, 480), window="drone sim"):
        self.drone = drone
        self.world = drone.world
        self.pw, self.ph = panel
        self.window = window
        self.obs = Renderer(self.world, intrinsics_fov(self.pw, self.ph, 60.0), panel, grid=True)
        lo, hi = self.world.floor_lo, self.world.floor_hi
        self.center = np.array([(lo[0] + hi[0]) / 2.0 - 100, (lo[1] + hi[1]) / 2.0, 0.0])
        self.set_view("overview")
        self._drag = None
        self.status = []                    # extra lines from SimIO
        self.left_w = 0                     # width of the program image in the last composed frame
        self.left_size = (1300, 848)        # program image size used while there is none yet

    # ---- observer camera ---------------------------------------------------
    def set_view(self, name):
        self.mode = name
        if name == "overview":
            self.target, self.az, self.el, self.dist = self.center.copy(), -115.0, 42.0, 1200.0
        elif name == "top":
            self.target, self.az, self.el, self.dist = self.center.copy(), -90.0, 89.0, 1350.0
        elif name == "side":
            self.target, self.az, self.el, self.dist = self.center + [0, 0, 80], -90.0, 6.0, 1300.0
        elif name == "follow":
            self.el, self.dist = 22.0, 260.0

    def camera(self):
        if self.mode == "follow":
            p = self.drone.pos
            self.target = np.array([p[0], p[1], p[2] + 10.0])
            h = math.radians(self.drone.heading)
            self.az = math.degrees(math.atan2(math.sin(h), -math.cos(h)))     # behind the drone
        el, az = math.radians(self.el), math.radians(self.az)
        eye = self.target + self.dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                                  math.sin(el)])
        return eye, look_at(eye, self.target)

    def zoom(self, k):
        self.dist = float(min(max(self.dist * k, 60.0), 4000.0))

    def on_mouse(self, event, x, y, flags, param=None):
        """x, y in composed (full resolution) image coordinates"""
        x -= self.left_w
        if x < 0 or y >= self.ph:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y, self.az, self.el)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            x0, y0, az0, el0 = self._drag
            if self.mode in ("top", "side"):
                self.mode = "free"
            self.az = az0 - (x - x0) * 0.4
            self.el = float(min(max(el0 + (y - y0) * 0.3, 2.0), 89.0))
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", -99):
            try:
                d = cv2.getMouseWheelDelta(flags)
            except Exception:
                d = 1 if flags > 0 else -1
            self.zoom(0.88 if d > 0 else 1.0 / 0.88)

    # ---- drawing -------------------------------------------------------------
    def render3d(self):
        d = self.drone
        eye, R = self.camera()
        hit = d.clock.t < d.hit_until
        dyn = list(d.human.objects()) if d.human is not None else []
        pitch, roll = d.body.pitch_roll()
        dyn.append(drone_model(d.pos, d.heading, pitch, roll, scale=2.5, hit=hit))
        img = self.obs.render(eye, R, dyn, highlight=d.hit_names if hit else ())
        o = self.obs
        # trail
        if d.trace:
            tr = np.array([(r[1], r[2], r[3] + 5.0) for r in d.trace[-4000:]])
            uv, z = o.project(tr, eye, R)
            ok = z > o.near
            pts = np.round(uv * 16).astype(np.int32)
            run = []
            for i in range(len(pts)):
                if ok[i] and abs(pts[i, 0]) < 1e6 and abs(pts[i, 1]) < 1e6:
                    run.append(pts[i])
                elif len(run) > 1:
                    cv2.polylines(img, [np.array(run)], False, (0, 160, 255), 1, cv2.LINE_AA, 4)
                    run = []
                else:
                    run = []
            if len(run) > 1:
                cv2.polylines(img, [np.array(run)], False, (0, 160, 255), 1, cv2.LINE_AA, 4)
        # drop line to the surface below + camera frustum
        p = d.pos
        o.line3d(img, (p[0], p[1], p[2]), (p[0], p[1], p[2] - d.true_tof()), eye, R, (90, 90, 90), 1)
        cp, cR = d.camera_pose()
        K = d.K
        far = 90.0
        cs = []
        for u, v in ((0, 0), (d.W, 0), (d.W, d.H), (0, d.H)):
            ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * far
            cs.append(cp + cR.T @ ray)
        for i in range(4):
            o.line3d(img, cp, cs[i], eye, R, (0, 220, 255), 1)
            o.line3d(img, cs[i], cs[(i + 1) % 4], eye, R, (0, 220, 255), 1)
        # marker labels
        for mk in self.world.markers:
            uv, z = o.project(mk.pos[None, :] + [0, 0, 14], eye, R)
            if z[0] > o.near and 0 <= uv[0, 0] < self.pw and 0 <= uv[0, 1] < self.ph:
                put(img, "M%d" % mk.id, (int(uv[0, 0]) - 10, int(uv[0, 1])), (0, 255, 255), 0.4)
        if d.human is not None and d.human.show_marker:
            uv, z = o.project(d.human.marker_center()[None, :] + [0, 0, 16], eye, R)
            if z[0] > o.near and 0 <= uv[0, 0] < self.pw and 0 <= uv[0, 1] < self.ph:
                put(img, "M%d" % d.human.mid, (int(uv[0, 0]) - 10, int(uv[0, 1])), (0, 255, 255), 0.4)
        if hit:
            put(img, "COLLISION: " + ", ".join(sorted(d.hit_names)), (8, self.ph - 12), (60, 60, 255), 0.6, 2)
        put(img, "3D view: %s" % self.mode, (8, 18))
        return img

    def info_lines(self):
        d = self.drone
        p, v = d.pos, d.vel
        pitch, roll = d.body.pitch_roll()
        st = d.get_current_state()
        lines = [("sim t %.1f s" % (d.clock.t - d.t0), (255, 255, 255)),
                 ("true pos %.0f %.0f %.0f cm   heading %.0f" % (p[0], p[1], p[2], d.heading), (255, 255, 255)),
                 ("speed %.0f cm/s  vz %.0f   pitch %.0f  roll %.0f" % (math.hypot(v[0], v[1]), v[2], pitch, roll),
                  (255, 255, 255)),
                 ("state packet: tof %d  h %d  bat %d%%  yaw %d  temp %d" % (st["tof"], st["h"], st["bat"], st["yaw"],
                                                                             st["temph"]), (200, 200, 200)),
                 ("motors %s   collisions %d" % ("ON" if d.body.motors else "off", len(d.collisions)),
                  (60, 60, 255) if d.collisions else (255, 255, 255))]
        if d.human is not None:
            lines.append(("human: %s%s, marker 0 %s" % (d.human.status, " (manual)" if d.human.manual else "",
                                                        "shown" if d.human.show_marker else "hidden"), (255, 255, 255)))
        lines += [(s, (0, 215, 255)) for s in self.status]
        return lines

    def compose(self, left_bgr):
        """full resolution window image: program image | (3D view over sim info)"""
        if left_bgr is None:
            lw, lh = self.left_size
            left = np.zeros((lh, lw, 3), np.uint8)
            put(left, "waiting for the flight program image ...", (20, 40), (200, 200, 200), 0.7)
        else:
            left = left_bgr
            self.left_size = (left.shape[1], left.shape[0])
        H = max(left.shape[0], self.ph + 230)
        right = np.full((H, self.pw, 3), 26, np.uint8)
        right[:self.ph] = self.render3d()
        y = self.ph + 22
        for text, col in self.info_lines():
            put(right, text, (10, y), col, 0.5)
            y += 21
        y += 8
        for i, text in enumerate(HELP):
            if y > H - 6:
                break
            put(right, text, (10, y), (220, 220, 220) if i else (0, 215, 255), 0.45)
            y += 19
        if left.shape[0] < H:
            left = np.vstack([left, np.zeros((H - left.shape[0], left.shape[1], 3), np.uint8)])
        self.left_w = left.shape[1]
        return np.hstack([left, right])


class SimIO(object):
    """io for the flight loop: wait_key(ms) advances the simulation, show(image) displays it.
    headless=True: no window, as fast as possible; keys come from script(io) -> key or -1.
    scale: display scale of the window (None = fit the screen); the composed image itself stays full size.
    attach_blackbox(bb, course) (called by core.app.run): writes truth.csv + collision events into the run's
    log directory; on_step callbacks (fn(io)) run after every simulation step (scenario scripts)."""

    def __init__(self, drone, viewer=None, headless=False, script=None, loop_dt=None, speed=1.0, scale=None):
        self.drone = drone
        self.clock = drone.clock
        self.viewer = viewer
        self.headless = headless or viewer is None
        self.script = script
        self.loop_dt = float(loop_dt or drone.P.loop_dt_s)
        self.speed = speed
        self.scale = scale
        self.paused = False
        self.step_once = False
        self.last_image = None
        self.n_calls = 0
        self._wall_next = None
        self._last_draw = 0.0
        self._next_draw = 0.0
        self.n_draw = 0
        self._t_open = _time.time()
        self.draw_ms = []
        self._window_ok = False
        self.bb = None
        self.course = None
        self.on_step = []
        self.states = []                                 # [(sim t, state)] from Course.enter
        self._truth_f = self._truth_w = None
        self._truth_i = 0
        if not self.headless:
            try:
                cv2.namedWindow(viewer.window, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(viewer.window, self._mouse)
                self._window_ok = True
            except Exception as e:                       # no display
                print("[sim] cannot open window (%s): running headless" % e)
                self.headless = True

    @property
    def t(self):
        return self.clock.t

    # ---- flight log hooks -------------------------------------------------------
    def attach_blackbox(self, bb, course):
        self.bb = bb
        self.course = course
        self.drone.on_collision = self._collision
        self._truth_i = len(self.drone.truth)
        path = bb.flog.path("truth.csv")
        if path is None:
            return
        try:
            self._truth_f = open(path, "w", newline="", encoding="utf-8")
            self._truth_w = csv.writer(self._truth_f, lineterminator="\n")
            self._truth_w.writerow(TRUTH_COLUMNS)
            bb.event("sim", world=self.drone.world.name, truth="truth.csv",
                     note="truth.csv: x east, y north, z up (cm, drone bottom), yaw clockwise from east (deg)")
        except Exception as e:
            print("[sim] truth.csv disabled (%s)" % e)
            self._truth_f = self._truth_w = None

    def on_enter(self, state, now):
        self.states.append((now, state))

    def _collision(self, t, name, pos, speed):
        if self.bb is not None:
            self.bb.event("collision", obj=name, x=round(float(pos[0]), 1), y=round(float(pos[1]), 1),
                          z=round(float(pos[2]), 1), speed=round(float(speed), 1),
                          state=self.course.state if self.course is not None else None)

    def _after_step(self):
        if self._truth_w is not None:
            rows = self.drone.truth
            if self._truth_i > len(rows):                # drone.reset() cleared it
                self._truth_i = 0
            t0 = self.bb.flog.t0
            try:
                for r in rows[self._truth_i:]:
                    self._truth_w.writerow(["%.3f" % (r[0] - t0)] + ["%.1f" % v for v in r[1:10]] +
                                           [r[10], "%.1f" % r[11], "%.1f" % r[12], r[13]])
            except Exception as e:
                print("[sim] truth.csv disabled (%s)" % e)
                self._truth_w = None
            self._truth_i = len(rows)
        for fn in self.on_step:
            fn(self)

    # ---- window -------------------------------------------------------------------
    def show(self, image):
        self.last_image = image

    def display_scale(self, img):
        return fit_scale(img.shape[1], img.shape[0]) if self.scale is None else self.scale

    def screenshot(self, path, displayed=False):
        img = self.viewer.compose(self.last_image)
        if displayed:
            img = shrink(img, self.display_scale(img))
        cv2.imwrite(path, img)
        return img

    def _mouse(self, event, x, y, flags, param=None):
        s = getattr(self, "_shown_scale", 1.0) or 1.0
        self.viewer.on_mouse(event, int(x / s), int(y / s), flags, param)

    def wait_key(self, ms=1):
        self.n_calls += 1
        dt = max(self.loop_dt, ms / 1000.0)
        if self.headless:
            self.drone.step(dt)
            self._after_step()
            k = self.script(self) if self.script is not None else -1
            return -1 if k is None else k
        # interactive
        if not self.paused or self.step_once:
            self.drone.step(dt)
            self._after_step()
            self.step_once = False
        now = _time.time()
        if self._wall_next is None or now - self._wall_next > 0.25:
            self._wall_next = now
        if not self.paused:
            self._wall_next += dt / self.speed
        if now >= self._next_draw:                      # window refresh ~30 fps (wall clock)
            self._next_draw = max(self._next_draw + 1.0 / 30.0, now - 0.05)
            self._draw()
        wait = 30 if self.paused else max(1, int((self._wall_next - _time.time()) * 1000))
        k = cv2.waitKeyEx(wait)
        if self._window_ok:
            try:
                if cv2.getWindowProperty(self.viewer.window, cv2.WND_PROP_VISIBLE) < 1:
                    return 27                                   # window closed -> ESC
            except Exception:
                pass
        if k == -1:
            k = self.script(self) if self.script is not None else -1
            return -1 if k is None else k
        return self._handle(k)

    def _draw(self):
        t0 = _time.time()
        v = self.viewer
        v.status = ["sim speed x%g%s" % (self.speed, "   PAUSED" if self.paused else "")]
        img = v.compose(self.last_image)
        s = self.display_scale(img)
        self._shown_scale = s
        cv2.imshow(v.window, shrink(img, s))
        self._last_draw = _time.time()
        self.n_draw += 1
        self.draw_ms.append((self._last_draw - t0) * 1000.0)
        if len(self.draw_ms) > 300:
            del self.draw_ms[:150]

    def _handle(self, k):
        """sim keys are consumed (-1); everything else goes to the flight program"""
        v = self.viewer
        h = self.drone.human
        if k in ARROWS:
            if h is not None:
                h.manual = True
                d = {"left": (-10, 0), "right": (10, 0), "up": (0, 10), "down": (0, -10)}[ARROWS[k]]
                h.move_manual(*d)
            return -1
        if k > 255:                                    # modifier bits / other special keys
            low = k & 0xFF
            if 32 <= low < 127 or low == 27:
                k = low
            else:
                return -1
        c = chr(k) if 32 <= k < 127 else ""
        if c in ("p", " "):
            self.paused = not self.paused
        elif c == "n":
            self.paused = True
            self.step_once = True
        elif c == "[":
            i = SPEEDS.index(self.speed) if self.speed in SPEEDS else 1
            self.speed = SPEEDS[max(0, i - 1)]
        elif c == "]":
            i = SPEEDS.index(self.speed) if self.speed in SPEEDS else 1
            self.speed = SPEEDS[min(len(SPEEDS) - 1, i + 1)]
        elif c == "r":
            self.drone.reset()
            print("[sim] reset")
        elif c in ("t", "y", "u", "i"):
            v.set_view({"t": "top", "y": "side", "u": "follow", "i": "overview"}[c])
        elif c in ("=", "+"):
            v.zoom(0.85)
        elif c == "-":
            v.zoom(1.0 / 0.85)
        elif c == "m" and h is not None:
            h.manual = not h.manual
        elif c == "b" and h is not None:
            h.show_marker = not h.show_marker
        else:
            return k
        if self.paused:
            self._draw()
        return -1

    def close(self):
        if self._truth_f is not None:
            try:
                self._truth_f.close()
            except Exception:
                pass
            self._truth_f = self._truth_w = None
        if not self.headless:
            try:
                cv2.destroyWindow(self.viewer.window)
            except Exception:
                pass
