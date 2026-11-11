"""
human.py - dummy human who carries marker 0 for segment 2 (a cooperative person).

wait  : stands at "start" holding marker 0 at hold_height (about the drone's flying height) until the drone is
        within trigger_dist and the marker is inside the drone camera's view.
walk  : walks the "path" (to the chair, around it, then to the blue line) at "speed" (half speed when the drone
        is farther than slow_dist). Pauses (stands still, keeps showing the marker) when the drone lags behind
        (marker-to-camera distance > lag_dist) or the marker is outside the drone camera's view cone
        (view_half_deg) for more than lost_grace s; walks on once the drone is within resume_dist again and has
        had the marker in view for resume_hold s.
end   : at the end of the path waits until the drone has caught up (resume_dist, marker in view) for end_wait s,
        then hides the marker and walks the "leave_path" out of the way (leave).
The body and the marker always turn toward the drone camera.
Manual mode (viewer): arrow keys move the human, the auto script is paused.
"""
import math

import numpy as np

from render import Obj, marker_texture
from world import Box, rgb2bgr


class Human(object):
    def __init__(self, spec):
        self.spec = spec or {}
        s = self.spec
        self.mid = int(s.get("marker_id", 0))
        self.speed = float(s.get("speed", 20.0))
        self.trigger_dist = float(s.get("trigger_dist", 130.0))
        self.slow_dist = float(s.get("slow_dist", 85.0))
        self.lag_dist = float(s.get("lag_dist", s.get("wait_dist", 100.0)))
        self.resume_dist = float(s.get("resume_dist", 80.0))
        self.view_half = float(s.get("view_half_deg", 20.0))
        self.lost_grace = float(s.get("lost_grace", 0.3))
        self.resume_hold = float(s.get("resume_hold", 0.5))
        self.hold_dist = float(s.get("hold_dist", 30.0))
        self.hold_height = float(s.get("hold_height", 100.0))
        self.follow_height = bool(s.get("follow_drone_height", False))
        self.zmin = float(s.get("height_min", 50.0))
        self.zmax = float(s.get("height_max", 200.0))
        self.path = [np.array(p, float) for p in s.get("path", [])]
        self.leave = [np.array(p, float) for p in s.get("leave_path", [])]
        self.end_wait = float(s.get("end_wait", 1.0))
        self.body_color = rgb2bgr(s.get("body_color", [60, 110, 170]))
        self.skin_color = rgb2bgr(s.get("skin_color", [225, 190, 160]))
        self.marker_size = float(s.get("marker_size", 15.0))
        self.margin = float(s.get("marker_margin", 2.5))
        self.manual = False
        self.reset()

    def reset(self):
        self.pos = np.array(self.spec.get("start", [0.0, 0.0]), float)
        self.heading = 180.0
        self.marker_z = self.hold_height
        self.show_marker = True
        self.state = "wait"
        self.route = list(self.path)
        self.t_state = 0.0
        self.cur_speed = 0.0
        self.paused = False
        self.t_out = 0.0                  # s the marker has been out of the drone's view
        self.t_in = 0.0                   # s the marker has been in view
        self.n_pauses = 0

    # ---- geometry ----------------------------------------------------------
    def _dir(self):
        a = math.radians(self.heading)
        return np.array([math.cos(a), -math.sin(a)])

    def marker_center(self):
        d = self._dir()
        p = self.pos + d * self.hold_dist
        return np.array([p[0], p[1], self.marker_z])

    # ---- behaviour -----------------------------------------------------------
    def update(self, dt, cam_pos, cam_fwd):
        cam_pos = np.asarray(cam_pos, float)
        to = cam_pos[:2] - self.pos
        if np.hypot(*to) > 1.0:
            self.heading = math.degrees(math.atan2(-to[1], to[0]))
        if self.follow_height:
            self.marker_z += (min(max(cam_pos[2], self.zmin), self.zmax) - self.marker_z) * min(1.0, dt / 0.4)
        mc = self.marker_center()
        dist = float(np.linalg.norm(mc - cam_pos))
        v = mc - cam_pos
        f = np.asarray(cam_fwd, float)
        cosang = float(v @ f) / max(np.linalg.norm(v) * np.linalg.norm(f), 1e-9)
        if cosang > math.cos(math.radians(self.view_half)):
            self.t_in += dt
            self.t_out = 0.0
        else:
            self.t_out += dt
            self.t_in = 0.0
        self.t_state += dt
        self.cur_speed = 0.0
        if self.manual:
            return
        if self.state == "wait":
            if dist < self.trigger_dist and cosang > math.cos(math.radians(25.0)):
                self._set("walk")
        elif self.state == "walk":
            if self.paused:
                if dist < self.resume_dist and self.t_in >= self.resume_hold:
                    self.paused = False
            elif dist > self.lag_dist or self.t_out > self.lost_grace:
                self.paused = True
                self.n_pauses += 1
            k = 0.0 if self.paused else (0.5 if dist > self.slow_dist else 1.0)
            if self._walk(self.route, self.speed * k, dt):
                self._set("end")
        elif self.state == "end":
            if not (dist < self.resume_dist and self.t_in > 0):
                self.t_state = 0.0                         # drone not there yet: keep waiting
            elif self.t_state >= self.end_wait:
                self.show_marker = False
                self.route = list(self.leave)
                self._set("leave")
        elif self.state == "leave":
            if self._walk(self.route, self.speed * 1.5, dt):
                self._set("gone")

    @property
    def status(self):
        return self.state + (" (waiting for the drone)" if self.state == "walk" and self.paused else "")

    def _set(self, st):
        self.state = st
        self.t_state = 0.0

    def _walk(self, route, v, dt):
        """move along route (consumed in place); True when finished"""
        step = v * dt
        self.cur_speed = v
        while route and step > 0:
            d = route[0] - self.pos
            n = float(np.hypot(*d))
            if n <= step:
                self.pos = route.pop(0).copy()
                step -= n
            else:
                self.pos = self.pos + d / n * step
                step = 0
        return not route

    def move_manual(self, dx, dy):
        self.pos = self.pos + np.array([dx, dy], float)

    # ---- drawing / collision -----------------------------------------------------
    def objects(self):
        a = math.radians(self.heading)
        f = np.array([math.cos(a), -math.sin(a), 0.0])
        r = np.array([-math.sin(a), -math.cos(a), 0.0])
        u = np.array([0.0, 0.0, 1.0])
        Rb = np.stack([f, r, u], axis=1)
        p = np.array([self.pos[0], self.pos[1], 0.0])
        o = Obj("human", multi=True)
        dark = tuple(int(c * 0.6) for c in self.body_color)
        for s in (-1, 1):
            o.add_box([-6, -6, -42], [6, 6, 42], dark, Rb, p + r * 10 * s + u * 42)
        o.add_box([-11, -20, -30], [11, 20, 30], self.body_color, Rb, p + u * 115)
        o.add_box([-10, -10, -11], [10, 10, 11], self.skin_color, Rb, p + u * 158)
        if self.show_marker:
            mc = self.marker_center()
            sh = p + u * 135 + f * 8 + r * 14
            v = mc - f * 1.0 - sh
            L = float(np.linalg.norm(v))
            if L > 1:
                ax = v / L
                b = np.cross(u, ax)
                b = b / np.linalg.norm(b) if np.linalg.norm(b) > 1e-6 else r
                c = np.cross(ax, b)
                o.add_box([-L / 2, -2.5, -2.5], [L / 2, 2.5, 2.5], self.skin_color,
                          np.stack([ax, b, c], axis=1), sh + v / 2)
            h = self.marker_size / 2 + self.margin
            o.add_box([-0.5, -h - 0.5, -h - 0.5], [0.5, h + 0.5, h + 0.5], (235, 235, 235), Rb, mc - f * 0.5)
            # viewer in front looks along -f: viewer's right is -r
            corners = np.array([mc + r * h + u * h, mc - r * h + u * h, mc - r * h - u * h, mc + r * h - u * h])
            o.add_quad(corners + f * 0.3, f, marker_texture(self.mid, self.marker_size, self.margin), bias=2.0)
        return [o.finish()]

    def colliders(self):
        p = self.pos
        out = [Box("human", [p[0] - 18, p[1] - 18, 0], [p[0] + 18, p[1] + 18, 170])]
        if self.show_marker:
            mc = self.marker_center()
            h = self.marker_size / 2 + self.margin
            out.append(Box("human_marker", mc - h, mc + h))
        return out
