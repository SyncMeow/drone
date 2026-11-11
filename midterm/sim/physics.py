"""
physics.py - Tello point-mass model with a thrust vector (config/sim_physics.json).

State (world frame, cm / s / deg): pos (x, y, z = bottom of the drone, TOF sensor height), vel,
heading psi (0 = east, clockwise positive, like Tello yaw), yaw rate, tilt (tan of the body tilt
as a horizontal world vector; horizontal acceleration = g * tilt).

Inner loop: rc -> target velocity (body frame) -> desired acceleration (P on velocity + drag
feed-forward) -> commanded tilt (limited to max_tilt) -> actual tilt (first-order lag) -> acceleration.
Vertical: target vz -> acceleration limited by thrust headroom. Gravity always acts; with motors
off the drone falls. Fixed time step (physics_dt_s).
"""
import copy
import math
import os

import numpy as np

from world import LAB6, Box, load_jsonc

DEFAULT_PHYSICS = os.path.join(LAB6, "config", "sim_physics.json")


class Params(object):
    """flat attribute view of the grouped JSON: p.control["max_tilt_deg"] -> p.max_tilt_deg"""

    def __init__(self, groups):
        self.groups = groups
        for g, vals in groups.items():
            if g == "presets" or not isinstance(vals, dict):
                continue
            for k, v in vals.items():
                setattr(self, k, v)


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def load_physics(path=None, ideal=False, overrides=None):
    groups = load_jsonc(path or DEFAULT_PHYSICS)
    if ideal:
        _merge(groups, groups.get("presets", {}).get("ideal", {}))
    if overrides:
        _merge(groups, overrides)
    return Params(groups)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def body_axes(psi, pitch=0.0, roll=0.0):
    """unit forward / right / up vectors of the body.
    psi: heading (0 = east, clockwise); pitch: nose down positive; roll: right side down positive (deg)"""
    a = math.radians(psi)
    f = np.array([math.cos(a), -math.sin(a), 0.0])
    r = np.array([-math.sin(a), -math.cos(a), 0.0])
    u = np.array([0.0, 0.0, 1.0])
    if pitch:
        p = math.radians(pitch)
        f, u = f * math.cos(p) - u * math.sin(p), u * math.cos(p) + f * math.sin(p)
    if roll:
        q = math.radians(roll)
        r, u = r * math.cos(q) - u * math.sin(q), u * math.cos(q) + r * math.sin(q)
    return f, r, u


class Body(object):
    def __init__(self, P, pos, heading, rng):
        self.P = P
        self.rng = rng
        self.g = float(P.gravity_cm_s2)
        self.k_drag = float(P.drag_kg_s) / float(P.mass_kg)       # 1/s
        self.half = np.array(P.half_size_cm, float)
        self.reset(pos, heading)

    def reset(self, pos, heading):
        self.pos = np.array(pos, float)
        self.vel = np.zeros(3)
        self.psi = float(heading)
        self.yaw_rate = 0.0
        self.tilt = np.zeros(2)            # tan(tilt) as world (x, y) vector
        self.motors = False
        self.mode = "off"                  # off / takeoff / rc / land
        self.gust = np.zeros(2)
        self.on_surface = True

    # attitude (for camera + telemetry)
    def pitch_roll(self):
        f, r, _ = body_axes(self.psi)
        pitch = math.degrees(math.atan(float(self.tilt @ f[:2])))   # accelerating forward -> nose down
        roll = math.degrees(math.atan(float(self.tilt @ r[:2])))    # accelerating right -> right side down
        return pitch, roll

    def step(self, dt, rc, world, extra_boxes=()):
        """advance dt with rc = (lr, fb, ud, yaw) already delayed; returns list of new collision boxes"""
        P = self.P
        lr, fb, ud, yw = rc
        f, r, _ = body_axes(self.psi)
        vt = (fb * f[:2] + lr * r[:2]) * float(P.rc_to_cm_s_horizontal)
        vzt = ud * float(P.rc_to_cm_s_vertical)
        yrt = yw * float(P.rc_to_deg_s_yaw)
        if self.mode == "takeoff":
            vt, yrt = np.zeros(2), 0.0
            err = float(P.takeoff_height_cm) - world.tof_below(self.pos)
            vzt = min(float(P.takeoff_speed_cm_s), max(0.0, 2.0 * err))   # slow down near the target
            if err <= 1.0 and abs(self.vel[2]) < 5.0:
                self.mode = "rc"
                vzt = 0.0
        elif self.mode == "rc":
            tof = self.pos[2] - world.support_z(self.pos[0], self.pos[1], self.pos[2] + 0.5)
            lim = float(P.min_hover_tof_cm)
            if tof < lim + 5.0 and vzt < 0:          # Tello will not descend onto the ground with rc
                vzt = max(vzt, (tof - lim) * 2.0)
        elif self.mode == "land":
            vt, yrt = np.zeros(2), 0.0
            vzt = -float(P.land_speed_cm_s)
        elif self.mode == "off":
            vt, vzt, yrt = np.zeros(2), 0.0, 0.0

        # disturbances
        gs = float(P.gust_std_cm_s2)
        if gs > 0 and self.motors:
            tau = max(float(P.gust_tau_s), 1e-3)
            self.gust += (-self.gust * dt / tau +
                          gs * math.sqrt(2.0 * dt / tau) * self.rng.standard_normal(2))
        dist_a = np.array(P.wind_cm_s2, float) + self.gust if self.motors else np.zeros(2)

        if self.motors and P.kinematic:
            self.vel[:2] = vt
            self.vel[2] = vzt
            self.yaw_rate = yrt
            self.tilt[:] = 0.0
        elif self.motors:
            # horizontal: velocity loop -> tilt -> acceleration
            a_des = float(P.vel_loop_gain_1_s) * (vt - self.vel[:2]) + self.k_drag * vt
            tcmd = a_des / self.g
            tmax = math.tan(math.radians(float(P.max_tilt_deg)))
            n = float(np.hypot(*tcmd))
            if n > tmax:
                tcmd *= tmax / n
            self.tilt += (tcmd - self.tilt) * min(1.0, dt / max(float(P.tilt_tau_s), 1e-3))
            a_xy = self.g * self.tilt - self.k_drag * self.vel[:2] + dist_a
            # vertical: thrust headroom limits
            az = (vzt - self.vel[2]) / max(float(P.vz_tau_s), 1e-3)
            az = min(max(az, -float(P.max_accel_down_cm_s2)), float(P.max_accel_up_cm_s2))
            self.vel[:2] += a_xy * dt
            self.vel[2] += az * dt
            self.yaw_rate += (yrt - self.yaw_rate) * min(1.0, dt / max(float(P.yaw_tau_s), 1e-3))
        else:
            self.tilt *= max(0.0, 1.0 - dt / 0.3)
            self.vel[:2] -= self.k_drag * self.vel[:2] * dt
            self.vel[2] -= (self.g + self.k_drag * self.vel[2]) * dt
            self.yaw_rate = 0.0
            if self.on_surface:
                self.vel[:2] *= max(0.0, 1.0 - 8.0 * dt)       # friction on the ground
        if self.motors:
            self.psi = wrap180(self.psi + (self.yaw_rate + float(P.yaw_drift_deg_s)) * dt)

        new_pos = self.pos + self.vel * dt
        hits = self._resolve(new_pos, world, extra_boxes)
        return hits

    def _resolve(self, new_pos, world, extra_boxes):
        """move to new_pos, handle ground / surfaces / obstacles. Returns crash boxes."""
        P = self.P
        h = self.half
        hits = []
        # surfaces below (floor / table tops): landing contact
        sup = world.support_z(new_pos[0], new_pos[1], self.pos[2] + 0.5, half=h[0] * 0.5)
        self.on_surface = False
        if new_pos[2] <= sup:
            if self.vel[2] < -float(P.land_contact_speed_cm_s) and self.motors:
                hits.append(Box("floor" if sup <= 0 else "surface", [0, 0, 0], [0, 0, 0]))
            new_pos[2] = sup
            if self.vel[2] < 0:
                self.vel[2] = 0.0
            self.on_surface = True
            if self.mode == "land":
                self.vel[:2] = 0.0
        # obstacles: body AABB (centre at pos + half z)
        for _ in range(3):
            c = new_pos + np.array([0.0, 0.0, h[2]])
            boxes = world.collide(c - h, c + h, extra_boxes)
            if not boxes:
                break
            b = boxes[0]
            lo, hi = c - h, c + h
            pen_neg = hi - b.lo        # push toward -axis by this
            pen_pos = b.hi - lo        # push toward +axis by this
            best = None
            for ax in range(3):
                for sgn, pen in ((-1, pen_neg[ax]), (1, pen_pos[ax])):
                    if best is None or pen < best[0]:
                        best = (pen, ax, sgn)
            pen, ax, sgn = best
            if ax == 2 and sgn > 0 and self.vel[2] >= -float(P.land_contact_speed_cm_s):
                # resting on top of a box (e.g. landing on the table): not a crash
                new_pos[2] += pen + 1e-6
                self.vel[2] = max(self.vel[2], 0.0)
                self.on_surface = True
                continue
            new_pos[ax] += sgn * (pen + 1e-6)
            if self.vel[ax] * sgn < 0:
                self.vel[ax] = 0.0
            hits.append(b)
        self.pos = new_pos
        return hits
