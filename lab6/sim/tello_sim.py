"""
tello_sim.py - SimTello: a stand-in for djitellopy.Tello driven by sim/physics.py.

    clock = SimClock()
    drone = SimTello(World(), load_physics(), clock)
    drone.step(0.02)            # advances physics, human, video, and clock.t

Implements the subset the flight code uses (connect, streamon/off, get_frame_read().frame (RGB,
960x720, new array only for a new video frame, delayed by the video latency), get_current_state(),
get_battery / get_yaw / get_height / get_distance_tof, takeoff / land (blocking: sim time advances),
send_rc_control, rotate_clockwise, is_flying) and harmless no-ops for the rest.
Yaw = Tello sense: clockwise (seen from above) positive, 0 = heading at start.
Like djitellopy, is_flying only changes with takeoff() / land(): when the drone lands by itself
(15 s without commands) or falls after a crash, is_flying stays True while h / tof read ~0.

Ground truth: pos (x, y, z bottom, cm), vel, heading, pitch, roll, collisions (events),
colliding (this step), trace [(t, x, y, z, heading, pitch, roll, vx, vy, vz, lr, fb, ud, yaw, tof, hit)].
"""
import collections
import math
import os
import time as _time

import cv2
import numpy as np

from physics import Body, body_axes, wrap180
from render import Renderer, drone_camera
from world import LAB6


def load_calib(path=None):
    path = path or os.path.join(LAB6, "calibrate.xml")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError("cannot read intrinsic/distortion from " + path)
    return K, dist


class SimClock(object):
    """time.time() / time.sleep() replacement. sleep() advances the simulation (physics included)."""

    def __init__(self, t0=1000.0):
        self.t = float(t0)
        self.stepper = None               # SimTello.step, set by SimTello

    def time(self):
        return self.t

    monotonic = perf_counter = time

    def sleep(self, s):
        if s <= 0:
            return
        if self.stepper is not None:
            self.stepper(s)
        else:
            self.t += s

    def __getattr__(self, name):          # strftime, localtime, ... from the real time module
        return getattr(_time, name)


class FrameRead(object):
    def __init__(self, drone):
        self._drone = drone
        self.stopped = False

    @property
    def frame(self):
        return self._drone._deliver_frame()

    def stop(self):
        self.stopped = True


class SimTello(object):
    W, H = 960, 720

    def __init__(self, world, physics, clock, human=None, calib=None, seed=None):
        self.world = world
        self.P = physics
        self.clock = clock
        self.human = human
        clock.stepper = self.step
        seed = int(physics.seed) if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.K, self.dist = calib if calib is not None else load_calib()
        self.renderer = Renderer(world, self.K, (self.W, self.H), self.dist)
        self.body = Body(physics, world.start_pos, world.start_heading, self.rng)
        self.pdt = float(physics.physics_dt_s)
        self.is_flying = False
        self.stream_on = False
        self._frame_read = None
        self.on_collision = None          # callback(t, name, pos, speed): e.g. write an event to the flight log
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        """back to the start pose on the ground; clears trace / collisions (clock keeps running)"""
        P = self.P
        self.body.reset(self.world.start_pos, self.world.start_heading)
        self.t0 = self.clock.t
        self.is_flying = False
        self._acc = 0.0
        self._rc = (0, 0, 0, 0)                       # rc acting now (after latency)
        self._rc_queue = collections.deque()
        self._override = None                         # rc used by blocking moves (rotate_*, move_*)
        self.last_cmd_t = self.clock.t
        self.auto_landed = False
        self.flight_time = 0.0
        self.battery = float(P.battery_start_pct)
        self.temph = float(P.temp_start_c)
        self.collisions = []                          # [(t, name, pos, speed)]
        self.colliding = False
        self.hit_names = set()
        self._contact = set()
        self.hit_until = 0.0
        self.trace = []
        self.truth = []                               # [(t, x, y, z, yaw, pitch, roll, vx, vy, vz, hit, hx, hy, m0)]
        self._next_trace = self.clock.t
        self._next_cap = self.clock.t
        self._pending = collections.deque()           # (t_cap, cam_pos, cam_R, dyn objects)
        self._shown_cap = None
        self._frame = None
        self.frames_captured = self.frames_dropped = self.frames_rendered = 0
        self.ground_z = 0.0
        if self.human is not None:
            self.human.reset()

    # ------------------------------------------------------------------ ground truth
    @property
    def pos(self):
        return self.body.pos

    @property
    def vel(self):
        return self.body.vel

    @property
    def heading(self):
        return self.body.psi

    @property
    def pitch(self):
        return self.body.pitch_roll()[0]

    @property
    def roll(self):
        return self.body.pitch_roll()[1]

    def camera_pose(self):
        pitch, roll = self.body.pitch_roll()
        return drone_camera(self.body.pos, self.body.psi, pitch, roll, float(self.P.camera_pitch_deg),
                            float(self.P.camera_forward_cm))

    def marker0_in_view(self):
        """ground truth: the human shows marker 0, it faces the camera and lies fully inside the image
        (occlusion by boxes is not checked)"""
        h = self.human
        if h is None or not h.show_marker:
            return False
        cp, cR = self.camera_pose()
        c = h.marker_center()
        pc = cR @ (c - cp)
        if pc[2] < 20.0:
            return False
        a = math.radians(h.heading)
        n = np.array([math.cos(a), -math.sin(a), 0.0])          # marker normal (toward where the human faces)
        if float(n @ (cp - c)) <= 0:
            return False
        r = h.marker_size / 2.0
        u = self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2]
        v = self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
        m = self.K[0, 0] * r / pc[2]
        return m < u < self.W - m and m < v < self.H - m

    def true_tof(self):
        return self.world.tof_below(self.body.pos)

    # ------------------------------------------------------------------ simulation
    def step(self, dt):
        """advance the simulation (and the clock) by dt seconds in fixed physics steps"""
        self._acc += dt
        n = int((self._acc + 1e-9) / self.pdt)
        self._acc -= n * self.pdt
        for _ in range(n):
            self._substep(self.pdt)
        if self.human is not None and n:
            cp, cR = self.camera_pose()
            self.human.update(n * self.pdt, cp, cR[2])

    def _substep(self, dt):
        P = self.P
        b = self.body
        t = self.clock.t + dt
        self.clock.t = t
        # command latency
        while self._rc_queue and self._rc_queue[0][0] <= t:
            self._rc = self._rc_queue.popleft()[1]
        rc = self._override if self._override is not None else self._rc
        # Tello lands by itself without commands
        if (b.motors and b.mode == "rc" and P.auto_land_no_command_s
                and t - self.last_cmd_t > float(P.auto_land_no_command_s)):
            b.mode = "land"
            self.auto_landed = True
            print("[sim] no command for %.0f s: Tello lands by itself" % float(P.auto_land_no_command_s))
        extra = self.human.colliders() if self.human is not None else ()
        hits = b.step(dt, rc, self.world, extra)
        if b.mode == "land" and b.on_surface:
            b.motors = False
            b.mode = "off"
        # collisions
        names = set(h.name for h in hits)
        self.colliding = bool(hits)
        for h in hits:
            if h.name not in self._contact:
                ev = (t, h.name, b.pos.copy(), float(np.linalg.norm(b.vel)))
                self.collisions.append(ev)
                print("[sim] COLLISION with %s at (%.0f, %.0f, %.0f) t=%.2f" % (h.name, b.pos[0], b.pos[1], b.pos[2], t - self.t0))
                if self.on_collision is not None:
                    try:
                        self.on_collision(t, h.name, b.pos.copy(), ev[3])
                    except Exception:
                        pass
                if P.behaviour == "fall" and h.name not in ("floor", "surface"):
                    b.motors = False
                    b.mode = "off"
        if hits:
            self.hit_names |= names
            self.hit_until = t + 1.0
        elif t > self.hit_until:
            self.hit_names = set()
        self._contact = names
        # power / temperature
        flying = b.motors
        drain = float(P.drain_flying_pct_min if flying else P.drain_idle_pct_min)
        self.battery = max(0.0, self.battery - drain * dt / 60.0)
        eq = float(P.temp_max_c) - (float(P.temp_flying_cool_c) if flying else 0.0)
        self.temph += (eq - self.temph) * dt / float(P.temp_tau_s)
        if flying:
            self.flight_time += dt
        # video capture
        if t >= self._next_cap:
            self._next_cap += 1.0 / float(P.fps)
            if self._next_cap < t:
                self._next_cap = t + 1.0 / float(P.fps)
            if self.stream_on:
                if self.rng.random() < float(P.drop_prob):
                    self.frames_dropped += 1
                else:
                    cp, cR = self.camera_pose()
                    dyn = self.human.objects() if self.human is not None else ()
                    self._pending.append((t, cp, cR, dyn))
                    self.frames_captured += 1
                    while len(self._pending) > 64:
                        self._pending.popleft()
        # trace
        if t >= self._next_trace:
            self._next_trace = t + float(P.trace_dt_s)
            pr = b.pitch_roll()
            self.trace.append((t, b.pos[0], b.pos[1], b.pos[2], b.psi, pr[0], pr[1],
                               b.vel[0], b.vel[1], b.vel[2]) + tuple(rc) +
                              (self.true_tof(), self.colliding))
            hp = self.human.pos if self.human is not None else (float("nan"), float("nan"))
            self.truth.append((t, b.pos[0], b.pos[1], b.pos[2], b.psi, pr[0], pr[1], b.vel[0], b.vel[1], b.vel[2],
                               int(self.colliding), hp[0], hp[1], int(self.marker0_in_view())))

    def _deliver_frame(self):
        if not self.stream_on:
            return self._frame
        now = self.clock.t
        lat = float(self.P.latency_s)
        snap = None
        while self._pending and self._pending[0][0] + lat <= now + 1e-9:
            snap = self._pending.popleft()
        if snap is not None:
            t, cp, cR, dyn = snap
            bgr = self.renderer.render(cp, cR, dyn, float(self.P.corner_noise_px), self.rng)
            self._frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._shown_cap = t
            self.frames_rendered += 1
        return self._frame

    def _blocking(self, until, max_time):
        t0 = self.clock.t
        while not until() and self.clock.t - t0 < max_time:
            self.step(0.02)

    # ------------------------------------------------------------------ djitellopy API
    def connect(self, wait_for_state=True):
        self.last_cmd_t = self.clock.t

    def streamon(self):
        self.stream_on = True

    def streamoff(self):
        self.stream_on = False

    def end(self):
        if self.is_flying:
            self.land()
        self.streamoff()

    def get_frame_read(self, *a, **k):
        if self._frame_read is None:
            self._frame_read = FrameRead(self)
        return self._frame_read

    def send_rc_control(self, left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity):
        rc = tuple(int(max(-100, min(100, round(v)))) for v in
                   (left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity))
        self.last_cmd_t = self.clock.t
        lat = float(self.P.command_latency_s)
        if lat <= 0:
            self._rc = rc
        else:
            self._rc_queue.append((self.clock.t + lat, rc))

    def send_keepalive(self):
        self.last_cmd_t = self.clock.t

    def takeoff(self):
        b = self.body
        self.last_cmd_t = self.clock.t
        self._rc = (0, 0, 0, 0)
        self._rc_queue.clear()
        if not b.motors:
            b.motors = True
            b.mode = "takeoff"
            self.ground_z = self.world.support_z(b.pos[0], b.pos[1], b.pos[2] + 0.5)
            if self.P.kinematic:
                b.pos[2] = self.ground_z + float(self.P.takeoff_height_cm)
                b.mode = "rc"
        self.is_flying = True
        self.auto_landed = False
        if self.P.blocking_takeoff_land:
            self._blocking(lambda: b.mode != "takeoff", 8.0)

    def land(self):
        b = self.body
        self.last_cmd_t = self.clock.t
        self._rc = (0, 0, 0, 0)
        self._rc_queue.clear()
        if b.motors:
            b.mode = "land"
            if self.P.kinematic:
                b.pos[2] = self.world.support_z(b.pos[0], b.pos[1], b.pos[2] + 0.5)
                b.vel[:] = 0.0
                b.motors = False
                b.mode = "off"
            elif self.P.blocking_takeoff_land:
                self._blocking(lambda: not b.motors, 10.0)
        self.is_flying = False

    def emergency(self):
        self.body.motors = False
        self.body.mode = "off"
        self.is_flying = False

    def _rotate(self, deg):
        b = self.body
        self.last_cmd_t = self.clock.t
        if not b.motors:
            return
        start = b.psi
        sgn = 1 if deg > 0 else -1
        self._override = (0, 0, 0, 60 * sgn)
        self._blocking(lambda: abs(wrap180(b.psi - start)) >= min(abs(deg), 179.5) - 3.0
                       if abs(deg) < 360 else False, abs(deg) / 50.0 + 2.0)
        self._override = (0, 0, 0, 0)
        self._blocking(lambda: abs(b.yaw_rate) < 3.0, 1.0)
        self._override = None

    def rotate_clockwise(self, x):
        self._rotate(abs(x))

    def rotate_counter_clockwise(self, x):
        self._rotate(-abs(x))

    def _move(self, lr, fb, ud, dist):
        b = self.body
        self.last_cmd_t = self.clock.t
        if not b.motors:
            return
        p0 = b.pos.copy()
        self._override = (lr, fb, ud, 0)
        self._blocking(lambda: np.linalg.norm(b.pos - p0) >= dist - 2.0, dist / 30.0 + 2.0)
        self._override = (0, 0, 0, 0)
        self._blocking(lambda: np.linalg.norm(b.vel) < 3.0, 1.5)
        self._override = None

    def move_forward(self, x): self._move(0, 50, 0, x)
    def move_back(self, x): self._move(0, -50, 0, x)
    def move_left(self, x): self._move(-50, 0, 0, x)
    def move_right(self, x): self._move(50, 0, 0, x)
    def move_up(self, x): self._move(0, 0, 50, x)
    def move_down(self, x): self._move(0, 0, -50, x)

    # ---- telemetry (state packet)
    def get_current_state(self):
        P = self.P
        b = self.body
        rng = self.rng
        pitch, roll = b.pitch_roll()
        yaw = wrap180(b.psi - self.world.start_heading + rng.normal(0, float(P.yaw_noise_deg) or 1e-12))
        tof_true = self.true_tof()
        if b.motors or tof_true > 1.0:
            tof = max(float(P.ground_tof_cm), tof_true + rng.normal(0, float(P.tof_noise_cm) or 1e-12))
        else:
            tof = float(P.ground_tof_cm)
        h = 0.0
        if b.motors:
            h = b.pos[2] - self.ground_z + rng.normal(0, float(P.baro_noise_cm) or 1e-12)
        f, r, _ = body_axes(b.psi)
        vgx = float(b.vel[:2] @ f[:2]) / 10.0
        vgy = float(b.vel[:2] @ r[:2]) / 10.0
        return {"mid": -1, "x": -100, "y": -100, "z": -100,
                "pitch": int(round(pitch)), "roll": int(round(roll)), "yaw": int(round(yaw)),
                "vgx": int(round(vgx)), "vgy": int(round(vgy)), "vgz": int(round(-b.vel[2] / 10.0)),
                "templ": int(self.temph - 2), "temph": int(self.temph),
                "tof": int(round(tof)), "h": int(round(h)), "bat": int(self.battery),
                "baro": round(b.pos[2] / 100.0, 2), "time": int(self.flight_time),
                "agx": 0.0, "agy": 0.0, "agz": -1000.0}

    def get_state_field(self, key):
        return self.get_current_state()[key]

    def get_battery(self): return self.get_state_field("bat")
    def get_yaw(self): return self.get_state_field("yaw")
    def get_pitch(self): return self.get_state_field("pitch")
    def get_roll(self): return self.get_state_field("roll")
    def get_height(self): return self.get_state_field("h")
    def get_distance_tof(self): return self.get_state_field("tof")
    def get_flight_time(self): return self.get_state_field("time")
    def get_barometer(self): return self.get_state_field("baro")
    def get_speed_x(self): return self.get_state_field("vgx")
    def get_speed_y(self): return self.get_state_field("vgy")
    def get_speed_z(self): return self.get_state_field("vgz")
    def get_lowest_temperature(self): return self.get_state_field("templ")
    def get_highest_temperature(self): return self.get_state_field("temph")

    def get_temperature(self):
        s = self.get_current_state()
        return (s["templ"] + s["temph"]) / 2.0

    query_battery = get_battery
    query_height = get_height
    query_distance_tof = get_distance_tof

    def __getattr__(self, name):
        """anything else (set_speed, set_video_*, flip_*, ...): harmless no-op"""
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None
