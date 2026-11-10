"""
run_sim.py - the real flight program (core.app.run: same loop, same HUD, same flight log) inside the 3D simulator.

    python sim/run_sim.py                    # window: HUD canvas (FPV + panel + messages) | 3D course view
                                             #   press g to fly the whole course, h j k l for segments 2-5
    python sim/run_sim.py --ideal            # no inertia / latency / noise (like the old kinematic sim)
    python sim/run_sim.py --scale 0.6        # display scale (default: auto, fits the screen; recordings stay full size)
    python sim/run_sim.py --no-record        # no fpv_raw.mp4 / fpv_hud.mp4 (logs + truth.csv are still written)
    python sim/run_sim.py --no-log           # no logs/..._sim directory at all
    python sim/run_sim.py --keys "1:g"       # scripted keys "sim_seconds:key,..." (e.g. headless runs)
    python sim/run_sim.py --headless --keys "1:g" --quit-after 200
    python sim/run_sim.py --manual-only      # small manual-flight demo (no course program)
    python sim/run_sim.py --physics my.json --world my_world.json --seed 3

Every run writes logs/YYYYmmdd_HHMMSS_sim/ through core/blackbox.py (flight.log, telemetry.csv, events.jsonl,
params.json, fpv_raw.mp4, fpv_hud.mp4) plus truth.csv (ground truth) and "collision" events.

make_sim() creates everything the flight program needs: drone (SimTello), clock (SimClock),
io (SimIO: wait_key advances the simulation, show composes the program image with the 3D view).
"""
import argparse
import os
import sys
import time as _time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LAB6 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, LAB6)

from human import Human                          # noqa: E402
from physics import load_physics                 # noqa: E402
from tello_sim import SimClock, SimTello         # noqa: E402
from viewer import SimIO, Viewer                 # noqa: E402
from world import World                          # noqa: E402
from core.display import parse_scale             # noqa: E402

KEY_HOLD = 0.2          # manual key: stop this long after the last key press (as main.py)
KEEPALIVE = 0.5         # send a hover command when nothing was sent for this long


def make_sim(world_path=None, physics_path=None, ideal=False, headless=False, script=None, seed=None, scale=None,
             overrides=None):
    world = World(world_path)
    phys = load_physics(physics_path, ideal=ideal, overrides=overrides)
    clock = SimClock()
    human = Human(world.human_spec) if world.human_spec else None
    drone = SimTello(world, phys, clock, human, seed=seed)
    viewer = None if headless else Viewer(drone)
    io = SimIO(drone, viewer, headless=headless, script=script, scale=scale)
    return drone, clock, io


def key_script(spec):
    """"1:g,30.5:j" -> script(io): returns the key once when the sim time passes each mark"""
    marks = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        t, k = part.split(":", 1)
        k = {"esc": 27, "space": 32}.get(k.lower(), None) if len(k) > 1 else ord(k)
        if k is None:
            raise ValueError("bad key in --keys: %r" % part)
        marks.append([float(t), k])
    marks.sort()

    def script(io):
        t = io.t - io.drone.t0
        if marks and t >= marks[0][0]:
            return marks.pop(0)[1]
        return -1
    return script


def manual_demo(drone, clock, io):
    """small stand-in for the flight program: course keyboard + marker detection"""
    from keyboard_djitellopy import keyboard
    from core.vision import detect_markers, make_detector, marker_obj_points
    detect = make_detector()
    obj_pts = marker_obj_points()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    key_expire = 0.0
    last_sent = clock.time()
    last_raw = None
    seen = {}
    try:
        while True:
            now = clock.time()
            if drone.is_flying and not key_expire and now - last_sent >= KEEPALIVE:
                drone.send_rc_control(0, 0, 0, 0)
                last_sent = now
            key = io.wait_key(1)
            if key != -1:
                key &= 0xFF
            if key in (27, ord('q')):
                break
            if key != -1:
                keyboard(drone, key)
                last_sent = clock.time()
                if key in (ord('1'), ord('2'), ord('3')):
                    key_expire = 0.0
                elif key not in (ord('5'), ord('6')):
                    key_expire = clock.time() + KEY_HOLD
            if key_expire and clock.time() >= key_expire:
                drone.send_rc_control(0, 0, 0, 0)
                last_sent = clock.time()
                key_expire = 0.0
            raw = frame_read.frame
            if raw is None or raw is last_raw:
                continue
            last_raw = raw
            frame = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            found = detect_markers(frame, detect, obj_pts, drone.K, drone.dist)
            for mid in found:
                seen[mid] = seen.get(mid, 0) + 1
            st = drone.get_current_state()
            cv2.putText(frame, "MANUAL DEMO  %s  bat %d%%  tof %d  yaw %d" % (
                "FLYING" if drone.is_flying else "ON GROUND", st["bat"], st["tof"], st["yaw"]),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "1 takeoff  2 land  w s a d move  z x up/down  c v turn  q quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            io.show(frame)
    finally:
        drone.send_rc_control(0, 0, 0, 0)
        if drone.is_flying:
            drone.land()
        drone.streamoff()
    return seen


def demo_script(io):
    """headless manual demo keys by sim time: take off, fly toward marker 1, turn a bit, land, quit"""
    t = io.t - io.drone.t0
    i = io.n_calls
    if 1.0 <= t < 1.02:
        return ord('1')
    if 2.0 <= t < 4.0 and i % 5 == 0:
        return ord('w')
    if 5.0 <= t < 5.6 and i % 5 == 0:
        return ord('c')
    if 7.0 <= t < 7.02:
        return ord('2')
    if t >= 8.0:
        return ord('q')
    return -1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tello course simulator running the real flight program")
    ap.add_argument("--headless", action="store_true", help="no window, run as fast as possible")
    ap.add_argument("--physics", default=None, help="physics json (default config/sim_physics.json)")
    ap.add_argument("--world", default=None, help="world json (default config/worlds/eab06.json)")
    ap.add_argument("--ideal", action="store_true", help="no inertia / latency / noise")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scale", default="auto", help="window display scale, or auto (fit the screen)")
    ap.add_argument("--keys", default=None, help='scripted keys "sim_seconds:key,...", e.g. "1:g"')
    ap.add_argument("--quit-after", type=float, default=None, help="quit after this many sim seconds")
    ap.add_argument("--no-record", action="store_true", help="no fpv_raw.mp4 / fpv_hud.mp4")
    ap.add_argument("--no-hud-video", action="store_true", help="record only fpv_raw.mp4")
    ap.add_argument("--no-log", action="store_true", help="no logs/..._sim directory (no truth.csv either)")
    ap.add_argument("--manual-only", action="store_true", help="manual flight demo instead of the flight program")
    a = ap.parse_args(argv)
    scale = parse_scale(a.scale)

    script = key_script(a.keys) if a.keys else (demo_script if a.headless and a.manual_only else None)
    if a.quit_after is not None:
        base = script
        script = lambda io: ord('q') if io.t - io.drone.t0 >= a.quit_after else (base(io) if base else -1)
    drone, clock, io = make_sim(a.world, a.physics, a.ideal, a.headless, script, a.seed, scale)
    t_wall = _time.time()
    if a.manual_only:
        seen = manual_demo(drone, clock, io)
        print("[sim] marker detections (frames): %s" % (seen or "none"))
    else:
        from core.app import run                       # the flight program (same code as main.py)
        try:
            run(drone, clock=clock, io=io, log=not a.no_log, record=not (a.no_record or a.no_log),
                hud_video=not a.no_hud_video, tag="sim")
        finally:
            bb = getattr(run, "blackbox", None)
            if bb is not None and bb.flog.dir is not None:
                print("[sim] log directory:", bb.flog.dir)
    io.close()
    if io.draw_ms:
        print("[sim] window: %.1f fps, redraw (3D view + compose + imshow) median %.1f ms"
              % (io.n_draw / max(_time.time() - io._t_open, 1e-6), sorted(io.draw_ms)[len(io.draw_ms) // 2]))
    sim_t = clock.time() - drone.t0
    wall = _time.time() - t_wall
    p = drone.pos
    print("[sim] sim time %.1f s, wall %.1f s (x%.1f), final pos (%.0f, %.0f, %.0f), collisions %d, "
          "frames rendered %d / dropped %d" % (sim_t, wall, sim_t / max(wall, 1e-6), p[0], p[1], p[2],
                                               len(drone.collisions), drone.frames_rendered, drone.frames_dropped))


if __name__ == "__main__":
    main()
