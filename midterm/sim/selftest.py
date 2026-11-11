"""
selftest.py - headless checks of the flight simulator. Run: python selftest.py [--quick]
(no display needed; --quick skips the performance test and the screenshots, ~1 s; used by ../../setup.sh)
 1 wall markers detected, solvePnP distance / lateral offset match ground truth
 2 a marker behind a board is not detected (and is detected with the board removed)
 3 physics: take-off + hover altitude, rc forward step response + coast, free fall with motors off
 4 tof over the landing table / floor / under the tunnel table
 5 collision with a board is reported ("stop" and "fall")
 6 performance: physics step, FPV render, 3D render, full viewer frame
Also writes sim/screenshots/overview.png and sim/screenshots/tunnel.png.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))                # lab6/ (core.vision: the flight program's detector)

from human import Human                                  # noqa: E402
from physics import body_axes, load_physics              # noqa: E402
from render import Renderer, drone_camera, look_at                     # noqa: E402
from tello_sim import SimClock, SimTello                 # noqa: E402
from viewer import SimIO, Viewer                         # noqa: E402
from core.vision import detect_markers as _detect, make_detector, marker_obj_points  # noqa: E402
from world import World, heading_vec                     # noqa: E402

QUIET = {"disturbance": {"yaw_drift_deg_s": 0.0, "gust_std_cm_s2": 0.0, "wind_cm_s2": [0, 0]},
         "sensors": {"tof_noise_cm": 0.0, "yaw_noise_deg": 0.0, "baro_noise_cm": 0.0},
         "video": {"latency_s": 0.0, "drop_prob": 0.0, "corner_noise_px": 0.0},
         "control": {"command_latency_s": 0.0}}
RESULTS = []
DETECT = make_detector()
OBJ_PTS = marker_obj_points()


def detect_markers(img, detect, K, dist, draw=True):
    """core.vision.detect_markers (draws on the image; draw=False works on a copy)"""
    return _detect(img if draw else img.copy(), detect, OBJ_PTS, K, dist)


def check(name, ok, msg):
    RESULTS.append((name, ok))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, msg))


def sim(ideal=False, overrides=None, human=True):
    w = World()
    p = load_physics(ideal=ideal, overrides=overrides)
    c = SimClock()
    h = Human(w.human_spec) if (human and w.human_spec) else None
    return SimTello(w, p, c, h)


def place(d, pos, heading):
    """put the drone in the air at pos (bottom, cm) with heading, motors on, hovering"""
    d.body.reset(pos, heading)
    d.body.motors = True
    d.body.mode = "rc"
    d.body.on_surface = False
    d.is_flying = True
    d.ground_z = 0.0
    d.last_cmd_t = d.clock.t


def grab(d, dt=0.12):
    d.streamon()
    d.step(dt)
    raw = d.get_frame_read().frame
    return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 1
def test_markers():
    d = sim(ideal=True, overrides=QUIET)
    worst_d = worst_x = 0.0
    n_ok = 0
    rows = []
    for mk in d.world.markers:
        n = heading_vec(mk.facing)
        for dist, lat in ((100.0, 0.0), (200.0, 35.0)):
            hdg = mk.facing + 180.0
            f, r, _ = body_axes(hdg)
            p2 = mk.pos[:2] + n * dist + r[:2] * lat
            place(d, [p2[0], p2[1], mk.pos[2] - 3.0], hdg)
            img = grab(d)
            found = detect_markers(img, DETECT, d.K, d.dist, draw=False)
            cp, cR = d.camera_pose()
            gt = cR @ (mk.pos - cp)
            if mk.id not in found:
                rows.append("M%d %.0f/%+.0f: NOT DETECTED" % (mk.id, dist, lat))
                continue
            t = found[mk.id]["t"]
            ed = abs(np.linalg.norm(t) - np.linalg.norm(gt))
            ex = abs(t[0] - gt[0])
            worst_d, worst_x = max(worst_d, ed), max(worst_x, ex)
            n_ok += 1
            rows.append("M%d %.0f/%+.0f: d %.1f (gt %.1f) x %.1f (gt %.1f)" % (
                mk.id, dist, lat, np.linalg.norm(t), np.linalg.norm(gt), t[0], gt[0]))
    # marker 0 held by the human at the start
    h = d.human
    h.reset()
    mc = h.marker_center()
    hdg = h.heading + 180.0
    n = heading_vec(h.heading)
    p2 = mc[:2] + n * 120.0
    place(d, [p2[0], p2[1], mc[2] - 3.0], hdg)
    d.step(0.5)
    h.marker_z = d.camera_pose()[0][2]
    img = grab(d)
    found = detect_markers(img, DETECT, d.K, d.dist, draw=False)
    mc = h.marker_center()
    cp, cR = d.camera_pose()
    gt = cR @ (mc - cp)
    tot = 2 * len(d.world.markers) + 1
    if 0 in found:
        t = found[0]["t"]
        ed, ex = abs(np.linalg.norm(t) - np.linalg.norm(gt)), abs(t[0] - gt[0])
        worst_d, worst_x = max(worst_d, ed), max(worst_x, ex)
        n_ok += 1
        rows.append("M0 (human) 120: d %.1f (gt %.1f) x %.1f (gt %.1f)" % (np.linalg.norm(t), np.linalg.norm(gt), t[0], gt[0]))
    else:
        rows.append("M0 (human): NOT DETECTED")
    for r in rows:
        print("      " + r)
    check("1 marker detection + solvePnP", n_ok == tot and worst_d < 4.0 and worst_x < 3.0,
          "%d/%d detected, worst distance error %.1f cm, worst lateral error %.1f cm" % (n_ok, tot, worst_d, worst_x))


# ---------------------------------------------------------------- 2
def test_occlusion():
    d = sim(ideal=True, overrides=QUIET, human=False)
    w = d.world
    r = Renderer(w, d.K, (960, 720), d.dist)
    eye = np.array([330.0, 250.0, 60.0])
    m5 = w.marker(5).pos
    R = look_at(eye, m5)
    img = r.render(eye, R)
    seen = set(detect_markers(img, DETECT, d.K, d.dist, draw=False))
    r.static = [o for o in r.static if o.name != "board3"]
    seen2 = set(detect_markers(r.render(eye, R), DETECT, d.K, d.dist, draw=False))
    check("2 marker behind board", 5 not in seen and 5 in seen2,
          "marker 5 behind board 3: detected=%s; with board 3 removed: detected=%s" % (5 in seen, 5 in seen2))


# ---------------------------------------------------------------- 3
def test_physics():
    d = sim(overrides=QUIET, human=False)
    P = d.P
    t0 = d.clock.t
    d.takeoff()
    t_to = d.clock.t - t0
    zs = []
    for _ in range(250):
        d.send_rc_control(0, 0, 0, 0)
        d.step(0.02)
        zs.append(d.pos[2])
    zs = np.array(zs)
    check("3a take-off + hover", abs(zs.mean() - P.takeoff_height_cm) < 5 and zs.std() < 1.0,
          "take-off took %.1f s, hover z mean %.1f (target %.0f) std %.2f cm" % (t_to, zs.mean(), P.takeoff_height_cm, zs.std()))
    # forward step
    rc = 50
    target = rc * P.rc_to_cm_s_horizontal
    f = body_axes(d.heading)[0]
    v, t90, pitch_max = [], None, 0.0
    for i in range(150):
        d.send_rc_control(0, rc, 0, 0)
        d.step(0.02)
        vf = float(d.vel @ f)
        v.append(vf)
        pitch_max = max(pitch_max, d.pitch)
        if t90 is None and vf >= 0.9 * target:
            t90 = (i + 1) * 0.02
    x0 = d.pos.copy()
    t_stop = None
    for i in range(250):
        d.send_rc_control(0, 0, 0, 0)
        d.step(0.02)
        if t_stop is None and np.linalg.norm(d.vel[:2]) < 1.0:
            t_stop = (i + 1) * 0.02
    coast = float(np.linalg.norm(d.pos[:2] - x0[:2]))
    vss = float(np.mean(v[-25:]))
    check("3b rc forward step", abs(vss - target) < 0.05 * target and t_stop is not None,
          "rc %d -> %.1f cm/s (target %.1f), 90%% at %.2f s, max pitch %.1f deg; release: stop after %.2f s, coast %.1f cm" %
          (rc, vss, target, t90 or -1, pitch_max, t_stop or -1, coast))
    # camera tilts with the body while accelerating
    place(d, [100, 0, 100], 0)
    cam0 = d.camera_pose()[1][2][2]
    for _ in range(10):
        d.send_rc_control(0, 100, 0, 0)
        d.step(0.02)
    cam1 = d.camera_pose()[1][2][2]
    check("3c camera tilts with body", cam1 < cam0 - 0.05,
          "camera forward z: hover %.3f, accelerating %.3f (looks further down)" % (cam0, cam1))
    # free fall
    place(d, [100, -150, 100], 0)
    d.emergency()
    t_fall = None
    for i in range(100):
        d.step(0.01)
        if t_fall is None and d.pos[2] <= 0.01:
            t_fall = (i + 1) * 0.01
    ideal = math.sqrt(2 * 100.0 / P.gravity_cm_s2)
    check("3d free fall (motors off)", t_fall is not None and abs(t_fall - ideal) < 0.1 * ideal + 0.02,
          "100 cm fall took %.2f s (vacuum %.2f s), z now %.1f" % (t_fall or -1, ideal, d.pos[2]))


# ---------------------------------------------------------------- 4
def test_tof():
    d = sim(overrides=QUIET, human=False)
    out = []
    lt = [t for t in d.world.spec.get("tables", []) if t.get("name") == "land_table"][0]
    lx, ly = sum(lt["x"]) / 2.0, sum(lt["y"]) / 2.0
    for name, p, want in (("landing table", [lx, ly, 150], 150.0 - lt["top_z"]), ("floor", [lx, 200, 150], 150.0),
                          ("under table 1", [460, 350, 30], 30.0)):
        place(d, p, 0)
        d.step(0.02)
        tof = d.get_current_state()["tof"]
        out.append((name, tof, want))
    ok = all(abs(t - w) <= 2 for _, t, w in out)
    check("4 tof", ok, ", ".join("%s %d (expect %.0f)" % o for o in out))


# ---------------------------------------------------------------- 5
def test_collision():
    msgs = []
    oks = []
    for beh in ("stop", "fall"):
        ov = {k: dict(v) for k, v in QUIET.items()}
        ov["collision"] = {"behaviour": beh}
        d = sim(overrides=ov, human=False)
        place(d, [120, 0, 80], 0)
        for _ in range(250):
            d.send_rc_control(0, 60, 0, 0)
            d.step(0.02)
        names = [c[1] for c in d.collisions]
        front = d.pos[0] + d.P.half_size_cm[0]
        if beh == "stop":
            ok = "board1" in names and front <= 200.5 and d.body.motors
        else:
            ok = "board1" in names and not d.body.motors and d.pos[2] < 0.5
        oks.append(ok)
        msgs.append("%s: events %s, drone front x %.1f (board at 200), z %.1f, motors %s" %
                    (beh, names[:3], front, d.pos[2], "on" if d.body.motors else "off"))
    check("5 collision", all(oks), "; ".join(msgs))


# ---------------------------------------------------------------- 6 + screenshots
def test_perf_and_shots():
    d = sim()
    v = Viewer(d)
    io = SimIO(d, v, headless=True)
    d.streamon()
    d.takeoff()
    # physics step cost (100 Hz, with human)
    n = 1000
    t = time.perf_counter()
    for _ in range(n // 2):
        d.send_rc_control(0, 0, 0, 40)
        d.step(0.02)
    ms_phys = (time.perf_counter() - t) * 1000.0 / n
    # FPV render
    poses = [([100, 0, 80], 0), ([620, 360, 40], 180), ([350, 350, 215], -90), ([200, 330, 215], -90)]
    fpv = []
    for p, hdg in poses * 8:
        cp, cR = drone_camera(p, hdg, 0, 0, 10)
        t = time.perf_counter()
        d.renderer.render(cp, cR, d.human.objects(), 0.3, d.rng)
        fpv.append((time.perf_counter() - t) * 1000.0)
    obs, comp = [], []
    for name in ("overview", "top", "follow", "side") * 6:
        v.set_view(name)
        t = time.perf_counter()
        v.render3d()
        obs.append((time.perf_counter() - t) * 1000.0)
    left = np.zeros((720, 960, 3), np.uint8)
    for _ in range(20):
        t = time.perf_counter()
        v.compose(left)
        comp.append((time.perf_counter() - t) * 1000.0)
    detect_ms = []
    img = grab(d)
    for _ in range(10):
        t = time.perf_counter()
        detect_markers(img.copy(), DETECT, d.K, d.dist)
        detect_ms.append((time.perf_counter() - t) * 1000.0)
    check("6 performance", np.median(fpv) + np.median(comp) < 33.0,
          "physics %.3f ms/step (100 Hz -> %.1f ms per sim second); FPV render %.1f ms; 3D view %.1f ms; "
          "full viewer frame (3D + compose) %.1f ms; ArUco detect %.1f ms -> FPV + viewer per frame %.1f ms"
          % (ms_phys, ms_phys * 100, np.median(fpv), np.median(obs), np.median(comp), np.median(detect_ms),
             np.median(fpv) + np.median(comp)))

    # screenshot 1: airborne near marker 1 (flown there, so there is a trail)
    d.reset()
    d.streamon()
    d.takeoff()
    for _ in range(100):
        d.send_rc_control(0, 45, 0, 0)
        d.step(0.02)
    for _ in range(40):
        d.send_rc_control(0, 0, 0, 0)
        d.step(0.02)
    shot(d, v, io, "overview.png", target=d.pos + [80, 40, 0], az=-125.0, el=32.0, dist=520.0)
    # screenshot 2: in front of the tunnel table, low, facing marker 3
    d.reset()
    d.streamon()
    place(d, [640, 330, 80], 180)
    for i in range(60):
        d.send_rc_control(0, 0, -30, 0)
        d.step(0.02)
    for i in range(70):
        d.send_rc_control(0, 30, 0, 0)
        d.step(0.02)
    shot(d, v, io, "tunnel.png", target=np.array([500.0, 340.0, 40.0]), az=-140.0, el=24.0, dist=330.0)


def shot(d, v, io, name, target, az, el, dist):
    d.step(0.25)
    img = cv2.cvtColor(d.get_frame_read().frame, cv2.COLOR_RGB2BGR)
    found = detect_markers(img, DETECT, d.K, d.dist)
    cv2.putText(img, "selftest  markers %s" % sorted(found), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    io.show(img)
    v.mode = "free"
    v.target, v.az, v.el, v.dist = np.asarray(target, float), az, el, dist
    os.makedirs(os.path.join(HERE, "screenshots"), exist_ok=True)
    path = os.path.join(HERE, "screenshots", name)
    io.screenshot(path)
    print("      wrote %s (markers in FPV: %s)" % (path, sorted(found)))


if __name__ == "__main__":
    test_markers()
    test_occlusion()
    test_physics()
    test_tof()
    test_collision()
    if "--quick" not in sys.argv[1:]:
        test_perf_and_shots()
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d checks passed%s" % (len(RESULTS) - len(bad), len(RESULTS), (": FAILED " + ", ".join(bad)) if bad else ""))
    sys.exit(1 if bad else 0)
