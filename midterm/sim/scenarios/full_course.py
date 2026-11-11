"""
full_course.py - headless run of the whole course with the real flight program (core.app.run) in the 3D sim.

    python sim/scenarios/full_course.py                 # default physics, seed 1
    python sim/scenarios/full_course.py --seed 3
    python sim/scenarios/full_course.py --ideal         # no inertia / latency / noise
    python sim/scenarios/full_course.py --seeds 1-5     # several runs, one summary table at the end
    python sim/scenarios/full_course.py --verbose       # also show the flight program's terminal output
    python sim/scenarios/full_course.py --record        # also write fpv_raw.mp4 / fpv_hud.mp4 (slower)
    python sim/scenarios/full_course.py --plot out.png  # top-down plot of the run (world + human + drone path)

Key script: g at 1 s, then nothing (no manual help). The run ends when the program lands after the course,
returns to MANUAL, sits in a waiting state for WAIT_GIVE_UP s, or after --max-time sim seconds (then q).
Output: state sequence with times, segment splits, collisions, minimum clearance in the tunnel, landing
position relative to the landing table, PASS / FAIL per segment. Exit code 0 = all segments passed.
Every run writes logs/..._sim/ (flight.log, telemetry.csv, events.jsonl, params.json, truth.csv) unless --no-log.
"""
import argparse
import os
import sys
import time as _time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
LAB6 = os.path.dirname(SIM)
sys.path.insert(0, SIM)
sys.path.insert(1, LAB6)

from run_sim import make_sim                                   # noqa: E402
from core import states as S                                   # noqa: E402

KEY_AT = 1.0             # s: press g
WAIT_GIVE_UP = 15.0      # s in a waiting state (S1_WAIT0 / S3_WAIT4 / S4_WAIT5) -> give up (q)
SEG_END = {1: S.S2_FOLLOW, 2: S.S3_ALIGN, 3: S.S4_CLIMB, 4: S.S5_ALIGN, 5: S.FINISHED}


class _Null(object):
    def write(self, s):
        return len(s)

    def flush(self):
        pass


def _seg_of(state):
    if state.startswith("S") and state[1:2].isdigit():
        return int(state[1])
    return 5 if state == S.FINISHED else 0


def run_once(seed=1, ideal=False, world=None, physics=None, max_time=400.0, record=False, log=True,
             verbose=False, plot=None):
    drone, clock, io = make_sim(world, physics, ideal, headless=True, seed=seed)
    w = drone.world
    seq = []                    # [(t rel, state)]
    st = {"done_t": None, "wait_t": None, "quit": None, "pressed": False}

    def script(io):
        t = io.t - drone.t0
        c = io.course
        if not st["pressed"]:
            if t >= KEY_AT:
                st["pressed"] = True
                return ord('g')
            return -1
        if st["quit"] is not None:
            return ord('q')
        state = c.state if c is not None else S.MANUAL
        if not seq or seq[-1][1] != state:
            seq.append((t, state))
        why = None
        if t >= max_time:
            why = "time limit %.0f s" % max_time
        elif state == S.MANUAL and len(seq) > 1 and S.FINISHED not in (getattr(c, "splits", None) or {}):
            why = "back in MANUAL"
        elif state in S.WAIT_STATES and state != S.FINISHED:
            st["wait_t"] = st["wait_t"] if st["wait_t"] is not None else t
            if t - st["wait_t"] >= WAIT_GIVE_UP:
                why = "waiting in %s for %.0f s" % (state, WAIT_GIVE_UP)
        else:
            st["wait_t"] = None
        if (state == S.FINISHED or S.FINISHED in (getattr(c, "splits", None) or {})) and not drone.body.motors:
            st["done_t"] = st["done_t"] if st["done_t"] is not None else t
            if t - st["done_t"] >= 1.0:
                why = "landed"
        if why:
            st["quit"] = why
            return ord('q')
        return -1

    io.script = script
    from core.app import run
    out, err = sys.stdout, sys.stderr
    if not verbose:
        sys.stdout = sys.stderr = _Null()                      # the flight log still gets every line (tee)
    t_wall = _time.time()
    try:
        run(drone, clock=clock, io=io, log=log, record=record and log, hud_video=record, tag="sim")
    finally:
        sys.stdout, sys.stderr = out, err
    wall = _time.time() - t_wall
    bb = getattr(run, "blackbox", None)
    course = io.course

    # ---- analysis ------------------------------------------------------------------
    tr = np.array([r[:14] for r in drone.truth]) if drone.truth else np.zeros((0, 14))
    t_rel = tr[:, 0] - drone.t0 if len(tr) else tr[:, 0]
    t_g = next((t for t, s in seq if s != S.MANUAL), None)
    splits = getattr(course, "splits", {}) or {}
    if S.FINISHED in splits and S.FINISHED not in [s for _, s in seq]:       # FINISHED -> land -> MANUAL in one cycle
        tf = splits[S.FINISHED] - drone.t0
        seq = sorted(seq + [(tf, S.FINISHED)], key=lambda x: x[0])
    reached = [s for _, s in seq]
    run_t0 = getattr(course, "run_t0", None)
    # collisions with their segment
    cols = []
    for (t, name, pos, speed) in drone.collisions:
        state = S.MANUAL
        for ts, s in seq:
            if ts <= t - drone.t0:
                state = s
        cols.append((t - drone.t0, name, pos, speed, state))
    # tunnel clearance: drone top vs table 1 underside while under it; drone top vs marker 3 sign bottom
    tbl = next((t for t in w.spec.get("tables", []) if t.get("name") == "table1"), None)
    half = np.asarray(drone.P.half_size_cm, float)
    clear_tbl = clear_m3 = None
    if tbl is not None and len(tr):
        under = ((tr[:, 1] + half[0] > tbl["x"][0]) & (tr[:, 1] - half[0] < tbl["x"][1]) &
                 (tr[:, 2] + half[1] > tbl["y"][0]) & (tr[:, 2] - half[1] < tbl["y"][1]) &
                 (tr[:, 3] < tbl["top_z"] - tbl.get("thickness", 4)))
        if np.any(under):
            clear_tbl = float(np.min(tbl["top_z"] - tbl.get("thickness", 4) - (tr[under, 3] + 2 * half[2])))
    sign = next((b for b in w.boxes if b.name == "sign3"), None)
    if sign is not None and len(tr):
        near = ((tr[:, 1] - half[0] < sign.hi[0] + 3) & (tr[:, 1] + half[0] > sign.lo[0] - 3) &
                (tr[:, 2] + half[1] > sign.lo[1]) & (tr[:, 2] - half[1] < sign.hi[1]) & (tr[:, 3] < sign.lo[2]))
        if np.any(near):
            clear_m3 = float(np.min(sign.lo[2] - (tr[near, 3] + 2 * half[2])))
    # landing
    lt = next((t for t in w.spec.get("tables", []) if t.get("name") == "land_table"), None)
    p = drone.pos.copy()
    land = None
    if lt is not None:
        cx, cy = sum(lt["x"]) / 2.0, sum(lt["y"]) / 2.0
        on = (lt["x"][0] <= p[0] <= lt["x"][1] and lt["y"][0] <= p[1] <= lt["y"][1]
              and abs(p[2] - lt["top_z"]) < 3 and not drone.body.motors)
        land = dict(dx=p[0] - cx, dy=p[1] - cy, on=on, z=p[2])
    # per segment verdicts
    verdict = {}
    for k in range(1, 6):
        start_ok = k == 1 or verdict.get(k - 1) == "PASS"
        if not start_ok:
            verdict[k] = "-"
            continue
        reached_end = SEG_END[k] in reached
        seg_cols = [c for c in cols if _seg_of(c[4]) == k and c[1] != "surface"]
        manual_j = k == 2 and any(s == S.S3_ALIGN for s in reached) and not reached_end
        ok = reached_end and not seg_cols and not manual_j
        if k == 5:
            ok = ok and land is not None and land["on"]
        verdict[k] = "PASS" if ok else "FAIL"
    res = dict(seed=seed, ideal=ideal, seq=seq, splits=splits, run_t0=run_t0, cols=cols, clear_tbl=clear_tbl,
               clear_m3=clear_m3, land=land, verdict=verdict, quit=st["quit"], t_g=t_g,
               sim_t=clock.time() - drone.t0, wall=wall, final=p, log_dir=bb.flog.dir if bb is not None else None,
               last=([s for s in reached if s != S.MANUAL] or [S.MANUAL])[-1],
               run_time=(splits.get(S.FINISHED) - run_t0) if (run_t0 and splits.get(S.FINISHED)) else None)
    if plot:
        plot_topdown(drone, seq, plot, res)
    return res


def report(r):
    print("=" * 78)
    print("full course  seed %s%s   (sim %.1f s in %.1f s wall)%s" % (
        r["seed"], "  IDEAL" if r["ideal"] else "", r["sim_t"], r["wall"],
        ("   log: " + os.path.relpath(r["log_dir"], LAB6)) if r["log_dir"] else ""))
    print("states:")
    for t, s in r["seq"]:
        print("  %7.1f  %s" % (t, s))
    print("ended: %s" % r["quit"])
    if r["run_t0"]:
        t0 = r["run_t0"]
        marks = [t0] + [r["splits"].get(SEG_END[k]) for k in range(1, 6)]
        parts = []
        for k in range(5):
            a, b = marks[k], marks[k + 1]
            parts.append("seg %d %s" % (k + 1, "%.1f s" % (b - a) if a is not None and b is not None else "-"))
        print("splits: " + "  ".join(parts) + ("   total %.1f s" % r["run_time"] if r["run_time"] else ""))
    if r["cols"]:
        for t, name, pos, speed, state in r["cols"]:
            print("COLLISION %.1f s  %s at (%.0f, %.0f, %.0f) %.0f cm/s in %s" % (t, name, pos[0], pos[1], pos[2],
                                                                                speed, state))
    else:
        print("collisions: none")
    print("tunnel: min clearance under table 1 %s, under marker 3 sign %s" % (
        "-" if r["clear_tbl"] is None else "%.1f cm" % r["clear_tbl"],
        "-" if r["clear_m3"] is None else "%.1f cm" % r["clear_m3"]))
    ld = r["land"]
    f = r["final"]
    print("final position (%.0f, %.0f, %.0f)%s" % (f[0], f[1], f[2], "" if ld is None else
          "  vs landing table centre: dx %+.0f dy %+.0f cm -> %s" % (ld["dx"], ld["dy"],
                                                                    "ON THE TABLE" if ld["on"] else "not on the table")))
    print("segments: " + "  ".join("%d %s" % (k, r["verdict"][k]) for k in range(1, 6)))


def summary(rs):
    print("\n%-8s %-12s %8s %5s %-22s %s" % ("run", "reached", "time", "coll", "landing (dx, dy)", "segments"))
    for r in rs:
        ld = r["land"]
        lt = "-" if ld is None or r["last"] != S.FINISHED else "%+.0f, %+.0f %s" % (ld["dx"], ld["dy"],
                                                                                    "on" if ld["on"] else "OFF")
        print("%-8s %-12s %8s %5d %-22s %s" % (
            ("ideal" if r["ideal"] else "seed %d" % r["seed"]), r["last"],
            "%.1f s" % r["run_time"] if r["run_time"] else "-", len(r["cols"]), lt,
            " ".join(r["verdict"][k][0] for k in range(1, 6))))


# ---------------------------------------------------------------- top-down plot (cv2 only)
SEG_COLORS = {0: (150, 150, 150), 1: (40, 160, 40), 2: (200, 120, 0), 3: (0, 140, 255), 4: (200, 0, 200),
              5: (30, 30, 220)}


def plot_topdown(drone, seq, path, res=None, ppcm=1.4):
    import cv2
    w = drone.world
    x0, y0 = -60.0, -140.0
    x1, y1 = 880.0, 540.0
    W, H = int((x1 - x0) * ppcm), int((y1 - y0) * ppcm)
    img = np.full((H + 60, W, 3), 245, np.uint8)

    def P(x, y):
        return int(round((x - x0) * ppcm)), int(round((y1 - y) * ppcm))

    for gx in range(-50, 900, 50):
        cv2.line(img, P(gx, y0), P(gx, y1), (232, 232, 232), 1)
    for gy in range(-100, 550, 50):
        cv2.line(img, P(x0, gy), P(x1, gy), (232, 232, 232), 1)
    for ln in w.lines:
        cv2.line(img, P(*ln["a"]), P(*ln["b"]), tuple(int(c) for c in ln["color"]), 3)
    for b in sorted(w.boxes, key=lambda b: b.hi[2]):
        col = tuple(int(c) for c in b.color) if hasattr(b, "color") else (200, 200, 200)
        cv2.rectangle(img, P(b.lo[0], b.hi[1]), P(b.hi[0], b.lo[1]), col, -1)
        cv2.rectangle(img, P(b.lo[0], b.hi[1]), P(b.hi[0], b.lo[1]), (90, 90, 90), 1)
        if not b.name.startswith(("sign", "table1_leg", "land_table_leg", "wall")):
            cv2.putText(img, "%s %d" % (b.name, b.hi[2]), (P(b.lo[0], b.hi[1])[0] + 2, P(b.lo[0], b.hi[1])[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
    for mk in w.markers:
        c = P(mk.pos[0], mk.pos[1])
        n = mk.normal if hasattr(mk, "normal") else None
        cv2.circle(img, c, 5, (0, 0, 0), -1)
        cv2.putText(img, "M%d" % mk.id, (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    tr = drone.truth
    if tr:
        a = np.array([r[:14] for r in tr])
        hp = a[:, 11:13]
        ok = ~np.isnan(hp[:, 0])
        pts = np.array([P(x, y) for x, y in hp[ok]], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(img, "human", tuple(pts[0] + [6, 14]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        # drone path coloured by segment
        tseq = [(t + drone.t0, s) for t, s in seq]
        segs = []
        j = 0
        cur = 0
        for r in a:
            while j < len(tseq) and tseq[j][0] <= r[0]:
                cur = _seg_of(tseq[j][1])
                j += 1
            segs.append(cur)
        dp = np.array([P(x, y) for x, y in a[:, 1:3]], np.int32)
        for i in range(1, len(dp)):
            cv2.line(img, tuple(dp[i - 1]), tuple(dp[i]), SEG_COLORS.get(segs[i], (0, 0, 0)), 2, cv2.LINE_AA)
        for r in a[a[:, 10] > 0]:
            cv2.circle(img, P(r[1], r[2]), 6, (0, 0, 255), 2)
        cv2.drawMarker(img, tuple(dp[-1]), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    # legend
    y = H + 22
    x = 10
    for k in range(1, 6):
        cv2.line(img, (x, y - 5), (x + 26, y - 5), SEG_COLORS[k], 3)
        cv2.putText(img, "seg %d" % k, (x + 32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        x += 100
    cv2.putText(img, "grey = human, red circle = collision, x = final", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (40, 40, 40), 1, cv2.LINE_AA)
    if res is not None:
        txt = "seed %s%s  reached %s  segments %s" % (res["seed"], " ideal" if res["ideal"] else "", res["last"],
                                                    " ".join("%d:%s" % (k, res["verdict"][k]) for k in range(1, 6)))
        cv2.putText(img, txt, (10, H + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "top view, 50 cm grid, x east / y north (wall at top)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description="full course, headless, flight program in the 3D sim")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None, help='e.g. "1-5" or "1,3,4"')
    ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--world", default=None)
    ap.add_argument("--physics", default=None)
    ap.add_argument("--max-time", type=float, default=400.0)
    ap.add_argument("--record", action="store_true", help="write fpv_raw.mp4 / fpv_hud.mp4")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--plot", default=None, help="write a top-down png of the (last) run")
    a = ap.parse_args()
    seeds = [a.seed]
    if a.seeds:
        seeds = []
        for part in a.seeds.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                seeds += list(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(part))
    rs = []
    for i, s in enumerate(seeds):
        r = run_once(s, a.ideal, a.world, a.physics, a.max_time, a.record, not a.no_log, a.verbose,
                     a.plot if i == len(seeds) - 1 else None)
        report(r)
        rs.append(r)
    if len(rs) > 1:
        summary(rs)
    sys.exit(0 if all(all(v == "PASS" for v in r["verdict"].values()) for r in rs) else 1)


if __name__ == "__main__":
    main()
