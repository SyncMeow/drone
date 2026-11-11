"""plot_log.py - one PNG overview of a flight log directory (logs/YYYYmmdd_HHMMSS_real/)

usage: python tools/plot_log.py logs/<dir> [out.png]      (default: <dir>/plot.png)
Panels (shared time axis, s since start): state timeline, rc sent, target distance, target lateral /
vertical, target angle, tof / h, yaw, loop dt + seconds since the last new video frame.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

SEG_COLORS = {0: "#9e9e9e", 1: "#2a6fdb", 2: "#e07b00", 3: "#1a9e5c", 4: "#9b4dca", 5: "#c23b4f"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def seg_of(state):
    if len(state) > 1 and state[0] == "S" and state[1].isdigit():
        return int(state[1])
    return 5 if state == "FINISHED" else 0


def load(d):
    with open(os.path.join(d, "telemetry.csv")) as f:
        rows = list(csv.DictReader(f))
    events = []
    p = os.path.join(d, "events.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    return rows, events


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "plot.png")
    rows, events = load(d)
    if not rows:
        print("telemetry.csv is empty")
        sys.exit(1)
    t = [num(r["t"]) for r in rows]
    col = lambda k: [num(r[k]) for r in rows]          # noqa: E731

    fig, ax = plt.subplots(8, 1, figsize=(14, 18), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, 2, 1.6, 1.6, 1.2, 1.4, 1.2, 1.2]})
    fig.suptitle(os.path.basename(os.path.normpath(d)), fontsize=13)

    # state timeline: one bar per state run, coloured by segment, labelled when wide enough
    a = ax[0]
    start = 0
    span = max(t[-1] - t[0], 1e-6)
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["state"] != rows[start]["state"]:
            t0, t1 = t[start], t[i] if i < len(rows) else t[-1]
            st = rows[start]["state"]
            a.broken_barh([(t0, max(t1 - t0, 1e-3))], (0, 1), facecolor=SEG_COLORS[seg_of(st)],
                          edgecolor="white", linewidth=1)
            if (t1 - t0) / span > 0.04:
                a.text((t0 + t1) / 2, 0.5, st, ha="center", va="center", fontsize=7, color="white")
            start = i
    for e in events:
        if e.get("ev") in ("key", "takeoff", "land", "warn", "exception", "manual"):
            a.axvline(e["t"], color="#333333", linewidth=0.8, alpha=0.6)
            label = e.get("key") or e.get("what") or e["ev"]
            a.text(e["t"], 1.05, str(label), fontsize=6, rotation=90, va="bottom", ha="center")
    a.set_yticks([])
    a.set_ylim(0, 1.6)
    a.set_ylabel("state")

    def lines(a, keys, labels, ylabel):
        for k, lb in zip(keys, labels):
            a.plot(t, col(k), linewidth=1.2, label=lb)
        a.set_ylabel(ylabel)
        a.grid(alpha=0.25)
        a.legend(loc="upper right", fontsize=8, ncol=len(keys))

    lines(ax[1], ["rc_lr", "rc_fb", "rc_ud", "rc_yaw"], ["left/right", "fwd/back", "up/down", "yaw"], "rc sent")
    lines(ax[2], ["sm_z", "raw_z"], ["distance smoothed", "distance raw"], "target z (cm)")
    lines(ax[3], ["sm_x", "sm_y"], ["lateral x", "vertical y"], "target x / y (cm)")
    lines(ax[4], ["sm_ang", "raw_ang"], ["angle smoothed", "angle raw"], "target angle (deg)")
    lines(ax[5], ["tof", "h"], ["tof", "h"], "height (cm)")
    lines(ax[6], ["yaw"], ["yaw"], "yaw (deg)")
    lines(ax[7], ["loop_dt", "since_frame"], ["loop dt", "since last frame"], "seconds")
    ax[7].set_xlabel("time since start (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
