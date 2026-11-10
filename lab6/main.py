"""
main.py - midterm course (EAB06), segments 1 to 5. Entry point: python main.py
(logic lives in core/, flight parameters in config/params.json)

Segment 1 (markers 1, 2)
    S1_APPROACH : fly toward marker 1, stop at APPROACH_DIST (never hit marker 1)
    S1_RIGHT    : slide right past the board; stop when straight in front of marker 2 (its x ~ 0)
    S1_FORWARD  : fly forward, stop M2_STOP_DIST before marker 2 (backs off if too close)
    S1_LEFT     : slide left until the hand-held marker 0 is seen
Segment 2 (marker 0)
    S2_FOLLOW   : follow the hand-held marker 0 (turn + slide to keep it centred, keep distance),
                  around the chair and over the blue line; if marker 0 is lost, keep turning left to find it
                  (FOLLOW_SEARCH_DIR); once marker 0 has been seen, "marker 0 gone + marker 3 in view"
                  for S2_TO_S3_TIME s (aligning with marker 3 meanwhile) switches to segment 3 automatically;
                  j switches at any time; seeing marker 0 again in S3_ALIGN returns to S2_FOLLOW
Segment 3 (markers 3, 4)
    S3_ALIGN    : face marker 3 squarely, stop at S3_DIST
    S3_DOWN     : descend until marker 3's bottom edge is at least S3_CLEAR above the camera
                  (guarantees passing under marker 3 without hitting it)
    S3_UNDER    : fly forward under the table at low height (distance based on marker 3, or S3_UNDER_TIME s)
    S3_TURN     : turn clockwise S3_TURN_DEG in place
    S3_UP       : climb until marker 4 is seen or TOF reaches S3_UP_ALT (marker 4 height)
    S3_FIND4    : at height but no marker 4: back off a little, then search left / right
    S3_ALIGN4   : align with marker 4 (lateral / height / distance / angle)
Segment 4 (marker 5)
    S4_LEFT     : keep the marker 4 height (the board is lower, markers 4 / 5 are at the same height)
                  and slide left over the board until marker 5 is seen (climbs S4_CLIMB_CM first if > 0)
Segment 5
    S5_ALIGN    : align with marker 5, stop M5_DIST from the wall (above the table)
    => land on the table

Each state only looks at its own target marker id; other markers in view are ignored.
Lateral / vertical / forward control uses the marker position in cm (solvePnP), smoothed (EMA)
and corrected with small gains, so the drone does not swing left and right.

Keys (the "drone" window must have focus, keyboard input method set to English)
    g h j k l  start segment 1 / 2 / 3 / 4 / 5 from manual (takes off first if on the ground)
    timer: starts on g, shown top-right; total and per-segment times are printed on landing (or exit)
        g: segment 1 start (approach marker 1)   h: segment 2 (follow marker 0, j switches to segment 3)
        j: segment 3 start (align marker 3)       k: segment 4 start (align marker 4 first; if not seen
                                                     within 2 s, the current heading is taken as the wall)
        l: segment 5 (align marker 5, then land)
    1 take off / 2 land / 3 stop
    w s a d z x c v  manual moves (stop about KEY_HOLD s after release); 5 height / 6 battery
    any operation key (w s a d x z c v 1 2 3) during autonomous flight => back to manual immediately;
        press g..l to resume
    q / ESC quit (stop and land); Ctrl+C also lands

Window: video + right panel (state, segment, target marker, rc bars, battery / temperature, video fps,
timer, key hints) + bottom bar with the latest messages (warnings in red/orange).

Logs: every run writes logs/YYYYmmdd_HHMMSS_real/ (flight.log, telemetry.csv, events.jsonl, params.json,
fpv_raw.mp4, fpv_hud.mp4). Plot one with: python tools/plot_log.py logs/<dir>
Options:
    --no-record     no mp4 files (logs are still written)
    --no-hud-video  record only fpv_raw.mp4 (saves CPU)
    --no-hud        plain video window (old 3-line overlay), no fpv_hud.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # 讓 core/ 和 keyboard_djitellopy 找得到

from djitellopy import Tello                                        # noqa: E402

from core.app import run                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="midterm course (Tello)")
    ap.add_argument("--no-record", action="store_true", help="do not record fpv_raw.mp4 / fpv_hud.mp4")
    ap.add_argument("--no-hud-video", action="store_true", help="do not record fpv_hud.mp4")
    ap.add_argument("--no-hud", action="store_true", help="plain video window instead of the HUD")
    args = ap.parse_args()
    try:
        run(Tello(), record=not args.no_record, hud_video=not args.no_hud_video, hud=not args.no_hud)
    finally:
        bb = getattr(run, "blackbox", None)
        if bb is not None and bb.flog.dir is not None:
            print("log directory:", bb.flog.dir)


if __name__ == '__main__':
    main()
