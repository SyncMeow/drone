# Simulator (lab6/sim)

Two simulators live here:

1. **3D sim** (`run_sim.py`, new): the real flight program (`core.app.run`, the same loop / HUD / flight log
   as `main.py`) flying a simulated Tello with inertia, latency, noise, collisions and a dummy human carrying
   marker 0, in the whole EAB06 course. numpy + OpenCV only, no GL.
2. **Kinematic regression scenarios** (`sim_world.py`, `run_seg*.py`, `run_follow.py`, `baseline/`, old):
   rc = cm/s, no physics. Their output must stay byte-identical; they are the regression test of the flight
   logic after any change in `core/`:

   ```bash
   bash baseline/run_all.sh /tmp/new && diff -r baseline /tmp/new     # only "Only in baseline: run_all.sh"
   ```

## Running the 3D sim

```bash
conda activate uav
cd lab6
python sim/run_sim.py                    # window; press g to fly the whole course (h j k l = segments 2-5)
python sim/run_sim.py --ideal            # no inertia / latency / noise (like the old kinematic sim)
python sim/run_sim.py --scale 0.6        # display scale; default auto = fit the screen
python sim/run_sim.py --no-record        # no mp4 (logs + truth.csv still written); --no-log = nothing written
python sim/run_sim.py --manual-only      # small manual-flight demo without the course program
python sim/run_sim.py --headless --keys "1:g" --quit-after 200     # scripted, no window
python sim/scenarios/full_course.py --seeds 1-5 --plot /tmp/top.png  # headless full course, PASS / FAIL report
python sim/selftest.py [--quick]         # headless checks (detection, physics, tof, collisions; --quick ~1 s)
```

Window (fits 1440x900 at the auto scale, ~0.72): left = the program's HUD canvas (FPV 960x720 + panel +
message bar, exactly what `main.py` shows), right = 3D course view (640x480) with ground truth and sim keys
below. Only the display is shrunk: logs and recordings keep full resolution. The screen size is asked from
tkinter in a subprocess (optional); without it 1440x900 is assumed. Override: `SIM_SCREEN=1280x800`.

### Keys

Flight keys go to the program unchanged: `g h j k l` start segments, `w s a d z x c v` move, `1 2 3` take off /
land / stop, `q` / ESC quit (lands). Sim keys (consumed by the sim):

| key | action |
| --- | --- |
| `p` / space, `n` | pause, single step |
| `[` `]` | sim speed x0.5 / x1 / x2 / x4 |
| `r` | reset drone + human to the start |
| `t` `y` `u` `i` | 3D view: top / side / follow / overview; `=` `-` zoom; mouse drag = orbit, wheel = zoom |
| `m`, arrow keys | human manual mode, move him |
| `b` | human hides / shows marker 0 |

### Output

Every run (unless `--no-log`) writes `logs/YYYYmmdd_HHMMSS_sim/` through `core/blackbox.py`, like a real flight:
`flight.log`, `telemetry.csv`, `events.jsonl`, `params.json`, `fpv_raw.mp4`, `fpv_hud.mp4`, plus sim-only:

- `truth.csv`: t (same clock as telemetry.csv), x, y, z (drone bottom, cm), yaw (heading, deg, clockwise from
  east), pitch, roll, vx, vy, vz, collision, human_x, human_y, m0_visible (marker 0 shown, facing the camera and
  fully inside the image; occlusion not checked)
- `"collision"` events in `events.jsonl` (object, position, speed, state)

`python tools/plot_log.py logs/<dir>` works on sim logs too.

## Files

| file | content |
| --- | --- |
| `run_sim.py` | entry point, `make_sim()` (drone, clock, io), `--manual-only` demo |
| `tello_sim.py` | `SimTello` (djitellopy subset incl. `get_current_state()` for the HUD), `SimClock` |
| `physics.py` | rigid body: velocity loop + tilt limit, drag, yaw, vertical, gusts, collision response |
| `render.py` | software renderer (FPV with lens distortion from `calibrate.xml`, 3D view) |
| `world.py` | loads the world file: boxes, tables, markers, floor lines |
| `human.py` | dummy human with marker 0 (segment 2) |
| `viewer.py` | window composition + `SimIO` (wait_key steps the sim, show; truth.csv / collision events) |
| `scenarios/full_course.py` | headless whole course: states, splits, collisions, tunnel clearance, landing |
| `selftest.py` | headless checks; also run by `../../setup.sh` |

Marker detection is `core.vision` (the flight program's own detector).

## World file (`config/worlds/eab06.json`)

JSON with `//` comments. cm; X east, Y north (toward the wall), Z up; headings / `facing_deg` clockwise from
east (0 = east, 90 = south, 180 = west, -90 = north), the same sense as Tello yaw.

- `start`: take-off `pos` + `heading_deg`; `floor`, `sky_color`
- `boxes`: `{"name", "min", "max", "color", "solid"}` axis-aligned; `tables`: top slab + 4 legs
- `markers`: `{"id", "pos" (centre), "facing_deg", "size", "mount"}`; a marker on a box face is drawn on the box,
  otherwise (`"mount": "sign"`) a thin solid sign plate is created
- `lines`: floor tape (drawn only)
- `human`: start, path, speed, wait / resume distances, hold height, leave path (see the comments in the file)

The world is an estimate. **When the world and the flight parameters disagree, change the world** (the
parameters come from real flights). Changes so far and why are commented in the file (marker 2 offset,
board 1 width, marker 5 105 cm from marker 4, human hold height and path).

## Physics file (`config/sim_physics.json`)

Groups `body`, `control`, `flight`, `video`, `disturbance`, `sensors`, `collision`, `power`, `sim`; `--ideal`
applies `presets.ideal`. All values are **estimates, not calibrated against the real drone**: mass 80 g,
rc -> speed 0.85 cm/s per rc (taken from `S3_UNDER_CM_PER_RC`, itself back-computed), response times, video
latency 0.15 s, 3 % dropped frames, noise levels, battery / temperature curves.

## What is approximate

- rc -> cm/s scale is unknown: measure it in a real flight from the slope of the marker distance in
  `telemetry.csv` (raw_z) during a constant-rc segment, then set `rc_to_cm_s_*`.
- No aerodynamic ground / ceiling effect under the table, no prop wash, no Wi-Fi stalls, no motion blur,
  constant lighting; yaw sign is the same as the program assumes (cannot catch a wrong `HEADING_SIGN`).
- Collisions are axis-aligned boxes; "stop" pushes the drone out (the real Tello may flip or fall).
- The human is idealised (always faces the drone, waits for it, never occludes his own marker).
