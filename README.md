# kuka-sim

Simulation of a KUKA KR 10 R1100-2 print cell, for planning nozzle orientation along non-planar 3D-printing toolpaths.

- `robot/kr10_r1100_2/`: URDF (kinematics, limits, inertias). Meshes are not included; see `meshes/README.md`.
- `config/cell.yaml`: tool, base, joint limits, bed and obstacles, each with a status (measured / copied / estimated / unknown).
- `docs/tasks/STEP1.md`: the first task, a dynamic-programming baseline for the redundancy angle about the nozzle axis.

Nothing here talks to a robot. Generated KRL files are marked NOT RUN.

## Step 1: DP baseline (simulation only)

```
pip install -r requirements.txt
pytest -q
python scripts/run_step1.py --out out          # all test toolpaths, tilt sweep, RESULTS data (~5 min)
python validate_fk.py data/fk_pairs_template.csv
```

- `kukasim/kinematics.py`: FK from the URDF with TOOL 3 / BASE 0, analytic IK returning all 32 configuration slots (shoulder, elbow, wrist flip, A4 turn, A6 turn). Rotation matrices inside, KUKA ABC only at the boundary.
- `kukasim/planner.py`: DP over (point x redundancy-angle bin x IK slot); reports infeasible stretches and breaks. Equal-cost ties are broken deterministically (smallest |A4| + |A6|, then lowest index), so a fresh plan reproduces the committed `trajectory.csv` on every platform (tested).
- `kukasim/collision.py`: capsules for links and tool, boxes for bed and clamps (sizes in `config/planner.yaml`, all estimated).
- `kukasim/transit.py`: moves between printed segments in the KRL export (issue #2): LIN retract along the nozzle axis to a safe height, PTP sampled in joint space and collision-checked (lifted further if needed, otherwise the export fails), LIN approach. The transit height also clears the highest Z printed so far. Before every transit PTP `$VEL_AXIS` is lowered so the model peak TCP speed is <= 0.8 x 100 mm/s (URDF axis maxima unverified); a transit that cannot meet it fails the export unless `transit.allow_fast_transit` is set. Transits turning A4 / A6 by more than 180 deg are flagged (hose / cable wrap not modelled). The min clearance per move is in the KRL header and `export_report.json`. `.src` files are written with LF line endings. `python scripts/export_krl.py --out results` rewrites only the `.src` files, their `export_report.json` and `trajectory.csv` (same plan, checked against the committed file first).
- `kukasim/first_ptp.py`: the first PTP of every generated program (KRL export and Hilbert corner test) starts from the pose the robot is in, which the program does not know. Its TCP speed is at most B(p) = sum_i (p % x URDF axis max speed) x r_i from any start, r_i = an upper bound of the distance from axis i to the tool tip (from the URDF joint offsets and TOOL 3: the offset to the next joint perpendicular to the axis plus the full lengths of all later offsets). If an integer p >= 1 gives B(p) <= 0.8 x 100 mm/s, the programs set `$VEL_AXIS[1..6] = p` before the first PTP (case `bound`). Otherwise (the repository's case: B(1) = 236.9 mm/s) they declare a start pose (`transit.start_axis`, default KUKA HOME {0, -90, 90, 0, 0, 0}, +-`start_tolerance_deg` 1 deg), lower `$VEL_AXIS` with the transit speed model from that start (and the 64 tolerance-box corners), and the header says the robot must be at that pose before the program starts (case `declared start`). The header of every program states the case and the peak TCP speed of the first PTP.
- Numbers in the `.src` and `trajectory.csv` files have one fixed form (`krl.fnum` / `krl.fang`): fixed decimals, no negative zero, orientation A / C in (-180, 180], so committed results do not change on last-bit noise.
- `kukasim/toolpaths.py`: dome, saddle, sine wave, 45° ramp and a bed perimeter path.
- `kukasim/slicer_import.py`: optional reader for `non_planar_slicer` KRL output (separate module; no GPL code).
- `validate_fk.py`: compares FK with pendant (A1..A6, X..C) pairs.
- `results/`, `RESULTS.md`: outputs of the run for the PR.

## Corner test: Hilbert curve with a per-corner speed ratio (simulation only)

```
python scripts/run_hilbert_corner.py                 # writes results/hilbert_corner/
python scripts/run_hilbert_corner.py --order 3 --seed 2 --min-straight 20 --out out/hc
python scripts/run_hilbert_corner.py --speed-mode pc  # the PC-side $OV_PRO variant (see below)
```

One Hilbert curve (order 3: 50 corners of 90 deg) on the first layer, nozzle vertical (A0 B90 C0, TOOL 3), extruder rpm fixed. The TCP speed around each corner is r x the straight speed, with r from 0.5 to 1.5 in a seeded random order (each ratio at least twice). Corners whose straight to a neighbour (between the corner zones) is shorter than 15 mm (flow-response assumption) get r = 1 and are listed.

- **Speed mode `krl` (default):** the program itself splits each leg at +-`d_zone_mm` (default 6 mm, must be >= 2 x C_DIS) around every corner and sets `$VEL.CP = r x v` on the two corner sub-segments, back to v after them, so the ratio is path-synchronous in the program. No PC is needed.
- **Speed mode `pc` (option):** one `$VEL.CP`; `$OV_PRO` is switched to `r x 60 %` around each corner per `schedule.csv`. This needs a PC-side schedule follower that reads the position and writes the override; **this repository does not provide one.**
- Layer Z is z(x, y) = z0 + kx (x - x0) + ky (y - y0) (`layer_z`); the default is flat at bed top_z + layer height.
- The default square is fitted to the reachable bed, kept >= `clamp_margin_mm` (XY) from the inflated clamp boxes and capped at `max_side_mm` (300 mm).
- **Scope of the checks: checked against cell.yaml only; site keep-outs must be checked locally before running.** `config/cell.yaml` has an optional `keep_out` list of boxes (`{name, x: [lo, hi], y: [lo, hi], z: [lo, hi], clearance_mm}`), checked like the clamps. It is empty in the repository on purpose: site keep-outs are added to a local copy and never committed.
- The programs assume the controller's INI / BAS(#INITMOV,0) defaults for everything they do not set (`$ACC.CP`, `$VEL.ORI`, `$ACC.ORI`, ...). `$APO.CDIS` is always set explicitly. The approach PTP gets `$VEL_AXIS` (at most 5 %, lowered for the first-PTP speed rule below) / `$ACC_AXIS` (10 %) and an IK target inside the hard limits; its path from the current pose is not collision-checked.
- The corner-speed model (C_DIS blend + trapezoid, optional `corner_dip_factor`, default 1.0 = no dip) is a **model estimate**; predicted bead widths are model-only. Blended 90 deg corners dip below line speed on real controllers: the logged speed decides.
- Guards: every segment >= 2 x C_DIS, corner zones disjoint, every LIN > 0.5 mm; otherwise `generate` raises `ValueError` with the offending segments.
- `kukasim/hilbert_corner.py`: geometry, square fit, schedule, corner-speed model, checks over the whole path (DP planner with one fixed angle, joint limits, singularities, collisions incl. keep-outs, clamp margin, TCP and joint speeds, nozzle-to-bed distance) and the outputs.
- Parameters: `hilbert_corner` in `config/planner.yaml` (all choices or assumptions).
- `results/hilbert_corner/`: `waypoints.csv`, `schedule.csv` (corner zones; the PC schedule in `pc` mode), `speed_profile.csv`, `hilbert_corner.src` (NOT RUN, LF line endings), `path.png`, `speed.png`, `report.md`.
