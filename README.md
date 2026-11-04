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
- `kukasim/collision.py`: capsules for links and tool, boxes for bed and clamps (sizes in `config/planner.yaml`, all estimated). `margins()` gives the clearance per primitive and obstacle.
- `kukasim/transit.py`: moves between printed segments in the KRL export (issue #2): LIN retract along the nozzle axis to a safe height, PTP sampled in joint space and collision-checked (lifted further if needed, otherwise the export fails), LIN approach. The transit height also clears the highest Z printed so far. Before every transit PTP `$VEL_AXIS` is lowered so the model peak TCP speed is <= 0.8 x 100 mm/s (URDF axis maxima unverified); a transit that cannot meet it fails the export unless `transit.allow_fast_transit` is set. Transits turning A4 / A6 by more than 180 deg are flagged (hose / cable wrap not modelled). The min clearance per move is in the KRL header and `export_report.json`. `.src` files are written with LF line endings. `python scripts/export_krl.py --out results` rewrites only the `.src` files and their `export_report.json`.
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
- The programs assume the controller's INI / BAS(#INITMOV,0) defaults for everything they do not set (`$ACC.CP`, `$VEL.ORI`, `$ACC.ORI`, ...). `$APO.CDIS` is always set explicitly. The approach PTP gets `$VEL_AXIS` / `$ACC_AXIS` (5 % / 10 %) and an IK target inside the hard limits; the PTP from the current pose is not checked.
- The corner-speed model (C_DIS blend + trapezoid, optional `corner_dip_factor`, default 1.0 = no dip) is a **model estimate**; predicted bead widths are model-only. Blended 90 deg corners dip below line speed on real controllers: the logged speed decides.
- Guards: every segment >= 2 x C_DIS, corner zones disjoint, every LIN > 0.5 mm; otherwise `generate` raises `ValueError` with the offending segments.
- `kukasim/hilbert_corner.py`: geometry, square fit, schedule, corner-speed model, checks over the whole path (DP planner with one fixed angle, joint limits, singularities, collisions incl. keep-outs, clamp margin, TCP and joint speeds, nozzle-to-bed distance) and the outputs.
- Parameters: `hilbert_corner` in `config/planner.yaml` (all choices or assumptions).
- `results/hilbert_corner/`: `waypoints.csv`, `schedule.csv` (corner zones; the PC schedule in `pc` mode), `speed_profile.csv`, `hilbert_corner.src` (NOT RUN, LF line endings), `path.png`, `speed.png`, `report.md`.

## Static check of a KRL program (simulation only)

```
python scripts/check_src.py results/hilbert_corner/hilbert_corner.src
```

`kukasim/src_check.py` reads a `.src` file and replays every PTP / LIN on the models above (URDF FK / IK, `CollisionModel`, the PTP speed model of `transit.py`); it adds no kinematics or collision model of its own. Per motion it reports: reach inside the joint limits minus `use_margin_deg` (a LIN is sampled and one IK branch is followed from its start; a configuration change or a joint step above `lin_max_joint_step_deg` is flagged), |A5|, elbow and shoulder margins, the TCP speed (LIN: `$VEL.CP`; PTP: model peak at the `$VEL_AXIS` / `$ACC_AXIS` in force; always at 100 % override, since a PC under RSI or the pendant can set any override), the minimum clearance and its primitive / obstacle pair (bed, clamps, keep-out boxes, floor; the nozzle tip is exempt from the bed on LIN moves only), the tip height above the highest bed top and, on LIN moves, the tip XY inside the bed outline minus a margin. Thresholds: `src_check` in `config/planner.yaml` (defaults taken from `transit` and `hilbert_corner`).

- Dialect: `BAS(#TOOL / #BASE / #INITMOV / #VEL_PTP)`, `$TOOL = TOOL_DATA[n]`, `$BASE = $NULLFRAME`, axis and Cartesian `PTP` / `LIN` targets (optional `S`, `T`, `E1..E4`, `C_DIS` / `C_PTP`), `$VEL.CP`, `$VEL_AXIS[i]`, `$ACC_AXIS[i]`, `$APO.CDIS`, `$OV_PRO`, `$ORI_TYPE`, `RSI_CREATE / RSI_ON / RSI_OFF`, `IF ... THEN / ENDIF`, `DECL`, `WAIT SEC`, `HALT`, `END` and user-variable assignments. Anything else is listed as "not interpreted" with its line number (a flag); a motion that cannot be read is a parse error. Only TOOL 3 and BASE 0 are modelled; any other frame is an error.
- Output: `<program>_check.md` and `<program>_check.json` next to the program or in `--out`; a program under `config/` or `results/` is reported into `out/src_check/` (the checker never writes into `config/` or `results/`). Exit code 0 = no flag, 1 = at least one flag, 2 = parse or frame error.
- The first PTP of a program starts from an unknown pose: it is reported as "start unknown, not checked" (target pose only) unless `--start-axis A1 A2 A3 A4 A5 A6` is given.
- Site keep-outs: boxes from `config/keep_out.local.yaml` (git-ignored; format of `config/keep_out.example.yaml`, which holds one fake box) or from `--keep-out FILE`. The report says which source was used and how many boxes it held, never their coordinates. No site keep-out coordinates are committed to this repository.
- Every program in `results/` passes; on the perimeter transits the checker reproduces the clearances and PTP speeds of `export_report.json` (tested to 0.05 mm / 0.5 mm/s, the rounding of the files).

**Limits.** "No flag" means no flag in this model: a URDF without meshes, capsules and tool geometry estimated, clamp positions estimated from photos, URDF axis speeds and the axis accelerations unverified, S / T bits unverified, blending (C_DIS / C_PTP) and RSI corrections not modelled. It does not replace a check on the controller (T1 dry run, workspace monitoring). Nothing in this repository has been run on the robot.
