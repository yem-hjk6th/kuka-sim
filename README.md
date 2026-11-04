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

`kukasim/src_check.py` reads a `.src` file and replays every PTP / LIN on the models above (URDF FK / IK, `CollisionModel`, the PTP speed model of `transit.py`); it adds no kinematics or collision model of its own. Per motion it reports: reach inside the joint limits minus `use_margin_deg` (a LIN is sampled and one IK branch is followed from its start; a configuration change or a joint step above `lin_max_joint_step_deg` is flagged), |A5|, elbow and shoulder margins, the TCP speed (LIN: `$VEL.CP`; PTP: the M1 flag uses the cruise bound, i.e. the TCP speed of the synchronous PTP with its leading axis at its `$VEL_AXIS` speed, which does not depend on the axis accelerations; the ramp-limited peak at `$ACC_AXIS` of the assumed axis accelerations is reported for information only; always at 100 % override, since a PC under RSI or the pendant can set any override), the minimum clearance and its primitive / obstacle pair (bed, clamps, keep-out boxes, floor; the nozzle tip is exempt from the bed on LIN moves only), the tip height above the highest bed top and, on LIN moves, the tip XY inside the bed outline minus a margin. Thresholds: `src_check` in `config/planner.yaml` (defaults taken from `transit` and `hilbert_corner`).

- Accepted statements (an explicit allow-list): axis and Cartesian `PTP` / `LIN` targets (optional `S`, `T`, `E1..E4`, `C_DIS` / `C_PTP`); settings `BAS(#TOOL / #BASE / #INITMOV / #VEL_PTP / #PTP_PARAMS)`, `$TOOL = TOOL_DATA[n]`, `$BASE = $NULLFRAME` / `BASE_DATA[n]`, `$VEL.CP`, `$VEL_AXIS[i]`, `$ACC_AXIS[i]`, `$APO.CDIS`, `$ORI_TYPE`, `PDAT_ACT.VEL` / `.ACC` (used by `BAS(#PTP_PARAMS, v)`: `$VEL_AXIS` = the larger of `v` and `PDAT_ACT.VEL`, an upper bound for either use, and `$ACC_AXIS` = `PDAT_ACT.ACC`); without effect on the checks and listed in the report: `PDAT_ACT.APO_DIST`, `$BWDSTART`, `$ADVANCE`; without motion: `DECL`, assignments to user variables whose right-hand side is a literal, a variable or arithmetic without calls, `$OV_PRO`, `WAIT SEC`, `HALT`, `END`, comments, `RSI_CREATE / RSI_ON / RSI_OFF`; control flow with call-free conditions: `IF / ELSE / ENDIF`, `SWITCH / CASE / DEFAULT / ENDSWITCH`, `FOR`, `WHILE`, `LOOP`, `REPEAT / UNTIL`, `EXIT`.
- Anything else, including any call of a function or subroutine (`ret = MyMoveFct(1)`, `MySub()`, `BAS(#VEL_CP, ...)`), is "not interpreted" (exit code 2). After it, every setting it may change is unknown (an assignment such as `$TOOL = $NULLFRAME`: that setting; a call: every modelled setting and the pose), and every later motion that depends on it is "not checked" until a modelled statement sets it again. A motion that cannot be read is a parse error. Only TOOL 3 and BASE 0 are modelled; any other frame is an error. A write to `TOOL_DATA[n]` / `BASE_DATA[n]` (a whole frame or one field, e.g. `TOOL_DATA[3].X = 400`) is not interpreted: frame n is unknown from there on, also after `BAS(#TOOL, n)` / `$TOOL = TOOL_DATA[n]`, so every later Cartesian motion that uses it is not checked. `$VEL.CP`, `$VEL_AXIS`, `$ACC_AXIS`, `BAS(#VEL_PTP / #PTP_PARAMS)` and `PDAT_ACT.VEL` / `.ACC` must be finite and > 0; any other value (e.g. `$VEL.CP = -0.5`) is an error (exit code 2) and leaves the setting unknown. An `$ORI_TYPE` other than `#VAR` / `#CONSTANT`, or one that is unknown, makes the path of every dependent LIN not checked.
- Branches and loops: every branch is followed and every loop is run until no new setting state appears. A motion is checked with the worst case of every value a setting may hold there (highest `$VEL.CP`, `$VEL_AXIS`, `$ACC_AXIS`; not checked if a TOOL / BASE other than 3 / 0 may be active). A motion whose start depends on a branch is checked from every possible start; one whose start depends on a loop is not checked. A `#VAR` LIN that turns the tool by about 180 deg (more than 180 - `lin_ori_ambiguous_deg`) has no unique shortest rotation: its path is not checked, only its target pose.
- DEF sections: each `DEF` in a file is checked on its own and reported as its own section. A `DEF` after the first one starts with an unknown pose and unknown settings (they come from the caller): its first motion is not checked, and every setting (`$TOOL`, `$BASE`, `$VEL.CP`, `$ORI_TYPE`, ...) must be set inside it before a motion that depends on it is checked.
- Output: `<program>_check.md` and `<program>_check.json` next to the program or in `--out`; a program under `config/` or `results/` is reported into `out/src_check/` (the checker never writes into `config/` or `results/`).
- Exit codes: **0** = every motion checked and no flag; **1** = at least one flag; **2** = parse / frame error or a statement not interpreted; **3** = no flag, but at least one motion not checked. The summary counts the motions not checked and the report lists why.
- The first motion of a program (PTP or Cartesian LIN) starts from an unknown pose: it is "not checked" (target pose only) unless `--start-axis A1 A2 A3 A4 A5 A6` is given, so a program without `--start-axis` gives at best exit code 3.
- Site keep-outs: boxes from `config/keep_out.local.yaml` (git-ignored; format of `config/keep_out.example.yaml`, which holds one fake box) or from `--keep-out FILE`. The report says which source was used and how many boxes it held, never their coordinates. No site keep-out coordinates are committed to this repository.
- Every program in `results/` gives no flag (exit code 3: only its first PTP, from the unknown current pose, is not checked). For every program with an `export_report.json` the checker reproduces its numbers: transit clearances and PTP speeds (cruise and ramp-limited peak; perimeter, the only one with transits), the `$VEL_AXIS` / `$ACC_AXIS` of the export, and the joint values of every printed point in `trajectory.csv` (tested to 0.05 mm, 0.5 mm/s and 0.05 mm / 1.5 m in joint space, the rounding of the files).

**Limits.** "No flag" means no flag in this model: a URDF without meshes, capsules and tool geometry estimated, clamp positions estimated from photos, URDF axis speeds and the axis accelerations unverified, S / T bits unverified, blending (C_DIS / C_PTP) and RSI corrections not modelled. The tool model is only the bracket, extruder barrel, nozzle and tip capsules: nothing else mounted on the tool (no camera cage or other attachment) is modelled. The bed is modelled flat at its highest measured top (`bed_top_max_z`), so the nozzle-height check of first-layer points on a sloped layer is conservative (the real bed is lower in places). It does not replace a check on the controller (T1 dry run, workspace monitoring). Nothing in this repository has been run on the robot.
