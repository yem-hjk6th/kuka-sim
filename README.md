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
- `kukasim/planner.py`: DP over (point x redundancy-angle bin x IK slot); reports infeasible stretches and breaks.
- `kukasim/collision.py`: capsules for links and tool, boxes for bed and clamps (sizes in `config/planner.yaml`, all estimated).
- `kukasim/toolpaths.py`: dome, saddle, sine wave, 45° ramp and a bed perimeter path.
- `kukasim/slicer_import.py`: optional reader for `non_planar_slicer` KRL output (separate module; no GPL code).
- `validate_fk.py`: compares FK with pendant (A1..A6, X..C) pairs.
- `results/`, `RESULTS.md`: outputs of the run for the PR.
