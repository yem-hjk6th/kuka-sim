# Step 1: classical baseline for nozzle orientation along non-planar toolpaths

## Goal
In simulation only, find joint trajectories for the KR 10 R1100-2 that follow a non-planar printing toolpath with the nozzle along a given direction at every point.
- **The free choice** is the rotation about the nozzle axis (the redundancy angle) and the IK configuration.
- **The cost** is joint-limit and singularity margin plus smooth joint motion. Collisions with the bed and the clamp are not allowed.
- **The method** is dynamic programming over the discretised redundancy angle.

This is the baseline that any later learning-based method must beat.

## Inputs
- `config/cell.yaml`: tool (TOOL 3), base, joint limits, bed and the known obstacle. Treat every value whose status is `estimated` or `unknown` as a parameter, never as a constant in code.
- `robot/kr10_r1100_2/urdf/kr10_r1100_2.urdf`. The meshes are absent, so use primitive collision shapes (see `robot/kr10_r1100_2/meshes/README.md`).
- Toolpaths: generate test surfaces procedurally (dome, saddle, sine wave, a 45° ramp) over the bed area, then layer paths with a nozzle direction from the surface normal and a maximum tilt parameter. Optionally read KRL `.src` produced by `non_planar_slicer` (GPL-3.0). Keep any code that touches it in a separate module.

## Deliverables
1. **Kinematics.**
   - FK and IK for the robot with TOOL 3, using KUKA's ABC convention (A about Z, then B about Y, then C about X).
   - IK returns all valid configurations. Handle A1/A4/A6 wrap and the turn-count ambiguity explicitly.
   - **B ≈ 90° is an Euler singularity.** The logged print poses sit there, for example A 84.1, B 89.9, C 116.2. Work with rotation matrices internally and convert to ABC only at the boundary.
2. **Validation hook.** `validate_fk.py` reads a CSV of (A1..A6, X, Y, Z, A, B, C) pairs and reports position and orientation errors. Ye will collect 20–30 such pairs from the pendant. Ship an example file with the single known pair from the README of this task's PR, if you can derive it, otherwise an empty template.
3. **Planner.**
   - DP over (toolpath point × redundancy angle bin × IK configuration).
   - Hard constraints: joint limits with a margin, collision-free against the bed (with a gap parameter) and the clamp box, and a maximum joint step per point at 12 mm/s with the URDF velocity limits.
   - Costs: distance to joint limits, a wrist-singularity measure (|A5| near 0) or manipulability, and joint-space smoothness.
   - Report infeasible stretches instead of failing silently.
4. **Outputs per toolpath:**
   - the joint trajectory (CSV);
   - plots of A1–A6, the chosen redundancy angle and the margins along the path;
   - a feasibility map (point × angle);
   - a short animation (pybullet or matplotlib) or a frame sequence;
   - a KRL export of the points with ABC. The export starts with a comment block `; NOT RUN - generated in simulation, never executed on a robot`.
5. **Tests.**
   - DP equals brute force on a small case.
   - FK/IK round trip on random reachable poses.
   - Constraints are respected on every output point.
   - The NOT RUN header is present in every KRL export.
6. **`RESULTS.md`:** for each test surface, the reachable tilt range, the infeasible stretches and the runtime.

## Known risks (address or report)
- **Narrow redundancy window.** The tip is ~73 mm off the A6 axis, and A5 is limited to ±120°. Earlier work found only about 235–285° of continuous turn. Some paths may have no feasible solution; report that as a result.
- **The URDF is checked at one pose only.** Do not tune anything to it.
- **Interpolation is not modelled.** The controller's LIN/C_DIS blending is not in the sim, so densify waypoints to 1–2 mm.

## Do not
- Connect to any robot or controller, or write any code that sends motion.
- Commit meshes, ROMs or anything over 5 MB.
- Add Co-Authored-By trailers. Commit author is `yem-hjk6th <183082151+yem-hjk6th@users.noreply.github.com>`.
