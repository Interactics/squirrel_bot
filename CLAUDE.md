# Working in this repo

## formulation.md is part of every change (user instruction, 2026-09-18)

The user reads `formulation.md` to check, **mathematically**, what changed each time
the code runs. Treat it as part of the code, not documentation:

- Any change to the OCP or how it is solved updates `formulation.md` **in the same
  change**. That covers `core/to_jump.py` (variables, objective, constraints,
  integrator, bounds, scaling, solver options, homotopy), `core/model_pin.py`
  (contact point, collision samples, trunk, margins), `core/verify_to.py` (the
  solve/verify procedure) and anything in `models/squirrel_leg.xml` or
  `core/model.py` that enters the OCP (masses, lengths, joint ranges, tail).
- Update the equations and tables so they match the code exactly, and add a
  **changelog entry at the top of §6**: dated, stating the mathematical difference
  (old equation → new equation) and why.
- **§5 numbers come from an actual run** of `python -m core.verify_to` (and
  `python -m studies.track` for the execution line) made after the change — never
  from memory or from an earlier run. If a run was not made, say so in §5.
- Formulas as short `$$...$$` blocks (GitHub renders them). One long `aligned`
  block with an annotation column overflowed the page once; keep each block narrow.

## model.md is the robot's spec sheet (user instruction, 2026-09-19)

- Any change to the physical model — masses, lengths, inertias, joint ranges, the
  tail, contact or actuator parameters, simulation settings — updates `model.md` in
  the same change, with a dated changelog entry (old value -> new value, why).
- Read the numbers out of the compiled model (`mujoco.MjModel` from
  `core.model.build`), not off the XML: the XML once said degrees where MuJoCo
  stored radians.
- Keep §9 (comparison with a real squirrel) honest when the tail or torso changes.

## Every run saves its data (user instruction, 2026-09-18)

- `python -m core.verify_to` writes `out/runs/<stamp>/plan.{csv,npz}` (per knot:
  t, phase, q, v, tau, f, L) and stamps `out/to_jump.npz` with the same run id.
- `python -m studies.track` writes each execution of that plan into its own
  subfolder `out/runs/<stamp>/track-<time>/`: `physics_ff.{csv,npz,gif}` and
  `physics_ffpd.{csv,npz,gif}`, `physics_fb.{csv,npz,gif}` (attitude feedback) (every 1 ms: q, v, the torque each joint actually
  received, toe contact force, reference q; GIF = plan | physics, 1/10 speed).
- Any new pipeline that produces a trajectory saves it the same way (via
  `core.model.save_table` into the run folder). Never overwrite a run folder.
- `formulation.md` §5 names the run id its numbers came from.

## Environment

- Python: `/opt/homebrew/anaconda3/envs/mujoco_env/bin/python` (mujoco 3.9,
  casadi 3.7.2, pinocchio 4.0). Run modules from the repo root: `python -m core.verify_to`.
- Live viewer on macOS: `PATH="/Library/Developer/CommandLineTools/usr/bin:$PATH" mjpython -m core.view live plantigrade`.
- Generated files go to `out/` (gitignored); `docs/` is tracked.

## Checking a run

Read the **whole** log, not a grep of the result lines: a `NameError` after the last
printed result once meant `to_jump.npz` was never written, and later tests silently
used a stale plan.
