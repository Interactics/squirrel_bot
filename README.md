# Squirrel Robot 
Squirrel-inspired robot

One squirrel hindlimb plus a tail, in the sagittal plane: model, whole-body
trajectory optimisation for a jump, and the checks that decide whether a solved
trajectory is real.

![jump and landing](docs/jump.gif)

Crouch, push 114 ms, fly 260 ms, land - 80 mm of COM rise, landing foot placed
150 mm ahead of the take-off foot to within 0.01 mm, peak joint torque 903 mN m.
This is the plan, played back kinematically; tracking it in closed loop is the
whole-body controller's job and is not written yet.

```
models/squirrel_leg.xml   the model.  Hand-written, with marker slots the tail is
                          spliced into - edit this, never a generated file
core/                     the library
  model.py                build the MJCF, pose it, stand it on the floor, film it
  model_pin.py            the same robot in Pinocchio, plus collision sample points
  to_jump.py              the optimal control problem
  verify_to.py            solve it, and generate collision constraints from MuJoCo
  view.py                 live window or GIF
studies/                  one-off experiments, each with its own selfcheck
out/                      everything generated (gitignored)
```

## Running

```bash
python -m core.verify_to            # solve the jump and verify it
python -m core.view gif out/to_jump.npz
python -m studies.stance          # plantigrade stands, digitigrade topples
python -m studies.tail_authority  # what one tail hinge buys in flight
python -m studies.tail_tuning     # does a redundant tail buy anything?  (no)
python -m studies.reach           # landing accuracy and how far it can go
```

macOS live viewer needs `mjpython`, which needs `otool`:

```bash
PATH="/Library/Developer/CommandLineTools/usr/bin:$PATH" mjpython -m core.view live plantigrade
```

## The problem

Direct collocation on the full planar dynamics. Three phases with free durations
and a fixed contact schedule — stance $\mathcal{C}$, flight $\mathcal{F}$, stance —
so no complementarity is needed. $N+1 = 53$ knots.

$q_k \in \mathbb{R}^8$ is the floating base $(x, z, \theta)$ followed by the tail,
hip, knee, ankle and MTP angles. Only the last five are actuated, which
$S \in \mathbb{R}^{5\times 8}$ selects — the three base rows carry no torque and
must be satisfied by contact force alone. That is the underactuation.

Minimise effort, with small regularisers on acceleration and body pitch:

$$\min_{q,\,v,\,a,\,\tau,\,f,\,\Delta t}\; \sum_k h_k \lVert \tau_k \rVert^2 \;+\; 0.02 \sum_k \lVert a_k \rVert^2 \;+\; 0.01 \sum_k \theta_k^2$$

Full rigid-body dynamics and trapezoidal collocation, at every knot:

$$M(q_k)\,a_k + b(q_k, v_k) = S^{\top}\tau_k + J(q_k)^{\top} f_k$$

$$q_{k+1} = q_k + \tfrac{h_k}{2}(v_k + v_{k+1}), \qquad v_{k+1} = v_k + \tfrac{h_k}{2}(a_k + a_{k+1})$$

In stance the toe stays put and the ground may only push, inside the friction cone:

$$p_z(q_k) = r, \quad p_x(q_k) = \text{const}, \quad f_{k,z} \ge 0, \quad \lvert f_{k,x} \rvert \le \mu f_{k,z} \qquad k \in \mathcal{C}$$

In flight there is no contact force, and nothing may go under the floor:

$$f_k = 0, \qquad p_z(q_k) \ge r \qquad k \in \mathcal{F}$$

Limits, and clearance $c(q) \ge 0$ — 17 rows of floor and trunk, plus whatever
segment pairs MuJoCo reports as overlapping:

$$\lvert \tau_k \rvert \le \bar\tau, \quad \lvert v_k \rvert \le \bar v, \quad q^- \le q_k \le q^+, \quad c(q_k) \ge 0$$

Start level and at rest; land level, at rest, and with the foot already matching
the ground so there is no impact to model:

$$v_0 = v_N = 0, \quad \theta_0 = \theta_N = 0, \quad J(q_{\mathrm{td}})\,v_{\mathrm{td}} = 0$$

And the two things actually being asked for — jump this high, land there:

$$\dot{c}_z(q_{\mathrm{lo}}, v_{\mathrm{lo}}) = \sqrt{2 g\, \Delta h}, \qquad p_x(q_{\mathrm{td}}) - p_x(q_0) = d$$

$p(q)$ is the toe-sphere centre and $r$ its radius, so $p_z = r$ is exact for a
sphere on a plane. $c_z$ is the centre of mass height. $\mathrm{lo}$ is the **first
flight** knot, not the last stance knot: trapezoidal integration keeps adding ground
reaction through $a_k$ at the last stance knot, so constraining liftoff velocity
there asks for 80 mm of rise and delivers 178 mm.

| | | |
|---|---|---|
| $\bar\tau$ | 1.5 N·m | generous, but finite — unbounded torque lets the optimiser mine integration error instead of the robot |
| $\mu$ | 0.9 | matches the MJCF |
| $r$ | 4 mm | toe sphere radius |
| $\Delta h$, $d$ | 80 mm, 150 mm | the two things actually being asked for |
| $h_k$ | 2–10 / 4–14 / 2–10 ms | stance steps stay short enough that trapezoidal integration is honest |

**Where to land is a constraint, not a cost.** Height, landing spot and landing
attitude are all equalities; the objective only buys them cheaply. That is why the
landing error is 0.01 mm rather than "close" — and why an unreachable target fails
outright instead of degrading gracefully. Maximising height *in the objective*,
with no ceiling, produced a 5.5 m jump on 750 N of ground reaction: integration
error, not a squirrel.

## Three layers of checking

A converged solve proves nothing.  Every trajectory passes through:

1. **IPOPT converged** - says only that the equations were satisfied.
2. **`core.to_jump.audit`** - in flight the COM must be a parabola with exactly
   -9.81, the rise must be the one that was asked for, nothing dips through the
   floor.  This caught a jump that was perfectly physical and 2.2x too high,
   because the liftoff constraint sat one knot too early.
3. **MuJoCo narrowphase** - the collision constraints in the OCP are point
   samples, an approximation.  MuJoCo's own detector is the referee, and it
   caught the tail sweeping 20 mm through the metatarsus while the sampled
   constraints reported everything fine.

Collision constraints are generated lazily from (3): solve, ask MuJoCo what
overlapped, constrain those pairs, solve again.  Two rounds is typical.
Constraining every pair up front is 64 rows a knot and segfaults IPOPT.

## What the model is not

Squirrels bound on four limbs.  This is one hindlimb in a plane, so it can speak
to hindlimb torque budget, ankle loading and tail attitude authority - and cannot
speak to fore/hind coordination, spine flexion, or roll.  The landing constraint
in particular asks a single hindlimb to absorb a landing the animal takes on its
forelimbs.
