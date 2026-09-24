# Formulation

The optimal control problem exactly as `core/to_jump.py` builds it, how it is
solved, and what changed each time. Every equation here maps to a line of code;
if they ever disagree, the code is wrong or this file is stale — both are bugs.

**Last updated:** 2026-09-18 · liftoff complementarity, flight toe margin, tail headroom, attitude feedback (run `20260918-172312`, track `20260918-172313`)

---

## 1. Model

Planar (sagittal) floating base, one hindlimb, one-hinge tail.

$$q = (x,\ z,\ \theta,\ \varphi_{\mathrm{tail}},\ q_{\mathrm{hip}},\ q_{\mathrm{knee}},\ q_{\mathrm{ankle}},\ q_{\mathrm{mtp}}) \in \mathbb{R}^8, \qquad v,\ a \in \mathbb{R}^8$$

$(x, z, \theta)$ is the unactuated base. The five joints are actuated:

$$S = \begin{bmatrix} 0_{5\times 3} & I_5 \end{bmatrix}, \qquad \tau \in \mathbb{R}^5$$

Contact is a single point: the toe-sphere centre $p(q) \in \mathbb{R}^2$ (x, z),
radius $r$. Contact force $f \in \mathbb{R}^2$ acts there, with
$J(q) = \partial p / \partial q \in \mathbb{R}^{2\times 8}$ (world-aligned). A point
contact transmits no moment.

Rigid-body dynamics (Pinocchio RNEA, identical to MuJoCo `mj_inverse` to 2.7e-15
with contact and joint limits disabled):

$$M(q)\,a + b(q, v) = S^{\top}\tau + J(q)^{\top} f$$

| parameter | value | source |
|---|---|---|
| total mass | 422.5 g | torso 350 g, tail rod 25 g + tip 25 g, leg 22.5 g |
| segments | femur 45, tibia 50, metatarsus 35, phalanx 18 mm | `models/squirrel_leg.xml` |
| tail | 180 mm, 1 hinge, root at torso $(-40, 45)$ mm | `core/model.py` `DEFAULT_TAIL` |
| joint limits $q^\pm$ [deg] | tail −150…60, hip −50…55, knee 5…135, ankle −160…−55, mtp −70…20 | MJCF ranges |
| $r$ | 4 mm | toe sphere radius |
| $\mu$ | 0.9 | MJCF friction |
| $\bar\tau$ | 1.5 N·m | `TAU_MAX`, every actuated joint |
| $\bar v$ | 40 rad/s or m/s | `V_MAX`, keeps the NLP bounded |

Full physical specs (inertias, contact, actuators, comparison with a real squirrel)
are in [model.md](model.md).

## 2. Discretisation

Fixed contact schedule, $N = 52$ intervals, $N+1 = 53$ knots:

| phase | knots | contact | step $h_k$ |
|---|---|---|---|
| push $\mathcal{P}$ | $0 \dots 13$ | toe | $\Delta t_{\mathcal{P}} \in [2, 10]$ ms |
| flight $\mathcal{F}$ | $14 \dots 37$ | none | $\Delta t_{\mathcal{F}} \in [4, 14]$ ms |
| land $\mathcal{L}$ | $38 \dots 52$ | toe | $\Delta t_{\mathcal{L}} \in [2, 10]$ ms |

The three $\Delta t$ are decision variables (so phase durations are free). Stance
knots $\mathcal{C} = \mathcal{P} \cup \mathcal{L}$; liftoff knot $\mathrm{lo} = 14$
(first flight knot); touchdown knot $\mathrm{td} = 38$.

Integration is **semi-implicit Euler**:

$$v_{k+1} = v_k + h_k\, a_k$$

$$q_{k+1} = q_k + h_k\, v_{k+1}$$

## 3. The optimal control problem

Decision variables: $q_k, v_k, a_k, \tau_k, f_k$ for $k = 0 \dots N$, and
$\Delta t_{\mathcal{P}}, \Delta t_{\mathcal{F}}, \Delta t_{\mathcal{L}}$.

### Objective

Every term is a time integral of a quantity divided by a reference scale, so the
weights are comparable and mean what they say:

$$J = J_\tau + w_a J_a + w_\theta J_\theta + w_L J_L + w_j J_{\mathrm{jerk}}$$

$$J_\tau = \sum_{k=0}^{N} h_k \left\lVert \frac{\tau_k}{\bar\tau} \right\rVert^2, \qquad J_a = \sum_{k=0}^{N} h_k \left\lVert \frac{a_k}{\bar a} \right\rVert^2$$

$$J_\theta = \sum_{k=0}^{N} h_k \left(\frac{\theta_k}{\bar\theta}\right)^2, \qquad J_L = \sum_{k=0}^{N} h_k \left(\frac{L_k}{\bar L}\right)^2$$

$L_k$ is the centroidal angular momentum about the centre of mass (the $y$
component, i.e. about the pitch axis), from Pinocchio's centroidal momentum:

$$L_k = \big[\, A_G(q_k)\, v_k \,\big]_{\text{angular},\,y}$$

In flight $L$ is conserved, so $w_L$ is mostly a price on the spin the body
leaves the ground with — which the tail would otherwise have to cancel.

$$J_{\mathrm{jerk}} = \sum_{k \notin \{13,\,37\}} \frac{\lVert a_{k+1} - a_k \rVert^2}{h_k}$$

The sums run over all knots $k = 0 \dots N$, with $h_N := h_{N-1}$. They must:
$a_N$, $\tau_N$, $f_N$ move nothing (integration stops at $N-1$), so a cost that
leaves them out leaves them free.

| scale / weight | value | role |
|---|---|---|
| $\bar\tau$ | 1.5 N·m | $=\bar\tau$ of the limits |
| $\bar a$, $w_a$ | 100 rad/s², $10^{-2}$ | acceleration — a regulariser, not a goal |
| $\bar\theta$, $w_\theta$ | 10°, $0.1$ | keep the body roughly level |
| $\bar L$, $w_L$ | 5 mN·m·s, $1$ | angular momentum |
| $w_j$ | $0$ | jerk, off (see changelog) |

### Constraints

**Dynamics and integration**

$$M(q_k)\,a_k + b(q_k, v_k) = S^{\top}\tau_k + J(q_k)^{\top} f_k \qquad k = 0 \dots N$$

$$v_{k+1} = v_k + h_k a_k, \qquad q_{k+1} = q_k + h_k v_{k+1} \qquad k = 0 \dots N-1$$

**Stance** $k \in \mathcal{C}$

$$f_{k,z} \ge 0, \qquad \lvert f_{k,x} \rvert \le \mu\, f_{k,z}$$

$$p_z(q_k) = r, \qquad p_x(q_k) = p_x(q_{k_0}) \qquad k \le N-1$$

where $k_0$ is the first knot of that stance phase. The position rows stop at
$N-1$: with $v_N = 0$, semi-implicit Euler gives $q_N = q_{N-1}$ exactly, and
repeating them would make the constraint Jacobian rank-deficient.

**Liftoff complementarity** — at the last push knot a contact that is still pushing
holds the toe still:

$$\ddot p(q_{13}, v_{13}, a_{13}) = J(q_{13})\,a_{13} + \dot J(q_{13}, v_{13})\,v_{13} = 0$$

**Flight** $k \in \mathcal{F}$

$$f_k = 0, \qquad p_z(q_k) \ge r$$

$$p_z(q_k) \ge r + c_{\mathcal{F}} \qquad 16 \le k \le 35, \quad c_{\mathcal{F}} = 10\ \text{mm}$$

— a toe margin mid-flight, not at the knots right after liftoff or before touchdown.

**Touchdown**, so there is no impact to model:

$$J(q_{\mathrm{td}})\, v_{\mathrm{td}} = 0$$

**Limits** for all $k$:

$$\lvert \tau_k \rvert \le \bar\tau, \qquad \lvert v_k \rvert \le \bar v, \qquad q^- \le q_{k,\,3:8} \le q^+$$

except the tail, which the plan may only use inside $[-110°, +20°]$ of its
$[-150°, +60°]$ joint range. The rest is reserved for attitude feedback.

**Clearance** $c(q_k) \ge 0$ for $k \le N-1$, three families of rows over sampled
points $s_i(q)$ with radii $\rho_i$ (`core/model_pin.py`):

$$s_{i,z}(q) - \rho_i \ge 0 \qquad i \in \text{FLOOR}$$

$$\left\lVert \frac{\ell_i(q) - c_T}{a_T + \rho_i} \right\rVert \ge 1 \qquad i \in \text{TRUNK\_OUT}$$

$$\lVert s_i(q) - s_j(q) \rVert \ge \rho_i + \rho_j + m \qquad (i, j) \in \mathcal{P}$$

FLOOR is 9 leg points + 5 tail points. $\ell_i$ is point $i$ in the torso frame;
the trunk is the ellipse centre $c_T = (-15, 35)$ mm, semi-axes $a_T = (45, 30)$ mm.
$m = 2$ mm. The pair set $\mathcal{P}$ starts empty and is grown by MuJoCo (§4).

**Boundary**

$$x_0 = 0, \quad \theta_0 = 0, \quad v_0 = 0, \qquad v_N = 0, \quad \theta_N = 0$$

The start *posture* (joint angles) is free — the optimiser picks its own crouch.

**What is being asked for** — jump height $\Delta h$ and landing stride $d$:

$$\dot c_z(q_{\mathrm{lo}}, v_{\mathrm{lo}}) - \tfrac{g}{2}\Delta t_{\mathcal{F}} = \sqrt{2 g\, \Delta h}$$

$$p_x(q_{\mathrm{td}}) - p_x(q_0) = d$$

with $\Delta h = 80$ mm, $d = 150$ mm by default, $c$ the centre of mass. The
$\tfrac{g}{2}\Delta t_{\mathcal{F}}$ term: semi-implicit Euler under constant
gravity lays the flight positions on the exact parabola launched with
$v - gh/2$, so this makes the *discrete* trajectory rise $\Delta h$.

### Scaling

Not cosmetic — each of these was the difference between converging and not.

| quantity | written as | why |
|---|---|---|
| $\Delta t$ | milliseconds | in seconds (~5e-3) IPOPT stalled beside O(1) torque rows |
| kinematic rows ($p$, $J v$, reach) | × 10³ (mm) | ~4e-3 m rows were left loose next to O(1) dynamics |
| clearance | distances in mm, not squared | squared residuals in the thousands segfaulted Hessian assembly |
| trunk row | × 10² | same |

## 4. How it is solved

**Solver.** CasADi `Opti` → IPOPT 3.14.11 with MUMPS, exact Hessian.
`tol 1e-5`, `acceptable_tol 1e-4` for 10 iterations, `mu_strategy adaptive`,
`max_iter 4000`.

**Initial guess** (cold): every knot at `start_pose()` — the crouch with the toe
exactly on the floor, level, analytic; $\Delta t = (6, 13, 6)$ ms; $f_z = 4$ N;
everything else zero.

**Homotopy on jump height.** Solve $\Delta h = 20 \to 40 \to 60 \to 80$ mm, each
warm-started from the previous solution. A cold start straight at 80 mm does not
converge.

**Lazy collision constraints** (`core/verify_to.py`):

1. Solve with $\mathcal{P} = \emptyset$.
2. Replay the trajectory through MuJoCo's narrowphase; collect every geom pair that
   overlaps by more than 0.1 mm anywhere.
3. Map each geom to its sample points, add those pairs to $\mathcal{P}$, re-solve.
4. Stop when MuJoCo reports nothing (at most 6 rounds; 2 is typical, with 8 pairs).

Constraining every pair up front is 64 rows per knot and segfaults IPOPT.

**Verification** — a trajectory counts only if all of these pass:

| check | passes when |
|---|---|
| IPOPT | converged |
| `audit()` | flight COM is a parabola with $\lvert g + 9.81\rvert < 0.05$; toe and tail never below the floor; rise within 10 % of $\Delta h$ |
| MuJoCo narrowphase | no geom pair overlaps anywhere |

**Outputs, every run.** `core.verify_to` writes `out/runs/<stamp>/plan.csv` (and
`.npz`): one row per knot with $t$, phase, $q$, $v$, $\tau$, $f$ and $L$.
`studies.track` writes each execution of that plan into its own subfolder,
`out/runs/<stamp>/track-<time>/`, so re-running never overwrites: `physics_ff`
(feedforward only) and `physics_ffpd` (feedforward + PD), every 1 ms — $q$, $v$, the
torque each joint actually received, toe contact force, reference $q$ — plus GIFs of
plan beside physics at 1/10 speed with time and phase stamped on each frame.
`out/to_jump.npz` carries the stamp so plan and executions stay paired.

**Execution harness** (`studies.track`). Each actuated joint gets
$\tau = \mathrm{clip}\big(\tau_{\mathrm{ff}} + K_p(q_{\mathrm{ref}} - q) + K_d(v_{\mathrm{ref}} - \dot q),\ \pm\bar\tau\big)$
with $\tau_{\mathrm{ff}}$ the plan torque held zero-order, $K_p = 15$, $K_d = 0.3$. The
whole sum goes through the MuJoCo actuator (gain 1, bias $[0, -K_p, -K_d]$,
$\mathrm{ctrl} = \tau_{\mathrm{ff}} + K_p q_{\mathrm{ref}} + K_d v_{\mathrm{ref}}$), so the
clip applies to the total and $-K_d\dot q$ is integrated implicitly.

The third execution, `physics_fb`, adds attitude feedback on top (pitch error
$e = \theta - \theta_{\mathrm{ref}}$). Tail, all phases — a tail swing of +1° turns the
body about −0.66°, so the tail target moves *with* the error:

$$\varphi_{\mathrm{cmd}} = \mathrm{clip}\big(\varphi_{\mathrm{ref}} + K_\theta e + K_\omega \dot e,\ [-145°, 58°]\big), \quad K_\theta = 3,\ K_\omega = 0.1\ \mathrm{s}$$

Hip, stance only — with the foot down the hip sits between the body and a leg that
is going nowhere, so shifting its target by $-e$ turns the body back:

$$q_{\mathrm{hip,cmd}} = q_{\mathrm{hip,ref}} - K_h e - K_{hd}\dot e, \quad K_h = 1,\ K_{hd} = 0.05\ \mathrm{s}$$

**Retargeting** (`solve_reach`): move $d$ from the current solution to a new
target in 4 warm-started steps.

## 5. Current result

Plan run `20260918-172312`, execution `track-20260918-172313`; data in
`out/runs/20260918-172312/`.

| | |
|---|---|
| COM rise | 79.9 mm (asked 80) |
| flight gravity (audit) | −9.81 m/s² |
| landing stride | +150.0 mm (asked 150) |
| phase durations | push 90 ms, flight 232 ms, land 115 ms |
| peak $\lvert\tau\rvert$ | 551 mN·m (was 1040 before the liftoff complementarity row) |
| push force, last three knots | 12.9, 12.9, 13.5 N — no liftoff spike (was 30.7 N) |
| collision rounds | 2, 8 pairs, MuJoCo clean |

**Executing it** (§4 harness):

| | feedforward only | + joint PD | + attitude feedback |
|---|---|---|---|
| peak joint torque [mN·m] | 551 | 1137 | 1500 (at the limit) |
| max pitch error, up to plan end | 137° | 86° | **23°** |
| max x / z error, up to plan end | 173 / 68 mm | 115 / 126 mm | 139 / 46 mm |

What the attitude-feedback run actually does, which the summary hides:

| | plan | physics |
|---|---|---|
| toe leaves the ground | 90 ms | **69 ms** (21 ms early) |
| toe lands | 322 ms | **423 ms** (100 ms late) |
| at plan end (437 ms) | — | x +139 mm, pitch −10°, toe just touching |
| after the plan ends | — | pitch +28° at 500 ms, +178° at 600 ms: falls |

It flies further and higher than planned and lands at the very end of the plan;
then nothing holds it up — the plan's last state is not a standing equilibrium on a
point foot, and there is no standing controller. Without the hip term the toe leaves
on time (91 ms vs 90) and does not drag in flight, but the angular momentum error at
liftoff then spins it; with the hip term the push is altered and the takeoff
velocity is off. Still to solve: a stance controller that corrects attitude
*without* changing the takeoff velocity (TVLQR / WBC on base pitch and COM).

**Why the tail moves** (plan). In flight $L \approx 0$, but the leg changes shape
for landing, and a shape change rotates the body even at zero angular momentum. For
run `20260918-115722`: the leg alone would pitch the body +3.03° over the flight;
the tail's swing gives −3.41°; net −0.38°.

## 6. Changelog

Newest first. Each entry is the mathematical difference, not the code diff.

### 2026-09-18 — liftoff complementarity, flight toe margin, tail headroom; attitude feedback

Found by executing the plan in MuJoCo and tracing why it fell (toe release time,
contact force in flight, angular momentum).

- **Added** $\ddot p(q_{13}, v_{13}, a_{13}) = 0$. *Why:* the plan put its largest push
  (30.7 N) into the interval in which the toe accelerated off the ground. Physics
  cannot deliver that; the real toe left 21–32 ms early, the jump came out low, and
  the leg — tracking its planned angles — dragged the foot through the "flight"
  (1–12 N of ground force), which spun the body up. After: push force 12.9, 12.9,
  13.5 N over the last three knots, peak torque 1040 → 551 mN·m, and executed
  without the hip term the toe leaves at 91 ms against 90 planned.
- **Added** $p_z(q_k) \ge r + 10$ mm for $16 \le k \le 35$ (was $\ge r$). *Why:* a
  slightly low jump dragged the toe; the margin absorbs it (drag in flight 0 N).
- **Tail range in the plan** $[-150°, 60°] \to [-110°, 20°]$. *Why:* once $w_L$
  made the tail unnecessary, the optimiser parked it at 56–60°, against its stop;
  feedback then saturated at the stop and could not stop a spin.
- **Execution** (not OCP): `physics_fb` adds tail and stance-hip attitude feedback
  (§4). Max pitch error to plan end 86° → 23°; still lands 100 ms late and falls
  after the plan ends (§5).

### 2026-09-18 — execution harness: torque limit on the total (not an OCP change)

- The OCP is unchanged; only how its plan is executed in `studies.track`.
- Before: $\tau = \mathrm{clip}\big(K_p(q_{\mathrm{ref}}-q) + K_d(v_{\mathrm{ref}}-\dot q), \pm\bar\tau\big) + \tau_{\mathrm{ff}}$
  — the feedforward was added outside the clip. After: the clip covers the sum (§4).
- Effect on the same plan: peak joint torque 4476 → 1500 mN·m; max z error
  688 → 43 mm; max pitch error 465° → 75°. The leg-runaway diagnosis is withdrawn;
  body pitch drift is what is left.

### 2026-09-18 — objective rescaled; angular momentum term

- **Objective** changed from

  $$\sum_{k<N} h_k\lVert\tau_k\rVert^2 + 0.02\sum_k\lVert a_k\rVert^2 + 0.01\sum_k\theta_k^2$$

  to $J_\tau + w_a J_a + w_\theta J_\theta + w_L J_L$ (§3). *Why:* the old "small" acceleration regulariser was unscaled and not
  time-weighted; with joint accelerations in the hundreds of rad/s² it was ~3×10⁴
  against an effort term of ~0.2. The TO was minimising acceleration, not torque,
  and an angular momentum term had no effect at any weight (tested $w$ = 10⁻³ …
  10⁻¹ in the old units: identical solutions).
- **Angular momentum** term added, $w_L = 1$. Sweep with the new scaling:

  | $w_L$ | $L$ in flight [mN·m·s] | tail range | $\int\lVert\tau\rVert^2dt$ |
  |---|---|---|---|
  | 0 | +3.11 | −50 … +28° | 0.0696 |
  | 0.1 | +0.44 | −8 … +6° | 0.0506 |
  | 1 | −0.02 | −23 … −15° | 0.0553 |
  | 10 | −0.04 | +54 … +60° | 0.0551 |

  $w_L = 0$ is not the cheapest in effort — it is a worse local optimum.
- **All knots in the cost**, including $k = N$. Leaving $a_N, \tau_N, f_N$ out made
  them free: the first run returned $f_N$ = 121 kN.
- Physics execution unchanged in character: still falls (§5).

### 2026-09-18 — semi-implicit Euler

- **Integration** changed from trapezoidal
  $q_{k+1} = q_k + \tfrac{h}{2}(v_k + v_{k+1})$,
  $v_{k+1} = v_k + \tfrac{h}{2}(a_k + a_{k+1})$
  to semi-implicit Euler $v_{k+1} = v_k + h a_k$, $q_{k+1} = q_k + h v_{k+1}$.
  *Why:* trapezoidal only fixes $a_k + a_{k+1}$. With the toe velocity zero at every
  stance knot, consecutive toe accelerations had to cancel, so the large toe
  acceleration liftoff needs echoed back through the whole push as a ±12.15 m/s²
  alternating chain (and touchdown did the same through landing). Joint torque
  zig-zagged ~300 mN·m knot to knot while $q$, $v$ and every audit looked clean.
  Zig-zag metric 461 → 30.
- **Liftoff condition** $\dot c_z(\mathrm{lo}) = \sqrt{2g\Delta h}$ became
  $\dot c_z(\mathrm{lo}) - \tfrac{g}{2}\Delta t_{\mathcal{F}} = \sqrt{2g\Delta h}$.
  Without it: 80 mm asked, 74.4 mm delivered.
- **Stance position rows and clearance** dropped at $k = N$ (duplicate of $N-1$ once
  $q_N = q_{N-1}$); first attempt failed with `Error_In_Step_Computation`.

### 2026-09-18 — jerk penalty (kept, weight 0)

- Added $w_j \sum_{k \notin \{13, 37\}} \lVert a_{k+1} - a_k \rVert^2 / h_k$.
  Swept $w_j = 10^{-7} \dots 10^{-2}$: zig-zag 461 → 197 at best, and $J_{\mathrm{jerk}}$
  plateaued at ~1.0e7 regardless of weight — the oscillation was forced by the
  constraints, not chosen. Superseded by the integrator change; $w_j = 0$.

### Before that — the path to the baseline

- Contact-knot and scaling fixes: $\Delta t$ in ms, kinematic rows in mm; toe velocity
  constraint only at touchdown (at $k=0$ it duplicated $v_0 = 0$).
- $\bar\tau$ 5 → 1.5 N·m and height moved from the objective (maximise) into a
  constraint: maximising with no ceiling found a 5.5 m jump on 750 N — integration
  error, not a squirrel.
- Liftoff condition moved from the last stance knot to the first flight knot
  (trapezoidal: asked 80 mm, got 178 mm).
- Landing stride $d$ added as an equality; default $d = +150$ mm (forward).
- Start posture freed (only $x_0, \theta_0, v_0$ pinned): the pinned crouch put the
  toe 30 mm in front of the COM, the wrong side to push forward from.
- Clearance $c(q) \ge 0$: floor, trunk exclusion, then lazily generated segment pairs.
- Joint limits tightened to anatomical ranges (§1); plantigrade is the home pose.
