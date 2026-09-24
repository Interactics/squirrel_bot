# Model

Every physical spec of the robot as the code builds it — read out of the compiled
MuJoCo model (`core.model.build(**DEFAULT_TAIL)`), not copied from the XML by hand.
The optimal control problem is in [formulation.md](formulation.md); this file is the
thing it is solved for.

**Last updated:** 2026-09-19 · removed stale actuator `ctrlrange` (see changelog)

---

## 1. Overview

One squirrel hindlimb and a one-hinge tail on a torso, in the sagittal plane.

| | |
|---|---|
| degrees of freedom | 8 = planar base 3 (x, z, pitch) + tail 1 + leg 4 (hip, knee, ankle, MTP) |
| actuated | 5 (tail, hip, knee, ankle, MTP); the base is unactuated |
| total mass | 422.5 g |
| sources | `models/squirrel_leg.xml` (torso, leg, contact), `core/model.py` (tail, poses) |
| frame | x forward, z up; the tail hangs off −x (the back) |

The planar base is a deliberate restriction: one leg cannot balance out of plane.
With a free base, the same stance falls over sideways from a 1 mm/s nudge.

## 2. Bodies

Mass, centre of mass in the body's own frame, and principal inertia about that COM.

| body | mass [g] | COM [mm] (x, z) | $I_{xx}, I_{yy}, I_{zz}$ [kg·m²] | geometry |
|---|---|---|---|---|
| torso | 350.0 | (−15, 35) | 1.18e-4, 2.05e-4, 1.97e-4 | ellipsoid, semi-axes 45 × 28 × 30 mm |
| tail | 50.0 | (−135, 0) | 1.82e-4, 1.82e-4, 3.8e-6 | capsule r 10 mm, 180 mm + tip sphere r 16 mm |
| femur | 9.0 | (0, −22.5) | 1.94e-6, 1.94e-6, 7.0e-8 | capsule r 4 mm, 45 mm |
| tibia | 7.0 | (0, −25) | 1.82e-6, 1.82e-6, 5.5e-8 | capsule r 4 mm, 50 mm |
| metatarsus | 4.5 | (0, −15.6) | 7.0e-7, 7.0e-7, 3.6e-8 | capsule r 4 mm, 35 mm + heel pad sphere r 5 mm |
| phalanx | 2.0 | (0, −11.2) | 8.5e-8, 8.5e-8, 4.9e-9 | capsule r 1.5 mm, 18 mm + toe pad sphere r 4 mm |

The tail's 50 g is 25 g of rod plus a **25 g sphere at the tip**. The leg is 22.5 g
in total — 5 % of the robot.

## 3. Kinematics

**Segment lengths** (joint to joint): femur 45, tibia 50, metatarsus 35, phalanx
18 mm. Leg, hip to toe tip: 148 mm. Foot (metatarsus + phalanx): 53 mm.

**Joints.** Zero is the leg straight down with the foot in line with the shank —
full pointe, which the animal never holds; that is why the ankle range sits far from
zero.

| joint | range [deg] | parent → child | note |
|---|---|---|---|
| tail | −150 … +60 | torso → tail | root at torso (−40, 45) mm; the **plan** may use only −110 … +20 |
| hip | −50 … +55 | torso → femur | at the torso frame origin |
| knee | +5 … +135 | femur → tibia | |
| ankle | −160 … −55 | tibia → metatarsus | the "extra gear" |
| MTP | −70 … +20 | metatarsus → phalanx | |

Joint limits are MuJoCo soft constraints (time constant 20 ms), not hard stops.

**Named poses** (hip, knee, ankle, MTP, deg):

| pose | angles | use |
|---|---|---|
| plantigrade | 10.8, 45.5, −146.2, 0.0 | **home pose** (MJCF keyframe), sole flat |
| digitigrade | 6.4, 58.0, −99.4, −55.0 | toes only, heel up |
| crouch | −7.8, 113.2, −140.4, −55.0 | deep crouch on the toes |

## 4. Tail

| | |
|---|---|
| construction | 1 hinge, rod 180 mm × r 10 mm (25 g) + tip sphere r 16 mm (25 g) |
| generator | `core.model.build(n=1, mass=0.025, tip=0.025)` — same tail can be cut into $n$ links |
| COM from hinge | 135 mm |
| inertia about the hinge | 1.09e-3 kg·m² |
| vs torso pitch inertia (about its COM) | **5.3×** (torso 2.05e-4 kg·m²) |
| attitude authority, flight | a tail swing of +1° turns the body about −0.66° (`studies.tail_authority`) |

## 5. Contact

| | |
|---|---|
| contact geoms | every geom collides with the floor; parent/child pairs excluded (MuJoCo default) |
| foot contact in the OCP | toe pad sphere centre, radius 4 mm — a point, transmits no moment |
| friction $\mu$ | 0.9 (sliding), pyramidal cone |
| contact margin | 0 — a contact only exists once geoms overlap |
| stiffness (`solref` time constant) | floor 0.02 s, foot geoms 0.008 s; **mixed ≈ 0.014 s** |
| `solimp` | 0.9, 0.95, 0.001 (MuJoCo default) |

Soft contact has consequences that matter here: contact force lags the plan by
~20 ms at both touch-down and release, and the toe sinks ~3 mm under ~5 N.

## 6. Actuation

| | |
|---|---|
| torque limit $\bar\tau$ | 1.5 N·m, every actuated joint (in the OCP and in the execution harness) |
| MJCF actuators | position servos, $K_p$ = 15 N·m/rad, $K_v$ = 0.15 N·m·s/rad, **no** force limit, **no** control limit — used only for standing and viewing |
| execution harness | overrides them: total torque through the actuator, clipped to ±1.5 N·m (see formulation.md §4) |
| velocity bound (OCP) | 40 rad/s — a numerical bound, not a motor spec; the plan peaks at 17 rad/s (ankle) |
| torque–speed curve | **not modelled** — full torque is allowed at any speed |

## 7. Simulation

| | |
|---|---|
| timestep | 0.2 ms |
| integrator | implicitfast |
| gravity | −9.81 m/s² |
| joint damping, armature | 0, 0 — removed so Pinocchio and MuJoCo describe the same robot (Pinocchio drops armature) |
| model identity | Pinocchio RNEA = MuJoCo `mj_inverse` to 2.7e-15 (contact and limits off); toe frame to 1.1e-16 (`python -m core.model_pin`) |

## 8. Derived quantities

| | |
|---|---|
| COM in the home pose (torso frame, hip at origin) | (−34.1, +103.8) mm — 34 mm **behind** the hip, pulled back by the tail |
| whole-body pitch inertia about the COM (home pose, locked) | 1.69e-3 kg·m² |
| plantigrade support | 53 mm sole; stands, COM 17 mm inside the back edge |
| digitigrade support | 18 mm toe line; topples without control |

## 9. Against a real grey squirrel

Eastern grey squirrel, *Sciurus carolinensis*. Ranges from
[Animal Diversity Web](https://animaldiversity.org/accounts/Sciurus_carolinensis/) and
[Wikipedia](https://en.wikipedia.org/wiki/Eastern_gray_squirrel); the single-animal
measurements are from Fukushima et al. 2021,
[*Inertial tail effects during righting of squirrels in unexpected falls*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8427179/),
Integr. Comp. Biol. 61(2).

| | this model | real squirrel | ratio |
|---|---|---|---|
| total mass | 422.5 g | 341 g (Fukushima); 400–680 g typical | in range |
| **tail mass** | **50 g (11.8 %)** | **11 g (3.3 %)** | **~3.6× the fraction** |
| tail length | 180 mm | 215 mm (Fukushima); 150–250 mm | 0.84× — *shorter* |
| tail inertia at its base | 1.09e-3 kg·m² | 9.99e-5 kg·m² (Fukushima) | **~11×** |
| body length | torso 90 mm | body 164 mm (Fukushima); head–body 230–300 mm | ~0.55× |
| hind foot | 53 mm | 54–76 mm | short end of range |

**So the tail is not too long — it is too heavy, and the torso is too short.**
- The tail is 16 % *shorter* than the measured animal's, but carries 4.5× its mass,
  half of it as a 25 g ball at the tip. Its inertia about the hinge is ~11× the real
  tail's. That tip mass came from the tail study, where it bought pitch authority
  (+1.8× over a bare rod); it was never checked against the animal.
- The torso is about half the real body length, so the tail *looks* long beside it,
  and the whole-body COM ends up 34 mm behind the hip.

A tail matched to the animal would be ~11 g spread along ~200 mm (no tip mass),
with far less attitude authority than now — worth knowing before trusting any
result that leans on the tail.

## 10. Known limitations

- **Point foot.** The OCP foot is one sphere: no ankle moment from the ground, so
  stance on the toe is an inverted pendulum. MuJoCo has more foot geoms (heel pad,
  metatarsus, phalanx) that can touch.
- **Soft contact and soft joint limits** (§5, §3) — the execution differs from the
  rigid-contact plan by ~20 ms at every contact switch.
- **No torque–speed curve** (§6).
- **Planar.** No roll, yaw or sideways motion.
- **One leg.** A squirrel bounds on four; landing is normally taken on the forelimbs.

## 11. Changelog

### 2026-09-19 — stale actuator `ctrlrange` removed

The MJCF gave the leg servos `ctrlrange="-69 92"` etc. and the tail servo
`"-150 60"`, meaning degrees. MuJoCo does not convert `ctrlrange` from degrees, so
they were stored as ±69…150 **radians** — effectively no limit — and the numbers
were also the joint limits from before they were tightened. Removed; the joint limits
do the job. Behaviour unchanged (all selfchecks pass; the execution harness already
disabled the control limit).
