"""Trajectory optimisation for a squirrel jump: crouch -> push -> fly -> land.

    python -m core.to_jump

Direct transcription (semi-implicit Euler) on the full planar dynamics, three phases with
free durations.  Contact schedule is fixed - stance, flight, stance - so no
complementarity is needed.

The interesting constraint is the landing: the toe must arrive with zero
velocity.  That removes the impact entirely, so the optimiser has to plan the
whole flight around arriving correctly - which is exactly what forces it to use
the tail.
"""
import numpy as np
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

from core.model import out
from core.model_pin import (ACT, FLOOR, MARGIN, MU, NB, TOE_R, TRUNK_AB, TRUNK_C,
                       TRUNK_OUT, TAIL_NAMES, load)

N_PUSH, N_FLY, N_LAND = 14, 24, 14
# 0.5 N m was the servo we started with; 1.5 is "unshackled but still a motor you
# could buy for a 420 g machine".  Leaving it huge lets the optimiser exploit the
# integrator instead of the robot - at 5 N m it found a 5.5 m jump on 750 N of
# ground force, which is integration error, not a squirrel.
TAU_MAX = 1.5                       # N m
V_MAX = 40.0                        # rad/s or m/s, keeps the NLP bounded
RISE = 0.080                        # m of COM rise to ask for
# +x is FORWARD; the tail hangs off -x, which is the back.  Left free, the
# optimiser jumps backwards, because the tail drags the COM 34 mm behind the hip
# and pushing off a foot that sits in front of the COM throws you that way.
REACH = 0.150                       # m of forward stride, foot to foot
# Jerk penalty, off.  It was tried against the trapezoidal zig-zag and could not
# remove it - the oscillation was forced by the constraints, and the jerk term
# plateaued at ~1e7 whatever its weight.  The integrator change fixed it instead.
# See formulation.md, changelog.
W_JERK = 0.0

# Objective: every term is a time integral of a quantity divided by a reference
# scale, so the weights mean what they say.  They did not before: the "small"
# regulariser 0.02*sum|a|^2 (unscaled, rad/s^2 in the hundreds) was ~3e4 against
# an effort term of ~0.2 - the TO was minimising acceleration, not torque, and an
# angular momentum term had no say at any weight.
A_REF = 100.0                  # rad/s^2 (or m/s^2 on the base)
TH_REF = np.deg2rad(10.0)      # body pitch
L_REF = 5e-3                   # N m s, centroidal angular momentum about the COM (y)
W_A = 1e-2                     # acceleration: a regulariser, not a goal
W_TH = 0.1                     # keep the body roughly level
# Angular momentum.  In flight L is conserved, so this is mostly a price on the
# spin the body leaves the ground with - which the tail then has to cancel.
W_L = 1.0

# Leave the execution something to work with.  Each of these was found by running
# the plan in MuJoCo and watching it fall.
# Tail range the PLAN may use; the rest of the joint range (-150..60) is reserved
# for attitude feedback.  Unrestricted, the optimiser parked the tail at 56-60 deg,
# against its stop, and feedback could only push it one way.
TAIL_PLAN = np.deg2rad([-110.0, 20.0])
# Toe height margin mid-flight.  With none, a jump a little lower than planned
# dragged the foot along the floor, and the ground spun the body up.
FLIGHT_CLEAR = 0.010


def make_funcs(model, toe_id, frames, pairs=()):
    """`pairs` are the point-pairs to keep apart, supplied by the caller after
    MuJoCo says which geoms actually overlapped."""
    seg, torso_f = frames["seg"], frames["torso"]
    cmodel = cpin.Model(model)
    nq, nv = cmodel.nq, cmodel.nv
    q, v, a, f = (ca.SX.sym("q", nq), ca.SX.sym("v", nv),
                  ca.SX.sym("a", nv), ca.SX.sym("f", 2))

    d1 = cmodel.createData()
    rnea = ca.Function("rnea", [q, v, a], [cpin.rnea(cmodel, d1, q, v, a)])

    d2 = cmodel.createData()
    cpin.framesForwardKinematics(cmodel, d2, q)
    toe = ca.Function("toe", [q], [d2.oMf[toe_id].translation[[0, 2]]])

    d3 = cmodel.createData()
    cpin.computeJointJacobians(cmodel, d3, q)
    cpin.updateFramePlacements(cmodel, d3)
    Jf = cpin.getFrameJacobian(cmodel, d3, toe_id, pin.LOCAL_WORLD_ALIGNED)[[0, 2], :]
    Jtoe = ca.Function("Jtoe", [q], [Jf])

    d4 = cmodel.createData()
    com = ca.Function("com", [q, v], [cpin.centerOfMass(cmodel, d4, q, v),
                                      d4.vcom[0]])

    # ---- collisions -------------------------------------------------------
    # All of it in ONE function returning residuals that must be >= 0.  Building
    # these as ~40 separate inline expressions per knot blew CasADi's expression
    # graph and segfaulted the process; one call per knot keeps the NLP small.
    d5 = cmodel.createData()
    cpin.framesForwardKinematics(cmodel, d5, q)
    res = []

    for name in FLOOR:                         # everything above the floor
        fid, r = seg[name]
        res.append((d5.oMf[fid].translation[2] - r) * 1e3)

    oMt = d5.oMf[torso_f]                      # leg outside the trunk ellipsoid
    for name in TRUNK_OUT:
        fid, r = seg[name]
        loc = oMt.actInv(d5.oMf[fid].translation)[[0, 2]]
        # Distances, not squared distances: squared residuals came out in the
        # thousands next to dynamics rows of order one, and IPOPT hates that.
        res.append((ca.sqrt(ca.sumsqr((loc - TRUNK_C) / (TRUNK_AB + r)) + 1e-12)
                    - 1.0) * 1e2)

    for na, nb in pairs:                       # only the pairs that actually bind
        fa, ra = seg[na]
        fb, rb = seg[nb]
        diff = (d5.oMf[fa].translation - d5.oMf[fb].translation)[[0, 2]]
        res.append((ca.sqrt(ca.sumsqr(diff) + 1e-12) - (ra + rb + MARGIN)) * 1e3)

    clear = ca.Function("clear", [q], [ca.vertcat(*res)])

    d9 = cmodel.createData()
    L = ca.Function("L", [q, v], [cpin.computeCentroidalMomentum(cmodel, d9, q, v).angular[1]])

    d10 = cmodel.createData()
    cpin.forwardKinematics(cmodel, d10, q, v, a)
    cpin.updateFramePlacements(cmodel, d10)
    toe_acc = ca.Function("toe_acc", [q, v, a], [cpin.getFrameClassicalAcceleration(
        cmodel, d10, toe_id, pin.LOCAL_WORLD_ALIGNED).linear[[0, 2]]])

    return dict(nq=nq, nv=nv, rnea=rnea, toe=toe, Jtoe=Jtoe, com=com, clear=clear, L=L,
                toe_acc=toe_acc)


def solve(q_start, rise=RISE, reach=REACH, warm=None, verbose=False, max_iter=4000,
          pairs=(), w_jerk=W_JERK, w_L=W_L, w_a=W_A, w_th=W_TH):
    model, toe_id, _, frames = load()
    F = make_funcs(model, toe_id, frames, pairs)
    nq, nv, nu = F["nq"], F["nv"], 5
    N = N_PUSH + N_FLY + N_LAND
    stance = np.array([k < N_PUSH or k >= N_PUSH + N_FLY for k in range(N + 1)])

    opti = ca.Opti()
    Q = opti.variable(nq, N + 1)
    V = opti.variable(nv, N + 1)
    A = opti.variable(nv, N + 1)
    U = opti.variable(nu, N + 1)
    Fc = opti.variable(2, N + 1)
    # Durations live in MILLISECONDS as decision variables.  In seconds they are
    # ~5e-3 while torques are ~1, and IPOPT stalls on that scaling: it parks at
    # residuals of 1e-4 (physically nothing on a 420 g robot) and never certifies.
    dt_ms = opti.variable(3)                    # one duration per phase
    dt = dt_ms / 1000.0

    opti.subject_to(opti.bounded(2.0, dt_ms[0], 10.0))     # push
    # 12 ms cap, not 20: at long reach the optimiser was drifting the flight-phase
    # COM to an implied g of -9.95 by leaning on trapezoidal error.
    opti.subject_to(opti.bounded(4.0, dt_ms[1], 14.0))     # flight
    opti.subject_to(opti.bounded(2.0, dt_ms[2], 10.0))     # land
    step = ca.vertcat(*[dt[0]] * N_PUSH, *[dt[1]] * N_FLY, *[dt[2]] * N_LAND)

    S = np.zeros((nu, nv))
    S[:, ACT] = np.eye(nu)

    for k in range(N + 1):
        opti.subject_to(F["rnea"](Q[:, k], V[:, k], A[:, k])
                        == S.T @ U[:, k] + F["Jtoe"](Q[:, k]).T @ Fc[:, k])
        opti.subject_to(opti.bounded(-TAU_MAX, U[:, k], TAU_MAX))
        opti.subject_to(opti.bounded(-V_MAX, V[:, k], V_MAX))
        for j, lo, hi in _limits(model):
            opti.subject_to(opti.bounded(lo, Q[j, k], hi))

        # With semi-implicit Euler and V[:, N] == 0, q[N] == q[N-1] exactly, so any
        # position-level row at the last knot duplicates the one before it - the
        # same rank deficiency that k == 0 once caused, now at the other end.
        last = k == N
        if not last:
            opti.subject_to(F["clear"](Q[:, k]) >= 0)

        p = F["toe"](Q[:, k]) * 1e3
        if stance[k]:
            if not last:
                opti.subject_to(p[1] == TOE_R * 1e3)             # on the ground
            # Velocity-level too would be differentially redundant with the
            # position constraint at every knot, and rank-deficient constraint
            # Jacobians read to IPOPT as infeasibility.  Impose it once per
            # stance phase; the position constraints carry the rest.
            # Only at touchdown.  At k == 0 this would duplicate V[:, 0] == 0
            # exactly, and a rank-deficient constraint Jacobian is what IPOPT
            # reports as "max iterations".
            if k == N_PUSH + N_FLY:
                opti.subject_to(F["Jtoe"](Q[:, k]) @ V[:, k] * 1e3 == 0)
            opti.subject_to(Fc[1, k] >= 0)                       # push, never pull
            # Complementarity at the last push knot: a contact that is pushing holds
            # the toe still.  Without it the plan put its largest push (30.7 N) into
            # the same interval in which the toe accelerated off the ground - which
            # physics cannot do, so the real toe left 20-30 ms early.
            if k == N_PUSH - 1:
                opti.subject_to(F["toe_acc"](Q[:, k], V[:, k], A[:, k]) == 0)
            opti.subject_to(opti.bounded(-MU * Fc[1, k], Fc[0, k], MU * Fc[1, k]))
        else:
            opti.subject_to(Fc[:, k] == 0)
            opti.subject_to(p[1] >= TOE_R * 1e3)                 # stay above the floor
            if N_PUSH + 2 <= k <= N_PUSH + N_FLY - 3:
                opti.subject_to(p[1] >= (TOE_R + FLIGHT_CLEAR) * 1e3)

    # Semi-implicit Euler, not trapezoidal.  Trapezoidal only pins a[k] + a[k+1],
    # and in stance the toe velocity is ~0 at every knot, so consecutive toe
    # accelerations must cancel.  The big toe acceleration that liftoff needs then
    # echoed back through the whole push as a +-12 m/s^2 chain, and touchdown did
    # the same forward through landing: torque zig-zagging 300 mN m knot to knot.
    # Here each a[k] owns exactly one interval, so a liftoff spike stays one spike.
    for k in range(N):
        h = step[k]
        opti.subject_to(V[:, k + 1] == V[:, k] + h * A[:, k])
        opti.subject_to(Q[:, k + 1] == Q[:, k] + h * V[:, k + 1])

    # the foot does not move within either stance phase
    for a, b in ((0, N_PUSH - 1), (N_PUSH + N_FLY, N - 1)):   # N duplicates N-1
        for k in range(a + 1, b + 1):
            opti.subject_to((F["toe"](Q[:, k])[0] - F["toe"](Q[:, a])[0]) * 1e3 == 0)

    # Start: level, at rest, on the toe, at the origin - but the CROUCH ITSELF is
    # free.  Pinning it forced the foot 30 mm in front of the COM, which is the
    # wrong side to push from, and no forward jump was reachable from there.
    # q_start is only the initial guess now.
    opti.subject_to(Q[0, 0] == 0)
    opti.subject_to(Q[2, 0] == 0)
    opti.subject_to(V[:, 0] == 0)
    opti.subject_to(V[:, N] == 0)
    opti.subject_to(Q[2, N] == 0)                                # land level

    # Ask for a specific jump and buy it as cheaply as possible.  Maximising
    # height with no ceiling just drives every bound to its limit.
    # The FIRST FLIGHT knot, not the last stance knot.  Trapezoidal integration
    # gives v[N_PUSH] = v[N_PUSH-1] + dt/2*(a[N_PUSH-1] + a[N_PUSH]), and
    # a[N_PUSH-1] still carries ground reaction - so the robot is still gaining
    # speed after the last stance knot.  Constraining there under-shoots: asking
    # for an 80 mm rise bought 178 mm.
    lift = N_PUSH
    _, vcom = F["com"](Q[:, lift], V[:, lift])
    # Semi-implicit Euler under constant gravity puts the flight positions on the
    # exact parabola launched with v - g h/2, not v.  Ask for THAT velocity, or
    # an 80 mm request comes out as 74 mm.
    opti.subject_to(vcom[2] - 9.81 * dt[1] / 2 == np.sqrt(2 * 9.81 * rise))

    # Where to land.  Without this the optimiser picks whatever is cheapest,
    # which turns out to be 54 mm BACKWARDS.  Measured foot-to-foot, so it is
    # the stride, not a base coordinate that drifts with posture.
    td = N_PUSH + N_FLY
    if reach is not None:
        opti.subject_to((F["toe"](Q[:, td])[0] - F["toe"](Q[:, 0])[0] - reach) * 1e3 == 0)
    # Sums run over ALL knots, k = 0..N.  a_N, tau_N and f_N move nothing (the
    # integration stops at N-1), so if the cost leaves them out they are free -
    # the first rescaled run came back with f_N = 121 kN.  The last knot reuses
    # the last step length.
    hk = lambda k: step[min(k, N - 1)]
    effort = sum(hk(k) * ca.sumsqr(U[:, k] / TAU_MAX) for k in range(N + 1))
    acc = sum(hk(k) * ca.sumsqr(A[:, k] / A_REF) for k in range(N + 1))
    tilt = sum(hk(k) * (Q[2, k] / TH_REF) ** 2 for k in range(N + 1))
    # sum h |jerk|^2 with jerk = (a[k+1] - a[k]) / h.  Skipped across liftoff and
    # touchdown: contact switching on or off makes acceleration jump for real,
    # and penalising that fights the physics instead of the artefact.
    switch = {N_PUSH - 1, N_PUSH + N_FLY - 1}
    jerk = sum(ca.sumsqr(A[:, k + 1] - A[:, k]) / step[k]
               for k in range(N) if k not in switch)
    ang = sum(hk(k) * (F["L"](Q[:, k], V[:, k]) / L_REF) ** 2 for k in range(N + 1))
    opti.minimize(effort + w_a * acc + w_th * tilt + w_L * ang + w_jerk * jerk)

    if warm is None:
        opti.set_initial(Q, np.tile(q_start.reshape(-1, 1), N + 1))
        opti.set_initial(dt_ms, [6.0, 13.0, 6.0])
        opti.set_initial(Fc[1, :], 4.0)
    else:
        for var, key in ((Q, "q"), (V, "v"), (A, "a"), (U, "tau"), (Fc, "f")):
            opti.set_initial(var, warm[key])
        opti.set_initial(dt_ms, warm["dt"] * 1000.0)
    # Exact Hessian.  The collision rows once segfaulted IPOPT here, but that was
    # their SCALING - squared distances gave residuals in the thousands next to
    # dynamics rows of order one.  With distances in millimetres it is fine, and
    # 14x faster than the limited-memory approximation that was papering over it.
    opts = {"ipopt.max_iter": max_iter, "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 1e-4, "ipopt.acceptable_iter": 10,
            "ipopt.mu_strategy": "adaptive"}
    if not verbose:
        opts.update({"ipopt.print_level": 0, "print_time": 0})
    opti.solver("ipopt", opts)

    try:
        s = opti.solve()
    except RuntimeError:
        print("\n--- solver failed; worst constraint violations ---")
        opti.debug.show_infeasibilities(1e-4)
        print("dt_ms =", opti.debug.value(dt_ms))
        print("pitch range deg:", np.rad2deg(opti.debug.value(Q[2, :])).round(1))
        print("toe z:", [float(opti.debug.value(F["toe"](Q[:, k])[1]))
                         for k in range(0, N + 1, 6)])
        print("fz:", opti.debug.value(Fc[1, :]).round(2))
        raise
    return dict(q=s.value(Q), v=s.value(V), a=s.value(A), tau=s.value(U),
                f=s.value(Fc), dt=s.value(dt), stance=stance, model=model,
                toe=toe_id, rise=rise,
                reach=float(s.value(F["toe"](Q[:, td])[0] - F["toe"](Q[:, 0])[0])))


def solve_reach(q_start, reach, warm, rise=RISE, n=4, pairs=()):
    """Walk the landing target from wherever the warm start lands to `reach`."""
    for x in np.linspace(warm["reach"], reach, n + 1)[1:]:
        warm = solve(q_start, rise=rise, reach=float(x), warm=warm, pairs=pairs)
    return warm


def solve_homotopy(q_start, target=RISE, reach=REACH, pairs=(),
                   steps=(0.02, 0.04, 0.06, None), **kw):
    """Walk the jump height up, warm-starting each solve from the last.

    Cold-starting straight at the target just runs IPOPT out of iterations: the
    all-knots-at-the-crouch guess is nowhere near any feasible jump."""
    warm = None
    for rise in steps:
        rise = target if rise is None else min(rise, target)
        warm = solve(q_start, rise=rise, reach=reach, warm=warm, pairs=pairs, **kw)
        print(f"  rise {rise*1000:4.0f} mm  ok   "
              f"peak |tau| {np.abs(warm['tau']).max()*1e3:5.0f} mNm   "
              f"push {warm['dt'][0]*N_PUSH*1000:4.0f} ms   "
              f"reach {warm['reach']*1000:+5.0f} mm")
        if rise >= target:
            break
    return warm


def audit(r):
    """In flight the COM must be a parabola with exactly -g, and the toe must
    never dip below the floor.  Cheap, and it is what caught the liftoff-knot
    bug: the trajectory was perfectly physical, just not the jump we asked for."""
    model, toe_id, _, frames = load()
    data = model.createData()
    q, dt = r["q"], r["dt"]
    z = np.array([pin.centerOfMass(model, data, q[:, k]).copy()[2]
                  for k in range(N_PUSH, N_PUSH + N_FLY)])
    t = np.arange(len(z)) * dt[1]
    g = np.polyfit(t, z, 2)[0] * 2
    toe_z = []
    for k in range(q.shape[1]):
        pin.framesForwardKinematics(model, data, q[:, k])
        toe_z.append(data.oMf[toe_id].translation[2])
    rise = z.max() - z[0]
    assert abs(g + 9.81) < 0.05, f"flight COM implies g = {g:.2f}, not -9.81"
    assert min(toe_z) > TOE_R - 1e-6, f"toe dips to {min(toe_z)*1e3:.3f} mm"
    clear = np.inf
    for k in range(q.shape[1]):
        pin.framesForwardKinematics(model, data, q[:, k])
        for name in TAIL_NAMES:
            fid, rad = frames["seg"][name]
            clear = min(clear, data.oMf[fid].translation[2] - rad)
    assert clear > -1e-6, f"tail goes {clear*1e3:.1f} mm through the floor"
    assert abs(rise - r["rise"]) < 0.1 * r["rise"], \
        f"asked for {r['rise']*1e3:.0f} mm of rise, got {rise*1e3:.0f} mm"
    return dict(g=g, rise=rise, apex=z.max(), min_toe=min(toe_z), tail_clear=clear)


def _limits(model):
    out = []
    for j in range(NB, model.nq):
        lo, hi = model.lowerPositionLimit[j], model.upperPositionLimit[j]
        if j == NB:                                   # the tail: keep headroom
            lo, hi = max(lo, TAIL_PLAN[0]), min(hi, TAIL_PLAN[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out.append((j, lo, hi))
    return out


def start_pose():
    """The crouch, level, with the toe exactly on the ground.

    Built analytically, not by letting MuJoCo settle: the settled crouch topples,
    so after 0.3 s it sits at -25 deg of pitch and resting on the wrong geom - a
    start state no jump should have to recover from."""
    from core.model import POSES
    model, toe_id, _, _ = load()
    data = model.createData()
    q = np.zeros(model.nq)
    q[NB:] = np.concatenate([[0.0], np.deg2rad(POSES["crouch"])])   # tail, then leg
    pin.framesForwardKinematics(model, data, q)
    q[1] += TOE_R - data.oMf[toe_id].translation[2]                 # drop onto the toe
    return q


if __name__ == "__main__":
    q0 = start_pose()
    print("homotopy on jump height:")
    r = solve_homotopy(q0)
    t = np.concatenate([[0], np.cumsum(np.repeat(r["dt"], [N_PUSH, N_FLY, N_LAND]))])
    print(f"\nphase durations  push {r['dt'][0]*N_PUSH*1000:.0f} ms   "
          f"flight {r['dt'][1]*N_FLY*1000:.0f} ms   land {r['dt'][2]*N_LAND*1000:.0f} ms")
    print(f"peak |tau| per joint [mNm]: "
          + "  ".join(f"{n} {abs(r['tau'][i]).max()*1e3:6.0f}"
                      for i, n in enumerate(("tail", "hip", "knee", "ankle", "mtp"))))
    print(f"peak contact force  fz {r['f'][1].max():.2f} N   "
          f"= {r['f'][1].max()/(0.4225*9.81):.1f} x body weight")
    print(f"apex torso z {r['q'][1].max()*1000:.1f} mm   "
          f"pitch range {np.rad2deg(r['q'][2]).min():+.0f} .. "
          f"{np.rad2deg(r['q'][2]).max():+.0f} deg   "
          f"tail range {np.rad2deg(r['q'][3]).min():+.0f} .. "
          f"{np.rad2deg(r['q'][3]).max():+.0f} deg")
    a = audit(r)
    print(f"landing foot {r['reach']*1000:+.1f} mm from the take-off foot")
    print(f"audit: flight g {a['g']:+.2f} m/s2   COM rise {a['rise']*1000:.1f} mm "
          f"(asked {r['rise']*1000:.0f})   apex {a['apex']*1000:.1f} mm   "
          f"min toe {a['min_toe']*1000:.2f} mm   "
          f"tail clearance {a['tail_clear']*1000:+.2f} mm")

    print("\nreachable landing targets (same 80 mm rise):")
    for tgt in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
        try:
            rr = solve_reach(q0, tgt, r)
            print(f"  target {tgt*1000:+5.0f} mm -> landed {rr['reach']*1000:+6.1f} mm"
                  f"   peak |tau| {np.abs(rr['tau']).max()*1e3:5.0f} mNm"
                  f"   fz {rr['f'][1].max():5.1f} N")
        except RuntimeError:
            print(f"  target {tgt*1000:+5.0f} mm -> no solution")

    np.savez(out("to_jump.npz"), **{k: r[k] for k in ("q", "v", "a", "tau", "f", "dt")},
             stance=r["stance"], t=t)
    print("\nwrote to_jump.npz")
