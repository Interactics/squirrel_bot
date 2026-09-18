"""How much body pitch does one tail hinge actually buy?

In flight there is no support polygon and no ground reaction - gravity acts at
the COM and applies no torque about it, so total angular momentum is conserved.
Swinging the tail one way therefore counter-rotates the body the other way.

    python -m studies.tail_authority

The point of this file: DOF count is not authority.  Authority is I_tail times
how far you can swing it.  One hinge already delivers most of what a 20-joint
tail could, for one actuator and one control input.
"""
import numpy as np
import mujoco

from core.model import POSES, Recorder, aid, qadr, setup, tail_joints


def predicted(m, d):
    """Pitch gain per radian of tail, from momentum conservation in free flight.

    The conserved quantity is momentum about the system COM, not about the pitch
    hinge - which sits at the hip, not at the COM.  So the base translation DOFs
    have to be eliminated too, or the tail comes out looking much weaker than it
    is.  With the base block b = (slide_x, slide_z, pitch) and the tail j:

        M_bb * v_b + M_bj * qdot_j = 0   ->   v_b = -inv(M_bb) * M_bj * qdot_j

    and the pitch row of that is the gain."""
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    dof = lambda j: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
    b = [dof("slide_x"), dof("slide_z"), dof("pitch")]
    j = dof(tail_joints(m)[0])
    gain = -np.linalg.solve(M[np.ix_(b, b)], M[b, j])[2]
    return gain, M[b[2], j], M[b[2], b[2]]


def swing(pose="digitigrade", start=60.0, end=-150.0, T=0.35, gif=None):
    """Launch into free flight, sweep the tail start -> end, report body pitch.

    Pass gif="name.gif" to also record it."""
    m, d = setup(pose, tail_deg=start)
    d.qpos[qadr(m, "slide_z")] += 0.5          # up in the air, no contacts
    m.opt.gravity[:] = 0                       # gravity applies no torque about the
                                               # COM anyway; dropping it keeps the
                                               # bookkeeping exact instead of merely
                                               # true-on-average
    mujoco.mj_forward(m, d)

    ratio0, i_tail, i_tot = predicted(m, d)
    tail = tail_joints(m)[0]
    zp, ta, tq = qadr(m, "pitch"), aid(m, tail), qadr(m, tail)
    pitch0, prev = d.qpos[zp], d.qpos[tq]

    # The gain is not constant - the coupling inertia swings with the tail - so
    # integrate it along the trajectory instead of freezing it at t=0.
    rec = Recorder(m, gif) if gif else None
    est = 0.0
    n = int(T / m.opt.timestep)
    for k in range(n):
        frac = 0.5 - 0.5 * np.cos(np.pi * k / n)          # smooth start and stop
        d.ctrl[ta] = np.deg2rad(start + (end - start) * frac)
        mujoco.mj_step(m, d)
        r, _, _ = predicted(m, d)
        est += r * (d.qpos[tq] - prev)
        prev = d.qpos[tq]
        if rec:
            rec.grab(d, k)
    if rec:
        rec.save()

    swept = np.rad2deg(d.qpos[tq]) - start                      # signed
    got = np.rad2deg(d.qpos[zp] - pitch0)
    est = np.rad2deg(est)
    print(f"  coupling M[pitch,tail] {i_tail:.3e}   locked M[pitch,pitch] {i_tot:.3e}"
          f"   gain at t=0 {ratio0:.2f}")
    print(f"  tail swept {swept:+7.1f} deg  ->  body pitched {got:+7.1f} deg "
          f"(momentum integral {est:+.1f})")
    return got, swept, est


if __name__ == "__main__":
    print("free flight, one tail hinge, leg servos holding the digitigrade shape:")
    got, swept, est = swing(gif="squirrel_tail_swing.gif", T=0.6)

    assert abs(got) > 20, "a 180 mm tail should be worth more than 20 deg of pitch"
    assert got * swept < 0, "body must counter-rotate against the tail"
    # momentum conservation is the physics; if these two drift apart, something
    # is injecting angular momentum that should not be (a stray contact, a limit).
    assert abs(got - est) < 0.15 * abs(est), \
        f"measured {got:.1f} deg should track the momentum integral {est:.1f} deg"
    print("\nselfcheck ok")
