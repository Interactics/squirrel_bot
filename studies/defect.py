"""How faithful is the plan to continuous-time physics?  One-step defects.

    python -m studies.defect

For every interval k: reset EXACTLY to the plan's state (q_k, v_k), hold the
plan's torque tau_k for h_k, integrate the true dynamics finely, and compare with
the plan's next state (q_k+1, v_k+1).  No controller and no error accumulation,
so this measures how wrong ONE step of the transcription is.

Two references:
  rigid   Pinocchio, rigid point contact via KKT, RK4 with SUB substeps
  mujoco  MuJoCo itself: soft contact, 0.2 ms steps

rigid large            -> the TO's time step is too coarse to stand for any
                          continuous trajectory
rigid small, mujoco big -> the gap is MuJoCo's soft contact
both small             -> the plan is faithful; tracking fails for lack of feedback
"""
import numpy as np
import pinocchio as pin
import mujoco

from core.model import DEFAULT_TAIL, build, out
from core.model_pin import load

SUB = 200                      # RK4 substeps per TO interval


def accel(m, d, toe, q, v, tau, contact):
    """Forward dynamics.  In contact the toe is held by a rigid bilateral point
    constraint: [M -J'; J 0][a; f] = [tau - b; -Jdot v]."""
    if not contact:
        return pin.aba(m, d, q, v, tau), np.zeros(2)
    M = pin.crba(m, d, q)
    M = np.triu(M) + np.triu(M, 1).T
    b = pin.rnea(m, d, q, v, np.zeros(m.nv))
    pin.computeJointJacobians(m, d, q)
    pin.updateFramePlacements(m, d)
    J = pin.getFrameJacobian(m, d, toe, pin.LOCAL_WORLD_ALIGNED)[[0, 2], :]
    pin.forwardKinematics(m, d, q, v, np.zeros(m.nv))
    pin.updateFramePlacements(m, d)
    drift = pin.getFrameClassicalAcceleration(m, d, toe, pin.LOCAL_WORLD_ALIGNED).linear[[0, 2]]
    K = np.block([[M, -J.T], [J, np.zeros((2, 2))]])
    sol = np.linalg.solve(K, np.concatenate([tau - b, -drift]))
    return sol[:m.nv], sol[m.nv:]


def rigid_step(m, d, toe, q, v, tau, h, contact):
    """RK4 with SUB substeps.  Returns the end state and the mean contact force."""
    dt, fs = h / SUB, []
    for _ in range(SUB):
        a1, f1 = accel(m, d, toe, q, v, tau, contact)
        a2, _ = accel(m, d, toe, q + dt / 2 * v, v + dt / 2 * a1, tau, contact)
        a3, _ = accel(m, d, toe, q + dt / 2 * (v + dt / 2 * a1), v + dt / 2 * a2, tau, contact)
        a4, _ = accel(m, d, toe, q + dt * (v + dt / 2 * a2), v + dt * a3, tau, contact)
        q = q + dt * (v + dt / 6 * (a1 + a2 + a3))
        v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        fs.append(f1)
    return q, v, np.mean(fs, axis=0)


def mujoco_step(mj, dj, dof, toe_geom, q, v, tau, h):
    mujoco.mj_resetData(mj, dj)
    dj.qpos[:], dj.qvel[:] = q, v
    mujoco.mj_forward(mj, dj)
    n = max(1, int(round(h / mj.opt.timestep)))
    fz, out6 = [], np.zeros(6)
    for _ in range(n):
        dj.qfrc_applied[:] = 0
        dj.qfrc_applied[dof] = tau
        mujoco.mj_step(mj, dj)
        s = 0.0
        for i in range(dj.ncon):
            if toe_geom in (dj.contact[i].geom1, dj.contact[i].geom2):
                mujoco.mj_contactForce(mj, dj, i, out6)
                s += out6[0]
        fz.append(s)
    return dj.qpos.copy(), dj.qvel.copy(), float(np.mean(fz))


def defects(qe, ve, q1, v1):
    return dict(pitch=np.rad2deg(abs(qe[2] - q1[2])), z=abs(qe[1] - q1[1]) * 1e3,
                x=abs(qe[0] - q1[0]) * 1e3, joint=np.rad2deg(np.abs(qe[3:] - q1[3:]).max()),
                vz=abs(ve[1] - v1[1]) * 1e3, wpitch=np.rad2deg(abs(ve[2] - v1[2])),
                vjoint=np.abs(ve[3:] - v1[3:]).max())


def main(path=None):
    z = np.load(path or out("to_jump.npz"))
    Q, V, A, TAU, F, st, t = z["q"], z["v"], z["a"], z["tau"], z["f"], z["stance"], z["t"]
    m, toe, xml, _ = load()
    d = m.createData()
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.actuator_gainprm[:] = 0              # plan torque only, through qfrc_applied
    mj.actuator_biasprm[:] = 0
    dj = mujoco.MjData(mj)
    names = ("tail1", "hip", "knee", "ankle", "mtp")
    dof = np.array([mj.jnt_dofadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in names])
    toe_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "toe_pad")
    N = Q.shape[1] - 1
    h = np.diff(t)

    print("plan: step per phase [ms]", np.round(1e3 * np.unique(np.round(h, 6)), 2),
          "  max|a| joint", np.round(np.abs(A[3:]).max(1)), "rad/s^2",
          "  max|v| joint", np.round(np.abs(V[3:]).max(1), 1), "rad/s\n")

    rows = []
    for k in range(N):
        # Semi-implicit Euler: f_k acts over the whole of interval k, so interval
        # 13 (last push knot -> first flight knot) is simulated IN contact - that
        # is the interval in which the plan both pushes hardest and leaves.
        contact = bool(st[k])
        phase = ("push" if k < 13 else "liftoff" if k == 13 else "flight" if k < 37
                 else "touchdown" if k == 37 else "land")
        tau = np.zeros(m.nv)
        tau[3:] = TAU[:, k]
        qr, vr, fr = rigid_step(m, d, toe, Q[:, k], V[:, k], tau, h[k], contact)
        qm, vm, fm = mujoco_step(mj, dj, dof, toe_geom, Q[:, k], V[:, k], TAU[:, k], h[k])
        rows.append((k, phase, defects(qr, vr, Q[:, k + 1], V[:, k + 1]),
                     defects(qm, vm, Q[:, k + 1], V[:, k + 1]), F[1, k], fr[1], fm))

    print(f"{'phase':<10}{'':>2}{'rigid  pitch':>13}{'z':>7}{'joint':>8}{'vz':>9}"
          f"{'  |  mujoco pitch':>18}{'z':>7}{'joint':>8}{'vz':>9}")
    print(f"{'':<12}{'[deg]':>13}{'[mm]':>7}{'[deg]':>8}{'[mm/s]':>9}"
          f"{'[deg]':>18}{'[mm]':>7}{'[deg]':>8}{'[mm/s]':>9}")
    for ph in ("push", "liftoff", "flight", "touchdown", "land"):
        sel = [r for r in rows if r[1] == ph]
        if not sel:
            continue
        mx = lambda i, key: max(r[i][key] for r in sel)
        print(f"{ph:<10}{len(sel):>2}{mx(2,'pitch'):13.3f}{mx(2,'z'):7.2f}{mx(2,'joint'):8.2f}"
              f"{mx(2,'vz'):9.1f}{mx(3,'pitch'):18.3f}{mx(3,'z'):7.2f}{mx(3,'joint'):8.2f}"
              f"{mx(3,'vz'):9.1f}")

    print("\nstance contact force, plan vs rigid vs mujoco (mean over the interval) [N]:")
    for k, ph, _, _, fp, fr, fm in rows:
        if ph in ("push", "liftoff", "land") and (k % 3 == 0 or k in (12, 13, 38, 39)):
            print(f"   k={k:2d} {ph:<5}  plan {fp:6.2f}   rigid {fr:6.2f}   mujoco {fm:6.2f}")
    return rows


def _selfcheck():
    """RK4 must conserve energy in flight with zero torque, or its defects mean
    nothing."""
    m, toe, _, _ = load()
    d = m.createData()
    q = np.zeros(m.nq); q[3:] = np.deg2rad([20, 10, 60, -100, -40]); q[1] = 0.3
    v = np.zeros(m.nv); v[2] = 3.0; v[3:] = [5, -3, 4, 2, -6]
    E = lambda q, v: (0.5 * v @ (np.triu(pin.crba(m, d, q)) + np.triu(pin.crba(m, d, q), 1).T) @ v
                      + pin.computePotentialEnergy(m, d, q))
    e0 = E(q, v)
    q1, v1, _ = rigid_step(m, d, toe, q, v, np.zeros(m.nv), 0.02, False)
    drift = abs(E(q1, v1) - e0) / abs(e0)
    assert drift < 1e-6, f"RK4 energy drift {drift:.1e}"
    print(f"selfcheck ok (RK4 energy drift over 20 ms: {drift:.1e})\n")


if __name__ == "__main__":
    _selfcheck()
    main()
