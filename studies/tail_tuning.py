"""Does a redundant tail buy attitude authority?  And what does a tip mass buy?

    python -m studies.tail_tuning

Builds tail variants into squirrel_leg.xml's marker slots and swings each one in
free flight.  The question is not "can we control it" - it is whether the extra
DOFs raise the ceiling at all.
"""
import numpy as np
import mujoco

from core.model import build


def dofs(m, *joints):
    return [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in joints]


def gain(m, d, tail_joints, drive):
    """Pitch produced per radian commanded, from momentum conservation about the
    system COM.  `drive` weights how the commanded motion is split across the
    tail joints, so a curling tail and a straight one are scored on equal terms."""
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    b = dofs(m, "slide_x", "slide_z", "pitch")
    t = dofs(m, *tail_joints)
    coupling = M[np.ix_(b, t)] @ np.asarray(drive)
    return -np.linalg.solve(M[np.ix_(b, b)], coupling)[2]


def swing(xml, drive, pose="digitigrade", start=60.0, end=-150.0, T=0.6, gif=None):
    """Free flight, sweep the tail along `drive`, measure the body's pitch."""
    from core.model import Recorder, aid, qadr, setup
    m, d = setup(pose, xml=xml)
    tail_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(m.njnt)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                   .startswith("tail")]
    drive = np.asarray(drive, float)
    drive = drive / np.abs(drive).max()          # base joint sweeps the full range

    for j, w in zip(tail_joints, drive):
        d.qpos[qadr(m, j)] = np.deg2rad(start) * w
    d.qpos[qadr(m, "slide_z")] += 0.5
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)

    zp = qadr(m, "pitch")
    pitch0 = d.qpos[zp]
    travel = 0.0
    prev = np.array([d.qpos[qadr(m, j)] for j in tail_joints])
    g0, est, n = gain(m, d, tail_joints, drive), 0.0, int(T / m.opt.timestep)
    rec = Recorder(m, gif) if gif else None
    tau_tail = 0.0                            # what the tail motors have to deliver

    for k in range(n):
        frac = 0.5 - 0.5 * np.cos(np.pi * k / n)
        for j, w in zip(tail_joints, drive):
            d.ctrl[aid(m, j)] = np.deg2rad(start + (end - start) * frac) * w
        mujoco.mj_step(m, d)
        now = np.array([d.qpos[qadr(m, j)] for j in tail_joints])
        step = now - prev
        travel += np.abs(step).sum()          # total actuator travel, all joints
        tau_tail = max(tau_tail, np.abs(d.sensordata[4:4 + len(tail_joints)]).max())
        if np.abs(step).max() > 0:
            est += gain(m, d, tail_joints, step / np.abs(step).max()) \
                   * np.abs(step).max() * np.sign(step[np.abs(step).argmax()])
        prev = now
        if rec:
            rec.grab(d, k)
    if rec:
        rec.save()

    base_swept = np.rad2deg(prev[0]) - start * drive[0]
    pitch = np.rad2deg(d.qpos[zp] - pitch0)
    travel = np.rad2deg(travel)
    return dict(pitch=pitch, swept=base_swept, g0=g0, est=np.rad2deg(est),
                mass=m.body_mass.sum(), travel=travel, eff=abs(pitch) / travel,
                tau_tail=tau_tail)


def cases():
    return [
        ("1 DOF, 25 g rod",           build(1, 0.025, 0.0),   [1]),
        ("3 DOF, 25 g rod, straight", build(3, 0.025, 0.0),   [1, 0, 0]),
        ("3 DOF, 25 g rod, curling",  build(3, 0.025, 0.0),   [1, 1, 1]),
        ("3 DOF, 25 g + 25 g tip",    build(3, 0.025, 0.025), [1, 0, 0]),
        ("1 DOF, 25 g + 25 g tip",    build(1, 0.025, 0.025), [1]),
    ]

if __name__ == "__main__":
    print(f"{'variant':<28} {'gain':>6} {'base swept':>11} {'body pitch':>11}"
          f" {'deg/deg':>8} {'tail tau':>9} {'robot':>7}")
    out = {}
    films = {}
    for name, xml, drive in cases():
        r = swing(xml, drive, gif=films.get(name))
        out[name] = r
        print(f"{name:<28} {r['g0']:+6.2f} {r['swept']:+10.1f}d {r['pitch']:+10.1f}d "
              f"{r['eff']:8.2f} {r['tau_tail']*1e3:7.1f}mNm {r['mass']*1000:6.0f}g")

    rod1 = out["1 DOF, 25 g rod"]
    rod3 = out["3 DOF, 25 g rod, straight"]
    curl = out["3 DOF, 25 g rod, curling"]
    tip3 = out["3 DOF, 25 g + 25 g tip"]
    tip1 = out["1 DOF, 25 g + 25 g tip"]

    # Splitting the same tail into three hinges changes nothing about the ceiling,
    # with or without a tip mass.  Redundancy is not authority.
    assert abs(rod3["pitch"] - rod1["pitch"]) < 0.1 * abs(rod1["pitch"]), \
        f"extra DOFs should not change authority: {rod1['pitch']:.1f} vs {rod3['pitch']:.1f}"
    assert abs(tip3["pitch"] - tip1["pitch"]) < 0.1 * abs(tip1["pitch"]), \
        f"...and still not, with a tip mass: {tip1['pitch']:.1f} vs {tip3['pitch']:.1f}"

    # Curling does win on raw pitch - driving three joints multiplies the distal
    # sweep faster than the shortening lever loses it.  But it spends three
    # actuators through their whole range to do it, so per degree of commanded
    # joint travel it is much worse.  That is the real cost of the redundancy.
    assert abs(curl["pitch"]) > abs(rod3["pitch"]), "curling should win on raw pitch"
    assert curl["eff"] < 0.6 * rod3["eff"], \
        f"...and lose badly per actuator-degree: {curl['eff']:.2f} vs {rod3['eff']:.2f}"

    # Mass at the end is what actually buys authority: I scales with r^2.
    assert abs(tip3["pitch"]) > 1.5 * abs(rod3["pitch"]), \
        f"a tip mass should clearly beat the bare rod: {rod3['pitch']:.1f} vs {tip3['pitch']:.1f}"
    # Every row above sits on the 0.5 Nm motor limit, so that table compares
    # variants at equal motor - the comparison you actually care about when
    # buying one.  Unclamp it to see what each design really demands.
    print("\nsame swing with the tail motor limit lifted (what it really asks for):")
    for label, kw, drive in (("25 g rod      ", dict(n=1, mass=0.025, torque=5.0), [1]),
                             ("25 g + 25 g tip", dict(n=1, mass=0.025, tip=0.025,
                                                      torque=5.0), [1])):
        r = swing(build(**kw), drive)
        print(f"  {label}  body pitch {r['pitch']:+7.1f}d   "
              f"peak tail torque {r['tau_tail']*1e3:6.1f} mNm")

    print("\nselfcheck ok")
