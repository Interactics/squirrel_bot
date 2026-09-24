"""Solve the jump with collision constraints generated lazily from MuJoCo.

    python -m core.verify_to

Constraining every segment pair up front is 64 rows a knot and segfaults IPOPT
while it assembles the Hessian.  In practice only two or three pairs ever bind.
So: solve, ask MuJoCo's own narrowphase what actually overlapped, add constraints
for exactly those, solve again.  MuJoCo is the referee, not my point sampling.
"""
import os
import time

import numpy as np
import mujoco
import pinocchio as pin

from core.model import Q_COLS, TAU_COLS, V_COLS, out, run_dir, save_table
from core.model_pin import GEOM_POINTS, load
from core.to_jump import N_FLY, N_LAND, N_PUSH, audit, solve_homotopy, start_pose

MAX_ROUNDS = 6


def penetrations(xml, q, tol=-1e-4):
    """Every geom pair that overlaps anywhere in the trajectory, worst first."""
    mj = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(mj)
    name = lambda g: mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g)
    hits = {}
    for k in range(q.shape[1]):
        d.qpos[:] = q[:, k]
        mujoco.mj_forward(mj, d)
        for c in range(d.ncon):
            con = d.contact[c]
            if con.dist < tol:
                key = tuple(sorted((name(con.geom1), name(con.geom2))))
                hits[key] = min(hits.get(key, 0.0), con.dist)
    return hits


def to_pairs(hits):
    """Turn MuJoCo geom pairs into the sample-point pairs the TO can constrain."""
    out, unmapped = set(), []
    for a, b in hits:
        if a == "floor" or b == "floor" or "trunk" in (a, b):
            # These have their own constraint families (FLOOR, TRUNK_OUT).  If one
            # shows up here the sample list is missing a point - say so loudly
            # rather than quietly hoping another pair fixes it.
            unmapped.append((a, b))
            continue
        pa, pb = GEOM_POINTS.get(a), GEOM_POINTS.get(b)
        if pa is None or pb is None:
            unmapped.append((a, b))
            continue
        out |= {tuple(sorted((x, y))) for x in pa for y in pb if x != y}
    return out, unmapped


def save_plan(r, t, folder):
    """The plan as a table: time, phase, q, v, tau, contact force, and centroidal
    angular momentum about the COM (y), one row per knot."""
    model, _, _, _ = load()
    data = model.createData()
    L = [pin.computeCentroidalMomentum(model, data, r["q"][:, k], r["v"][:, k]).angular[1]
         for k in range(r["q"].shape[1])]
    phase = np.where(r["stance"], np.where(np.arange(len(t)) < N_PUSH, 0, 2), 1)
    cols = {"t_s": t, "phase_0push_1flight_2land": phase}
    cols.update({n: r["q"][i] for i, n in enumerate(Q_COLS)})
    cols.update({n: r["v"][i] for i, n in enumerate(V_COLS)})
    cols.update({n: r["tau"][i] for i, n in enumerate(TAU_COLS)})
    cols.update({"fx_N": r["f"][0], "fz_N": r["f"][1], "L_Nms": L})
    return save_table(os.path.join(folder, "plan"), cols)


def main():
    _, _, xml, _ = load()
    q0 = start_pose()
    pairs = set()

    for rnd in range(1, MAX_ROUNDS + 1):
        print(f"\n=== round {rnd}: {len(pairs)} collision pairs constrained")
        r = solve_homotopy(q0, pairs=tuple(sorted(pairs)))
        hits = penetrations(xml, r["q"])
        if not hits:
            print("MuJoCo: no penetration anywhere in the trajectory")
            break
        print(f"MuJoCo found {len(hits)} overlapping geom pairs:")
        for pair, d in sorted(hits.items(), key=lambda kv: kv[1]):
            print(f"   {pair[0]:<12} x {pair[1]:<12} {d*1000:8.2f} mm")
        new, unmapped = to_pairs(hits)
        if unmapped:
            print(f"   (not mappable to sample points: {unmapped})")
        if not new - pairs:
            print("   no new constraints to add - denser sampling needed")
            break
        pairs |= new
    else:
        print("gave up after", MAX_ROUNDS, "rounds")

    a = audit(r)
    print(f"\naudit   g {a['g']:+.2f} m/s2   rise {a['rise']*1000:.1f} mm "
          f"(asked {r['rise']*1000:.0f})   apex {a['apex']*1000:.1f} mm")
    print(f"reach   {r['reach']*1000:+.1f} mm   peak |tau| "
          f"{np.abs(r['tau']).max()*1e3:.0f} mNm   fz {r['f'][1].max():.1f} N")

    t = np.concatenate([[0], np.cumsum(np.repeat(r["dt"], [N_PUSH, N_FLY, N_LAND]))])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    np.savez(out("to_jump.npz"), q=r["q"], v=r["v"], a=r["a"], tau=r["tau"], f=r["f"],
             dt=r["dt"], stance=r["stance"], t=t, run=stamp)
    print("wrote", out("to_jump.npz"))
    print("wrote", save_plan(r, t, run_dir(stamp)) + ".csv")
    assert not penetrations(xml, r["q"]), "the plan must not pass through anything"
    return r


if __name__ == "__main__":
    main()
