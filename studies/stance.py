"""Stand the squirrel leg up in MuJoCo and see what the joints have to carry.

    python -m studies.stance

The three poses were solved with closed-form 2-link IK on the same segment
lengths the MJCF uses, so the postures and the simulated model agree by
construction rather than by eye.
"""
import numpy as np
import mujoco

from core.model import POSES, qadr, setup

def settle(pose, T=1.5):
    """Hold `pose` for T seconds and report what the joints carried."""
    m, d = setup(pose)
    zp = qadr(m, "pitch")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    n = int(T / m.opt.timestep)
    z, pitch, tau = np.empty(n), np.empty(n), np.empty((n, 4))
    for k in range(n):
        mujoco.mj_step(m, d)
        z[k] = d.xpos[torso][2]
        pitch[k] = d.qpos[zp]
        tau[k] = d.sensordata[:4]

    xs = [d.contact[i].pos[0] for i in range(d.ncon)]
    com = d.subtree_com[0][0]
    margin = (com - min(xs), max(xs) - com) if xs else (0.0, 0.0)

    return dict(pose=pose, z0=z[0], z=z[-1], pitch=np.rad2deg(pitch[-1]),
                tau_peak=np.abs(tau).max(axis=0), ncon=d.ncon, margin=margin,
                fell=abs(np.rad2deg(pitch[-1])) > 30 or z[-1] < 0.5 * z[0])


def report(r):
    print(f"\n{r['pose']:>12}   hip {r['z0']*1000:5.1f} -> {r['z']*1000:5.1f} mm   "
          f"pitch {r['pitch']:+6.1f} deg   contacts {r['ncon']}   "
          f"{'FELL' if r['fell'] else 'stood'}")
    print(f"{'':>12}   peak |tau| [mNm]  hip {r['tau_peak'][0]*1e3:5.1f}  "
          f"knee {r['tau_peak'][1]*1e3:5.1f}  ankle {r['tau_peak'][2]*1e3:5.1f}  "
          f"mtp {r['tau_peak'][3]*1e3:5.1f}")
    print(f"{'':>12}   COM inside support polygon by  back {r['margin'][0]*1000:+5.1f}  "
          f"front {r['margin'][1]*1000:+5.1f}  mm")


if __name__ == "__main__":
    plant = settle("plantigrade")
    digi = settle("digitigrade")
    report(plant)
    report(digi)

    # The whole point of the model: a flat sole is a contact patch and can resist
    # pitch, a toe is a point and cannot.  If this ever flips, the foot geometry
    # or the torso COM moved.
    assert not plant["fell"], "plantigrade should stand"
    assert digi["fell"], "digitigrade on a point toe should topple without control"
    # ...and it stands for exactly one reason: the COM projection sits inside
    # the support polygon, with room on both sides.
    assert min(plant["margin"]) > 0.005, \
        f"plantigrade COM should sit inside the polygon, margins {plant['margin']}"
    print("\nselfcheck ok")
