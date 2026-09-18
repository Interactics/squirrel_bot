"""Crouch, jump, fly, land - and see whether the tail is what keeps it upright.

    python -m studies.jump_landing

Phases are driven by contact, not by a stopwatch: the robot is in flight when
nothing touches the floor, and it has landed when something does again.
"""
import numpy as np
import mujoco

from core.model import POSES, Recorder, aid, qadr, setup, tail_joints

TAKEOFF = [6.4, 58.0, -99.4, -55.0]      # digitigrade, extended
CROUCH = POSES["crouch"]
PUSH_T = 0.10                            # seconds to extend the leg
TAIL_LIMIT = np.deg2rad([-150, 60])


def run(tail_on=True, T=1.2, kp=2.5, kd=0.12, gif=None):
    m, d = setup("crouch")
    zp, zx = qadr(m, "pitch"), qadr(m, "slide_x")
    tails = tail_joints(m)
    rec = None
    if gif:
        rec = Recorder(m, gif)

    log = dict(phase=[], pitch=[], z=[], t=[])
    took_off = False
    n = int(T / m.opt.timestep)

    for k in range(n):
        t = k * m.opt.timestep
        flying = d.ncon == 0
        took_off = took_off or flying

        # --- leg: ramp crouch -> extended to push off, then hold for landing
        frac = min(1.0, t / PUSH_T)
        for j, a, b in zip(("hip", "knee", "ankle", "mtp"), CROUCH, TAKEOFF):
            d.ctrl[aid(m, j)] = np.deg2rad(a + (b - a) * frac)

        # --- tail: in flight, swing it to null the body's pitch
        if tail_on and flying:
            err = d.qpos[zp]                      # want the body level at 0
            rate = d.qvel[m.jnt_dofadr[mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, "pitch")]]
            cmd = np.clip(kp * err + kd * rate, *TAIL_LIMIT)
            for j in tails:
                d.ctrl[aid(m, j)] = cmd / len(tails)
        elif not tail_on:
            for j in tails:
                d.ctrl[aid(m, j)] = 0.0

        mujoco.mj_step(m, d)
        if rec:
            rec.grab(d, k)

        log["t"].append(t)
        log["pitch"].append(np.rad2deg(d.qpos[zp]))
        log["z"].append(d.xpos[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "torso")][2])
        log["phase"].append("air" if flying else ("push" if not took_off else "land"))
    if rec:
        rec.save()

    z = np.array(log["z"])
    pitch = np.array(log["pitch"])
    phase = np.array(log["phase"])
    air = phase == "air"
    if not air.any():
        return dict(took_off=False, apex=z.max(), flight=0.0, touchdown=np.nan,
                    final_pitch=pitch[-1], travel=d.qpos[zx])

    last_air = np.where(air)[0][-1]
    return dict(took_off=True, apex=z.max(),
                flight=air.sum() * m.opt.timestep,
                touchdown=pitch[min(last_air + 1, len(pitch) - 1)],
                final_pitch=pitch[-1], travel=d.qpos[zx])


def show(label, r):
    if not r["took_off"]:
        print(f"{label:<12} never left the ground (apex {r['apex']*1000:.1f} mm)")
        return
    print(f"{label:<12} apex {r['apex']*1000:6.1f} mm   flight {r['flight']*1000:5.0f} ms"
          f"   pitch at touchdown {r['touchdown']:+7.1f} deg"
          f"   after landing {r['final_pitch']:+7.1f} deg")


if __name__ == "__main__":
    off = run(tail_on=False)
    on = run(tail_on=True)
    show("tail OFF", off)
    show("tail ON", on)

    assert on["took_off"] and off["took_off"], "the leg has to actually jump first"
    assert abs(on["touchdown"]) < abs(off["touchdown"]), \
        f"the tail should land it flatter: {on['touchdown']:.1f} vs {off['touchdown']:.1f}"
    print("\nselfcheck ok")
