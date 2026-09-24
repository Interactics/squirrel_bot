"""Does the plan actually execute?  Run the TO torques in MuJoCo physics.

    python -m studies.track

Every jump video so far was the PLAN played back kinematically - qpos written
straight from the optimiser.  Here MuJoCo integrates the real dynamics: soft
contact instead of the TO's rigid one, 0.2 ms steps instead of 2-14 ms.

    tau = tau_TO + Kp (q_ref - q) + Kd (v_ref - v)      on the 5 actuated joints

The three base DOFs get nothing: they move only through contact, exactly as in
the TO.

The PD runs through MuJoCo's position actuators, NOT qfrc_applied.  qfrc_applied
is integrated explicitly, and with leg inertias of ~1e-5 kg m^2 a Kd of 0.3 gives
damping * dt ~ 6: numerically unstable, and it showed up as the PD *creating*
4 deg of error in 10 ms that feedforward alone did not have.  Actuator damping is
integrated implicitly (implicitfast), so the same gains are fine there.

The WHOLE torque goes through the actuator, feedforward included: gain 1, bias
[0, -Kp, -Kd], ctrl = tau_ff + Kp q_ref + Kd v_ref gives exactly
tau_ff + Kp (q_ref - q) + Kd (v_ref - qdot), and forcerange clamps that total to
the motor limit.  It used to add the feedforward through qfrc_applied, which no
limit touches: the simulated joints received up to 4.5 N m against a 1.5 N m
motor, from 119 ms on - enough to fling the 2 g toe through its joint limit.
"""
import numpy as np
import mujoco
from PIL import Image, ImageDraw

import os
import shutil
import time

from core.model import (DEFAULT_TAIL, Q_COLS, TAU_COLS, V_COLS, build, out, run_dir,
                        save_table)
from core.to_jump import TAU_MAX

ACT_J = ("tail1", "hip", "knee", "ankle", "mtp")     # TO order: q[3:8]
HOLD = 0.3                                            # s after the plan ends
SLOW = 10                  # GIF playback: 10x slow motion, liftoff is a ~30 ms event


# Tail attitude feedback.  The base is unactuated, so joint PD alone never sees the
# body tilt; the tail is the one actuator that can turn the body, in flight by
# momentum exchange.  A tail swing of +1 deg turns the body about -0.66 deg, so a
# body pitch error e is cancelled by moving the tail +e/0.66: K_TH ~ 1.5.
TAIL_LIM = np.deg2rad([-145.0, 58.0])     # stay off the soft joint limits (-150, 60)
# Attitude feedback used for the physics_fb run: hip strategy in stance, tail in
# flight and stance.  Tuned by sweep; see formulation.md.
FB = dict(k_hip=1.0, k_hipd=0.05, k_th=3.0, k_om=0.1)


def run(plan, kp=15.0, kd=0.3, record=None, k_th=0.0, k_om=0.0, k_hip=0.0, k_hipd=0.0):
    q_p, v_p, tau_p, t_p = plan["q"], plan["v"], plan["tau"], plan["t"]
    m = mujoco.MjModel.from_xml_string(build(**DEFAULT_TAIL))
    act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ACT_J])
    m.actuator_gainprm[act, 0] = 1.0               # force = ctrl + bias, see docstring
    m.actuator_biasprm[act, :3] = [0.0, -kp, -kd]  # -kd qdot is integrated implicitly
    m.actuator_forcelimited[act] = 1               # the motor limit is on the total
    m.actuator_forcerange[act] = [-TAU_MAX, TAU_MAX]
    m.actuator_ctrllimited[act] = 0
    d = mujoco.MjData(m)
    d.qpos[:] = q_p[:, 0]
    mujoco.mj_forward(m, d)

    dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                    for j in ACT_J])
    ref = lambda arr, t: np.array([np.interp(t, t_p, row) for row in arr])
    # The TO integrates with semi-implicit Euler: tau[k] acts over the whole of
    # [t_k, t_k+1).  Feed it forward the same way - zero-order hold, not a ramp.
    hold = lambda arr, t: arr[:, min(np.searchsorted(t_p, t, side="right") - 1,
                                     arr.shape[1] - 1)]

    T = t_p[-1] + HOLD
    n = int(T / m.opt.timestep)
    ghost = mujoco.MjData(m)
    frame_every = max(1, int(0.005 / m.opt.timestep))     # GIF: 5 ms
    log_every = max(1, int(0.001 / m.opt.timestep))       # data: 1 ms
    toe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "toe_pad")
    rows, frames, f6 = [], [], np.zeros(6)
    if record:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = 90, -8, 0.62
        cam.lookat[:] = [0.07, 0, 0.09]
        r = mujoco.Renderer(m, 300, 400)

    for k in range(n):
        t = k * m.opt.timestep
        tc = min(t, t_p[-1])
        q_r, v_r = ref(q_p, tc), ref(v_p, tc) * (t <= t_p[-1])
        q_cmd = q_r[3:].copy()
        # tail target moves WITH the body error: tail + turns the body -
        q_cmd[0] = np.clip(q_cmd[0] + k_th * (d.qpos[2] - q_r[2])
                           + k_om * (d.qvel[2] - v_r[2]), *TAIL_LIM)
        # Hip strategy, stance only: with the foot on the ground the hip sits between
        # the body and a leg that is going nowhere, so shifting the hip target by
        # -e turns the body back without moving the leg.  In flight the tail owns
        # attitude (angular momentum is fixed then; the hip would only fling the leg).
        if k_hip or k_hipd:
            in_stance = t < t_p[14] or t_p[38] <= t
            if in_stance:
                q_cmd[1] -= k_hip * (d.qpos[2] - q_r[2]) + k_hipd * (d.qvel[2] - v_r[2])
        d.ctrl[act] = hold(tau_p, tc) + kp * q_cmd + kd * v_r[3:]
        mujoco.mj_step(m, d)
        if k % log_every == 0:
            fz = 0.0
            for i in range(d.ncon):
                if toe in (d.contact[i].geom1, d.contact[i].geom2):
                    mujoco.mj_contactForce(m, d, i, f6)
                    fz += f6[0]
            # the torque the joint actually received: servo PD + feedforward
            tau = d.actuator_force[act]         # everything, already clamped
            rows.append((t, *d.qpos, *d.qvel, *tau, fz, *q_r))
        if record and k % frame_every == 0:
            ghost.qpos[:] = q_r
            mujoco.mj_forward(m, ghost)
            r.update_scene(ghost, cam); a = r.render()
            r.update_scene(d, cam);     b = r.render()
            img = Image.fromarray(np.hstack([a, b]))
            draw = ImageDraw.Draw(img)
            draw.text((8, 6), "plan", fill=(255, 255, 255))
            draw.text((a.shape[1] + 8, 6), "physics", fill=(255, 255, 255))
            phase = ("push" if t < t_p[14] else "flight" if t < t_p[38]
                     else "land" if t <= t_p[-1] else "after plan")
            draw.text((8, a.shape[0] - 16), f"t = {t*1e3:5.0f} ms   {phase}   (1/{SLOW} speed)",
                      fill=(255, 255, 255))
            frames.append(img)
    if record:
        r.close()
        frames[0].save(out(record), save_all=True, append_images=frames[1:],
                       duration=int(1000 * frame_every * m.opt.timestep * SLOW), loop=0)
    R = np.array(rows).T
    return dict(t=R[0], q=R[1:9], v=R[9:17], tau=R[17:22], fz=R[22], qr=R[23:31])


def save_physics(sim, folder, name):
    cols = {"t_s": sim["t"]}
    cols.update({n: sim["q"][i] for i, n in enumerate(Q_COLS)})
    cols.update({n: sim["v"][i] for i, n in enumerate(V_COLS)})
    cols.update({n: sim["tau"][i] for i, n in enumerate(TAU_COLS)})
    cols["fz_toe_N"] = sim["fz"]
    cols.update({"ref_" + n: sim["qr"][i] for i, n in enumerate(Q_COLS)})
    return save_table(os.path.join(folder, name), cols)


def report(label, sim, t_end):
    t = sim["t"]
    x, z, th = sim["q"][:3]
    xr, zr, thr = sim["qr"][:3]
    end = t <= t_end
    ex, ez = np.abs(x - xr)[end].max() * 1e3, np.abs(z - zr)[end].max() * 1e3
    eth = np.rad2deg(np.abs(th - thr))[end].max()
    fell = np.rad2deg(abs(th[-1])) > 45
    print(f"{label:<22} max err  x {ex:5.1f} mm  z {ez:5.1f} mm  pitch {eth:5.1f} deg"
          f"   |  final pitch {np.rad2deg(th[-1]):+6.1f} deg  x {x[-1]*1e3:+6.1f} mm"
          f"  {'FELL' if fell else 'upright'}")
    return dict(ex=ex, ez=ez, eth=eth, fell=fell, x=x[-1], th=th[-1])


if __name__ == "__main__":
    plan = np.load(out("to_jump.npz"))
    t_end = plan["t"][-1]
    stamp = str(plan["run"]) if "run" in plan else time.strftime("%Y%m%d-%H%M%S")
    # one subfolder per tracking run, so re-running a plan never overwrites
    folder = os.path.join(run_dir(stamp), "track-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder)
    print(f"plan {stamp}: {t_end*1000:.0f} ms, lands at base x {plan['q'][0,-1]*1e3:+.1f} mm\n")
    ff = run(plan, kp=0, kd=0, record=os.path.join(folder, "physics_ff.gif"))
    pd = run(plan, record=os.path.join(folder, "physics_ffpd.gif"))
    fb = run(plan, record=os.path.join(folder, "physics_fb.gif"), **FB)
    shutil.copy(os.path.join(folder, "physics_fb.gif"), out("track_plan_vs_physics.gif"))
    report("feedforward only", ff, t_end)
    report("feedforward + PD", pd, t_end)
    report("+ attitude feedback", fb, t_end)
    for name, sim in (("physics_ff", ff), ("physics_ffpd", pd), ("physics_fb", fb)):
        save_physics(sim, folder, name)
        print(f"  {name:<13} peak |tau| {np.abs(sim['tau']).max()*1e3:5.0f} mNm (limit {TAU_MAX*1e3:.0f})")
    print(f"\n  wrote {folder}/  (csv, npz, gif; plan | physics at 1/{SLOW} speed)")
    print(f"  latest gif: {out('track_plan_vs_physics.gif')}")
