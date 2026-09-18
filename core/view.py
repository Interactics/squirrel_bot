"""Look at the squirrel: live window, or an animated GIF.

    mjpython -m core.view live plantigrade        # interactive (macOS needs mjpython)
    python   -m core.view gif  plantigrade        # a stance, servos holding
    python   -m core.view gif  out/to_jump.npz        # a planned trajectory, kinematic

macOS note: plain `python -m core.view live` opens no window and exits 0, silently.
mjpython itself needs `otool`, so if Xcode's licence is unaccepted, prefix with
    PATH="/Library/Developer/CommandLineTools/usr/bin:$PATH"

The GIF path replaces three near-identical recorders that had drifted apart.
"""
import sys
import time

import numpy as np
import mujoco

from core.model import POSES, Recorder, setup

SYNC_EVERY = 25          # steps between viewer syncs; 2e-4 s each -> 200 Hz redraw


def live(pose="plantigrade"):
    import mujoco.viewer
    m, d = setup(pose)
    print(f"holding {pose}  -  close the window to quit")
    with mujoco.viewer.launch_passive(m, d) as v:
        wall = time.time()
        while v.is_running():
            for _ in range(SYNC_EVERY):
                mujoco.mj_step(m, d)
            v.sync()
            wall += SYNC_EVERY * m.opt.timestep
            lag = wall - time.time()
            if lag > 0:
                time.sleep(lag)      # wall-clock speed, not flat out


def gif(target="plantigrade", out=None, seconds=1.5):
    """A stance (stepped, servos holding) or an .npz plan (kinematic playback)."""
    if target.endswith(".npz"):
        z = np.load(target)
        m, d = setup("crouch")
        rec = Recorder(m, out or target.replace(".npz", ".gif"), distance=0.75)
        for k in range(z["q"].shape[1]):
            d.qpos[:] = z["q"][:, k]
            mujoco.mj_forward(m, d)
            rec.grab(d, every=False)         # one frame per knot: it IS the plan
        return rec.save()

    m, d = setup(target)
    rec = Recorder(m, out or f"squirrel_{target}.gif")
    for k in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        rec.grab(d, k)
    return rec.save()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "gif"
    arg = sys.argv[2] if len(sys.argv) > 2 else "plantigrade"
    if what not in ("live", "gif"):
        sys.exit("usage: view.py [live|gif] [pose|file.npz]")
    if not arg.endswith(".npz") and arg not in POSES:
        sys.exit(f"pose must be one of {list(POSES)} or an .npz")
    (live if what == "live" else gif)(arg)
