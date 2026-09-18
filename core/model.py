"""The squirrel model: build it, pose it, stand it on the floor, film it.

Everything else imports from here.  It used to live in sim_squirrel_leg.py (a
standing experiment) and tail_tuning.py (a tail experiment), which meant the two
imported each other and the cycle had to be broken with a lazy import inside a
function.  Core belongs in one place that depends on nothing of ours.
"""
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = os.path.join(REPO, "models", "squirrel_leg.xml")
OUT = os.path.join(REPO, "out")          # everything generated lands here, gitignored


def out(name):
    """Path for a generated file.  Keeps GIFs and .npz out of the source tree."""
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)

LEG = ("hip", "knee", "ankle", "mtp")
PADS = ("heel_pad", "toe_pad", "metatarsus", "phalanx")
# The XML ships with empty tail slots; this is the tail the model gets by default.
DEFAULT_TAIL = dict(n=1, mass=0.025, tip=0.025)

TAIL_LENGTH, TAIL_RADIUS = 0.180, 0.010   # total length [m], fluff radius [m]
TAIL_ROOT = "-0.040 0 0.045"              # tail base, in the torso frame

# hip, knee, ankle, mtp  [deg].  Zero = leg straight down, foot in line with the
# shank - full pointe, which the animal never holds.  plantigrade is the home pose.
POSES = {
    "plantigrade": [10.8, 45.5, -146.2, 0.0],    # sole flat -> a real contact patch
    "digitigrade": [6.4, 58.0, -99.4, -55.0],    # toe only  -> a point contact
    "crouch":      [-7.8, 113.2, -140.4, -55.0],
}


# ---------------------------------------------------------------- model source
def build(n=1, mass=0.025, tip=0.0, torque=None):
    """Splice an n-link tail (total `mass`, plus `tip` kg at the very end) into
    the marker slots in squirrel_leg.xml.  One generator, so every tail variant is
    the same tail cut into a different number of pieces - otherwise comparing them
    proves nothing."""
    seg = TAIL_LENGTH / n
    bodies, acts, sens, ind = [], [], [], "      "
    for i in range(n):
        nm = f"tail{i + 1}"
        pos = TAIL_ROOT if i == 0 else f"-{seg:.4f} 0 0"
        pad = ind + "  " * i
        bodies.append(f'{pad}<body name="{nm}" pos="{pos}">')
        bodies.append(f'{pad}  <joint name="{nm}" range="-150 60"/>')
        bodies.append(f'{pad}  <geom name="{nm}" fromto="0 0 0  -{seg:.4f} 0 0" '
                      f'size="{TAIL_RADIUS}" mass="{mass / n:.6f}" rgba=".55 .42 .30 1"/>')
        if i == n - 1 and tip:
            bodies.append(f'{pad}  <geom name="tip" type="sphere" pos="-{seg:.4f} 0 0" '
                          f'size="0.016" mass="{tip:.6f}" rgba=".30 .30 .34 1"/>')
        lim = f' forcerange="-{torque} {torque}"' if torque else ""
        acts.append(f'    <position name="{nm}" joint="{nm}" ctrlrange="-150 60"{lim}/>')
        sens.append(f'    <jointactuatorfrc name="tau_{nm}" joint="{nm}"/>')
    for i in range(n - 1, -1, -1):
        bodies.append(ind + "  " * i + "</body>")

    xml = open(BASE).read()
    for tag, block in (("TAIL", bodies), ("TAILACT", acts), ("TAILSENS", sens)):
        head, foot = f"<!-- {tag}:BEGIN -->", f"<!-- {tag}:END -->"
        a, b = xml.index(head) + len(head), xml.index(foot)
        xml = xml[:a] + "\n" + "\n".join(block) + "\n" + xml[b:]
    return xml


# ------------------------------------------------------------------- accessors
# Everything is addressed by name.  Adding a body anywhere in the XML reshuffles
# qpos, so hardcoded indices are a bug waiting for the next edit - one cost a whole
# debugging round when the tail went in ahead of the leg.
def _id(m, objtype, name, what):
    """mj_name2id returns -1 for an unknown name, and -1 is a VALID python index -
    so a typo silently addresses the last joint instead of raising.  That cost a
    round: the tail joint was renamed tail -> tail1 and tail_authority spent its
    life reading the mtp joint while reporting tail numbers."""
    i = mujoco.mj_name2id(m, objtype, name)
    if i < 0:
        raise KeyError(f"no {what} named {name!r} in this model")
    return i


def qadr(m, joint):
    return m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, joint, "joint")]


def gid(m, geom):
    return _id(m, mujoco.mjtObj.mjOBJ_GEOM, geom, "geom")


def aid(m, act):
    return _id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act, "actuator")


def tail_joints(m):
    names = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
             for i in range(m.njnt))
    return [n for n in names if n.startswith("tail")]


def bottom(m, d, g):
    """Lowest world z of a capsule or sphere geom (capsules run along local z)."""
    half = m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
    return d.geom_xpos[g][2] - abs(d.geom_xmat[g].reshape(3, 3)[2, 2]) * half \
        - m.geom_size[g][0]


def setup(pose, tail_deg=0.0, xml=None):
    """Build the model, put the leg in `pose`, and stand it on the floor with the
    position servos already commanded to hold that pose.

    `xml` takes a model string (see build) instead of the default tail."""
    m = mujoco.MjModel.from_xml_string(build(**DEFAULT_TAIL) if xml is None else xml)
    d = mujoco.MjData(m)

    for joint, ang in zip(LEG, np.deg2rad(POSES[pose])):
        d.qpos[qadr(m, joint)] = ang
        d.ctrl[aid(m, joint)] = ang              # ctrl is radians at runtime,
                                                 # even though the XML says degrees
    for joint in tail_joints(m):
        d.qpos[qadr(m, joint)] = np.deg2rad(tail_deg)
        d.ctrl[aid(m, joint)] = np.deg2rad(tail_deg)

    # drop the torso so the lowest foot point just kisses the floor
    mujoco.mj_forward(m, d)
    d.qpos[qadr(m, "slide_z")] -= min(bottom(m, d, gid(m, g)) for g in PADS)
    mujoco.mj_forward(m, d)
    return m, d


# --------------------------------------------------------------------- filming
class Recorder:
    """Grab frames at FPS of SIMULATED time, play them back at PLAY fps, so a
    120 ms event is watchable.  There were three copies of this."""

    FPS, PLAY, W, H = 100, 25, 480, 360

    def __init__(self, m, path, track="torso", distance=0.62, elevation=-8):
        from PIL import Image                   # only needed when recording
        self._Image = Image
        self.path, self.frames = (path if os.sep in str(path)
                                  else out(path)), []
        self.every = max(1, int(round(1.0 / (self.FPS * m.opt.timestep))))
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 90, elevation, distance
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, track)
        self.r = mujoco.Renderer(m, self.H, self.W)

    def grab(self, d, k=0, every=True):
        if every and k % self.every:
            return
        self.r.update_scene(d, self.cam)
        self.frames.append(self._Image.fromarray(self.r.render()))

    def save(self):
        self.r.close()
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                            duration=int(1000 / self.PLAY), loop=0)
        print(f"  wrote {self.path}  ({len(self.frames)} frames, "
              f"{self.FPS // self.PLAY}x slow motion)")
        return self.path


if __name__ == "__main__":
    m, d = setup("plantigrade")
    print(f"model ok   nq={m.nq} nv={m.nv} nu={m.nu}   mass {m.body_mass.sum()*1000:.1f} g")
    assert m.nq == 8 and m.nu == 5, "1-DOF tail model should be 8 DOF, 5 actuators"
    assert abs(min(bottom(m, d, gid(m, g)) for g in PADS)) < 1e-9, "foot must touch"
    print("selfcheck ok")
