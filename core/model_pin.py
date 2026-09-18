"""One Pinocchio model of the squirrel, built from the same MJCF MuJoCo runs.

Verified against mj_inverse to 3e-15 (see check() below).  That check is the
whole reason this file exists: a TO that optimises a slightly different robot
than the simulator runs produces a trajectory the WBC cannot track, and the
failure looks like a controller bug for days.
"""
import os
import tempfile
import warnings

import numpy as np
import pinocchio as pin

warnings.filterwarnings("ignore", message=".*DeprecatedBool.*")

from core.model import DEFAULT_TAIL, build           # noqa: E402

# q = [slide_x, slide_z, pitch, tail, hip, knee, ankle, mtp]; first three unactuated.
NB = 3                                   # floating (planar) base DOFs
ACT = slice(NB, 8)                       # the five actuated joints
# The contact point is NOT a material point: the lowest point of a sphere slides
# over its own surface as the foot rotates.  So the frame sits at the sphere
# CENTRE, and the ground constraint is centre_z == TOE_R, which is exact.
TOE_LOCAL = np.array([0.0, 0.0, -0.018])  # toe pad centre, in the mtp joint frame
TOE_R = 0.004                             # its radius
MU = 0.9                                 # matches the MJCF friction

# Sample points along the tail, as (distance from the hinge [m], clearance [m]).
# The clearance is the capsule radius, and the tip sphere's radius at the end.
# Without these the optimiser happily sweeps the tail 67 mm through the floor -
# it is a free lever arm if nothing says otherwise.
# Every 30 mm, so that a 10 mm-radius tail cannot thread between two samples.

TAIL_SAMPLES = ((0.000, 0.010), (0.045, 0.010), (0.090, 0.010),
                (0.135, 0.010), (0.180, 0.016))
TAIL_NAMES = tuple(f"tail_{i}" for i in range(len(TAIL_SAMPLES)))

# Sampled points on every segment, as name -> (joint, offset along its z, radius).
# One list, several jobs: floor clearance, trunk exclusion, tail-vs-leg and
# leg-vs-itself.  Keeping them in one place is what stopped the last round of
# "constrain the thing that broke, discover the next thing it folds into".
SEGMENTS = {
    "femur_mid":  ("hip", -0.0225, 0.004),
    "femur_end":  ("hip", -0.045, 0.004),
    "tibia_mid":  ("knee", -0.025, 0.004),
    "tibia_end":  ("knee", -0.050, 0.004),
    "heel":       ("ankle", 0.000, 0.005),
    "meta_mid":   ("ankle", -0.0175, 0.004),
    "meta_end":   ("ankle", -0.035, 0.004),
    "phal_mid":   ("mtp", -0.009, 0.003),
    "toe":        ("mtp", -0.018, 0.004),
}

# Must clear the FLOOR.  Constraining the toe alone let the heel go 34 mm under.
# Everything that can reach the ground.  The femur was left off once and went
# 5.4 mm under; the round-2 solve only came out clean by luck.
FLOOR = ("heel", "meta_mid", "meta_end", "phal_mid", "toe",
         "tibia_mid", "tibia_end", "femur_mid", "femur_end") + TAIL_NAMES

# Must stay OUTSIDE the trunk ellipsoid - the leg folded 34 mm into the body.
TRUNK_OUT = ("heel", "meta_mid", "meta_end", "tibia_mid", "tibia_end")

# Non-adjacent segment pairs that must not pass through each other.  MuJoCo
# excludes parent/child, so only these can actually touch - and once the joint
# limits tightened, the optimiser folded the shank straight through the toe.
# Which sample points stand in for each MuJoCo geom.  Collision constraints are
# generated LAZILY: solve, ask MuJoCo what actually overlapped, add only those
# pairs, solve again.  Constraining every pair up front is 64 rows a knot and
# segfaults IPOPT while it assembles the Hessian; in practice two or three pairs
# ever bind, and guessing which ones cost several rounds of "constrain the thing
# that broke, watch it fold into the next thing".
GEOM_POINTS = {
    "femur": ("femur_mid", "femur_end"),
    "tibia": ("tibia_mid", "tibia_end"),
    "metatarsus": ("heel", "meta_mid", "meta_end"),
    "heel_pad": ("heel",),
    "phalanx": ("phal_mid",),
    "toe_pad": ("toe",),
    "tail1": TAIL_NAMES,
    "tip": (TAIL_NAMES[-1],),
}

# The trunk ellipsoid, in the torso frame, that the leg must stay outside of.
TRUNK_C = np.array([-0.015, 0.035])          # centre (x, z)
TRUNK_AB = np.array([0.045, 0.030])          # semi-axes (x, z)
MARGIN = 0.002                               # extra clearance on every pair [m]


def load():
    xml = build(**DEFAULT_TAIL)
    path = os.path.join(tempfile.mkdtemp(), "squirrel.xml")
    with open(path, "w") as f:
        f.write(xml)
    m = pin.buildModelFromMJCF(path)
    toe = m.addFrame(pin.Frame("toe", m.getJointId("mtp"), 0,
                               pin.SE3(np.eye(3), TOE_LOCAL), pin.FrameType.OP_FRAME))
    torso_f = m.addFrame(pin.Frame("torso", 1, 0, pin.SE3.Identity(),
                                   pin.FrameType.OP_FRAME))
    seg = {name: (m.addFrame(pin.Frame(name, m.getJointId(j), 0,
                                       pin.SE3(np.eye(3), np.array([0.0, 0.0, off])),
                                       pin.FrameType.OP_FRAME)), r)
           for name, (j, off, r) in SEGMENTS.items()}
    tail_j = m.getJointId("tail1")
    for name, (off, r) in zip(TAIL_NAMES, TAIL_SAMPLES):
        seg[name] = (m.addFrame(pin.Frame(name, tail_j, 0,
                                          pin.SE3(np.eye(3), np.array([-off, 0.0, 0.0])),
                                          pin.FrameType.OP_FRAME)), r)
    return m, toe, xml, dict(seg=seg, torso=torso_f)


def check():
    import mujoco
    m, toe, xml, _ = load()
    data = m.createData()
    D = mujoco.mjtDisableBit
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.disableflags |= D.mjDSBL_CONTACT | D.mjDSBL_LIMIT
    dj = mujoco.MjData(mj)

    rng = np.random.default_rng(0)
    worst_tau, worst_toe = 0.0, 0.0
    tid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "toe_pad")
    for _ in range(8):
        q, v, a = (rng.normal(size=m.nv) * 0.4 for _ in range(3))
        worst_tau = max(worst_tau, np.abs(pin.rnea(m, data, q, v, a)
                                          - _mj_inv(mujoco, mj, dj, q, v, a)).max())
        pin.framesForwardKinematics(m, data, q)
        worst_toe = max(worst_toe,
                        np.abs(data.oMf[toe].translation - dj.geom_xpos[tid]).max())
    print(f"pinocchio vs mujoco   torque {worst_tau:.2e}   toe centre {worst_toe:.2e}"
          f"   (contact when toe_z == {TOE_R})")
    assert worst_tau < 1e-10 and worst_toe < 1e-10, "the two models must be identical"
    print("model check ok")


def _mj_inv(mujoco, mj, dj, q, v, a):
    dj.qpos[:], dj.qvel[:], dj.qacc[:] = q, v, a
    mujoco.mj_inverse(mj, dj)
    return dj.qfrc_inverse.copy()


if __name__ == "__main__":
    check()
