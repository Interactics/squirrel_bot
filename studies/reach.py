"""How accurately can the jump hit a landing spot, and how far can it go?

    python -m studies.reach

Accuracy is only worth reporting if the trajectory is real, so every target is
checked three ways before its error is believed:
  1. IPOPT converged            - says nothing about physics
  2. audit()                    - flight COM is a parabola with -g, rise as asked
  3. MuJoCo narrowphase         - nothing passes through anything
A target that fails any of them is reported as failed, not as "landed".
"""
import numpy as np

from core.model_pin import load
from core.to_jump import REACH, RISE, audit, solve_homotopy, solve_reach, start_pose
from core.verify_to import main as solve_verified
from core.verify_to import penetrations

TARGETS = (0.050, 0.150, 0.250)          # m, foot to foot


def check(xml, r):
    """Return (ok, note) for a solved trajectory."""
    try:
        a = audit(r)
    except AssertionError as e:
        return False, f"audit: {e}"
    hits = penetrations(xml, r["q"])
    if hits:
        worst = min(hits.values())
        pair = min(hits, key=hits.get)
        return False, f"collision {pair[0]}x{pair[1]} {worst*1000:.1f} mm"
    return True, (f"g {a['g']:+.2f}  rise {a['rise']*1000:.1f} mm  "
                  f"apex {a['apex']*1000:.0f} mm")


def main():
    _, _, xml, _ = load()
    q0 = start_pose()
    base = solve_verified()               # the verified nominal jump + its pairs
    pairs = tuple(sorted({("femur_end", "phal_mid"), ("femur_end", "toe"),
                          ("femur_mid", "phal_mid"), ("femur_mid", "toe"),
                          ("tibia_end", "phal_mid"), ("tibia_end", "toe"),
                          ("tibia_mid", "phal_mid"), ("tibia_mid", "toe")}))

    print("\n" + "=" * 78)
    print(f"{'target':>9} {'landed':>10} {'error':>9} {'peak tau':>9} {'fz':>7}  verdict")
    print("=" * 78)
    warm = base
    for tgt in TARGETS:
        try:
            warm = solve_reach(q0, tgt, warm, pairs=pairs)
        except RuntimeError:
            print(f"{tgt*1000:+8.0f}mm {'-':>10} {'-':>9} {'-':>9} {'-':>7}  solver failed")
            continue
        ok, note = check(xml, warm)
        err = (warm["reach"] - tgt) * 1000
        print(f"{tgt*1000:+8.0f}mm {warm['reach']*1000:+9.1f}mm {err:+8.2f}mm "
              f"{np.abs(warm['tau']).max()*1e3:8.0f} {warm['f'][1].max():6.1f}N  "
              f"{note if ok else 'REJECTED ' + note}")

    print("\nhow far can it go?  stepping out until a check fails")
    good, warm = TARGETS[-1], warm
    step, tgt = 0.050, TARGETS[-1] + 0.050
    while step >= 0.0125:
        try:
            w = solve_reach(q0, tgt, warm, pairs=pairs)
            ok, note = check(xml, w)
        except RuntimeError:
            ok, note, w = False, "solver failed", None
        print(f"   {tgt*1000:+6.0f} mm  {'ok  ' + note if ok else 'NO  ' + note}")
        if ok:
            good, warm = tgt, w
            tgt += step
        else:
            step /= 2
            tgt = good + step
    print(f"\nfurthest verified landing: {good*1000:+.0f} mm "
          f"({good/0.090:.1f} torso lengths) at tau <= 1.5 N m, 80 mm rise")


if __name__ == "__main__":
    main()
