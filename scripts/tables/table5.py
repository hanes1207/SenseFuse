"""Table V: the fusion rule, against its alternatives.

Open-YOLO3D with EVA02-E and Uni3D on ScanNet200, the setting Table I leads with. Every rule in
:mod:`models.rules` is read at the weight Algorithm 1 estimates *in that rule's own space*, so that
no rule is penalised for another's estimator, and each is reported against the baseline of its own
group: the 2D head alone for the score-level and decision-level rules, and Open-YOLO3D's own label
scores for the vote-space rules.

The three groups answer three different questions. The score-level rules ask whether a different
pooling of the same two rows would do better. The decision-level rules ask whether the combination
could happen after the argmax instead of before it. The vote-space rules ask what the shape channel
can contribute when the 2D head emits counts rather than a continuous score, which is the setting
the fusion law does not cover.

    python scripts/tables/table5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from models import rules as R  # noqa: E402
from utils.settings import instance_columns, load  # noqa: E402

# What the paper prints: rule -> (accuracy, change against that group's baseline).
PRINTED = {"linear": (53.0, 4.5), "poe": (52.8, 4.3), "maxpool": (48.4, 0.0),
           "rank": (32.0, -16.4), "router": (46.9, -1.6), "veto": (47.2, -1.2),
           "dirichlet": (49.6, 1.0)}
ROWS = ("linear", "poe", "maxpool", "rank", "router", "veto")


def main():
    from utils.scannet200 import build_scores, load_index, raw_votes
    from utils.paths import FUSION

    us, uc, zuni = load_index()
    y = np.load(f"{FUSION}/gtmatch_full.npz")["ycls"].astype(np.int64)
    S2, S3 = build_scores("eva", us, uc, zuni)
    has = S2.max(1) > -0.5

    cols = instance_columns()                      # the evaluator's 198 instance classes
    inv = -np.ones(200, np.int64)
    inv[cols] = np.arange(len(cols))
    y198 = np.where(y >= 0, inv[np.maximum(y, 0)], -1)

    A2 = SE.ctr(S2[:, cols][has].astype(np.float64))
    A3 = SE.ctr(S3[:, cols][has].astype(np.float64))
    ok = ((y198 >= 0) & has)[has]
    yy = y198[has][ok]
    inst = lambda lab: 100.0 * float((lab == yy).mean())
    base2d = inst(A2[ok].argmax(1))

    print(f"Open-YOLO3D (EVA) + Uni3D on ScanNet200: {int(has.sum())} proposals fitted, "
          f"{int(ok.sum())} scored over {len(cols)} instance classes\n")
    print(f"{'rule':10s} {'w':>7s} {'acc':>7s} {'delta':>7s}    printed")
    print(f"{'2D only':10s} {'':7s} {base2d:7.2f} {'':7s}    {48.4:5.1f}")
    worst = abs(round(base2d, 1) - 48.4)

    for name in ROWS:
        tf = R.SPACE_OF[name]
        w = float(np.clip(SE.soft_em(tf(A2), tf(A3)), 0.0, 1.0))
        a = inst(R.RULES[name](A2[ok], A3[ok], w).argmax(1))
        p = PRINTED[name]
        print(f"{name:10s} {w:7.4f} {a:7.2f} {a - base2d:+7.2f}    {p[0]:5.1f} ({p[1]:+.1f})")
        worst = max(worst, abs(round(a, 1) - p[0]), abs(round(a - base2d, 1) - p[1]))

    # Vote space: Open-YOLO3D's own label scores, and the shape channel entering as the Dirichlet
    # prior the multinomial asks for. The placebo replaces the shape prior by a uniform one of the
    # same mass, which smooths the histogram by exactly as much while carrying no shape evidence.
    V = raw_votes(us, uc)[:, cols][has]
    Vc = V - V.mean(1, keepdims=True)
    base_vote = inst(Vc[ok].argmax(1))
    print(f"\n{'votes':10s} {'':7s} {base_vote:7.2f} {'':7s}    {48.6:5.1f}")
    worst = max(worst, abs(round(base_vote, 1) - 48.6))
    for name, uniform in (("dirichlet", False), ("dir-placebo", True)):
        a = inst(R.r_dirichlet(V[ok], A3[ok], 1.0, uniform=uniform).argmax(1))
        if name in PRINTED:
            p = PRINTED[name]
            print(f"{name:10s} {1.0:7.4f} {a:7.2f} {a - base_vote:+7.2f}    {p[0]:5.1f} ({p[1]:+.1f})")
            worst = max(worst, abs(round(a, 1) - p[0]), abs(round(a - base_vote, 1) - p[1]))
        else:
            print(f"{name:10s} {1.0:7.4f} {a:7.2f} {a - base_vote:+7.2f}    (not printed)")

    print(f"\nlargest difference from a printed value: {worst:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
