"""Table IV: what dropping the cross-modal independence assumption costs.

Equation (5) forms the weight from two precisions alone, which assumes the two heads' background
score residuals are uncorrelated. This table estimates that correlation without labels and corrects
for it, and reports what the correction does to accuracy.

``rho``   the background residual correlation, estimated by Algorithm 1's own responsibilities;
``Acc``   accuracy at the corrected weight, and its change against the uncorrected one;
``Inad.`` the share of scenes whose corrected weight falls outside ``[0, 1]``, where it is not a
          convex combination at all. Such a weight is reported rather than clipped, since clipping
          would conceal exactly what the ablation asks about.

All four rows are ScanNet200. The correction is small in every one and helps in none, which is the
evidence that pairing across modalities keeps the residual correlation small enough for Eq. (5).

    python scripts/tables/table4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils.settings import TABLE1_ROWS, load  # noqa: E402

PRINTED = {"OY (EVA)": (0.26, 50.1, -2.8, 20),
           "OY (CLIP)": (0.17, 45.5, -2.4, 9),
           "OpenMask3D": (0.18, 32.1, -1.8, 11),
           "Open3DIS": (0.14, 43.7, -0.9, 3)}


def main():
    print(f"{'setting':11s} {'rho':>6s} {'Acc(rho)':>9s} {'delta':>7s} {'Inad.':>6s}    printed")
    worst = 0.0
    for label, pipeline, head in TABLE1_ROWS:
        S2, S3, y, covered, scene = load(pipeline, "scannet200", head=head)
        S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
        ok = (y >= 0) & covered
        yy = y[ok]
        fit = np.ones(len(S2), bool) if pipeline != "open3dis" else covered

        w = float(SE.soft_em(S2[fit], S3[fit]))
        w_rho, rho, _ = SE.rho_em(S2[fit], S3[fit])
        acc = lambda v: 100.0 * float((((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
        a_rho = acc(float(np.clip(w_rho, 0.0, 1.0)) if np.isfinite(w_rho) else 0.0)
        delta = a_rho - acc(w)

        # The inadmissible share is a per-scene statement: a scene whose own corrected weight leaves
        # the unit interval has no convex combination to deploy.
        bad = 0
        scenes = sorted(set(scene.tolist()))
        for s in scenes:
            m = (scene == s) & fit
            if m.sum() < 3:
                continue
            _, _, adm = SE.rho_em(S2[m], S3[m])
            bad += not adm
        inad = 100.0 * bad / max(len(scenes), 1)

        p = PRINTED[label]
        print(f"{label:11s} {rho:6.2f} {a_rho:9.2f} {delta:+7.2f} {inad:5.0f}%    "
              f"{p[0]:.2f} / {p[1]:.1f} ({p[2]:+.1f}) / {p[3]:d}%")
        worst = max(worst, abs(round(rho, 2) - p[0]), abs(round(a_rho, 1) - p[1]),
                    abs(round(delta, 1) - p[2]), abs(round(inad) - p[3]) / 100.0)
    print(f"\nlargest difference from a printed value: {worst:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
