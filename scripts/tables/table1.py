"""Table I: labelling accuracy under fusion, on three validation sets.

Every cell is computed here from the score matrices :mod:`utils.settings` supplies, and printed
beside the value the paper carries. The columns are the paper's:

``w-hat``     Algorithm 1's weight, fitted on that dataset's proposals and reading no label;
``sd``        the standard deviation of the per-scene weights, each fitted on one scene alone;
``w*``        the oracle weight, the accuracy-maximising scalar on a 0.01 grid, found with labels;
``2D``        the pipeline alone, at ``w = 0``;
``@w-hat``    accuracy under the dataset-level weight, which is what is deployed;
``@w-hat_i``  accuracy under each scene's own weight;
``rec``       the share of the oracle gain that ``w-hat`` recovers.

``w*`` measures nothing on its own: it is the denominator of ``rec``. ``sd`` is the spread of a
second estimator rather than an error bar on the first; the mean of the per-scene weights agrees
with ``w-hat`` to within 0.04 in every cell.

    python scripts/tables/table1.py                  every cell
    python scripts/tables/table1.py --dataset replica
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils.settings import DATASETS, TABLE1_ROWS, load  # noqa: E402

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 3)

# The paper's cells: (w-hat, sd, w*, 2D, @w-hat, @w-hat_i, rec). A dash marks a cell the paper
# leaves empty, namely OpenMask3D on ScanNet++, for which it publishes no proposals.
PRINTED = {
    ("OY (EVA)", "scannet200"): (0.352, 0.09, 0.35, 48.4, 53.0, 52.9, 99),
    ("OY (CLIP)", "scannet200"): (0.231, 0.10, 0.24, 41.1, 48.0, 48.0, 97),
    ("OpenMask3D", "scannet200"): (0.219, 0.09, 0.30, 29.4, 33.9, 33.9, 81),
    ("Open3DIS", "scannet200"): (0.236, 0.09, 0.21, 37.5, 44.6, 43.5, 100),
    ("OY (EVA)", "replica"): (0.467, 0.08, 0.31, 61.2, 71.2, 69.1, 93),
    ("OY (CLIP)", "replica"): (0.352, 0.07, 0.18, 65.5, 71.2, 72.7, 67),
    ("OpenMask3D", "replica"): (0.376, 0.07, 0.16, 60.4, 67.6, 69.1, 67),
    ("Open3DIS", "replica"): (0.163, 0.05, 0.15, 51.1, 58.4, 58.2, 97),
    ("OY (EVA)", "scannetpp"): (0.472, 0.08, 0.57, 44.2, 69.3, 70.1, 91),
    ("OY (CLIP)", "scannetpp"): (0.352, 0.13, 0.44, 50.2, 70.5, 70.5, 93),
    ("Open3DIS", "scannetpp"): (0.085, 0.04, 0.12, 68.0, 71.8, 71.6, 96),
}


def cell(pipeline, dataset, head, classes=198):
    """One cell of Table I."""
    S2, S3, y, covered, scene = load(pipeline, dataset, head=head, classes=classes)
    S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
    ok = (y >= 0) & covered
    yy = y[ok]

    w = float(SE.soft_em(S2, S3))                    # Alg. 1, over every proposal, reading no label
    acc = lambda v: 100.0 * float((((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
    sweep = np.array([acc(v) for v in GRID])

    # Each scene fitted on its own proposals and nothing else. A scene whose iteration does not
    # return a finite weight falls back to the pooled one rather than being dropped, so that the
    # per-scene column is scored over the same proposals as the pooled one.
    per, wsc = [], np.empty(len(S2))
    for s in sorted(set(scene.tolist())):
        m = scene == s
        v = float(SE.soft_em(S2[m], S3[m]))
        if not np.isfinite(v):
            v = w
        wsc[m] = v
        per.append(v)
    per = np.asarray(per)
    lab_i = ((1 - wsc)[:, None] * S2 + wsc[:, None] * S3).argmax(1)

    a2, aw = acc(0.0), acc(w)
    return dict(n=int(ok.sum()), n_scenes=len(per), w=w, sd=float(per.std()),
                w_mean=float(per.mean()), w_star=float(GRID[int(sweep.argmax())]),
                acc_2d=a2, acc_w=aw, acc_wi=100.0 * float((lab_i[ok] == yy).mean()),
                acc_oracle=float(sweep.max()),
                rec=100.0 * (aw - a2) / max(1e-9, float(sweep.max()) - a2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--classes", type=int, default=198, choices=(198, 200))
    a = ap.parse_args()
    datasets = [a.dataset] if a.dataset else list(DATASETS)

    print(f"{'dataset':11s} {'method':11s} {'n':>6s} {'w':>7s} {'sd':>6s} {'w*':>6s} "
          f"{'2D':>7s} {'@w':>7s} {'@w_i':>7s} {'rec':>5s}   (printed below each row)")
    worst = 0.0
    for ds in datasets:
        for label, pipeline, head in TABLE1_ROWS:
            if ds == "scannetpp" and pipeline == "openmask3d":
                continue                     # no ScanNet++ proposals are published for it
            r = cell(pipeline, ds, head, classes=a.classes)
            p = PRINTED[(label, ds)]
            print(f"{ds:11s} {label:11s} {r['n']:6d} {r['w']:7.3f} {r['sd']:6.2f} "
                  f"{r['w_star']:6.2f} {r['acc_2d']:7.2f} {r['acc_w']:7.2f} {r['acc_wi']:7.2f} "
                  f"{r['rec']:4.0f}%")
            print(f"{'':11s} {'printed':11s} {'':6s} {p[0]:7.3f} {p[1]:6.2f} {p[2]:6.2f} "
                  f"{p[3]:7.1f} {p[4]:7.1f} {p[5]:7.1f} {p[6]:4.0f}%")
            for got, want, nd in ((r["w"], p[0], 3), (r["sd"], p[1], 2), (r["w_star"], p[2], 2),
                                  (r["acc_2d"], p[3], 1), (r["acc_w"], p[4], 1),
                                  (r["acc_wi"], p[5], 1)):
                worst = max(worst, abs(round(got, nd) - want))
    print(f"\nlargest difference from a printed value, at its printed precision: {worst:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
