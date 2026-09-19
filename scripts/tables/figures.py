"""The quantities plotted in Figures 3, 5 and 6.

The figures themselves are typeset with the paper; what is reproducible from a checkout is the data
behind them, which is what this prints, together with the claims the captions and the text make
about that data.

``--fig3``  The score plane of Figure 3. Each ground-truth-matched proposal contributes one correct
            class score and its wrong-class scores, in the standardised units ``u_h = S_h / sigma_h``
            the figure uses. Printed here as the two marginals' moments and the background
            covariance the ellipses are drawn from.
``--fig5``  The per-class accuracy change at the deployed weight, over classes with at least ten
            matched masks. The caption's claim is that each of the five classes it shows as degraded
            has a median point count below one thousand, which is checked: what fusion costs is
            confined to instances too small for the shape encoder to read.
``--fig6``  The accuracy curve over the weight, for one setting on each of the three datasets, with
            the oracle weight, the estimate, and the band within one point of the peak.

    python scripts/tables/figures.py --all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils.settings import DATASETS, load  # noqa: E402

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 3)


def _headline(dataset="scannet200"):
    S2, S3, y, covered, scene = load("openyolo3d", dataset, head="eva")
    S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
    return S2, S3, y, (y >= 0) & covered, scene


def fig3():
    """The score plane: correct-class scores against background, in standardised units."""
    S2, S3, y, ok, _ = _headline()
    yy = y[ok]
    A, B = S2[ok], S3[ok]
    rows = np.arange(len(yy))
    hit = np.zeros(A.shape, bool)
    hit[rows, yy] = True

    print("Figure 3: Open-YOLO3D (EVA02-E) and Uni3D on ScanNet200, "
          f"{len(yy)} ground-truth-matched proposals\n")
    for name, S in (("2D (EVA02-E)", A), ("shape (Uni3D)", B)):
        sd = S.std(1, keepdims=True) + 1e-12
        u = S / sd
        print(f"  {name:14s} correct class  mean {u[hit].mean():+.3f}  sd {u[hit].std():.3f}")
        print(f"  {'':14s} background     mean {u[~hit].mean():+.3f}  sd {u[~hit].std():.3f}")
    # The ellipses are the background covariance: the figure's claim is that it is nearly diagonal,
    # which is Assumption 1 and what Table IV then tests by correcting for what is left of it.
    u2 = (A / (A.std(1, keepdims=True) + 1e-12))[~hit]
    u3 = (B / (B.std(1, keepdims=True) + 1e-12))[~hit]
    r = float(np.corrcoef(u2, u3)[0, 1])
    print(f"\n  background correlation between the two channels: {r:+.3f}")


def fig5():
    """Per-class accuracy change at the deployed weight, and the point-count claim."""
    from utils.scannet200 import get_masks, load_index
    S2, S3, y, ok, _ = _headline()
    w = float(SE.soft_em(S2, S3))
    lab2 = S2.argmax(1)
    labf = ((1 - w) * S2 + w * S3).argmax(1)
    yy, l2, lf = y[ok], lab2[ok], labf[ok]

    us, uc, _zuni = load_index()
    sizes = np.zeros(len(y), np.int64)
    for s in sorted(set(us.tolist())):
        m = get_masks(s)
        r = np.where(us == s)[0]
        sizes[r] = m[:, uc[r]].sum(0)
    sz = sizes[ok]

    rows = []
    for c in np.unique(yy):
        sel = yy == c
        if sel.sum() < 10:
            continue
        rows.append((int(c), int(sel.sum()),
                     100.0 * float((lf[sel] == c).mean() - (l2[sel] == c).mean()),
                     float(np.median(sz[sel]))))
    rows.sort(key=lambda r: -r[2])
    import scannet200_constants as C200
    from utils.settings import instance_columns
    names = [list(C200.CLASS_LABELS_200)[i] for i in instance_columns()]

    print(f"Figure 5: per-class accuracy change at w-hat = {w:.4f}, over the "
          f"{len(rows)} classes with at least ten matched masks\n")
    print("  five most improved")
    for c, n, d, med in rows[:5]:
        print(f"    {names[c]:24s} n {n:4d}  {d:+6.1f} pp   median points {med:8.0f}")
    print("  five most degraded")
    shown = rows[-5:][::-1]
    for c, n, d, med in shown:
        print(f"    {names[c]:24s} n {n:4d}  {d:+6.1f} pp   median points {med:8.0f}")
    worst = max(r[3] for r in shown)
    print(f"\n  every class the figure shows as degraded has a median point count below 1,000: "
          f"{'yes' if worst < 1000 else 'NO'} (largest {worst:.0f})")


def fig6():
    """The accuracy curve over the weight, on one setting per dataset."""
    print("Figure 6: Open-YOLO3D (EVA02-E) and Uni3D, accuracy against the fusion weight\n")
    print(f"  {'dataset':11s} {'w*':>6s} {'peak':>7s} {'w-hat':>7s} {'@w-hat':>7s} "
          f"{'within 1 pp':>16s}")
    for ds in DATASETS:
        S2, S3, y, ok, _ = _headline(ds)
        yy = y[ok]
        sweep = np.array([100.0 * float((((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
                          for v in GRID])
        w = float(SE.soft_em(S2, S3))
        band = GRID[sweep >= sweep.max() - 1.0]
        # at the exact fixed point, not at its nearest grid node: snapping first would print one
        # weight and measure another.
        at_w = 100.0 * float((((1 - w) * S2 + w * S3).argmax(1)[ok] == yy).mean())
        print(f"  {ds:11s} {GRID[int(sweep.argmax())]:6.2f} {sweep.max():7.2f} {w:7.3f} "
              f"{at_w:7.2f}   [{band.min():.2f}, {band.max():.2f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("fig3", "fig5", "fig6", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    ran = False
    for flag, fn in (("fig3", fig3), ("fig5", fig5), ("fig6", fig6)):
        if a.all or getattr(a, flag):
            print()
            fn()
            ran = True
    if not ran:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
