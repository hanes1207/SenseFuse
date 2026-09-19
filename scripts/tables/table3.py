"""Table III: what makes a pair of heads complementary.

Six pairs drawn from four frozen encoders, all on Open-YOLO3D's ScanNet200 proposals: two image
towers (EVA02-E and CLIP ViT-L) and two shape towers (Uni3D and OpenShape). Every pair is fused by
the same rule at its own label-free weight, so the only thing that varies across rows is which two
encoders are paired.

``Acc1``, ``Acc2``  each head alone;
``rho``             the mean per-mask Spearman rank correlation between the two heads' class
                    scores, computed over every proposal both heads score and reading no label;
``Ident.``          among the masks both heads get wrong, the share on which they are wrong in the
                    same way;
``Resc.``           the share of masks the first head gets wrong and the second gets right;
``Acc``             the fused accuracy.

The two image towers agree on 41 % of their joint errors and rescue little; a cross-modal pair
agrees on about 6 % and rescues more. Competence still matters: OpenShape is the least correlated
partner of all and gains the least, because it is also the weakest.

    python scripts/tables/table3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils.settings import instance_columns  # noqa: E402

PAIRS = (("EVA02-E", "CLIP-L", "2D+2D"), ("Uni3D", "OpenShape", "3D+3D"),
         ("EVA02-E", "Uni3D", "2D+3D"), ("CLIP-L", "Uni3D", "2D+3D"),
         ("EVA02-E", "OpenShape", "2D+3D"), ("CLIP-L", "OpenShape", "2D+3D"))

# What the paper prints: (Acc1, Acc2, rho, Ident., Resc., Acc).
PRINTED = {("EVA02-E", "CLIP-L"): (48.4, 41.1, 0.677, 41.4, 7.8, 49.6),
           ("Uni3D", "OpenShape"): (16.1, 3.6, 0.394, 4.4, 2.5, 12.4),
           ("EVA02-E", "Uni3D"): (48.4, 16.1, 0.226, 5.9, 7.5, 53.0),
           ("CLIP-L", "Uni3D"): (41.1, 16.1, 0.124, 6.0, 9.0, 48.0),
           ("EVA02-E", "OpenShape"): (48.4, 3.6, 0.209, 2.4, 1.6, 49.3),
           ("CLIP-L", "OpenShape"): (41.1, 3.6, 0.079, 2.2, 2.2, 42.8)}


def spearman_rho(A, B):
    """Mean per-mask Spearman correlation between two heads' class rankings."""
    ra = np.argsort(np.argsort(-A, 1), 1).astype(np.float64)
    rb = np.argsort(np.argsort(-B, 1), 1).astype(np.float64)
    a = ra - ra.mean(1, keepdims=True)
    b = rb - rb.mean(1, keepdims=True)
    return float(((a * b).sum(1)
                  / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean())


def channels():
    """The four encoders' score matrices over Open-YOLO3D's ScanNet200 proposals."""
    from utils.paths import FEATS, FUSION
    from utils.scannet200 import build_scores, l2, load_index, loadcat, text_anchors

    us, uc, zuni = load_index()
    y = np.load(f"{FUSION}/gtmatch_full.npz")["ycls"].astype(np.int64)
    S_eva, S_uni = build_scores("eva", us, uc, zuni)
    S_cl, _ = build_scores("clipL", us, uc, zuni)

    D = loadcat(f"{FEATS}/openshape_oy_s*.npz", ["scene", "col", "z"])
    k = {(D["scene"].astype(str)[i], int(D["col"][i])): i for i in range(len(D["scene"]))}
    sel = np.array([k[(s, c)] for s, c in zip(us, uc)])
    S_os = (l2(D["z"].astype(np.float32)[sel])
            @ l2(np.load(f"{FEATS}/bigg_text_200.npy").astype(np.float32)).T)
    return {"EVA02-E": S_eva, "CLIP-L": S_cl, "Uni3D": S_uni, "OpenShape": S_os}, y


def main():
    raw, y = channels()
    valid = {n: np.asarray(raw[n]).max(1) > -0.5 for n in raw}

    cols = instance_columns()
    inv = -np.ones(200, np.int64)
    inv[cols] = np.arange(len(cols))
    y198 = np.where(y >= 0, inv[np.maximum(y, 0)], -1)
    F = {n: SE.ctr(np.asarray(raw[n])[:, cols].astype(np.float64)) for n in raw}
    ok = (y198 >= 0) & valid["EVA02-E"]
    yy = y198[ok]
    print(f"Open-YOLO3D on ScanNet200: {int(ok.sum())} ground-truth-matched proposals "
          f"over {len(cols)} instance classes\n")
    print(f"{'pair':22s} {'mod':7s} {'Acc1':>6s} {'Acc2':>6s} {'rho':>6s} {'Ident':>7s} "
          f"{'Resc':>6s} {'Acc':>6s}    printed Acc (rho)")

    worst = 0.0
    for a, b, modality in PAIRS:
        Fa, Fb = F[a][ok], F[b][ok]
        pa, pb = Fa.argmax(1), Fb.argmax(1)
        acc_a, acc_b = 100 * float((pa == yy).mean()), 100 * float((pb == yy).mean())
        wa, wb = pa != yy, pb != yy
        both = wa & wb
        ident = 100 * float((pa[both] == pb[both]).mean())
        resc = 100 * float((wa & ~wb).mean())
        v = valid[a] & valid[b]
        w = float(SE.soft_em(F[a][v], F[b][v]))
        acc = 100 * float((((1 - w) * Fa + w * Fb).argmax(1) == yy).mean())
        rho = spearman_rho(F[a][v], F[b][v])

        p = PRINTED[(a, b)]
        print(f"{a + '+' + b:22s} {modality:7s} {acc_a:6.1f} {acc_b:6.1f} {rho:6.3f} "
              f"{ident:6.1f}% {resc:5.1f}% {acc:6.1f}    {p[5]:.1f} ({p[2]:.3f})")
        for got, want, nd in ((acc_a, p[0], 1), (acc_b, p[1], 1), (rho, p[2], 3),
                              (ident, p[3], 1), (resc, p[4], 1), (acc, p[5], 1)):
            worst = max(worst, abs(round(got, nd) - want))
    print(f"\nlargest difference from a printed value: {worst:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
