"""The estimator ablations of Section IV-D.

Four claims are made there about Algorithm 1, and each is checked here against the value the paper
states. They concern the estimator rather than the fusion, so all four are accuracy measurements and
none involves a submission rule.

``--w0``        Initialisation. The weight is re-estimated from five starting points. Under a hard
                argmax pseudo-label the estimate moves by more than 0.01 across starts on many
                scenes; the softmax relaxation suppresses that, at no cost in accuracy.
``--tau-beta``  The two constants of the relaxation, swept over a twentyfold and a sixtyfourfold
                range. The accuracy spans are small enough that one default serves every setting.
``--plateau``   The width of the region within one accuracy point of the peak. The estimate does not
                have to be exact, it has to land on this plateau.
``--constant``  What a hand-chosen constant weight would cost. The comparison is generous to the
                constant: it is tuned on the labels of all three validation sets at once, which is
                supervision the estimator never receives.

    python scripts/tables/ablations.py --all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils.settings import DATASETS, TABLE1_ROWS, load  # noqa: E402

GRID = np.round(np.arange(0.0, 1.0001, 0.01), 3)
STARTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _cells(datasets=DATASETS):
    """(label, dataset, S2, S3, y, ok, scene) for every setting Table I reports."""
    for ds in datasets:
        for label, pipeline, head in TABLE1_ROWS:
            if ds == "scannetpp" and pipeline == "openmask3d":
                continue
            S2, S3, y, covered, scene = load(pipeline, ds, head=head)
            S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
            yield f"{label}/{ds}", S2, S3, y, (y >= 0) & covered, scene


def w0_sweep():
    """Claim: across five initialisations the hard rule moves on 69 of ScanNet200's 312 scenes and
    the soft rule on 10, with labelling accuracy within 0.3 points.

    Both quantities are per-scene: the weight is estimated on one scene at a time, from each of the
    five starts, and the corpus accuracy is then read off with each scene carrying its own weight.
    Measuring instead one weight per corpus would hide the effect, since the instability the sweep is
    about is the instability of a single scene's estimate.
    """
    S2, S3, y, covered, scene = load("openyolo3d", "scannet200", head="eva")
    S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
    ok = (y >= 0) & covered
    scenes = sorted(set(scene.tolist()))
    print("initialisation sweep, Open-YOLO3D (EVA) on ScanNet200, w0 in "
          f"{{{', '.join(str(s) for s in STARTS)}}}\n")
    for name, em in (("hard", SE.hard_em), ("soft", SE.soft_em)):
        moved, hits, n = 0, np.zeros(len(STARTS)), 0
        for s in scenes:
            m = scene == s
            ws = [float(em(S2[m], S3[m], w0=v)) for v in STARTS]
            if max(ws) - min(ws) > 0.01:
                moved += 1
            sel = m & ok
            if not sel.any():
                continue
            a, b, yy = S2[sel], S3[sel], y[sel]
            n += int(sel.sum())
            for i, w in enumerate(ws):
                hits[i] += float((((1 - w) * a + w * b).argmax(1) == yy).sum())
        accs = 100.0 * hits / max(1, n)
        print(f"  {name:5s}  scenes whose estimate moves by more than 0.01: {moved:3d} / "
              f"{len(scenes)}   accuracy over the five starts "
              f"{accs.min():.2f}-{accs.max():.2f}, span {accs.max() - accs.min():.2f} points")
    print("\n  printed: 69 scenes hard, 10 soft, the soft rule's accuracy within 0.3 points")


def tau_beta_sweep():
    """Claim: median accuracy spans of 0.7 and 0.2 points, worst 2.2 and 0.7."""
    taus = (0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0)      # a twentyfold range about the default 0.25
    betas = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)     # a sixtyfourfold range about the default 8
    spans = {"tau0": [], "beta": []}
    print(f"{'setting':22s} {'tau0 span':>10s} {'beta span':>10s}")
    for label, S2, S3, y, ok, _ in _cells():
        yy = y[ok]
        acc = lambda w: 100.0 * float((((1 - w) * S2 + w * S3).argmax(1)[ok] == yy).mean())
        a_tau = [acc(float(SE.soft_em(S2, S3, tau0=t))) for t in taus]
        a_beta = [acc(float(SE.soft_em(S2, S3, beta=b))) for b in betas]
        spans["tau0"].append(max(a_tau) - min(a_tau))
        spans["beta"].append(max(a_beta) - min(a_beta))
        print(f"{label:22s} {spans['tau0'][-1]:10.2f} {spans['beta'][-1]:10.2f}")
    for k, printed in (("tau0", (0.7, 2.2)), ("beta", (0.2, 0.7))):
        v = np.asarray(spans[k])
        print(f"\n  {k:5s} median {np.median(v):.2f}  worst {v.max():.2f}"
              f"   printed median {printed[0]} worst {printed[1]}")


def plateau():
    """Claim: the region within one point of the peak spans 0.06 to 0.27 across configurations."""
    print(f"{'setting':22s} {'w-hat':>7s} {'w*':>6s} {'plateau':>16s} {'width':>7s}  in it?")
    widths = []
    for label, S2, S3, y, ok, _ in _cells():
        yy = y[ok]
        sweep = np.array([100.0 * float((((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
                          for v in GRID])
        inside = GRID[sweep >= sweep.max() - 1.0]
        w = float(SE.soft_em(S2, S3))
        widths.append(float(inside.max() - inside.min()))
        print(f"{label:22s} {w:7.3f} {GRID[int(sweep.argmax())]:6.2f} "
              f"[{inside.min():.2f}, {inside.max():.2f}]{'':4s} {widths[-1]:7.2f}  "
              f"{'yes' if inside.min() <= w <= inside.max() else 'NO'}")
    print(f"\n  widths span {min(widths):.2f} to {max(widths):.2f}   printed 0.06 to 0.27")


def constant():
    """Claim: the best jointly tuned constant recovers a mean 81 % against 89 % for the estimate,
    and a fixed 0.30 drives ScanNet++/Open3DIS below its own 2D accuracy."""
    rows, accs = [], []
    for label, S2, S3, y, ok, _ in _cells():
        yy = y[ok]
        acc = lambda v, S2=S2, S3=S3, ok=ok, yy=yy: 100.0 * float(
            (((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
        sweep = np.array([acc(v) for v in GRID])
        rows.append((label, acc(0.0), float(sweep.max()), float(SE.soft_em(S2, S3)), sweep))
        accs.append(acc)

    def recovery(v):
        return np.asarray([100.0 * (sweep[int(round(v * 100))] - a2) / max(1e-9, best - a2)
                           for _, a2, best, _w, sweep in rows])

    # at the exact fixed point rather than at its nearest grid node
    rec_hat = np.asarray([100.0 * (acc_at(w) - a2) / max(1e-9, best - a2)
                          for (_, a2, best, w, _sweep), acc_at in zip(rows, accs)])
    best_const, best_mean = None, -1e9
    for v in GRID:
        r = np.asarray([100.0 * (sweep[int(round(v * 100))] - a2) / max(1e-9, best - a2)
                        for _, a2, best, _w, sweep in rows])
        if r.mean() > best_mean:
            best_const, best_mean = float(v), float(r.mean())
    print(f"  Algorithm 1                    mean recovery {rec_hat.mean():5.1f} %   "
          f"(printed 89 %)")
    print(f"  best constant, tuned on labels mean recovery {best_mean:5.1f} %   "
          f"at w = {best_const:.2f}   (printed 81 %)")
    print()
    for (label, a2, best, _w, sweep), r in zip(rows, recovery(0.30)):
        mark = "  <- below its 2D head" if sweep[30] < a2 else ""
        print(f"  w = 0.30 on {label:22s} 2D {a2:5.2f} -> {sweep[30]:5.2f}   "
              f"recovery {r:6.1f} %{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("w0", "tau-beta", "plateau", "constant", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    ran = False
    for flag, fn, title in (("w0", w0_sweep, "Initialisation"),
                            ("tau_beta", tau_beta_sweep, "The relaxation's two constants"),
                            ("plateau", plateau, "The accuracy plateau"),
                            ("constant", constant, "What a constant weight would cost")):
        if a.all or getattr(a, flag):
            print(f"\n=== {title} ===")
            fn()
            ran = True
    if not ran:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
