"""SenseFuse: fusion of a 2D open-vocabulary labeller with a frozen 3D shape encoder.

Both heads score the same instance mask over the same class names. This module is the entry point
for the two tasks the method performs, namely assigning a label and scoring a benchmark:

    python main.py                        labelling accuracy, every setting in the paper
    python main.py --dataset scannet200   one dataset
    python main.py --task ap              instance-segmentation AP
    python main.py --gate                 a single assertion on the first row of Table I

or as a library:

    from main import accuracy
    r = accuracy("openyolo3d", "scannet200", head="eva")
    r["acc_2d"], r["w"], r["acc_fused"]        # 48.44, 0.3522, 52.97

The procedure implemented by `accuracy`, in the order given in the method section:

  1. load the pipeline's mask proposals and the two heads' score matrices over the same class
     names: `S2` from the 2D labeller and `S3` from the frozen 3D shape encoder (Uni3D);
  2. row-centre both matrices. A row's mean carries the labeller's per-mask confidence, which is
     not comparable across heads; centring removes it;
  3. estimate the fusion weight `w` with Algorithm 1 if no weight is supplied. No ground-truth
     label is read at any stage of this step;
  4. report the accuracy of `argmax_c[(1-w) S2 + w S3]` over the ground-truth-matched masks,
     together with the 2D head alone (`w = 0`) and the oracle sweep over `w`.

ScanNet200 is scored over the benchmark's 198 instance classes. `wall` and `floor` are stuff classes
for which the instance evaluator provides no column, and predictions assigned either label are
discarded.
"""
import argparse
import os
import sys

from utils.paths import *  # noqa: F403

import numpy as np  # noqa: E402

from utils.settings import SETTINGS, load  # noqa: E402


def accuracy(pipeline, dataset, head="eva", w=None, estimator="soft", classes=198):
    """Labelling accuracy of the fused head, the 2D head alone, and the oracle sweep.

    ``w=None`` estimates the weight with Alg. 1 from the scores alone (no labels). ``classes=198``
    restricts ScanNet200 to the instance classes the AP evaluator scores; pass 200 to reproduce
    the pre-2026-09-12 numbers. Returns percentages, not fractions.
    """
    if estimator not in ("soft", "hard"):
        raise ValueError("estimator must be 'soft' (what the paper deploys) or 'hard'")

    S2, S3, y, covered, _scene = load(pipeline, dataset, head=head, classes=classes)

    from models import estimator as SE
    S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
    ok = (y >= 0) & covered
    if w is None:
        w = float(SE.EM[estimator](S2, S3))        # Alg. 1; reads no label

    yy = y[ok]
    acc = lambda v: 100.0 * float((((1 - v) * S2 + v * S3).argmax(1)[ok] == yy).mean())
    grid = np.round(np.arange(0.0, 1.0001, 0.01), 3)
    sweep = np.array([acc(v) for v in grid])
    a2, af = acc(0.0), acc(w)
    return dict(pipeline=pipeline, dataset=dataset, head=head, classes=classes,
                n=int(ok.sum()), w=w, acc_2d=a2, acc_fused=af, delta=af - a2,
                acc_oracle=float(sweep.max()), w_oracle=float(grid[int(sweep.argmax())]),
                recovery=100.0 * (af - a2) / max(1e-9, float(sweep.max()) - a2))


# Table I's first row, as printed. Any change to the loaders or the estimator that moves these
# has moved the paper, and this assertion is meant to be the first thing that says so.
GATE = dict(n=6534, w=0.352, acc_2d=48.4, acc_fused=53.0)


def _gate():
    r = accuracy("openyolo3d", "scannet200", head="eva")
    bad = [f"{k}: {r[k]:.4f} vs Table I's {v}" for k, v in GATE.items()
           if abs(r[k] - v) > (0.051 if isinstance(v, float) else 0)]
    print("GATE (Table I, OY(EVA)/ScanNet200):", "PASS" if not bad else "FAIL " + "; ".join(bad))
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pipeline", choices=sorted({p for p, _ in SETTINGS}))
    ap.add_argument("--dataset", choices=sorted({d for _, d in SETTINGS}))
    ap.add_argument("--head")
    ap.add_argument("--w", type=float, default=None, help="skip Alg. 1 and use this weight")
    ap.add_argument("--classes", type=int, default=198, choices=(198, 200))
    ap.add_argument("--gate", action="store_true", help="only check Table I's first row")
    ap.add_argument("--task", choices=("label", "ap"), default="label")
    a, rest = ap.parse_known_args()
    if a.task == "ap":
        from utils import ap as scorer
        argv = rest or [a.dataset or "sn200"] + ([a.head] if a.head else [])
        return scorer.main(argv)
    if a.gate:
        sys.exit(0 if _gate() else 1)
    print(f"{'pipeline/dataset':28s} {'head':7s} {'n':>6s} {'w':>7s} {'2D':>7s} "
          f"{'fused':>7s} {'delta':>7s} {'oracle':>7s} {'rec':>6s}")
    for (pipeline, dataset), heads in sorted(SETTINGS.items()):
        if a.pipeline and pipeline != a.pipeline:
            continue
        if a.dataset and dataset != a.dataset:
            continue
        for head in heads:
            if a.head and head != a.head:
                continue
            r = accuracy(pipeline, dataset, head=head, w=a.w, classes=a.classes)
            print(f"{pipeline + '/' + dataset:28s} {head:7s} {r['n']:6d} {r['w']:7.4f} "
                  f"{r['acc_2d']:7.2f} {r['acc_fused']:7.2f} {r['delta']:+7.2f} "
                  f"{r['acc_oracle']:7.2f} {r['recovery']:5.0f}%", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
