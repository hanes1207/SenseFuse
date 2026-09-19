"""Table II: instance-segmentation AP, under two submission rules.

Each cell is scored here and printed beside the value the paper carries. The two rules are held
fixed across methods so that the columns are comparable: ``top-1`` submits one prediction per
proposal, and ``flat`` submits the top-600 entries of the flattened proposal-class matrix. Every
prediction carries confidence 1.0, as each released implementation does. The flattening itself is a
property of the pipeline rather than of the table: Open-YOLO3D and OpenMask3D rank per-mask
max-normalised scores, Open3DIS ranks raw ones.

The fusion weight is not a constant here. Each ``+Uni3D`` cell is scored at the weight Algorithm 1
estimates for that setting, which is the same weight Table I prints, so the two tables cannot drift
apart. A block therefore begins by estimating its weight and reporting it.

The Open3DIS rows carry a dagger in the paper: they are a re-run of that pipeline in cosine space
rather than a reading of its released artefact, because the released rows carry a collapsed label
instead of a score matrix.

    python scripts/tables/table2.py                 every block, roughly four hours on one CPU
    python scripts/tables/table2.py --block oy_sn200
    python scripts/tables/table2.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from models import estimator as SE  # noqa: E402
from utils import ap as A  # noqa: E402
from utils.settings import load  # noqa: E402

# block -> (label, setting, runner, printed 2D (top-1, flat), printed +Uni3D (top-1, flat))
BLOCKS = {
    "oy_sn200_eva": ("OY (EVA) / ScanNet200", ("openyolo3d", "scannet200", "eva"),
                     lambda w, r: A.run_sn200("eva", w, r), (23.9, 28.3), (24.1, 28.2)),
    "oy_sn200_clip": ("OY (CLIP) / ScanNet200", ("openyolo3d", "scannet200", "clipL"),
                      lambda w, r: A.run_sn200("clipL", w, r), (19.5, 27.1), (20.3, 27.5)),
    "om_sn200": ("OM3D / ScanNet200", ("openmask3d", "scannet200", "clipL"),
                 A.run_sn200_om, (15.5, 24.4), (17.1, 25.8)),
    "o3d_sn200": ("O3DIS / ScanNet200", ("open3dis", "scannet200", "own"),
                  A.run_sn200_o3d_cosine, (15.3, 22.4), (16.6, 23.6)),
    "oy_replica_eva": ("OY (EVA) / Replica", ("openyolo3d", "replica", "eva"),
                       lambda w, r: A.run_replica("eva", w, r), (17.5, 18.3), (19.1, 18.4)),
    "oy_replica_clip": ("OY (CLIP) / Replica", ("openyolo3d", "replica", "clipL"),
                        lambda w, r: A.run_replica("clipL", w, r), (15.9, 17.9), (16.2, 18.4)),
    "om_replica": ("OM3D / Replica", ("openmask3d", "replica", "clipL"),
                   A.run_replica_om, (15.8, 18.0), (16.3, 18.5)),
    "o3d_replica": ("O3DIS / Replica", ("open3dis", "replica", "own"),
                    A.run_replica_o3d, (11.6, 10.0), (14.3, 14.5)),
    "oy_spp_eva": ("OY (EVA) / ScanNet++", ("openyolo3d", "scannetpp", "eva"),
                   lambda w, r: A.run_spp_oy("eva", w, r), (6.1, 6.9), (7.0, 7.5)),
    "oy_spp_clip": ("OY (CLIP) / ScanNet++", ("openyolo3d", "scannetpp", "clipL"),
                    lambda w, r: A.run_spp_oy("clipL", w, r), (6.3, 6.8), (7.4, 7.3)),
    "o3d_spp": ("O3DIS / ScanNet++", ("open3dis", "scannetpp", "own"),
                A.run_spp_o3d, (11.2, 10.5), (12.0, 12.4)),
}


def deployed_weight(pipeline, dataset, head):
    """Algorithm 1's weight for one setting, in the class space that setting is scored in."""
    S2, S3, _, covered, _ = load(pipeline, dataset, head=head)
    S2, S3 = SE.ctr(S2.astype(np.float64)), SE.ctr(S3.astype(np.float64))
    if pipeline == "open3dis" and dataset == "scannet200":
        S2, S3 = S2[covered], S3[covered]     # the shape head does not cover every proposal here
    return float(SE.soft_em(S2, S3))


def _round_half_up(x):
    """Round to one decimal away from zero, which is what the table does; Python rounds halves to
    even, so 16.55 would otherwise read 16.5 against a printed 16.6."""
    import decimal
    return float(decimal.Decimal(repr(x)).quantize(decimal.Decimal("0.1"),
                                                   rounding=decimal.ROUND_HALF_UP))


def run_block(key):
    label, setting, runner, p2d, pfused = BLOCKS[key]
    w = deployed_weight(*setting)
    print(f"\n=== {label}   (Algorithm 1's weight {w:.4f}) ===", flush=True)
    worst = 0.0
    for arm, weight, printed in (("2D only", 0.0, p2d), ("+Uni3D", w, pfused)):
        got = []
        for rule in ("top1", "flat"):
            got.append(runner(weight, rule)["ap"])
        # three decimals, so that a cell sitting on a rounding boundary is visible rather than
        # showing up as a difference of 0.1 that is not one.
        d = [abs(_round_half_up(g) - p) for g, p in zip(got, printed)]
        print(f"  {arm:8s} top-1 {got[0]:7.3f}  flat {got[1]:7.3f}"
              f"   printed {printed[0]:5.1f} / {printed[1]:5.1f}"
              f"   diff {d[0]:.1f} / {d[1]:.1f}", flush=True)
        worst = max(worst, *d)
    print(f"  largest difference from a printed value: {worst:.1f}", flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block", action="append", choices=sorted(BLOCKS))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (label, *_rest) in BLOCKS.items():
            print(f"{k:16s} {label}")
        return 0
    worst = 0.0
    for key in (a.block or list(BLOCKS)):
        worst = max(worst, run_block(key))
    print(f"\nlargest difference over every block run: {worst:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
