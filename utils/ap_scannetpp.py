"""ScanNet++ AP against the official benchmark ground truth.

The scorer uses the official ``ScanNetEval`` together with ground truth built from
``segments_anno.json`` and the official ``map_benchmark.csv``. The ground-truth encoding is dictated
by the evaluator: ``ScanNetEval(scannetpp_benchmark)`` applies ``gts_sem = gts_sem - 16 + 1`` and
constructs ``class * encode + inst`` internally, so ``gt_sem`` must be supplied as
``benchmark_idx + 16``, placing the instance classes at 16 to 99, and ``gt_ins`` as a per-point
running instance index with void encoded as ``0 - sem``. The encoding was verified by scoring the
ground truth against itself, which yields an AP of 1.0."""
import os, sys
from utils.paths import *  # noqa: F403
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
import os, sys, glob, argparse
import numpy as np, torch
from scannetv2_inst_eval import ScanNetEval
from open3dis.dataset.scannetpp import (INSTANCE_BENCHMARK84_SCANNET_PP,
                                        SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP)

# The OV3DIS literature evaluates ScanNet++ on the "100-class subset" (Any3DIS Tab. 3,
# evaluation/eval_openvocab.py:29 passes SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP, 100 names).
# That list is [16 stuff classes, then the 84 instance classes] -- S100[16:] == I84 exactly.
# It is NOT a no-op relabelling of the 84-class eval: segments_anno annotates 892 instances in
# those 16 stuff classes across the 50 val scenes (wall 438, ceiling 65, floor 64, ...), so they
# are has_gt=True. A labeller prompted with only the 84 instance names never predicts them, which
# lands in scannetv2_inst_eval.py's `elif has_gt: ap_current = 0.0` branch, not the `else: nan`
# branch -- so the 16 enter the nanmean as hard zeros and mAP scales by 84/100.
# Open3DIS's own configs/scannetpp*.yaml set num_classes: 84. Scoring over 84 classes rather than
# 100 therefore reads approximately 1.19 times high relative to the published numbers.
_MODE100 = os.environ.get("SPP_CLASSES", "84") == "100"
NAMES = (list(SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP) if _MODE100
         else list(INSTANCE_BENCHMARK84_SCANNET_PP))
N2C = {n: i for i, n in enumerate(NAMES)}          # name -> 0..len(NAMES)-1
_PSHIFT = 16 if _MODE100 else 0                    # idx84 -> idx into the 100-name list
GT_GROUND = f"{O3D_ROOT}/Scannetpp/Scannetpp_3D/val/groundtruth"
# raw_sem in groundtruth/*.pth indexes the FULL instance label list (2752 names), NOT
# semantic_classes.txt. This was verified by decoding: instance_classes.txt yields coherent scene
# contents ("3d printer" x15, "arm chair" x14, "conference table" x6), whereas semantic_classes.txt
# yields incoherent labels.
_SEMNAMES = [l.rstrip("\n") for l in open(os.path.join(ASSETS, "scannetpp", "instance_classes.txt"))]

import csv
# assets/scannetpp/map_benchmark.csv is the official ScanNet++ release file (2,878 classes) as of
# 2026-08-12. An alternative Pointcept copy (1,651 classes) may be selected with MAP_CSV=<path>.
# The official file maps two misspelled labels that the alternative discards -- 'ceilng light' to
# 'ceiling lamp' (7 instances, scene e7af285f7d) and 'carboard box' to 'box' (1 instance, scene
# c50d2d1d42) -- adding 8 ground-truth instances out of 2,653. The measured effect on every
# ScanNet++ AP cell is below 0.1 mAP: only two classes move ('ceiling lamp' by -0.4 AP and 'box' by
# +0.1 AP), which amounts to -0.003 mAP once averaged over the 100 class names.
_MAP_CSV = os.environ.get("MAP_CSV", os.path.join(ASSETS, "scannetpp", "map_benchmark.csv"))
_RAW2INST = {}
with open(_MAP_CSV) as f:
    for row in csv.DictReader(f):
        cls = row["class"]; imt = (row.get("instance_map_to") or "").strip()
        _RAW2INST[cls] = imt if (imt and imt != "None") else cls
def raw2bench(n): return N2C.get(_RAW2INST.get(n, n), -1)

def rle_decode(rle):
    length = rle["length"]; s = rle["counts"]
    starts, nums = [np.asarray(x, dtype=np.int64) for x in (s[0::2], s[1::2])]
    starts -= 1; ends = starts + nums
    m = np.zeros(length, np.uint8)
    for lo, hi in zip(starts, ends): m[lo:hi] = 1
    return m

ANNO_DIR = f"{SCANNETPP_EVAL}/data"

def build_gt_auth(sc):
    """Build the ground truth from the official ``segments_anno.json``.

    The annotation supplies string labels, so no semantic-id space has to be inferred. The mesh
    vertex order was verified to equal the prepared point-cloud order, with identical coordinates.

    Returns ``(gt_sem, gt_ins, inst_list)``, where ``gt_sem`` is ``benchmark_idx + 16`` because the
    evaluator applies ``-16 + 1``, ``gt_ins`` is a running per-point instance index, and
    ``inst_list`` is a list of ``(class_id, point_mask)`` pairs.
    """
    import json
    segs = json.load(open(f"{ANNO_DIR}/{sc}/scans/segments.json"))
    anno = json.load(open(f"{ANNO_DIR}/{sc}/scans/segments_anno.json"))
    seg_ids = np.asarray(segs["segIndices"], dtype=np.int64)
    npts = len(seg_ids)
    gt_sem = np.zeros(npts, np.int32); gt_ins = -np.ones(npts, np.int32)
    inst_list = []
    run = 0
    for obj in anno["segGroups"]:
        c = raw2bench(obj["label"])            # official string label -> benchmark 0..83
        if c < 0: continue
        m = np.isin(seg_ids, np.asarray(obj["segments"], dtype=np.int64))
        if not m.any(): continue
        gt_sem[m] = c + 16                     # instance classes live at 16..99 in benchmark sem space
        gt_ins[m] = run
        inst_list.append((c + 1, m.astype(np.uint8)))
        run += 1
    return gt_sem, gt_ins, inst_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pred", choices=["pred", "gtmask"])
    ap.add_argument("--conf", default="one", choices=["one", "rank", "score"])
    ap.add_argument("--ov-dir", required=True,
                    help="directory of written per-scene predictions to score")
    ap.add_argument("--scenes", default="", help="comma-separated scene ids; empty = all")
    args = ap.parse_args()
    ovdir = args.ov_dir
    scan_eval = ScanNetEval(class_labels=NAMES, dataset_name="scannetpp_benchmark")
    scenes = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{ovdir}/*.pth")
                    if os.path.exists(f"{GT_GROUND}/{os.path.basename(f)}"))
    if args.scenes:
        keep = set(args.scenes.split(","))
        scenes = [s for s in scenes if s in keep]
    gtsem, gtins, res = [], [], []
    for sc in scenes:
        gs, gi, inst_list = build_gt_auth(sc)
        d = torch.load(f"{ovdir}/{sc}.pth", map_location="cpu", weights_only=False)
        npts = int(d['ins'][0]['length'])
        if len(gs) != npts:
            print(f"  LEN MISMATCH {sc}: gt={len(gs)} pred={npts} -> skip", flush=True); continue
        gtsem.append(gs); gtins.append(gi)
        tmp = []
        if args.mode == "gtmask":
            for cid, m in inst_list:
                tmp.append({"scan_id": sc, "label_id": int(cid), "conf": 1.0, "pred_mask": m})
        else:
            cls = np.asarray(d['class']); n = len(d['ins'])
            confs = d.get('conf', None)
            for k in range(n):
                m = rle_decode(d['ins'][k])
                if m.sum() < 1: continue
                if args.conf == "score" and confs is not None: cf = float(confs[k])
                elif args.conf == "rank": cf = float(n - k) / n
                else: cf = 1.0
                # Predictions are indices into the 84 instance names. In 100-class mode the
                # evaluator's label ids index the 100-name list, where those 84 sit at 16..99,
                # so the id is (idx84 + 16) + 1.
                tmp.append({"scan_id": sc, "label_id": int(cls[k]) + _PSHIFT + 1,
                            "conf": cf, "pred_mask": m})
        res.append(tmp)
    print(f"mode={args.mode} conf={args.conf} scenes={len(res)}", flush=True)
    scan_eval.evaluate(res, gtsem, gtins)

if __name__ == "__main__":
    main()
