#!/usr/bin/env python3
"""Rebuild ``assets/fusion/o3drel_gtmatch.npz`` from source and compare it with the shipped copy.

That cache carries the ground-truth class of every proposal of Open3DIS's released ScanNet200
result, in the order ``data/features/o3drel_uni`` indexes them. Its original generator was lost,
leaving the cache as an artefact from which the matching rule behind it could not be re-derived.
This module restores that generator.

The matching rule is identical to ``utils.scannet200.gt_match``: ScanNet200's own ground-truth file,
with instance ids decoded as ``inst_id // 1000`` to a semantic id and then to a benchmark index in
0..199; a proposal is matched when its IoU against a ground-truth instance is at least 0.5 and is
assigned -1 otherwise; and ground-truth instances of fewer than ``utils.scannet200.MIN_GT_PTS``
points are excluded, as the AP evaluator excludes them at ``scannetv2_inst_eval.py`` line 88. The
proposals are the released Open3DIS hierarchical-agglomerative masks, deduplicated to the ``uidx``
rows indexed by ``data/features/o3drel_uni/<scene>.npz``.

The shipped cache applies that size floor. An earlier revision of this module omitted it, on the
incorrect reading that the evaluator's ground-truth filter was the commented-out statement at line
407; the filter is in fact the live statement at line 88. With the floor restored, 28 of 48,846
proposals are assigned to a different ground-truth instance.

  python -m utils.gtmatch           # rebuild and compare, writing nothing
  python -m utils.gtmatch --write   # additionally write the .npz
"""
import os, sys
from utils.paths import *  # noqa: F403
import argparse
import os
import sys

import numpy as np
import torch


from utils import scannet200_ap as AP
import scannet200_constants as C200
# the size floor is declared in a single place, so this module cannot diverge from the accuracy axis
from utils import scannet200 as SN

UNID = f"{FEATS}/o3drel_uni"
SRC = f"{O3D_DL}/Result_OpenVocab_ISBNet-GSAM/final_result_hier_agglo"
OUT = f"{FUSION}/o3drel_gtmatch.npz"

IDARR = np.array(list(C200.VALID_CLASS_IDS_200))
ID2IDX = {int(v): i for i, v in enumerate(IDARR)}


def rle_decode(rle):
    """Decode a run-length encoding. Open3DIS writes 1-indexed (start, run-length) pairs."""
    n = int(rle["length"])
    c = rle["counts"]
    starts = np.asarray(c[0::2], dtype=np.int64) - 1
    runs = np.asarray(c[1::2], dtype=np.int64)
    m = np.zeros(n, bool)
    for s, r in zip(starts, starts + runs):
        m[s:r] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    scenes = sorted(f[:-4] for f in os.listdir(UNID) if f.endswith(".npz"))
    ycls, offs, cur = [], [], 0
    for si, s in enumerate(scenes):
        if si % 50 == 0:
            print(f"  {si}/{len(scenes)}", flush=True)
        uidx = np.load(f"{UNID}/{s}.npz")["uidx"].astype(np.int64)
        ins = torch.load(f"{SRC}/{s}.pth", map_location="cpu", weights_only=False)["ins"]
        P = np.stack([rle_decode(ins[int(i)]) for i in uidx], 1)          # (V, R)

        gt = np.loadtxt(f"{AP.GTDIR}/{s}.txt", dtype=np.int64)
        y = np.full(len(uidx), -1, np.int64)
        if len(gt) == P.shape[0]:
            iids = np.unique(gt)
            iids = iids[iids >= 1000]
            keep = [(int(i), ID2IDX[int(i) // 1000]) for i in iids
                    if int(i) // 1000 in ID2IDX and int((gt == i).sum()) >= SN.MIN_GT_PTS]
            if keep:
                G = np.stack([gt == i for i, _ in keep], 1).astype(np.float32)   # (V, I)
                gcls = np.array([c for _, c in keep], np.int64)
                Pf = P.astype(np.float32)
                inter = Pf.T @ G
                union = Pf.sum(0)[:, None] + G.sum(0)[None, :] - inter
                iou = inter / np.maximum(union, 1e-9)
                j = iou.argmax(1)
                hit = iou[np.arange(len(j)), j] >= 0.5
                y[hit] = gcls[j[hit]]
        else:
            print(f"  !! {s}: GT {len(gt)} vs mask {P.shape[0]} vertices -- left unmatched")
        ycls.append(y)
        offs.append((cur, cur + len(y)))
        cur += len(y)

    ycls = np.concatenate(ycls)
    offs = np.asarray(offs, np.int64)
    print(f"\nrebuilt: {len(ycls)} proposals, {(ycls >= 0).sum()} matched")

    if os.path.exists(OUT):
        old = np.load(OUT, allow_pickle=True)
        oy = old["ycls"].astype(np.int64)
        same_y = len(oy) == len(ycls) and bool((oy == ycls).all())
        same_o = bool((old["offs"].astype(np.int64) == offs).all())
        print(f"shipped cache: {len(oy)} proposals, {(oy >= 0).sum()} matched")
        print(f"ycls identical: {same_y}   offs identical: {same_o}")
        if not same_y:
            d = np.where(oy != ycls)[0]
            print(f"  {len(d)} differing rows, first few: {d[:10].tolist()}")
            if not a.write:
                return 1
    if a.write:
        np.savez(OUT, ycls=ycls, offs=offs)
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
