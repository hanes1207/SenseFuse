"""OpenShape shape embeddings for Open-YOLO3D's ScanNet200 proposals
-> ``features/openshape_oy_s{k}.npz``.

The second shape encoder. Table III pairs it against both image towers and against Uni3D, and it is
the least correlated partner of all four, which is what makes it the counterexample to reading
complementarity off correlation alone: it also gains the least, because it is the weakest.

Preprocessing is Uni3D's, so that the two shape channels differ in the encoder and in nothing else.
OpenShape's own text tower is OpenCLIP ViT-bigG, so its class anchors are the separate
``bigg_text_200.npy``, written by ``text_anchors.py``.

The package imports ``dgl`` only for furthest-point sampling, which ``_lib.openshape_shim`` supplies
in pure torch, so no graph library is required.

    python scripts/extract/openshape.py --shard 0 --num_shards 1
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from _lib import ply, uni3d  # noqa: F401  (the package import also sets sys.path)
from _lib.openshape_shim import openshape
from utils.paths import FEATS, OPENYOLO3D

OYDIR = f"{OPENYOLO3D}/output/scannet200/scannet200_masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    model = openshape.load_pc_encoder("openshape-pointbert-vitg14-rgb").eval()

    def encode(batch):
        x = torch.from_numpy(np.stack(batch)).transpose(1, 2).cuda()      # (B, 6, N)
        with torch.no_grad():
            return F.normalize(model(x), dim=-1).cpu().numpy()

    scenes = sorted(f[:-3] for f in os.listdir(OYDIR) if f.endswith(".pt"))[a.shard::a.num_shards]
    Z, SC, COL = [], [], []
    t0 = time.time()
    for si, s in enumerate(scenes):
        m, _ = torch.load(f"{OYDIR}/{s}.pt", map_location="cpu")
        m = m.numpy() > 0.5
        d = ply.read_preprocessed(ply.preprocessed_path(s), want_xyz=True)
        feats, cols = [], []
        for c in range(m.shape[1]):
            idx = np.nonzero(m[:, c])[0]
            if len(idx) < 10:
                continue
            feats.append(uni3d.prep(d["xyz"][idx], d["rgb"][idx], 10000))
            cols.append(c)
        for b in range(0, len(feats), a.batch):
            Z.append(encode(feats[b:b + a.batch]))
        SC.extend([s] * len(cols))
        COL.extend(cols)
        if si % 25 == 0:
            print(f"  [s{a.shard}] {si+1}/{len(scenes)} cum {len(SC)} masks "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    out = f"{FEATS}/openshape_oy_s{a.shard}.npz"
    np.savez(out, scene=np.array(SC), col=np.array(COL, np.int32),
             z=np.concatenate(Z).astype(np.float32))
    print(f"dumped {out}: {len(SC)} masks", flush=True)


if __name__ == "__main__":
    main()
