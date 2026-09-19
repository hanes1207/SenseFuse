"""Uni3D shape embeddings for Open-YOLO3D's ScanNet200 proposals -> ``features/oyuni_s{k}.npz``.

One embedding per proposal that carries at least ten vertices, over the 312 validation scenes. The
proposals index the axis-aligned mesh, so the mask is applied to that mesh and never to the raw one.
Shards are independent and may be run on separate devices; the shard files are concatenated by the
loader through the ``(scene, col)`` key, so the split need not be the one used here.

    python scripts/extract/oy_uni3d.py --dump data/features/oyuni_s0.npz --shard 0 --num_shards 2
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from _lib import ply, uni3d  # noqa: F401  (the package import also sets sys.path)
from utils.paths import OPENYOLO3D

OYDIR = f"{OPENYOLO3D}/output/scannet200/scannet200_masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    scenes = sorted(f[:-3] for f in os.listdir(OYDIR) if f.endswith(".pt"))[a.shard::a.num_shards]
    enc = uni3d.build_uni3d(a.device)
    SC, COL, ZU = [], [], []
    for si, s in enumerate(scenes):
        t0 = time.time()
        m, _sc = torch.load(f"{OYDIR}/{s}.pt", map_location="cpu")
        m = m.numpy() > 0.5
        d = ply.read_preprocessed(ply.preprocessed_path(s), want_xyz=True)
        assert len(d["xyz"]) == m.shape[0], f"{s} vertex mismatch"
        feats, cols = [], []
        for c in range(m.shape[1]):
            idx = np.nonzero(m[:, c])[0]
            if len(idx) < 10:
                continue
            feats.append(uni3d.prep(d["xyz"][idx], d["rgb"][idx], 10000))
            cols.append(c)
        for b in range(0, len(feats), a.batch):
            pc = torch.from_numpy(np.stack(feats[b:b + a.batch])).to(a.device)
            with torch.no_grad():
                ZU.append(F.normalize(enc(pc[:, :, :3].contiguous(),
                                          pc[:, :, 3:].contiguous()), dim=-1).cpu().numpy())
        SC.extend([s] * len(cols))
        COL.extend(cols)
        print(f"  [s{a.shard}] {si+1}/{len(scenes)} {s} {len(cols)} props "
              f"{time.time()-t0:.0f}s cum {len(SC)}", flush=True)
    np.savez(a.dump, scene=np.array(SC), col=np.array(COL, np.int64),
             zuni=np.concatenate(ZU).astype(np.float32) if ZU else np.zeros((0, 1024), np.float32))
    print(f"dumped {a.dump}: {len(SC)}", flush=True)


if __name__ == "__main__":
    main()
