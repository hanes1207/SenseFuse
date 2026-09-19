"""Uni3D embeddings for the unique masks of Open3DIS's released ScanNet200 predictions
-> ``features/o3drel_uni/<scene>.npz``.

Open3DIS releases roughly six hundred predictions per scene over far fewer distinct masks, so the
writer deduplicates by the MD5 of the run-length counts and encodes each distinct mask once. Each
file therefore carries ``zuni`` for the ``U`` unique masks, ``uidx`` mapping a unique mask back to a
released row, and ``urows`` mapping every released row to its unique mask; the scorer folds the
released class scores onto the embeddings through ``urows``.

The masks index the raw mesh, which is why this writer reads the pose-frame mesh rather than the
axis-aligned one. Output is per scene and resumable: an existing file is skipped, so any shard split
reproduces the same directory.

    python scripts/extract/o3d_released_uni3d.py --shard 0 --num_shards 1
"""
import argparse
import glob
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F

from _lib import uni3d  # noqa: F401  (the package import also sets sys.path)
from utils.paths import FEATS, O3D_DL, SCANNET_V2

SRC = f"{O3D_DL}/Result_OpenVocab_ISBNet-GSAM/final_result_hier_agglo"
SCANS = f"{SCANNET_V2}/scans"
OUT = f"{FEATS}/o3drel_uni"


def rle_pts(rle):
    c = np.asarray(rle["counts"])
    out = [np.arange(lo, lo + n) for lo, n in zip(c[0::2] - 1, c[1::2])]
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from plyfile import PlyData
    enc = uni3d.build_uni3d(a.device)
    rng = np.random.default_rng(0)
    scenes = sorted(os.path.basename(f)[:-4]
                    for f in glob.glob(f"{SRC}/scene*.pth"))[a.shard::a.num_shards]
    for si, s in enumerate(scenes):
        out = f"{OUT}/{s}.npz"
        if os.path.exists(out):
            continue
        d = torch.load(f"{SRC}/{s}.pth", map_location="cpu", weights_only=False)
        v = PlyData.read(f"{SCANS}/{s}/{s}_vh_clean_2.ply")["vertex"]
        V = np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)
        C = np.stack([v[k] for k in ("red", "green", "blue")], 1).astype(np.float32)
        seen, uidx, urows = {}, [], []
        for k, r in enumerate(d["ins"]):
            h = hashlib.md5(np.ascontiguousarray(np.asarray(r["counts"])).tobytes()).hexdigest()
            if h not in seen:
                seen[h] = len(uidx)
                uidx.append(k)
            urows.append(seen[h])
        feats = []
        for k in uidx:
            vi = rle_pts(d["ins"][k])
            feats.append(uni3d.prep(V[vi], C[vi], 10000, rng))
        feats = np.stack(feats)
        zs = []
        for b in range(0, len(feats), 8):
            pc = torch.from_numpy(feats[b:b + 8]).to(a.device)
            with torch.no_grad():
                zs.append(F.normalize(enc(pc[:, :, :3].contiguous(),
                                          pc[:, :, 3:].contiguous()), dim=-1).cpu().numpy())
            torch.cuda.empty_cache()
        np.savez(out, zuni=np.concatenate(zs), urows=np.array(urows), uidx=np.array(uidx))
        print(f"[s{a.shard}] {si+1}/{len(scenes)} {s}: {len(uidx)} unique masks", flush=True)
    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
