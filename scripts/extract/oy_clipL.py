"""CLIP ViT-L-14-336 crop embeddings for Open-YOLO3D's ScanNet200 proposals
-> ``features/s2doy_s{k}.npz``.

The second 2D head. It shares every step with ``oy_eva.py`` except the encoder, so a difference
between the two rows of Table I is a difference between the image towers and not between two
pipelines.

    python scripts/extract/oy_clipL.py --dump data/features/s2doy_s0.npz --shard 0 --num_shards 4
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from _lib import ply  # noqa: F401  (the package import also sets sys.path)
from _lib.crops import build_clipL, mask_crops
from _lib.frames import load_frames
from utils.paths import OPENYOLO3D

OYDIR = f"{OPENYOLO3D}/output/scannet200/scannet200_masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = a.device

    clipL, preprocess = build_clipL(dev)
    scenes = sorted(f[:-3] for f in os.listdir(OYDIR) if f.endswith(".pt"))[a.shard::a.num_shards]
    if a.limit:
        scenes = scenes[:a.limit]
    SC, COL, ZC, HC = [], [], [], []
    for si, s in enumerate(scenes):
        t0 = time.time()
        masks, _ = torch.load(f"{OYDIR}/{s}.pt", map_location="cpu")
        masks = masks.numpy() > 0.5
        frames = load_frames(s)
        raw_pts = None
        if frames is not None:
            try:
                raw_pts = ply.read_raw(s)
            except Exception:
                raw_pts = None
        cols = [c for c in range(masks.shape[1]) if masks[:, c].sum() >= 10]
        img_cache, depth_cache = {}, {}
        crop_lists = []
        for c in cols:
            idx = np.nonzero(masks[:, c])[0]
            crop_lists.append(mask_crops(raw_pts[idx], frames, preprocess, img_cache,
                                         depth_cache=depth_cache)
                              if (raw_pts is not None and len(idx)) else [])
        flat = [cr for cl in crop_lists for cr in cl]
        feats = None
        if flat:
            with torch.no_grad():
                out = []
                for b in range(0, len(flat), 32):     # the 336 px tower is memory-heavy on 12 GB
                    out.append(F.normalize(clipL.encode_image(
                        torch.stack(flat[b:b + 32]).to(dev)).float(), dim=-1))
                feats = torch.cat(out, 0)
        pp = 0
        for c, cl in zip(cols, crop_lists):
            SC.append(s)
            COL.append(c)
            if cl:
                ZC.append(F.normalize(feats[pp:pp + len(cl)].mean(0), dim=-1)
                          .cpu().numpy().astype(np.float32))
                HC.append(1)
                pp += len(cl)
            else:
                ZC.append(np.zeros(768, np.float32))
                HC.append(0)
        print(f"  [s{a.shard}] {si+1}/{len(scenes)} {s} {len(cols)}masks {time.time()-t0:.1f}s "
              f"cum {len(SC)} ({sum(HC)} with a view)", flush=True)
    np.savez(a.dump, scene=np.array(SC), col=np.array(COL, np.int64),
             zclipL=np.stack(ZC) if ZC else np.zeros((0, 768), np.float32),
             hasclip=np.array(HC, np.int8))
    print(f"dumped {a.dump}: {len(SC)} masks", flush=True)


if __name__ == "__main__":
    main()
