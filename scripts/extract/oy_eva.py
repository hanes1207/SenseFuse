"""EVA02-E-14-plus crop embeddings for Open-YOLO3D's ScanNet200 proposals
-> ``features/evaoy_s{k}.npz``.

EVA02-E is the image tower Uni3D's shape embeddings are aligned to, so this is the 2D head against
which the shape head is compared in the same space. The weights are the ones the Uni3D release
ships, and the crop selection is the shared one in ``_lib/crops.py``, so this writer and
``oy_clipL.py`` differ only in the encoder.

The mask is projected using the raw mesh, whose frame the camera poses are expressed in, while the
proposal itself indexes the axis-aligned mesh; the two share a vertex order. A proposal that no
frame sees is written as a zero row with ``hasclip = 0``, and the loader sets that row's 2D scores
to -1 rather than letting a zero vector score.

    python scripts/extract/oy_eva.py --dump data/features/evaoy_s0.npz --shard 0 --num_shards 4
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from _lib import ply  # noqa: F401  (the package import also sets sys.path)
from _lib.crops import mask_crops
from _lib.frames import load_frames
from utils.paths import OPENYOLO3D, UNI3D

os.environ.setdefault("HF_HUB_OFFLINE", "1")
EVA_PRE = f"{UNI3D}/models/open_clip_pytorch_model.bin"
OYDIR = f"{OPENYOLO3D}/output/scannet200/scannet200_masks"


def main():
    import open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    dev = a.device

    me, _, ppe = open_clip.create_model_and_transforms("EVA02-E-14-plus", pretrained=EVA_PRE)
    me = me.half().to(dev).eval()
    print(f"[s{a.shard}] EVA02-E loaded", flush=True)

    scenes = sorted(f[:-3] for f in os.listdir(OYDIR) if f.endswith(".pt"))[a.shard::a.num_shards]
    if a.limit:
        scenes = scenes[:a.limit]
    SC, COL, ZE, HC = [], [], [], []
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
            crop_lists.append(mask_crops(raw_pts[idx], frames, ppe, img_cache,
                                         depth_cache=depth_cache)
                              if (raw_pts is not None and len(idx)) else [])
        flat = [cr for cl in crop_lists for cr in cl]
        feats = None
        if flat:
            with torch.no_grad():
                out = []
                for b in range(0, len(flat), a.bs):
                    out.append(F.normalize(me.encode_image(
                        torch.stack(flat[b:b + a.bs]).half().to(dev)).float(), dim=-1))
                feats = torch.cat(out, 0)
        pp = 0
        for c, cl in zip(cols, crop_lists):
            SC.append(s)
            COL.append(c)
            if cl:
                ZE.append(F.normalize(feats[pp:pp + len(cl)].mean(0), dim=-1)
                          .cpu().numpy().astype(np.float32))
                HC.append(1)
                pp += len(cl)
            else:
                ZE.append(np.zeros(1024, np.float32))
                HC.append(0)
        print(f"  [s{a.shard}] {si+1}/{len(scenes)} {s} {len(cols)}masks {len(flat)}crops "
              f"{time.time()-t0:.1f}s cum {len(SC)}", flush=True)
    np.savez(a.dump, scene=np.array(SC), col=np.array(COL, np.int64),
             zeva=np.stack(ZE) if ZE else np.zeros((0, 1024), np.float32),
             hasclip=np.array(HC, np.int8))
    print(f"[s{a.shard}] dumped {a.dump}: {len(SC)} masks", flush=True)


if __name__ == "__main__":
    main()
