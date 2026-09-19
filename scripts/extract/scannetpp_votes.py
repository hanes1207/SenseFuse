"""Open-YOLO3D's YOLO-World vote labelling, carried to ScanNet++
-> ``features/yolovotes_scannetpp/yolovotes_<scene>.npz``.

The same labeller as ``oy_yolo_votes.py``, prompted with the 84 instance names of ScanNet++'s
top-100 list and run over the iPhone frames of the ScanNet++ 2D release rather than ScanNet's
export. The histogram is 85 wide: the 84 names and one no-label column. The proposals are Mask3D's,
prepared under ``$SENSEFUSE_SCANNETPP_DUMPS/mask3d_props``.

Depth is resized to the colour resolution before it is read, because ScanNet++'s two streams differ
in resolution; the visibility threshold is 0.10 m, looser than ScanNet's, for the same reason. The
projection is stored as ``int16``, which is not a micro-optimisation: at ``int64`` a single scene's
projections reach roughly 11 GB and the process is killed.

This writer needs Open-YOLO3D's own environment, not the environment of this repository.

    python scripts/extract/scannetpp_votes.py --shard 0 --num_shards 1
"""
import argparse
import glob
import os
import sys
import time
import types

import numpy as np
import torch

from _lib import pil_shim  # noqa: F401  (detectron2 0.6 against Pillow 11; also sets sys.path)
from utils.paths import FEATS, O3D_ROOT, OPENYOLO3D, SCANNETPP_DUMPS, SCANNETPP_EVAL

# ---- stub the proposal stage before importing the checkout ----
for _name in ("models", "models.Mask3D"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_mk = types.ModuleType("models.Mask3D.mask3d")
for _fn in ("load_mesh_or_pc", "get_model", "load_mesh", "prepare_data",
            "map_output_to_pointcloud", "save_colorized_mesh"):
    setattr(_mk, _fn, None)
sys.modules["models.Mask3D.mask3d"] = _mk
sys.modules.setdefault("open3d", types.ModuleType("open3d"))
os.chdir(OPENYOLO3D)
sys.path.insert(0, OPENYOLO3D)
from utils.utils_2d import Network_2D                                      # noqa: E402
import yaml, imageio.v2 as imageio, trimesh                                # noqa: E402

# The ScanNet++ 2D release (iPhone colour, depth, pose and intrinsics at a five-frame interval) is
# reached through O3D_ROOT, which is the tree Open3DIS's ScanNet++ preparation writes.
DATA = f"{O3D_ROOT}/Scannetpp/Scannetpp_2D_5interval/val"
MESHD = f"{SCANNETPP_EVAL}/data"
PROPS = f"{SCANNETPP_DUMPS}/mask3d_props"
OUT = f"{FEATS}/yolovotes_scannetpp"
INTERVAL = 2
TOPK_FRAMES = 40
VIS_TH = 0.10
DEPTH_SCALE = 1000.0


def load_classes():
    p = glob.glob(f"{SCANNETPP_EVAL}/**/top100_instance.txt", recursive=True)[0]
    return [l.strip() for l in open(p) if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = load_classes()
    cfg = yaml.safe_load(open(f"{OPENYOLO3D}/pretrained/config_scannet200.yaml"))
    cfg["network2d"]["text_prompts"] = names
    net2d = Network_2D(cfg)
    NC = len(names) + 1
    scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{PROPS}/*.npz"))[a.shard::a.num_shards]
    for si, sc in enumerate(scenes):
        out = f"{OUT}/yolovotes_{sc}.npz"
        if os.path.exists(out):
            print(f"[s{a.shard}] {si+1}/{len(scenes)} {sc} skipped", flush=True)
            continue
        t0 = time.time()
        d = np.load(f"{PROPS}/{sc}.npz")
        V = np.asarray(trimesh.load(f"{MESHD}/{sc}/scans/mesh_aligned_0.05.ply", process=False).vertices, np.float32)
        flat, ns = d["flat"], d["n"]; offs = np.concatenate([[0], np.cumsum(ns)])
        props = [flat[offs[i]:offs[i+1]] for i in range(len(ns))]
        fids = sorted(f[:-4] for f in os.listdir(f"{DATA}/{sc}/color"))[::INTERVAL]
        Pt = torch.from_numpy(V).cuda()
        # per-frame: labelmap + per-prop visible count/labels
        Fn = len(fids)
        vis_count = np.zeros((len(props), Fn), np.int32)
        frame_votes = [None] * Fn
        color_paths = [f"{DATA}/{sc}/color/{f}.jpg" for f in fids]
        preds2d = net2d.get_bounding_boxes(color_paths)
        img0 = imageio.imread(color_paths[0]); H, W = img0.shape[:2]
        lab_frames = []
        projs = []
        for fi, f in enumerate(fids):
            K = np.loadtxt(f"{DATA}/{sc}/intrinsic/{f}.txt")
            pose = np.loadtxt(f"{DATA}/{sc}/pose/{f}.txt")
            if not np.isfinite(pose).all():
                lab_frames.append(None); projs.append(None); continue
            w2c = np.linalg.inv(pose).astype(np.float32)
            R = torch.from_numpy(w2c[:3, :3]).cuda(); t = torch.from_numpy(w2c[:3, 3]).cuda()
            Xc = Pt @ R.T + t; z = Xc[:, 2]
            ok = z > 0.05
            u = (K[0, 0] * Xc[:, 0] / torch.clamp(z, min=1e-6) + K[0, 2])
            v = (K[1, 1] * Xc[:, 1] / torch.clamp(z, min=1e-6) + K[1, 2])
            ok &= (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
            ui = u.long().clamp(0, W - 1); vi = v.long().clamp(0, H - 1)
            dp = imageio.imread(f"{DATA}/{sc}/depth/{f}.png").astype(np.float32) / DEPTH_SCALE
            import cv2
            dp = cv2.resize(dp, (W, H), interpolation=cv2.INTER_NEAREST)
            dmt = torch.from_numpy(dp).cuda()
            ok &= (torch.abs(dmt[vi, ui] - z) <= VIS_TH)
            # label map
            pred = list(preds2d.values())[fi]
            lm = np.full((H, W), -1, np.int16)
            bb = pred["bbox"].long().numpy(); lb = pred["labels"].numpy().astype(np.int16)
            widths = (bb[:, 2] - bb[:, 0]) + (bb[:, 3] - bb[:, 1])
            for j in np.argsort(-widths):
                x0, y0, x1, y1 = bb[j]; lm[y0:y1, x0:x1] = lb[j]
            lab_frames.append(lm)
            uin = ui.cpu().numpy().astype(np.int16); vin = vi.cpu().numpy().astype(np.int16); okn = ok.cpu().numpy()
            projs.append((uin, vin, okn))
            for mi, vi_ in enumerate(props):
                vis_count[mi, fi] = int(okn[vi_].sum())
        votes = np.zeros((len(props), NC), np.int64)
        for mi, vi_ in enumerate(props):
            top = np.argsort(-vis_count[mi])[:TOPK_FRAMES]
            for fi in top:
                if vis_count[mi, fi] < 1 or projs[fi] is None: continue
                uin, vin, okn = projs[fi]
                sel = vi_[okn[vi_]]
                lab = lab_frames[fi][vin[sel], uin[sel]]
                lab = lab[lab != -1]
                if len(lab): votes[mi] += np.bincount(lab, minlength=NC)
        np.savez(out, votes=votes, nmask=len(props))
        print(f"[s{a.shard}] {si+1}/{len(scenes)} {sc} Nm {len(props)} F {Fn} {time.time()-t0:.0f}s", flush=True)
    print("SHARD_DONE", flush=True)

if __name__ == "__main__":
    main()
