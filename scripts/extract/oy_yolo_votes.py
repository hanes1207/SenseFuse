"""Open-YOLO3D's YOLO-World vote histograms for its own ScanNet200 and Replica masks
-> ``features/yolovotes{,_replica}/yolovotes_<scene>.npz``.

The third 2D head is not an embedding but a count. Open-YOLO3D labels a proposal by projecting it
into its most visible frames and reading off a label map painted from YOLO-World's boxes; this
writer reproduces that stack in place and records the resulting per-proposal histogram, from which
the published scoring is reproduced downstream. The histogram is 199 wide: the 198 ``text_prompts``
of ``pretrained/config_scannet200.yaml`` and one no-label column, and 49 wide on Replica.

Open-YOLO3D's own classes are used unchanged -- ``WORLD_2_CAM``, ``Network_2D``, the label-map
construction and ``get_visibility_mat`` -- with its Mask3D proposal stage stubbed out, since the
proposals are read from its released files rather than regenerated. All parameters are its config's:
``topk`` 40 frames per proposal, ``vis_depth_threshold`` 0.05 m, ``depth_scale`` 1000.

This writer needs Open-YOLO3D's own environment (YOLO-World and detectron2), not the environment of
this repository. It stubs a module named ``models`` before importing that checkout, so it must not be
imported into a process that uses this repository's ``models`` package.

    python scripts/extract/oy_yolo_votes.py --dataset scannet200 --shard 0 --num_shards 1
    python scripts/extract/oy_yolo_votes.py --dataset replica
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import torch

from _lib import pil_shim  # noqa: F401  (detectron2 0.6 against Pillow 11; also sets sys.path)
from utils.paths import FEATS, OPENYOLO3D, SCANNET_V2

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
import utils as U                                                          # noqa: E402
from utils.utils_2d import Network_2D, load_yaml                           # noqa: E402
from plyfile import PlyData                                                # noqa: E402

SV2 = f"{SCANNET_V2}/scans"

# The two datasets differ only in the configuration read, the masks scored and where the frames of a
# scene already sit: Replica ships as one directory per scene inside the checkout, whereas ScanNet
# has to be presented to it by linking the raw export into the layout it expects.
DATASETS = {
    "scannet200": dict(config="pretrained/config_scannet200.yaml",
                       masks=f"{OPENYOLO3D}/output/scannet200/scannet200_masks",
                       out=f"{FEATS}/yolovotes"),
    "replica": dict(config="pretrained/config_replica.yaml",
                    masks=f"{OPENYOLO3D}/output/replica/replica_masks",
                    out=f"{FEATS}/yolovotes_replica"),
}


def load_ply(path):
    """Read the pose-frame mesh. The preprocessed validation meshes are axis-aligned while the
    poses are not, so the raw mesh is required; the two share a vertex order."""
    v = PlyData.read(path)["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    coords = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], -1)
    return coords, np.zeros_like(pts)


U.WORLD_2_CAM.load_ply = staticmethod(load_ply)


def scene_dir(s, dataset):
    """Present one scene to the checkout under the directory layout it expects."""
    if dataset == "replica":
        d = f"{OPENYOLO3D}/replica/{s}"
        p = f"{d}/{s}.ply"
        if not os.path.islink(p) and not os.path.exists(p):
            os.symlink(f"{d}/{s}_mesh.ply", p)
        return d
    d = f"{OPENYOLO3D}/data/scannet200/{s}"
    os.makedirs(d, exist_ok=True)
    for src, dst in ((f"{SV2}/{s}/data/pose", "poses"),
                     (f"{SV2}/{s}/data_compressed/color", "color"),
                     (f"{SV2}/{s}/data_compressed/depth", "depth"),
                     (f"{SV2}/{s}/data/intrinsic/intrinsic_color.txt", "intrinsics.txt"),
                     (f"{SV2}/{s}/{s}_vh_clean_2.ply", f"{s}.ply")):
        p = f"{d}/{dst}"
        if not os.path.islink(p) and not os.path.exists(p):
            os.symlink(src, p)
    return d


def projections(w2c):
    """Per-frame projection in float32, with the semantics of ``WORLD_2_CAM.get_mesh_projections``.

    The colour intrinsic is adjusted to depth resolution, and a point counts as inside when it falls
    within the image and its camera depth agrees with the sensor depth to within the configured
    threshold.
    """
    import imageio.v2 as imageio
    pts4, _ = U.WORLD_2_CAM.load_ply(w2c.mesh)
    pts = pts4[:, :3].astype(np.float32)
    K = w2c.adjust_intrinsic(np.loadtxt(w2c.intrinsics[0]),
                             w2c.image_resolution, w2c.depth_resolution)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    nf, N = len(w2c.poses), len(pts)
    P = torch.from_numpy(pts).cuda()
    proj = torch.zeros((nf, N, 2), dtype=torch.int16)
    inside = torch.zeros((nf, N), dtype=torch.bool)
    H, W = w2c.height, w2c.width
    for f in range(nf):
        pose = np.loadtxt(w2c.poses[f])
        if not np.isfinite(pose).all():
            continue
        m = np.linalg.inv(pose).astype(np.float32)
        R = torch.from_numpy(m[:3, :3]).cuda()
        t = torch.from_numpy(m[:3, 3]).cuda()
        Xc = P @ R.T + t
        z = Xc[:, 2]
        ok = z > 0.05
        u = fx * Xc[:, 0] / torch.clamp(z, min=1e-6) + cx
        v = fy * Xc[:, 1] / torch.clamp(z, min=1e-6) + cy
        ok &= (u > 0) & (u < W) & (v > 0) & (v < H)
        ui = u.long().clamp(0, W - 1)
        vi = v.long().clamp(0, H - 1)
        dm = torch.from_numpy(imageio.imread(w2c.depth_maps_paths[f]).astype(np.float32)
                              / w2c.depth_scale).cuda()
        ok &= torch.abs(dm[vi, ui] - z) <= w2c.vis_depth_threshold
        proj[f, :, 0] = ui.to(torch.int16).cpu()
        proj[f, :, 1] = vi.to(torch.int16).cpu()
        inside[f] = ok.cpu()
    return proj, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet200", choices=sorted(DATASETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes", default="", help="file of scene ids; empty = every released mask")
    a = ap.parse_args()
    ds = DATASETS[a.dataset]
    os.makedirs(ds["out"], exist_ok=True)

    cfg = load_yaml(ds["config"])
    NC = len(cfg["network2d"]["text_prompts"]) + 1          # prompts + one no-label column
    topk_frames = cfg["openyolo3d"]["topk"]                 # 40
    net2d = Network_2D(cfg)
    if a.scenes:
        scenes = [l.strip() for l in open(a.scenes) if l.strip()][a.shard::a.num_shards]
    else:
        scenes = sorted(f[:-3] for f in os.listdir(ds["masks"])
                        if f.endswith(".pt"))[a.shard::a.num_shards]
    if a.limit:
        scenes = scenes[:a.limit]

    for si, s in enumerate(scenes):
        out = f"{ds['out']}/yolovotes_{s}.npz"
        if os.path.exists(out):
            print(f"  [s{a.shard}] {si+1}/{len(scenes)} {s} skipped (exists)", flush=True)
            continue
        t0 = time.time()
        d = scene_dir(s, a.dataset)
        w2c = U.WORLD_2_CAM(d, cfg["openyolo3d"]["depth_scale"], cfg)
        proj, inside = projections(w2c)                      # (F,N,2) int16, (F,N) bool
        masks, _ = torch.load(f"{ds['masks']}/{s}.pt", map_location="cpu")
        masks = masks > 0.5                                  # (N, Nm)
        t1 = time.time()
        preds2d = net2d.get_bounding_boxes(w2c.color_paths)
        t2 = time.time()

        # label maps, with the semantics of the checkout's construct_label_maps
        sp = [w2c.depth_resolution[0] / w2c.image_resolution[0],
              w2c.depth_resolution[1] / w2c.image_resolution[1]]
        nf = len(preds2d)
        label_maps = (torch.ones((nf, w2c.height, w2c.width)) * -1).type(torch.int16)
        for frame_id, pred in enumerate(preds2d.values()):
            bboxes = pred["bbox"].long().clone()
            labels = pred["labels"].type(torch.int16)
            bboxes[:, 0] = (bboxes[:, 0].float() * sp[1]).long()
            bboxes[:, 2] = (bboxes[:, 2].float() * sp[1]).long()
            bboxes[:, 1] = (bboxes[:, 1].float() * sp[0]).long()
            bboxes[:, 3] = (bboxes[:, 3].float() * sp[0]).long()
            extent = (bboxes[:, 2] - bboxes[:, 0]) + (bboxes[:, 3] - bboxes[:, 1])
            for j in extent.sort(descending=True).indices:   # larger boxes are painted over
                b = bboxes[j]
                label_maps[frame_id, b[1]:b[3], b[0]:b[2]] = labels[j]

        vis = U.get_visibility_mat(masks.permute(1, 0).cuda(), inside.cuda(), topk=topk_frames)
        vis = vis.cpu().numpy()
        inside_np, proj_np = inside.numpy(), proj.numpy()
        lm = label_maps.numpy()
        masks_np = masks.permute(1, 0).numpy()               # (Nm, N)
        votes = np.zeros((masks_np.shape[0], NC), np.int64)
        for mi in range(masks_np.shape[0]):
            for f in np.nonzero(vis[mi])[0]:
                vpm = inside_np[f] & masks_np[mi]
                if vpm.sum() == 0:
                    continue
                xy = proj_np[f][vpm]
                lab = lm[f, xy[:, 1], xy[:, 0]]
                lab = lab[lab != -1]
                if len(lab):
                    votes[mi] += np.bincount(lab, minlength=NC)
        np.savez(out, votes=votes, nmask=masks_np.shape[0])
        del proj, inside, label_maps, lm, proj_np, inside_np
        torch.cuda.empty_cache()
        print(f"  [s{a.shard}] {si+1}/{len(scenes)} {s} Nm {masks_np.shape[0]} F {nf} "
              f"proj {t1-t0:.0f}s yolo {t2-t1:.0f}s aggregate {time.time()-t2:.0f}s", flush=True)
    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
