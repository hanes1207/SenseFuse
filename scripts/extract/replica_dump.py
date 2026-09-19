"""Replica embeddings and text anchors -> ``features/replica/``.

Replica is the cross-dataset setting: the same two encoders, the same crop selection and the same
fusion, applied to a different sensor and a 48-class vocabulary. Open-YOLO3D's released Replica
masks are used, so nothing here is a proposal of ours.

The writer runs in two stages because the two image towers do not fit beside one another:

    python scripts/extract/replica_dump.py --stage clipL   # replica_dump.npz, clipL_text_48.npy
    python scripts/extract/replica_dump.py --stage eva     # replica_eva_crops.npz, eva_text_48.npy

``clipL`` also writes the Uni3D shape embeddings, which the ``eva`` stage does not repeat. The crop
selection is Replica's own rather than the ScanNet one in ``_lib/crops.py``: Replica poses are given
per frame in the mask's own frame, its depth carries a different scale, and the visibility tolerance
is 0.4 m against ScanNet's 0.2 m. Both stages keep every proposal, with no minimum point count, so
the dump has one row per released mask.

Text anchors are the twenty-template ensemble, built for whichever tower the stage loads, so the two
2D heads differ in their encoder and in nothing else. Class names are the evaluator's own, with
hyphens replaced by spaces.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _lib import ply  # noqa: F401  (the package import also sets sys.path)
from utils.paths import FEATS, OPENYOLO3D, UNI3D

from evaluate import SCENE_NAMES_REPLICA                                    # noqa: E402
from evaluate.replica.eval_semantic_instance import CLASS_LABELS as RNAMES  # noqa: E402

OUT = f"{FEATS}/replica"
OY = OPENYOLO3D
DEPTH_SCALE = 6553.5
VIS_TH = 0.4
NAMES = [n.replace("-", " ") for n in RNAMES]

TEMPLATES = ["{}", "a {}", "the {}", "a {} in a scene", "there is a {} in the room",
             "a photo of a {}", "a photo of the {}", "a photo of one {}",
             "a close-up photo of a {}", "a cropped photo of a {}", "a bright photo of a {}",
             "a dark photo of a {}", "a blurry photo of a {}", "a photo of a small {}",
             "a photo of a large {}", "a {} in a living room", "a {} in a bedroom",
             "a {} in an office", "an indoor photo of a {}", "a 3d render of a {}"]


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def text_anchors(model, tokenizer, dev):
    """The twenty-template ensemble over the 48 class names, in one tower's space."""
    with torch.no_grad():
        per = [F.normalize(model.encode_text(
            tokenizer([t.format(n) for n in NAMES]).to(dev)).float(), dim=-1) for t in TEMPLATES]
        return F.normalize(torch.stack(per).mean(0), dim=-1).cpu().numpy()


def scene_crops(s, preprocess):
    """Crop selection for one Replica scene: per proposal, two paddings of its five best views.

    Returns ``(xyz, rgb, masks, crop_lists)``. A view counts only when at least thirty of the
    proposal's points are visible in it and their sensor depth agrees to within ``VIS_TH``.
    """
    import imageio.v2 as imageio
    from plyfile import PlyData
    base = f"{OY}/replica/{s}"
    v = PlyData.read(f"{base}/{s}_mesh.ply")["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
    m, _ = torch.load(f"{OY}/output/replica/replica_masks/{s}.pt", map_location="cpu")
    m = m.numpy() > 0.5
    K = np.loadtxt(f"{base}/intrinsics.txt")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fids = sorted(int(f[:-4]) for f in os.listdir(f"{base}/color") if f.endswith(".jpg"))[::2]
    w2c = {f: np.linalg.inv(np.loadtxt(f"{base}/poses/{f}.txt")) for f in fids}
    depths = {f: imageio.imread(f"{base}/depth/{f}.png").astype(np.float32) / DEPTH_SCALE
              for f in fids}
    H, W = depths[fids[0]].shape

    crop_lists = []
    for c in range(m.shape[1]):
        idx = np.nonzero(m[:, c])[0]
        pts = xyz[idx if len(idx) <= 3000 else np.random.choice(idx, 3000, replace=False)]
        best = []
        for f in fids:
            R, t = w2c[f][:3, :3], w2c[f][:3, 3]
            Xc = pts @ R.T + t
            z = Xc[:, 2]
            ok = z > 0.1
            u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx
            v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if ok.sum() < 30:
                continue
            ui, vi = u[ok].astype(int), v[ok].astype(int)
            dvis = np.abs(depths[f][vi, ui] - z[ok]) <= VIS_TH
            if dvis.sum() < 30:
                continue
            best.append((int(dvis.sum()), f, ui[dvis], vi[dvis]))
        best.sort(key=lambda x: -x[0])
        crops = []
        for _, f, ui, vi in best[:5]:
            img = Image.open(f"{base}/color/{f}.jpg").convert("RGB")
            x0, x1, y0, y1 = ui.min(), ui.max(), vi.min(), vi.max()
            for pad in (0.1, 0.3):
                pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
                a, b = max(0, int(x0 - pw)), min(W, int(x1 + pw))
                c0, d0 = max(0, int(y0 - ph)), min(H, int(y1 + ph))
                if b - a >= 4 and d0 - c0 >= 4:
                    crops.append(preprocess(img.crop((a, c0, b, d0))))
        crop_lists.append(crops)
    return xyz, rgb, m, crop_lists


def encode_crops(model, crop_lists, dev, dim, batch, half):
    """Mean of a proposal's encoded crops, renormalised; a zero row where no view saw it."""
    flat = [c for cl in crop_lists for c in cl]
    feats = None
    if flat:
        with torch.no_grad():
            out = []
            for b in range(0, len(flat), batch):
                t = torch.stack(flat[b:b + batch])
                t = t.half().to(dev) if half else t.to(dev)
                out.append(F.normalize(model.encode_image(t).float(), dim=-1))
            feats = torch.cat(out, 0)
    Z, HV, p = [], [], 0
    for cl in crop_lists:
        if cl:
            Z.append(F.normalize(feats[p:p + len(cl)].mean(0), dim=-1).cpu().numpy())
            HV.append(True)
            p += len(cl)
        else:
            Z.append(np.zeros(dim, np.float32))
            HV.append(False)
    return Z, HV


def main():
    import open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("clipL", "eva"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    if a.stage == "clipL":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai")
        model = model.to(dev).eval()
        tok = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
        dim, batch, half = 768, 32, False
        anchors_out = f"{OUT}/clipL_text_48.npy"
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "EVA02-E-14-plus", pretrained=f"{UNI3D}/models/open_clip_pytorch_model.bin")
        model = model.half().to(dev).eval()
        tok = open_clip.get_tokenizer("EVA02-E-14-plus")
        dim, batch, half = 1024, 16, True
        anchors_out = f"{OUT}/eva_text_48.npy"

    np.save(anchors_out, text_anchors(model, tok, dev))
    print(f"wrote {anchors_out}", flush=True)

    enc3d = None
    if a.stage == "clipL":
        from _lib import uni3d
        enc3d = uni3d.build_uni3d(dev)

    SC, COL, Z2, HV, Z3 = [], [], [], [], []
    for s in SCENE_NAMES_REPLICA:
        t0 = time.time()
        xyz, rgb, m, crop_lists = scene_crops(s, preprocess)
        z, hv = encode_crops(model, crop_lists, dev, dim, batch, half)
        Z2 += z
        HV += hv
        SC += [s] * m.shape[1]
        COL += list(range(m.shape[1]))
        if enc3d is not None:
            from _lib import uni3d
            feats = [uni3d.prep(xyz[np.nonzero(m[:, c])[0]], rgb[np.nonzero(m[:, c])[0]], 10000)
                     for c in range(m.shape[1])]
            with torch.no_grad():
                for b in range(0, len(feats), 8):
                    pc = torch.from_numpy(np.stack(feats[b:b + 8])).to(dev)
                    Z3.append(F.normalize(enc3d(pc[:, :, :3].contiguous(),
                                                pc[:, :, 3:].contiguous()), dim=-1).cpu().numpy())
        print(f"  {s}: {m.shape[1]} proposals {time.time()-t0:.0f}s", flush=True)

    sc, col = np.array(SC), np.array(COL)
    z2 = l2(np.stack(Z2).astype(np.float32))
    hv = np.array(HV)
    if a.stage == "clipL":
        np.savez(f"{OUT}/replica_dump.npz", scene=sc, col=col,
                 zuni=l2(np.concatenate(Z3).astype(np.float32)), zclipL=z2, hasclip=hv)
        print(f"wrote {OUT}/replica_dump.npz: {len(sc)} proposals", flush=True)
    else:
        np.savez(f"{OUT}/replica_eva_crops.npz", scene=sc, col=col, zeva=z2, hv=hv)
        print(f"wrote {OUT}/replica_eva_crops.npz: {len(sc)} proposals", flush=True)


if __name__ == "__main__":
    main()
