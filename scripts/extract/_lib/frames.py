"""Posed RGB frames of a ScanNet scene, and the projection they are used through.

``load_frames`` reads the raw ``.sens`` export directly rather than any cached frame list, so that
it is unaffected by a cleared cache and applies unchanged to the train scenes. ScanNet marks frames
whose tracking was lost with an infinite pose; those are dropped here, which is why the returned
``fids`` is not simply every file in the colour directory.

``project`` follows the ScanNet convention: poses are camera-to-world, the camera looks down ``+z``,
and a point in front of the camera has positive depth.
"""
import glob
import os

import numpy as np
from PIL import Image

from utils.paths import SCANNET_V2

SCANS = os.path.join(SCANNET_V2, "scans")


def load_frames(scene_id, frame_skip=20):
    """Read posed RGB frames of one scene, subsampled by ``frame_skip``.

    Returns ``None`` when the scene has no colour export, and otherwise a dictionary with the
    camera-to-world poses, the 4x4 colour intrinsics, the colour resolution, the retained frame
    ids and a map from frame id to image path.
    """
    sd = os.path.join(SCANS, scene_id)
    cdir = os.path.join(sd, "data_compressed", "color")
    pdir = os.path.join(sd, "data", "pose")
    kpath = os.path.join(sd, "data", "intrinsic", "intrinsic_color.txt")
    if not (os.path.isdir(cdir) and os.path.isfile(kpath)):
        return None
    fids = sorted(int(os.path.splitext(os.path.basename(p))[0])
                  for p in glob.glob(os.path.join(cdir, "*.jpg")))[::frame_skip]
    poses, keep, colors = [], [], {}
    for fi in fids:
        pp = os.path.join(pdir, f"{fi}.txt")
        if not os.path.isfile(pp):
            continue
        P = np.loadtxt(pp)
        if P.shape != (4, 4) or not np.isfinite(P).all():   # ScanNet marks lost frames with -inf
            continue
        poses.append(P)
        keep.append(fi)
        colors[fi] = os.path.join(cdir, f"{fi}.jpg")
    if not keep:
        return None
    K = np.loadtxt(kpath)
    im = Image.open(colors[keep[0]])
    return {"poses": np.stack(poses), "intr": K, "color_size": np.array(im.size),
            "fids": keep, "colors": colors}


def project(points_world, pose_c2w, K):
    """Project world points into one frame. Returns ``(uv (N,2), depth (N,))`` in metres."""
    w2c = np.linalg.inv(pose_c2w)
    hom = np.concatenate([points_world, np.ones((len(points_world), 1), points_world.dtype)], 1)
    cam = (hom @ w2c.T)[:, :3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], 1), z
