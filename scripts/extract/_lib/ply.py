"""Readers for the two point-cloud representations of a ScanNet scene.

A scene is read in one of two frames, and the writers are not free to choose between them:

``read_preprocessed`` reads ``$SCANNET200_ROOT/{split}/{scene}.ply``, the axis-aligned Mask3D-format
mesh. Its vertex order is the one the released proposals index, so a proposal mask may only be
applied to it. The packed vertex block is sliced directly rather than parsed, which is two orders of
magnitude faster than a general PLY parser on a 240k-vertex scene.

``read_raw`` reads ``$SCANNET_V2_ROOT/scans/{scene}/{scene}_vh_clean_2.ply``, the mesh in the frame
the camera poses are expressed in. Projecting a mask into an image therefore requires these
coordinates and not the axis-aligned ones. The two files share a vertex order, which was verified by
matching their per-vertex colours exactly, so a proposal mask indexes both.
"""
import os

import numpy as np

from utils.paths import SCANNET200, SCANNET_V2

_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ("label", "<u4"), ("inst", "<u4")])


def read_preprocessed(path, want_xyz=False):
    """Read a Mask3D-format ``.ply`` by slicing its packed vertex block."""
    with open(path, "rb") as f:
        raw = f.read()
    i = raw.index(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:i].decode("ascii", "ignore")
    nvert = int([l for l in hdr.splitlines() if l.startswith("element vertex")][0].split()[-1])
    v = np.frombuffer(raw, dtype=_DT, count=nvert, offset=i)
    out = {"label": v["label"].astype(np.int64), "inst": v["inst"].astype(np.int64)}
    if want_xyz:
        out["xyz"] = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        out["rgb"] = np.stack([v["r"], v["g"], v["b"]], 1).astype(np.float32)
    return out


def preprocessed_path(scene_id, split="val"):
    return os.path.join(SCANNET200, split, f"{scene_id}.ply")


def read_raw(scene_id):
    """Vertices of ``{scene}_vh_clean_2.ply``, in the frame the camera poses use."""
    from plyfile import PlyData
    p = os.path.join(SCANNET_V2, "scans", scene_id, f"{scene_id}_vh_clean_2.ply")
    v = PlyData.read(p)["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
