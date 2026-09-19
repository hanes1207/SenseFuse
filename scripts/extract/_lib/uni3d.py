"""Frozen Uni3D-giant as the shape encoder, and the preprocessing its checkpoint expects.

Uni3D's only custom-CUDA dependency is furthest-point sampling, supplied by ``pointnet2_ops``. That
extension is not required here: the module is stubbed at import time and ``fps`` is replaced by a
batched pure-torch implementation, so the giant checkpoint loads and runs without a compiler
toolchain. The point encoder is loaded with ``strict=False`` and the result asserted to have no
missing and no unexpected key, which is what establishes that the substitution changed nothing.

``prep`` reproduces Uni3D's own zero-shot preprocessing: the ScanNet ``z``-up convention is mapped to
the ``y``-up convention of the Objaverse pretraining, the mask is centred and scaled to the unit
sphere, and colours are divided by 255. Uni3D never saw ScanNet or ScanNet++, so the shape channel is
open-vocabulary in the same sense as the image channel.
"""
import os
import sys
import types

import numpy as np
import torch

from utils.paths import UNI3D

CKPT = f"{UNI3D}/models/model.pt"


def build_uni3d(dev):
    """Load the frozen Uni3D-giant point encoder onto ``dev``."""
    sys.path.insert(0, UNI3D)                    # the Uni3D checkout supplies its own dependencies
    stub = types.ModuleType("pointnet2_ops")
    stub.pointnet2_utils = types.SimpleNamespace(furthest_point_sample=None, gather_operation=None)
    sys.modules["pointnet2_ops"] = stub
    # Uni3D's point encoder is ``models/point_encoder.py`` of its checkout, and this repository has
    # a ``models`` package of its own. The Uni3D directory carries no ``__init__.py``, so it would
    # lose the name to this repository's regular package whatever the order of ``sys.path``; the
    # module is therefore loaded from its file rather than by name.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "uni3d_point_encoder", os.path.join(UNI3D, "models", "point_encoder.py"))
    PE = importlib.util.module_from_spec(spec)
    sys.modules["uni3d_point_encoder"] = PE
    spec.loader.exec_module(PE)

    def fps_torch(data, number):                 # (B,N,3) -> (B,number,3), batched, on device
        B, N, _ = data.shape
        d = data
        idx = torch.zeros(B, number, dtype=torch.long, device=d.device)
        dist = torch.full((B, N), 1e10, device=d.device)
        far = torch.zeros(B, dtype=torch.long, device=d.device)
        ar = torch.arange(B, device=d.device)
        for i in range(number):
            idx[:, i] = far
            cent = d[ar, far].unsqueeze(1)
            dist = torch.minimum(dist, ((d - cent) ** 2).sum(-1))
            far = dist.argmax(1)
        return torch.gather(d, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    PE.fps = fps_torch

    import timm
    from types import SimpleNamespace
    args = SimpleNamespace(pc_feat_dim=1408, embed_dim=1024, group_size=64, num_group=512,
                           pc_encoder_dim=512, patch_dropout=0.0)
    pt = timm.create_model("eva_giant_patch14_560", pretrained=False, drop_path_rate=0.0)
    enc = PE.PointcloudEncoder(pt, args)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["module"] if "module" in ck else ck
    sd2 = {k[len("point_encoder."):]: v for k, v in sd.items() if k.startswith("point_encoder.")}
    miss, unexp = enc.load_state_dict(sd2, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, f"load mismatch {len(miss)}/{len(unexp)}"
    return enc.to(dev).eval()


def prep(xyz, rgb, n=10000, rng=None):
    """One mask as the ``(n, 6)`` array the encoder takes, in Uni3D's own convention."""
    rng = rng or np.random.default_rng(0)
    m = len(xyz)
    idx = rng.choice(m, n, replace=(m < n)) if m != n else np.arange(n)
    x = xyz[idx].astype(np.float32).copy()
    c = rgb[idx].astype(np.float32) / 255.0
    x[:, [1, 2]] = x[:, [2, 1]]                      # z-up (ScanNet) -> y-up (Objaverse)
    x = x - x.mean(0)
    mx = np.sqrt((x ** 2).sum(1)).max()
    x = x / mx if mx > 1e-6 else x
    return np.concatenate([x, c], 1).astype(np.float32)
