"""Import OpenShape without dgl: the package only uses dgl.geometry.farthest_point_sampler,
which we provide as a pure-torch shim (registered in sys.modules before the import)."""
import sys
import types

import torch


def farthest_point_sampler(pos, npoints, start_idx=None):
    B, N, _ = pos.shape
    device = pos.device
    idx = torch.zeros(B, npoints, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    far = (torch.randint(0, N, (B,), device=device) if start_idx is None
           else torch.full((B,), start_idx, dtype=torch.long, device=device))
    for i in range(npoints):
        idx[:, i] = far
        centroid = pos[torch.arange(B, device=device), far][:, None, :]
        dist = torch.minimum(dist, ((pos - centroid) ** 2).sum(-1))
        far = dist.argmax(1)
    return idx


_geom = types.ModuleType("dgl.geometry")
_geom.farthest_point_sampler = farthest_point_sampler
_dgl = types.ModuleType("dgl")
_dgl.geometry = _geom
sys.modules.setdefault("dgl", _dgl)
sys.modules.setdefault("dgl.geometry", _geom)

import openshape  # noqa: E402,F401
