"""All paths used by this repository, declared in a single module.

The repository carries code together with the few megabytes of text anchors and ground-truth
matchings in ``assets/``. Every other input -- the per-mask embeddings, the baselines' released
masks, and the ScanNet++ artefacts -- is downloaded once and unpacked into ``data/``, whose sources
are listed in ``data/DATASETS.md``. Each location is also settable through an environment variable,
so an existing copy may be used in place rather than relocated.
"""
import os
import tempfile

__all__ = [
    "REPO", "ASSETS", "FUSION", "DATA", "FEATS", "OPENYOLO3D", "OPENMASK3D", "OM_REPRO",
    "O3D_DL", "O3D_ROOT", "OPEN3DIS", "SCANNETPP_EVAL", "SCANNETPP_DUMPS", "UNI3D",
    "SCANNET200", "SCANNET_V2", "TMPDIR",
    "env_path", "asset", "fusion", "dump",
]


def env_path(var, default):
    return os.environ.get(var, default)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
FUSION = os.path.join(ASSETS, "fusion")     # text anchors and GT <-> proposal matchings
DATA = env_path("SENSEFUSE_DATA", os.path.join(REPO, "data"))

# ---- downloaded inputs (see data/DATASETS.md) ---------------------------------------------
FEATS = env_path("SENSEFUSE_FEATS", os.path.join(DATA, "features"))
OPENYOLO3D = env_path("OPENYOLO3D_ROOT", os.path.join(DATA, "openyolo3d"))
OPENMASK3D = env_path("OPENMASK3D_ROOT", os.path.join(DATA, "openmask3d"))
OM_REPRO = env_path("SENSEFUSE_OM_REPRO", os.path.join(DATA, "openmask3d_features"))
O3D_DL = env_path("SENSEFUSE_O3D_DL", os.path.join(DATA, "open3dis_released"))
O3D_ROOT = env_path("SENSEFUSE_O3D", os.path.join(DATA, "scannetpp_open3dis_3d"))
OPEN3DIS = env_path("OPEN3DIS_ROOT", os.path.join(DATA, "open3dis"))
SCANNETPP_EVAL = env_path("SENSEFUSE_SCANNETPP_EVAL", os.path.join(DATA, "scannetpp"))
SCANNETPP_DUMPS = env_path("SENSEFUSE_SCANNETPP_DUMPS", os.path.join(DATA, "scannetpp_dumps"))
UNI3D = env_path("UNI3D_ROOT", os.path.join(DATA, "uni3d"))

# ---- raw scans, read only by scripts/extract/ when rebuilding data/features ----------------
SCANNET200 = env_path("SCANNET200_ROOT", os.path.join(DATA, "scannet200"))
SCANNET_V2 = env_path("SCANNET_V2_ROOT", os.path.join(DATA, "scannet_v2"))

# The official evaluators require a report file to be written; its contents are never read back.
TMPDIR = env_path("SENSEFUSE_TMPDIR", tempfile.gettempdir())


def asset(*parts):
    return os.path.join(ASSETS, *parts)


def fusion(*parts):
    return os.path.join(FUSION, *parts)


def dump(*parts):
    return os.path.join(SCANNETPP_DUMPS, *parts)
