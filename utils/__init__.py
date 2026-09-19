"""Loaders, scorers, and the benchmark code against which they are run.

``utils/benchmark`` carries three third-party sources unmodified: ScanNet200's official constants
and splits, OpenMask3D's instance-AP evaluator, and Open3DIS's ScanNet++ evaluator and dataset
tables. Open-YOLO3D's Replica evaluator and its two prompt configurations are not carried here: that
project publishes no licence, so this repository reads them from the checkout ``OPENYOLO3D_ROOT``
already points at, in the same way it reads that project's masks and ground truth.

All of these import one another by bare module name, so their directories must be importable.
Performing the insertion here means that no module below modifies ``sys.path``.
"""
import os
import sys

_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
from utils.paths import OPENYOLO3D  # noqa: E402  (only os and tempfile are imported below it)

for _p in (
    os.path.join(_B, "scannet200"),                      # scannet200_constants, scannet200_splits
    os.path.join(_B, "openmask3d"),                      # eval_semantic_instance, util, util_3d
    os.path.join(_B, "open3dis"),                        # open3dis.dataset.*, open3dis.evaluation.*
    os.path.join(_B, "open3dis", "open3dis", "evaluation"),   # bare scannetv2_inst_eval
    OPENYOLO3D,                                          # evaluate.replica, evaluate.scannet200
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def openyolo3d_file(*parts):
    """A path inside the Open-YOLO3D checkout, with the message to give when it is not there."""
    p = os.path.join(OPENYOLO3D, *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. This repository does not carry Open-YOLO3D's evaluator or its prompt "
            f"configurations, because that project publishes no licence; clone it and point "
            f"OPENYOLO3D_ROOT at the checkout. data/DATASETS.md says where.")
    return p
