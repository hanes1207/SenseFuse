"""ScanNet200's instance ground truth and the official AP evaluator that reads it.

The evaluator is OpenMask3D's copy of the benchmark's `eval_semantic_instance`, carried unmodified
under `utils/benchmark/openmask3d`. This module declares the location of the ground truth and
re-exports `evaluate`, so that every scorer in this repository reaches the benchmark through a
single point.

Note that the evaluator discards ground-truth instances of fewer than 100 vertices, through the
condition `gt["vert_count"] >= min_region_size` in `evaluate_matches`. The accuracy axis applies the
same threshold, so that the accuracy and AP results are computed over the same set of instances.
"""
import os

from utils.paths import *  # noqa: F403

from eval_semantic_instance import evaluate  # noqa: E402,F401  (re-exported)
import scannet200_constants as C200          # noqa: E402

IDS200 = list(C200.VALID_CLASS_IDS_200)
NAMES200 = list(C200.CLASS_LABELS_200)
IDSET = set(IDS200)

MASKDIR = os.path.join(OPENMASK3D, "masks")               # class-agnostic proposals
GTDIR = os.path.join(OPENMASK3D, "instance_gt", "validation")   # <scene>.txt, id*1000 + inst
