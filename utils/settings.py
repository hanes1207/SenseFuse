"""The eleven (pipeline, dataset) settings the paper reports, behind one interface.

Every table in the paper is a function of the same four arrays: the 2D head's score matrix ``S2``,
the shape head's ``S3``, the ground-truth class ``y`` of each proposal, and the scene each proposal
belongs to. :func:`load` returns them for any setting, so that Table I, the ablations and the
figures are all read off one construction rather than off separate scripts that can drift apart.

The three pipelines differ in what they publish, and the differences are all here:

* **Open-YOLO3D** and **OpenMask3D** release a class-agnostic mask set, so a head is applied to it
  directly. Their proposals are scored by whichever 2D head the row names, against text anchors in
  that head's own space.
* **Open3DIS** releases labelled rows rather than a score matrix, so its settings are read from a
  re-run of its pipeline with the per-proposal scores dumped before they are collapsed. That is what
  the dagger marks in the paper; see :mod:`utils.open3dis`.

ScanNet200 is scored over the evaluator's 198 instance classes. ``wall`` and ``floor`` are stuff
classes with no instance column, and predictions assigned either are discarded, so the denominator
is 6,534 rather than 6,750. Pass ``classes=200`` to recover the earlier convention.

The ``votes`` head is Open-YOLO3D's own sparse detection count. It is reported in the fusion-rule
ablation rather than in Table I, because it is not a continuous cosine channel and the fusion law
does not apply to it.
"""
import os

import numpy as np

from utils.paths import FEATS, FUSION, OM_REPRO, dump

# (pipeline, dataset) -> the heads the paper scores it with. The first is the one Table I prints
# for that pipeline when the pipeline supplies its own 2D head.
SETTINGS = {
    ("openyolo3d", "scannet200"): ("eva", "clipL", "votes"),
    ("openmask3d", "scannet200"): ("clipL",),
    ("open3dis", "scannet200"): ("own",),
    ("openyolo3d", "replica"): ("eva", "clipL", "votes"),
    ("openmask3d", "replica"): ("clipL",),
    ("open3dis", "replica"): ("own",),
    ("openyolo3d", "scannetpp"): ("eva", "clipL", "votes"),
    ("open3dis", "scannetpp"): ("own",),
}

# Table I's rows, in the order it prints them: (row label, pipeline, head).
TABLE1_ROWS = (("OY (EVA)", "openyolo3d", "eva"),
               ("OY (CLIP)", "openyolo3d", "clipL"),
               ("OpenMask3D", "openmask3d", "clipL"),
               ("Open3DIS", "open3dis", "own"))
DATASETS = ("scannet200", "replica", "scannetpp")

_STUFF = ("wall", "floor")


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def instance_columns():
    """The 198 of ScanNet200's 200 columns that the instance evaluator scores."""
    import scannet200_constants as C200
    names = list(C200.CLASS_LABELS_200)
    return np.array([i for i, n in enumerate(names) if n not in _STUFF], np.int64)


# ------------------------------------------------------------------ ScanNet200
def _scannet200(pipeline, head):
    from utils.scannet200 import build_scores, load_index
    us, uc, zuni = load_index()
    if pipeline == "openyolo3d":
        y = np.load(f"{FUSION}/gtmatch_full.npz")["ycls"].astype(np.int64)
        S2, S3 = build_scores(head, us, uc, zuni)
        return S2, S3, y, S2.max(1) > -0.5, us
    if pipeline == "openmask3d":
        y = np.load(f"{FUSION}/gtmatch_full.npz")["ycls"].astype(np.int64)
        feat = os.path.join(OM_REPRO, "features")
        have = [s for s in sorted(set(us.tolist())) if os.path.exists(f"{feat}/{s}.npy")]
        if not have:
            raise FileNotFoundError(
                f"OpenMask3D per-scene features not found under {feat}. Set SENSEFUSE_OM_REPRO to "
                f"it; data/DATASETS.md states its contents.")
        Z = np.zeros((len(us), 768), np.float32)
        for s in have:
            rows = np.where(us == s)[0]
            f = np.load(f"{feat}/{s}.npy")
            n = min(len(rows), len(f))
            Z[rows[:n]] = f[:n]
        S2 = (l2(Z) @ l2(np.load(f"{FUSION}/text_clipL_om_single.npy").astype(np.float32)).T
              ).astype(np.float32)
        # One denominator for every row of Table I: a proposal counts only where the shared crop
        # machinery saw it at all. Four ScanNet200 `ceiling' proposals are covered by OpenMask3D's
        # own feature stage but by neither Open-YOLO3D head, and counting them here would score
        # this row over a different population from the two above it.
        covered = np.isin(us, have) & (build_scores("eva", us, uc, zuni)[0].max(1) > -0.5)
        S2[~covered] = -1.0
        S3 = (zuni @ l2(np.load(f"{FUSION}/text_eva.npy").astype(np.float32)).T).astype(np.float32)
        return S2, S3, y, covered, us
    # Open3DIS: its own proposals, in raw-cosine space, already in the 198 instance columns.
    from utils import open3dis as OC
    S2, S3, y, cov, rows, scenes = OC.sn200_cosine_state()
    scene_of = np.concatenate([np.full(len(rows[s]), s) for s in scenes])
    return S2, S3, y, cov, scene_of


# ------------------------------------------------------------------ Replica
def _replica(pipeline, head):
    from utils import replica as RU
    if pipeline == "open3dis":
        from utils import open3dis as OC
        S2, S3, y, scene_of, _ = OC.replica_state()
        return S2, S3, y, np.ones(len(S2), bool), scene_of
    D = np.load(f"{RU.FEATS}/replica_dump.npz", allow_pickle=True)
    us, uc = D["scene"].astype(str), D["col"].astype(int)
    zuni = RU.l2(D["zuni"].astype(np.float32))
    y = RU.gt_match(us, uc)
    if pipeline == "openyolo3d":
        S2, S3 = RU.build_scores(head, us, uc, zuni)
        return S2, S3, y, S2.max(1) > -0.5, us
    # OpenMask3D on Replica: its own CLIP-L features against the same 48 anchors.
    feat = f"{OM_REPRO}/replica_features"
    have = [s for s in sorted(set(us.tolist())) if os.path.exists(f"{feat}/{s}.npy")]
    if not have:
        raise FileNotFoundError(f"OpenMask3D Replica features not found under {feat}")
    Z = np.zeros((len(us), 768), np.float32)
    for s in have:
        rows = np.where(us == s)[0]
        f = np.load(f"{feat}/{s}.npy")
        n = min(len(rows), len(f))
        Z[rows[:n]] = f[:n]
    S2 = (RU.l2(Z) @ np.load(f"{FUSION}/text_clipL_om_single_replica.npy").T).astype(np.float32)
    # The same shared denominator as on ScanNet200: one Replica `clock' proposal (office2, #22) is
    # covered by OpenMask3D's feature stage and by neither Open-YOLO3D head.
    covered = np.isin(us, have) & D["hasclip"].astype(bool)
    S2[~covered] = -1.0
    S3 = (zuni @ RU.l2(np.load(f"{RU.FEATS}/eva_text_48.npy").astype(np.float32)).T
          ).astype(np.float32)
    return S2, S3, y, covered, us


# ------------------------------------------------------------------ ScanNet++
def _spp_gt(entry):
    """Ground-truth class of each proposal, at IoU 0.5 against the scene's annotated instances."""
    y = -np.ones(len(entry["props"]), np.int64)
    for k, p in enumerate(entry["props"]):
        ps = set(p.tolist())
        best = (0.0, -1)
        for c, vs in entry["gt"]:
            i = len(ps & vs)
            if i:
                u = len(ps) + len(vs) - i
                if i / u > best[0]:
                    best = (i / u, c)
        if best[0] >= 0.5:
            y[k] = best[1]
    return y


def _scannetpp(pipeline, head):
    import pickle
    if pipeline == "open3dis":
        z = np.load(dump("state_o3d_spp.npz"), allow_pickle=True)
        y = z["Y"].astype(np.int64)
        return (z["S2"].astype(np.float32), z["S3"].astype(np.float32), y,
                np.ones(len(y), bool), z["scene"].astype(str))
    De = pickle.load(open(dump("spp_stageB_dump.pkl"), "rb"))
    Dc = pickle.load(open(dump("spp_stageB_dump_clipL.pkl"), "rb"))
    A = l2(De[0]["anchors"].astype(np.float32))
    S2l, S3l, yl, scl = [], [], [], []
    for se, sc_ in zip(De, Dc):
        n = len(se["props"])
        if head == "eva":
            S2l.append(l2(se["z2"].astype(np.float32)) @ A.T)
        elif head == "clipL":
            S2l.append(l2(sc_["z2"].astype(np.float32))
                       @ l2(sc_["anchors"].astype(np.float32)).T)
        else:
            # Open-YOLO3D's own vote / max convention, sliced to the 84 benchmark columns.
            v = np.load(f"{FEATS}/yolovotes_scannetpp/yolovotes_{se['scene']}.npz"
                        )["votes"].astype(np.float32)[:, :A.shape[0]]
            if v.shape[0] < n:
                v = np.vstack([v, np.zeros((n - v.shape[0], v.shape[1]), np.float32)])
            v = v[:n]
            S2l.append(v / np.maximum(v.max(1, keepdims=True), 1.0))
        S3l.append(l2(se["z3"].astype(np.float32)) @ A.T)
        yl.append(_spp_gt(se))
        scl.append(np.full(n, se["scene"]))
    y = np.concatenate(yl)
    return (np.concatenate(S2l).astype(np.float32), np.concatenate(S3l).astype(np.float32),
            y, np.ones(len(y), bool), np.concatenate(scl))


# ------------------------------------------------------------------ entry point
def load(pipeline, dataset, head="eva", classes=198):
    """``(S2, S3, y, covered, scene)`` for one setting.

    ``covered`` marks the proposals the 2D head could score at all; a proposal no frame saw carries
    no image embedding and is excluded from the accuracy denominator rather than counted wrong.
    """
    key = (pipeline, dataset)
    if key not in SETTINGS:
        raise NotImplementedError(
            f"{pipeline} on {dataset} is not a setting the paper reports "
            f"({', '.join(f'{p}/{d}' for p, d in SETTINGS)})")
    if head not in SETTINGS[key]:
        raise NotImplementedError(
            f"{pipeline}/{dataset} is scored with {SETTINGS[key]} in the paper, not {head!r}")
    if dataset == "scannet200":
        S2, S3, y, covered, scene = _scannet200(pipeline, head)
        if pipeline != "open3dis":                  # already in the 198 instance columns
            if classes == 198:
                cols = instance_columns()
                inv = -np.ones(200, np.int64)
                inv[cols] = np.arange(len(cols))
                S2, S3 = S2[:, cols], S3[:, cols]
                y = np.where(y >= 0, inv[np.maximum(y, 0)], -1)   # wall/floor proposals leave
            elif classes != 200:
                raise ValueError("classes must be 198 (the evaluator's) or 200")
    elif dataset == "replica":
        S2, S3, y, covered, scene = _replica(pipeline, head)
    else:
        S2, S3, y, covered, scene = _scannetpp(pipeline, head)
    return S2, S3, y, covered, scene
