"""Open3DIS's own proposals and dense class scores, for the settings it is scored on.

Open3DIS differs from the other two pipelines in what it releases. Open-YOLO3D and OpenMask3D
publish a class-agnostic mask set that every head then scores, so one mask loader serves all of
them. Open3DIS publishes, per scene, the result of its own labelling: a duplicated and top-k
expanded list of (mask, class, confidence) rows. Those rows cannot be scored by a second head,
because the class scores they came from have already been collapsed.

What a second head needs is the proposal set the pipeline scored and the dense per-class matrix it
produced, both of which exist inside ``get_final_instances`` before the collapse. This module reads
them back:

* :func:`replica_masks` rebuilds the proposal set in the order the pipeline scored it, namely its
  hierarchical-agglomerative 2D proposals followed by its 3D class-agnostic set, and then indexes
  that stack by the column list recorded when the shape encoder ran, so that row *i* of the score
  matrices and row *i* of the mask stack are the same proposal;
* :func:`replica_state` builds the two score matrices and the ground-truth match from source.

The 2D channel is ``centre(log p) / 300``, where ``p`` is ``inst_class_scores`` and 300 is the
pipeline's own ``scale_semantic_score``; this is the representation in which both channels are
cosine-valued, which is what the fusion law requires and is why the Open3DIS rows carry a dagger in
the paper. The shape channel is built exactly as Open-YOLO3D's is, from Uni3D against the 48
EVA02-E text anchors.
"""
import glob
import os

import numpy as np
import torch

from utils.paths import DATA, FEATS, FUSION, OPEN3DIS, OPENYOLO3D, dump, env_path

SCALE = 300.0                      # Open3DIS's final_instance.scale_semantic_score

DENSE = dump("o3d_rep_instscore")          # per-scene inst_class_scores, dumped before the collapse
PROPS = dump("o3d_rep_zuni_props")         # the Uni3D pass, proposal-indexed to match them
GTDIR = f"{OPENYOLO3D}/replica/ground_truth"
_MASKS = {}


def _rle(r):
    c = r["counts"]
    st = np.asarray(c[0::2], np.int64) - 1
    nm = np.asarray(c[1::2], np.int64)
    m = np.zeros(r["length"], bool)
    for a, b in zip(st, st + nm):
        m[a:b] = True
    return m


def replica_scenes():
    """The scenes for which both the dense scores and the shape pass exist."""
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{DENSE}/*.pth")
                  if os.path.exists(f"{PROPS}/{os.path.basename(f)[:-4]}.npz"))


def replica_masks(sc):
    """``(n_proposals, n_points)`` boolean masks, in the order the score matrices index.

    One scene is cached at a time: the stack is large and the scorers walk the scenes in order.
    """
    if sc not in _MASKS:
        _MASKS.clear()
        g = torch.load(f"{OPEN3DIS}/data/replica/replica_3d/{sc}.pth",
                       map_location="cpu", weights_only=False)
        npts = len(np.asarray(g[0]))
        P = []
        h = torch.load(f"{OPEN3DIS}/exp/version_8scenes/hier_agglo/{sc}.pth",
                       map_location="cpu", weights_only=False)
        for r in h["ins"]:
            P.append(_rle(r) if isinstance(r, dict) else np.asarray(r, bool))
        a = torch.load(f"{OPEN3DIS}/data/replica/cls_agnostic_replica_scannet200/{sc}.pth",
                       map_location="cpu", weights_only=False)
        ins = a["ins"]
        A = ins.numpy() if hasattr(ins, "numpy") else np.asarray(ins)
        if A.ndim == 2 and A.shape[1] == npts:
            P += [A[i].astype(bool) for i in range(A.shape[0])]
        else:
            P += [A[:, i].astype(bool) for i in range(A.shape[1])]
        M = np.stack(P)
        _MASKS[sc] = M[np.load(f"{PROPS}/{sc}.npz")["col"].astype(np.int64)]
    return _MASKS[sc]


def replica_state():
    """Build the Replica/Open3DIS arm from source.

    Returns ``(S2, S3, Y, scene, offsets)``: the two centred score matrices over Replica's 48
    classes, the ground-truth class of each proposal (-1 when unmatched at IoU 0.5), the scene of
    each row, and a map from scene to its row indices.
    """
    from evaluate.replica import eval_semantic_instance as REV
    from open3dis.dataset.replica import INSTANCE_CAT_REPLICA as OCAT

    names = list(REV.CLASS_LABELS)
    o2v = np.array([names.index(n) for n in OCAT], np.int64)   # Open3DIS order -> evaluator order
    l2 = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    ctr = lambda S: S - S.mean(1, keepdims=True)
    anchors = l2(np.load(f"{FEATS}/replica/eva_text_48.npy").astype(np.float32))

    scenes = replica_scenes()
    if len(scenes) < 8:
        raise FileNotFoundError(
            f"only {len(scenes)} of Replica's 8 scenes have Open3DIS dense scores under {DENSE} "
            f"and a shape pass under {PROPS}; data/DATASETS.md states where both come from")
    Pa, ZUa, off, cur, sc_of = [], [], {}, 0, []
    for sc in scenes:
        p = torch.load(f"{DENSE}/{sc}.pth", map_location="cpu",
                       weights_only=False)["inst_class_scores"].numpy().astype(np.float64)
        z = np.load(f"{PROPS}/{sc}.npz")
        col = z["col"].astype(int)
        if p.shape[0] <= col.max():
            raise ValueError(f"{sc}: the dense scores {p.shape} do not cover column {col.max()}; "
                             f"the dump and the shape pass saw different proposals")
        Pa.append(p[col])
        ZUa.append(l2(z["zuni"].astype(np.float32)))
        off[sc] = np.arange(cur, cur + len(col))
        cur += len(col)
        sc_of += [sc] * len(col)
    S2 = (ctr(np.log(np.maximum(np.concatenate(Pa), 1e-12))) / SCALE).astype(np.float32)
    S3 = ctr((np.concatenate(ZUa) @ anchors[o2v].T).astype(np.float32))

    # Ground-truth match, at IoU 0.5 against the evaluator's own instance file. No minimum instance
    # size is applied here: this arm is matched over every annotated instance, as the run that
    # produced the published row did.
    id2idx = {int(v): i for i, v in enumerate(REV.VALID_CLASS_IDS)}
    v2o = np.full(len(names), -1, np.int64)
    v2o[o2v] = np.arange(len(o2v))
    Y = np.full(len(S2), -1, np.int64)
    for sc in scenes:
        gt = np.loadtxt(f"{GTDIR}/{sc}.txt", dtype=np.int64)
        M = replica_masks(sc)
        if M.shape[1] != len(gt):
            print(f"  !! {sc} vertex mismatch {M.shape[1]} vs {len(gt)}", flush=True)
            continue
        iids = np.unique(gt)
        iids = iids[iids >= 1000]
        keep = [(int(i), id2idx[int(i) // 1000]) for i in iids if int(i) // 1000 in id2idx]
        if not keep:
            continue
        G = np.stack([gt == i for i, _ in keep], 1).astype(np.float32)
        gcls = np.array([c for _, c in keep], np.int64)
        Pm = M.T.astype(np.float32)
        inter = Pm.T @ G
        iou = inter / np.maximum(Pm.sum(0)[:, None] + G.sum(0)[None, :] - inter, 1e-9)
        bi = iou.argmax(1)
        best = iou[np.arange(len(bi)), bi]
        r = off[sc]
        Y[r[best >= 0.5]] = v2o[gcls[bi[best >= 0.5]]]
    return S2, S3, Y, np.array(sc_of), off


# ------------------------------------------------------------------ ScanNet200, raw cosine
#
# The ScanNet200 row is a self-run rather than a reading of the released artefact, for the same
# reason as Replica: the released rows carry a collapsed label, not a score matrix. The pipeline was
# re-run with its per-proposal features dumped, and the 2D channel is the raw cosine
# ``l2(inst_feat_mean) @ text_feat``, taken before the softmax the pipeline applies. Two dumps are
# kept: ``dense`` carries the pipeline's own posterior and ``denseraw`` the features, and the
# proposal identity between them is asserted by comparing the run-length encodings rather than
# assumed, because only that makes the ground-truth match of one reusable for the other.
#
# These cells are not row-comparable with the released-artefact rows: the proposals are this run's,
# and its own 2D flat reads 21.7 against the released 23.0 on the same scenes. That is why they
# carry a dagger in the paper.
SN200_COS = env_path("SENSEFUSE_O3D_COSINE", os.path.join(DATA, "open3dis_cosine"))
_SN200_MK = {}


def sn200_cosine_scenes():
    return sorted(f[:-4] for f in os.listdir(f"{SN200_COS}/dense") if f.endswith(".pth"))


def _rle_key(ins):
    return [tuple(np.asarray(r["counts"]).tolist()) for r in ins]


def sn200_cosine_masks(sc):
    """``(n_proposals, n_points)`` masks of the self-run, cached one scene at a time."""
    if sc not in _SN200_MK:
        _SN200_MK.clear()
        d = torch.load(f"{SN200_COS}/dense/{sc}.pth", map_location="cpu", weights_only=False)
        _SN200_MK[sc] = np.stack([_rle(r) if isinstance(r, dict) else np.asarray(r, bool)
                                  for r in d["ins"]])
    return _SN200_MK[sc]


def sn200_cosine_state():
    """Build the Open3DIS/ScanNet200 arm in raw-cosine space.

    Returns ``(S2, S3, y, cov, rows, scenes)``. ``cov`` marks the proposals the shape encoder
    covered; the fusion is applied only there, and the 2D score stands elsewhere.
    """
    from utils import scannet200_ap as AP

    ctr = lambda S: S - S.mean(1, keepdims=True)
    l2 = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    scenes = [s for s in sn200_cosine_scenes()
              if os.path.exists(f"{SN200_COS}/uni/{s}.npz")
              and os.path.exists(f"{SN200_COS}/denseraw/{s}.pth")]
    if not scenes:
        raise FileNotFoundError(
            f"no Open3DIS ScanNet200 self-run under {SN200_COS}; it needs dense/, denseraw/ and "
            f"uni/, and data/DATASETS.md states where they come from")
    Rl, Zl, cl, rows, cur = [], [], [], {}, 0
    for sc in scenes:
        dn = torch.load(f"{SN200_COS}/denseraw/{sc}.pth", map_location="cpu", weights_only=False)
        do = torch.load(f"{SN200_COS}/dense/{sc}.pth", map_location="cpu", weights_only=False)
        if _rle_key(dn["ins"]) != _rle_key(do["ins"]):
            raise ValueError(f"{sc}: the two dumps hold different proposals")
        R = (l2(dn["inst_feat_mean"].float().numpy().astype(np.float32))
             @ dn["text_feat"].float().numpy().astype(np.float32).T).astype(np.float32)
        z = np.load(f"{SN200_COS}/uni/{sc}.npz")
        col = z["col"].astype(int)
        if int(z["n"]) != len(R):
            raise ValueError(f"{sc}: shape pass covers {int(z['n'])} proposals, scores {len(R)}")
        Z = np.zeros((len(R), z["zuni"].shape[1]), np.float32)
        Z[col] = l2(z["zuni"].astype(np.float32))
        c = np.zeros(len(R), bool)
        c[col] = True
        Rl.append(R)
        Zl.append(Z)
        cl.append(c)
        rows[sc] = np.arange(cur, cur + len(R))
        cur += len(R)
    S2 = ctr(np.concatenate(Rl)).astype(np.float32)
    anchors = l2(np.load(f"{FUSION}/text_eva.npy").astype(np.float32))
    names200, o2v = _sn200_o2v()
    S3 = ctr((np.concatenate(Zl) @ anchors[o2v].T).astype(np.float32))
    cov = np.concatenate(cl)

    # Ground-truth match at IoU 0.5, in the evaluator's own 198 instance classes.
    import scannet200_constants as C200
    idarr = np.array(list(C200.VALID_CLASS_IDS_200))
    id2col = {int(idarr[names200.index(n)]): i for i, n in enumerate(_OCAT())}
    Y = []
    for sc in scenes:
        M = sn200_cosine_masks(sc)
        gt = np.loadtxt(f"{AP.GTDIR}/{sc}.txt").astype(np.int64)
        if len(gt) != M.shape[1]:
            raise ValueError(f"{sc}: {len(gt)} ground-truth points against {M.shape[1]}")
        iids = np.unique(gt)
        iids = iids[iids >= 1000]
        keep = [(int(i), id2col[int(i) // 1000]) for i in iids if int(i) // 1000 in id2col]
        y = -np.ones(len(M), np.int64)
        if keep:
            G = np.stack([gt == i for i, _ in keep], 1)
            gcls = np.array([c for _, c in keep], np.int64)
            inter = M.astype(np.float32) @ G.astype(np.float32)
            iou = inter / np.maximum(M.sum(1)[:, None] + G.sum(0)[None, :] - inter, 1e-9)
            bi = iou.argmax(1)
            best = iou[np.arange(len(bi)), bi]
            y[best >= 0.5] = gcls[bi[best >= 0.5]]
        Y.append(y)
    return S2, S3, np.concatenate(Y), cov, rows, scenes


def _OCAT():
    from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 as OCAT
    return list(OCAT)


def _sn200_o2v():
    """(the 200 benchmark names, the 198 instance columns as indices into them)."""
    import scannet200_constants as C200
    names = list(C200.CLASS_LABELS_200)
    return names, np.array([names.index(n) for n in _OCAT()], np.int64)


# ------------------------------------------------------------------ ScanNet++
#
# The ScanNet++ arm has the same shape as Replica's: the pipeline's own proposals -- its
# hierarchical-agglomerative 2D set followed by ISBNet's class-agnostic 3D set -- and its dense
# per-class scores, dumped before the collapse. ``centre(log p) / 300`` is again the 2D channel.
# Predictions are written out per scene and scored by the ScanNet++ evaluator, because that
# benchmark's submission format is a written directory rather than an in-memory list.
SPP_PROPS = env_path("SENSEFUSE_O3D_SPP_PROPOSALS",
                     os.path.join(DATA, "open3dis_scannetpp_proposals"))


def spp_state():
    """``(S2, S3, y, scene)`` for Open3DIS on ScanNet++, centred, over the 84 instance classes."""
    ctr = lambda S: S - S.mean(1, keepdims=True)
    z = np.load(dump("state_o3d_spp.npz"), allow_pickle=True)
    return (ctr(z["S2"].astype(np.float32)), ctr(z["S3"].astype(np.float32)),
            z["Y"].astype(np.int64), z["scene"].astype(str))


def spp_proposals(sc):
    """The run-length encodings of one scene's proposals, in the order the scores index them."""
    from utils.paths import O3D_ROOT
    col = np.load(f"{dump('o3d_spp_zuni')}/{sc}.npz")["col"].astype(int)
    h2 = torch.load(f"{SPP_PROPS}/{sc}.pth", map_location="cpu", weights_only=False)
    i3 = torch.load(f"{O3D_ROOT}/Scannetpp/Scannetpp_3D/val/isbnet_clsagnostic_scannetpp/{sc}.pth",
                    map_location="cpu", weights_only=False)
    props = list(h2["ins"]) + list(i3["ins"])
    return [props[c] for c in col]


def spp_write(S, scene_of, rule, outdir, topk=600):
    """Write one arm's ScanNet++ submission.

    ``rule`` is ``top1`` (one prediction per proposal) or ``flat`` (the top-600 entries of the raw
    flattened matrix -- ScanNet++'s convention, without the per-mask normalisation ScanNet200 uses).
    Every prediction carries confidence 1.0, as each released implementation does.
    """
    import json
    from utils.paths import SCANNETPP_EVAL
    os.makedirs(outdir, exist_ok=True)
    for sc in sorted(set(scene_of.tolist())):
        rows = np.flatnonzero(scene_of == sc)
        npts = len(json.load(open(f"{SCANNETPP_EVAL}/data/{sc}/scans/segments.json"))["segIndices"])
        rles = [{"length": npts, "counts": r["counts"]} for r in spp_proposals(sc)]
        Ps = S[rows]
        if rule == "top1":
            cls = Ps.argmax(1)
            keep = list(range(len(cls)))
        else:
            fl = Ps.reshape(-1)
            k = min(topk, fl.size)
            idx = np.argpartition(-fl, k - 1)[:k]
            idx = idx[np.argsort(-fl[idx])]
            keep, cls = (idx // Ps.shape[1]).tolist(), idx % Ps.shape[1]
        torch.save({"ins": [rles[i] for i in keep],
                    "class": torch.from_numpy(np.asarray(cls, np.int64)),
                    "conf": torch.from_numpy(np.ones(len(keep), np.float32))},
                   f"{outdir}/{sc}.pth")
    return outdir
