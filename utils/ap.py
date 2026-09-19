#!/usr/bin/env python3
"""Instance-segmentation AP, computed independently of the production path.

This module re-computes every cell of the AP table: the fusion, the submission rule and the scoring
are implemented here, sharing nothing with the production path except the stored features and text
embeddings, which are data rather than computation.

    fusion      (1-w) * center(S2) + w * center(S3), with w taken from the reported w_law column
    submission  top-1, that is one class per mask, or flatten-K, that is the top K (mask, class)
                pairs per scene, both at conf = 1.0 and both implemented here rather than imported
    scoring     the evaluator's matching semantics, replicated from scannetv2_inst_eval.py: the
                prediction floor at line 326, the ground-truth floor at line 88, the strict
                inequality at line 110, the duplicate false positive at line 115, and the geometric
                false-positive exemption at line 139; followed by Proposition 2,
                AP_c = 1/2 r_c (1 + p_c), in place of a precision-recall curve. mAP averages AP over
                IoU thresholds 0.5:0.05:0.95, as the evaluator does.

Agreement between this module and the production path therefore tests three propositions at once:
that the reported numbers are correct, that Proposition 2 holds on the predictions the table reports,
and that the submission rules behave as stated.

    python -m utils.ap sn200 eva          # both arms, both rules, seven cells each
"""
import os, sys
from utils.paths import *  # noqa: F403
import contextlib
import io
import os
import sys

import numpy as np

MIN_REGION = 100
IOUS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)      # evaluator's list, :36
ctr = lambda S: S - S.mean(1, keepdims=True)


# ------------------------------------------------------------------ scoring
def associate(pred, gt_masks, gt_cls):
    """IoU association, computed once and reused for every threshold.

    pred: list of (class, bool mask, npoints), already size-filtered.
    gt_masks/gt_cls: parallel lists; masks are the encoded ones, unfiltered by size.
    Returns (sizes, by_c_gt, matched_pred, matched_gt), where matched_pred[j] is in prediction
    order because that is the order the evaluator's gt["matched_pred"] is built in (:346).
    """
    sizes = np.array([int(m.sum()) for m in gt_masks], np.int64)
    by_c_gt, by_c_pred = {}, {}
    for j, c in enumerate(gt_cls):
        by_c_gt.setdefault(int(c), []).append(j)
    for i, (c, _, _) in enumerate(pred):
        by_c_pred.setdefault(int(c), []).append(i)
    matched_pred = {j: [] for j in range(len(gt_masks))}
    matched_gt = {i: [] for i in range(len(pred))}
    for c, pidx in by_c_pred.items():
        gidx = by_c_gt.get(c, [])
        if not gidx:
            continue
        # a matrix product rather than a per-prediction boolean AND: flatten-600 submits about 600
        # masks per scene against roughly 200k points, and the elementwise form is approximately
        # 15 times slower than the BLAS one.
        G = np.stack([gt_masks[j] for j in gidx], 1).astype(np.float32)          # (V, I)
        P = np.stack([pred[i][1] for i in pidx], 1).astype(np.float32)           # (V, R)
        inter = (P.T @ G).astype(np.int64)                                       # (R, I)
        for a, i in enumerate(pidx):
            pn = pred[i][2]
            nz = np.where(inter[a] > 0)[0]
            for k in nz:
                j = gidx[k]
                iou = float(inter[a, k]) / (sizes[j] + pn - inter[a, k])
                matched_pred[j].append((i, iou))
                matched_gt[i].append((j, iou, int(inter[a, k])))
        del G, P, inter
    # gt["matched_pred"] is built in prediction order (:346); grouping by class above visits
    # predictions in index order within a class, so the lists are already in that order.
    return sizes, by_c_gt, matched_pred, matched_gt


def count_at(pred, assoc, void_int, K, iou_th):
    """(tp, fp, ngt) per class at one threshold, from a precomputed association."""
    sizes, by_c_gt, matched_pred, matched_gt = assoc
    tp = np.zeros(K); fp = np.zeros(K); ngt = np.zeros(K)
    visited = np.zeros(len(pred), bool)
    for c, gidx in by_c_gt.items():
        keep_g = [j for j in gidx if sizes[j] >= MIN_REGION]         # :88
        ngt[c] += len(keep_g)
        for j in keep_g:
            cur = False
            for i, iou in matched_pred[j]:
                if visited[i]:
                    continue                                         # :104
                if iou > iou_th:                                     # :110 strict
                    if cur:
                        fp[c] += 1                                   # :115 duplicate
                    else:
                        cur = True
                        visited[i] = True
            if cur:
                tp[c] += 1
    for i, (c, _, pn) in enumerate(pred):
        if any(iou > iou_th for _, iou, _ in matched_gt[i]):          # :140 geometric escape
            continue
        ign = void_int[i]                                             # :144
        for j, _, inter in matched_gt[i]:
            if sizes[j] < MIN_REGION:                                 # :150
                ign += inter
        if ign / pn <= iou_th:                                        # :158
            fp[int(c)] += 1
    return tp, fp, ngt


def sweep(scene_iter, K):
    """Every IoU threshold in one pass over the scenes. Returns a (thresholds, K) AP array."""
    TP = {t: np.zeros(K) for t in IOUS}
    FP = {t: np.zeros(K) for t in IOUS}
    NG = {t: np.zeros(K) for t in IOUS}
    for pred, gt_masks, gt_cls, void in scene_iter:
        assoc = associate(pred, gt_masks, gt_cls)
        void_int = [int(np.count_nonzero(void & pm)) for _, pm, _ in pred]
        for t in IOUS:
            a, b, c = count_at(pred, assoc, void_int, K, t)
            TP[t] += a; FP[t] += b; NG[t] += c
    ap = np.full((len(IOUS), K), np.nan)
    for ti, t in enumerate(IOUS):
        for c in range(K):
            if NG[t][c] == 0:
                continue                                   # evaluator reports nan -> nanmean
            r = TP[t][c] / NG[t][c]
            p = TP[t][c] / (TP[t][c] + FP[t][c]) if (TP[t][c] + FP[t][c]) > 0 else 0.0
            ap[ti, c] = 0.5 * r * (1 + p)
    return ap


def summarise(ap, names, head, common, tail):
    """mAP over 0.5:0.05:0.95, AP50, AP25, and the three ScanNet200 splits."""
    i50, i25 = list(IOUS).index(0.5), len(IOUS) - 1
    swept = ap[:len(IOUS) - 1]                              # 0.25 is not part of the mean
    out = dict(ap=100 * np.nanmean(swept), ap50=100 * np.nanmean(ap[i50]),
               ap25=100 * np.nanmean(ap[i25]))
    per = {n: np.nanmean(swept[:, i]) for i, n in enumerate(names)}
    for key, cats in (("head", head), ("common", common), ("tail", tail)):
        v = [per[n] for n in cats if n in per and not np.isnan(per[n])]
        out[key] = 100 * float(np.mean(v)) if v else float("nan")
    return out


# ------------------------------------------------------------------ submission rules
def submit_norm(S):
    """Per-mask min-max normalisation, matching the form of Open-YOLO3D's ``vote / max``.

    The top-1 rule is invariant to this transform, since it is monotone within a row and the argmax
    therefore cannot move. The flatten-K rule is not invariant: it ranks (mask, class) pairs across
    masks, and the normalisation places every mask's highest-scoring class at exactly 1.0. Omitting
    it reads 1.4 mAP low on the CLIP-L channel and 0.0 on the vote channel, which is already
    max-normalised; this asymmetry is why the omission was visible in only one of the two.
    """
    lo = S.min(1, keepdims=True)
    hi = S.max(1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


def top1(S, rows):
    lab = S.argmax(1)
    return [(int(r), int(lab[r])) for r in rows]


def flatten_raw(S, rows, topk):
    """The ScanNet++ flattening rule, which differs from the ScanNet200 rule.

    The reference implementation is ``fl = P.reshape(-1); k = min(600, fl.size);
    argpartition(-fl, k-1)``: the raw score, without per-mask normalisation and without a positivity
    filter. Applying the ScanNet200 rule of normalising and then discarding non-positive entries
    reads 0.1 to 0.2 mAP high on every flattened cell.
    """
    D = S[rows]
    flat = D.reshape(-1)
    k = min(topk, len(flat))
    idx = np.argpartition(-flat, k - 1)[:k]
    return [(int(rows[int(ix) // D.shape[1]]), int(ix) % D.shape[1]) for ix in idx]


def flatten_rowmax(S, rows, topk):
    """Open3DIS's cosine-space flattening: divide each row by its maximum, then take the top K
    positive entries.

    This is not the same normalisation as :func:`submit_norm`, which maps each row onto ``[0, 1]``
    by its minimum and maximum. Dividing by the maximum alone leaves a centred row's negative
    entries negative, and they are then dropped; the min-max form would rank them against one
    another instead. On this pipeline's centred cosines the difference is 0.3 to 0.5 mAP, which is
    why the rule is a property of the pipeline and not of the table.
    """
    D = S[rows]
    Dn = D / np.maximum(D.max(1, keepdims=True), 1e-9)
    flat = Dn.reshape(-1)
    k = min(topk, len(flat))
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[flat[idx] > 0]
    return [(int(rows[int(ix) // D.shape[1]]), int(ix) % D.shape[1]) for ix in idx]


def flatten(S, rows, topk):
    D = submit_norm(S)[rows]
    flat = D.reshape(-1)
    k = min(topk, len(flat))
    idx = np.argpartition(-flat, k - 1)[:k]
    out = []
    for ix in idx:
        rr, c = divmod(int(ix), D.shape[1])
        if flat[ix] > 0:
            out.append((int(rows[rr]), c))
    return out


# ------------------------------------------------------------------ ScanNet200
def _instance_cols(names198):
    """The columns of a 200-way score matrix that the instance evaluator scores, in its own order.

    The evaluator's 198 names are ScanNet200's 200 with `wall` and `floor` removed, in order, so the
    restriction is a column selection and the resulting index is already the evaluator's.
    """
    import scannet200_constants as C200
    names200 = list(C200.CLASS_LABELS_200)
    cols = np.array([names200.index(n) for n in names198], np.int64)
    assert len(cols) == len(names198)
    return cols


def run_sn200(source, w, rule, topk=600):
    from utils import scannet200_ap as AP
    import scannet200_constants as C200
    from scannet200_splits import (HEAD_CATS_SCANNET_200 as HCAT,
                                   COMMON_CATS_SCANNET_200 as CCAT,
                                   TAIL_CATS_SCANNET_200 as TCAT)
    from utils import scannet200 as SN

    # The AP evaluator scores ScanNet200 instances over 198 classes rather than 200:
    # eval_semantic_instance removes `wall` and `floor` from CLASS_LABELS when dataset ==
    # "scannet200". The difference is not cosmetic, and it is not enough to discard the predictions
    # that land on those two labels. A 200-way argmax spends 97 to 353 of the 13,165 proposals on
    # them, and line 488 (`if not label_id in ID_TO_LABEL: continue`) then drops each one silently;
    # restricting the score matrix to the 198 instance columns BEFORE the argmax instead turns those
    # proposals into scorable predictions, which is what the table reports and is worth up to 0.15
    # mAP on the top-1 rule. Since bool_void is `not in VALID_CLASS_IDS`, every wall and floor point
    # is also void, so a prediction falling on one is not a false positive either.
    import eval_semantic_instance as OM
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            OM.evaluate({}, "/nonexistent", output_file=f"{TMPDIR}/u_probe.txt",
                        dataset="scannet200")
        except Exception:
            pass                                    # only run for its dataset switch
    names = list(OM.CLASS_LABELS)
    ids198 = np.asarray(OM.VALID_CLASS_IDS)
    id2idx = {int(v): i for i, v in enumerate(ids198)}
    IDARR = np.array(list(C200.VALID_CLASS_IDS_200))      # the space predictions are written in
    K = len(names)

    us, uc, zuni = SN.load_index()
    S2, S3 = SN.build_scores(source, us, uc, zuni)
    cols = _instance_cols(names)
    S = (1 - w) * ctr(S2[:, cols]) + w * ctr(S3[:, cols])
    scenes = sorted(set(us.tolist()))

    def gen():
        for si, s in enumerate(scenes):
            if si % 50 == 0:
                print(f"    {si}/{len(scenes)}", flush=True)
            rows = np.where(us == s)[0]
            gt = np.loadtxt(f"{AP.GTDIR}/{s}.txt", dtype=np.int64)
            m = SN.get_masks(s)
            if m.shape[0] != len(gt):
                SN.MASKS.pop(s, None)
                continue
            ids = np.unique(gt); ids = ids[ids >= 1000]
            keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
            gt_masks = [gt == i for i, _ in keep]
            gt_cls = [c for _, c in keep]
            void = ~np.isin(gt // 1000, ids198)        # wall and floor are void here, see above
            ent = top1(S, rows) if rule == "top1" else flatten(S, rows, topk)
            pred = []
            for r, c in ent:                           # c already indexes the evaluator's classes
                pm = m[:, uc[r]]
                n = int(pm.sum())
                if n >= MIN_REGION:                    # :326
                    pred.append((c, pm, n))
            yield pred, gt_masks, gt_cls, void
            del m
            SN.MASKS.pop(s, None)

    return summarise(sweep(gen(), K), names, HCAT, CCAT, TCAT)




# ------------------------------------------------------------------ Replica
def _replica_eval_globals():
    """Class list and ids the Replica evaluator uses, read from it rather than restated."""
    from evaluate.replica import eval_semantic_instance as REV   # from the OPENYOLO3D_ROOT checkout
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            REV.evaluate({}, "/nonexistent", output_file=f"{TMPDIR}/u_probe2.txt",
                         dataset="replica")
        except Exception:
            pass
    return list(REV.CLASS_LABELS), np.asarray(REV.VALID_CLASS_IDS)


def run_replica(source, w, rule, topk=600):
    # IMPORT ORDER: utils.scannet200 registers OpenMask3D's `eval_semantic_instance` under that
    # bare module name; importing utils.replica afterwards would silently hand it the 18-class
    # ScanNet module and drop Replica's GT matching from 139 to 22, with no exception raised.
    from utils import replica as RU
    names, ids48 = _replica_eval_globals()
    id2idx = {int(v): i for i, v in enumerate(ids48)}
    K = len(names)

    D = np.load(f"{RU.FEATS}/replica_dump.npz", allow_pickle=True)
    us, uc = D["scene"].astype(str), D["col"].astype(np.int64)
    zuni = RU.l2(D["zuni"].astype(np.float32))
    S2, S3 = RU.build_scores(source, us, uc, zuni)
    S = (1 - w) * ctr(S2) + w * ctr(S3)
    scenes = sorted(set(us.tolist()))

    def gen():
        for s in scenes:
            rows = np.where(us == s)[0]
            gt = np.loadtxt(f"{RU.GTDIR}/{s}.txt", dtype=np.int64)
            m = RU.get_masks(s)
            if m.shape[0] != len(gt):
                RU.MASKS.pop(s, None)
                continue
            ids = np.unique(gt); ids = ids[ids >= 1000]
            keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
            gt_masks = [gt == i for i, _ in keep]
            gt_cls = [c for _, c in keep]
            void = ~np.isin(gt // 1000, ids48)         # :320
            ent = top1(S, rows) if rule == "top1" else flatten(S, rows, topk)
            pred = []
            for r, c in ent:                           # Replica predictions are 0-based indices
                pm = m[:, uc[r]]                       # into VALID_CLASS_IDS (:285 PRED_ID_TO_ID)
                n = int(pm.sum())
                if n >= MIN_REGION:
                    pred.append((c, pm, n))
            yield pred, gt_masks, gt_cls, void
            del m
            RU.MASKS.pop(s, None)

    ap = sweep(gen(), K)
    i50 = list(IOUS).index(0.5)
    swept = ap[:len(IOUS) - 1]
    return dict(ap=100 * np.nanmean(swept), ap50=100 * np.nanmean(ap[i50]),
                ap25=100 * np.nanmean(ap[-1]))




# ------------------------------------------------------------------ ScanNet++
def run_spp(ovdir):
    """Score a written ScanNet++ prediction directory.

    The semantics match the reference implementation that reproduced three of these cells to the
    precision the evaluator prints. All ten IoU thresholds are swept, so that the mAP column is
    verified and not only AP50.
    """
    os.environ.setdefault("SPP_CLASSES", "100")
    import torch
    from utils import ap_scannetpp as E
    K = len(E.NAMES)
    valid = np.arange(K) + 1
    GT_GROUND = f"{O3D_ROOT}/Scannetpp/Scannetpp_3D/val/groundtruth"
    scenes = sorted(os.path.basename(f)[:-4] for f in os.listdir(ovdir) if f.endswith(".pth"))
    scenes = [s for s in scenes if os.path.exists(f"{GT_GROUND}/{s}.pth")]

    def gen():
        for i, sc in enumerate(scenes):
            if i % 10 == 0:
                print(f"    {i}/{len(scenes)}", flush=True)
            gs, gi, _ = E.build_gt_auth(sc)
            ss = gs.astype(np.int64) - 16 + 1
            ss[ss < 0] = 0
            gts = ss * 1000 + (gi.astype(np.int64) + 1)      # the evaluator's own encoding
            gt_masks, gt_cls = [], []
            for gid in np.unique(gts):
                if gid == 0:
                    continue
                lab = int(gid // 1000)
                if lab not in valid:
                    continue
                gt_masks.append(gts == gid)
                gt_cls.append(lab - 1)
            void = ~np.in1d(gts // 1000, valid)
            p = torch.load(f"{ovdir}/{sc}.pth", map_location="cpu", weights_only=False)
            pcls = np.asarray(p["class"]).astype(int) + E._PSHIFT
            pred = []
            for c, r in zip(pcls, p["ins"]):
                m = E.rle_decode(r).astype(bool)
                n = int(m.sum())
                if n >= MIN_REGION:
                    pred.append((int(c), m, n))
            yield pred, gt_masks, gt_cls, void

    ap = sweep(gen(), K)
    i50 = list(IOUS).index(0.5)
    swept = ap[:len(IOUS) - 1]
    return dict(ap=100 * np.nanmean(swept), ap50=100 * np.nanmean(ap[i50]),
                ap25=100 * np.nanmean(ap[-1]), n=len(scenes))


# ------------------------------------------------------------------ ScanNet200, other methods
def _sn200_eval_space():
    import eval_semantic_instance as OM
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            OM.evaluate({}, "/nonexistent", output_file=f"{TMPDIR}/u_probe.txt",
                        dataset="scannet200")
        except Exception:
            pass
    return list(OM.CLASS_LABELS), np.asarray(OM.VALID_CLASS_IDS)


def _sn200_gen(S, us, uc, SN, AP, IDARR, ids198, id2idx, rule, topk, scenes, masks_of=None,
               cols=None):
    """One scene at a time: predictions under `rule`, GT and void in the evaluator's 198 space.

    ``cols`` restricts ``S`` to the evaluator's 198 instance classes before the argmax, which is what
    the table reports; see the comment in :func:`run_sn200`. Passing ``None`` keeps the 200-way
    argmax and drops whatever lands on `wall` or `floor`, which is the earlier convention.
    """
    if cols is not None:
        S = S[:, cols]
    for si, s in enumerate(scenes):
        if si % 50 == 0:
            print(f"    {si}/{len(scenes)}", flush=True)
        rows = np.where(us == s)[0]
        gt = np.loadtxt(f"{AP.GTDIR}/{s}.txt", dtype=np.int64)
        m = SN.get_masks(s) if masks_of is None else masks_of(s)
        if m.shape[0] != len(gt):
            SN.MASKS.pop(s, None)
            continue
        ids = np.unique(gt); ids = ids[ids >= 1000]
        keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
        ent = top1(S, rows) if rule == "top1" else flatten(S, rows, topk)
        pred = []
        base = int(rows[0]) if len(rows) else 0
        for r, c in ent:
            if cols is None:
                raw = int(IDARR[c])
                if raw not in id2idx:                  # wall / floor under the 200-way argmax
                    continue
                c = id2idx[raw]
            pm = m[:, uc[r]] if masks_of is None else m[:, r - base]
            n = int(pm.sum())
            if n >= MIN_REGION:
                pred.append((c, pm, n))
        yield pred, [gt == i for i, _ in keep], [c for _, c in keep], ~np.isin(gt // 1000, ids198)
        del m
        SN.MASKS.pop(s, None)


def run_sn200_om(w, rule, topk=600):
    """OpenMask3D on ScanNet200: its own crop features against its own single-template text."""
    from utils import scannet200_ap as AP
    import scannet200_constants as C200
    from utils import scannet200 as SN
    from scannet200_splits import (HEAD_CATS_SCANNET_200 as H, COMMON_CATS_SCANNET_200 as C,
                                   TAIL_CATS_SCANNET_200 as T)
    names, ids198 = _sn200_eval_space()
    id2idx = {int(v): i for i, v in enumerate(ids198)}
    IDARR = np.array(list(C200.VALID_CLASS_IDS_200))
    us, uc, zuni = SN.load_index()
    zuni = SN.l2(zuni)
    FEAT = f"{OM_REPRO}/features"
    have = [s for s in sorted(set(us.tolist())) if os.path.exists(f"{FEAT}/{s}.npy")]
    Z = np.zeros((len(us), 768), np.float32)
    for s in have:
        r_ = np.where(us == s)[0]
        f = np.load(f"{FEAT}/{s}.npy")
        n = min(len(r_), len(f))
        Z[r_[:n]] = f[:n]
    TA = SN.l2(np.load(f"{SN.CACHE}/text_eva.npy").astype(np.float32))
    S2 = (SN.l2(Z) @ np.load(f"{SN.CACHE}/text_clipL_om_single.npy").T).astype(np.float32)
    S2[~np.isin(us, have)] = -1.0
    S = (1 - w) * ctr(S2) + w * ctr((zuni @ TA.T).astype(np.float32))
    gen = _sn200_gen(S, us, uc, SN, AP, IDARR, ids198, id2idx, rule, topk,
                     sorted(set(us.tolist())), cols=_instance_cols(names))
    return summarise(sweep(gen, len(names)), names, H, C, T)


def run_sn200_o3d(w, rule, topk=600):
    """Open3DIS on ScanNet200: its released top-600 confidences over its own 198 names,
    scattered into the benchmark's 200 columns, on its own released hier-agglo masks."""
    import torch
    from utils import scannet200_ap as AP
    import scannet200_constants as C200
    from utils import scannet200 as SN
    from utils.gtmatch import rle_decode
    from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 as OCAT
    from scannet200_splits import (HEAD_CATS_SCANNET_200 as H, COMMON_CATS_SCANNET_200 as C,
                                   TAIL_CATS_SCANNET_200 as T)
    names, ids198 = _sn200_eval_space()
    id2idx = {int(v): i for i, v in enumerate(ids198)}
    NAMES200 = list(C200.CLASS_LABELS_200)
    IDARR = np.array(list(C200.VALID_CLASS_IDS_200))
    O2V = np.array([NAMES200.index(n) for n in OCAT], np.int64)
    SRC = f"{O3D_DL}/Result_OpenVocab_ISBNet-GSAM/final_result_hier_agglo"
    UNID = f"{FEATS}/o3drel_uni"
    TA = SN.l2(np.load(f"{SN.CACHE}/text_eva.npy").astype(np.float32))
    scenes = sorted(f[:-4] for f in os.listdir(UNID) if f.endswith(".npz"))

    def gen():
        for si, sc in enumerate(scenes):
            if si % 50 == 0:
                print(f"    {si}/{len(scenes)}", flush=True)
            gt = np.loadtxt(f"{AP.GTDIR}/{sc}.txt", dtype=np.int64)
            z = np.load(f"{UNID}/{sc}.npz")
            uidx = z["uidx"].astype(np.int64)
            d_ = torch.load(f"{SRC}/{sc}.pth", map_location="cpu", weights_only=False)
            D = np.zeros((len(uidx), len(OCAT)), np.float32)
            np.maximum.at(D, (z["urows"].astype(np.int64),
                              np.asarray(d_["final_class"], np.int64)),
                          np.asarray(d_["conf"], np.float32))
            F2 = np.zeros((len(uidx), 200), np.float32)
            F2[:, O2V] = D
            ZU = SN.l2(z["zuni"].astype(np.float32))
            S = (1 - w) * ctr(F2) + w * ctr((ZU @ TA.T).astype(np.float32))
            P = np.stack([rle_decode(d_["ins"][int(i)]) for i in uidx], 1).astype(bool)
            if P.shape[0] != len(gt):
                continue
            rows = np.arange(len(uidx))
            ent = top1(S, rows) if rule == "top1" else flatten(S, rows, topk)
            pred = []
            for r, c in ent:
                raw = int(IDARR[c])
                if raw not in id2idx:
                    continue
                pm = P[:, r]
                n = int(pm.sum())
                if n >= MIN_REGION:
                    pred.append((id2idx[raw], pm, n))
            ids = np.unique(gt); ids = ids[ids >= 1000]
            keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
            yield (pred, [gt == i for i, _ in keep], [c for _, c in keep],
                   ~np.isin(gt // 1000, ids198))
            del P, D, F2, S

    return summarise(sweep(gen(), len(names)), names, H, C, T)


def run_spp_oy(source, w, rule, topk=600):
    """Open-YOLO3D on ScanNet++, with both channels rebuilt from the stage-B dumps.

    These rows are produced from the dumps because, unlike the Open3DIS rows, no written prediction
    directory exists for this setting.
    """
    import pickle
    os.environ.setdefault("SPP_CLASSES", "100")
    from utils import ap_scannetpp as E
    K = len(E.NAMES)
    valid = np.arange(K) + 1
    VOTES = f"{FEATS}/yolovotes_scannetpp"
    De = pickle.load(open(dump("spp_stageB_dump.pkl"), "rb"))
    Dc = pickle.load(open(dump("spp_stageB_dump_clipL.pkl"), "rb"))
    l2 = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    A = l2(De[0]["anchors"].astype(np.float32))

    def gen():
        for i, (se, sc_) in enumerate(zip(De, Dc)):
            if i % 10 == 0:
                print(f"    {i}/{len(De)}", flush=True)
            sc = se["scene"]
            gs, gi, _ = E.build_gt_auth(sc)
            npts = len(gs)
            ss = gs.astype(np.int64) - 16 + 1
            ss[ss < 0] = 0
            gts = ss * 1000 + (gi.astype(np.int64) + 1)
            gt_masks, gt_cls = [], []
            for gid in np.unique(gts):
                if gid == 0:
                    continue
                lab = int(gid // 1000)
                if lab in valid:
                    gt_masks.append(gts == gid)
                    gt_cls.append(lab - 1)
            void = ~np.in1d(gts // 1000, valid)
            n_p = len(se["props"])
            if source == "eva":
                S2 = l2(se["z2"].astype(np.float32)) @ A.T
            elif source == "clipL":
                S2 = l2(sc_["z2"].astype(np.float32)) @ l2(sc_["anchors"].astype(np.float32)).T
            else:                                          # OY's own `vote / max`
                v = np.load(f"{VOTES}/yolovotes_{sc}.npz")["votes"].astype(np.float32)
                v = v[:, :A.shape[0]]
                if v.shape[0] < n_p:
                    v = np.vstack([v, np.zeros((n_p - v.shape[0], v.shape[1]), np.float32)])
                v = v[:n_p]
                S2 = v / np.maximum(v.max(1, keepdims=True), 1.0)
            S3 = l2(se["z3"].astype(np.float32)) @ A.T
            S = (1 - w) * ctr(S2.astype(np.float32)) + w * ctr(S3.astype(np.float32))
            rows = np.arange(n_p)
            ent = top1(S, rows) if rule == "top1" else flatten_raw(S, rows, topk)
            pred = []
            for r, c in ent:
                pm = np.zeros(npts, bool)
                pm[se["props"][r]] = True
                n = int(pm.sum())
                if n >= MIN_REGION:
                    pred.append((int(c) + E._PSHIFT, pm, n))   # 84 names sit at 16..99
            yield pred, gt_masks, gt_cls, void

    ap = sweep(gen(), K)
    i50 = list(IOUS).index(0.5)
    return dict(ap=100 * np.nanmean(ap[:len(IOUS) - 1]), ap50=100 * np.nanmean(ap[i50]),
                ap25=100 * np.nanmean(ap[-1]))


def run_replica_om(w, rule, topk=600):
    """OpenMask3D on Replica: same shape as run_sn200_om against the 48-class evaluator space."""
    from utils import replica as RU
    names, ids48 = _replica_eval_globals()
    id2idx = {int(v): i for i, v in enumerate(ids48)}
    K = len(names)
    D = np.load(f"{RU.FEATS}/replica_dump.npz", allow_pickle=True)
    us, uc = D["scene"].astype(str), D["col"].astype(np.int64)
    zuni = RU.l2(D["zuni"].astype(np.float32))
    FEAT = f"{OM_REPRO}/replica_features"
    have = [s for s in sorted(set(us.tolist())) if os.path.exists(f"{FEAT}/{s}.npy")]
    Z = np.zeros((len(us), 768), np.float32)
    for s in have:
        r_ = np.where(us == s)[0]
        f = np.load(f"{FEAT}/{s}.npy")
        n = min(len(r_), len(f))
        Z[r_[:n]] = f[:n]
    TAr = np.load(f"{FUSION}/text_clipL_om_single_replica.npy")
    EV = RU.l2(np.load(f"{RU.FEATS}/eva_text_48.npy").astype(np.float32))
    S2 = (RU.l2(Z) @ TAr.T).astype(np.float32)
    S2[~np.isin(us, have)] = -1.0
    S = (1 - w) * ctr(S2) + w * ctr((zuni @ EV.T).astype(np.float32))
    return _replica_score(S, us, uc, RU, ids48, id2idx, K, rule, topk)


def _replica_score(S, us, uc, RU, ids48, id2idx, K, rule, topk, masks_of=None, flat_fn=flatten):
    """Score one Replica arm. ``flat_fn`` selects the flattening rule, which is a property of the
    pipeline: Open-YOLO3D and OpenMask3D rank max-normalised scores, Open3DIS ranks raw ones."""
    def gen():
        for s in sorted(set(us.tolist())):
            rows = np.where(us == s)[0]
            gt = np.loadtxt(f"{RU.GTDIR}/{s}.txt", dtype=np.int64)
            m = RU.get_masks(s) if masks_of is None else masks_of(s)
            if m.shape[0] != len(gt):
                RU.MASKS.pop(s, None)
                continue
            ids = np.unique(gt); ids = ids[ids >= 1000]
            keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
            ent = top1(S, rows) if rule == "top1" else flat_fn(S, rows, topk)
            pred = []
            for r, c in ent:
                pm = m[:, uc[r]]
                n = int(pm.sum())
                if n >= MIN_REGION:
                    pred.append((c, pm, n))
            yield (pred, [gt == i for i, _ in keep], [c for _, c in keep],
                   ~np.isin(gt // 1000, ids48))
            del m
            RU.MASKS.pop(s, None)

    ap = sweep(gen(), K)
    i50 = list(IOUS).index(0.5)
    return dict(ap=100 * np.nanmean(ap[:len(IOUS) - 1]), ap50=100 * np.nanmean(ap[i50]),
                ap25=100 * np.nanmean(ap[-1]))


def run_sn200_o3d_cosine(w, rule, topk=600):
    """Open3DIS on ScanNet200, over its own proposals in raw-cosine space.

    This is the daggered row of the paper: the released artefact carries a collapsed label rather
    than a score matrix, so the pipeline was re-run with its per-proposal features dumped. The
    columns are already the evaluator's 198 instance classes, so no class-space mapping is applied.
    """
    from utils import open3dis as OC
    from utils import scannet200_ap as AP
    from scannet200_splits import (HEAD_CATS_SCANNET_200 as HCAT,
                                   COMMON_CATS_SCANNET_200 as CCAT,
                                   TAIL_CATS_SCANNET_200 as TCAT)
    import eval_semantic_instance as OM
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            OM.evaluate({}, "/nonexistent", output_file=f"{TMPDIR}/u_probe.txt",
                        dataset="scannet200")
        except Exception:
            pass
    names = list(OM.CLASS_LABELS)
    ids198 = np.asarray(OM.VALID_CLASS_IDS)
    id2idx = {int(v): i for i, v in enumerate(ids198)}
    K = len(names)

    S2, S3, _, cov, rows, scenes = OC.sn200_cosine_state()
    S = S2.astype(np.float32).copy()
    S[cov] = (1 - w) * ctr(S2)[cov] + w * ctr(S3)[cov]

    def gen():
        for si, sc in enumerate(scenes):
            if si % 50 == 0:
                print(f"    {si}/{len(scenes)}", flush=True)
            M = OC.sn200_cosine_masks(sc)
            gt = np.loadtxt(f"{AP.GTDIR}/{sc}.txt", dtype=np.int64)
            ids = np.unique(gt); ids = ids[ids >= 1000]
            keep = [(int(i), id2idx[int(i) // 1000]) for i in ids if int(i) // 1000 in id2idx]
            r = rows[sc]
            ent = top1(S, r) if rule == "top1" else flatten_rowmax(S, r, topk)
            base = int(r[0])
            pred = []
            for rr, c in ent:
                pm = M[rr - base]
                n = int(pm.sum())
                if n >= MIN_REGION:
                    pred.append((c, pm, n))
            yield (pred, [gt == i for i, _ in keep], [c for _, c in keep],
                   ~np.isin(gt // 1000, ids198))

    return summarise(sweep(gen(), K), names, HCAT, CCAT, TCAT)


def run_spp_o3d(w, rule, topk=600):
    """Open3DIS on ScanNet++, written out as a submission and scored by the ScanNet++ evaluator."""
    from utils import open3dis as OC
    S2, S3, _, scene = OC.spp_state()
    S = ((1 - w) * S2 + w * S3).astype(np.float32)
    d = os.path.join(TMPDIR, f"sensefuse_spp_o3d_{rule}_{w:.4f}")
    OC.spp_write(S, scene, rule, d, topk=topk)
    return run_spp(d)


def run_replica_o3d(w, rule, topk=600):
    """Open3DIS on Replica, over its own proposals and its own dense class scores.

    The pipeline scores its hierarchical-agglomerative proposals followed by its 3D class-agnostic
    set rather than Open-YOLO3D's Mask3D masks, so both the masks and the score matrices come from
    ``utils.open3dis`` rather than from the shared Replica loader.
    """
    from utils import open3dis as OC
    from utils import replica as RU
    names, ids48 = _replica_eval_globals()
    id2idx = {int(v): i for i, v in enumerate(ids48)}
    K = len(names)
    S2, S3, _, us, _ = OC.replica_state()
    S = (1 - w) * ctr(S2.astype(np.float32)) + w * ctr(S3.astype(np.float32))
    uc = np.concatenate([np.arange(int((us == s).sum())) for s in sorted(set(us.tolist()))])
    # utils.open3dis stacks proposals first; the scorer indexes points first. Open3DIS's own
    # submission ranks the raw flattened matrix, without the per-mask normalisation the other two
    # pipelines apply, so the flattening rule is the ScanNet++ one here.
    return _replica_score(S, us, uc, RU, ids48, id2idx, K, rule, topk,
                          masks_of=lambda s: OC.replica_masks(s).T, flat_fn=flatten_raw)


def main(argv=None):
    """Delegate to the Table II driver, which owns the block list and the printed values.

    ``main.py --task ap`` reaches the scorer through here. The runners above are the machinery; the
    weights, the blocks and the values they are compared against live in ``scripts/tables/table2.py``
    so that there is one statement of what the table contains.
    """
    import runpy
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.argv = ["table2.py"] + list(argv or sys.argv[1:])
    runpy.run_path(os.path.join(root, "scripts", "tables", "table2.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
