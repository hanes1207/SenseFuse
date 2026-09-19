"""Replica score matrices and instance-AP evaluation, over Open-YOLO3D's released Mask3D proposals.

The submission protocol was verified in the official repository (aminebdj/OpenYOLO3D):
``pretrained/config_replica.yaml`` sets ``topk_per_image: -1``, and ``utils/__init__.py`` line 243
gates the flattening branch on ``topk_per_image != -1``. On Replica, Open-YOLO3D therefore does not
flatten a top-600 list -- that is the ScanNet200 setting, where ``topk_per_image: 600`` -- but emits
one label per mask, with ``conf = torch.ones_like`` (``run_evaluation.py`` line 57).

Consequently the method's own protocol on Replica is top-1 with conf = 1.0, and the native and
cross-method rows coincide by construction. ``--topk -1`` reproduces that protocol; ``--topk 600`` is
retained only as a deliberate off-protocol probe.

Fusion is performed on row-centred cosines, the space in which the precision law is defined, and each
method's submission normalisation is applied only afterwards, at ranking time. AP is reported at
w = 0 and at w*, the weight maximising labelling accuracy over the ground-truth-matched masks.

The Replica benchmark comprises 48 flat classes without a head, common and tail split, so only mAP,
AP50 and AP25 are emitted, matching the Replica columns of Table I.
"""
import os, sys
from utils import openyolo3d_file  # noqa: E402
from utils.paths import *  # noqa: F403
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
import argparse, io, contextlib, json, os, sys
import numpy as np, torch

from evaluate.replica import eval_semantic_instance as REV   # from the OPENYOLO3D_ROOT checkout

VOTES = os.path.join(FEATS, "yolovotes_replica")
FEATS = os.path.join(FEATS, "replica")            # this dataset's dump dir under paths.FEATS
MASKD = f"{OPENYOLO3D}/output/replica/replica_masks"
GTDIR = f"{OPENYOLO3D}/replica/ground_truth"
NAMES = list(REV.CLASS_LABELS)
IDARR = np.array(list(REV.VALID_CLASS_IDS))
NC = len(NAMES)
# see utils.scannet200.MIN_GT_PTS; scannetv2_inst_eval.py line 88, min_region_sizes = [100].
# The effect is largest on Replica: 38 of its 427 ground-truth instances (8.9 %) fall below 100
# 1.5% on ScanNet200.
MIN_GT_PTS = int(os.environ.get("MIN_GT_PTS", "100"))


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def center(S):
    return S - S.mean(1, keepdims=True)


def submit_norm(S):
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


MASKS = {}
def get_masks(s):
    if s not in MASKS:
        m = torch.load(f"{MASKD}/{s}.pt", map_location="cpu")
        m = m[0] if isinstance(m, (list, tuple)) else m
        MASKS[s] = m.numpy() > 0.5
    return MASKS[s]


def gt_match(us, uc):
    id2idx = {int(v): i for i, v in enumerate(IDARR)}
    y = np.full(len(us), -1, np.int64)
    for s in sorted(set(us.tolist())):
        rows = np.where(us == s)[0]
        gt = np.loadtxt(f"{GTDIR}/{s}.txt", dtype=np.int64)
        m = get_masks(s)
        if m.shape[0] != len(gt):
            print(f"  !! {s} vertex mismatch {m.shape[0]} vs {len(gt)}", flush=True)
            continue
        iids = np.unique(gt)
        iids = iids[iids >= 1000]
        keep = [(int(i), id2idx[int(i) // 1000]) for i in iids
                if int(i) // 1000 in id2idx and int((gt == i).sum()) >= MIN_GT_PTS]
        if not keep:
            continue
        G = np.stack([gt == i for i, _ in keep], 1).astype(np.float32)
        gcls = np.array([c for _, c in keep], np.int64)
        P = m[:, uc[rows]].astype(np.float32)
        inter = P.T @ G
        union = P.sum(0)[:, None] + G.sum(0)[None, :] - inter
        iou = inter / np.maximum(union, 1e-9)
        bi = iou.argmax(1)
        best = iou[np.arange(len(rows)), bi]
        hit = best >= 0.5
        y[rows[hit]] = gcls[bi[hit]]
    return y


def build_scores(source, us, uc, zuni):
    evat = l2(np.load(f"{FEATS}/eva_text_48.npy").astype(np.float32))
    S3 = zuni @ evat.T
    D = np.load(f"{FEATS}/replica_dump.npz", allow_pickle=True)
    E = np.load(f"{FEATS}/replica_eva_crops.npz", allow_pickle=True)
    if source == "eva":
        S2 = l2(E["zeva"].astype(np.float32)) @ evat.T
        S2[~E["hv"].astype(bool)] = -1.0
    elif source == "clipL":
        ct = l2(np.load(f"{FEATS}/clipL_text_48.npy").astype(np.float32))
        S2 = l2(D["zclipL"].astype(np.float32)) @ ct.T
        S2[~D["hasclip"].astype(bool)] = -1.0
    elif source == "votes":
        import yaml
        cfg = yaml.safe_load(open(openyolo3d_file("pretrained", "config_replica.yaml")))["network2d"]["text_prompts"]
        p2n = np.array([NAMES.index(p) for p in cfg if p in NAMES])
        keepc = np.array([i for i, p in enumerate(cfg) if p in NAMES])
        S2 = np.zeros((len(us), NC), np.float32)
        for s in sorted(set(us.tolist())):
            v = np.load(f"{VOTES}/yolovotes_{s}.npz")["votes"].astype(np.float64)
            v = v / np.maximum(v.max(1, keepdims=True), 1)      # OY: vote / max
            for r in np.where(us == s)[0]:
                if uc[r] < v.shape[0]:
                    S2[r, p2n] = v[uc[r], keepc]
    else:
        raise ValueError(source)
    return S2.astype(np.float32), S3.astype(np.float32)


def evaluate(entries, us, uc, tag):
    preds = {}
    for s in sorted(set(us.tolist())):
        m = get_masks(s)
        cols, cls = [], []
        for row, c in entries[s]:
            cols.append(m[:, uc[row]])
            # Replica's evaluator indexes PRED_ID_TO_ID by a zero-based class
            # index rather than by a raw ScanNet id (eval_semantic_instance.py
            # lines 71 to 78).
            cls.append(int(c))
        preds[s] = {"pred_masks": np.stack(cols, 1) if cols else np.zeros((m.shape[0], 0), bool),
                    "pred_scores": np.ones(len(cls), np.float32),      # conf = 1.0
                    "pred_classes": np.array(cls, np.int64)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = REV.evaluate(preds, GTDIR, output_file=f"{TMPDIR}/rep_{tag}.txt", dataset="replica")
    a = r[0] if isinstance(r, tuple) else r
    out = dict(ap=a["all_ap"] * 100, ap50=a["all_ap_50%"] * 100, ap25=a["all_ap_25%"] * 100)
    print(f"[{tag:26s}] mAP {out['ap']:5.1f}  AP50 {out['ap50']:5.1f}  AP25 {out['ap25']:5.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["eva", "clipL", "votes"])
    ap.add_argument("--protocol", nargs="+", default=["native", "top1"])
    ap.add_argument("--w", nargs="+", default=["0", "law", "wstar"])
    ap.add_argument("--topk", type=int, default=-1)   # Replica: topk_per_image = -1
    args = ap.parse_args()

    D = np.load(f"{FEATS}/replica_dump.npz", allow_pickle=True)
    us = D["scene"].astype(str)
    uc = D["col"].astype(int)
    zuni = l2(D["zuni"].astype(np.float32))
    y = gt_match(us, uc)
    print(f"proposals {len(us)}  matched {int((y >= 0).sum())}", flush=True)

    S2, S3 = build_scores(args.source, us, uc, zuni)
    F2, F3 = center(S2), center(S3)
    from models.law import weight_from_stats, mu_sigma
    sel = y >= 0
    mu2, sig2 = mu_sigma(F2[sel], y[sel])
    mu3, sig3 = mu_sigma(F3[sel], y[sel])
    w_law = float(weight_from_stats(mu2, sig2, mu3, sig3))
    grid = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 4)
    A, B, yy = F2[sel], F3[sel], y[sel]
    acc = np.array([((1 - w) * A + w * B).argmax(1).__eq__(yy).mean() for w in grid])
    i = int(acc.argmax()); thr = acc[i] - 0.005
    lo = hi = i
    while lo > 0 and acc[lo - 1] >= thr: lo -= 1
    while hi < len(grid) - 1 and acc[hi + 1] >= thr: hi += 1
    wstar = float(grid[i])
    print(f"w_law {w_law:.3f}  w* {wstar:.2f}  acc {acc[0]:.4f}->{acc[i]:.4f} "
          f"acc@law {np.interp(w_law, grid, acc):.4f}  plateau [{grid[lo]:.2f},{grid[hi]:.2f}]", flush=True)

    res = {"_meta": dict(source=args.source, dataset="replica", w_law=w_law, w_star=wstar,
                         acc_w0=float(acc[0]), acc_max=float(acc[i]),
                         acc_at_law=float(np.interp(w_law, grid, acc)),
                         plateau=[float(grid[lo]), float(grid[hi])],
                         n_prop=int(len(y)), n_matched=int(sel.sum())),
           "_accsweep": {"grid": grid.tolist(), "acc": acc.tolist()}}
    for wtag in args.w:
        w = wstar if wtag == "wstar" else (w_law if wtag == "law" else float(wtag))
        F = F2 if w == 0 else (1 - w) * F2 + w * F3
        for proto in args.protocol:
            tag = f"{args.source}_{proto}_w{w:.2f}"
            if proto == "native" and args.topk > 0:
                Sn = submit_norm(F)
                ent = {}
                for s in sorted(set(us.tolist())):
                    r = np.where(us == s)[0]
                    Ds = Sn[r]; flat = Ds.reshape(-1)
                    k = min(args.topk, len(flat))
                    idx = np.argpartition(-flat, k - 1)[:k]
                    ent[s] = [(int(r[ix // Ds.shape[1]]), int(ix % Ds.shape[1]))
                              for ix in idx if flat[ix] > 0]
            else:
                lab = F.argmax(1)
                ent = {s: [(int(i2), int(lab[i2])) for i2 in np.where(us == s)[0]]
                       for s in sorted(set(us.tolist()))}
            res[tag] = evaluate(ent, us, uc, tag)
            res[tag]["w"] = w
    out = f"{SCANNETPP_DUMPS}/u_replica_{args.source}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
