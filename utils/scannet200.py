"""UNIFIED ScanNet200 Table-1 driver.

For a given labeling SOURCE, evaluates under TWO submission protocols:
  (a) native   -- that method's own published protocol
                  OY  : per-mask max-norm (vote/max), flatten top-600, conf=1.0
                  O3D : pointwise softmax(300*cos) -> mask mean (row-sum 1),
                        flatten top-300 (2D-only) / top-600 (3D+2D), conf=1.0
                  OM  : top-1 argmax, conf=1.0 (identical to (b) by construction)
  (b) top1     -- top-1 argmax + conf=1.0  (the fair cross-method protocol)

and at TWO fusion weights: w=0 (2D only) and w=w* (argmax labeling accuracy on
GT-matched masks).  Always emits the FULL metric set:
    mAP / AP50 / AP25 / head / common / tail.

All confidences are uniform 1.0 -- verified against each method's released code
(OpenMask3D `np.ones`, Open3DIS `conf = 1.0  # Same as OpenMask3D`,
 OpenYOLO3D `ones_like`, Any3DIS = Open3DIS head).

Usage:
  python sn200_unified.py --source eva --protocol native top1 --w 0 wstar
"""
import os, sys
from utils import openyolo3d_file  # noqa: E402
from utils.paths import *  # noqa: F403
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
import argparse, glob, io, contextlib, json, gc
import numpy as np, torch

from utils import scannet200_ap as AP
import scannet200_constants as C200
from scannet200_splits import (HEAD_CATS_SCANNET_200 as HCAT,
                               COMMON_CATS_SCANNET_200 as CCAT,
                               TAIL_CATS_SCANNET_200 as TCAT)

NAMES = list(C200.CLASS_LABELS_200)
IDARR = np.array(list(C200.VALID_CLASS_IDS_200))
# FEATS comes from utils.paths (SENSEFUSE_FEATS); re-exported for the modules that import it here.
OYDIR = f"{OPENYOLO3D}/output/scannet200/scannet200_masks"
CACHE = FUSION          # text anchors and ground-truth matchings, carried in assets/fusion
os.makedirs(CACHE, exist_ok=True)

# GT instances smaller than this are not scored by the official AP evaluator:
# scannetv2_inst_eval.py:88 drops them inside evaluate_matches, via
#     `gt["vert_count"] >= min_region_size`,   min_region_sizes = np.array([100]) at :43.
# The commented-out filter at :407 belongs to assign_boxes_for_scan, a different entry point;
# reading that one as "the evaluator does not filter GT" is what put this package's accuracy axis
# on a different GT population from its own AP table.  Def. 1 states no size condition, but
# tab:accuracy and tab:ap have to be about the same instances, so the evaluator decides.
MIN_GT_PTS = int(os.environ.get("MIN_GT_PTS", "100"))


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def maxn(S):
    """per-mask max-normalisation -- OpenYOLO3D's `vote / max` convention."""
    return S / np.maximum(S.max(1, keepdims=True), 1e-9)


def center(S):
    """row-centering -- the space the precision law is defined in (models.law).

    The law's w depends on the RELATIVE noise scale (sigma) of the two channels.
    Row max-normalisation rescales every row independently and therefore destroys
    exactly that quantity: on SN200/EVA the law reads 0.365 on centered scores
    (matching the empirical w*=0.37) but 0.758 on max-normed scores. So fusion is
    performed on centered scores and each method's submission normalisation is
    applied only afterwards, at ranking time.
    """
    return S - S.mean(1, keepdims=True)


def softmax_norm(S, T=300.0):
    """per-mask class posterior, softmax(T * score), row-sum 1.

    Needed for the objectness x semantic ranking: submit_norm() is a per-row min-max whose
    maximum is 1.0 for EVERY mask, so multiplying it by objectness makes the cross-mask
    ordering depend on objectness alone and the semantic term contributes nothing. A softmax
    posterior is comparable across masks, which is what makes closed-set
    `class_prob * mask_quality` ranking work.
    """
    Z = T * (S - S.max(1, keepdims=True))
    E = np.exp(Z)
    return E / np.maximum(E.sum(1, keepdims=True), 1e-12)


def submit_norm(S):
    """per-mask [0,1] normalisation used for the flatten-top-K ranking.

    min-max rather than /max because centered scores are signed; monotone per row,
    so it preserves each mask's label ordering and puts every mask's best class at
    1.0 -- the same shape as OpenYOLO3D's `vote / max`.
    """
    lo = S.min(1, keepdims=True)
    hi = S.max(1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


def loadcat(pat, keys):
    parts = {k: [] for k in keys}
    for f in sorted(glob.glob(pat)):
        d = np.load(f, allow_pickle=True)
        for k in keys:
            parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------- proposals
def load_index():
    """canonical proposal ordering = the Uni3D shard order (scene, col)."""
    U = loadcat(f"{FEATS}/oyuni_s*.npz", ["scene", "col", "zuni"])
    us = U["scene"].astype(str)
    uc = U["col"].astype(int)
    zuni = l2(U["zuni"].astype(np.float32))
    return us, uc, zuni


MASKS = {}
OBJ = {}
def get_masks(s):
    if s not in MASKS:
        m, o = torch.load(f"{OYDIR}/{s}.pt", map_location="cpu")
        MASKS[s] = m.numpy() > 0.5
        # Mask3D per-mask objectness = scores_per_query * mask_scores_per_image
        # (Mask3D __init__.py:231-236). Open-YOLO3D loads it as preds_3d[1] and uses it
        # for score-thresholding and NMS (utils/__init__.py:116-117) but then drops it at
        # submission time. It is the only quantity here that answers "is this a real
        # instance?", whereas every open-vocabulary score answers "which class?",
        # ranking actually needs.
        OBJ[s] = o.numpy().astype(np.float32)
    return MASKS[s]


def get_obj(s):
    get_masks(s)
    return OBJ[s]


# ---------------------------------------------------------------- GT match
def gt_match(us, uc):
    """per-proposal GT class index (0..199) via IoU>=0.5, else -1. Cached.

    GT instances under MIN_GT_PTS points are excluded, matching the AP evaluator (:88).
    """
    # ``assets/fusion/gtmatch_full.npz`` is this function's committed output, and is what every
    # accuracy measurement reads, so that only ScanNet200's ground truth and Open-YOLO3D's masks are
    # needed to rebuild it rather than to use it. It is keyed ``ycls`` for historical reasons.
    fp = f"{CACHE}/gtmatch_full.npz"
    if MIN_GT_PTS == 100 and os.path.exists(fp):
        d = np.load(fp)
        if len(d["ycls"]) == len(us):
            return d["ycls"].astype(np.int64)
    id2idx = {int(v): i for i, v in enumerate(IDARR)}
    y = np.full(len(us), -1, np.int64)
    scenes = sorted(set(us.tolist()))
    for si, s in enumerate(scenes):
        if si % 25 == 0:
            print(f"  gt_match {si}/{len(scenes)}", flush=True)
        rows = np.where(us == s)[0]
        gt = np.loadtxt(f"{AP.GTDIR}/{s}.txt", dtype=np.int64)
        m = get_masks(s)
        if m.shape[0] != len(gt):
            print(f"  !! {s} vertex mismatch {m.shape[0]} vs {len(gt)}", flush=True)
            continue
        inst_ids = np.unique(gt)
        inst_ids = inst_ids[inst_ids >= 1000]
        keep = [(int(i), id2idx[int(i) // 1000]) for i in inst_ids
                if int(i) // 1000 in id2idx and int((gt == i).sum()) >= MIN_GT_PTS]
        if not keep:
            del m
            MASKS.pop(s, None)
            continue
        G = np.stack([gt == iid for iid, _ in keep], 1).astype(np.float32)   # (V, I)
        gcls = np.array([ci for _, ci in keep], np.int64)
        P = m[:, uc[rows]].astype(np.float32)                                 # (V, R)
        inter = P.T @ G                                                       # (R, I)
        areas = P.sum(0)[:, None] + G.sum(0)[None, :] - inter
        iou = inter / np.maximum(areas, 1e-9)
        bi = iou.argmax(1)
        best = iou[np.arange(len(rows)), bi]
        hit = best >= 0.5
        y[rows[hit]] = gcls[bi[hit]]
        del m, G, P, inter, iou
        MASKS.pop(s, None)
    np.savez_compressed(fp, ycls=y)
    return y


# ---------------------------------------------------------------- scoring
T20 = ["{}", "a {}", "the {}", "a {} in a scene", "there is a {} in the room",
       "a photo of a {}", "a photo of the {}", "a photo of one {}",
       "a close-up photo of a {}", "a cropped photo of a {}",
       "a bright photo of a {}", "a dark photo of a {}", "a blurry photo of a {}",
       "a photo of a small {}", "a photo of a large {}", "a {} in a living room",
       "a {} in a bedroom", "a {} in an office", "an indoor photo of a {}",
       "a 3d render of a {}"]


def text_anchors(kind, dev="cuda"):
    fp = f"{CACHE}/text_{kind}.npy"
    if os.path.exists(fp):
        return np.load(fp)
    import open_clip, torch.nn.functional as F
    os.environ["HF_HUB_OFFLINE"] = "1"
    if kind == "eva":
        m, _, _ = open_clip.create_model_and_transforms(
            "EVA02-E-14-plus",
            pretrained=f"{UNI3D}/models/open_clip_pytorch_model.bin")
        tok = open_clip.get_tokenizer("EVA02-E-14-plus")
    else:
        m, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai")
        tok = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
    m.visual = None
    gc.collect()
    m = m.to(dev).eval()
    with torch.no_grad():
        T = F.normalize(torch.stack([
            F.normalize(m.encode_text(tok([t.format(n) for n in NAMES]).to(dev)).float(), dim=-1)
            for t in T20]).mean(0), dim=-1).cpu().numpy()
    del m
    torch.cuda.empty_cache()
    np.save(fp, T)
    return T


def raw_votes(us, uc):
    """Open-YOLO3D's vote counts, unnormalised, in the 200-class benchmark order.

    ``build_scores`` divides each row by its maximum, which is the pipeline's own convention. The
    Dirichlet rule is a statement about counts, and dividing by the row maximum discards the sample
    size the posterior is conditioned on, so that rule reads the counts through this function.
    """
    import yaml
    cfg = yaml.safe_load(open(openyolo3d_file("pretrained", "config_scannet200.yaml")))["network2d"]["text_prompts"]
    p2n = np.array([NAMES.index(p) for p in cfg])
    V = np.zeros((len(us), 200), np.float32)
    for s in sorted(set(us.tolist())):
        v = np.load(f"{FEATS}/yolovotes/yolovotes_{s}.npz")["votes"].astype(np.float64)
        rows = np.where(us == s)[0]
        for r in rows:
            if uc[r] < v.shape[0]:
                V[r, p2n] = v[uc[r], :len(p2n)]
    return V


def build_scores(source, us, uc, zuni):
    """returns (S2_raw, S3_raw) un-normalised cosine-like score matrices."""
    evat = text_anchors("eva")
    S3 = zuni @ evat.T                       # Uni3D shape-text channel
    if source == "eva":
        E = loadcat(f"{FEATS}/evaoy_s*.npz", ["scene", "col", "zeva", "hasclip"])
        k = {(E["scene"].astype(str)[i], int(E["col"][i])): i for i in range(len(E["scene"]))}
        sel = np.array([k[(s, c)] for s, c in zip(us, uc)])
        z = l2(E["zeva"].astype(np.float32)[sel])
        h = E["hasclip"].astype(bool)[sel]
        S2 = z @ evat.T
        S2[~h] = -1.0
    elif source == "clipL":
        D = loadcat(f"{FEATS}/s2doy_s*.npz", ["scene", "col", "zclipL", "hasclip"])
        k = {(D["scene"].astype(str)[i], int(D["col"][i])): i for i in range(len(D["scene"]))}
        sel = np.array([k[(s, c)] for s, c in zip(us, uc)])
        z = l2(D["zclipL"].astype(np.float32)[sel])
        h = D["hasclip"].astype(bool)[sel]
        S2 = z @ text_anchors("clipL").T
        S2[~h] = -1.0
    elif source == "votes":
        import yaml
        cfg = yaml.safe_load(open(openyolo3d_file("pretrained", "config_scannet200.yaml")))["network2d"]["text_prompts"]
        p2n = np.array([NAMES.index(p) for p in cfg])
        S2 = np.zeros((len(us), 200), np.float32)
        for s in sorted(set(us.tolist())):
            v = np.load(f"{FEATS}/yolovotes/yolovotes_{s}.npz")["votes"].astype(np.float64)
            v = v / np.maximum(v.max(1, keepdims=True), 1)   # OY: vote / max
            rows = np.where(us == s)[0]
            for r in rows:
                if uc[r] < v.shape[0]:
                    S2[r, p2n] = v[uc[r], :len(p2n)]
    else:
        raise ValueError(source)
    return S2.astype(np.float32), S3.astype(np.float32)


# ---------------------------------------------------------------- law / w*
def law_weight(S2, S3, y):
    """plug-in precision law w = (mu3/sig3^2) / (mu2/sig2^2 + mu3/sig3^2)."""
    from models.law import weight_from_stats, mu_sigma
    sel = y >= 0
    mu2, sig2 = mu_sigma(S2[sel], y[sel])
    mu3, sig3 = mu_sigma(S3[sel], y[sel])
    return float(weight_from_stats(mu2, sig2, mu3, sig3)), (mu2, sig2, mu3, sig3)


def acc_sweep(S2n, S3n, y, step=0.01):
    sel = y >= 0
    A, B, yy = S2n[sel], S3n[sel], y[sel]
    grid = np.round(np.arange(0, 1.0 + 1e-9, step), 4)
    acc = np.array([((1 - w) * A + w * B).argmax(1).__eq__(yy).mean() for w in grid])
    i = int(acc.argmax())
    lo = hi = i
    thr = acc[i] - 0.005
    while lo > 0 and acc[lo - 1] >= thr:
        lo -= 1
    while hi < len(grid) - 1 and acc[hi + 1] >= thr:
        hi += 1
    return dict(w_star=float(grid[i]), acc_max=float(acc[i]), acc_w0=float(acc[0]),
                plateau=[float(grid[lo]), float(grid[hi])],
                grid=grid.tolist(), acc=acc.tolist())


# ---------------------------------------------------------------- AP eval
def _agg(d, grp=None):
    g = set(grp) if grp is not None else None
    v = [d[n] for n in d if d[n] == d[n] and (g is None or n in g)]
    return float(np.mean(v) * 100) if v else float("nan")


class LazyPreds:
    """Streams one scene's predictions at a time.

    Holding all 312 scenes' (V x 600) bool masks at once needs ~28 GB, which is why
    the earlier version split the run into NGROUP class-passes and paid for 4 full
    AP evaluations. AP.evaluate only ever iterates preds.items(), so streaming gives
    the identical result in a single pass -- and lets us read all_ap straight from
    the evaluator instead of averaging per-class APs across passes.
    """
    def __init__(self, entries, uc, conf="one"):
        self.entries, self.uc, self.conf = entries, uc, conf
        self.scenes = sorted(entries.keys())

    def __len__(self):
        return len(self.scenes)

    def keys(self):
        return list(self.scenes)

    def items(self):
        for s in self.scenes:
            m = get_masks(s)
            ent = self.entries[s]
            # e[0] is the GLOBAL proposal row; the mask column is uc[row]
            cols = np.array([self.uc[e[0]] for e in ent], np.int64)
            cls = np.array([IDARR[e[1]] for e in ent], np.int64)
            if self.conf in ("objsem", "sem"):
                o = get_obj(s)
                sc = np.array([(o[self.uc[e[0]]] if self.conf == "objsem" else 1.0) * e[3]
                               for e in ent], np.float32)
            elif self.conf == "obj":
                o = get_obj(s)
                sc = np.array([o[self.uc[e[0]]] for e in ent], np.float32)
            elif self.conf == "score":
                # rank by the fused, submission-normalised score: the Mask3D
                # closed-set convention, as opposed to the uniform conf = 1.0
                # imposed by the open-vocabulary baselines
                sc = np.array([e[2] for e in ent], np.float32)
            else:
                sc = np.ones(len(cls), np.float32)              # conf = 1.0
            yield s, {
                "pred_masks": m[:, cols] if len(cols) else np.zeros((m.shape[0], 0), bool),
                "pred_scores": sc,
                "pred_classes": cls}
            # masks are cached (approximately 9 GB for all 312 scenes) so that the six weight
            # and protocol combinations do not each re-read 9.5 GB of .pt files from disk


def evaluate(entries, us, uc, tag, ngroup=None, conf="one"):
    """entries[scene] = list of (row, class_idx, score). conf: 'one' or 'score'."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = AP.evaluate(LazyPreds(entries, uc, conf), AP.GTDIR,
                        output_file=f"{TMPDIR}/u_{tag}.txt", dataset="scannet200")
    a = r[0]
    per = {n: a["classes"][n]["ap"] for n in a["classes"]}
    out = dict(ap=a["all_ap"] * 100, ap50=a["all_ap_50%"] * 100, ap25=a["all_ap_25%"] * 100,
               head=_agg(per, HCAT), common=_agg(per, CCAT), tail=_agg(per, TCAT))
    print(f"[{tag:26s}] mAP {out['ap']:5.1f}  AP50 {out['ap50']:5.1f}  AP25 {out['ap25']:5.1f}"
          f" | h {out['head']:5.1f}  c {out['common']:5.1f}  t {out['tail']:5.1f}", flush=True)
    gc.collect()
    return out


def entries_flatten(S, us, topk, P=None):
    """flatten (mask x class), keep top-K per scene -- OY / Open3DIS convention."""
    e = {}
    for s in sorted(set(us.tolist())):
        r = np.where(us == s)[0]
        Ds = S[r]
        flat = Ds.reshape(-1)
        k = min(topk, len(flat))
        idx = np.argpartition(-flat, k - 1)[:k]
        lst = []
        for ix in idx:
            rr, c = divmod(int(ix), Ds.shape[1])
            if flat[ix] > 0:
                lst.append((int(r[rr]), c, float(flat[ix]),
                            float(P[int(r[rr]), c]) if P is not None else float(flat[ix])))
        e[s] = lst
    return e


def entries_top1(S, us, S_conf=None, P=None):
    """one label per proposal -- OpenMask3D convention.

    S drives the argmax; S_conf (if given) supplies the confidence. They must differ:
    the flatten ranking uses submit_norm(), whose per-row max is exactly 1.0 by
    construction, so scoring top-1 with it makes every mask tie at 1.0 and conf='score'
    silently collapses onto conf=1.0. The un-normalised fused score is used instead.
    """
    lab = S.argmax(1)
    sc = (S if S_conf is None else S_conf).max(1)
    e = {}
    for s in sorted(set(us.tolist())):
        r = np.where(us == s)[0]
        e[s] = [(int(i), int(lab[i]), float(sc[i]),
                 float(P[i, lab[i]]) if P is not None else float(sc[i])) for i in r]
    return e


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["eva", "clipL", "votes"])
    ap.add_argument("--protocol", nargs="+", default=["native", "top1"])
    ap.add_argument("--w", nargs="+", default=["0", "law", "wstar"])
    ap.add_argument("--conf", nargs="+", default=["one"], choices=["one", "score", "obj", "sem", "objsem"])
    ap.add_argument("--space", default="centered", choices=["centered", "maxn"])
    ap.add_argument("--topk", type=int, default=600)
    ap.add_argument("--ngroup", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    us, uc, zuni = load_index()
    print(f"proposals {len(us)}  scenes {len(set(us.tolist()))}", flush=True)
    y = gt_match(us, uc)
    print(f"GT-matched {int((y >= 0).sum())} / {len(y)}", flush=True)

    S2, S3 = build_scores(args.source, us, uc, zuni)
    # fusion space: row-centered (law-consistent) unless explicitly overridden
    if args.space == "centered":
        F2, F3 = center(S2), center(S3)
    else:
        F2, F3 = maxn(S2), maxn(S3)

    wl_raw, stats = law_weight(S2, S3, y)
    wl_f, _ = law_weight(F2, F3, y)
    sw = acc_sweep(F2, F3, y)
    print(f"space={args.space}  w_law(raw) {wl_raw:.3f}  w_law({args.space}) {wl_f:.3f}  "
          f"w*(acc) {sw['w_star']:.2f}  acc {sw['acc_w0']:.4f}->{sw['acc_max']:.4f} "
          f"plateau {sw['plateau']}", flush=True)

    res = {"_meta": dict(source=args.source, space=args.space, w_law_raw=wl_raw,
                         w_law_space=wl_f, w_star=sw["w_star"], acc_w0=sw["acc_w0"],
                         acc_max=sw["acc_max"], plateau=sw["plateau"],
                         n_matched=int((y >= 0).sum()),
                         mu2=stats[0], sig2=stats[1], mu3=stats[2], sig3=stats[3]),
           "_accsweep": {"grid": sw["grid"], "acc": sw["acc"]}}

    for wtag in args.w:
        w = sw["w_star"] if wtag == "wstar" else (wl_f if wtag == "law" else float(wtag))
        F = F2 if w == 0 else (1 - w) * F2 + w * F3
        for proto in args.protocol:
            # submission normalisation applied AFTER fusion; top-1 is argmax-invariant to it
            Fp = softmax_norm(F) if any(c in ("sem", "objsem") for c in args.conf) else None
            ent = (entries_flatten(submit_norm(F), us, args.topk, P=Fp) if proto == "native"
                   else entries_top1(submit_norm(F), us, S_conf=F, P=Fp))
            for cf in args.conf:
                tag = f"{args.source}_{proto}_{cf}_w{w:.2f}"
                res[tag] = evaluate(ent, us, uc, tag, args.ngroup, conf=cf)
                res[tag]["w"] = w
                res[tag]["conf"] = cf
    out = args.out or dump(f"u_sn200_{args.source}_{args.space}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
