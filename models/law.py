"""Estimators and analyses for two-head score fusion.

Everything operates on matched-mask score matrices (row-centered cosines).
The theory: under the signal+noise Gaussian model the Bayes-optimal fusion is
S = (1-w)*S2 + w*Sw with w = (mu_w/sig_w^2) / (mu_w/sig_w^2 + mu_2/sig_2^2).
"""
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- population stats

def mu_sigma(S, y):
    """Signal and noise of one head on matched masks.

    mu    : mean over masks of (correct-class score - mean wrong-class score)
    sigma : mean over masks of the within-row std of wrong-class scores
    """
    n = len(y)
    rows = np.arange(n)
    s_correct = S[rows, y]
    mean_wrong = (S.sum(1) - s_correct) / (S.shape[1] - 1)
    mu = float((s_correct - mean_wrong).mean())

    wrong = S.copy()
    wrong[rows, y] = np.nan
    sigma = float(np.nanstd(wrong, axis=1).mean())
    return mu, sigma


def weight_from_stats(mu2, sig2, muw, sigw):
    """Diag precision law: w_h proportional to mu_h / sigma_h^2 (weak head's share)."""
    prec = np.maximum([mu2 / sig2 ** 2, muw / sigw ** 2], 0)
    return float(prec[1] / (prec[0] + prec[1] + 1e-12))


def predict_weight(S2, Sw, y):
    """Weight predicted by the diag law from GT-matched masks."""
    return weight_from_stats(*mu_sigma(S2, y), *mu_sigma(Sw, y))


# ---------------------------------------------------------------- sweeps

def fuse(S2, Sw, w):
    return (1 - w) * S2 + w * Sw


def accuracy(S, y):
    return float((S.argmax(1) == y).mean())


def sweep(S2, Sw, y, step=0.005):
    """Accuracy over the full weight grid. Returns (grid, acc)."""
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    acc = np.array([accuracy(fuse(S2, Sw, w), y) for w in grid])
    return grid, acc


def bootstrap_wstar(S2, Sw, y, scenes, step=0.001, n_boot=1000, seed=0):
    """w* on a fine grid plus a scene-level bootstrap 95% CI and 0.5pt plateau."""
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    correct = np.stack([(fuse(S2, Sw, w).argmax(1) == y) for w in grid])
    acc = correct.mean(1)
    w_star = float(grid[acc.argmax()])
    plateau = grid[acc >= acc.max() - 0.005]

    uniq = np.unique(scenes)
    by_scene = {s: np.where(scenes == s)[0] for s in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([by_scene[uniq[p]] for p in pick])
        boots.append(float(grid[correct[:, idx].mean(1).argmax()]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(w_star=w_star, acc_max=float(acc.max()),
                ci95=[float(lo), float(hi)],
                plateau=[float(plateau[0]), float(plateau[-1])],
                grid=grid, acc=acc)


# ---------------------------------------------------------------- label-free (EM)

def em_weight(S2, Sw, w0=0.25, iters=6, confidence_quantile=0.5):
    """Zero-label weight estimation.

    The iteration proceeds as follows: fuse at ``w``, take pseudo-labels from the fused argmax
    while retaining only the top ``confidence_quantile`` fraction of masks by margin, re-estimate
    ``(mu, sigma)``, and form a new ``w``. Convergence is reached in approximately four iterations,
    and the weight obtained from ground-truth labels is recovered without using any label.
    """
    w = w0
    trajectory = [w]
    for _ in range(iters):
        S = fuse(S2, Sw, w)
        pseudo = S.argmax(1)
        if confidence_quantile is not None:
            top2 = np.partition(S, -2, 1)
            margin = top2[:, -1] - top2[:, -2]
            keep = margin >= np.quantile(margin, 1 - confidence_quantile)
        else:
            keep = np.ones(len(pseudo), bool)
        w = weight_from_stats(*mu_sigma(S2[keep], pseudo[keep]),
                              *mu_sigma(Sw[keep], pseudo[keep]))
        trajectory.append(round(w, 3))
    return w, trajectory


# ---------------------------------------------------------------- class-wise weights

def classwise_weights(S2, Sw, y, K, w_global, min_count=8, shrink_count=50):
    """Per-class precision weights with James-Stein-style shrinkage to the global w.

    w_c from (mu_c, sigma_c) of masks of class c; classes with fewer than
    `min_count` masks fall back to the global weight; the rest are shrunk with
    lambda_c = n_c / (n_c + shrink_count).
    """
    w_raw = np.full(K, w_global)
    counts = np.zeros(K)
    for c in set(y.tolist()):
        m = y == c
        if m.sum() < min_count:
            continue
        mu2, sig2 = mu_sigma(S2[m], y[m])
        muw, sigw = mu_sigma(Sw[m], y[m])
        if sig2 > 0 and sigw > 0 and (mu2 > 0 or muw > 0):
            w_raw[c] = weight_from_stats(mu2, sig2, muw, sigw)
            counts[c] = m.sum()
    if shrink_count is None:                 # per-class weights before shrinkage
        return w_raw
    lam = counts / (counts + shrink_count)
    return lam * w_raw + (1 - lam) * w_global


def fuse_classwise(S2, Sw, w_per_class):
    W = w_per_class[None, :]
    return (1 - W) * S2 + W * Sw


# ---------------------------------------------------------------- diagnostics

def margin_gaussian_check(S2, Sw, y, weights=np.arange(0, 1.001, 0.05)):
    """Predict acc(w) from a central-Gaussian fit of the margin distribution.

    The decision statistic is M = s_correct - max wrong; acc = P(M > 0) exactly.
    Fit N(median, (IQR/1.349)^2) on M's central region and compare Phi(median/sigma)
    against the empirical accuracy across the sweep.
    """
    rows = np.arange(len(y))
    emp, pred = [], []
    for w in weights:
        S = fuse(S2, Sw, w)
        s_correct = S[rows, y]
        S_wrong = S.copy()
        S_wrong[rows, y] = -1e9
        margin = s_correct - S_wrong.max(1)
        med = float(np.median(margin))
        sig = float((np.quantile(margin, 0.75) - np.quantile(margin, 0.25)) / 1.349)
        emp.append(float((margin > 0).mean()))
        pred.append(float(norm.cdf(med / sig)))
    emp, pred = np.array(emp), np.array(pred)
    return dict(weights=np.array(weights), acc_emp=emp, acc_pred=pred,
                max_gap=float(np.abs(emp - pred).max()),
                wstar_emp=float(weights[emp.argmax()]),
                wstar_pred=float(weights[pred.argmax()]))


def plugin_mi_bits(S, y, K):
    """Plug-in mutual information (bits) between the head's argmax and the label."""
    pred = S.argmax(1)
    cm = np.zeros((K, K))
    for a, b in zip(pred, y):
        cm[a, b] += 1
    p = cm / len(y)
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(p * (np.log2(p) - np.log2(px) - np.log2(py))))


# ---------------------------------------------------------------- fusion gate

def orthogonal_component(S_strong, S_weak, labels):
    """d'_perp of the weak head after orthogonalizing against the strong head's evidence.

    d'_fused^2 = d'_strong^2 + d'_perp^2 with d'_perp = (d'_w - rho*d'_s)/sqrt(1-rho^2).
    A value of d'_perp near zero indicates that the weak head carries no evidence absent from the
    strong head, in which case fusion cannot improve accuracy.
    """
    n = len(labels)
    rows = np.arange(n)

    def dprime(S):
        s_correct = S[rows, labels]
        mean_wrong = (S.sum(1) - s_correct) / (S.shape[1] - 1)
        E = S.copy()
        E[rows, labels] = np.nan
        return float((s_correct - mean_wrong).mean() / np.nanstd(E, 1).mean())

    def noise(S):
        E = S.copy()
        E[rows, labels] = np.nan
        return E - np.nanmean(E, 1, keepdims=True)

    e1, e2 = noise(S_strong), noise(S_weak)
    ok = ~np.isnan(e1) & ~np.isnan(e2)
    rho = float((e1[ok] * e2[ok]).mean() / (np.nanstd(e1) * np.nanstd(e2)))
    d_s, d_w = dprime(S_strong), dprime(S_weak)
    return (d_w - rho * d_s) / np.sqrt(max(1 - rho ** 2, 1e-9)), rho


def should_fuse(S_strong, S_weak, threshold=0.1):
    """Label-free fusion gate: pseudo-label with the strong head and measure d'_perp.

    Pseudo-labelling deflates d'_perp by approximately a factor of five relative to ground-truth
    labels, but preserves the separation between settings: heads sharing an encoder (S3DIS
    prototype and shape-text) yield 0.007, whereas genuinely complementary heads (ScanNet 2D and
    shape) yield 0.248, a ratio of 35. A threshold of 0.1 separates the two regimes. The gate is
    evaluated before ``em_weight``; a negative result indicates that the strong head should be used
    alone.
    """
    pseudo = S_strong.argmax(1)
    d_perp, rho = orthogonal_component(S_strong, S_weak, pseudo)
    return bool(d_perp > threshold), float(d_perp), rho
