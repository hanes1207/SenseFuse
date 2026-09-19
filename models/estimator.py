"""Algorithm 1: the soft (pre-argmax) relaxation of the precision-weight estimator.

The estimator replaces the two discontinuities of the EM loop -- the argmax that supplies
pseudo-labels and the median-gap threshold that supplies the confident subset -- with a softmax and a
sigmoid respectively. The update map is thereby continuous in ``w``, so Brouwer's theorem applies and
multi-start initialisation converges to a single point.

Two arms are provided:

    soft_em(S2, S3)   the deployed estimator: w0 = 0.5, |dw| < 1e-3, at most 12 iterations
    hard_em(S2, S3)   the hard pseudo-label rule, retained as the control arm

The deployed stopping rule is the one under which every reported number is measured. A convergence
study conducted at tol 1e-6 with a cap of 200 iterations is reported in the paper; values obtained
under that rule differ in the third decimal of ``w`` and are not those printed in the tables.

Both arms require already-centred score rows (see ``ctr``), which is the space in which Table I is
measured: centring only, without row standardisation, and fitted over every proposal rather than over
the ground-truth-matched subset.

Executed as a script, the module performs a wiring check and is expected to report 52.058 for the
soft arm and 52.073 for the hard arm, with a six-start spread exceeding 0.01 in 11 scenes for the
soft arm and 78 for the hard arm, on ScanNet200 with Open-YOLO3D (EVA02-E) and Uni3D.
"""
import os, sys
from utils.paths import *  # noqa: F403

import numpy as np

from models import law as stats

# ---- the deployed stopping rule. These constants are fixed and are not exposed as arguments.
W0 = 0.5
TOL = 1e-3
RMAX = 12
TAU0 = 0.25
BETA = 8.0
STARTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def ctr(S):
    return S - S.mean(1, keepdims=True)


def w_from(mu2, sd2, mu3, sd3):
    """``models.law.weight_from_stats``, written out so that the two arms cannot diverge."""
    p2 = max(mu2 / (sd2 ** 2), 0.0) if sd2 > 1e-12 else 0.0
    p3 = max(mu3 / (sd3 ** 2), 0.0) if sd3 > 1e-12 else 0.0
    return float(p3 / (p2 + p3 + 1e-12))


def soft_update(S2, S3, w, tau0=TAU0, beta=BETA, with_moments=False):
    """One continuous EM step. This is the sole implementation of Algorithm 1's update.

    When ``with_moments`` is set, the function additionally returns
    ``[(mu_2, sd_2), (mu_3, sd_3)]``, the plug-in moments from which the weight was formed; these
    are the quantities plotted in the iteration trace of Fig. 3. The flag does not affect any
    computed value.
    """
    F = (1 - w) * S2 + w * S3
    scale = F.std(1, keepdims=True) + 1e-12
    P = np.exp((F - F.max(1, keepdims=True)) / (tau0 * scale))
    P /= P.sum(1, keepdims=True)

    q = np.sort(F, axis=1)
    gap = q[:, -1] - q[:, -2]
    g = 1.0 / (1.0 + np.exp(-beta * (gap - np.median(gap)) / (gap.std() + 1e-12)))
    g = g / (g.sum() + 1e-12)

    K = F.shape[1]
    out = []
    for S in (S2, S3):
        sig = (P * S).sum(1)
        mu = float((g * (sig * K / (K - 1.0))).sum())
        Q = (1.0 - P)
        Q = Q / Q.sum(1, keepdims=True)
        mean_bg = (Q * S).sum(1, keepdims=True)
        var_bg = (Q * (S - mean_bg) ** 2).sum(1)
        sd = float((g * np.sqrt(np.maximum(var_bg, 0.0))).sum())
        out += [mu, sd]
    nw = w_from(out[0], out[1], out[2], out[3])
    if with_moments:
        return nw, [(out[0], out[1]), (out[2], out[3])]
    return nw


def soft_em(S2, S3, w0=W0, tol=TOL, iters=RMAX, tau0=TAU0, beta=BETA):
    """Algorithm 1 with the soft update, under the deployed stopping rule.

    ``tau0`` and ``beta`` are the relaxation's two constants and are fixed at their defaults for
    every reported number; they are arguments only so that the sweep of Section IV-D can move them.
    """
    w = w0
    for _ in range(iters):
        nw = soft_update(S2, S3, w, tau0=tau0, beta=beta)
        if not np.isfinite(nw):
            return w
        if abs(nw - w) < tol:
            return nw
        w = nw
    return w


def hard_em(S2, S3, w0=W0, tol=TOL, iters=RMAX):
    """The hard pseudo-label rule, retained as the control arm for every soft measurement."""
    w = w0
    for _ in range(iters):
        F = (1 - w) * S2 + w * S3
        lab = F.argmax(1)
        p = np.partition(F, -2, axis=1)
        keep = (p[:, -1] - p[:, -2]) >= np.median(p[:, -1] - p[:, -2])
        if keep.sum() < 3:
            keep = np.ones(len(F), bool)
        nw = float(stats.predict_weight(S2[keep], S3[keep], lab[keep]))
        if not np.isfinite(nw):
            return w
        if abs(nw - w) < tol:
            return nw
        w = nw
    return w


# ------------------------------------------------------------------ the correlation correction
#
# Equation (5) assumes the two heads' background score residuals are uncorrelated. The ablation asks
# what happens if that assumption is dropped and the residual correlation is estimated and corrected
# for instead. The correction is read off the SAME responsibilities, background weights and subset
# weights as the moments, so the two arms differ only in which formula turns (mu, sigma, rho) into a
# weight, and not in the estimator around it.


def soft_parts(F, tau0=TAU0, beta=BETA):
    """``(P, Q, g)``: responsibilities, background weights and subset weights of one update."""
    scale = F.std(1, keepdims=True) + 1e-12
    P = np.exp((F - F.max(1, keepdims=True)) / (tau0 * scale))
    P /= P.sum(1, keepdims=True)
    q = np.sort(F, axis=1)
    gap = q[:, -1] - q[:, -2]
    z = np.clip(-beta * (gap - np.median(gap)) / (gap.std() + 1e-12), -700.0, 700.0)
    g = 1.0 / (1.0 + np.exp(z))
    g = g / (g.sum() + 1e-12)
    Q = 1.0 - P
    Q = Q / Q.sum(1, keepdims=True)
    return P, Q, g


def soft_moments(S2, S3, w, tau0=TAU0, beta=BETA):
    """``(mu2, sd2, mu3, sd3)`` exactly as :func:`soft_update` forms them, returned rather than
    consumed. ``w_from(*soft_moments(...))`` is ``soft_update(...)`` to machine precision, which
    :func:`_selftest_moments` asserts."""
    F = (1 - w) * S2 + w * S3
    P, Q, g = soft_parts(F, tau0, beta)
    K = F.shape[1]
    out = []
    for S in (S2, S3):
        mu = float((g * ((P * S).sum(1) * K / (K - 1.0))).sum())
        mean_bg = (Q * S).sum(1, keepdims=True)
        sd = float((g * np.sqrt(np.maximum((Q * (S - mean_bg) ** 2).sum(1), 0.0))).sum())
        out += [mu, sd]
    return out


def soft_rho(S2, S3, w, tau0=TAU0, beta=BETA):
    """The background residual correlation of Assumption 1, estimated without labels.

    The label indicator is replaced by the responsibilities, ``every class but the label`` by the
    background weights, and the equal weight on confident rows by the subset weights, so that this
    reduces to the hard estimator as ``tau0 -> 0`` and ``beta -> inf``.
    """
    F = (1 - w) * S2 + w * S3
    _, Q, g = soft_parts(F, tau0, beta)
    W = g[:, None] * Q
    e2 = S2 - (Q * S2).sum(1, keepdims=True)
    e3 = S3 - (Q * S3).sum(1, keepdims=True)
    s2 = float(np.sqrt(max((W * e2 * e2).sum(), 0.0)))
    s3 = float(np.sqrt(max((W * e3 * e3).sum(), 0.0)))
    return float((W * e2 * e3).sum()) / (s2 * s3 + 1e-12)


def w_from_rho(mu2, sd2, mu3, sd3, rho):
    """The correlation-corrected weight, and whether it is admissible.

    ``w = a3 / (a2 + a3)`` with ``a_h = (d_h - rho d_other) / sd_h`` and ``d_h = mu_h / sd_h``.
    A correlation large enough to make one numerator negative places the weight outside ``[0, 1]``,
    where it is not a convex combination at all; such a cell is reported as inadmissible rather than
    clipped, since clipping would hide exactly what the ablation is asking about.
    """
    if sd2 <= 1e-12 or sd3 <= 1e-12:
        return float("nan"), False
    d2, d3 = mu2 / sd2, mu3 / sd3
    a2, a3 = (d2 - rho * d3) / sd2, (d3 - rho * d2) / sd3
    if abs(a2 + a3) < 1e-12:
        return float("nan"), False
    w = a3 / (a2 + a3)
    return float(w), bool(0.0 <= w <= 1.0)


def rho_em(S2, S3, w0=W0, tol=TOL, iters=RMAX):
    """Algorithm 1 with :func:`w_from_rho` in place of the closed form. Returns ``(w, rho, ok)``."""
    w, rho, ok = w0, 0.0, True
    for _ in range(iters):
        mu2, sd2, mu3, sd3 = soft_moments(S2, S3, w)
        rho = soft_rho(S2, S3, w)
        nw, ok = w_from_rho(mu2, sd2, mu3, sd3, rho)
        if not np.isfinite(nw):
            return w, rho, ok
        if abs(nw - w) < tol:
            return nw, rho, ok
        w = nw
    return w, rho, ok


def _selftest_moments(rng=None, trials=200):
    """``soft_moments`` must not drift from ``soft_update``; the guard against a re-typed copy."""
    rng = rng or np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        n, K = int(rng.integers(5, 40)), int(rng.integers(8, 60))
        A, B = rng.normal(size=(n, K)), rng.normal(size=(n, K))
        A, B = ctr(A), ctr(B)
        w = float(rng.uniform())
        worst = max(worst, abs(w_from(*soft_moments(A, B, w)) - soft_update(A, B, w)))
    assert worst < 1e-12, f"soft_moments has drifted from soft_update: {worst:g}"
    return worst


EM = {"hard": hard_em, "soft": soft_em}


def _wiring_check():
    from utils.scannet200 import load_index, build_scores, CACHE
    us, uc, zuni = load_index()
    y = np.load(f"{CACHE}/gtmatch_full.npz")["ycls"].astype(np.int64)
    S2, S3 = build_scores("eva", us, uc, zuni)
    S2, S3 = ctr(S2), ctr(S3)
    scenes = [s for s in np.unique(us) if (us == s).sum() >= 3]

    rows = []
    for s in scenes:
        m = us == s
        A, B = S2[m], S3[m]
        soft = [soft_em(A, B, w0) for w0 in STARTS]
        hard = [hard_em(A, B, w0) for w0 in STARTS]
        rows.append(dict(scene=str(s), m=m,
                         soft=soft_em(A, B), hard=hard_em(A, B),
                         spread_soft=max(soft) - min(soft),
                         spread_hard=max(hard) - min(hard)))

    def corpus(key):
        num = den = 0
        for r in rows:
            yy = y[r["m"]]
            ok = yy >= 0
            if ok.sum() == 0:
                continue
            w = r[key]
            p = ((1 - w) * S2[r["m"]] + w * S3[r["m"]])[ok].argmax(1)
            num += int((p == yy[ok]).sum()); den += int(ok.sum())
        return 100.0 * num / den, den

    ss = np.array([r["spread_soft"] for r in rows])
    sh = np.array([r["spread_hard"] for r in rows])
    a_s, n = corpus("soft")
    a_h, _ = corpus("hard")
    print("scenes %d   masks %d" % (len(rows), n))
    print("  soft  acc %.3f   spread>0.01 in %d scenes   max %.4f"
          % (a_s, (ss > .01).sum(), ss.max()))
    print("  hard  acc %.3f   spread>0.01 in %d scenes   max %.4f"
          % (a_h, (sh > .01).sum(), sh.max()))
    ok = (abs(a_s - 52.058) < 5e-4 and abs(a_h - 52.073) < 5e-4
          and (ss > .01).sum() == 11 and (sh > .01).sum() == 78)
    print("WIRING %s" % ("OK" if ok else "MISMATCH -- abort"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _wiring_check() else 1)
