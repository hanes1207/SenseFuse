"""Alternatives to the linear fusion rule, against which it is measured.

``models/law.py`` states that the optimal weight is not invariant to the score space. That is a
claim about the weight. It leaves open whether the *rule* is right at all, which is what this module
measures: every alternative here maps two per-mask score rows to one, carries a single scalar, and
is estimated label-free in the space it fuses in, so that no rule is penalised for another's
estimator.

The alternatives fall into three groups.

**Pooled differently, still before the argmax.** ``rowstd`` and ``minmax`` are the linear rule after
a per-row rescaling, ``rank`` replaces each row by its Borda ranking and therefore keeps only the
ordering, ``lop`` averages the two softmaxes and ``poe`` averages them geometrically, and
``maxpool`` takes the elementwise maximum. ``poe`` is included because it is not a new rule at all:
a geometric mean of softmaxes is a linear combination of logits, so it can differ from ``linear``
only through the per-channel temperature, which is ``rowstd``.

**After the argmax.** ``router`` awards the mask to whichever channel has the larger standardised
margin, and ``veto`` keeps the 2D label unless the shape channel ranks it outside its top *r*.
Neither ever adds a score, and the gap between these and the first group is what the paper's claim
that fusion must precede the argmax rests on.

**In the labeller's own count space.** Open-YOLO3D's votes are multinomial counts, so a prior
belongs in the same units: ``dirichlet`` enters the shape channel as one, with a mass measured in
units of the row's own vote total, rather than summing a count with a cosine.
"""
import numpy as np

__all__ = ["RULES", "SPACE_OF", "t_none", "t_std", "t_minmax", "t_rank", "softmax",
           "r_dirichlet", "margin"]


# ------------------------------------------------------------------ row transforms
def t_none(S):
    return S


def t_std(S):
    return S / (S.std(1, keepdims=True) + 1e-9)


def t_minmax(S):
    lo = S.min(1, keepdims=True)
    hi = S.max(1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-9)


def t_rank(S):
    """Borda: 0 for the worst class, 1 for the best. Scale-free by construction, which is the
    point, since it keeps only the ordering each channel induces and discards every magnitude."""
    r = np.argsort(np.argsort(S, 1), 1).astype(np.float32)
    return r / max(S.shape[1] - 1, 1)


def softmax(S):
    """Softmax of standardised rows, so that the temperature is set by the data and is not a
    second free parameter."""
    Z = t_std(S)
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


# ------------------------------------------------------------------ rules
def r_linear(A, B, w, tf=t_none):
    return (1 - w) * tf(A) + w * tf(B)


def r_lop(A, B, w):
    """Linear opinion pool: the arithmetic mean of the two probability rows. Not a monotone
    transform of the linear rule, since a class the 2D head is merely unsure about survives here,
    where in log space one confident rejection removes it."""
    return (1 - w) * softmax(A) + w * softmax(B)


def r_poe(A, B, w):
    """Product of experts: the geometric mean of the two probability rows, taken in logs."""
    return (1 - w) * np.log(softmax(A) + 1e-12) + w * np.log(softmax(B) + 1e-12)


def r_max(A, B, w):
    return np.maximum((1 - w) * t_std(A), w * t_std(B))


def margin(S):
    """Top-1 minus top-2 on standardised rows: how strongly a channel prefers its own answer, in
    units comparable across two channels of different scale."""
    Z = t_std(S)
    p = np.partition(Z, -2, 1)
    return p[:, -1] - p[:, -2]


def r_router(A, B, w):
    """Decision level: no score is ever added. Each channel keeps its own argmax and the mask is
    awarded to whichever is locally more confident, with ``w`` tilting the contest. A one-hot
    matrix is returned so that the caller may take an argmax of it as of any other rule."""
    take3 = w * margin(B) > (1 - w) * margin(A)
    lab = np.where(take3, B.argmax(1), A.argmax(1))
    out = np.zeros_like(A)
    out[np.arange(len(lab)), lab] = 1.0
    return out


def r_veto(A, B, w):
    """Decision level, asymmetric: the 2D head proposes and the shape head may reject. The 2D label
    stands unless the shape channel ranks it outside its top *r*, where ``r = 1 + round((1-w)(K-1))``,
    so that ``w = 0`` never rejects and ``w = 1`` always defers."""
    K = A.shape[1]
    r = 1 + int(round((1 - w) * (K - 1)))
    lab2 = A.argmax(1)
    rank3 = np.argsort(np.argsort(-B, 1), 1)          # 0 is the shape head's favourite
    lab = np.where(rank3[np.arange(len(lab2)), lab2] >= r, B.argmax(1), lab2)
    out = np.zeros_like(A)
    out[np.arange(len(lab)), lab] = 1.0
    return out


def r_dirichlet(N, B, a0, uniform=False):
    """Vote space. The counts ``n_c`` come from a multinomial over classes, so a prior belongs in
    the same units: the posterior mean is proportional to ``n_c + a0 * p3_c``, with ``a0`` measured
    in units of the row's own vote total. ``a0 = 1`` therefore means that the prior is worth as much
    as everything the mask observed, which is a fixed default rather than a tuned one.

    ``uniform=True`` replaces the shape prior by ``1/K``. That is the placebo: it smooths the sparse
    histogram by exactly as much, in exactly the same places, while carrying no shape information.
    Most classes receive no votes, so any prior breaks ties across that mass, and the placebo is the
    only way to tell whether the shape channel supplies evidence or merely breaks those ties."""
    P = np.full_like(N, 1.0 / N.shape[1]) if uniform else softmax(B)
    return N + a0 * P * np.maximum(N.sum(1, keepdims=True), 1.0)


RULES = {
    "linear": lambda A, B, w: r_linear(A, B, w, t_none),
    "rowstd": lambda A, B, w: r_linear(A, B, w, t_std),
    "minmax": lambda A, B, w: r_linear(A, B, w, t_minmax),
    "rank": lambda A, B, w: r_linear(A, B, w, t_rank),
    "lop": r_lop,
    "poe": r_poe,
    "maxpool": r_max,
    "router": r_router,
    "veto": r_veto,
}

# The space each rule fuses in. Algorithm 1 is run on these rows rather than on the raw ones, so
# that every rule is read at the label-free weight its own space implies.
SPACE_OF = {"linear": t_none, "rowstd": t_std, "minmax": t_minmax, "rank": t_rank,
            "lop": t_std, "poe": t_std, "maxpool": t_std, "router": t_std, "veto": t_std}
