"""Metrics for decision quality, calibration and selective prediction.

Conventions that the plan insists on and that are enforced here:

* Accuracy is always reported on the *full* sample before any abstention; the
  selective numbers are reported alongside coverage, never instead of it
  (plan 9.3).
* Bootstrap resampling clusters on the parent image, because questions on one
  image and the degraded variants of one image are not independent draws.
* Risk--coverage thresholds are fitted on a calibration split and then applied
  unchanged; `coverage_at_risk` never searches the test set for a threshold.
"""
from __future__ import annotations

import collections
import math
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# proper scoring rules
# --------------------------------------------------------------------------- #
def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score (sum of squared error over the simplex)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def macro_f1(probs: np.ndarray, labels: np.ndarray, n_classes: int | None = None) -> float:
    pred = np.argmax(probs, axis=1)
    classes = range(n_classes or probs.shape[1])
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        if tp + fn == 0:          # class absent from the reference: skip
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width confidence ECE.  Auxiliary only: it is bin-sensitive and not
    a proper scoring rule, so the plan keeps NLL/Brier as the primary numbers."""
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# binary discrimination (answerability)
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie handling; labels in {0,1}."""
    pos, neg = labels == 1, labels == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-wise, no interpolation)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / y.sum())


# --------------------------------------------------------------------------- #
# selective prediction
# --------------------------------------------------------------------------- #
def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error rate among accepted) as a function of coverage.

    Sorting by descending confidence and accepting prefixes gives the standard
    empirical risk--coverage curve.
    """
    order = np.argsort(-confidence, kind="mergesort")
    c = correct[order].astype(float)
    n = len(c)
    cum_correct = np.cumsum(c)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    return coverage, risk


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk--coverage curve.  Lower is better."""
    cov, risk = risk_coverage(confidence, correct)
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0] + 1e-12)) if len(cov) > 1 else float("nan")


def threshold_for_risk(
    confidence: np.ndarray, correct: np.ndarray, target_risk: float
) -> float:
    """Smallest confidence threshold on THIS split whose empirical risk is <=
    target.  Intended to be called on the calibration split only."""
    order = np.argsort(-confidence, kind="mergesort")
    c = correct[order].astype(float)
    conf_sorted = confidence[order]
    cum = np.cumsum(c)
    k = np.arange(1, len(c) + 1)
    risk = 1.0 - cum / k
    ok = np.where(risk <= target_risk)[0]
    if len(ok) == 0:
        return float(np.max(confidence) + 1e-6)     # accept nothing
    return float(conf_sorted[ok[-1]])


def apply_threshold(
    confidence: np.ndarray, correct: np.ndarray, thr: float
) -> dict[str, float]:
    acc = confidence >= thr
    cov = float(np.mean(acc))
    return {
        "coverage": cov,
        "selective_risk": float(1.0 - np.mean(correct[acc])) if acc.any() else float("nan"),
        "selective_accuracy": float(np.mean(correct[acc])) if acc.any() else float("nan"),
        "threshold": float(thr),
        "full_accuracy": float(np.mean(correct)),
    }


def selective_accuracy_at_coverage(
    confidence: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    n_keep = max(1, int(round(coverage * len(correct))))
    order = np.argsort(-confidence, kind="mergesort")[:n_keep]
    return float(np.mean(correct[order]))


# --------------------------------------------------------------------------- #
# clustered bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    cluster_ids: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample *clusters* (parent images) with replacement.

    `stat_fn` receives the array of row indices selected and returns the scalar.
    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    idx_by_cluster: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(cluster_ids):
        idx_by_cluster[c].append(i)
    clusters = list(idx_by_cluster)
    point = stat_fn(np.arange(len(cluster_ids)))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        rows: list[int] = []
        for p in pick:
            rows.extend(idx_by_cluster[clusters[p]])
        if not rows:
            continue
        v = stat_fn(np.asarray(rows))
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_cluster_bootstrap_delta(
    stat_fn_a: Callable[[np.ndarray], float],
    stat_fn_b: Callable[[np.ndarray], float],
    cluster_ids: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI on the paired difference a - b over the same resampled clusters.

    Plan 13.1 asks for paired CIs to support a claimed gain, not just two point
    estimates that happen to differ.
    """
    rng = np.random.default_rng(seed)
    idx_by_cluster: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(cluster_ids):
        idx_by_cluster[c].append(i)
    clusters = list(idx_by_cluster)
    all_rows = np.arange(len(cluster_ids))
    point = stat_fn_a(all_rows) - stat_fn_b(all_rows)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        rows: list[int] = []
        for p in pick:
            rows.extend(idx_by_cluster[clusters[p]])
        r = np.asarray(rows)
        va, vb = stat_fn_a(r), stat_fn_b(r)
        if not (math.isnan(va) or math.isnan(vb)):
            vals.append(va - vb)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# intervention specificity (E3)
# --------------------------------------------------------------------------- #
def intervention_specificity(
    suff_orig: np.ndarray,
    suff_relevant: np.ndarray,
    suff_irrelevant: np.ndarray,
) -> dict[str, float]:
    """The central diagnostic of C2.

    `drop_relevant` should be large and `drop_irrelevant` near zero.  A model
    that merely detects degradation shows the two moving together, which
    `specificity_gap` makes visible.
    """
    d_rel = suff_orig - suff_relevant
    d_irr = suff_orig - suff_irrelevant
    return {
        "suff_orig_mean": float(np.mean(suff_orig)),
        "drop_relevant_mean": float(np.mean(d_rel)),
        "drop_irrelevant_mean": float(np.mean(d_irr)),
        "specificity_gap": float(np.mean(d_rel - d_irr)),
        "frac_correctly_ordered": float(np.mean(d_rel > d_irr)),
        # AUROC of separating the relevant arm from the irrelevant arm using the
        # sufficiency score alone: 0.5 means no question-conditioning at all.
        "pair_auroc": auroc(
            np.concatenate([-suff_relevant, -suff_irrelevant]),
            np.concatenate([np.ones_like(suff_relevant), np.zeros_like(suff_irrelevant)]),
        ),
    }


# --------------------------------------------------------------------------- #
# post-hoc calibration
# --------------------------------------------------------------------------- #
def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 400) -> float:
    """Single-parameter temperature scaling by golden-section search on NLL.

    Fitted on the calibration split, then applied unchanged (plan 7.4).
    """
    def obj(t: float) -> float:
        z = logits / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return nll(p, labels)

    lo, hi = 0.05, 10.0
    invphi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    for _ in range(iters):
        if obj(c) < obj(d):
            b = d
        else:
            a = c
        c, d = b - invphi * (b - a), a + invphi * (b - a)
        if abs(b - a) < 1e-4:
            break
    return float((a + b) / 2)


def softmax(logits: np.ndarray, t: float = 1.0, mask: np.ndarray | None = None) -> np.ndarray:
    z = logits / max(t, 1e-6)
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p = np.where(np.isfinite(p), p, 0.0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
