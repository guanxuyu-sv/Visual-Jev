"""Frozen risk thresholds must be realizable when confidences tie.

Run with `python code/tests/test_threshold.py` or pytest.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm.eval.metrics import threshold_for_risk


def test_tied_group_is_accepted_or_rejected_whole():
    # two questions tie at 0.9, one right and one wrong: accepting either means
    # accepting both (risk 0.5), so a 5% target must accept nothing
    conf, correct = np.array([0.9, 0.9, 0.8]), np.array([1, 0, 0])
    thr = threshold_for_risk(conf, correct, 0.05)
    assert not (conf >= thr).any()


def test_calibration_risk_never_exceeds_target_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(5, 60))
        conf = np.round(rng.random(n), 1)              # heavy ties, like 2-decimal API output
        correct = (rng.random(n) < 0.85).astype(int)
        target = float(rng.choice([0.05, 0.1, 0.2]))
        accept = conf >= threshold_for_risk(conf, correct, target)
        if accept.any():
            assert 1 - correct[accept].mean() <= target + 1e-12


def test_no_ties_unchanged():
    rng = np.random.default_rng(1)
    conf = rng.random(500)
    correct = (rng.random(500) < conf).astype(int)
    order = np.argsort(-conf, kind="mergesort")
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, 501)
    expected = conf[order][np.where(risk <= 0.1)[0][-1]]
    assert threshold_for_risk(conf, correct, 0.1) == expected


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
