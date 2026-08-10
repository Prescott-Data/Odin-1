from __future__ import annotations
from typing import List, Tuple, Set
from sklearn.isotonic import IsotonicRegression
import numpy as np
import numpy as np


def recall_at_k(predicted: List[Tuple[str, float]], relevant: Set[str], k: int = 10) -> float:
    if not predicted or not relevant:
        return 0.0
    top = [n for n, _ in predicted[:k]]
    hits = sum(1 for n in top if n in relevant)
    denom = min(k, len(relevant)) or 1
    return hits / denom


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi)
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(conf - acc) * (mask.sum() / max(N, 1))
    return float(ece)


class SimpleLLMCalibrator:
    """
    Wraps isotonic regression to calibrate LLM self-reported confidences.
    """
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, raw_conf: np.ndarray, labels: np.ndarray):
        self.iso.fit(raw_conf, labels)
        self.is_fitted = True

    def transform(self, raw_conf: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return raw_conf
        return self.iso.transform(raw_conf)


