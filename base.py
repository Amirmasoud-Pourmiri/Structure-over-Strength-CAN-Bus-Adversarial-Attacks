"""Common interface every reproduced IDS implements.

The whole point of the black-box pipeline is that the attacker never sees this
class -- only the yes/no `alert()` decision (and, where a deployment leaks it, a
scalar `score()`).  Keeping a uniform interface lets the evaluation harness
swap one IDS family for another without touching the attack code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IDS(ABC):
    family: str = "abstract"

    @abstractmethod
    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "IDS":
        ...

    @abstractmethod
    def score(self, x: np.ndarray) -> float:
        """Anomaly score for a single (W, F) window; higher = more anomalous."""

    def alert(self, x: np.ndarray) -> bool:
        """The only signal the attacker is guaranteed to observe."""
        return self.score(x) > self.threshold

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.score(x) for x in X])

    threshold: float = 0.0

    def calibrate(self, X_cal: np.ndarray, target_fpr: float = 0.01) -> None:
        """Set the threshold to a chosen false-positive rate on clean data."""
        s = self.score_batch(X_cal)
        self.threshold = float(np.quantile(s, 1.0 - target_fpr))
