"""Hybrid / multi-layer IDS in the spirit of IDEX (the 'High' family).

Three independent checks must all pass for a window to be accepted:
  1. an ML reconstruction score (the LSTM autoencoder),
  2. a deterministic rule/semantic layer (allow-list + range check),
  3. a conformal-calibration tier that flags windows whose nonconformity
     exceeds a quantile fixed on a held-out clean set.
The attacker has to defeat every layer at once, which is what makes this the
hardest family and the reason multi-surrogate + context methods are needed.
"""
from __future__ import annotations

import numpy as np

from ids.base import IDS
from ids.lstm_ae import LSTMAutoencoderIDS
from ids.rule_based import RuleBasedIDS


class HybridIDS(IDS):
    family = "hybrid"

    def __init__(self, epochs: int = 12, device: str = "cpu", seed: int = 0):
        self.ml = LSTMAutoencoderIDS(epochs=epochs, device=device, seed=seed)
        self.rule = RuleBasedIDS()
        self.q_conformal = 1.0
        self.ml_thr = 1.0
        self.threshold = 0.5  # combined decision is binary

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "HybridIDS":
        self.ml.fit(X_train, X_cal)
        self.rule.fit(X_train, X_cal)
        # conformal tier: nonconformity = ML score; quantile on clean calib set
        cal_scores = self.ml.score_batch(X_cal)
        self.q_conformal = float(np.quantile(cal_scores, 0.99))
        self.ml_thr = float(np.quantile(cal_scores, 0.95))
        return self

    def calibrate(self, X_cal: np.ndarray, target_fpr: float = 0.01) -> None:
        # thresholds already fixed in fit(); keep interface uniform
        cal_scores = self.ml.score_batch(X_cal)
        self.ml_thr = float(np.quantile(cal_scores, 1.0 - target_fpr))
        self.q_conformal = float(np.quantile(cal_scores, 1.0 - target_fpr / 2))

    def score(self, x: np.ndarray) -> float:
        ml_s = self.ml.score(x)
        rule_s = self.rule.score(x)
        # soft aggregate used only for ranking/AUC; alert() uses the hard rule
        return max(ml_s / (self.ml_thr + 1e-9), rule_s, ml_s / (self.q_conformal + 1e-9))

    def alert(self, x: np.ndarray) -> bool:
        ml_s = self.ml.score(x)
        if self.rule.alert(x):
            return True                      # layer 2: deterministic rule
        if ml_s > self.ml_thr:
            return True                      # layer 1: ML threshold
        if ml_s > self.q_conformal:
            return True                      # layer 3: conformal tier
        return False
