"""Rule-based / signature IDS (the 'Low difficulty' family).

Learns, from clean traffic, the set of legal arbitration ids and a per-id,
per-byte legal value range.  At test time any frame whose id is unknown, or
whose payload byte falls outside the observed envelope, trips the alert.  This
is the classic allow-list + range-check gateway -- instantaneous and binary.
"""
from __future__ import annotations

import numpy as np

from canbus import N_BYTES, ID_MAX
from ids.base import IDS


class RuleBasedIDS(IDS):
    family = "rule"

    def __init__(self, margin: float = 0.10):
        self.margin = margin
        self.known_ids: set[int] = set()
        self.lo = {}
        self.hi = {}
        self.threshold = 0.5  # any single violation => alert

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "RuleBasedIDS":
        ids_seen = {}
        for x in X_train:
            for row in x:
                sid = int(round(row[0] * ID_MAX))
                bytes_ = row[2:2 + N_BYTES]
                ids_seen.setdefault(sid, []).append(bytes_)
        for sid, rows in ids_seen.items():
            arr = np.stack(rows)
            self.known_ids.add(sid)
            self.lo[sid] = arr.min(axis=0) - self.margin
            self.hi[sid] = arr.max(axis=0) + self.margin
        return self

    def calibrate(self, X_cal: np.ndarray, target_fpr: float = 0.01) -> None:
        # a signature IDS fires on ANY range violation; the threshold is fixed,
        # not learned from a quantile (which would mask single-frame violations).
        self.threshold = 0.5

    def score(self, x: np.ndarray) -> float:
        violations = 0
        for row in x:
            sid = int(round(row[0] * ID_MAX))
            if sid not in self.known_ids:
                violations += 1
                continue
            b = row[2:2 + N_BYTES]
            out = np.sum((b < self.lo[sid]) | (b > self.hi[sid]))
            violations += int(out > 0)
        # normalise to a soft score in [0, 1]; alert() fires above 0.5
        return min(1.0, violations / 1.0)
