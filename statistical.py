"""Statistical / frequency IDS (the 'Medium difficulty' family).

Models, per arbitration id, the normal message frequency inside a window and
the mean inter-arrival time.  It is tolerant of payload variation but reacts
when an id appears more often than usual or the bus rhythm shifts -- exactly
the footprint that naive frame injection leaves behind.
"""
from __future__ import annotations

import numpy as np

from canbus import ID_MAX, N_BYTES
from ids.base import IDS


class StatisticalIDS(IDS):
    family = "statistical"

    def __init__(self):
        self.mu = {}
        self.sigma = {}
        self.iat_mu = {}
        self.iat_sigma = {}

    def _counts(self, x: np.ndarray):
        counts = {}
        last_t = {}
        iats = {}
        t = 0.0
        for row in x:
            sid = int(round(row[0] * ID_MAX))
            t += row[2 + N_BYTES]  # accumulated normalised iat
            counts[sid] = counts.get(sid, 0) + 1
            if sid in last_t:
                iats.setdefault(sid, []).append(t - last_t[sid])
            last_t[sid] = t
        return counts, iats

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "StatisticalIDS":
        per_id_counts = {}
        per_id_iats = {}
        for x in X_train:
            counts, iats = self._counts(x)
            for sid, c in counts.items():
                per_id_counts.setdefault(sid, []).append(c)
            for sid, vs in iats.items():
                per_id_iats.setdefault(sid, []).extend(vs)
        for sid, cs in per_id_counts.items():
            self.mu[sid] = float(np.mean(cs))
            self.sigma[sid] = float(np.std(cs) + 1e-3)
        for sid, vs in per_id_iats.items():
            self.iat_mu[sid] = float(np.mean(vs))
            self.iat_sigma[sid] = float(np.std(vs) + 1e-3)
        return self

    def score(self, x: np.ndarray) -> float:
        counts, iats = self._counts(x)
        z = 0.0
        for sid, c in counts.items():
            mu = self.mu.get(sid, 0.0)
            sg = self.sigma.get(sid, 1.0)
            z = max(z, abs(c - mu) / sg)  # count anomaly
        for sid, vs in iats.items():
            if sid in self.iat_mu and vs:
                dev = abs(np.mean(vs) - self.iat_mu[sid]) / self.iat_sigma[sid]
                z = max(z, dev)
        return float(z)
