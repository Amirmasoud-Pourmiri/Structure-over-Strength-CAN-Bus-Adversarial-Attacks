"""OTIDS-style timing-based IDS adapter.

Faithful re-implementation of the per-ID inter-arrival timing detection
principle from Lee, Jeong, and Kim -- "OTIDS: A Novel Intrusion Detection
System for In-Vehicle Network by Using Remote Frame," PST 2017 -- adapted
to our window-based interface.

OTIDS detects intrusions by monitoring whether per-ID message timing deviates
from learned norms.  For each arbitration ID the detector learns the expected
inter-arrival interval (mu) and its spread (sigma) from clean traffic.  A
window is anomalous when the maximum z-score of observed inter-arrival times
across all IDs in that window exceeds a calibrated threshold.

This is an *independent* implementation: the code has no shared state with the
four author-controlled IDS families in ids/ and reproduces the OTIDS detection
principle from the published description rather than reusing any internal
component of our own statistical detector.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from canbus import ID_MAX
from ids.base import IDS


class OTIDSStyleIDS(IDS):
    """Timing-based IDS in the spirit of OTIDS (Lee et al., PST 2017).

    Detection logic
    ---------------
    Training: for each arbitration ID observed in clean traffic, record the
    inter-arrival times and fit a Gaussian (mu, sigma) per ID.

    Scoring: for a test window, compute the mean observed inter-arrival time
    for each ID, convert it to a z-score against the training distribution,
    and return the maximum z-score across all IDs as the anomaly score.
    A large positive z means frames arrived faster than usual (possible
    injection); a large negative z means frames arrived slower (possible
    suppression).  We take the absolute z-score so both directions count.

    This mirrors the OTIDS intuition: injecting extra frames shortens the
    observed inter-arrival time and shifts the z-score upward.
    """

    family = "otids"

    def __init__(self):
        self.iat_mu: dict[int, float] = {}
        self.iat_sigma: dict[int, float] = {}
        self.threshold: float = 1e9

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_id_iats(self, x: np.ndarray) -> dict[int, list[float]]:
        """Extract per-ID inter-arrival times from an encoded window."""
        last_t: dict[int, float] = {}
        iats: dict[int, list[float]] = defaultdict(list)
        t = 0.0
        for row in x:
            sid = int(round(float(row[0]) * ID_MAX))
            # column 2+N_BYTES holds the normalised IAT increment
            # use index -1 which is the last column = IAT field regardless
            # of N_BYTES value; this avoids importing N_BYTES here.
            iat_col = row[-1]
            t += float(iat_col)
            if sid in last_t:
                iats[sid].append(t - last_t[sid])
            last_t[sid] = t
        return dict(iats)

    def _score_window(self, x: np.ndarray) -> float:
        iats = self._per_id_iats(x)
        z_max = 0.0
        for sid, vs in iats.items():
            if sid not in self.iat_mu or not vs:
                continue
            mu = self.iat_mu[sid]
            sg = self.iat_sigma[sid]
            obs = float(np.mean(vs))
            z = abs(obs - mu) / (sg + 1e-9)
            z_max = max(z_max, z)
        return z_max

    # ------------------------------------------------------------------
    # IDS interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "OTIDSStyleIDS":
        per_id_iats: dict[int, list[float]] = defaultdict(list)
        for x in X_train:
            for sid, vs in self._per_id_iats(x).items():
                per_id_iats[sid].extend(vs)
        for sid, vs in per_id_iats.items():
            self.iat_mu[sid] = float(np.mean(vs))
            self.iat_sigma[sid] = float(np.std(vs) + 1e-4)
        return self

    def score(self, x: np.ndarray) -> float:
        return self._score_window(x)

    def calibrate(self, X_cal: np.ndarray, target_fpr: float = 0.01) -> None:
        scores = self.score_batch(X_cal)
        self.threshold = float(np.quantile(scores, 1.0 - target_fpr))
