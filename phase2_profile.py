"""Phase 2 -- black-box reconnaissance of the IDS decision boundary.

We query the real IDS with structured perturbations of recorded normal windows
and record only its yes/no answer.  Three products come out:
  * a labelled query set (window -> alerted?) that trains the surrogate,
  * a per-byte sensitivity map (which payload bytes are 'risky'),
  * a rate map (how many extra frames of an id are tolerated).
Everything here uses only `ids.alert`, never any internal state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from canbus import BYTE_COLS, ID_MAX, N_BYTES, encode_window
from dataset import BUS_SCHEMA, TARGET_ID


@dataclass
class ProfileResult:
    X_query: np.ndarray              # (N, W, F) probed windows
    y_query: np.ndarray              # (N,) 1 = IDS alerted
    byte_sensitivity: np.ndarray     # (N_BYTES,) fraction of alerts per byte
    rate_tolerance: float            # extra frames tolerated before alert
    n_queries: int = 0


def _perturb(x: np.ndarray, rng, severity: float) -> np.ndarray:
    """Random but legal perturbation of a window; severity scales the change."""
    x = x.copy()
    n_edit = max(1, int(severity * len(x)))
    rows = rng.choice(len(x), size=n_edit, replace=False)
    for p in rows:
        cols = rng.choice(BYTE_COLS, size=rng.integers(1, N_BYTES + 1), replace=False)
        x[p, cols] = np.clip(x[p, cols] + rng.normal(0, severity, size=len(cols)), 0, 1)
        if rng.random() < severity * 0.5:    # occasionally also spoof an id
            x[p, 0] = rng.choice([s[0] for s in BUS_SCHEMA]) / ID_MAX
    return x


def profile_ids(ids, pool_raw, rng, n_queries: int = 600) -> ProfileResult:
    X, y = [], []
    n = 0

    # 1. graded random perturbations -> the bulk of the labelled query set
    for _ in range(n_queries):
        idx = int(rng.integers(0, len(pool_raw)))
        base = encode_window(pool_raw[idx])
        sev = float(rng.uniform(0.0, 0.6))
        xp = _perturb(base, rng, sev) if sev > 1e-3 else base
        X.append(xp)
        y.append(int(ids.alert(xp)))
        n += 1

    # 2. per-byte sensitivity sweep on the target id
    byte_hits = np.zeros(N_BYTES)
    sweeps = 20
    for b in range(N_BYTES):
        hits = 0
        for _ in range(sweeps):
            idx = int(rng.integers(0, len(pool_raw)))
            base = encode_window(pool_raw[idx])
            pos = int(rng.integers(0, len(base)))
            base[pos, 0] = TARGET_ID / ID_MAX
            base[pos, BYTE_COLS[b]] = rng.random()
            hits += int(ids.alert(base))
            n += 1
        byte_hits[b] = hits / sweeps

    # 3. rate tolerance: add k copies of a busy id until the IDS reacts
    rate_tol = float(len(pool_raw[0]))
    sid = BUS_SCHEMA[0][0]
    for k in range(1, len(pool_raw[0]) // 2):
        alerts = 0
        for _ in range(10):
            idx = int(rng.integers(0, len(pool_raw)))
            base = encode_window(pool_raw[idx])
            pos = rng.choice(len(base), size=k, replace=False)
            base[pos, 0] = sid / ID_MAX
            alerts += int(ids.alert(base))
            n += 1
        if alerts / 10 > 0.5:
            rate_tol = float(k)
            break

    return ProfileResult(
        X_query=np.stack(X),
        y_query=np.array(y),
        byte_sensitivity=byte_hits,
        rate_tolerance=rate_tol,
        n_queries=n,
    )
