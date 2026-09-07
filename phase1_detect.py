"""Phase 1 -- decide whether an IDS is on the bus and guess its family.

The attacker can only watch the bus and observe reactions.  Instead of throwing
one confounded probe at the model, we send three *mechanism-isolating* probes,
each designed to trip exactly one detection principle (Section 3.2--3.3):

  * ``small`` -- a payload byte is nudged *just* past the legal envelope.  A
                 deterministic range check (rule, or the rule tier inside a
                 hybrid) trips instantly; a reconstruction model barely moves
                 because the error is tiny, and a frequency monitor is blind.
  * ``gross`` -- the same byte is driven far out of range.  Both the range check
                 and the reconstruction model react; the frequency monitor still
                 does not.
  * ``rate``  -- a busy id is duplicated into a couple of extra slots carrying a
                 *legal* payload, so only the per-id frequency / inter-arrival
                 rhythm changes.  Only a statistical monitor reacts.

The binary pattern of which probes fire then votes for one of the four families.
A reconstruction autoencoder behaves like a *soft* range check (it reacts to the
gross violation but rides over the small one), which is exactly what separates it
from a hard signature rule.  A layered hybrid is dominated by its deterministic
rule tier under this cheap battery and therefore looks like a rule gateway from
the outside -- a deliberately reported blind spot (Section 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from canbus import BYTE_COLS, ID_MAX, N_BYTES
from dataset import BUS_SCHEMA, TARGET_ID

ID_COL = 0
DLC_COL = 1
IAT_COL = 2 + N_BYTES
_BUSY_ID = BUS_SCHEMA[0][0]          # fastest-cycling id -> tightest rate model


@dataclass
class DetectionReport:
    ids_present: bool
    confidence: float
    family_guess: str
    reaction: Dict[str, float]


def _baseline_window(pool_raw, idx: int) -> np.ndarray:
    from canbus import encode_window
    return encode_window(pool_raw[idx])


def _row_for_id(x: np.ndarray, sid: int):
    """Return the byte columns of some in-window frame carrying `sid`, or None."""
    key = sid / ID_MAX
    hits = np.where(np.isclose(x[:, ID_COL], key, atol=0.5 / ID_MAX))[0]
    if len(hits):
        return x[hits[0], BYTE_COLS].copy()
    return None


def _probe_small(x: np.ndarray, rng) -> np.ndarray:
    """A byte nudged just past the legal envelope (rule trips, AE rides over)."""
    x = x.copy()
    pos = int(rng.integers(0, len(x)))
    x[pos, BYTE_COLS[-1]] = 0.16     # byte 7 is ~0 normally; rule margin is 0.10
    return x


def _probe_gross(x: np.ndarray, rng) -> np.ndarray:
    """Bytes driven far out of range (both rule and AE react).

    The high byte positions sit at ~0 on every id in normal traffic, so driving
    several of them up at once is strongly out of distribution -- enough that a
    reconstruction model trips reliably, not just the hard range check.
    """
    x = x.copy()
    pos = int(rng.integers(0, len(x)))
    for c in BYTE_COLS[-4:]:
        x[pos, c] = 0.85
    return x


def _probe_rate(x: np.ndarray, rng) -> np.ndarray:
    """Frequency/IAT anomaly only: duplicate a busy id with a *legal* payload."""
    x = x.copy()
    payload = _row_for_id(x, _BUSY_ID)
    if payload is None:
        payload = x[0, BYTE_COLS]
    slots = rng.choice(np.arange(1, len(x)), size=2, replace=False)
    for p in slots:
        x[p, ID_COL] = _BUSY_ID / ID_MAX
        x[p, BYTE_COLS] = payload
        x[p, IAT_COL] = 0.02         # arrives hard on the heels of its sibling
    return x


PROBES = {
    "small": _probe_small,
    "gross": _probe_gross,
    "rate": _probe_rate,
}


def detect_ids(ids, pool_raw, rng, n_trials: int = 30) -> DetectionReport:
    reaction: Dict[str, float] = {}
    base_alarm = 0
    n_base = 0
    for name, probe in PROBES.items():
        hits = 0
        for _ in range(n_trials):
            idx = int(rng.integers(0, len(pool_raw)))
            base = _baseline_window(pool_raw, idx)
            base_alarm += int(ids.alert(base))
            n_base += 1
            hits += int(ids.alert(probe(base, rng)))
        reaction[name] = hits / n_trials
    base_rate = base_alarm / max(n_base, 1)

    overall = float(max(reaction.values()))
    present = (overall - base_rate) > 0.1
    confidence = float(np.clip(overall - base_rate, 0, 1))
    family = _classify_family(reaction)
    return DetectionReport(present, confidence, family, reaction)


def _classify_family(r: Dict[str, float]) -> str:
    """Vote for a family from the binary signature of which mechanisms fired."""
    small = r["small"] > 0.5         # hard range check fires on a tiny violation
    gross = r["gross"] > 0.5         # any content check fires on a large one
    rate = r["rate"] > 0.5           # frequency / inter-arrival layer

    if rate and not gross:
        return "statistical"         # rhythm only, blind to payload
    if small and gross:
        return "rule"                # hard range layer (a hybrid hides here too)
    if gross and not small:
        return "lstm_ae"             # soft reconstruction: large yes, small no
    if rate:
        return "statistical"
    return "rule"
