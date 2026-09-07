"""Method M9 -- context engineering, applied by every other method.

The malicious frame is fixed, but the frames around it are the attacker's to
choose.  We (a) place the malicious frame near the middle of the window where
window-based models weight it least, (b) build the companion frames from exact
replays of recently recorded normal frames so they are normal by construction,
and (c) keep the command bytes pinned through any later optimisation.
"""
from __future__ import annotations

import numpy as np

from canbus import BYTE_COLS, ID_MAX, N_BYTES, encode_window
from dataset import CMD_BYTE_INDEX, CMD_VALUE, TARGET_ID

IAT_COL = 2 + N_BYTES      # the inter-arrival-time feature column


def middle_position(window_len: int) -> int:
    return window_len // 2


def pick_target_slot(carrier: np.ndarray) -> int:
    """Index of an existing target-id frame nearest the middle (rate-preserving).

    Replacing a slot that already carries the target id keeps the per-id message
    count and rhythm identical to normal traffic -- the 'match local rhythm /
    replay where possible' rule -- so a frequency IDS sees nothing unusual.
    Falls back to the middle if the carrier happens to hold no target frame.
    """
    ids = np.rint(carrier[:, 0] * ID_MAX).astype(int)
    slots = np.where(ids == TARGET_ID)[0]
    if len(slots) == 0:
        return middle_position(len(carrier))
    mid = middle_position(len(carrier))
    return int(slots[np.argmin(np.abs(slots - mid))])


def non_target_slot(carrier: np.ndarray) -> int:
    """A slot that does NOT carry the target id (used by the naive baseline, so
    its injection adds an extra target frame and disturbs the rate)."""
    ids = np.rint(carrier[:, 0] * ID_MAX).astype(int)
    mid = middle_position(len(carrier))
    for off in range(len(carrier)):
        for p in (mid + off, mid - off):
            if 0 <= p < len(carrier) and ids[p] != TARGET_ID:
                return p
    return mid


def make_carrier(pool_raw, rng) -> np.ndarray:
    """A replayed clean window the malicious frame will be embedded into."""
    idx = int(rng.integers(0, len(pool_raw)))
    return encode_window(pool_raw[idx]).copy()


def embed_malicious(carrier: np.ndarray, pos: int, rng) -> np.ndarray:
    """Place the door-unlock spoof at `pos` (a rate-preserving target slot).

    The seven non-command bytes start arbitrary -- this is the raw, un-crafted
    spoof.  The command byte is pinned to the unlock value; everything else is
    what the Phase-4 methods must shape into something the IDS accepts.
    """
    x = carrier.copy()
    x[pos, 0] = TARGET_ID / ID_MAX
    x[pos, 1] = N_BYTES / N_BYTES
    x[pos, BYTE_COLS] = rng.random(len(BYTE_COLS))
    x[pos, BYTE_COLS[CMD_BYTE_INDEX]] = CMD_VALUE / 255.0
    return x


def embed_naive(carrier: np.ndarray, pos: int, rng) -> np.ndarray:
    """The dumb attacker: a freshly minted command frame with arbitrary payload
    dropped at a non-target slot (adds a frame, disturbs the rate, no shaping)."""
    x = carrier.copy()
    x[pos, 0] = TARGET_ID / ID_MAX
    x[pos, 1] = N_BYTES / N_BYTES
    x[pos, BYTE_COLS] = rng.random(len(BYTE_COLS))
    x[pos, BYTE_COLS[CMD_BYTE_INDEX]] = CMD_VALUE / 255.0
    return x


def command_mask(window_len: int, pos: int) -> np.ndarray:
    """Features the attack must NOT touch.

    The attacker only shapes the bytes of the *injected* frame; the surrounding
    companions are genuine recorded frames replayed unchanged (so they cannot
    drift out of spec), the arbitration ids/dlcs stay legal, the timing/rhythm is
    preserved, and the command byte is pinned.  Hence everything is protected
    except the seven non-command payload bytes of the malicious frame at `pos`.
    """
    mask = np.ones((window_len, 11), dtype=bool)
    free_cols = [c for c in BYTE_COLS if c != BYTE_COLS[CMD_BYTE_INDEX]]
    mask[pos, free_cols] = False                   # only the spoofed frame's bytes are free
    return mask


def enforce_command(x: np.ndarray, pos: int) -> np.ndarray:
    """Re-pin the command byte after any optimisation step."""
    x[pos, BYTE_COLS[CMD_BYTE_INDEX]] = CMD_VALUE / 255.0
    return x


def command_intact(x: np.ndarray, pos: int, tol: float = 0.5 / 255.0) -> bool:
    return abs(x[pos, BYTE_COLS[CMD_BYTE_INDEX]] - CMD_VALUE / 255.0) <= tol
