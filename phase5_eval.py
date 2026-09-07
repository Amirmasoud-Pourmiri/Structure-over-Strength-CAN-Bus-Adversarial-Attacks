"""Phase 5 -- deliver crafted windows and measure attack power.

For each (method, IDS) we craft `n_attacks` windows, push every one through the
delivery channel (loopback offline, SocketCAN on hardware), and record:

  stealth      = P(real IDS does not alert)
  effectiveness= P(command byte survived -> ECU would obey)
  success      = P(stealth AND effective)
  transfer     = P(real IDS silent | surrogate was silent)   [the transfer rate]
  craft_queries= mean surrogate queries spent optimising

Only `ids.alert` is ever consulted on the 'real' model, exactly as a black-box
attacker would experience it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from attack.context import (command_intact, embed_malicious, embed_naive,
                            make_carrier, middle_position, non_target_slot,
                            pick_target_slot)
from attack.phase4_craft import CONSTRAINED_METHODS, CRAFT_METHODS
from attack.phase3_surrogate import Ensemble
from deploy.can_io import LoopbackChannel

ALL_METHODS = {**CRAFT_METHODS, **CONSTRAINED_METHODS}

# which surrogate should lead the decision-only / single-surrogate methods,
# chosen to match each real IDS family (this is the Phase-1 family guess in use)
PRIMARY_BY_FAMILY = {
    "rule": "iforest",
    "statistical": "iforest",
    "lstm_ae": "ae",
    "hybrid": "ae",
}
GRADIENT_METHODS = {"M1_fgsm", "M2_pgd", "M3_cw", "M8_transfer", "M10_slowdrift"}


@dataclass
class MethodResult:
    method: str
    family: str
    stealth: float
    effectiveness: float
    success: float
    transfer: float
    craft_queries: float
    surrogate_stealth: float


def _order_surrogates(surrogates, family):
    primary = PRIMARY_BY_FAMILY.get(family, "ae")
    ordered = sorted(surrogates, key=lambda s: 0 if s.name == primary else 1)
    return ordered


def evaluate_method(method, surrogates, real_ids, pool_raw, rng,
                    n_attacks=60, gan=None, normal_target_frame=None,
                    target_byte_lohi=None) -> MethodResult:
    fn = ALL_METHODS[method]
    ordered = _order_surrogates(surrogates, real_ids.family)
    surr_for_stealth = Ensemble(surrogates) if method == "M8_transfer" else ordered[0]
    channel = LoopbackChannel()

    stealth, effect, succ, sur_ok, transfer_num, transfer_den, qs = 0, 0, 0, 0, 0, 0, []
    W = len(pool_raw[0])
    for _ in range(n_attacks):
        carrier = make_carrier(pool_raw, rng)
        pos = pick_target_slot(carrier)          # rate-preserving placement (M9)
        carrier = embed_malicious(carrier, pos, rng)

        if method == "M7_gan":
            x, info = fn(ordered, carrier, pos, rng, gan=gan)
        elif method == "M9_context":
            x, info = fn(ordered, carrier, pos, rng, normal_target_frame=normal_target_frame)
        elif method == "M12_pgd_legal":
            x, info = fn(ordered, carrier, pos, rng, target_byte_lohi=target_byte_lohi)
        else:
            x, info = fn(ordered, carrier, pos, rng)

        qs.append(info.get("queries", 0))
        eff = command_intact(x, pos)
        # deliver (records frames; on hardware this injects them)
        channel.send_window(x)

        if isinstance(surr_for_stealth, Ensemble):
            s_ok = not surr_for_stealth.alert_any(x)
        else:
            s_ok = not surr_for_stealth.alert(x)
        r_ok = not real_ids.alert(x)

        stealth += int(r_ok)
        effect += int(eff)
        succ += int(r_ok and eff)
        sur_ok += int(s_ok)
        if s_ok:
            transfer_den += 1
            transfer_num += int(r_ok)

    n = n_attacks
    return MethodResult(
        method=method, family=real_ids.family,
        stealth=stealth / n,
        effectiveness=effect / n,
        success=succ / n,
        transfer=(transfer_num / transfer_den) if transfer_den else 0.0,
        craft_queries=float(np.mean(qs)),
        surrogate_stealth=sur_ok / n,
    )


def naive_baseline(real_ids, pool_raw, rng, n_attacks=60) -> MethodResult:
    """Uncrafted injection: shows what the IDS catches before any attack effort."""
    channel = LoopbackChannel()
    W = len(pool_raw[0])
    stealth = effect = succ = 0
    for _ in range(n_attacks):
        carrier = make_carrier(pool_raw, rng)
        pos = non_target_slot(carrier)           # naive: adds a frame, disturbs rate
        x = embed_naive(carrier, pos, rng)       # arbitrary payload, no context repair
        channel.send_window(x)
        eff = command_intact(x, pos)
        r_ok = not real_ids.alert(x)
        stealth += int(r_ok); effect += int(eff); succ += int(r_ok and eff)
    n = n_attacks
    return MethodResult("M0_naive", real_ids.family, stealth / n, effect / n,
                        succ / n, 0.0, 0.0, 0.0)
