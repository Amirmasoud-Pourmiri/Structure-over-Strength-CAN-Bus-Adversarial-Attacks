"""Phase 4 -- craft a CAN window that carries the command yet scores normal.

Every method here optimises only the 'free' features (companion-frame bytes and
the non-command bytes of the malicious frame); arbitration ids, DLCs, and the
command byte itself are pinned by the context layer so the result stays a legal,
effective attack.  Each method returns the crafted window plus a small info dict
(query count, final surrogate margin) used by the evaluation harness.

Methods: M1 FGSM, M2 PGD, M3 C&W, M4 boundary, M5 CMA-ES, M6 NES,
M7 GAN, M8 transfer-ensemble, M9 context-only, M10 slow-drift.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from attack.context import (command_intact, command_mask, embed_malicious,
                            enforce_command, make_carrier, middle_position)
from attack.phase3_surrogate import AESurrogate, Ensemble, Surrogate


def _free_mask(window_len: int, pos: int) -> np.ndarray:
    """Features we are allowed to change = everything except the command mask."""
    return ~command_mask(window_len, pos)


def _primary_diff(surrogates: List[Surrogate]) -> AESurrogate:
    for s in surrogates:
        if getattr(s, "differentiable", False):
            return s
    raise ValueError("no differentiable surrogate available")


# --- M1 FGSM ----------------------------------------------------------------
def craft_fgsm(surrogates, carrier, pos, rng, eps=0.08):
    s = _primary_diff(surrogates)
    free = _free_mask(carrier.shape[0], pos)
    xt = torch.tensor(carrier[None], dtype=torch.float32, requires_grad=True)
    loss = s.torch_score(xt).sum()
    loss.backward()
    grad = xt.grad.detach().numpy()[0]
    x = carrier - eps * np.sign(grad) * free
    x = np.clip(x, 0, 1)
    enforce_command(x, pos)
    return x, {"queries": 1, "margin": s.score(x) - s.threshold}


# --- M2 PGD -----------------------------------------------------------------
def craft_pgd(surrogates, carrier, pos, rng, steps=40, alpha=0.05, eps=1.0):
    s = _primary_diff(surrogates)
    free = _free_mask(carrier.shape[0], pos)
    x = carrier.copy()
    q = 0
    for _ in range(steps):
        xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
        loss = s.torch_score(xt).sum()
        loss.backward()
        grad = xt.grad.detach().numpy()[0]
        x = x - alpha * np.sign(grad) * free
        x = np.clip(carrier + np.clip(x - carrier, -eps, eps), 0, 1)
        enforce_command(x, pos)
        q += 1
        if s.score(x) <= s.threshold:
            break
    return x, {"queries": q, "margin": s.score(x) - s.threshold}


# --- M3 Carlini-Wagner ------------------------------------------------------
def craft_cw(surrogates, carrier, pos, rng, steps=60, c=1.0, lr=0.02, kappa=0.0):
    s = _primary_diff(surrogates)
    free = torch.tensor(_free_mask(carrier.shape[0], pos), dtype=torch.float32)
    base = torch.tensor(carrier, dtype=torch.float32)
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    thr = s.threshold
    for _ in range(steps):
        x = torch.clamp(base + delta * free, 0, 1)
        score = s.torch_score(x[None])[0]
        l2 = (delta * free).pow(2).sum()
        loss = l2 + c * torch.clamp(score - thr + kappa, min=0)
        opt.zero_grad(); loss.backward(); opt.step()
    x = torch.clamp(base + delta.detach() * free, 0, 1).numpy()
    enforce_command(x, pos)
    return x, {"queries": steps, "margin": s.score(x) - thr}


# --- M4 Boundary (decision-only) -------------------------------------------
def craft_boundary(surrogates, carrier, pos, rng, budget=120):
    s = surrogates[0]                      # uses only the boolean alert()
    free = _free_mask(carrier.shape[0], pos)
    x = carrier.copy()
    enforce_command(x, pos)
    best = x.copy()
    best_ok = not s.alert(best)
    q = 0
    sigma = 0.15
    for _ in range(budget):
        cand = best + rng.normal(0, sigma, size=best.shape) * free
        cand = np.clip(cand, 0, 1)
        enforce_command(cand, pos)
        q += 1
        if not s.alert(cand):
            best, best_ok = cand, True
            sigma *= 0.97                  # tighten once inside the normal region
        else:
            sigma *= 1.01
        sigma = float(np.clip(sigma, 0.01, 0.4))
    return best, {"queries": q, "margin": s.score(best) - s.threshold, "ok": best_ok}


# --- M5 CMA-ES --------------------------------------------------------------
def craft_cma(surrogates, carrier, pos, rng, budget=160):
    import cma
    s = surrogates[0]
    free = _free_mask(carrier.shape[0], pos)
    idx = np.argwhere(free)
    # optimise a compact subset: the malicious frame + its two neighbours
    keep = [tuple(i) for i in idx if abs(i[0] - pos) <= 1]
    x0 = np.array([carrier[i, j] for i, j in keep])
    es = cma.CMAEvolutionStrategy(
        x0.tolist(), 0.2,
        {"bounds": [0, 1], "maxfevals": budget, "verbose": -9,
         "seed": int(rng.integers(1, 1_000_000))})
    q = 0

    def build(vec):
        x = carrier.copy()
        for (i, j), v in zip(keep, vec):
            x[i, j] = v
        enforce_command(x, pos)
        return x

    while not es.stop():
        sols = es.ask()
        batch = np.stack([build(np.clip(v, 0, 1)) for v in sols])
        fits = s.score_batch(batch)               # one vectorised call per generation
        es.tell(sols, list(map(float, fits)))
        q += len(sols)
    x = build(np.clip(es.result.xbest, 0, 1))
    return x, {"queries": q, "margin": s.score(x) - s.threshold}


# --- M6 NES (estimated gradients) ------------------------------------------
def craft_nes(surrogates, carrier, pos, rng, steps=12, pop=12, sigma=0.1, lr=0.05):
    from canbus import BYTE_COLS
    from dataset import CMD_BYTE_INDEX
    s = surrogates[0]
    free = _free_mask(carrier.shape[0], pos)
    cmd_col = BYTE_COLS[CMD_BYTE_INDEX]
    x = carrier.copy()
    q = 0
    for _ in range(steps):
        noise = rng.normal(0, 1, size=(pop,) + x.shape) * free
        plus = np.clip(x[None] + sigma * noise, 0, 1)
        minus = np.clip(x[None] - sigma * noise, 0, 1)
        for arr in (plus, minus):
            arr[:, pos, cmd_col] = carrier[pos, cmd_col]
        fp = s.score_batch(plus)                  # vectorised antithetic sampling
        fm = s.score_batch(minus)
        grad = ((fp - fm)[:, None, None] * noise).sum(axis=0) / (2 * pop * sigma)
        q += 2 * pop
        x = np.clip(x - lr * grad * free, 0, 1)
        enforce_command(x, pos)
        if s.score(x) <= s.threshold:
            break
    return x, {"queries": q, "margin": s.score(x) - s.threshold}


# --- M7 GAN-based realistic carrier -----------------------------------------
def craft_gan(surrogates, carrier, pos, rng, gan=None, refine=15):
    """Adopt GAN-generated payload bytes (keeping ids/dlc/timing/command from the
    real carrier), then NES-refine so the window blends into the surrogate's
    notion of normal."""
    x = carrier.copy()
    if gan is not None:
        free = _free_mask(carrier.shape[0], pos)
        g = np.clip(gan.sample_window(rng), 0, 1)
        x = np.where(free, g, x)
    enforce_command(x, pos)
    return craft_nes(surrogates, x, pos, rng, steps=refine, pop=12)


# --- M8 Transfer attack (multi-surrogate ensemble) --------------------------
def craft_transfer(surrogates, carrier, pos, rng, steps=40, alpha=0.05, eps=1.0):
    ens = Ensemble(surrogates)
    diff = ens.torch_members()
    free = _free_mask(carrier.shape[0], pos)
    x = carrier.copy()
    q = 0
    for _ in range(steps):
        xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
        loss = sum((m.torch_score(xt) / (m.threshold + 1e-9)).sum() for m in diff)
        loss.backward()
        grad = xt.grad.detach().numpy()[0]
        x = x - alpha * np.sign(grad) * free
        x = np.clip(carrier + np.clip(x - carrier, -eps, eps), 0, 1)
        enforce_command(x, pos)
        q += 1
        if not ens.alert_any(x):
            break
    return x, {"queries": q, "margin": ens.mean_norm_score(x) - 1.0}


# --- M9 Context engineering only --------------------------------------------
def craft_context(surrogates, carrier, pos, rng, normal_target_frame=None):
    """Replay a real recorded target frame into the slot and only flip the
    command byte -- the companion frames are already exact replays, so the whole
    window is normal by construction apart from the pinned command."""
    from canbus import BYTE_COLS
    x = carrier.copy()
    if normal_target_frame is not None:
        x[pos, BYTE_COLS] = normal_target_frame
    enforce_command(x, pos)
    return x, {"queries": 0, "margin": surrogates[0].score(x) - surrogates[0].threshold}


# --- M10 Slow-drift spoofing -------------------------------------------------
def craft_slow_drift(surrogates, carrier, pos, rng, steps=70, alpha=0.02, eps=1.0):
    """Many tiny steps -> the smallest perturbation that crosses the boundary."""
    return craft_pgd(surrogates, carrier, pos, rng, steps=steps, alpha=alpha, eps=eps)


# --- M11 Constrained PGD: projection onto the integer byte grid -------------
def craft_pgd_grid(surrogates, carrier, pos, rng, steps=40, alpha=0.05, eps=1.0):
    """PGD that, at every step, snaps the free payload bytes onto the legal
    integer grid (0..255, i.e. multiples of 1/255).  This is the reviewer's
    "projection onto the integer grid of legal byte values" variant; it isolates
    the *discreteness* gap from the *range-legality* gap (cf. M12).  The command
    byte is pinned as in every other method."""
    s = _primary_diff(surrogates)
    free = _free_mask(carrier.shape[0], pos)
    x = carrier.copy()
    q = 0
    for _ in range(steps):
        xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
        loss = s.torch_score(xt).sum()
        loss.backward()
        grad = xt.grad.detach().numpy()[0]
        x = x - alpha * np.sign(grad) * free
        x = np.clip(carrier + np.clip(x - carrier, -eps, eps), 0, 1)
        # discrete projection: round free bytes onto the 0..255 integer grid
        x = np.where(free, np.round(x * 255.0) / 255.0, x)
        enforce_command(x, pos)
        q += 1
        if s.score(x) <= s.threshold:
            break
    return x, {"queries": q, "margin": s.score(x) - s.threshold}


# --- M12 Constrained PGD: projection onto the legal byte envelope -----------
def craft_pgd_legal(surrogates, carrier, pos, rng, target_byte_lohi=None,
                    steps=40, alpha=0.05, eps=1.0):
    """The strongest constrained attacker: at every step, project the free
    payload bytes onto the per-byte legal envelope the attacker has estimated
    for the target id from her own recorded normal traffic, then quantise onto
    the integer grid.  `target_byte_lohi` is a pair of length-N_BYTES arrays
    (lo, hi) in normalised [0, 1] units; if None this degrades to M11."""
    from canbus import BYTE_COLS
    from dataset import CMD_BYTE_INDEX
    s = _primary_diff(surrogates)
    free = _free_mask(carrier.shape[0], pos)
    lo, hi = (None, None) if target_byte_lohi is None else target_byte_lohi
    x = carrier.copy()
    q = 0
    for _ in range(steps):
        xt = torch.tensor(x[None], dtype=torch.float32, requires_grad=True)
        loss = s.torch_score(xt).sum()
        loss.backward()
        grad = xt.grad.detach().numpy()[0]
        x = x - alpha * np.sign(grad) * free
        x = np.clip(carrier + np.clip(x - carrier, -eps, eps), 0, 1)
        if lo is not None:
            for k, col in enumerate(BYTE_COLS):
                if k == CMD_BYTE_INDEX:
                    continue
                x[pos, col] = float(np.clip(x[pos, col], lo[k], hi[k]))
        # quantise onto the integer grid after the legal-range projection
        x = np.where(free, np.round(x * 255.0) / 255.0, x)
        enforce_command(x, pos)
        q += 1
        if s.score(x) <= s.threshold:
            break
    return x, {"queries": q, "margin": s.score(x) - s.threshold}


# The ten methods that make up the head-to-head main grid (Table II).
CRAFT_METHODS = {
    "M1_fgsm": craft_fgsm,
    "M2_pgd": craft_pgd,
    "M3_cw": craft_cw,
    "M4_boundary": craft_boundary,
    "M5_cmaes": craft_cma,
    "M6_nes": craft_nes,
    "M7_gan": craft_gan,
    "M8_transfer": craft_transfer,
    "M9_context": craft_context,
    "M10_slowdrift": craft_slow_drift,
}

# Constrained-optimization ablation methods (Section: constrained-PGD ablation).
# Kept out of CRAFT_METHODS so the main 10x4 grid is unchanged; the standalone
# experiments/run_constrained.py driver runs these directly.
CONSTRAINED_METHODS = {
    "M11_pgd_grid": craft_pgd_grid,
    "M12_pgd_legal": craft_pgd_legal,
}
