"""Single-model ML IDS: an LSTM autoencoder (the 'Med-High' family).

Trains on clean windows to reconstruct them; a window with an injected frame
reconstructs poorly and scores high.  Being a differentiable neural net, this
is the family that white-box-on-the-surrogate gradient methods (FGSM/PGD/C&W)
target, and the one whose gradients transfer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from canbus import N_FEATURES
from ids.base import IDS


class _AE(nn.Module):
    """Sequence autoencoder with a per-frame bottleneck.

    A single latent vector for a whole 32-frame window (different ids on every
    row) is far too lossy to reconstruct the bytes, so the baseline error stays
    high and the detector goes blind.  Keeping a small per-frame latent lets the
    model reconstruct normal, id-conditioned byte patterns accurately while a
    frame whose bytes leave the learned envelope still reconstructs poorly.
    """

    def __init__(self, n_feat: int, hidden: int = 40, latent: int = 8):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        h, _ = self.enc(x)                  # (B, W, hidden)
        z = torch.tanh(self.to_z(h))        # (B, W, latent) -- per-frame bottleneck
        d = torch.relu(self.from_z(z))      # (B, W, hidden)
        dec, _ = self.dec(d)
        return torch.sigmoid(self.out(dec))


class LSTMAutoencoderIDS(IDS):
    family = "lstm_ae"

    def __init__(self, epochs: int = 18, lr: float = 2e-3, device: str = "cpu", seed: int = 0):
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.seed = seed
        self.model: _AE | None = None

    def fit(self, X_train: np.ndarray, X_cal: np.ndarray) -> "LSTMAutoencoderIDS":
        torch.manual_seed(self.seed)
        self.model = _AE(N_FEATURES).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.model.train()
        bs = 64
        for _ in range(self.epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), bs):
                idx = perm[i:i + bs]
                xb = X[idx]
                opt.zero_grad()
                loss = loss_fn(self.model(xb), xb)
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def _recon_err(self, X: torch.Tensor) -> torch.Tensor:
        # frame-level score: the worst-reconstructed frame in the window.  A
        # single injected frame is not diluted by the surrounding normal frames.
        out = self.model(X)
        per_frame = ((out - X) ** 2).mean(dim=2)   # (B, W)
        return per_frame.max(dim=1).values

    def score(self, x: np.ndarray) -> float:
        with torch.no_grad():
            xt = torch.tensor(x[None], dtype=torch.float32, device=self.device)
            return float(self._recon_err(xt).item())

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self._recon_err(xt).cpu().numpy()

    # exposed for white-box surrogate gradient attacks
    def torch_score(self, xt: torch.Tensor) -> torch.Tensor:
        return self._recon_err(xt)
