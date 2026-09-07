"""Phase 3 -- train local surrogates that imitate the IDS decision boundary.

The attacker fits these only on Phase-2 query labels, never on the real model.
We provide the four families named in the scenario (isolation forest, one-class
SVM, autoencoder, and a small Transformer encoder) plus an ensemble wrapper for
the multi-surrogate transfer attack (M8).  The autoencoder/Transformer expose a
differentiable `torch_score` so PGD/FGSM/C&W can run white-box on the surrogate.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from canbus import N_FEATURES


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


class Surrogate:
    differentiable = False
    name = "base"

    def fit(self, X_normal):
        """Fit the one-class model on captured normal traffic."""
        raise NotImplementedError

    def score(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.score(x) for x in X])

    def alert(self, x: np.ndarray) -> bool:
        return self.score(x) > self.threshold

    threshold: float = 0.0

    def calibrate(self, X_query, y_query):
        """Match the alert fraction the real IDS showed on the query set."""
        s = self.score_batch(X_query)
        frac = float(np.mean(y_query)) if len(y_query) else 0.05
        frac = min(max(frac, 0.01), 0.5)
        self.threshold = float(np.quantile(s, 1.0 - frac))


class IsolationForestSurrogate(Surrogate):
    name = "iforest"

    def __init__(self, seed=0):
        self.model = IsolationForest(n_estimators=120, random_state=seed, contamination="auto")

    def fit(self, X_normal):
        self.model.fit(_flatten(X_normal))
        return self

    def score(self, x):
        return float(-self.model.score_samples(_flatten(x[None]))[0])

    def score_batch(self, X):
        return -self.model.score_samples(_flatten(X))


class OCSVMSurrogate(Surrogate):
    name = "ocsvm"

    def __init__(self, seed=0):
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)

    def fit(self, X_normal):
        self.model.fit(_flatten(X_normal))
        return self

    def score(self, x):
        return float(-self.model.decision_function(_flatten(x[None]))[0])

    def score_batch(self, X):
        return -self.model.decision_function(_flatten(X))


class _AENet(nn.Module):
    def __init__(self, n_feat, hidden=36, latent=8):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        h, _ = self.enc(x)                  # per-frame bottleneck (see ids.lstm_ae)
        z = torch.tanh(self.to_z(h))
        d = torch.relu(self.from_z(z))
        dec, _ = self.dec(d)
        return torch.sigmoid(self.out(dec))


class AESurrogate(Surrogate):
    differentiable = True
    name = "ae"

    def __init__(self, seed=0, epochs=14, lr=2e-3, device="cpu"):
        self.seed, self.epochs, self.lr, self.device = seed, epochs, lr, device
        self.model = None

    def fit(self, X_normal):
        torch.manual_seed(self.seed)
        self.model = _AENet(N_FEATURES).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X = torch.tensor(X_normal, dtype=torch.float32, device=self.device)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), 64):
                xb = X[perm[i:i + 64]]
                opt.zero_grad()
                loss = ((self.model(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def torch_score(self, xt: torch.Tensor) -> torch.Tensor:
        per_frame = ((self.model(xt) - xt) ** 2).mean(dim=2)
        return per_frame.max(dim=1).values

    def score(self, x):
        with torch.no_grad():
            xt = torch.tensor(x[None], dtype=torch.float32, device=self.device)
            return float(self.torch_score(xt).item())

    def score_batch(self, X):
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.torch_score(xt).cpu().numpy()


class _TransformerNet(nn.Module):
    def __init__(self, n_feat, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d_model, n_feat)

    def forward(self, x):
        h = self.tr(self.inp(x))
        return torch.sigmoid(self.out(h))


class TransformerSurrogate(AESurrogate):
    differentiable = True
    name = "transformer"

    def fit(self, X_normal):
        torch.manual_seed(self.seed)
        self.model = _TransformerNet(N_FEATURES).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X = torch.tensor(X_normal, dtype=torch.float32, device=self.device)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), 64):
                xb = X[perm[i:i + 64]]
                opt.zero_grad()
                loss = ((self.model(xb) - xb) ** 2).mean()
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def torch_score(self, xt):
        per_frame = ((self.model(xt) - xt) ** 2).mean(dim=2)
        return per_frame.max(dim=1).values


class Ensemble:
    """Multi-surrogate wrapper for the transfer attack (M8)."""

    def __init__(self, members: List[Surrogate]):
        self.members = members

    def torch_members(self):
        return [m for m in self.members if getattr(m, "differentiable", False)]

    def alert_any(self, x) -> bool:
        return any(m.alert(x) for m in self.members)

    def mean_norm_score(self, x) -> float:
        return float(np.mean([m.score(x) / (m.threshold + 1e-9) for m in self.members]))


SURROGATE_FAMILIES = {
    "iforest": IsolationForestSurrogate,
    "ocsvm": OCSVMSurrogate,
    "ae": AESurrogate,
    "transformer": TransformerSurrogate,
}


def train_surrogates(names, X_normal, X_query, y_query, seed=0) -> List[Surrogate]:
    """Fit each surrogate on the attacker's captured normal traffic, then
    calibrate its threshold against the Phase-2 probe labels."""
    out = []
    for nm in names:
        s = SURROGATE_FAMILIES[nm](seed=seed)
        s.fit(X_normal)
        s.calibrate(X_query, y_query)
        out.append(s)
    return out
