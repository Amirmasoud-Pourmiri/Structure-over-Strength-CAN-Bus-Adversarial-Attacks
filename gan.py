"""Small GAN used by method M7 to synthesise realistic normal CAN windows.

A fully-connected generator/discriminator pair over flattened windows.  It is
intentionally light (a few hundred steps) -- enough to learn the marginal byte
structure of normal traffic so the carrier the attacker embeds into matches the
real distribution rather than looking artificially clean.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from canbus import N_FEATURES


class _Gen(nn.Module):
    def __init__(self, zdim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class _Disc(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)


class WindowGAN:
    def __init__(self, window_len, seed=0, zdim=24, device="cpu"):
        self.window_len = window_len
        self.zdim = zdim
        self.device = device
        self.dim = window_len * N_FEATURES
        torch.manual_seed(seed)
        self.G = _Gen(zdim, self.dim).to(device)
        self.D = _Disc(self.dim).to(device)

    def fit(self, X_normal: np.ndarray, steps=400, bs=64, lr=2e-4):
        X = torch.tensor(X_normal.reshape(len(X_normal), -1), dtype=torch.float32, device=self.device)
        optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        for _ in range(steps):
            idx = torch.randint(0, X.size(0), (bs,))
            real = X[idx]
            z = torch.randn(bs, self.zdim, device=self.device)
            fake = self.G(z).detach()
            optD.zero_grad()
            lossD = bce(self.D(real), torch.ones(bs, 1)) + bce(self.D(fake), torch.zeros(bs, 1))
            lossD.backward(); optD.step()
            z = torch.randn(bs, self.zdim, device=self.device)
            optG.zero_grad()
            lossG = bce(self.D(self.G(z)), torch.ones(bs, 1))
            lossG.backward(); optG.step()
        self.G.eval()
        return self

    def sample_window(self, rng) -> np.ndarray:
        z = torch.randn(1, self.zdim, generator=torch.Generator().manual_seed(int(rng.integers(1, 1e9))))
        with torch.no_grad():
            w = self.G(z).numpy().reshape(self.window_len, N_FEATURES)
        return w
