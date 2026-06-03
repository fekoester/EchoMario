from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class ESNConfig:
    input_dim: int
    type: str = "sparse_esn"
    size: int = 500
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    input_scale: float = 0.5
    sparsity: float = 0.02
    seed: int = 123
    state_norm: str = "rms"


class SparseESN:
    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.W_in = self.rng.normal(
            loc=0.0,
            scale=cfg.input_scale / np.sqrt(max(1, cfg.input_dim)),
            size=(cfg.size, cfg.input_dim),
        ).astype(np.float32)
        self.W_res = self._make_sparse_reservoir(cfg.size, cfg.sparsity, cfg.spectral_radius)
        self.x = np.zeros((1, cfg.size), dtype=np.float32)

    def _make_sparse_reservoir(self, size: int, sparsity: float, spectral_radius: float) -> np.ndarray:
        W = np.zeros((size, size), dtype=np.float32)
        n_connections = int(size * size * sparsity)
        rows = self.rng.integers(0, size, size=n_connections)
        cols = self.rng.integers(0, size, size=n_connections)
        vals = self.rng.normal(0.0, 1.0 / np.sqrt(max(1, size * sparsity)), size=n_connections)
        W[rows, cols] = vals.astype(np.float32)
        radius = self._estimate_spectral_radius(W)
        if radius > 1e-8:
            W *= spectral_radius / radius
        return W.astype(np.float32)

    @staticmethod
    def _estimate_spectral_radius(W: np.ndarray, n_iter: int = 60) -> float:
        rng = np.random.default_rng(0)
        v = rng.normal(size=W.shape[0]).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        for _ in range(n_iter):
            v = W @ v
            norm = np.linalg.norm(v) + 1e-8
            v = v / norm
        wv = W @ v
        return float(np.linalg.norm(wv))

    def _ensure_batch_size(self, batch_size: int) -> None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self.x.shape[0] != batch_size:
            self.x = np.zeros((batch_size, self.cfg.size), dtype=np.float32)

    def _normalize_state(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.state_norm == 'rms':
            rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True)) + 1e-6
            return (x / rms).astype(np.float32)
        return x.astype(np.float32)

    def reset(self, mask: np.ndarray | list[int] | None = None, batch_size: int | None = None) -> None:
        if batch_size is not None:
            self._ensure_batch_size(batch_size)
        if mask is None:
            self.x.fill(0.0)
            return
        idx = np.asarray(mask, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return
        if idx.max(initial=-1) >= self.x.shape[0]:
            self._ensure_batch_size(int(idx.max()) + 1)
        self.x[idx] = 0.0

    def step_batch(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float32)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.ndim != 2 or u.shape[1] != self.cfg.input_dim:
            raise ValueError(f"Expected batch of shape (N, {self.cfg.input_dim}), got {u.shape}")
        self._ensure_batch_size(u.shape[0])
        pre = self.x @ self.W_res.T + u @ self.W_in.T
        new_x = np.tanh(pre).astype(np.float32)
        self.x = ((1.0 - self.cfg.leak_rate) * self.x + self.cfg.leak_rate * new_x).astype(np.float32)
        return self._normalize_state(self.x).copy()

    def step(self, u: np.ndarray) -> np.ndarray:
        return self.step_batch(np.asarray(u, dtype=np.float32).reshape(1, -1))[0]

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "sparse_esn",
            "cfg": self.cfg.__dict__,
            "W_in": self.W_in,
            "W_res": self.W_res,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SparseESN":
        cfg = ESNConfig(**state["cfg"])
        obj = cls(cfg)
        obj.W_in = state["W_in"].astype(np.float32)
        obj.W_res = state["W_res"].astype(np.float32)
        obj.reset()
        return obj


class IdentityReservoir:
    def __init__(self, input_dim: int):
        self.cfg = ESNConfig(input_dim=input_dim, type="none", size=input_dim)
        self.x = np.zeros((1, input_dim), dtype=np.float32)

    def _ensure_batch_size(self, batch_size: int) -> None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self.x.shape[0] != batch_size:
            self.x = np.zeros((batch_size, self.cfg.input_dim), dtype=np.float32)

    def reset(self, mask: np.ndarray | list[int] | None = None, batch_size: int | None = None) -> None:
        if batch_size is not None:
            self._ensure_batch_size(batch_size)
        if mask is None:
            self.x.fill(0.0)
            return
        idx = np.asarray(mask, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return
        if idx.max(initial=-1) >= self.x.shape[0]:
            self._ensure_batch_size(int(idx.max()) + 1)
        self.x[idx] = 0.0

    def step_batch(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float32)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.ndim != 2 or u.shape[1] != self.cfg.input_dim:
            raise ValueError(f"Expected batch of shape (N, {self.cfg.input_dim}), got {u.shape}")
        self._ensure_batch_size(u.shape[0])
        self.x = u.astype(np.float32, copy=True)
        return self.x.copy()

    def step(self, u: np.ndarray) -> np.ndarray:
        return self.step_batch(np.asarray(u, dtype=np.float32).reshape(1, -1))[0]

    def state_dict(self) -> dict[str, Any]:
        return {"type": "none", "cfg": self.cfg.__dict__}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "IdentityReservoir":
        cfg = state.get("cfg", {})
        input_dim = int(cfg["input_dim"])
        return cls(input_dim=input_dim)


ReservoirLike = SparseESN | IdentityReservoir


def load_reservoir_from_state_dict(state: dict[str, Any]) -> ReservoirLike:
    reservoir_type = state.get("type", state.get("cfg", {}).get("type", "sparse_esn"))
    if reservoir_type == "none":
        return IdentityReservoir.from_state_dict(state)
    return SparseESN.from_state_dict(state)


def make_reservoir(config: dict[str, Any], input_dim: int) -> ReservoirLike:
    r_cfg = config["reservoir"]
    reservoir_type = str(r_cfg.get("type", "sparse_esn"))
    if reservoir_type == "none":
        return IdentityReservoir(input_dim=input_dim)

    cfg = ESNConfig(
        input_dim=input_dim,
        type=reservoir_type,
        size=int(r_cfg.get("size", 500)),
        spectral_radius=float(r_cfg.get("spectral_radius", 0.9)),
        leak_rate=float(r_cfg.get("leak_rate", 0.5)),
        input_scale=float(r_cfg.get("input_scale", 0.5)),
        sparsity=float(r_cfg.get("sparsity", 0.02)),
        seed=int(r_cfg.get("seed", 123)),
        state_norm=str(r_cfg.get("state_norm", "rms")),
    )
    return SparseESN(cfg)
