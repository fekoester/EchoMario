from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import sparse

# The sparse input projection reads one W_in row per non-zero input, the dense one reads all of
# W_in once per batch. Use the sparse path while the batch has fewer non-zeros than this fraction
# of the input dimension (one tile-observation env is ~10 % non-zero).
_SPARSE_INPUT_MAX_NONZERO_FRACTION = 0.3


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
    device: str = "cpu"
    # Multi-timescale reservoir: when both are set, every neuron gets its own leak rate drawn
    # log-uniformly from [leak_rate_min, leak_rate_max] (fast neurons track velocity, slow ones context).
    leak_rate_min: float | None = None
    leak_rate_max: float | None = None
    # Input layout. 'dense': every neuron sees a random mix of the whole observation.
    # 'local': every neuron sees one random (2r+1)x(2r+1) patch of the tile screen (all channels),
    # like a random convolution feature; the camera follows the player, so a patch is a fixed offset
    # from the player. Extra state features (after the screen) still connect densely.
    input_structure: str = "dense"
    screen_shape: tuple[int, int, int] | None = None  # (rows, cols, channels), needed for 'local'
    patch_radius: int = 2


class SparseESN:
    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.W_in = self.rng.normal(
            loc=0.0,
            scale=cfg.input_scale / np.sqrt(max(1, cfg.input_dim)),
            size=(cfg.size, cfg.input_dim),
        ).astype(np.float32)
        if cfg.input_structure == "local":
            self.W_in = self._make_local_input(cfg)
        elif cfg.input_structure != "dense":
            raise ValueError(f"unknown reservoir.input_structure {cfg.input_structure!r}")
        self.W_res = self._make_sparse_reservoir(cfg.size, cfg.sparsity, cfg.spectral_radius)
        self.leak = self._make_leak(cfg)
        self.x = np.zeros((1, cfg.size), dtype=np.float32)
        self._prepare_fast_path()

    def _prepare_fast_path(self) -> None:
        """
        Caches a CSR copy of the recurrent matrix (it is ~1 % dense), a row-major W_in.T for the
        sparse input path and a torch copy of W_in.T (optionally on GPU) for the dense path.
        """
        self._W_res_csr = sparse.csr_matrix(self.W_res)
        self._W_in_T = np.ascontiguousarray(self.W_in.T)
        self._torch_device = torch.device(getattr(self.cfg, "device", "cpu") or "cpu")
        self._W_in_T_torch = torch.from_numpy(self._W_in_T).to(self._torch_device)

    def _input_projection(self, u: np.ndarray) -> np.ndarray:
        """Computes u @ W_in.T, skipping the zero entries of sparse (one-hot tile) observations."""
        on_cpu = self._torch_device.type == "cpu"
        if on_cpu and np.count_nonzero(u) <= _SPARSE_INPUT_MAX_NONZERO_FRACTION * u.shape[1]:
            return np.asarray(sparse.csr_matrix(u) @ self._W_in_T, dtype=np.float32)
        u_t = torch.from_numpy(np.ascontiguousarray(u)).to(self._torch_device, non_blocking=True)
        return (u_t @ self._W_in_T_torch).cpu().numpy()

    def _recurrent_projection(self, x: np.ndarray) -> np.ndarray:
        """Computes x @ W_res.T using the sparse recurrent matrix."""
        return np.asarray((self._W_res_csr @ x.T).T, dtype=np.float32)

    @staticmethod
    def _make_local_input(cfg: ESNConfig) -> np.ndarray:
        if cfg.screen_shape is None:
            raise ValueError("input_structure 'local' needs screen_shape (rows, cols, channels)")
        rows, cols, channels = (int(v) for v in cfg.screen_shape)
        screen_dim = rows * cols * channels
        if screen_dim > cfg.input_dim:
            raise ValueError(f"screen_shape {cfg.screen_shape} is larger than input_dim {cfg.input_dim}")
        rng = np.random.default_rng(cfg.seed + 104729)
        W = np.zeros((cfg.size, cfg.input_dim), dtype=np.float32)
        r = int(cfg.patch_radius)
        for n in range(cfg.size):
            r0, c0 = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            rr = np.arange(max(0, r0 - r), min(rows, r0 + r + 1))
            cc = np.arange(max(0, c0 - r), min(cols, c0 + r + 1))
            cells = (rr[:, None] * cols + cc[None, :]).reshape(-1)
            idx = (cells[:, None] * channels + np.arange(channels)[None, :]).reshape(-1)
            # One channel is active per cell, so the drive has std ~ input_scale.
            W[n, idx] = rng.normal(0.0, cfg.input_scale / np.sqrt(len(cells)), size=len(idx))
        extra = cfg.input_dim - screen_dim
        if extra > 0:
            W[:, screen_dim:] = rng.normal(0.0, cfg.input_scale / np.sqrt(extra), size=(cfg.size, extra))
        return W

    def _make_leak(self, cfg: ESNConfig) -> float | np.ndarray:
        """Scalar leak rate, or one leak rate per neuron for a multi-timescale reservoir."""
        lo, hi = cfg.leak_rate_min, cfg.leak_rate_max
        if lo is None or hi is None:
            return float(cfg.leak_rate)
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError('need 0 < leak_rate_min <= leak_rate_max <= 1')
        # Separate generator so W_in / W_res stay identical to a single-timescale reservoir.
        rng = np.random.default_rng(cfg.seed + 7919)
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=cfg.size)).astype(np.float32)

    def _next_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        new_x = np.tanh(self._recurrent_projection(x) + self._input_projection(u))
        return ((1.0 - self.leak) * x + self.leak * new_x).astype(np.float32)

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
        self.x = self._next_state(self.x, u)
        return self._normalize_state(self.x).copy()

    def peek_batch(self, u: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
        """
        Returns the (normalized) state the reservoirs at ``indices`` would have after receiving
        ``u``, without changing the stored state. Used to value terminal observations.
        """
        u = np.asarray(u, dtype=np.float32).reshape(len(indices), -1)
        idx = np.asarray(indices, dtype=np.int64)
        return self._normalize_state(self._next_state(self.x[idx], u))

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
    def from_state_dict(cls, state: dict[str, Any]) -> SparseESN:
        cfg_dict = dict(state["cfg"])
        if cfg_dict.get("screen_shape") is not None:
            cfg_dict["screen_shape"] = tuple(cfg_dict["screen_shape"])
        if str(cfg_dict.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            cfg_dict["device"] = "cpu"
        cfg = ESNConfig(**cfg_dict)
        obj = cls(cfg)
        obj.W_in = state["W_in"].astype(np.float32)
        obj.W_res = np.asarray(state["W_res"], dtype=np.float32)
        obj._prepare_fast_path()
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

    def peek_batch(self, u: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
        return np.asarray(u, dtype=np.float32).reshape(len(indices), -1).copy()

    def state_dict(self) -> dict[str, Any]:
        return {"type": "none", "cfg": self.cfg.__dict__}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> IdentityReservoir:
        cfg = state.get("cfg", {})
        input_dim = int(cfg["input_dim"])
        return cls(input_dim=input_dim)


ReservoirLike = SparseESN | IdentityReservoir


def load_reservoir_from_state_dict(state: dict[str, Any]) -> ReservoirLike:
    reservoir_type = state.get("type", state.get("cfg", {}).get("type", "sparse_esn"))
    if reservoir_type == "none":
        return IdentityReservoir.from_state_dict(state)
    return SparseESN.from_state_dict(state)


def screen_shape_from_config(config: dict[str, Any]) -> tuple[int, int, int] | None:
    """(rows, cols, channels) of the toy platformer's tile screen, or None for other envs."""
    env_cfg = config.get("env", {})
    if env_cfg.get("name") == "smw":
        from echomario.envs.smw_env import smw_screen_shape

        return smw_screen_shape(config)
    if env_cfg.get("name") != "toy_platformer" or env_cfg.get("observation_mode", "full_screen") != "full_screen":
        return None
    from echomario.envs.toy_platformer_env import ToyPlatformerEnv

    channels = len(ToyPlatformerEnv.SCREEN_CHANNEL_NAMES) * int(env_cfg.get("frame_stack", 1))
    return int(env_cfg.get("height", 18)), int(env_cfg.get("camera_width", 36)), channels


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
        device=str(r_cfg.get("device", "cpu")),
        leak_rate_min=None if r_cfg.get("leak_rate_min") is None else float(r_cfg["leak_rate_min"]),
        leak_rate_max=None if r_cfg.get("leak_rate_max") is None else float(r_cfg["leak_rate_max"]),
        input_structure=str(r_cfg.get("input_structure", "dense")),
        screen_shape=screen_shape_from_config(config) if r_cfg.get("input_structure") == "local" else None,
        patch_radius=int(r_cfg.get("patch_radius", 2)),
    )
    return SparseESN(cfg)
