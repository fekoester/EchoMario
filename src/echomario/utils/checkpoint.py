from __future__ import annotations

from pathlib import Path
from typing import Any
import torch


def save_checkpoint(
    path: str | Path,
    *,
    config: dict[str, Any],
    policy_state: dict[str, Any],
    reservoir_state: dict[str, Any],
    global_step: int,
    stats: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "policy_state": policy_state,
        "reservoir_state": reservoir_state,
        "global_step": global_step,
        "stats": stats or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
