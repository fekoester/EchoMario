from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ResetSpec:
    seed: int | None = None
    options: dict[str, Any] | None = None

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.options is not None:
            kwargs["options"] = self.options
        return kwargs


class EpisodeSeedManager:
    def __init__(self, config: dict[str, Any]):
        project_cfg = config.get("project", {})
        env_cfg = config.get("env", {})

        self.project_seed = int(project_cfg.get("seed", 42))
        self.env_name = str(env_cfg.get("name", ""))
        self.randomize = bool(env_cfg.get("randomize", False))
        self.train_seed_mode = str(env_cfg.get("train_seed_mode", "random" if self.randomize else "fixed"))
        self.fixed_level_seed = env_cfg.get("level_seed", self.project_seed)
        self.eval_seed_start = int(env_cfg.get("eval_seed_start", 10000))
        self.num_eval_seeds = int(env_cfg.get("num_eval_seeds", 20))
        self.rng = np.random.default_rng(self.project_seed)

    def _toy_level_reset(self, level_seed: int | None) -> ResetSpec:
        if level_seed is None:
            return ResetSpec()
        return ResetSpec(options={"level_seed": int(level_seed)})

    def training_reset_spec(self) -> ResetSpec:
        if self.env_name == "toy_platformer" and self.randomize:
            if self.train_seed_mode == "random":
                level_seed = int(self.rng.integers(0, 2**31 - 1))
                return self._toy_level_reset(level_seed)
            if self.train_seed_mode == "fixed":
                return self._toy_level_reset(int(self.fixed_level_seed))
            raise ValueError(f"Unsupported train_seed_mode: {self.train_seed_mode}")

        if self.train_seed_mode == "fixed":
            return ResetSpec(seed=self.project_seed)
        if self.train_seed_mode == "random":
            seed = int(self.rng.integers(0, 2**31 - 1))
            return ResetSpec(seed=seed)
        raise ValueError(f"Unsupported train_seed_mode: {self.train_seed_mode}")

    def evaluation_reset_specs(self) -> list[ResetSpec]:
        specs: list[ResetSpec] = []
        for offset in range(self.num_eval_seeds):
            eval_seed = self.eval_seed_start + offset
            if self.env_name == "toy_platformer" and self.randomize:
                specs.append(self._toy_level_reset(eval_seed))
            else:
                specs.append(ResetSpec(seed=eval_seed))
        return specs
