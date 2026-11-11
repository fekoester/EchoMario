from __future__ import annotations

import torch
from torch import nn

from echomario.agents.base import ActorCriticBase


class ReadoutActorCritic(ActorCriticBase):
    def __init__(
        self,
        reservoir_dim: int,
        num_actions: int,
        continuous: bool = False,
        log_std_init: float = -1.0,
        min_log_std: float = -3.0,
        max_log_std: float = 0.0,
    ):
        super().__init__()

        self.continuous = bool(continuous)
        self.num_actions = int(num_actions)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.policy = nn.Linear(reservoir_dim, num_actions)
        self.value = nn.Linear(reservoir_dim, 1)

        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

        if self.continuous:
            self.log_std = nn.Parameter(torch.full((num_actions,), float(log_std_init)))
        else:
            self.log_std = None

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_out = self.policy(states)
        values = self.value(states).squeeze(-1)
        return policy_out, values
