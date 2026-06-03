from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class ReadoutActorCritic(nn.Module):
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

    def _continuous_dist(self, mean: torch.Tensor) -> Normal:
        assert self.log_std is not None
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        raw_log_prob = dist.log_prob(raw_action).sum(dim=-1)

        correction = torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)

        return raw_log_prob - correction

    def get_action_and_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        policy_out, values = self.forward(states)

        if self.continuous:
            mean = policy_out
            dist = self._continuous_dist(mean)

            if actions is None:
                if deterministic:
                    raw_action = mean
                else:
                    raw_action = dist.sample()
            else:
                raw_action = self._atanh(actions)

            env_action = torch.tanh(raw_action)
            log_probs = self._squashed_log_prob(dist, raw_action)

            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            normal_entropy = 0.5 * torch.log(
                torch.tensor(2.0 * torch.pi * torch.e, device=mean.device, dtype=mean.dtype)
            )
            entropy = (log_std + normal_entropy).sum().expand_as(log_probs)

            return env_action, log_probs, entropy, values

        dist = Categorical(logits=policy_out)

        if actions is None:
            if deterministic:
                actions = torch.argmax(policy_out, dim=-1)
            else:
                actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy, values