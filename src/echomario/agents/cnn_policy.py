from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class TileCNNActorCritic(nn.Module):
    def __init__(
        self,
        *,
        screen_channels: int,
        screen_height: int,
        screen_width: int,
        state_dim: int,
        num_actions: int,
        continuous: bool = False,
        hidden_dim: int = 256,
        log_std_init: float = -1.0,
        min_log_std: float = -3.0,
        max_log_std: float = 0.0,
    ):
        super().__init__()

        self.screen_channels = int(screen_channels)
        self.screen_height = int(screen_height)
        self.screen_width = int(screen_width)
        self.state_dim = int(state_dim)
        self.continuous = bool(continuous)
        self.num_actions = int(num_actions)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.screen_dim = self.screen_channels * self.screen_height * self.screen_width
        self.cnn = nn.Sequential(
            nn.Conv2d(self.screen_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.screen_channels, self.screen_height, self.screen_width)
            cnn_dim = int(self.cnn(dummy).shape[-1])

        trunk_dim = cnn_dim + self.state_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, self.num_actions)
        self.value = nn.Linear(hidden_dim, 1)

        if self.continuous:
            self.log_std = nn.Parameter(torch.full((self.num_actions,), float(log_std_init)))
        else:
            self.log_std = None

    def _split_obs(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.shape[-1] != self.screen_dim + self.state_dim:
            raise ValueError(
                f'Expected flattened obs dim {self.screen_dim + self.state_dim}, got {states.shape[-1]}'
            )
        screen = states[..., : self.screen_dim]
        screen = screen.view(-1, self.screen_channels, self.screen_height, self.screen_width)
        state = states[..., self.screen_dim :] if self.state_dim > 0 else None
        return screen, state

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        screen, state = self._split_obs(states)
        screen_features = self.cnn(screen)
        if state is not None:
            features = torch.cat([screen_features, state], dim=-1)
        else:
            features = screen_features
        trunk = self.trunk(features)
        policy_out = self.policy(trunk)
        values = self.value(trunk).squeeze(-1)
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
                raw_action = mean if deterministic else dist.sample()
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
            actions = torch.argmax(policy_out, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy, values
