from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class ActorCriticBase(nn.Module):
    """
    Shared action-distribution logic for all actor-critics.

    Subclasses implement ``forward(states) -> (policy_out, values)`` and set ``continuous``,
    ``num_actions``, ``min_log_std``, ``max_log_std`` and ``log_std``.

    Continuous actions use a tanh-squashed Gaussian. PPO must re-evaluate the *raw* (pre-tanh)
    sample: reconstructing it with ``atanh`` from the squashed action is lossy once the policy
    saturates (|a| -> 1), which corrupts the importance ratio. ``act`` therefore returns the raw
    action for storage and the squashed one for the environment.
    """

    is_recurrent = False
    continuous: bool
    num_actions: int
    min_log_std: float
    max_log_std: float
    log_std: nn.Parameter | None

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

    def _distribution_outputs(
        self,
        policy_out: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor | None,
        deterministic: bool,
        actions_are_raw: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (stored_action, env_action, log_prob, entropy, value)."""
        if self.continuous:
            mean = policy_out
            dist = self._continuous_dist(mean)
            if actions is None:
                raw_action = mean if deterministic else dist.sample()
            elif actions_are_raw:
                raw_action = actions
            else:
                raw_action = self._atanh(actions)
            log_probs = self._squashed_log_prob(dist, raw_action)
            log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
            normal_entropy = 0.5 * torch.log(
                torch.tensor(2.0 * torch.pi * torch.e, device=mean.device, dtype=mean.dtype)
            )
            entropy = (log_std + normal_entropy).sum().expand_as(log_probs)
            return raw_action, torch.tanh(raw_action), log_probs, entropy, values

        dist = Categorical(logits=policy_out)
        if actions is None:
            actions = torch.argmax(policy_out, dim=-1) if deterministic else dist.sample()
        return actions, actions, dist.log_prob(actions), dist.entropy(), values

    def act(self, states: torch.Tensor, deterministic: bool = False):
        """Samples actions. Returns (action_to_store, env_action, log_prob, value)."""
        policy_out, values = self.forward(states)
        stored, env_action, log_probs, _entropy, values = self._distribution_outputs(
            policy_out, values, None, deterministic, False
        )
        return stored, env_action, log_probs, values

    def get_action_and_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
        actions_are_raw: bool = False,
    ):
        """
        Returns (env_action, log_prob, entropy, value). Pass ``actions_are_raw=True`` when
        re-evaluating raw continuous actions stored by :meth:`act`.
        """
        policy_out, values = self.forward(states)
        _stored, env_action, log_probs, entropy, values = self._distribution_outputs(
            policy_out, values, actions, deterministic, actions_are_raw
        )
        return env_action, log_probs, entropy, values

    def predict_value(self, states: torch.Tensor, indices=None) -> torch.Tensor:
        """
        Value estimate that never changes internal (recurrent) state. ``indices`` selects which
        environments' recurrent state to use when ``states`` covers only some of them.
        """
        return self.forward(states)[1]

    def reset_state(self, mask=None, batch_size: int | None = None) -> None:
        """Resets recurrent state (no-op for feed-forward policies)."""

    def snapshot_state(self):
        """Returns a copy of the recurrent state (None for feed-forward policies)."""
        return None

    def restore_state(self, snapshot) -> None:
        """Restores a state returned by :meth:`snapshot_state`."""


def reset_policy_state(policy, mask=None, batch_size: int | None = None) -> None:
    """Resets recurrent policy state if the policy has any; safe to call for every policy."""
    reset = getattr(policy, "reset_state", None)
    if reset is not None:
        reset(mask=mask, batch_size=batch_size)
