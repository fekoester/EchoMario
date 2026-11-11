from __future__ import annotations

import numpy as np
import torch
from torch import nn

from echomario.agents.base import ActorCriticBase


class GRUActorCritic(ActorCriticBase):
    """
    Trainable recurrent baseline: observation encoder -> GRU cell -> policy/value heads, all trained
    end-to-end with PPO (truncated BPTT over ``seq_len`` chunks).

    This is the natural counterpart to the frozen reservoir: same recurrent role, but every weight is
    learned. Use it with ``reservoir.type: none`` so the GRU sees raw observations.

    The policy is *stateful* during interaction: ``forward``/``act``/``get_action_and_value`` advance
    an internal hidden state per environment, mirroring how the reservoir is used. Call
    ``reset_state`` (or ``echomario.agents.base.reset_policy_state``) whenever episodes reset.
    """

    is_recurrent = True

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        continuous: bool = False,
        hidden_dim: int = 256,
        encoder_dim: int = 256,
        log_std_init: float = -1.0,
        min_log_std: float = -3.0,
        max_log_std: float = 0.0,
    ):
        super().__init__()
        self.continuous = bool(continuous)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.encoder = nn.Sequential(nn.Linear(obs_dim, encoder_dim), nn.ReLU())
        self.gru = nn.GRUCell(encoder_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)

        if self.continuous:
            self.log_std = nn.Parameter(torch.full((num_actions,), float(log_std_init)))
        else:
            self.log_std = None
        self._hidden: torch.Tensor | None = None

    # ------------------------------------------------------------------ recurrent state

    def hidden_batch_size(self) -> int:
        return 0 if self._hidden is None else int(self._hidden.shape[0])

    def _ensure_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._hidden is None or self._hidden.shape[0] != batch_size or self._hidden.device != device:
            self._hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return self._hidden

    def get_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self._ensure_hidden(batch_size, torch.device(device)).clone()

    def reset_state(self, mask=None, batch_size: int | None = None) -> None:
        if batch_size is not None:
            device = self._hidden.device if self._hidden is not None else self.value.weight.device
            self._hidden = torch.zeros(int(batch_size), self.hidden_dim, device=device)
            return
        if self._hidden is None:
            return
        if mask is None:
            self._hidden.zero_()
            return
        idx = np.asarray(mask, dtype=np.int64).reshape(-1)
        if idx.size:
            self._hidden[torch.as_tensor(idx, device=self._hidden.device)] = 0.0

    def snapshot_state(self):
        return None if self._hidden is None else self._hidden.clone()

    def restore_state(self, snapshot) -> None:
        self._hidden = None if snapshot is None else snapshot.clone()

    # ------------------------------------------------------------------ forward passes

    def _heads(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(hidden), self.value(hidden).squeeze(-1)

    def _step(self, states: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.gru(self.encoder(states), hidden)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stateful single step: advances the stored hidden state."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        hidden = self._ensure_hidden(states.shape[0], states.device)
        new_hidden = self._step(states, hidden.to(states.dtype))
        self._hidden = new_hidden.detach().float()
        return self._heads(new_hidden)

    def predict_value(self, states: torch.Tensor, indices=None) -> torch.Tensor:
        """Value of ``states`` given the current hidden state, without advancing it."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if indices is None:
            hidden = self._ensure_hidden(states.shape[0], states.device)
        else:
            hidden = self._ensure_hidden(max(self.hidden_batch_size(), 1), states.device)
            hidden = hidden[torch.as_tensor(indices, device=hidden.device)]
        return self._heads(self._step(states, hidden.to(states.dtype)))[1]

    def evaluate_sequence(
        self,
        states: torch.Tensor,
        hidden0: torch.Tensor,
        prev_dones: torch.Tensor,
        actions: torch.Tensor,
        actions_are_raw: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-runs the GRU over sequences for PPO.

        Args:
            states: (L, B, obs_dim)
            hidden0: (B, H) hidden state before the first step
            prev_dones: (L, B) 1.0 where the previous step ended an episode (hidden reset to 0)
            actions: (L, B, ...) stored actions

        Returns:
            log_probs, entropy, values, each of shape (L, B)
        """
        L, B = states.shape[:2]
        encoded = self.encoder(states.reshape(L * B, -1)).reshape(L, B, -1)
        hidden = hidden0
        outputs = []
        for t in range(L):
            hidden = hidden * (1.0 - prev_dones[t]).unsqueeze(-1)
            hidden = self.gru(encoded[t], hidden)
            outputs.append(hidden)
        hiddens = torch.stack(outputs).reshape(L * B, -1)
        policy_out, values = self._heads(hiddens)
        flat_actions = actions.reshape(L * B, *actions.shape[2:])
        _stored, _env, log_probs, entropy, values = self._distribution_outputs(
            policy_out, values, flat_actions, False, actions_are_raw
        )
        return log_probs.reshape(L, B), entropy.reshape(L, B), values.reshape(L, B)
