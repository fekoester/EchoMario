from __future__ import annotations

import numpy as np
from gymnasium import spaces

from echomario.agents.cnn_policy import TileCNNActorCritic
from echomario.agents.readout_policy import ReadoutActorCritic


PolicyLike = ReadoutActorCritic | TileCNNActorCritic


def policy_kind(config: dict) -> str:
    return str(config.get('agent', {}).get('model', 'readout'))


def build_policy(config: dict, env, reservoir_dim: int) -> PolicyLike:
    continuous = isinstance(env.action_space, spaces.Box)
    num_actions = int(env.action_space.shape[0]) if continuous else int(env.action_space.n)
    kind = policy_kind(config)

    if kind == 'cnn':
        screen_channels = len(getattr(env, 'SCREEN_CHANNEL_NAMES'))
        screen_height = int(getattr(env, 'height'))
        screen_width = int(getattr(env, 'camera_width'))
        state_dim = len(getattr(env, 'STATE_FEATURE_NAMES')) if bool(getattr(env, 'include_state_features', False)) else 0
        return TileCNNActorCritic(
            screen_channels=screen_channels,
            screen_height=screen_height,
            screen_width=screen_width,
            state_dim=state_dim,
            num_actions=num_actions,
            continuous=continuous,
            hidden_dim=int(config.get('agent', {}).get('hidden_dim', 256)),
            log_std_init=float(config.get('agent', {}).get('log_std_init', -1.0)),
            min_log_std=float(config.get('agent', {}).get('min_log_std', -3.0)),
            max_log_std=float(config.get('agent', {}).get('max_log_std', 0.0)),
        )

    return ReadoutActorCritic(
        reservoir_dim=reservoir_dim,
        num_actions=num_actions,
        continuous=continuous,
        log_std_init=float(config.get('agent', {}).get('log_std_init', -1.0)),
        min_log_std=float(config.get('agent', {}).get('min_log_std', -3.0)),
        max_log_std=float(config.get('agent', {}).get('max_log_std', 0.0)),
    )


def policy_contribution(policy: PolicyLike, state: np.ndarray, selected_action: int) -> np.ndarray:
    if hasattr(policy, 'policy') and getattr(policy.policy, 'weight', None) is not None:
        weight = policy.policy.weight.detach().cpu().numpy()
        if weight.ndim == 2 and weight.shape[1] == state.shape[0] and 0 <= selected_action < weight.shape[0]:
            return weight[selected_action] * state
    return np.zeros_like(state, dtype=np.float32)
