from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from echomario.agents.base import reset_policy_state
from echomario.agents.factory import PolicyLike
from echomario.envs.reset import ResetSpec
from echomario.reservoirs.esn import ReservoirLike
from echomario.training.env_pool import SubprocEnvPool

StepResult = tuple[np.ndarray, float, bool, bool, dict]


@dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_value: torch.Tensor
    episode_returns: list[float]
    episode_lengths: list[int]
    last_obs: np.ndarray
    last_state: np.ndarray
    timings: dict[str, float]
    # Raw (pre-tanh) continuous actions are stored so PPO can re-evaluate them exactly.
    actions_are_raw: bool = False
    # Recurrent policies: hidden state *before* each step, shape (T * N, H).
    hiddens: torch.Tensor | None = None
    num_envs: int = 1


def env_is_continuous(env: gym.Env) -> bool:
    return isinstance(env.action_space, spaces.Box)


def reservoir_is_identity(reservoir: ReservoirLike) -> bool:
    return str(getattr(getattr(reservoir, 'cfg', None), 'type', '')) == 'none'


def _collect(
    *,
    num_envs: int,
    continuous: bool,
    action_shape: tuple[int, ...],
    step_fn: Callable[[np.ndarray], list[StepResult]],
    reset_fn: Callable[[int], np.ndarray],
    reservoir: ReservoirLike,
    policy: PolicyLike,
    obs_batch: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    state: np.ndarray | None,
    progress_fn: Callable[[int], None] | None,
    progress_interval: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    gamma: float | None,
) -> Rollout:
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.reshape(1, -1)
    if obs_batch.shape[0] != num_envs:
        raise ValueError(f'Expected obs_batch for {num_envs} envs, got shape {obs_batch.shape}')

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    running_returns = np.zeros(num_envs, dtype=np.float32)
    running_lengths = np.zeros(num_envs, dtype=np.int32)

    current_state = None if state is None else np.asarray(state, dtype=np.float32)
    identity_reservoir = reservoir_is_identity(reservoir)
    recurrent = bool(getattr(policy, 'is_recurrent', False))
    rollout_state_dim = obs_batch.shape[1] if identity_reservoir else int(reservoir.cfg.size)
    total = rollout_steps * num_envs
    flat_action_shape = (total, *action_shape) if continuous else (total,)

    states_tensor = torch.empty((total, rollout_state_dim), dtype=torch.float32, device=device)
    actions_tensor = torch.empty(
        flat_action_shape, dtype=torch.float32 if continuous else torch.long, device=device
    )
    log_probs_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    hiddens_tensor = None
    if recurrent:
        hiddens_tensor = torch.empty((total, policy.hidden_dim), dtype=torch.float32, device=device)
        if policy.hidden_batch_size() != num_envs:
            policy.reset_state(batch_size=num_envs)
    timings = {'state': 0.0, 'policy': 0.0, 'env': 0.0, 'store': 0.0, 'bootstrap': 0.0}

    def autocast():
        return torch.autocast(
            device_type='cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'
        )

    for step_idx in range(rollout_steps):
        flat_start = step_idx * num_envs
        flat_end = flat_start + num_envs

        state_start = time.perf_counter()
        if current_state is None:
            x_np = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
        else:
            x_np = current_state
            current_state = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        timings['state'] += time.perf_counter() - state_start

        policy_start = time.perf_counter()
        if recurrent:
            hiddens_tensor[flat_start:flat_end].copy_(policy.get_hidden(num_envs, device))
        with torch.no_grad(), autocast():
            stored_action, env_action, log_prob, value = policy.act(x)
        env_actions = env_action.float().cpu().numpy()
        env_actions = env_actions.astype(np.float32) if continuous else env_actions.astype(np.int64)
        timings['policy'] += time.perf_counter() - policy_start

        env_start = time.perf_counter()
        results = step_fn(env_actions)
        next_obs_batch = np.empty_like(obs_batch)
        reward_batch = np.empty(num_envs, dtype=np.float32)
        done_batch = np.empty(num_envs, dtype=np.float32)
        truncated_idx: list[int] = []
        for env_idx, (next_obs, reward, terminated, truncated, _info) in enumerate(results):
            reward_batch[env_idx] = float(reward)
            done_batch[env_idx] = float(terminated or truncated)
            next_obs_batch[env_idx] = next_obs
            running_returns[env_idx] += float(reward)
            running_lengths[env_idx] += 1
            if truncated and not terminated:
                truncated_idx.append(env_idx)
        timings['env'] += time.perf_counter() - env_start

        # Time-limit truncation is not a real terminal state: bootstrap with V(s_T) so the agent is
        # not taught that running out of time is as bad as dying.
        if gamma is not None and truncated_idx:
            bootstrap_start = time.perf_counter()
            terminal_x = reservoir.peek_batch(next_obs_batch[truncated_idx], truncated_idx)
            with torch.no_grad(), autocast():
                terminal_values = policy.predict_value(
                    torch.as_tensor(terminal_x, dtype=torch.float32, device=device),
                    indices=truncated_idx,
                ).float().cpu().numpy()
            reward_batch[truncated_idx] += gamma * terminal_values
            timings['bootstrap'] += time.perf_counter() - bootstrap_start

        env_start = time.perf_counter()
        for env_idx in np.flatnonzero(done_batch):
            episode_returns.append(float(running_returns[env_idx]))
            episode_lengths.append(int(running_lengths[env_idx]))
            reservoir.reset(mask=[env_idx])
            reset_policy_state(policy, mask=[env_idx])
            next_obs_batch[env_idx] = reset_fn(int(env_idx))
            running_returns[env_idx] = 0.0
            running_lengths[env_idx] = 0
        timings['env'] += time.perf_counter() - env_start

        store_start = time.perf_counter()
        states_tensor[flat_start:flat_end].copy_(x)
        actions_tensor[flat_start:flat_end].copy_(stored_action.detach())
        log_probs_tensor[step_idx].copy_(log_prob.detach().float())
        values_tensor[step_idx].copy_(value.detach().float())
        rewards_tensor[step_idx].copy_(torch.as_tensor(reward_batch, device=device))
        dones_tensor[step_idx].copy_(torch.as_tensor(done_batch, device=device))
        timings['store'] += time.perf_counter() - store_start

        obs_batch = next_obs_batch
        if progress_fn is not None and progress_interval > 0 and (step_idx + 1) % progress_interval == 0:
            progress_fn(num_envs * progress_interval)

    bootstrap_start = time.perf_counter()
    if current_state is None:
        last_state = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
    else:
        last_state = current_state
    with torch.no_grad(), autocast():
        last_value = policy.predict_value(torch.as_tensor(last_state, dtype=torch.float32, device=device))
    timings['bootstrap'] += time.perf_counter() - bootstrap_start

    return Rollout(
        states=states_tensor,
        actions=actions_tensor,
        log_probs=log_probs_tensor.reshape(-1),
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        last_value=last_value.detach().float(),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        last_obs=obs_batch.copy(),
        last_state=np.asarray(last_state).copy(),
        timings=timings,
        actions_are_raw=continuous,
        hiddens=hiddens_tensor,
        num_envs=num_envs,
    )


def collect_rollout_parallel(
    *,
    envs: list[gym.Env],
    reservoir: ReservoirLike,
    policy: PolicyLike,
    obs_batch: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    reset_fn: Callable[[int], np.ndarray],
    state: np.ndarray | None = None,
    progress_fn: Callable[[int], None] | None = None,
    progress_interval: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    gamma: float | None = None,
) -> Rollout:
    """Collects a rollout from in-process environments. ``gamma`` enables truncation bootstrapping."""
    if not envs:
        raise ValueError('envs must contain at least one environment')
    continuous = env_is_continuous(envs[0])

    def step_fn(actions: np.ndarray) -> list[StepResult]:
        return [
            env.step(actions[i] if continuous else int(actions[i])) for i, env in enumerate(envs)
        ]

    return _collect(
        num_envs=len(envs),
        continuous=continuous,
        action_shape=tuple(envs[0].action_space.shape or ()),
        step_fn=step_fn,
        reset_fn=reset_fn,
        reservoir=reservoir,
        policy=policy,
        obs_batch=obs_batch,
        rollout_steps=rollout_steps,
        device=device,
        state=state,
        progress_fn=progress_fn,
        progress_interval=progress_interval,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        gamma=gamma,
    )


def collect_rollout_env_pool(
    *,
    env_pool: SubprocEnvPool,
    continuous: bool,
    reservoir: ReservoirLike,
    policy: PolicyLike,
    obs_batch: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    reset_spec_fn: Callable[[int], ResetSpec],
    state: np.ndarray | None = None,
    progress_fn: Callable[[int], None] | None = None,
    progress_interval: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    gamma: float | None = None,
) -> Rollout:
    """Collects a rollout from subprocess environments. ``gamma`` enables truncation bootstrapping."""
    return _collect(
        num_envs=env_pool.num_envs,
        continuous=continuous,
        action_shape=tuple(env_pool.action_shape),
        step_fn=env_pool.step,
        reset_fn=lambda env_idx: env_pool.reset_at(env_idx, reset_spec_fn(env_idx)),
        reservoir=reservoir,
        policy=policy,
        obs_batch=obs_batch,
        rollout_steps=rollout_steps,
        device=device,
        state=state,
        progress_fn=progress_fn,
        progress_interval=progress_interval,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        gamma=gamma,
    )


def collect_rollout(
    *,
    env: gym.Env,
    reservoir: ReservoirLike,
    policy: PolicyLike,
    obs: np.ndarray,
    rollout_steps: int,
    device: torch.device,
    reset_fn: Callable[[], np.ndarray],
    state: np.ndarray | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    gamma: float | None = None,
) -> Rollout:
    rollout = collect_rollout_parallel(
        envs=[env],
        reservoir=reservoir,
        policy=policy,
        obs_batch=np.asarray(obs, dtype=np.float32).reshape(1, -1),
        rollout_steps=rollout_steps,
        device=device,
        reset_fn=lambda _env_idx: reset_fn(),
        state=None if state is None else np.asarray(state, dtype=np.float32).reshape(1, -1),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        gamma=gamma,
    )
    rollout.last_obs = rollout.last_obs[0].copy()
    rollout.last_state = rollout.last_state[0].copy()
    rollout.last_value = rollout.last_value[0].detach()
    return rollout
