from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from echomario.agents.factory import PolicyLike
from echomario.envs.reset import ResetSpec
from echomario.reservoirs.esn import ReservoirLike
from echomario.training.env_pool import SubprocEnvPool


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


def env_is_continuous(env: gym.Env) -> bool:
    return isinstance(env.action_space, spaces.Box)


def reservoir_is_identity(reservoir: ReservoirLike) -> bool:
    return str(getattr(getattr(reservoir, 'cfg', None), 'type', '')) == 'none'


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
) -> Rollout:
    if not envs:
        raise ValueError('envs must contain at least one environment')

    num_envs = len(envs)
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.reshape(1, -1)
    if obs_batch.shape[0] != num_envs:
        raise ValueError(f'Expected obs_batch for {num_envs} envs, got shape {obs_batch.shape}')

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    running_returns = np.zeros(num_envs, dtype=np.float32)
    running_lengths = np.zeros(num_envs, dtype=np.int32)

    continuous = env_is_continuous(envs[0])
    current_state = None if state is None else np.asarray(state, dtype=np.float32)
    identity_reservoir = reservoir_is_identity(reservoir)
    rollout_state_dim = obs_batch.shape[1] if identity_reservoir else int(reservoir.cfg.size)
    action_dtype = torch.float32 if continuous else torch.long
    if continuous:
        action_shape = (rollout_steps * num_envs, *envs[0].action_space.shape)
    else:
        action_shape = (rollout_steps * num_envs,)
    states_tensor = torch.empty(
        (rollout_steps * num_envs, rollout_state_dim), dtype=torch.float32, device=device
    )
    actions_tensor = torch.empty(action_shape, dtype=action_dtype, device=device)
    log_probs_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    timings = {'state': 0.0, 'policy': 0.0, 'env': 0.0, 'store': 0.0, 'bootstrap': 0.0}

    for step_idx in range(rollout_steps):
        state_start = time.perf_counter()
        if current_state is None:
            x_np = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
        else:
            x_np = current_state
            current_state = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        timings['state'] += time.perf_counter() - state_start

        policy_start = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                action, log_prob, _, value = policy.get_action_and_value(x)
        action = action.float()
        log_prob = log_prob.float()
        value = value.float()

        if continuous:
            action_np = action.detach().cpu().numpy().astype(np.float32)
            env_actions = action_np
        else:
            action_np = action.detach().cpu().numpy().astype(np.int64)
            env_actions = action_np
        timings['policy'] += time.perf_counter() - policy_start

        next_obs_batch = np.empty_like(obs_batch)
        reward_batch = np.empty(num_envs, dtype=np.float32)
        done_batch = np.empty(num_envs, dtype=np.float32)

        env_start = time.perf_counter()
        for env_idx, env in enumerate(envs):
            env_action = env_actions[env_idx] if continuous else int(env_actions[env_idx])
            next_obs, reward, terminated, truncated, _info = env.step(env_action)
            done = bool(terminated or truncated)

            reward_batch[env_idx] = float(reward)
            done_batch[env_idx] = float(done)
            running_returns[env_idx] += float(reward)
            running_lengths[env_idx] += 1

            if done:
                episode_returns.append(float(running_returns[env_idx]))
                episode_lengths.append(int(running_lengths[env_idx]))
                reservoir.reset(mask=[env_idx])
                next_obs_batch[env_idx] = reset_fn(env_idx)
                running_returns[env_idx] = 0.0
                running_lengths[env_idx] = 0
            else:
                next_obs_batch[env_idx] = next_obs
        timings['env'] += time.perf_counter() - env_start

        store_start = time.perf_counter()
        flat_start = step_idx * num_envs
        flat_end = flat_start + num_envs
        states_tensor[flat_start:flat_end].copy_(x)
        actions_tensor[flat_start:flat_end].copy_(action.detach())
        log_probs_tensor[step_idx].copy_(log_prob.detach())
        values_tensor[step_idx].copy_(value.detach())
        rewards_tensor[step_idx].copy_(torch.as_tensor(reward_batch, dtype=torch.float32, device=device))
        dones_tensor[step_idx].copy_(torch.as_tensor(done_batch, dtype=torch.float32, device=device))
        timings['store'] += time.perf_counter() - store_start

        obs_batch = next_obs_batch
        if progress_fn is not None and progress_interval > 0 and (step_idx + 1) % progress_interval == 0:
            progress_fn(num_envs * progress_interval)

    bootstrap_start = time.perf_counter()
    if current_state is None:
        last_state = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
    else:
        last_state = current_state
    x = torch.as_tensor(last_state, dtype=torch.float32, device=device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            _, last_value = policy.forward(x)
    timings['bootstrap'] += time.perf_counter() - bootstrap_start

    return Rollout(
        states=states_tensor,
        actions=actions_tensor,
        log_probs=log_probs_tensor.reshape(-1),
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        last_value=last_value.detach(),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        last_obs=obs_batch.copy(),
        last_state=last_state.copy(),
        timings=timings,
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
) -> Rollout:
    num_envs = env_pool.num_envs
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
    rollout_state_dim = obs_batch.shape[1] if identity_reservoir else int(reservoir.cfg.size)
    action_dtype = torch.float32 if continuous else torch.long
    if continuous:
        action_shape = (rollout_steps * num_envs, *env_pool.action_shape)
    else:
        action_shape = (rollout_steps * num_envs,)
    states_tensor = torch.empty(
        (rollout_steps * num_envs, rollout_state_dim), dtype=torch.float32, device=device
    )
    actions_tensor = torch.empty(action_shape, dtype=action_dtype, device=device)
    log_probs_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values_tensor = torch.empty((rollout_steps, num_envs), dtype=torch.float32, device=device)
    timings = {'state': 0.0, 'policy': 0.0, 'env': 0.0, 'store': 0.0, 'bootstrap': 0.0}

    for step_idx in range(rollout_steps):
        state_start = time.perf_counter()
        if current_state is None:
            x_np = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
        else:
            x_np = current_state
            current_state = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        timings['state'] += time.perf_counter() - state_start

        policy_start = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                action, log_prob, _, value = policy.get_action_and_value(x)
        action = action.float()
        log_prob = log_prob.float()
        value = value.float()

        if continuous:
            action_np = action.detach().cpu().numpy().astype(np.float32)
        else:
            action_np = action.detach().cpu().numpy().astype(np.int64)
        timings['policy'] += time.perf_counter() - policy_start

        env_start = time.perf_counter()
        step_results = env_pool.step(action_np)
        next_obs_batch = np.empty_like(obs_batch)
        reward_batch = np.empty(num_envs, dtype=np.float32)
        done_batch = np.empty(num_envs, dtype=np.float32)

        for env_idx, (next_obs, reward, terminated, truncated, _info) in enumerate(step_results):
            done = bool(terminated or truncated)
            reward_batch[env_idx] = float(reward)
            done_batch[env_idx] = float(done)
            running_returns[env_idx] += float(reward)
            running_lengths[env_idx] += 1

            if done:
                episode_returns.append(float(running_returns[env_idx]))
                episode_lengths.append(int(running_lengths[env_idx]))
                reservoir.reset(mask=[env_idx])
                next_obs_batch[env_idx] = env_pool.reset_at(env_idx, reset_spec_fn(env_idx))
                running_returns[env_idx] = 0.0
                running_lengths[env_idx] = 0
            else:
                next_obs_batch[env_idx] = next_obs
        timings['env'] += time.perf_counter() - env_start

        store_start = time.perf_counter()
        flat_start = step_idx * num_envs
        flat_end = flat_start + num_envs
        states_tensor[flat_start:flat_end].copy_(x)
        actions_tensor[flat_start:flat_end].copy_(action.detach())
        log_probs_tensor[step_idx].copy_(log_prob.detach())
        values_tensor[step_idx].copy_(value.detach())
        rewards_tensor[step_idx].copy_(torch.as_tensor(reward_batch, dtype=torch.float32, device=device))
        dones_tensor[step_idx].copy_(torch.as_tensor(done_batch, dtype=torch.float32, device=device))
        timings['store'] += time.perf_counter() - store_start

        obs_batch = next_obs_batch
        if progress_fn is not None and progress_interval > 0 and (step_idx + 1) % progress_interval == 0:
            progress_fn(num_envs * progress_interval)

    bootstrap_start = time.perf_counter()
    if current_state is None:
        last_state = obs_batch if identity_reservoir else reservoir.step_batch(obs_batch)
    else:
        last_state = current_state
    x = torch.as_tensor(last_state, dtype=torch.float32, device=device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            _, last_value = policy.forward(x)
    timings['bootstrap'] += time.perf_counter() - bootstrap_start

    return Rollout(
        states=states_tensor,
        actions=actions_tensor,
        log_probs=log_probs_tensor.reshape(-1),
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        last_value=last_value.detach(),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        last_obs=obs_batch.copy(),
        last_state=last_state.copy(),
        timings=timings,
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
    )
    rollout.last_obs = rollout.last_obs[0].copy()
    rollout.last_state = rollout.last_state[0].copy()
    rollout.last_value = rollout.last_value[0].detach()
    return rollout
