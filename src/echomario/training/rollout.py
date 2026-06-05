from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def env_is_continuous(env: gym.Env) -> bool:
    return isinstance(env.action_space, spaces.Box)


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
) -> Rollout:
    if not envs:
        raise ValueError('envs must contain at least one environment')

    num_envs = len(envs)
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.reshape(1, -1)
    if obs_batch.shape[0] != num_envs:
        raise ValueError(f'Expected obs_batch for {num_envs} envs, got shape {obs_batch.shape}')

    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    running_returns = np.zeros(num_envs, dtype=np.float32)
    running_lengths = np.zeros(num_envs, dtype=np.int32)

    continuous = env_is_continuous(envs[0])
    current_state = None if state is None else np.asarray(state, dtype=np.float32)

    for _ in range(rollout_steps):
        if current_state is None:
            x_np = reservoir.step_batch(obs_batch)
        else:
            x_np = current_state
            current_state = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(x)

        if continuous:
            action_np = action.detach().cpu().numpy().astype(np.float32)
            env_actions = action_np
            stored_actions = action_np
        else:
            action_np = action.detach().cpu().numpy().astype(np.int64)
            env_actions = action_np
            stored_actions = action_np

        next_obs_batch = np.empty_like(obs_batch)
        reward_batch = np.empty(num_envs, dtype=np.float32)
        done_batch = np.empty(num_envs, dtype=np.float32)

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

        states.append(x_np)
        actions.append(stored_actions)
        log_probs.append(log_prob.detach().cpu().numpy().astype(np.float32))
        rewards.append(reward_batch)
        dones.append(done_batch)
        values.append(value.detach().cpu().numpy().astype(np.float32))

        obs_batch = next_obs_batch

    last_state = reservoir.step_batch(obs_batch) if current_state is None else current_state
    x = torch.as_tensor(last_state, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, last_value = policy.forward(x)

    action_dtype = torch.float32 if continuous else torch.long

    return Rollout(
        states=torch.as_tensor(np.concatenate(states, axis=0), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.concatenate(actions, axis=0), dtype=action_dtype, device=device),
        log_probs=torch.as_tensor(np.concatenate(log_probs, axis=0), dtype=torch.float32, device=device),
        rewards=torch.as_tensor(np.stack(rewards, axis=0), dtype=torch.float32, device=device),
        dones=torch.as_tensor(np.stack(dones, axis=0), dtype=torch.float32, device=device),
        values=torch.as_tensor(np.stack(values, axis=0), dtype=torch.float32, device=device),
        last_value=last_value.detach(),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        last_obs=obs_batch.copy(),
        last_state=last_state.copy(),
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
) -> Rollout:
    num_envs = env_pool.num_envs
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.reshape(1, -1)
    if obs_batch.shape[0] != num_envs:
        raise ValueError(f'Expected obs_batch for {num_envs} envs, got shape {obs_batch.shape}')

    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    running_returns = np.zeros(num_envs, dtype=np.float32)
    running_lengths = np.zeros(num_envs, dtype=np.int32)

    current_state = None if state is None else np.asarray(state, dtype=np.float32)

    for _ in range(rollout_steps):
        if current_state is None:
            x_np = reservoir.step_batch(obs_batch)
        else:
            x_np = current_state
            current_state = None
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(x)

        if continuous:
            action_np = action.detach().cpu().numpy().astype(np.float32)
            stored_actions = action_np
        else:
            action_np = action.detach().cpu().numpy().astype(np.int64)
            stored_actions = action_np

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

        states.append(x_np)
        actions.append(stored_actions)
        log_probs.append(log_prob.detach().cpu().numpy().astype(np.float32))
        rewards.append(reward_batch)
        dones.append(done_batch)
        values.append(value.detach().cpu().numpy().astype(np.float32))

        obs_batch = next_obs_batch

    last_state = reservoir.step_batch(obs_batch) if current_state is None else current_state
    x = torch.as_tensor(last_state, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, last_value = policy.forward(x)

    action_dtype = torch.float32 if continuous else torch.long

    return Rollout(
        states=torch.as_tensor(np.concatenate(states, axis=0), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.concatenate(actions, axis=0), dtype=action_dtype, device=device),
        log_probs=torch.as_tensor(np.concatenate(log_probs, axis=0), dtype=torch.float32, device=device),
        rewards=torch.as_tensor(np.stack(rewards, axis=0), dtype=torch.float32, device=device),
        dones=torch.as_tensor(np.stack(dones, axis=0), dtype=torch.float32, device=device),
        values=torch.as_tensor(np.stack(values, axis=0), dtype=torch.float32, device=device),
        last_value=last_value.detach(),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        last_obs=obs_batch.copy(),
        last_state=last_state.copy(),
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
    )
    rollout.last_obs = rollout.last_obs[0].copy()
    rollout.last_state = rollout.last_state[0].copy()
    rollout.last_value = rollout.last_value[0].detach()
    return rollout
