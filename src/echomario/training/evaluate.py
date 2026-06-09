from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from echomario.agents.readout_policy import ReadoutActorCritic
from echomario.envs.reset import ResetSpec
from echomario.reservoirs.esn import ReservoirLike
from echomario.training.env_pool import SubprocEnvPool


def evaluate_policy(
    *,
    env: gym.Env,
    reservoir: ReservoirLike,
    policy: ReadoutActorCritic,
    episodes: int,
    device: torch.device,
    reset_specs: list[ResetSpec] | None = None,
) -> dict[str, float]:
    returns = []
    max_xs = []
    successes = []
    fell_count = 0
    enemy_hit_count = 0
    stagnated_count = 0
    truncated_count = 0

    continuous = isinstance(env.action_space, spaces.Box)
    specs = reset_specs if reset_specs is not None else [ResetSpec() for _ in range(episodes)]
    if not specs:
        raise ValueError('reset_specs must contain at least one episode reset specification')

    for spec in specs[:episodes]:
        obs, _ = env.reset(**spec.as_kwargs())
        reservoir.reset()
        done = False
        ep_return = 0.0
        max_x = 0.0
        reached_goal = False

        while not done:
            x_np = reservoir.step(obs)
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                if continuous:
                    action, _, _, _ = policy.get_action_and_value(x, deterministic=True)
                    env_action = action.squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    logits, _ = policy.forward(x)
                    env_action = int(torch.argmax(logits, dim=-1).item())

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            max_x = max(max_x, float(info.get('max_x', info.get('x_pos', 0.0))))
            reached_goal = reached_goal or bool(info.get('reached_goal', info.get('flag_get', False)))

            if done:
                fell_count += int(bool(info.get('fell', False)))
                enemy_hit_count += int(bool(info.get('hit_enemy', False)))
                stagnated_count += int(bool(info.get('stagnated', False)))
                truncated_count += int(bool(truncated))

        returns.append(ep_return)
        max_xs.append(max_x)
        successes.append(1.0 if reached_goal else 0.0)

    denom = float(len(returns))
    return {
        'eval_return_mean': float(np.mean(returns)),
        'eval_return_std': float(np.std(returns)),
        'eval_max_x_mean': float(np.mean(max_xs)),
        'eval_success_rate': float(np.mean(successes)),
        'eval_fell_rate': fell_count / denom,
        'eval_enemy_hit_rate': enemy_hit_count / denom,
        'eval_stagnation_rate': stagnated_count / denom,
        'eval_truncation_rate': truncated_count / denom,
    }


def evaluate_policy_env_pool(
    *,
    env_pool: SubprocEnvPool,
    continuous: bool,
    reservoir: ReservoirLike,
    policy: ReadoutActorCritic,
    reset_specs: list[ResetSpec],
    device: torch.device,
) -> dict[str, float]:
    if str(getattr(getattr(reservoir, 'cfg', None), 'type', '')) != 'none':
        raise ValueError('evaluate_policy_env_pool currently supports identity reservoirs only')
    if not reset_specs:
        raise ValueError('reset_specs must contain at least one episode reset specification')
    if len(reset_specs) != env_pool.num_envs:
        raise ValueError(f'Expected {env_pool.num_envs} reset specs, got {len(reset_specs)}')

    obs_batch = env_pool.reset_all(reset_specs)
    reservoir.reset(batch_size=env_pool.num_envs)

    returns = np.zeros(env_pool.num_envs, dtype=np.float32)
    max_xs = np.zeros(env_pool.num_envs, dtype=np.float32)
    successes = np.zeros(env_pool.num_envs, dtype=np.float32)
    fell = np.zeros(env_pool.num_envs, dtype=np.float32)
    enemy_hit = np.zeros(env_pool.num_envs, dtype=np.float32)
    stagnated = np.zeros(env_pool.num_envs, dtype=np.float32)
    truncated_flags = np.zeros(env_pool.num_envs, dtype=np.float32)
    done = np.zeros(env_pool.num_envs, dtype=bool)

    while not bool(done.all()):
        active_indices = np.flatnonzero(~done)
        active_obs = obs_batch[active_indices]
        x = torch.as_tensor(active_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            if continuous:
                action, _, _, _ = policy.get_action_and_value(x, deterministic=True)
                env_actions = action.cpu().numpy().astype(np.float32)
            else:
                logits, _ = policy.forward(x)
                env_actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)

        for env_idx, (next_obs, reward, terminated, truncated, info) in env_pool.step_at(
            active_indices,
            env_actions,
        ):
            returns[env_idx] += float(reward)
            max_xs[env_idx] = max(max_xs[env_idx], float(info.get('max_x', info.get('x_pos', 0.0))))
            successes[env_idx] = max(
                successes[env_idx],
                1.0 if bool(info.get('reached_goal', info.get('flag_get', False))) else 0.0,
            )
            obs_batch[env_idx] = next_obs

            episode_done = bool(terminated or truncated)
            if episode_done:
                done[env_idx] = True
                fell[env_idx] = 1.0 if bool(info.get('fell', False)) else 0.0
                enemy_hit[env_idx] = 1.0 if bool(info.get('hit_enemy', False)) else 0.0
                stagnated[env_idx] = 1.0 if bool(info.get('stagnated', False)) else 0.0
                truncated_flags[env_idx] = 1.0 if bool(truncated) else 0.0

    return {
        'eval_return_mean': float(np.mean(returns)),
        'eval_return_std': float(np.std(returns)),
        'eval_max_x_mean': float(np.mean(max_xs)),
        'eval_success_rate': float(np.mean(successes)),
        'eval_fell_rate': float(np.mean(fell)),
        'eval_enemy_hit_rate': float(np.mean(enemy_hit)),
        'eval_stagnation_rate': float(np.mean(stagnated)),
        'eval_truncation_rate': float(np.mean(truncated_flags)),
    }
