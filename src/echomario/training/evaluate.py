from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from echomario.agents.readout_policy import ReadoutActorCritic
from echomario.envs.reset import ResetSpec
from echomario.reservoirs.esn import ReservoirLike


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
