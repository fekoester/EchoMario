from __future__ import annotations

import inspect
from typing import Any

import gymnasium as gym

from echomario.envs.toy_platformer_env import ToyPlatformerEnv


def make_env(config: dict[str, Any]) -> gym.Env:
    env_cfg = config['env']
    name = env_cfg['name']

    if name == 'toy_platformer':
        signature = inspect.signature(ToyPlatformerEnv.__init__)
        kwargs: dict[str, Any] = {}
        for key, value in env_cfg.items():
            if key in {'name', 'train_seed_mode', 'eval_seed_start', 'num_eval_seeds'}:
                continue
            if key in signature.parameters:
                kwargs[key] = value
        kwargs.setdefault('seed', int(config['project'].get('seed', 42)))
        return ToyPlatformerEnv(**kwargs)

    if name.startswith('SuperMarioBros'):
        from echomario.envs.mario_env import make_mario_env

        return make_mario_env(config)

    raise ValueError(f'Unknown env name: {name}')
