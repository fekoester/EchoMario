from __future__ import annotations

from collections import deque
import inspect
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from echomario.envs.toy_platformer_env import ToyPlatformerEnv


class DiscreteToyActionWrapper(gym.Wrapper):
    ACTIONS = (
        (0.0, -1.0, -1.0),   # idle
        (1.0, -1.0, -1.0),   # right
        (1.0, -1.0, 1.0),    # right + run
        (0.0, 1.0, -1.0),    # jump
        (1.0, 1.0, -1.0),    # right + jump
        (1.0, 1.0, 1.0),     # right + run + jump
        (-1.0, -1.0, -1.0),  # left
        (-1.0, 1.0, -1.0),   # left + jump
    )
    ACTION_NAMES = (
        'idle',
        'right',
        'right_run',
        'jump',
        'right_jump',
        'right_run_jump',
        'left',
        'left_jump',
    )

    def __init__(self, env: ToyPlatformerEnv):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.ACTION_NAMES = list(self.ACTION_NAMES)
        self.observation_mode = str(env.observation_mode)
        self.height = int(env.height)
        self.camera_width = int(env.camera_width)
        self.include_state_features = bool(env.include_state_features)
        self.SCREEN_CHANNEL_NAMES = list(env.SCREEN_CHANNEL_NAMES)
        self.STATE_FEATURE_NAMES = list(env.STATE_FEATURE_NAMES)
        self.input_feature_names = list(env.get_input_feature_names())

    def step(self, action):
        action_idx = int(action)
        if not 0 <= action_idx < len(self.ACTIONS):
            raise ValueError(f'Invalid discrete action {action_idx}')
        continuous_action = np.asarray(self.ACTIONS[action_idx], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        info['discrete_action'] = action_idx
        info['discrete_action_name'] = self.ACTION_NAMES[action_idx]
        return obs, reward, terminated, truncated, info

    def get_input_feature_names(self) -> list[str]:
        return list(self.input_feature_names)


class FullScreenFrameStack(gym.Wrapper):
    def __init__(self, env: ToyPlatformerEnv, num_frames: int):
        super().__init__(env)
        self.num_frames = int(num_frames)
        if self.num_frames < 1:
            raise ValueError('frame_stack must be at least 1')
        if env.observation_mode != 'full_screen':
            raise ValueError('frame_stack is only supported for full_screen observations')

        self.height = int(env.height)
        self.camera_width = int(env.camera_width)
        self.include_state_features = bool(env.include_state_features)
        self.base_screen_channels = len(env.SCREEN_CHANNEL_NAMES)
        self.screen_dim = self.height * self.camera_width * self.base_screen_channels
        self.state_dim = len(env.STATE_FEATURE_NAMES) if self.include_state_features else 0

        self.SCREEN_CHANNEL_NAMES = [
            f'{name}_t-{frame_offset}'
            for frame_offset in reversed(range(self.num_frames))
            for name in env.SCREEN_CHANNEL_NAMES
        ]
        self.STATE_FEATURE_NAMES = list(env.STATE_FEATURE_NAMES)
        self.input_feature_names = self._build_input_feature_names()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.input_feature_names),),
            dtype=np.float32,
        )
        self.frames: deque[np.ndarray] = deque(maxlen=self.num_frames)

    def _build_input_feature_names(self) -> list[str]:
        names: list[str] = []
        for frame_offset in reversed(range(self.num_frames)):
            for screen_row in range(self.height):
                for screen_col in range(self.camera_width):
                    for channel_name in self.env.SCREEN_CHANNEL_NAMES:
                        names.append(f'screen_t-{frame_offset}_r{screen_row}_c{screen_col}_{channel_name}')
        if self.include_state_features:
            names.extend(self.STATE_FEATURE_NAMES)
        return names

    def _split_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        screen = obs[: self.screen_dim]
        state = obs[self.screen_dim :] if self.state_dim > 0 else np.empty(0, dtype=np.float32)
        return screen, state

    def _stack_obs(self, state: np.ndarray) -> np.ndarray:
        if len(self.frames) != self.num_frames:
            raise RuntimeError('frame stack is not initialized')
        screen = np.concatenate(list(self.frames), axis=0)
        if self.state_dim <= 0:
            return screen.astype(np.float32, copy=False)
        return np.concatenate((screen, state)).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        screen, state = self._split_obs(obs)
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(screen.copy())
        return self._stack_obs(state), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        screen, state = self._split_obs(obs)
        self.frames.append(screen.copy())
        return self._stack_obs(state), reward, terminated, truncated, info

    def get_input_feature_names(self) -> list[str]:
        return list(self.input_feature_names)


def make_env(config: dict[str, Any]) -> gym.Env:
    env_cfg = config['env']
    name = env_cfg['name']

    if name == 'toy_platformer':
        signature = inspect.signature(ToyPlatformerEnv.__init__)
        kwargs: dict[str, Any] = {}
        for key, value in env_cfg.items():
            if key in {
                'name',
                'train_seed_mode',
                'eval_seed_start',
                'num_eval_seeds',
                'frame_stack',
                'action_mode',
                'discrete_actions',
            }:
                continue
            if key in signature.parameters:
                kwargs[key] = value
        kwargs.setdefault('seed', int(config['project'].get('seed', 42)))
        env = ToyPlatformerEnv(**kwargs)
        action_mode = str(env_cfg.get('action_mode', 'continuous'))
        if bool(env_cfg.get('discrete_actions', False)) or action_mode == 'discrete_basic':
            env = DiscreteToyActionWrapper(env)
        frame_stack = int(env_cfg.get('frame_stack', 1))
        if frame_stack > 1:
            return FullScreenFrameStack(env, frame_stack)
        return env

    if name.startswith('SuperMarioBros'):
        from echomario.envs.mario_env import make_mario_env

        return make_mario_env(config)

    raise ValueError(f'Unknown env name: {name}')
