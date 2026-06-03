from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarioObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        frame_width: int = 32,
        frame_height: int = 32,
        frame_stack: int = 1,
        crop_top: int = 32,
        crop_bottom: int = 16,
    ):
        super().__init__(env)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.frame_stack = int(frame_stack)
        self.crop_top = int(crop_top)
        self.crop_bottom = int(crop_bottom)
        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

        obs_dim = self.frame_width * self.frame_height * self.frame_stack
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.crop_bottom > 0:
            cropped = frame[self.crop_top : -self.crop_bottom]
        else:
            cropped = frame[self.crop_top :]
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(
            gray,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA,
        )
        return small.astype(np.float32) / 255.0

    def observation(self, observation: np.ndarray) -> np.ndarray:
        processed = self._process_frame(observation)
        if not self.frames:
            for _ in range(self.frame_stack):
                self.frames.append(processed)
        else:
            self.frames.append(processed)
        stacked = np.stack(list(self.frames), axis=0)
        return stacked.reshape(-1).astype(np.float32)

    def reset(self, **kwargs):
        self.frames.clear()
        return super().reset(**kwargs)


class MarioInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info['reached_goal'] = bool(info.get('flag_get', False))
        info['max_x'] = float(info.get('x_pos', info.get('x_pos_screen', 0.0)))
        return obs, reward, terminated, truncated, info


MOVEMENT_SETS = {
    'simple': 'SIMPLE_MOVEMENT',
    'right_only': 'RIGHT_ONLY',
    'complex': 'COMPLEX_MOVEMENT',
}


def make_mario_env(config: dict[str, Any]):
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros import actions as mario_actions
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise ImportError(
            'Mario backend is optional. Install optional dependencies later. '
            'This repo does not include ROMs or game assets.'
        ) from exc

    env_cfg = config['env']
    env_name = env_cfg['name']
    movement_set_name = str(env_cfg.get('movement_set', 'simple')).lower()
    movement_attr = MOVEMENT_SETS.get(movement_set_name, 'SIMPLE_MOVEMENT')
    movement_set = getattr(mario_actions, movement_attr)

    env = gym_super_mario_bros.make(env_name, render_mode='rgb_array')
    env = JoypadSpace(env, movement_set)
    env = MarioInfoWrapper(env)
    env = MarioObservationWrapper(
        env,
        frame_width=int(env_cfg.get('frame_width', 32)),
        frame_height=int(env_cfg.get('frame_height', 32)),
        frame_stack=int(env_cfg.get('frame_stack', 1)),
        crop_top=int(env_cfg.get('crop_top', 32)),
        crop_bottom=int(env_cfg.get('crop_bottom', 16)),
    )
    return env
