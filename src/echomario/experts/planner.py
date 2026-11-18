"""
Look-ahead planner that plays the toy platformer well enough to act as a teacher.

The planner never sees observations. It clones the environment once per level, copies the live
dynamic state into the clone before each decision, and scores short action sequences by
simulating them. It picks the first action of the best sequence (receding horizon).

A sequence is: ``first`` action held for ``first_hold`` frames, then ``second`` action held for
``second_hold`` frames, then the default ``right_run`` for ``tail`` frames. Its score is the
summed environment reward plus a large penalty if the player dies, so the teacher is cautious.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from echomario.envs.make_env import DiscreteToyActionWrapper
from echomario.envs.toy_platformer_env import ToyPlatformerEnv

ACTIONS = np.asarray(DiscreteToyActionWrapper.ACTIONS, dtype=np.float32)
ACTION_NAMES = DiscreteToyActionWrapper.ACTION_NAMES
RIGHT_RUN = ACTION_NAMES.index('right_run')
RIGHT_RUN_JUMP = ACTION_NAMES.index('right_run_jump')
RIGHT_JUMP = ACTION_NAMES.index('right_jump')
IDLE = ACTION_NAMES.index('idle')
LEFT = ACTION_NAMES.index('left')
RIGHT = ACTION_NAMES.index('right')

# Tie-break order: prefer simple, consistent labels so the student sees a learnable mapping.
_PRIORITY = [RIGHT_RUN, RIGHT_RUN_JUMP, RIGHT_JUMP, RIGHT, IDLE, ACTION_NAMES.index('jump'), LEFT,
             ACTION_NAMES.index('left_jump')]


@dataclass
class PlannerConfig:
    # 3 + 14 + 14 frames of look-ahead. The 14-frame second macro allows a full-height jump
    # (jump boost lasts 13 frames), which is needed to get onto raised bridges over wide gaps.
    first_hold: int = 3
    second_hold: int = 14
    tail: int = 14
    second_actions: tuple[int, ...] = (RIGHT_RUN, RIGHT_RUN_JUMP, RIGHT_JUMP, LEFT)
    # Ending with a run-jump lets "back up, then run-jump" plans get onto a bridge from below.
    tail_actions: tuple[int, ...] = (RIGHT_RUN, RIGHT_RUN_JUMP)
    death_penalty: float = 200.0
    tie_eps: float = 1e-3


def unwrap_toy(env) -> ToyPlatformerEnv:
    base = env
    while not isinstance(base, ToyPlatformerEnv):
        base = base.env
    return base


class LookaheadPlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()
        self._sim: ToyPlatformerEnv | None = None
        self._level_key = None

    def attach(self, env) -> None:
        """Clones the current level. Call after every ``env.reset``."""
        base = unwrap_toy(env)
        sim = copy.deepcopy(base)
        sim.compute_obs = False
        self._sim = sim

    def _run(self, action_idx: int, frames: int) -> tuple[float, bool]:
        """Holds one action in the clone. Returns (reward incl. death penalty, episode_over)."""
        sim = self._sim
        action = ACTIONS[action_idx]
        total = 0.0
        for _ in range(frames):
            _obs, reward, terminated, truncated, info = sim.step(action)
            total += reward
            if terminated or truncated:
                if info['fell'] or info['hit_enemy'] or info['stagnated']:
                    total -= self.cfg.death_penalty
                return total, True
        return total, False

    def scores(self, env) -> np.ndarray:
        """Best achievable score for each first action from the env's current state."""
        if self._sim is None:
            self.attach(env)
        cfg = self.cfg
        sim = self._sim
        state = unwrap_toy(env).get_sim_state()
        out = np.full(len(ACTIONS), -np.inf, dtype=np.float64)
        for first in range(len(ACTIONS)):
            sim.set_sim_state(state)
            head, over = self._run(first, cfg.first_hold)
            if over:
                out[first] = head
                continue
            mid = sim.get_sim_state()  # the first macro is shared by all continuations
            best = -np.inf
            for second in cfg.second_actions:
                sim.set_sim_state(mid)
                score, over = self._run(second, cfg.second_hold)
                if not over and cfg.tail > 0:
                    branch = sim.get_sim_state() if len(cfg.tail_actions) > 1 else None
                    tails = []
                    for k, tail_action in enumerate(cfg.tail_actions):
                        if k > 0:
                            sim.set_sim_state(branch)
                        tails.append(self._run(tail_action, cfg.tail)[0])
                    score += max(tails)
                best = max(best, score)
            out[first] = head + best
        return out

    def act(self, env) -> int:
        return self.choose(self.scores(env))

    def choose(self, scores: np.ndarray) -> int:
        best = scores.max()
        for action_idx in _PRIORITY:
            if scores[action_idx] >= best - self.cfg.tie_eps:
                return int(action_idx)
        return int(np.argmax(scores))
