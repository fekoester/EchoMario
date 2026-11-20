"""
Look-ahead teacher for Super Mario World, using emulator save-states.

At each decision it tries every first action for ``first_hold`` steps, then each continuation
macro for ``horizon`` steps, and scores an action by its best continuation: pixels of new
progress, minus penalties for dying or losing a power-up, plus a bonus for the goal tape.
In SMW a jump starts only on a fresh B press, so the continuations include 'hop' (press, release,
press again) as well as one long held jump.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv

A = {n: i for i, n in enumerate(ACTION_NAMES)}


def _macro(name: str, n: int) -> list[int]:
    if name == 'run':
        return [A['right_run']] * n
    if name == 'hop':
        return [(A['right_run_jump'] if (i % 14) < 12 else A['right_run']) for i in range(n)]
    if name == 'longjump':
        return [A['right_run_jump']] * n
    if name == 'wait_hop':
        return [A['idle']] * 10 + _macro('hop', n - 10)
    if name == 'wait_long':  # moving platforms, enemies passing by
        return [A['idle']] * 20 + _macro('hop', n - 20)
    if name == 'back_hop':
        return [A['left']] * 6 + _macro('hop', n - 6)
    if name == 'runup':  # back up for a running start, then run and jump (wide gaps)
        k = n // 3
        return [A['left']] * k + [A['right_run']] * k + [A['right_run_jump']] * (n - 2 * k)
    raise ValueError(name)


@dataclass
class SMWPlannerConfig:
    first_actions: tuple[str, ...] = ('right_run', 'right_run_jump', 'right_jump', 'right', 'idle', 'left')
    first_hold: int = 4
    horizon: int = 30
    continuations: tuple[str, ...] = ('run', 'hop', 'longjump', 'wait_hop', 'runup')
    # if > 0 and airborne when the look-ahead ends, keep going until landed (or dead). Tried 40:
    # twice as slow and no better on YoshiIsland3/4, DonutPlains4, Forest3, so off by default.
    settle_steps: int = 0
    death_penalty: float = 2000.0
    hurt_penalty: float = 300.0
    goal_bonus: float = 3000.0
    priority: tuple[str, ...] = ('right_run', 'right_run_jump', 'right_jump', 'right', 'idle', 'left')


class SMWPlanner:
    def __init__(self, cfg: SMWPlannerConfig | None = None):
        self.cfg = cfg or SMWPlannerConfig()
        self.first = [A[n] for n in self.cfg.first_actions]
        self.conts = [_macro(n, self.cfg.horizon) for n in self.cfg.continuations]
        self.rank = {A[n]: i for i, n in enumerate(self.cfg.priority)}
        self.rollout_steps = 0

    @staticmethod
    def _doomed(env: SMWEnv, start_y: int) -> bool:
        """Falling, below the starting height, with no ground under Mario anywhere in view:
        a pit death that has not happened yet when the look-ahead ends."""
        m = env.mario()
        if m['on_ground'] or m['vy'] <= 0 or m['y'] <= start_y + 8:
            return False
        g = env.grid()
        below = g[env.rows_above:, max(0, env.cols_behind - 1):env.cols_behind + 2, :2]
        return not below.any()

    def _run(self, env: SMWEnv, actions: list[int], start_x: int, start_y: int | None = None) -> tuple[float, bool]:
        """Plays the actions; returns (score so far, ended)."""
        for a in actions:
            env.emulate(a)
            _, _, terminated, _, info = env._after_step(compute_obs=False)
            self.rollout_steps += 1
            if info['hurt']:
                return env.track.max_x - start_x - self.cfg.hurt_penalty, True
            if terminated:
                if info['died']:
                    return env.track.max_x - start_x - self.cfg.death_penalty, True
                return env.track.max_x - start_x + self.cfg.goal_bonus, True
        if start_y is not None and self.cfg.settle_steps and not env.mario()['on_ground']:
            # A plan that ends in mid-air over a pit is not a good plan yet: let it land.
            last = actions[-1] if actions else A['right_run']
            for _ in range(self.cfg.settle_steps):
                env.emulate(last)
                _, _, terminated, _, info = env._after_step(compute_obs=False)
                self.rollout_steps += 1
                if terminated or info['hurt']:
                    if info['died'] or info['hurt']:
                        pen = self.cfg.death_penalty if info['died'] else self.cfg.hurt_penalty
                        return env.track.max_x - start_x - pen, True
                    return env.track.max_x - start_x + self.cfg.goal_bonus, True
                if env.mario()['on_ground']:
                    break
        if start_y is not None and self._doomed(env, start_y):
            return env.track.max_x - start_x - self.cfg.death_penalty, True
        return float(env.track.max_x - start_x), False

    def scores(self, env: SMWEnv) -> np.ndarray:
        """Score of every action in ACTION_NAMES (-inf for actions not tried)."""
        root = env.get_sim_state()
        start_x = env.track.max_x
        start_y = env.mario()['y']
        out = np.full(len(ACTION_NAMES), -np.inf, np.float64)
        for a in self.first:
            env.set_sim_state(root)
            s1, ended = self._run(env, [a] * self.cfg.first_hold, start_x)
            if ended:
                out[a] = s1
                continue
            mid = env.get_sim_state()
            best = -np.inf
            for cont in self.conts:
                env.set_sim_state(mid)
                s, _ = self._run(env, cont, start_x, start_y)
                best = max(best, s)
                if best >= self.cfg.goal_bonus:
                    break
            out[a] = best
        env.set_sim_state(root)
        return out

    def choose(self, scores: np.ndarray) -> int:
        best = scores.max()
        cands = [a for a in range(len(scores)) if scores[a] >= best - 1e-6]
        return min(cands, key=lambda a: self.rank.get(a, 99))

    def act(self, env: SMWEnv) -> int:
        return self.choose(self.scores(env))
