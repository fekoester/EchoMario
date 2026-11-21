"""
Look-ahead teacher for Super Mario World, using emulator save-states.

At each decision it tries every first action for ``first_hold`` steps, then each continuation
macro for ``horizon`` steps, and scores an action by its best continuation: pixels of new
progress, minus penalties for dying or losing a power-up, plus a bonus for the goal tape.

With ``deep_stages`` > 0 (teacher v2) a 3-second safety check is added: actions keep their
1-second score, but an action is vetoed when every 3-second continuation found from it dies
(running into a pit or a hazard whose timing is wrong), so the next safe action wins, e.g. waiting.
The check grows each action's best ``deep_beam[k]`` plans (the furthest one, then the furthest one
standing on the ground) by another ``horizon`` steps of ``deep_continuations``, and keeps growing
further plans while all of them die. Ranking actions by the 3-second scores directly was tried
first: with a narrow beam those scores are noisy (+-70 px), and the teacher procrastinated, idling
forever because "idle, then the lucky plan" kept winning in an unchanged state.
Macros that start with the same buttons share that prefix instead of emulating it twice.
In SMW a jump starts only on a fresh B press, so the continuations include 'hop' (press, release,
press again) as well as one long held jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    # teacher v2: extra look-ahead stages of `horizon` steps each (0 = the 1-second teacher)
    deep_stages: int = 0
    deep_beam: tuple[int, ...] = (2, 1)
    deep_continuations: tuple[str, ...] = ('run', 'hop', 'longjump', 'wait_long', 'runup')
    # actions within this many pixels of the best count as tied (priority decides)
    tie_tolerance: float = 0.0
    # the safety check runs for actions within this many pixels of the best safe action (anything
    # further behind gets a zero target with q_clip <= this anyway, and is never chosen)
    deep_margin: float = 32.0
    # if every plan grown from the beam dies, keep growing the next ones, up to this many per stage
    deep_max_expand: int = 6
    # collector bonuses, in pixels of progress: per coin, per point of the game's score (stomps 200+,
    # shell kills 1000, dragon coins), per power-up step (small -> big -> cape/fire). 0 = speedrunner.
    coin_bonus: float = 0.0
    score_bonus: float = 0.0
    powerup_bonus: float = 0.0
    # stomp combos escalate (200 ... 8000 points); counted in full they turn the teacher into an enemy
    # farmer, so the score gain since the decision saturates: C * (1 - exp(-gain / C)) points
    score_saturation: float = 1000.0
    # pit check: 'window' looks for ground in the bottom of the observation grid (v1, so it depends on the
    # student's window: with a 16x20 grid v2 stalled at Donut Plains 1's gap), 'column' looks straight down
    # to the bottom of the level, platform sprites included (SMWEnv.ground_below)
    ground_check: str = 'window'
    # the safety check vetoes an action when every 3-s plan from it dies. When no plan in the macro set gets
    # past a hazard (a jump that needs exact timing), only waiting survives and the teacher would wait forever:
    # after this many decisions without progress, the veto pauses until Mario moves forward again
    veto_patience: int = 20


def teacher_config(name: str) -> SMWPlannerConfig:
    """'v1': 1-second look-ahead. 'v2': 3 seconds (two extra stages, beam 2 then 1)."""
    if name == 'v1':
        return SMWPlannerConfig()
    if name == 'v2':
        return SMWPlannerConfig(deep_stages=2, deep_beam=(2, 1), ground_check='column')
    if name == 'v2-collector':  # v2, but coins, stomped enemies and power-ups count too
        return SMWPlannerConfig(deep_stages=2, deep_beam=(2, 1), ground_check='column', coin_bonus=16.0,
                                score_bonus=0.16, powerup_bonus=200.0)
    raise ValueError(f'unknown teacher {name!r}')


@dataclass
class _Leaf:
    score: float
    ended: bool
    state: tuple | None = None
    on_ground: bool = False
    falling_into_void: bool = False  # falling with no ground below in view
    doomed: bool = False  # the 1-second teacher's pit-doom check (below the start height, no ground)


def _prefix_plan(macros: list[list[int]]) -> list[tuple[int, int]]:
    """For each macro: (parent macro, shared prefix length) with the earlier macro sharing the
    longest prefix; (-1, 0) when nothing is shared."""
    plan = []
    for i, m in enumerate(macros):
        best = (-1, 0)
        for j in range(i):
            n = 0
            while n < min(len(m), len(macros[j])) and m[n] == macros[j][n]:
                n += 1
            if n > best[1]:
                best = (j, n)
        plan.append(best)
    return plan


class SMWPlanner:
    def __init__(self, cfg: SMWPlannerConfig | None = None):
        self.cfg = cfg or SMWPlannerConfig()
        self.first = [A[n] for n in self.cfg.first_actions]
        self.conts = [_macro(n, self.cfg.horizon) for n in self.cfg.continuations]
        self.deep_conts = [_macro(n, self.cfg.horizon) for n in self.cfg.deep_continuations]
        self._plans = {id(self.conts): _prefix_plan(self.conts), id(self.deep_conts): _prefix_plan(self.deep_conts)}
        self.rank = {A[n]: i for i, n in enumerate(self.cfg.priority)}
        self.rollout_steps = 0
        self._best_x, self._stalled = -1, 0
        self._root_counts = (0, 0, 0)
        self._collect = bool(self.cfg.coin_bonus or self.cfg.score_bonus or self.cfg.powerup_bonus)

    @staticmethod
    def _counts(env: SMWEnv) -> tuple[int, int, int]:
        w = env._wram()
        return (int(w[0xDBF]), int(w[0xF34]) | (int(w[0xF35]) << 8) | (int(w[0xF36]) << 16), int(w[0x19]))

    def _value(self, env: SMWEnv, start_x: int) -> float:
        """Pixels of new progress, plus the collector bonuses since the decision being planned."""
        v = float(env.track.max_x - start_x)
        if self._collect:
            c, sc, p = self._counts(env)
            c0, sc0, p0 = self._root_counts
            coins = c - c0 if c >= c0 else c + 100 - c0  # the coin counter wraps at 100 (1-up)
            sat = self.cfg.score_saturation
            points = sat * (1.0 - math.exp(-10.0 * max(0, sc - sc0) / sat))  # the game stores score / 10
            v += self.cfg.coin_bonus * coins + self.cfg.score_bonus * points + self.cfg.powerup_bonus * max(0, p - p0)
        return v

    def _no_ground(self, env: SMWEnv) -> bool:
        if self.cfg.ground_check == 'column':
            return not env.ground_below()
        g = env.grid()
        return not g[env.rows_above:, max(0, env.cols_behind - 1):env.cols_behind + 2, :2].any()

    def _doomed(self, env: SMWEnv, start_y: int) -> bool:
        """Falling, below the starting height, with no ground under Mario: a pit death that has not
        happened yet when the look-ahead ends."""
        m = env.mario()
        if m['on_ground'] or m['vy'] <= 0 or m['y'] <= start_y + 8:
            return False
        return self._no_ground(env)

    def _run(self, env: SMWEnv, actions: list[int], start_x: int, start_y: int | None = None) -> tuple[float, bool]:
        """Plays the actions; returns (score so far, ended)."""
        for a in actions:
            env.emulate(a)
            _, _, terminated, _, info = env._after_step(compute_obs=False)
            self.rollout_steps += 1
            if info['hurt']:
                return self._value(env, start_x) - self.cfg.hurt_penalty, True
            if terminated:
                if info['died']:
                    return self._value(env, start_x) - self.cfg.death_penalty, True
                return self._value(env, start_x) + self.cfg.goal_bonus, True
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
                        return self._value(env, start_x) - pen, True
                    return self._value(env, start_x) + self.cfg.goal_bonus, True
                if env.mario()['on_ground']:
                    break
        if start_y is not None and self._doomed(env, start_y):
            return self._value(env, start_x) - self.cfg.death_penalty, True
        return self._value(env, start_x), False

    def _expand(self, env: SMWEnv, state: tuple, macros: list[list[int]], start_x: int, start_y: int,
                final: bool, keep_state: bool) -> list[_Leaf]:
        """Plays every macro from ``state``; macros sharing a prefix replay it from a saved state.
        ``final``: apply the pit-doom check at the end (last look-ahead stage)."""
        plan = self._plans[id(macros)]
        need: dict[int, set[int]] = {}
        for j, n in plan:
            if j >= 0:
                need.setdefault(j, set()).add(n)
        saved: dict[tuple[int, int], tuple] = {}  # (macro, step) -> (sim state, score, ended)
        leaves = []
        for i, actions in enumerate(macros):
            j, n = plan[i]
            score, ended = self._value(env, start_x), False
            if j >= 0 and (j, n) in saved:
                st, score, ended = saved[(j, n)]
                if ended:
                    leaves.append(_Leaf(score, True))
                    continue
                env.set_sim_state(st)
            else:
                env.set_sim_state(state)
                n = 0
            saves = need.get(i, set())
            for k in range(n, len(actions)):
                if k in saves:
                    saved[(i, k)] = (env.get_sim_state(), self._value(env, start_x), False)
                score, ended = self._step(env, actions[k], start_x)
                if ended:
                    break
            if ended:
                for k in saves:  # later macros sharing a prefix past the end end the same way
                    saved.setdefault((i, k), (None, score, True))
                leaves.append(_Leaf(score, True))
                continue
            if final and self._doomed(env, start_y):
                leaves.append(_Leaf(self._value(env, start_x) - self.cfg.death_penalty, True))
                continue
            m = env.mario()
            void = not m['on_ground'] and m['vy'] > 0 and self._no_ground(env)
            leaves.append(_Leaf(self._value(env, start_x), False,
                                env.get_sim_state() if keep_state else None, m['on_ground'], bool(void),
                                bool(void and m['y'] > start_y + 8)))
        return leaves

    def _step(self, env: SMWEnv, a: int, start_x: int) -> tuple[float, bool]:
        env.emulate(a)
        _, _, terminated, _, info = env._after_step(compute_obs=False)
        self.rollout_steps += 1
        if info['hurt']:
            return self._value(env, start_x) - self.cfg.hurt_penalty, True
        if terminated:
            if info['died']:
                return self._value(env, start_x) - self.cfg.death_penalty, True
            return self._value(env, start_x) + self.cfg.goal_bonus, True
        return self._value(env, start_x), False

    @staticmethod
    def _rank(leaves: list[_Leaf]) -> list[_Leaf]:
        """Open plans in the order they are grown: the furthest, then the furthest one standing on
        the ground (so 'wait for it' plans are not crowded out by jumps), then the rest by score;
        plans falling into the void go last."""
        open_ = sorted((lf for lf in leaves if not lf.ended), key=lambda lf: (lf.falling_into_void, -lf.score))
        if len(open_) <= 1:
            return open_
        out = [open_[0]]
        grounded = [lf for lf in open_[1:] if lf.on_ground and not lf.falling_into_void]
        if grounded:
            out.append(grounded[0])
        return out + [lf for lf in open_[1:] if lf is not out[-1]]

    def _survives(self, lf: _Leaf) -> bool:
        return not lf.ended or lf.score >= self.cfg.goal_bonus / 2

    def scores(self, env: SMWEnv) -> np.ndarray:
        """Score of every action in ACTION_NAMES (-inf for actions not tried)."""
        if self.cfg.settle_steps:
            return self._scores_settle(env)
        rendering = env.rendering
        env.set_render(False)  # nobody looks at the rollouts
        if self._collect:
            self._root_counts = self._counts(env)
        root = env.get_sim_state()
        start_x = env.track.max_x
        start_y = env.mario()['y']
        out = np.full(len(ACTION_NAMES), -np.inf, np.float64)
        stages = self.cfg.deep_stages
        stage1: dict[int, list[_Leaf]] = {}
        for a in self.first:
            env.set_sim_state(root)
            s1, ended = self._run(env, [a] * self.cfg.first_hold, start_x)
            if ended:
                out[a] = s1
                continue
            mid = env.get_sim_state()
            leaves = self._expand(env, mid, self.conts, start_x, start_y, final=stages == 0, keep_state=stages > 0)
            # the 1-second score (with the pit-doom check, exactly as without deep stages)
            out[a] = max(lf.score - (self.cfg.death_penalty if lf.doomed and not lf.ended else 0.0) for lf in leaves)
            stage1[a] = leaves
        if start_x > self._best_x:
            self._best_x, self._stalled = start_x, 0
        else:
            self._stalled += 1
        if stages and self._stalled < self.cfg.veto_patience:
            out = self._safety_check(env, out, stage1, start_x, start_y)
        env.set_sim_state(root)
        env.set_render(rendering)
        return out

    def _safety_check(self, env: SMWEnv, s1: np.ndarray, stage1: dict[int, list[_Leaf]], start_x: int,
                      start_y: int) -> np.ndarray:
        """Vetoes actions whose every 3-second continuation dies (see the module docstring)."""
        out = s1.copy()
        best_safe = None
        for a in sorted(stage1, key=lambda a_: -s1[a_]):
            if best_safe is not None and s1[a] < best_safe - self.cfg.deep_margin:
                break
            leaves = stage1[a]
            for k in range(self.cfg.deep_stages):
                width = self.cfg.deep_beam[min(k, len(self.cfg.deep_beam) - 1)]
                done = [lf for lf in leaves if lf.ended]
                grown = []
                for n, lf in enumerate(self._rank(leaves)):
                    if n >= self.cfg.deep_max_expand or (n >= width and any(self._survives(c) for c in grown)):
                        break
                    grown += self._expand(env, lf.state, self.deep_conts, start_x, start_y,
                                          final=k == self.cfg.deep_stages - 1, keep_state=k < self.cfg.deep_stages - 1)
                leaves = done + grown
            if any(self._survives(lf) for lf in leaves):
                best_safe = s1[a] if best_safe is None else max(best_safe, s1[a])
            else:
                out[a] = min(s1[a], max(lf.score for lf in leaves))
        return out if best_safe is not None else s1  # nothing survives 3 s: keep the 1-second view

    def _scores_settle(self, env: SMWEnv) -> np.ndarray:
        """The original implementation, kept for ``settle_steps`` > 0."""
        rendering = env.rendering
        env.set_render(False)
        if self._collect:
            self._root_counts = self._counts(env)
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
        env.set_render(rendering)
        return out

    def choose(self, scores: np.ndarray) -> int:
        best = scores.max()
        cands = [a for a in range(len(scores)) if scores[a] >= best - max(1e-6, self.cfg.tie_tolerance)]
        return min(cands, key=lambda a: self.rank.get(a, 99))

    def act(self, env: SMWEnv) -> int:
        return self.choose(self.scores(env))
