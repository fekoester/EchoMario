"""
A* teacher for Super Mario World: a deterministic best-first search over emulator save-states.

Teacher v3 scores every button with a handful of fixed macros plus a random rescue search. Its verdicts at
borderline states are partly random (the same state rescored with another seed flips 2-8 % of the deadly/safe
verdicts; identical inputs within an episode disagree in 20 %), and the fixed macros miss many escapes.

This teacher scores each first button (held ``seg`` steps, the decision cadence) by the best plan found
within ``horizon`` steps, searching over sequences of ``seg``-step segments of any button. The segments are
aligned with the decisions, so a plan found at one decision is still a plan at the next one.

- value of a plan: progress beyond Mario's current x at the horizon (or the goal bonus), minus penalties for
  being hurt, plus ``earliness`` x the mean progress over the plan's segments (progress now beats the same
  progress later: no procrastination). A plan that dies, or ends falling into a pit, is deadly.
- A* bound: the furthest Mario could get from his current x and speed, accelerating at most ``acc`` px per
  step per step up to run speed (``v_run``), and up to P-speed (``vmax``) only once the P-meter could be full
  (measured on flat ground: +0.375 px/step per step, 4.6 and 6.1 px/step, P-meter +4 per step at run speed);
  the same bound for the earliness term. Nodes are simulated lazily: a child is emulated only when it is
  popped, so a straight run costs one segment per level of the tree.
- phase 1, verdicts: each button is searched depth first (deepest node, then best bound, then button prior)
  until a plan reaches the horizon alive (safe) or ``verdict_steps`` are spent / its tree is exhausted
  (deadly).
- phase 2, values: the node with the highest bound is expanded among the buttons that could still matter
  (bound above their best plan and within ``q_clip`` of the best button), until no bound exceeds what has
  been found or the step budget is spent.
- duplicates: a node in the same coarse state as an earlier one of the same first button (Mario's 8-px cell,
  speed bucket, ground flag, power, time bucket of 16 steps) with at least as much progress is dropped. This
  keeps the tree from exploding where nothing makes progress, so detours can be found.
- stalls: when no button gains ``stall_px`` within the horizon (a wall, a ledge or gap that needs a run-up),
  the teacher enters stall mode until Mario gets ``stall_px`` past the stall x (or ``stall_decisions`` pass):
  progress is then measured beyond the stall x (so backing up for a run-up is not undone by "run right
  again"), and the search is breadth first over distinct coarse Mario states (time ignored) with
  ``long_horizon`` and ``long_budget``, which finds run-ups and detours. Scores in stall mode depend on the
  stall x, i.e. on the obstacle Mario is trying to pass.
- memory: the plan behind the executed button is kept; at the next decision it is replayed (deterministic
  emulator), and if it still survives, its first button cannot be scored deadly. A found escape is not lost
  to a search that misses it the second time. (This is the only history the scores depend on; it can only
  turn a wrong "deadly" verdict into a correct "safe" one.)

Only the choice of the action to execute may be randomized among buttons within ``drive_tolerance`` px of the
best (diverse trajectories); the labels (scores) do not depend on it.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv
from echomario.experts.smw_planner import SMWPlanner, SMWPlannerConfig

A = {n: i for i, n in enumerate(ACTION_NAMES)}
PRIOR = ('right_run', 'right_run_jump', 'right_jump', 'right', 'idle', 'jump', 'left', 'left_jump')


@dataclass
class AStarConfig:
    seg: int = 4  # = the decision cadence (plan_every)
    horizon: int = 88
    vmax: float = 6.25  # px per step (2 frames): P-speed is ~3 px per frame
    v_run: float = 4.75  # run speed without a full P-meter
    acc: float = 0.45  # px per step per step (measured 0.375)
    p_rate: float = 4.5  # P-meter per step at run speed (measured 4; full at 112)
    earliness: float = 0.1
    eps: float = 4.0  # stop expanding a button when no bound beats its best plan by more than this
    q_clip: float = 32.0  # buttons further than this behind the best get a zero target anyway
    budget_steps: int = 1500  # emulator steps per decision (phase 1 may use more)
    verdict_steps: int = 400  # per button in phase 1
    stall_px: float = 16.0
    long_horizon: int = 240
    long_budget: int = 8000
    stall_decisions: int = 60
    stall_retry_budget: int = 2000  # after a stall search found no way past, the next ones search less
    time_bucket: int = 16
    death_penalty: float = 2000.0
    hurt_penalty: float = 300.0
    goal_bonus: float = 3000.0
    drive_tolerance: float = 0.0  # when driving, pick randomly among buttons this close to the best
    seed: int = 0


class _Node:
    __slots__ = ('state', 't', 'gain', 'gsum', 'pen', 'x', 'v', 'pm', 'parent', 'a')

    def __init__(self, state, t, gain, gsum, pen, x=0.0, v=0.0, pm=0, parent=None, a=-1):
        self.state, self.t, self.gain, self.gsum, self.pen = state, t, gain, gsum, pen
        self.x, self.v, self.pm, self.parent, self.a = x, v, pm, parent, a

    def actions(self) -> list[int]:
        out, n = [], self
        while n is not None and n.a >= 0:
            out.append(n.a)
            n = n.parent
        return out[::-1]


class _Entry:
    """A child not emulated yet (lazy expansion); sits in both the depth-first and the best-first heap."""
    __slots__ = ('node', 'a', 'done')

    def __init__(self, node: _Node, a: int):
        self.node, self.a, self.done = node, a, False


class SMWAStar:
    def __init__(self, cfg: AStarConfig | None = None):
        self.cfg = cfg or AStarConfig()
        self.first = [A[n] for n in PRIOR]
        self.children = [A[n] for n in PRIOR]
        self.rank = {A[n]: i for i, n in enumerate(PRIOR)}
        self._pit = SMWPlanner(SMWPlannerConfig(ground_check='column'))  # for its pit-doom check
        self._rng = np.random.default_rng(self.cfg.seed)
        self.rollout_steps = 0
        self.last_stats: dict = {}
        self._paths: dict[int, list[int]] = {}  # best plan per first button (last decision)
        self._plan: list[int] = []  # rest of the plan behind the executed button
        self.rescued = self.rescue_tried = self.followed = 0  # rescued: deadly verdicts overturned by memory
        self._stall_x: int | None = None
        self._stall_n = 0
        self._stall_fails = 0
        self.stall_decisions = 0

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ search
    def _reach(self, n: _Node, rem: int) -> np.ndarray:
        """Upper bound of Mario's x - root x after 1..rem more steps."""
        c = self.cfg
        k = np.arange(1, rem + 1, dtype=np.float64)
        v0 = max(0.0, n.v)
        if v0 >= c.v_run or n.pm >= 112:
            k_full = 0.0 if n.pm >= 112 or v0 > c.v_run + 0.5 else max(0.0, (112 - n.pm) / c.p_rate)
        else:
            k_full = (c.v_run - v0) / c.acc + (112 - n.pm) / c.p_rate
        cap = np.where(k >= k_full, c.vmax, c.v_run)
        v = np.minimum(cap, v0 + c.acc * k)
        return n.x + np.cumsum(v)

    def _bound(self, n: _Node, H: int) -> float:
        c = self.cfg
        rem = max(0, H - n.t)
        if rem == 0:
            return self._leaf_value(n, False, H)
        g = np.maximum(n.gain, self._reach(n, rem))
        ends = np.arange(c.seg, rem + 1, c.seg)  # future segment ends (t is a multiple of seg)
        early = n.gsum + g[ends - 1].sum()
        return float(g[-1]) + c.earliness * early / (H // c.seg) - n.pen

    def _leaf_value(self, n: _Node, goal: bool, H: int) -> float:
        c = self.cfg
        r = max(0, H - n.t) // c.seg  # an early goal keeps its progress for the rest of the plan
        return n.gain + (c.goal_bonus if goal else 0.0) + c.earliness * (n.gsum + r * n.gain) / (H // c.seg) - n.pen

    def _simulate(self, env: SMWEnv, parent: _Node, a: int, root_x: int, start_y: int, H: int):
        """Plays ``a`` for one segment from the parent; returns (node, status) with status 'open', 'dead',
        'goal' or 'leaf' (reached the horizon alive)."""
        c = self.cfg
        env.set_sim_state(parent.state)
        pen = parent.pen
        for _ in range(c.seg):
            env.emulate(a)
            _, _, terminated, _, info = env._after_step(compute_obs=False)
            self.rollout_steps += 1
            self._steps += 1
            if info['hurt']:
                pen += c.hurt_penalty
            if terminated:
                gain = float(env.track.max_x - root_x)
                return _Node(None, parent.t + c.seg, gain, parent.gsum + gain, pen, parent=parent, a=a), \
                    ('dead' if info['died'] else 'goal')
        gain = float(env.track.max_x - root_x)
        t = parent.t + c.seg
        w = env._wram()
        m = env.mario(w)
        node = _Node(env.get_sim_state(), t, gain, parent.gsum + gain, pen, float(m['x'] - root_x), m['vx'] / 8.0,
                     int(w[0x13E4]), parent, a)
        if t >= H:
            return node, ('dead' if self._pit._doomed(env, start_y) else 'leaf')
        return node, 'open'

    def _key(self, env: SMWEnv, b: int, t: int, bfs: bool) -> tuple:
        m = env.mario()
        return (b, -1 if bfs else t // self.cfg.time_bucket, m['x'] >> 3, m['y'] >> 3, m['vx'] // 6, m['vy'] // 12,
                m['on_ground'], m['power'])

    def _search(self, env: SMWEnv, root: _Node, root_x: int, start_y: int, H: int, budget: int, bfs: bool = False):
        c = self.cfg
        self._steps = 0
        best_ok = {b: -np.inf for b in self.first}  # best plan that survives
        best_node: dict[int, _Node] = {}
        best_dead = {b: -np.inf for b in self.first}
        deep: dict[int, list] = {b: [] for b in self.first}  # phase 1: deepest first
        heaps: dict[int, list] = {b: [] for b in self.first}  # phase 2: best bound first
        seen: dict[tuple, float] = {}
        counter = 0

        wide: list = []  # stall mode: breadth first over all buttons
        best_open: dict[int, tuple[float, _Node]] = {}  # stall mode: furthest surviving node (horizon not reached)

        def push(b: int, node: _Node):
            nonlocal counter
            if bfs:
                for a in self.children:
                    heapq.heappush(wide, (node.t, self.rank[b], self.rank[a], counter, _Entry(node, a), b))
                    counter += 1
                return
            f = self._bound(node, H)
            for a in self.children:  # lazy: children are emulated when popped
                e = _Entry(node, a)
                heapq.heappush(deep[b], (-node.t, -f, self.rank[a], counter, e))
                heapq.heappush(heaps[b], (-f, -node.t, self.rank[a], counter, e))
                counter += 1

        def clean(h: list) -> list:
            while h and h[0][-1].done:
                heapq.heappop(h)
            return h

        def settle(b: int, node: _Node, status: str) -> None:
            if status == 'open':
                k = self._key(env, b, node.t, bfs)
                if seen.get(k, -np.inf) >= node.gain:
                    return
                seen[k] = node.gain
                if bfs and node.gain > best_open.get(b, (-np.inf, None))[0] and not self._pit._doomed(env, start_y):
                    best_open[b] = (node.gain, node)
                push(b, node)
            elif status == 'dead':
                best_dead[b] = max(best_dead[b], node.gain - c.death_penalty)
            else:
                v = self._leaf_value(node, status == 'goal', H)
                if v > best_ok[b]:
                    best_ok[b], best_node[b] = v, node

        def pop(h: list, b: int) -> None:
            e = heapq.heappop(h)[-1]
            e.done = True
            child, status = self._simulate(env, e.node, e.a, root_x, start_y, H)
            settle(b, child, status)

        for b in self.first:
            child, status = self._simulate(env, root, b, root_x, start_y, H)
            settle(b, child, status)
        if bfs:  # stall mode: breadth first; stop a while after the first way past the stall x
            found_at = None
            while wide and self._steps < budget and (found_at is None or self._steps < 1.5 * found_at):
                *_, e, b = heapq.heappop(wide)
                child, status = self._simulate(env, e.node, e.a, root_x, start_y, H)
                settle(b, child, status)
                if found_at is None and max(max(best_ok.values()), max((g for g, _ in best_open.values()), default=0)) >= c.stall_px:
                    found_at = self._steps
            out = np.full(len(ACTION_NAMES), -np.inf, np.float64)
            paths = {}
            for b in self.first:
                if best_ok[b] > -np.inf:
                    out[b], paths[b] = best_ok[b], best_node[b].actions()
                elif b in best_open:
                    out[b], paths[b] = self._leaf_value(best_open[b][1], False, H), best_open[b][1].actions()
                else:
                    out[b] = best_dead[b] if best_dead[b] > -np.inf else -c.death_penalty
            return out, paths, {'steps': self._steps, 'verdict_steps': self._steps, 'proved': not wide, 'horizon': H}
        for b in self.first:  # phase 1: a verdict for every button
            s0 = self._steps
            while clean(deep[b]) and best_ok[b] == -np.inf and self._steps - s0 < c.verdict_steps:
                pop(deep[b], b)
        phase1 = self._steps
        while self._steps < budget:  # phase 2: values of the buttons that matter, best bound first
            top = max(best_ok.values())
            cand = None
            for b in self.first:
                if not clean(heaps[b]):
                    continue
                f = -heaps[b][0][0]
                if f > best_ok[b] + c.eps and f > top - c.q_clip and (cand is None or f > cand[0]):
                    cand = (f, b)
            if cand is None:
                break
            pop(heaps[cand[1]], cand[1])
        out = np.full(len(ACTION_NAMES), -np.inf, np.float64)
        for b in self.first:
            out[b] = best_ok[b] if best_ok[b] > -np.inf else (best_dead[b] if best_dead[b] > -np.inf else -c.death_penalty)
        paths = {b: n.actions() for b, n in best_node.items()}
        stats = {'steps': self._steps, 'verdict_steps': phase1, 'proved': self._steps < budget, 'horizon': H}
        return out, paths, stats

    def _replay_plan(self, env: SMWEnv, root: _Node, root_x: int, start_y: int, H: int) -> float | None:
        """Value of the remembered plan from this state (``None`` if it dies or ends in a pit)."""
        node = root
        for a in self._plan:
            node, status = self._simulate(env, node, a, root_x, start_y, H)
            if status == 'dead':
                return None
            if status in ('goal', 'leaf'):
                return self._leaf_value(node, status == 'goal', H)
        if node.t <= 0:
            return None
        node = _Node(node.state, H, node.gain, node.gsum + node.gain * ((H - node.t) // self.cfg.seg), node.pen)
        env.set_sim_state(node.state)  # a plan shorter than the horizon: judge it where it ends
        return None if self._pit._doomed(env, start_y) else self._leaf_value(node, False, H)

    def scores(self, env: SMWEnv) -> np.ndarray:
        c = self.cfg
        rendering = env.rendering
        env.set_render(False)
        orig = env.get_sim_state()
        m = env.mario()
        root_x, start_y = m['x'], m['y']
        vx, pm = m['vx'] / 8.0, int(env._wram()[0x13E4])
        if self._stall_x is not None:
            self._stall_n += 1
            if root_x >= self._stall_x + c.stall_px or self._stall_n > c.stall_decisions:
                self._stall_x = None
        steps, ref = 0, root_x
        if self._stall_x is None:
            env.track.max_x = root_x  # progress from Mario's x now: the scores depend on the game state only
            root = _Node(env.get_sim_state(), 0, 0.0, 0.0, 0.0, 0.0, vx, pm)
            out, paths, stats = self._search(env, root, root_x, start_y, c.horizon, c.budget_steps)
            steps = stats['steps']
            if out.max() > -c.death_penalty / 2 and out.max() < c.stall_px:
                self._stall_x, self._stall_n, self._stall_fails = root_x, 0, 0
        if self._stall_x is not None:  # progress beyond the stall x, breadth first, longer horizon
            ref = self._stall_x
            env.set_sim_state(orig)
            env.track.max_x = max(root_x, ref)
            root = _Node(env.get_sim_state(), 0, float(max(root_x, ref) - ref), 0.0, 0.0, float(root_x - ref), vx, pm)
            budget = c.long_budget if self._stall_fails < 2 else c.stall_retry_budget
            out, paths, stats = self._search(env, root, ref, start_y, c.long_horizon, budget, bfs=True)
            steps += stats['steps']
            self.stall_decisions += 1
            self._stall_fails = 0 if out.max() >= c.stall_px else self._stall_fails + 1
        if self._plan and out[self._plan[0]] <= -c.death_penalty / 2:
            v = self._replay_plan(env, root, ref, start_y, stats['horizon'])
            if v is not None:
                out[self._plan[0]] = v
                paths[self._plan[0]] = list(self._plan)
                self.rescued += 1
        self._paths = paths
        self.last_stats = {**stats, 'steps': steps}
        env.set_sim_state(orig)
        env.set_render(rendering)
        return out

    def choose(self, scores: np.ndarray) -> int:
        best = scores.max()
        tol = max(1e-6, self.cfg.drive_tolerance)
        cands = [a for a in range(len(scores)) if scores[a] >= best - tol]
        if self.cfg.drive_tolerance > 0 and len(cands) > 1:
            a = int(cands[self._rng.integers(len(cands))])
        else:
            a = min(cands, key=lambda a_: self.rank.get(a_, 99))
        self._plan = self._paths.get(a, [a])[1:]
        return a

    def act(self, env: SMWEnv) -> int:
        return self.choose(self.scores(env))


def astar_config(name: str) -> AStarConfig:
    """'astar': deterministic choice; 'astar-div': random choice among buttons within 4 px of the best."""
    if name == 'astar':
        return AStarConfig()
    if name == 'astar-div':
        return AStarConfig(drive_tolerance=4.0)
    raise ValueError(name)
