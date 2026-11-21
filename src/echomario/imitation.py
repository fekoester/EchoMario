"""
Readout training by imitation: the classic reservoir-computing recipe applied to control.

A look-ahead planner (``echomario.experts.planner``) labels every visited state with a good
discrete action. The frozen reservoir turns the observation stream into features, and the
readout is fitted in closed form by ridge regression onto the one-hot labels. DAgger rounds
then let the learner drive (so the data covers the states the learner actually reaches) while
the planner keeps labelling.

This separates two questions that PPO mixes together:
1. Can a linear readout of the reservoir *represent* a policy that plays the game?
2. Can PPO *find* that readout from reward alone?
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from echomario.envs.make_env import make_env
from echomario.experts.planner import LookaheadPlanner, PlannerConfig, unwrap_toy
from echomario.reservoirs.esn import IdentityReservoir, make_reservoir

FEATURE_MODES = ('reservoir', 'input', 'reservoir+input')


def discrete_config(config: dict) -> dict:
    """Copy of ``config`` whose environment uses the 8 discrete actions."""
    cfg = {key: (dict(value) if isinstance(value, dict) else value) for key, value in config.items()}
    cfg['env']['discrete_actions'] = True
    cfg['env']['action_mode'] = 'discrete_basic'
    return cfg


class FeatureMap:
    """Turns the observation stream into readout features for one environment."""

    def __init__(self, config: dict, obs_dim: int, mode: str):
        if mode not in FEATURE_MODES:
            raise ValueError(f'mode must be one of {FEATURE_MODES}')
        self.mode = mode
        self.reservoir = None
        if mode != 'input':
            self.reservoir = make_reservoir(config, input_dim=obs_dim)
            if isinstance(self.reservoir, IdentityReservoir):
                raise ValueError(f'feature mode {mode!r} needs reservoir.type: sparse_esn')
        res_dim = 0 if self.reservoir is None else int(self.reservoir.cfg.size)
        in_dim = obs_dim if mode != 'reservoir' else 0
        self.dim = res_dim + in_dim + 1  # +1 bias

    def reset(self) -> None:
        if self.reservoir is not None:
            self.reservoir.reset(batch_size=1)

    def step(self, obs: np.ndarray) -> np.ndarray:
        parts = []
        if self.reservoir is not None:
            parts.append(self.reservoir.step_batch(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0])
        if self.mode != 'reservoir':
            parts.append(np.asarray(obs, dtype=np.float32))
        parts.append(np.ones(1, dtype=np.float32))
        return np.concatenate(parts)


@dataclass
class EpisodeStats:
    level_seed: int
    reached_goal: bool
    progress: float  # max_x / goal_x
    length: int
    death: str  # 'fell', 'enemy', 'stagnated', 'timeout' or ''
    agreement: float = float('nan')  # fraction of steps where learner == expert (labelled episodes)


def _death_kind(info: dict, truncated: bool) -> str:
    if info.get('fell'):
        return 'fell'
    if info.get('hit_enemy'):
        return 'enemy'
    if info.get('stagnated'):
        return 'stagnated'
    if truncated and not info.get('reached_goal'):
        return 'timeout'
    return ''


def run_episode(
    env,
    features: FeatureMap,
    level_seed: int,
    *,
    weights: np.ndarray | None = None,
    planner: LookaheadPlanner | None = None,
    beta: float = 0.0,
    rng: np.random.Generator | None = None,
    collect: bool = False,
    plan_every: int = 2,
    target: str = 'onehot',
    q_clip: float = 5.0,
    student_hold: int = 1,
):
    """
    Plays one level.

    With a ``planner`` the expert re-plans every ``plan_every`` frames; each plan labels the current
    state (when ``collect``) and decides who controls the next ``plan_every`` frames: the expert
    (probability ``beta``, or always when there are no weights yet) holding its action, or the
    learner acting greedily (re-deciding every ``student_hold`` frames). Without a planner the
    learner plays alone. The reservoir sees every frame either way.
    """
    obs, _ = env.reset(options={'level_seed': int(level_seed)})
    features.reset()
    if planner is not None:
        planner.attach(env)
    rng = rng or np.random.default_rng(level_seed)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    targets: list[np.ndarray] = []
    agree = 0
    planned = 0
    steps = 0
    info: dict = {}
    truncated = False
    expert_in_control = False
    expert_action = None
    learner_action = None
    while True:
        f = features.step(obs)
        if weights is not None and steps % student_hold == 0:
            learner_action = int(np.argmax(f @ weights))
        if planner is not None and steps % plan_every == 0:
            scores = planner.scores(env)
            expert_action = planner.choose(scores)
            planned += 1
            if collect:
                feats.append(f)
                labels.append(expert_action)
                targets.append(make_targets(scores, expert_action, target, q_clip))
            if learner_action is not None:
                agree += int(expert_action == learner_action)
            expert_in_control = learner_action is None or rng.random() < beta
        action = expert_action if expert_in_control else learner_action
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    base = unwrap_toy(env)
    stats = EpisodeStats(
        level_seed=int(level_seed),
        reached_goal=bool(info.get('reached_goal', False)),
        progress=float(base.max_x / max(1e-6, base.goal_x)),
        length=steps,
        death=_death_kind(info, truncated),
        agreement=agree / planned if (weights is not None and planned) else float('nan'),
    )
    if collect:
        return stats, np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.int64), np.stack(targets)
    return stats


TARGET_MODES = ('onehot', 'q', 'graded')


def make_targets(scores: np.ndarray, expert_action: int, mode: str, q_clip: float = 5.0) -> np.ndarray:
    """
    Regression targets for one planned state.

    ``onehot``: 1 for the planner's choice. ``q``: how close each action's planned score is to the
    best one, 1 for the best and 0 for anything ``q_clip`` or more worse (e.g. every deadly action).
    The q targets keep the information that several actions are equally good, which one-hot labels
    throw away with an arbitrary tie-break. ``q_clip`` must be small (a few reward points): idling
    for a planning step costs only ~1 point of progress, and with a large clip that difference
    drowns in readout noise, so the student learns to stand still until the stall timeout.
    """
    if mode == 'onehot':
        y = np.zeros(len(scores), dtype=np.float32)
        y[expert_action] = 1.0
        return y
    if mode == 'q':
        rel = np.clip(np.asarray(scores, dtype=np.float64) - np.max(scores), -q_clip, 0.0)
        return (1.0 + rel / q_clip).astype(np.float32)
    if mode == 'graded':
        # Two scales in one target: losing progress costs up to 0.5 (saturating at q_clip points),
        # dying (>= 100 points worse, i.e. the planner's death penalty) costs the other 0.5.
        regret = np.max(scores) - np.asarray(scores, dtype=np.float64)
        y = 1.0 - 0.5 * np.minimum(regret, q_clip) / q_clip - 0.5 * (regret >= 100.0)
        return y.astype(np.float32)
    raise ValueError(f'target must be one of {TARGET_MODES}')


@dataclass
class RidgeAccumulator:
    dim: int
    num_actions: int
    gram: np.ndarray = field(init=False)
    cross: np.ndarray = field(init=False)
    count: int = 0
    label_counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.gram = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.cross = np.zeros((self.dim, self.num_actions), dtype=np.float64)
        self.label_counts = np.zeros(self.num_actions, dtype=np.int64)

    def add(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray | None = None,
        critical_weight: float = 0.0,
        tried: np.ndarray | None = None,
    ) -> None:
        """
        Adds samples. With ``critical_weight`` k > 0 each sample is weighted by
        1 + k * (max target - min target): frames where some action is fatal (a jump must happen
        now) count more than the many frames where every action is fine. ``tried`` (bool, same shape
        as targets) limits that spread to the actions the teacher actually evaluated; without it an
        action the teacher never tries (target 0 everywhere) makes every frame look critical.
        """
        f = feats.astype(np.float64)
        if targets is None:
            y = np.zeros((len(labels), self.num_actions), dtype=np.float64)
            y[np.arange(len(labels)), labels] = 1.0
        else:
            y = np.asarray(targets, dtype=np.float64)
        if critical_weight > 0.0 and targets is not None:
            if tried is not None:
                yt = np.where(tried, y, np.nan)
                spread = np.nan_to_num(np.nanmax(yt, axis=1) - np.nanmin(yt, axis=1))
            else:
                spread = y.max(axis=1) - y.min(axis=1)
            w = 1.0 + critical_weight * spread
            fw = f * w[:, None]
            self.gram += fw.T @ f
            self.cross += fw.T @ y
        else:
            self.gram += f.T @ f
            self.cross += f.T @ y
        self.count += len(labels)
        self.label_counts += np.bincount(labels, minlength=self.num_actions)

    def merge(self, other: RidgeAccumulator) -> None:
        self.gram += other.gram
        self.cross += other.cross
        self.count += other.count
        self.label_counts += other.label_counts

    def solve(self, alpha: float) -> np.ndarray:
        """Ridge solution; ``alpha`` is relative to the mean feature energy, the bias is not penalised."""
        n = max(1, self.count)
        gram = self.gram / n
        reg = alpha * float(np.trace(gram)) / self.dim
        penalty = np.full(self.dim, reg)
        penalty[-1] = 0.0
        return np.linalg.solve(gram + np.diag(penalty), self.cross / n).astype(np.float32)


def summarize(stats: list[EpisodeStats]) -> dict[str, float]:
    if not stats:
        return {}
    deaths: dict[str, int] = {}
    for s in stats:
        if s.death:
            deaths[s.death] = deaths.get(s.death, 0) + 1
    out = {
        'episodes': len(stats),
        'success_rate': float(np.mean([s.reached_goal for s in stats])),
        'progress_mean': float(np.mean([s.progress for s in stats])),
        'length_mean': float(np.mean([s.length for s in stats])),
    }
    agreements = [s.agreement for s in stats if not np.isnan(s.agreement)]
    if agreements:
        out['expert_agreement'] = float(np.mean(agreements))
    for kind, count in sorted(deaths.items()):
        out[f'death_{kind}'] = count / len(stats)
    return out


# ----- worker entry points (module level so multiprocessing can pickle them) -----

def collect_worker(args: tuple) -> tuple[RidgeAccumulator, list[EpisodeStats], float]:
    config, mode, seeds, weights, beta, planner_cfg, worker_seed, target, critical_weight, *rest = args
    q_clip = float(rest[0]) if rest else 5.0
    start = time.perf_counter()
    env = make_env(discrete_config(config))
    features = FeatureMap(config, int(env.observation_space.shape[0]), mode)
    planner = LookaheadPlanner(planner_cfg or PlannerConfig())
    acc = RidgeAccumulator(features.dim, int(env.action_space.n))
    rng = np.random.default_rng(worker_seed)
    stats = []
    for seed in seeds:
        ep, feats, labels, targets = run_episode(
            env, features, seed, weights=weights, planner=planner, beta=beta, rng=rng, collect=True, target=target,
            q_clip=q_clip,
        )
        acc.add(feats, labels, targets, critical_weight=critical_weight)
        stats.append(ep)
    env.close()
    return acc, stats, time.perf_counter() - start


def eval_worker(args: tuple) -> list[EpisodeStats]:
    config, mode, seeds, weights, *rest = args
    student_hold = int(rest[0]) if rest else 1
    env = make_env(discrete_config(config))
    features = FeatureMap(config, int(env.observation_space.shape[0]), mode)
    out = [run_episode(env, features, seed, weights=weights, student_hold=student_hold) for seed in seeds]
    env.close()
    return out


def expert_worker(args: tuple) -> list[EpisodeStats]:
    config, seeds, planner_cfg = args
    env = make_env(discrete_config(config))
    planner = LookaheadPlanner(planner_cfg or PlannerConfig())
    out = []
    for seed in seeds:
        env.reset(options={'level_seed': int(seed)})
        planner.attach(env)
        steps = 0
        while True:
            if steps % 2 == 0:
                action = planner.act(env)
            _obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        base = unwrap_toy(env)
        out.append(EpisodeStats(int(seed), bool(info.get('reached_goal', False)),
                                float(base.max_x / max(1e-6, base.goal_x)), steps, _death_kind(info, truncated)))
    env.close()
    return out
