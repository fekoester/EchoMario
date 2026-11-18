import numpy as np

from echomario.envs.toy_platformer_env import ToyPlatformerEnv
from echomario.experts.planner import LookaheadPlanner
from echomario.imitation import FeatureMap, RidgeAccumulator, run_episode
from echomario.reservoirs.esn import ESNConfig, SparseESN

EASY = dict(randomize=True, width=60, camera_width=20, num_gaps=0, num_enemies=0, max_steps=400)


def _play(env, actions):
    out = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((None if obs is None else obs.copy(), r, term, trunc, info['x'], info['y']))
        if term or trunc:
            break
    return out


def test_sim_state_roundtrip_reproduces_trajectory():
    env = ToyPlatformerEnv(randomize=True, max_steps=300)
    env.reset(options={'level_seed': 7})
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step(rng.uniform(-1, 1, 3).astype(np.float32))
    snap = env.get_sim_state()
    actions = [rng.uniform(-1, 1, 3).astype(np.float32) for _ in range(120)]
    first = _play(env, actions)
    env.set_sim_state(snap)
    second = _play(env, actions)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a[1:] == b[1:]
        np.testing.assert_array_equal(a[0], b[0])


def test_compute_obs_flag_skips_observation_only():
    env = ToyPlatformerEnv(randomize=True)
    env.reset(options={'level_seed': 3})
    env.compute_obs = False
    obs, reward, *_ = env.step(np.array([1, -1, 1], dtype=np.float32))
    assert obs is None and np.isfinite(reward)


def test_planner_does_not_touch_live_env_and_clears_easy_level():
    env = ToyPlatformerEnv(**EASY)
    env.reset(options={'level_seed': 11})
    planner = LookaheadPlanner()
    planner.attach(env)
    before = env.get_sim_state()
    planner.act(env)
    after = env.get_sim_state()
    assert before[0] == after[0] and before[1] == after[1]

    from echomario.envs.make_env import DiscreteToyActionWrapper

    denv = DiscreteToyActionWrapper(ToyPlatformerEnv(**EASY))
    denv.reset(options={'level_seed': 11})
    planner.attach(denv)
    while True:
        _o, _r, term, trunc, info = denv.step(planner.act(denv))
        if term or trunc:
            break
    assert info['reached_goal']


def test_multi_timescale_leaks_keep_weights_identical():
    base = SparseESN(ESNConfig(input_dim=5, size=50, seed=1))
    multi = SparseESN(ESNConfig(input_dim=5, size=50, seed=1, leak_rate_min=0.05, leak_rate_max=0.9))
    np.testing.assert_array_equal(base.W_in, multi.W_in)
    np.testing.assert_array_equal(base.W_res, multi.W_res)
    assert np.ndim(base.leak) == 0
    assert multi.leak.shape == (50,) and 0.05 <= multi.leak.min() and multi.leak.max() <= 0.9
    x = multi.step(np.ones(5, dtype=np.float32))
    assert x.shape == (50,) and np.all(np.isfinite(x))


def test_ridge_recovers_linear_labels():
    rng = np.random.default_rng(0)
    feats = np.hstack([rng.normal(size=(500, 6)), np.ones((500, 1))]).astype(np.float32)
    true_w = rng.normal(size=(7, 3))
    labels = np.argmax(feats @ true_w, axis=1)
    acc = RidgeAccumulator(7, 3)
    acc.add(feats[:250], labels[:250])
    other = RidgeAccumulator(7, 3)
    other.add(feats[250:], labels[250:])
    acc.merge(other)
    w = acc.solve(1e-4)
    assert np.mean(np.argmax(feats @ w, axis=1) == labels) > 0.85


def test_run_episode_collects_features_and_labels():
    from echomario.envs.make_env import DiscreteToyActionWrapper

    config = {'reservoir': {'type': 'sparse_esn', 'size': 40, 'seed': 2}}
    env = DiscreteToyActionWrapper(ToyPlatformerEnv(**EASY))
    fmap = FeatureMap(config, int(env.observation_space.shape[0]), 'reservoir')
    stats, feats, labels, targets = run_episode(env, fmap, 5, planner=LookaheadPlanner(), collect=True, target="q")
    assert targets.shape == (feats.shape[0], 8) and np.allclose(targets.max(axis=1), 1.0)
    assert feats.shape == ((stats.length + 1) // 2, 41) and labels.shape == (feats.shape[0],)
    assert stats.reached_goal
