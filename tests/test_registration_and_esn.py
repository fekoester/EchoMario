import gymnasium as gym
import numpy as np
import pytest

import echomario  # noqa: F401  (registers the environments)
from echomario.envs.registration import ENVIRONMENTS
from echomario.reservoirs.esn import SparseESN, make_reservoir


@pytest.mark.parametrize("env_id", list(ENVIRONMENTS))
def test_registered_envs_pass_gymnasium_checker(env_id):
    from gymnasium.utils.env_checker import check_env

    env = gym.make(env_id)
    check_env(env.unwrapped)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_observations_are_not_aliased():
    env = gym.make("EchoMario/ToyPlatformerEasy-v0")
    first, _ = env.reset(seed=0)
    snapshot = first.copy()
    env.step(env.action_space.sample())
    env.reset(seed=1)
    np.testing.assert_array_equal(first, snapshot)


def _config(**overrides):
    reservoir = {"type": "sparse_esn", "size": 120, "sparsity": 0.05, "seed": 3, **overrides}
    return {"reservoir": reservoir}


@pytest.mark.parametrize("density", [0.05, 0.9])
def test_fast_reservoir_matches_dense_reference(density):
    esn = make_reservoir(_config(), input_dim=300)
    rng = np.random.default_rng(0)
    u = (rng.random((4, 300)) < density).astype(np.float32) * rng.normal(size=(4, 300)).astype(np.float32)
    esn.reset(batch_size=4)
    x = np.zeros((4, 120), np.float32)
    for _ in range(5):
        pre = x @ esn.W_res.T + u @ esn.W_in.T
        x = (1 - esn.cfg.leak_rate) * x + esn.cfg.leak_rate * np.tanh(pre)
        out = esn.step_batch(u)
    rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True)) + 1e-6
    np.testing.assert_allclose(out, x / rms, atol=1e-5)


def test_peek_does_not_change_state():
    esn = make_reservoir(_config(), input_dim=10)
    esn.reset(batch_size=3)
    u = np.ones((3, 10), np.float32)
    esn.step_batch(u)
    before = esn.x.copy()
    peeked = esn.peek_batch(u[[2]], [2])
    np.testing.assert_array_equal(esn.x, before)
    np.testing.assert_allclose(peeked[0], esn.step_batch(u)[2], atol=1e-6)


def test_state_dict_roundtrip_without_device_key():
    esn = make_reservoir(_config(), input_dim=10)
    state = esn.state_dict()
    state["cfg"].pop("device")  # checkpoints from before the device option
    restored = SparseESN.from_state_dict(state)
    u = np.ones((1, 10), np.float32)
    esn.reset()
    np.testing.assert_allclose(restored.step_batch(u), esn.step_batch(u), atol=1e-6)


def test_gru_policy_from_config_collects_rollout():
    import torch

    from echomario.agents.factory import build_policy
    from echomario.envs.make_env import make_env
    from echomario.training.rollout import collect_rollout

    config = {
        "project": {"seed": 1},
        "env": {"name": "toy_platformer", "width": 60, "camera_width": 20, "num_enemies": 0},
        "reservoir": {"type": "none"},
        "agent": {"model": "gru", "hidden_dim": 16, "encoder_dim": 16},
    }
    env = make_env(config)
    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    policy = build_policy(config, env, reservoir.cfg.size)
    obs, _ = env.reset(seed=0)
    rollout = collect_rollout(env=env, reservoir=reservoir, policy=policy, obs=obs, rollout_steps=8,
                              device=torch.device("cpu"), reset_fn=lambda: env.reset()[0], gamma=0.99)
    assert rollout.hiddens.shape == (8, 16)
