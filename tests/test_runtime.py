from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from echomario.agents.factory import build_policy
from echomario.agents.readout_policy import ReadoutActorCritic
from echomario.envs.make_env import make_env
from echomario.envs.mario_env import MarioInfoWrapper, MarioObservationWrapper
from echomario.envs.reset import EpisodeSeedManager
from echomario.reservoirs.esn import load_reservoir_from_state_dict, make_reservoir
from echomario.training.evaluate import evaluate_policy
from echomario.training.rollout import collect_rollout, collect_rollout_parallel
from echomario.utils.checkpoint import load_checkpoint, save_checkpoint


class DummyMarioEnv(gym.Env):
    observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        obs = np.zeros((240, 256, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        obs = np.full((240, 256, 3), self.step_count, dtype=np.uint8)
        info = {'flag_get': self.step_count > 1, 'x_pos': 12 + self.step_count}
        return obs, 1.0, self.step_count > 1, False, info


def test_rollout_carries_bootstrap_state():
    config = {
        'project': {'seed': 42},
        'env': {'name': 'toy_platformer'},
        'reservoir': {'type': 'none'},
    }
    env = make_env(config)
    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    policy = ReadoutActorCritic(reservoir_dim=reservoir.cfg.size, num_actions=3, continuous=True)

    def reset_fn():
        obs, _ = env.reset()
        reservoir.reset()
        return obs

    obs = reset_fn()
    rollout = collect_rollout(
        env=env,
        reservoir=reservoir,
        policy=policy,
        obs=obs,
        rollout_steps=4,
        device=torch.device('cpu'),
        reset_fn=reset_fn,
    )
    assert rollout.last_state.shape == obs.shape


def test_parallel_rollout_batches_multiple_envs():
    config = {
        'project': {'seed': 42},
        'env': {'name': 'toy_platformer', 'observation_mode': 'full_screen', 'include_state_features': True},
        'reservoir': {'type': 'none'},
        'agent': {'model': 'cnn', 'hidden_dim': 64},
    }
    envs = [make_env(config) for _ in range(3)]
    reservoir = make_reservoir(config, input_dim=int(envs[0].observation_space.shape[0]))
    reservoir.reset(batch_size=len(envs))
    policy = build_policy(config, envs[0], reservoir.cfg.size)

    def reset_fn(env_idx: int):
        obs, _ = envs[env_idx].reset()
        return obs

    obs_batch = np.stack([reset_fn(i) for i in range(len(envs))], axis=0)
    rollout = collect_rollout_parallel(
        envs=envs,
        reservoir=reservoir,
        policy=policy,
        obs_batch=obs_batch,
        rollout_steps=4,
        device=torch.device('cpu'),
        reset_fn=reset_fn,
    )
    assert rollout.states.shape[0] == 12
    assert rollout.last_state.shape[0] == 3
    assert rollout.last_obs.shape[0] == 3


def test_rollout_runs_with_cnn_policy():
    config = {
        'project': {'seed': 42},
        'env': {'name': 'toy_platformer', 'observation_mode': 'full_screen', 'include_state_features': True},
        'reservoir': {'type': 'none'},
        'agent': {'model': 'cnn', 'hidden_dim': 64},
    }
    env = make_env(config)
    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    policy = build_policy(config, env, reservoir.cfg.size)

    def reset_fn():
        obs, _ = env.reset()
        reservoir.reset()
        return obs

    obs = reset_fn()
    rollout = collect_rollout(
        env=env,
        reservoir=reservoir,
        policy=policy,
        obs=obs,
        rollout_steps=4,
        device=torch.device('cpu'),
        reset_fn=reset_fn,
    )
    assert rollout.states.shape[1] == env.observation_space.shape[0]


def test_checkpoint_roundtrip_for_linear_baseline(tmp_path: Path):
    config = {
        'project': {'seed': 42, 'device': 'cpu'},
        'env': {'name': 'toy_platformer', 'randomize': True, 'train_seed_mode': 'random', 'eval_seed_start': 10, 'num_eval_seeds': 3},
        'reservoir': {'type': 'none'},
        'agent': {'lr': 1e-4},
        'training': {},
        'logging': {},
    }
    env = make_env(config)
    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    policy = ReadoutActorCritic(reservoir_dim=reservoir.cfg.size, num_actions=3, continuous=True)

    ckpt_path = tmp_path / 'baseline.pt'
    save_checkpoint(ckpt_path, config=config, policy_state=policy.state_dict(), reservoir_state=reservoir.state_dict(), global_step=123, stats={})
    ckpt = load_checkpoint(ckpt_path)
    restored = load_reservoir_from_state_dict(ckpt['reservoir_state'])
    assert restored.cfg.size == int(env.observation_space.shape[0])

    seed_manager = EpisodeSeedManager(config)
    stats = evaluate_policy(
        env=env,
        reservoir=restored,
        policy=policy,
        episodes=3,
        reset_specs=seed_manager.evaluation_reset_specs(),
        device=torch.device('cpu'),
    )
    assert 'eval_success_rate' in stats


def test_mario_wrappers_transform_observation_and_info():
    env = MarioObservationWrapper(MarioInfoWrapper(DummyMarioEnv()), frame_width=16, frame_height=16, frame_stack=2)
    obs, _ = env.reset()
    assert obs.shape == (16 * 16 * 2,)

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (16 * 16 * 2,)
    assert 'reached_goal' in info
    assert 'max_x' in info
