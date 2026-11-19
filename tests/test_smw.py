"""Super Mario World env and teacher. Skipped unless stable-retro is installed and a ROM was imported."""

import numpy as np
import pytest

retro = pytest.importorskip('stable_retro')


def _rom_available() -> bool:
    try:
        return retro.data.get_romfile_path('SuperMarioWorld-Snes-v0', retro.data.Integrations.ALL) is not None
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _rom_available(), reason='Super Mario World ROM not imported')


def test_tile_classes():
    from echomario.envs.smw_env import tile_class

    t = np.array([0x25, 0x100, 0x110, 0x111, 0x130, 0x2B, 0x3F])
    assert tile_class(t).tolist() == [-1, 1, 1, 0, 0, 2, -1]


def test_env_obs_grid_and_sim_state_round_trip():
    from echomario.envs.smw_env import OBS_DIM, SMWEnv

    env = SMWEnv('YoshiIsland2')
    try:
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,) and info['x'] > 0
        assert env.grid().sum() > 0  # ground is visible
        for _ in range(40):
            env.step(2)
        snap = env.get_sim_state()
        xs = []
        for _ in range(2):
            env.set_sim_state(snap)
            for _ in range(30):
                env.step(5)
            xs.append(env.mario()['x'])
        assert xs[0] == xs[1] and xs[0] > snap[1].max_x  # deterministic replay, moved right
    finally:
        env.close()


def test_planner_prefers_moving_right_at_start():
    from echomario.envs.smw_env import ACTION_NAMES, SMWEnv
    from echomario.experts.smw_planner import SMWPlanner

    env = SMWEnv('YoshiIsland2')
    try:
        env.reset()
        x0 = env.mario()['x']
        a = SMWPlanner().act(env)
        assert ACTION_NAMES[a].startswith('right')
        assert env.mario()['x'] == x0  # planning leaves the live game untouched
    finally:
        env.close()
