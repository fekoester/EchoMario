"""Super Mario World env and teacher. Skipped unless stable-retro is installed and a ROM was imported."""

import numpy as np
import pytest

import echomario.envs.smw_env  # noqa: F401  (before stable_retro: it picks the emulator core on import)

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
    from echomario.envs.smw_env import OBS_DIM, SMWEnv, grid_kwargs, smw_obs_dim

    env = SMWEnv('YoshiIsland2')
    try:
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,) and info['x'] > 0
        assert smw_obs_dim({'env': {'grid': grid_kwargs({})}}) == OBS_DIM
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


def test_prefix_plan_shares_common_starts():
    from echomario.experts.smw_planner import _prefix_plan

    assert _prefix_plan([[1, 1, 2], [1, 1, 3], [4], [1, 2]]) == [(-1, 0), (0, 2), (-1, 0), (0, 1)]


def test_render_switch_keeps_the_game_identical():
    """With the patched core (scripts/build_snes_core.sh) not drawing must not change the game."""
    from echomario.envs.smw_env import SMWEnv

    env = SMWEnv('DonutPlains1')
    try:
        if not env.fast_core:
            pytest.skip('stock snes9x core: no render switch')
        runs = []
        for render in (True, False):
            env.reset()
            env.set_render(render)
            ram = []
            for i in range(150):
                env.step(5 if (i // 10) % 3 else 2)
                ram.append(env._wram().copy())
            runs.append(np.array(ram))
        env.set_render(True)
        assert np.array_equal(runs[0], runs[1])
    finally:
        env.close()


def test_teacher_v2_keeps_v1_scores_when_safe_and_leaves_the_game_alone():
    from echomario.envs.smw_env import SMWEnv
    from echomario.experts.smw_planner import SMWPlanner, teacher_config

    env = SMWEnv('YoshiIsland2')
    try:
        env.reset()
        x0 = env.mario()['x']
        v1, v2 = SMWPlanner(teacher_config('v1')), SMWPlanner(teacher_config('v2'))
        s1, s2 = v1.scores(env), v2.scores(env)
        assert np.array_equal(s1, s2)  # level start: nothing to veto, the 1-second scores stand
        assert v2.rollout_steps > v1.rollout_steps  # but the 3-second check did run
        assert env.mario()['x'] == x0
    finally:
        env.close()


def test_replay_log_reproduces_observations_with_rewinds():
    from echomario.envs.smw_env import SMWEnv, replay_observations

    kw = {'rows': 22, 'cols': 24, 'rows_above': 12, 'cols_behind': 6, 'obs': 'v2'}
    env = SMWEnv('DonutPlains1', max_steps=10 ** 6, **kw)
    try:
        obs, _ = env.reset()
        for _ in range(5):  # start delay
            obs, *_ = env.step(0)
        steps, actions, seen, saved = [], [], [], None
        plan = [(k, 5 if (k // 10) % 3 else 2) for k in range(60)] + [(k, 2) for k in range(20, 50)]
        for p, (k, a) in enumerate(plan):
            if p and k != plan[p - 1][0] + 1:
                env.set_sim_state(saved[0])
                obs = saved[1]
            elif k == 20 and saved is None:
                saved = (env.get_sim_state(), obs.copy())
            steps.append(k)
            actions.append(a)
            seen.append(obs.copy())
            obs, *_ = env.step(a)
    finally:
        env.close()
    log = {'level': 'DonutPlains1', 'noop': 5, 'steps': np.array(steps), 'actions': np.array(actions)}
    assert np.array_equal(replay_observations(log, **kw), np.array(seen))
