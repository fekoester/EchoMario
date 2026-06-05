import numpy as np

from echomario.envs.toy_platformer_env import Enemy, ToyPlatformerEnv


def test_toy_env_runs():
    env = ToyPlatformerEnv()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert len(env.get_input_feature_names()) == env.observation_space.shape[0]

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        if terminated or truncated:
            break


def test_random_toy_env_generates_richer_structures():
    env = ToyPlatformerEnv(randomize=True, seed=123)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert len(env.platforms) > 0
    assert len(env.gaps) > 0
    assert len(env.coins) > 0
    assert len(env.question_blocks) > 0
    assert len(env.enemies) > 0


def test_full_screen_observation_shape_matches_camera_tiles_and_state():
    env = ToyPlatformerEnv(randomize=False, seed=123, observation_mode='full_screen', include_state_features=True)
    obs, _ = env.reset()

    expected = env.camera_width * env.height * len(env.SCREEN_CHANNEL_NAMES) + len(env.STATE_FEATURE_NAMES)
    assert obs.shape == (expected,)
    assert len(env.get_input_feature_names()) == expected
    assert env._static_screen_grid.shape == (env.height, env.width, len(env.SCREEN_CHANNEL_NAMES))


def test_engineered_observation_mode_still_available():
    env = ToyPlatformerEnv(randomize=False, seed=123, observation_mode='engineered')
    obs, _ = env.reset()

    assert obs.shape == (len(env.ENGINEERED_FEATURE_NAMES),)
    assert len(env.get_input_feature_names()) == len(env.ENGINEERED_FEATURE_NAMES)


def test_level_seed_reset_is_reproducible():
    env = ToyPlatformerEnv(randomize=True, seed=123)
    env.reset(options={'level_seed': 999})
    first_gaps = list(env.gaps)
    first_platforms = [(p.x0, p.x1, p.y) for p in env.platforms]
    first_blocks = [(b.x, b.y, b.item_kind) for b in env.question_blocks]

    env.reset(options={'level_seed': 999})
    assert first_gaps == list(env.gaps)
    assert first_platforms == [(p.x0, p.x1, p.y) for p in env.platforms]
    assert first_blocks == [(b.x, b.y, b.item_kind) for b in env.question_blocks]


def test_random_seed_reset_changes_layout():
    env = ToyPlatformerEnv(randomize=True, seed=123)
    env.reset(options={'level_seed': 100})
    first_layout = (list(env.gaps), [(p.x0, p.x1, p.y) for p in env.platforms])
    env.reset(options={'level_seed': 101})
    second_layout = (list(env.gaps), [(p.x0, p.x1, p.y) for p in env.platforms])
    assert first_layout != second_layout


def test_continuous_action_manual_right_jump():
    env = ToyPlatformerEnv(randomize=False, seed=123)
    obs, _ = env.reset()

    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert 'move_axis' in info
    assert 'jump_pressed' in info
    assert 'run_pressed' in info


def test_ground_horizontal_motion_is_not_blocked_by_floor():
    env = ToyPlatformerEnv(randomize=False, seed=123)
    env.reset()

    start_x = env.player.x
    _obs, _reward, _terminated, _truncated, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))

    assert env.player.on_ground is True
    assert env.player.x > start_x
    assert info['x'] > start_x


def test_stomp_enemy_defeats_it_and_awards_score():
    env = ToyPlatformerEnv(randomize=False, seed=123)
    env.reset()

    env.enemies = [Enemy(x=6.0, y=1.0, vx=0.0, left=5.0, right=7.0)]
    env.player.x = 6.0
    env.player.y = 1.95
    env.player.vy = -0.25
    env.player.on_ground = False

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))

    assert info['stomped_enemy'] is True
    assert info['hit_enemy'] is False
    assert len(env.enemies) == 0
    assert env.player.score >= 100


def test_shaped_reward_reports_components_and_checkpoints():
    env = ToyPlatformerEnv(
        randomize=False,
        seed=123,
        progress_reward_scale=0.0,
        step_penalty=0.0,
        checkpoint_reward=3.0,
        checkpoint_interval=3,
    )
    env.reset()
    env.enemies = []
    env.player.x = 2.95
    env.max_x = 2.95
    env.next_checkpoint_x = 3.0

    _obs, reward, _terminated, _truncated, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))

    assert info['checkpoint_count'] == 1
    assert info['reward_components']['checkpoint'] == 3.0
    assert reward >= 3.0


def test_gap_clear_shaped_reward_fires_once():
    env = ToyPlatformerEnv(
        randomize=False,
        seed=123,
        progress_reward_scale=0.0,
        step_penalty=0.0,
        gap_clear_reward=4.0,
    )
    env.reset()
    env.gaps = [(3, 4)]
    env.enemies = []
    env.player.x = 4.1
    env.player.y = 1.0
    env.max_x = 4.1

    env.player.vx = 1.0
    _obs, reward, _terminated, _truncated, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))
    assert info['gap_clear_count'] == 1
    assert info['reward_components']['gap_clear'] == 4.0
    assert reward >= 4.0

    _obs, _reward, _terminated, _truncated, info = env.step(np.array([1.0, -1.0, 1.0], dtype=np.float32))
    assert info['gap_clear_count'] == 0


def test_stagnation_termination():
    env = ToyPlatformerEnv(randomize=False, seed=123, stagnation_timeout=5, stagnation_epsilon=0.5)
    env.reset()

    done = False
    info = {}
    for _ in range(10):
        _obs, _reward, terminated, truncated, info = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))
        done = bool(terminated or truncated)
        if done:
            break

    assert done
    assert info.get('stagnated', False) is True
