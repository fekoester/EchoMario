from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Player:
    x: float = 2.0
    y: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    jump_hold_frames: int = 0
    collected_coins: int = 0
    collected_items: int = 0
    score: int = 0


@dataclass
class Platform:
    x0: int
    x1: int
    y: int


@dataclass
class Enemy:
    x: float
    y: float
    vx: float
    left: float
    right: float


@dataclass
class Coin:
    x: float
    y: float
    collected: bool = False


@dataclass
class QuestionBlock:
    x: int
    y: int
    item_kind: str = 'coin'
    used: bool = False


@dataclass
class Item:
    x: float
    y: float
    kind: str = 'mushroom'
    vx: float = 0.0
    vy: float = 0.0
    collected: bool = False
    emerging_frames: int = 10


class ToyPlatformerEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    ENGINEERED_FEATURE_NAMES = [
        'x_norm',
        'y_norm',
        'vx',
        'vy',
        'on_ground',
        'falling',
        'coins_norm',
        'items_norm',
        'enemy_dx',
        'enemy_dy',
        'enemy_vx',
        'coin_dx',
        'coin_dy',
        'item_dx',
        'item_dy',
        'block_dx',
        'block_dy',
        'platform_dx',
        'platform_dy',
        'platform_height',
        'gap_start_dx',
        'gap_end_dx',
        'bridge_ahead',
        'goal_dx',
        'time',
        'solid_low+1',
        'solid_mid+1',
        'solid_high+1',
        'enemy+1',
        'coin+1',
        'solid_low+2',
        'solid_mid+2',
        'solid_high+2',
        'enemy+2',
        'coin+2',
        'solid_low+3',
        'solid_mid+3',
        'solid_high+3',
        'enemy+3',
        'coin+3',
        'solid_low+4',
        'solid_mid+4',
        'solid_high+4',
        'enemy+4',
        'coin+4',
        'solid_low+5',
        'solid_mid+5',
        'solid_high+5',
        'enemy+5',
        'coin+5',
        'solid_low+6',
        'solid_mid+6',
        'solid_high+6',
        'enemy+6',
        'coin+6',
        'solid_low+7',
        'solid_mid+7',
        'solid_high+7',
        'enemy+7',
        'coin+7',
        'solid_low+8',
        'solid_mid+8',
        'solid_high+8',
        'enemy+8',
        'coin+8',
    ]
    SCREEN_CHANNEL_NAMES = [
        'empty',
        'ground',
        'platform',
        'question_block',
        'used_block',
        'goal',
        'enemy',
        'coin',
        'item',
        'player',
    ]
    STATE_FEATURE_NAMES = [
        'player_screen_x',
        'player_y_norm',
        'vx',
        'vy',
        'on_ground',
        'score_norm',
        'goal_dx',
        'time',
        'gap_start_dx',
        'gap_end_dx',
        'bridge_ahead',
    ]

    def __init__(
        self,
        max_steps: int = 1400,
        render_scale: int = 24,
        seed: int | None = None,
        randomize: bool = False,
        width: int = 220,
        height: int = 18,
        camera_width: int = 36,
        num_gaps: int = 10,
        num_bridge_gaps: int = 4,
        num_enemies: int = 16,
        num_coins: int = 40,
        num_question_blocks: int = 8,
        num_upper_platforms: int = 14,
        num_mid_platforms: int = 12,
        min_gap_width: int = 2,
        max_gap_width: int = 4,
        min_bridge_gap_width: int = 6,
        max_bridge_gap_width: int = 10,
        min_platform_width: int = 3,
        max_platform_width: int = 8,
        min_spacing: int = 8,
        stagnation_timeout: int = 150,
        stagnation_epsilon: float = 0.05,
        stagnation_penalty: float = -5.0,
        level_seed: int | None = None,
        enable_gaps: bool = True,
        enable_bridge_platforms: bool = True,
        enable_upper_platforms: bool = True,
        enable_moving_enemies: bool = True,
        enable_coins: bool = True,
        enable_question_blocks: bool = True,
        enable_items: bool = True,
        enemy_speed: float = 0.04,
        observation_mode: str = 'full_screen',
        include_state_features: bool = True,
        progress_reward_scale: float = 1.25,
        step_penalty: float = -0.002,
        coin_reward: float = 0.4,
        item_reward: float = 1.5,
        stomp_reward: float = 2.0,
        goal_reward: float = 45.0,
        death_penalty: float = -12.0,
        safe_landing_reward: float = 0.0,
        gap_clear_reward: float = 0.0,
        gap_jump_reward: float = 0.0,
        gap_jump_lookahead: float = 5.0,
        gap_jump_hold_reward: float = 0.0,
        gap_jump_hold_lookahead: float = 5.0,
        enemy_pass_reward: float = 0.0,
        checkpoint_reward: float = 0.0,
        checkpoint_interval: int = 25,
        forced_gaps: list | None = None,
    ):
        super().__init__()

        self.width = int(width)
        self.height = int(height)
        self.camera_width = int(camera_width)
        self.max_steps = int(max_steps)
        self.render_scale = int(render_scale)

        self.randomize = bool(randomize)
        self.num_gaps = int(num_gaps)
        self.num_bridge_gaps = int(num_bridge_gaps)
        self.num_enemies = int(num_enemies)
        self.num_coins = int(num_coins)
        self.num_question_blocks = int(num_question_blocks)
        self.num_upper_platforms = int(num_upper_platforms)
        self.num_mid_platforms = int(num_mid_platforms)
        self.min_gap_width = int(min_gap_width)
        self.max_gap_width = int(max_gap_width)
        self.min_bridge_gap_width = int(min_bridge_gap_width)
        self.max_bridge_gap_width = int(max_bridge_gap_width)
        self.min_platform_width = int(min_platform_width)
        self.max_platform_width = int(max_platform_width)
        self.min_spacing = int(min_spacing)

        self.enable_gaps = bool(enable_gaps)
        self.enable_bridge_platforms = bool(enable_bridge_platforms)
        self.enable_upper_platforms = bool(enable_upper_platforms)
        self.enable_moving_enemies = bool(enable_moving_enemies)
        self.enable_coins = bool(enable_coins)
        self.enable_question_blocks = bool(enable_question_blocks)
        self.enable_items = bool(enable_items)
        self.enemy_speed = float(enemy_speed)
        self.observation_mode = str(observation_mode)
        self.include_state_features = bool(include_state_features)
        self.progress_reward_scale = float(progress_reward_scale)
        self.step_penalty = float(step_penalty)
        self.coin_reward = float(coin_reward)
        self.item_reward = float(item_reward)
        self.stomp_reward = float(stomp_reward)
        self.goal_reward = float(goal_reward)
        self.death_penalty = float(death_penalty)
        self.safe_landing_reward = float(safe_landing_reward)
        self.gap_clear_reward = float(gap_clear_reward)
        self.gap_jump_reward = float(gap_jump_reward)
        self.gap_jump_lookahead = float(gap_jump_lookahead)
        self.gap_jump_hold_reward = float(gap_jump_hold_reward)
        self.gap_jump_hold_lookahead = float(gap_jump_hold_lookahead)
        self.enemy_pass_reward = float(enemy_pass_reward)
        self.checkpoint_reward = float(checkpoint_reward)
        self.checkpoint_interval = int(checkpoint_interval)
        self.forced_gaps = self._parse_forced_gaps(forced_gaps)

        self.stagnation_timeout = int(stagnation_timeout)
        self.stagnation_epsilon = float(stagnation_epsilon)
        self.stagnation_penalty = float(stagnation_penalty)
        self.stagnation_counter = 0

        self.base_seed = seed
        self.level_seed = level_seed
        self.rng = np.random.default_rng(seed if level_seed is None else level_seed)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.num_screen_channels = len(self.SCREEN_CHANNEL_NAMES)
        self.screen_dim = self.height * self.camera_width * self.num_screen_channels
        self.state_dim = len(self.STATE_FEATURE_NAMES) if self.include_state_features else 0
        self.screen_channel_index = {name: idx for idx, name in enumerate(self.SCREEN_CHANNEL_NAMES)}
        self.input_feature_names = self._build_input_feature_names()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.input_feature_names),),
            dtype=np.float32,
        )

        self.player = Player()
        self.t = 0
        self.max_x = 0.0
        self.goal_x = self.width - 5
        self.solid_tiles: set[tuple[int, int]] = set()
        self.platforms: list[Platform] = []
        self.gaps: list[tuple[int, int]] = []
        self.bridge_platforms: list[Platform] = []
        self.enemies: list[Enemy] = []
        self.coins: list[Coin] = []
        self.question_blocks: list[QuestionBlock] = []
        self.items: list[Item] = []
        self._static_screen_grid = np.zeros((self.height, self.width, self.num_screen_channels), dtype=np.float32)
        self._static_screen_views = np.zeros((self._camera_left_count(), self.screen_dim), dtype=np.float32)
        self._screen_obs_buffer = np.zeros(self.screen_dim + self.state_dim, dtype=np.float32)
        self.cleared_gap_indices: set[int] = set()
        self.passed_enemy_ids: set[int] = set()
        self.next_checkpoint_x = float(self.checkpoint_interval)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and 'level_seed' in options:
            self.rng = np.random.default_rng(int(options['level_seed']))

        self.player = Player()
        self.t = 0
        self.max_x = self.player.x
        self.stagnation_counter = 0
        self.cleared_gap_indices = set()
        self.passed_enemy_ids = set()
        self.next_checkpoint_x = float(self.checkpoint_interval)
        self.goal_x = self.width - 5
        self._generate_level()
        self.player.y = self._surface_top_below(self.player.x, 4.0) or 1.0
        self.player.on_ground = True
        return self._obs(), {}

    def _generate_level(self) -> None:
        self.solid_tiles.clear()
        self.platforms = []
        self.gaps = []
        self.bridge_platforms = []
        self.enemies = []
        self.coins = []
        self.question_blocks = []
        self.items = []

        for x in range(self.width):
            self._add_solid_tile(x, 0)

        if not self.randomize:
            self._build_handcrafted_level()
        else:
            self._build_random_level()

        self._apply_forced_gaps()
        self._rebuild_static_screen_grid()

    def _parse_forced_gaps(self, forced_gaps: list | None) -> list[tuple[int, int]]:
        parsed: list[tuple[int, int]] = []
        for gap in forced_gaps or []:
            if len(gap) != 2:
                raise ValueError(f'forced_gaps entries must be [start, end], got {gap!r}')
            start, end = int(gap[0]), int(gap[1])
            if start > end:
                raise ValueError(f'forced_gaps start must be <= end, got {gap!r}')
            parsed.append((start, end))
        return parsed

    def _apply_forced_gaps(self) -> None:
        for start, end in self.forced_gaps:
            self._carve_gap(start, end)

    def _build_handcrafted_level(self) -> None:
        self._carve_gap(18, 20)
        self._carve_gap(39, 47)
        self._add_platform(37, 49, 2, bridge=True)
        self._carve_gap(71, 73)
        self._carve_gap(94, 103)
        self._add_platform(92, 105, 3, bridge=True)
        self._add_platform(54, 61, 3)
        self._add_platform(66, 74, 5)
        self._add_platform(116, 126, 3)
        self._add_platform(129, 137, 6)
        self._add_platform(145, 154, 3)
        self._add_platform(165, 174, 5)
        self._add_platform(186, 198, 3)
        self._place_coin_arc(11, 16, 3)
        self._place_coin_arc(38, 48, 5)
        self._place_coin_arc(93, 105, 6)
        self._place_coin_arc(145, 154, 5)
        self._add_question_block(58, 4, 'coin')
        self._add_question_block(119, 4, 'mushroom')
        self._spawn_enemy_on_surface(27, 0)
        self._spawn_enemy_on_surface(58, 3)
        self._spawn_enemy_on_surface(97, 3)
        self._spawn_enemy_on_surface(167, 5)
        self._sprinkle_default_coins()

    def _build_random_level(self) -> None:
        cursor = 12
        guarantee_bridge = False

        while cursor < self.width - 18:
            segment_roll = float(self.rng.random())
            plain_len = int(self.rng.integers(5, 12))
            cursor += plain_len
            if cursor >= self.width - 18:
                break

            if self.enable_gaps and segment_roll < 0.25 and len(self.gaps) < self.num_gaps:
                width = int(self.rng.integers(self.min_gap_width, self.max_gap_width + 1))
                self._carve_gap(cursor, cursor + width - 1)
                if self.enable_coins:
                    self._place_coin_arc(cursor - 1, cursor + width, 4)
                cursor += width + 3
                continue

            if self.enable_bridge_platforms and segment_roll < 0.45 and len(self.bridge_platforms) < self.num_bridge_gaps:
                width = int(self.rng.integers(self.min_bridge_gap_width, self.max_bridge_gap_width + 1))
                a = cursor
                b = min(self.width - 12, cursor + width - 1)
                self._carve_gap(a, b)
                bridge_y = int(self.rng.integers(2, 5))
                self._add_platform(max(4, a - 1), min(self.width - 8, b + 1), bridge_y, bridge=True)
                if self.enable_coins:
                    self._place_coin_arc(a - 1, b + 1, bridge_y + 2)
                if self.enable_moving_enemies and self.rng.random() < 0.5:
                    self._spawn_enemy_on_surface((a + b) // 2, bridge_y)
                guarantee_bridge = True
                cursor = b + 4
                continue

            if self.enable_upper_platforms and segment_roll < 0.75:
                count = int(self.rng.integers(2, 5))
                segment_start = cursor
                for idx in range(count):
                    width = int(self.rng.integers(self.min_platform_width, self.max_platform_width + 1))
                    x0 = min(self.width - 12, segment_start + idx * int(self.rng.integers(4, 7)))
                    y = int(self.rng.choice([2, 3, 5, 6]))
                    self._add_platform(x0, min(self.width - 6, x0 + width), y)
                    if self.enable_coins and self.rng.random() < 0.7:
                        self._place_coin_arc(x0, min(self.width - 6, x0 + width), y + 2)
                    if self.enable_question_blocks and self.rng.random() < 0.35:
                        block_x = min(self.width - 8, x0 + width // 2)
                        self._add_question_block(block_x, max(2, y + 1), 'mushroom' if self.rng.random() < 0.35 else 'coin')
                    if self.enable_moving_enemies and self.rng.random() < 0.35:
                        self._spawn_enemy_on_surface(x0 + width // 2, y)
                cursor += int(self.rng.integers(10, 18))
                continue

            if self.enable_question_blocks and self.rng.random() < 0.35:
                self._add_question_block(cursor + 1, int(self.rng.choice([2, 3, 4])), 'coin')
                if self.enable_items and self.rng.random() < 0.3:
                    self._add_question_block(cursor + 3, int(self.rng.choice([3, 4])), 'mushroom')

            if self.enable_moving_enemies and self.rng.random() < 0.5:
                self._spawn_enemy_on_surface(cursor + 2, 0)

            if self.enable_coins and self.rng.random() < 0.7:
                self._place_coin_arc(cursor, cursor + 4, int(self.rng.choice([3, 4])))

        if self.enable_bridge_platforms and not guarantee_bridge:
            a = self.width // 2
            b = min(self.width - 10, a + 7)
            self._carve_gap(a, b)
            self._add_platform(a - 1, b + 1, 3, bridge=True)
            self._place_coin_arc(a - 1, b + 1, 5)

        self._sprinkle_default_coins()
        self._pad_enemies()
        self._pad_question_blocks()

    def _pad_enemies(self) -> None:
        attempts = 0
        while self.enable_moving_enemies and len(self.enemies) < self.num_enemies and attempts < 2000:
            attempts += 1
            x = int(self.rng.integers(10, self.width - 10))
            y = self._highest_surface_y_at_column(x)
            if y is None or y < 0:
                continue
            if any(abs(enemy.x - x) < 4.0 and abs(enemy.y - (y + 1.0)) < 1.0 for enemy in self.enemies):
                continue
            self._spawn_enemy_on_surface(x, y)

    def _pad_question_blocks(self) -> None:
        attempts = 0
        while self.enable_question_blocks and len(self.question_blocks) < self.num_question_blocks and attempts < 2000:
            attempts += 1
            x = int(self.rng.integers(12, self.width - 12))
            y = int(self.rng.choice([2, 3, 4, 5]))
            if self._is_solid_tile(x, y):
                continue
            if self._tile_has_support_nearby(x, y - 2):
                self._add_question_block(x, y, 'mushroom' if self.enable_items and self.rng.random() < 0.3 else 'coin')

    def _sprinkle_default_coins(self) -> None:
        attempts = 0
        while self.enable_coins and len(self.coins) < self.num_coins and attempts < 3000:
            attempts += 1
            x = int(self.rng.integers(8, self.width - 8))
            surface = self._highest_surface_y_at_column(x)
            if surface is None:
                continue
            y = float(surface + int(self.rng.choice([2, 3])))
            if self._is_solid_tile(x, int(round(y))):
                continue
            if any(abs(coin.x - x) < 2.0 and abs(coin.y - y) < 1.0 for coin in self.coins):
                continue
            self.coins.append(Coin(x=float(x), y=y))

    def _tile_has_support_nearby(self, x: int, y: int) -> bool:
        for dx in range(-2, 3):
            if self._is_solid_tile(x + dx, max(0, y)):
                return True
        return False

    def _place_coin_arc(self, x0: int, x1: int, y: int) -> None:
        if not self.enable_coins:
            return
        for x in range(max(2, x0), min(self.width - 2, x1 + 1), 2):
            offset = 1 if (x - x0) % 4 == 0 else 0
            self.coins.append(Coin(x=float(x), y=float(y + offset)))

    def _carve_gap(self, a: int, b: int) -> None:
        a = max(3, int(a))
        b = min(self.width - 6, int(b))
        if a > b:
            return
        for x in range(a, b + 1):
            self._remove_solid_tile(x, 0)
        self.gaps.append((a, b))

    def _add_platform(self, x0: int, x1: int, y: int, bridge: bool = False) -> None:
        x0 = max(2, int(x0))
        x1 = min(self.width - 3, int(x1))
        y = max(1, min(self.height - 4, int(y)))
        if x0 > x1:
            return
        for x in range(x0, x1 + 1):
            self._add_solid_tile(x, y)
        platform = Platform(x0=x0, x1=x1, y=y)
        self.platforms.append(platform)
        if bridge:
            self.bridge_platforms.append(platform)

    def _add_question_block(self, x: int, y: int, item_kind: str) -> None:
        if not self.enable_question_blocks:
            return
        x = max(3, min(self.width - 4, int(x)))
        y = max(2, min(self.height - 4, int(y)))
        if self._is_solid_tile(x, y):
            return
        self._add_solid_tile(x, y)
        self.question_blocks.append(QuestionBlock(x=x, y=y, item_kind=item_kind))

    def _spawn_enemy_on_surface(self, x: int, tile_y: int) -> None:
        if not self.enable_moving_enemies:
            return
        left = x
        while self._is_solid_tile(left - 1, tile_y):
            left -= 1
        right = x
        while self._is_solid_tile(right + 1, tile_y):
            right += 1
        if right - left < 2:
            return
        enemy = Enemy(
            x=float(x),
            y=float(tile_y + 1.0),
            vx=float(self.enemy_speed if self.rng.random() < 0.5 else -self.enemy_speed),
            left=float(left),
            right=float(right),
        )
        self.enemies.append(enemy)

    def _add_solid_tile(self, x: int, y: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.solid_tiles.add((int(x), int(y)))

    def _remove_solid_tile(self, x: int, y: int) -> None:
        self.solid_tiles.discard((int(x), int(y)))

    def _is_solid_tile(self, x: int, y: int) -> bool:
        return (int(x), int(y)) in self.solid_tiles

    def _highest_surface_y_at_column(self, x: int) -> int | None:
        ys = [y for sx, y in self.solid_tiles if sx == int(x)]
        return max(ys) if ys else None

    def _surface_top_below(self, x: float, y_limit: float) -> float | None:
        sampled_x = {int(round(x)), int(np.floor(x - 0.25)), int(np.ceil(x + 0.25))}
        candidates: list[float] = []
        for xi in sampled_x:
            for sx, sy in self.solid_tiles:
                if sx != xi:
                    continue
                top = float(sy + 1)
                if top <= y_limit + 1e-6:
                    candidates.append(top)
        if not candidates:
            return None
        return max(candidates)

    def _resolve_horizontal(self, old_x: float, old_y: float, new_x: float) -> float:
        direction = np.sign(new_x - old_x)
        if direction == 0.0:
            return float(np.clip(new_x, 0.0, self.width - 1.0))

        # `player.y` is the feet position on the top surface of a tile.
        # Sample slightly above the feet; sampling below would treat the floor
        # itself as a horizontal blocker and freeze ground movement.
        edge_x = new_x + 0.35 * direction
        sample_ys = [old_y + 0.05, old_y + 0.45, old_y + 0.85]
        xi = int(np.floor(edge_x))
        for y in sample_ys:
            yi = int(np.floor(y))
            if self._is_solid_tile(xi, yi):
                return old_x
        return float(np.clip(new_x, 0.0, self.width - 1.0))

    def _check_head_collision(self, old_x: float, old_y: float, new_y: float) -> tuple[float, bool]:
        head_old = old_y + 0.9
        head_new = new_y + 0.9
        sampled_x = {int(round(old_x)), int(np.floor(old_x - 0.25)), int(np.ceil(old_x + 0.25))}
        ceiling_hit = False
        candidate_bottoms: list[float] = []
        for xi in sampled_x:
            for sx, sy in self.solid_tiles:
                if sx != xi:
                    continue
                bottom = float(sy)
                if head_old < bottom <= head_new + 1e-6:
                    candidate_bottoms.append(bottom)
        if candidate_bottoms:
            ceiling_hit = True
            new_y = min(candidate_bottoms) - 0.91
            self._activate_block_below(old_x, min(candidate_bottoms))
        return new_y, ceiling_hit

    def _activate_block_below(self, x: float, block_bottom: float) -> None:
        xi = int(round(x))
        yi = int(round(block_bottom))
        for block in self.question_blocks:
            if block.x == xi and block.y == yi and not block.used:
                block.used = True
                self._spawn_block_reward(block)
                return

    def _spawn_block_reward(self, block: QuestionBlock) -> None:
        if block.item_kind == 'coin':
            self.player.collected_coins += 1
            self.player.score += 10
        elif self.enable_items:
            self.items.append(
                Item(
                    x=float(block.x),
                    y=float(block.y + 1.0),
                    kind=block.item_kind,
                    vx=0.03 if self.rng.random() < 0.5 else -0.03,
                    vy=0.05,
                    emerging_frames=12,
                )
            )

    def _check_landing(self, old_x: float, old_y: float, new_x: float, new_y: float) -> tuple[float, bool]:
        sampled_x = {int(round(new_x)), int(np.floor(new_x - 0.25)), int(np.ceil(new_x + 0.25))}
        landing_surfaces: list[float] = []
        for xi in sampled_x:
            for sx, sy in self.solid_tiles:
                if sx != xi:
                    continue
                top = float(sy + 1)
                if new_y <= top <= old_y + 1e-6:
                    landing_surfaces.append(top)
        if landing_surfaces:
            return max(landing_surfaces), True
        return new_y, False

    def _update_enemies(self) -> None:
        for enemy in self.enemies:
            enemy.x += enemy.vx
            if enemy.x <= enemy.left + 0.1:
                enemy.x = enemy.left + 0.1
                enemy.vx = abs(enemy.vx)
            elif enemy.x >= enemy.right - 0.1:
                enemy.x = enemy.right - 0.1
                enemy.vx = -abs(enemy.vx)

    def _update_items(self) -> None:
        for item in self.items:
            if item.collected:
                continue
            if item.emerging_frames > 0:
                item.y += 0.05
                item.emerging_frames -= 1
                continue
            item.vy -= 0.02
            old_y = item.y
            item.x = float(np.clip(item.x + item.vx, 0.0, self.width - 1.0))
            item.y += item.vy
            landing_y, landed = self._check_landing(item.x, old_y, item.x, item.y)
            if landed:
                item.y = landing_y
                item.vy = 0.0
            if item.x <= 1.0 or item.x >= self.width - 2.0:
                item.vx *= -1.0

    def _collect_objects(self) -> tuple[int, int]:
        coin_hits = 0
        item_hits = 0
        for coin in self.coins:
            if not coin.collected and abs(self.player.x - coin.x) <= 0.7 and abs(self.player.y - coin.y) <= 1.0:
                coin.collected = True
                coin_hits += 1
        for item in self.items:
            if not item.collected and abs(self.player.x - item.x) <= 0.8 and abs(self.player.y - item.y) <= 1.0:
                item.collected = True
                item_hits += 1
        self.player.collected_coins += coin_hits
        self.player.collected_items += item_hits
        self.player.score += 10 * coin_hits + 200 * item_hits
        return coin_hits, item_hits

    def _gap_clear_count(self, old_x: float, new_x: float, y: float) -> int:
        if y < 0.8:
            return 0
        count = 0
        for gap_idx, (_a, b) in enumerate(self.gaps):
            if gap_idx in self.cleared_gap_indices:
                continue
            if old_x <= b + 0.2 < new_x:
                self.cleared_gap_indices.add(gap_idx)
                count += 1
        return count

    def _near_uncleared_gap(self, x: float, lookahead: float) -> bool:
        for gap_idx, (a, b) in enumerate(self.gaps):
            if gap_idx in self.cleared_gap_indices:
                continue
            if b < x - 1.0:
                continue
            if -0.5 <= float(a - x) <= lookahead:
                return True
        return False

    def _enemy_pass_count(self, old_x: float, new_x: float) -> int:
        count = 0
        for enemy in self.enemies:
            enemy_id = id(enemy)
            if enemy_id in self.passed_enemy_ids:
                continue
            if old_x <= enemy.x < new_x - 0.5:
                self.passed_enemy_ids.add(enemy_id)
                count += 1
        return count

    def _checkpoint_count(self) -> int:
        if self.checkpoint_interval <= 0 or self.checkpoint_reward == 0.0:
            return 0
        count = 0
        while self.max_x >= self.next_checkpoint_x:
            count += 1
            self.next_checkpoint_x += float(self.checkpoint_interval)
        return count

    def _find_enemy_collision(self) -> Enemy | None:
        px = self.player.x
        py = self.player.y
        for enemy in self.enemies:
            if abs(px - enemy.x) <= 0.6 and abs(py - enemy.y) <= 0.85:
                return enemy
        return None

    def step(self, action):
        self.t += 1
        p = self.player
        was_on_ground = p.on_ground
        previous_coin_count = p.collected_coins
        previous_item_count = p.collected_items

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f'Expected continuous action with shape (3,), got {action.shape}')

        move_axis = float(np.clip(action[0], -1.0, 1.0))
        jump_signal = float(np.clip(action[1], -1.0, 1.0))
        run_signal = float(np.clip(action[2], -1.0, 1.0))

        jump_pressed = jump_signal > 0.0
        run_pressed = run_signal > 0.0
        run_strength = 0.5 * (run_signal + 1.0)

        if p.on_ground:
            base_accel = 0.035
            run_accel = 0.055
            friction = 0.80
            max_speed = 0.18 + 0.15 * run_strength
        else:
            base_accel = 0.018
            run_accel = 0.028
            friction = 0.94
            max_speed = 0.16 + 0.12 * run_strength

        accel = move_axis * (base_accel + run_accel * run_strength)
        p.vx = friction * p.vx + accel
        p.vx = float(np.clip(p.vx, -max_speed, max_speed))

        if jump_pressed:
            p.jump_hold_frames += 1
        else:
            p.jump_hold_frames = 0

        jump_started = bool(jump_pressed and p.on_ground)
        if jump_started:
            p.vy = 0.46
            p.on_ground = False
            p.jump_hold_frames = 1

        gravity = 0.032
        if jump_pressed and p.vy > 0.0 and p.jump_hold_frames <= 13:
            gravity = 0.018
        if (not jump_pressed) and p.vy > 0.0:
            gravity = 0.052
        p.vy -= gravity

        self._update_enemies()
        self._update_items()

        old_x = p.x
        old_y = p.y
        new_x = self._resolve_horizontal(old_x, old_y, p.x + p.vx)
        new_y = p.y + p.vy

        if p.vy > 0.0:
            new_y, ceiling_hit = self._check_head_collision(new_x, old_y, new_y)
            if ceiling_hit:
                p.vy = min(0.0, p.vy)

        landed = False
        if p.vy <= 0.0:
            landing_y, landed = self._check_landing(old_x, old_y, new_x, new_y)
            if landed:
                new_y = landing_y
                p.vy = 0.0
                p.on_ground = True
                p.jump_hold_frames = 0
            else:
                p.on_ground = False
        else:
            p.on_ground = False

        p.x = new_x
        p.y = new_y

        fell = p.y < -2.0
        reached_goal = p.x >= self.goal_x

        hit_enemy = False
        stomped_enemy = False
        enemy = self._find_enemy_collision()
        if enemy is not None:
            stomp_from_above = p.vy < -0.02 and old_y >= enemy.y + 0.2 and p.y >= enemy.y - 0.05
            if stomp_from_above:
                stomped_enemy = True
                p.vy = 0.30
                p.y = enemy.y + 1.2
                p.score += 100
                self.enemies = [other for other in self.enemies if other is not enemy]
            else:
                hit_enemy = True

        self._collect_objects()

        prev_max_x = self.max_x
        self.max_x = max(self.max_x, p.x)
        progress_reward = self.max_x - prev_max_x
        coin_delta = self.player.collected_coins - previous_coin_count
        item_delta = self.player.collected_items - previous_item_count
        safe_landing = bool(landed and not was_on_ground and not fell and not hit_enemy)
        gap_clear_count = self._gap_clear_count(old_x, p.x, p.y)
        gap_jump = bool(jump_started and self._near_uncleared_gap(old_x, self.gap_jump_lookahead))
        gap_jump_hold = bool(
            jump_pressed
            and p.vy > 0.0
            and not p.on_ground
            and self._near_uncleared_gap(old_x, self.gap_jump_hold_lookahead)
        )
        enemy_pass_count = 0 if hit_enemy else self._enemy_pass_count(old_x, p.x)
        checkpoint_count = self._checkpoint_count()

        if progress_reward > self.stagnation_epsilon:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        stagnated = self.stagnation_counter >= self.stagnation_timeout

        reward_components = {
            'progress': self.progress_reward_scale * progress_reward,
            'step': self.step_penalty,
            'coin': self.coin_reward * coin_delta,
            'item': self.item_reward * item_delta,
            'stomp': self.stomp_reward if stomped_enemy else 0.0,
            'safe_landing': self.safe_landing_reward if safe_landing else 0.0,
            'gap_clear': self.gap_clear_reward * gap_clear_count,
            'gap_jump': self.gap_jump_reward if gap_jump else 0.0,
            'gap_jump_hold': self.gap_jump_hold_reward if gap_jump_hold else 0.0,
            'enemy_pass': self.enemy_pass_reward * enemy_pass_count,
            'checkpoint': self.checkpoint_reward * checkpoint_count,
            'goal': self.goal_reward if reached_goal else 0.0,
            'death': self.death_penalty if fell or hit_enemy else 0.0,
            'stagnation': self.stagnation_penalty if stagnated else 0.0,
        }
        reward = sum(reward_components.values())
        if stomped_enemy:
            self.passed_enemy_ids.add(id(enemy))
        if reached_goal:
            p.score += 1000

        terminated = bool(fell or hit_enemy or reached_goal or stagnated)
        truncated = self.t >= self.max_steps

        info = {
            'x': p.x,
            'y': p.y,
            'vx': p.vx,
            'vy': p.vy,
            'max_x': self.max_x,
            'reached_goal': reached_goal,
            'fell': fell,
            'hit_enemy': hit_enemy,
            'stomped_enemy': stomped_enemy,
            'safe_landing': safe_landing,
            'gap_clear_count': gap_clear_count,
            'gap_jump': gap_jump,
            'gap_jump_hold': gap_jump_hold,
            'enemy_pass_count': enemy_pass_count,
            'checkpoint_count': checkpoint_count,
            'reward_components': reward_components,
            'stagnated': stagnated,
            'stagnation_counter': self.stagnation_counter,
            'move_axis': move_axis,
            'jump_pressed': jump_pressed,
            'run_pressed': run_pressed,
            'run_strength': run_strength,
            'coins_collected': self.player.collected_coins,
            'items_collected': self.player.collected_items,
            'score': self.player.score,
            'gaps': self.gaps,
            'num_platforms': len(self.platforms),
            'num_enemies': len(self.enemies),
            'num_coins': sum(0 if coin.collected else 1 for coin in self.coins),
        }

        return self._obs(), float(reward), terminated, truncated, info

    def _nearest_enemy_features(self) -> tuple[float, float, float]:
        nearest_dx = 12.0
        nearest_dy = 0.0
        nearest_vx = 0.0
        for enemy in self.enemies:
            dx = float(enemy.x - self.player.x)
            if dx >= -1.0 and dx < nearest_dx:
                nearest_dx = dx
                nearest_dy = float(enemy.y - self.player.y)
                nearest_vx = float(enemy.vx)
        return nearest_dx / 12.0, nearest_dy / 6.0, nearest_vx / max(1e-6, self.enemy_speed)

    def _nearest_collectible_features(self, objects: list[Coin] | list[Item]) -> tuple[float, float]:
        nearest_dx = 12.0
        nearest_dy = 0.0
        for obj in objects:
            if getattr(obj, 'collected', False):
                continue
            dx = float(obj.x - self.player.x)
            if dx >= -1.0 and dx < nearest_dx:
                nearest_dx = dx
                nearest_dy = float(obj.y - self.player.y)
        return nearest_dx / 12.0, nearest_dy / 6.0

    def _nearest_block_features(self) -> tuple[float, float]:
        nearest_dx = 12.0
        nearest_dy = 0.0
        for block in self.question_blocks:
            dx = float(block.x - self.player.x)
            if dx >= -1.0 and dx < nearest_dx:
                nearest_dx = dx
                nearest_dy = float(block.y + 1 - self.player.y)
        return nearest_dx / 12.0, nearest_dy / 6.0

    def _nearest_platform_features(self) -> tuple[float, float, float]:
        nearest_dx = 12.0
        nearest_dy = 0.0
        nearest_h = 0.0
        for platform in self.platforms:
            if platform.x1 < self.player.x - 1:
                continue
            px = float(np.clip(self.player.x, platform.x0, platform.x1))
            dx = px - self.player.x
            if dx >= -1.0 and dx < nearest_dx:
                nearest_dx = dx
                nearest_dy = float(platform.y + 1 - self.player.y)
                nearest_h = float(platform.y)
        return nearest_dx / 12.0, nearest_dy / 6.0, nearest_h / max(1, self.height)

    def _gap_features(self) -> tuple[float, float, float]:
        nearest_gap_start_dx = 12.0
        nearest_gap_end_dx = 12.0
        bridge_ahead = 0.0
        for a, b in self.gaps:
            if b < self.player.x - 1:
                continue
            nearest_gap_start_dx = float(a - self.player.x)
            nearest_gap_end_dx = float(b - self.player.x)
            if any(platform.x0 <= a and platform.x1 >= b for platform in self.bridge_platforms):
                bridge_ahead = 1.0
            break
        return nearest_gap_start_dx / 12.0, nearest_gap_end_dx / 12.0, bridge_ahead

    def _lookahead_features(self) -> list[float]:
        xi = int(round(self.player.x))
        features: list[float] = []
        for d in range(1, 9):
            xj = xi + d
            solid_low = 1.0 if self._is_solid_tile(xj, 0) else 0.0
            solid_mid = 1.0 if any(self._is_solid_tile(xj, y) for y in (2, 3, 4)) else 0.0
            solid_high = 1.0 if any(self._is_solid_tile(xj, y) for y in (5, 6, 7, 8)) else 0.0
            enemy = 1.0 if any(abs(enemy.x - xj) <= 0.6 for enemy in self.enemies) else 0.0
            coin = 1.0 if any((not coin.collected) and abs(coin.x - xj) <= 0.5 for coin in self.coins) else 0.0
            features.extend([solid_low, solid_mid, solid_high, enemy, coin])
        return features


    def _camera_left_count(self) -> int:
        return max(1, self.width - self.camera_width + 1)

    def _cache_static_screen_views(self) -> None:
        views = np.empty((self._camera_left_count(), self.screen_dim), dtype=np.float32)
        for left in range(views.shape[0]):
            right = left + self.camera_width
            views[left] = self._static_screen_grid[:, left:right, :].reshape(-1)
        self._static_screen_views = views

    def _rebuild_static_screen_grid(self) -> None:
        grid = np.zeros((self.height, self.width, self.num_screen_channels), dtype=np.float32)
        empty_idx = self.screen_channel_index['empty']
        grid[:, :, empty_idx] = 1.0

        block_map = {(block.x, block.y): block for block in self.question_blocks}
        for world_x, world_y in self.solid_tiles:
            if not (0 <= world_x < self.width and 0 <= world_y < self.height):
                continue
            screen_row = self.height - 1 - world_y
            grid[screen_row, world_x, empty_idx] = 0.0
            block = block_map.get((world_x, world_y))
            if block is not None:
                channel = 'used_block' if block.used else 'question_block'
                grid[screen_row, world_x, self.screen_channel_index[channel]] = 1.0
            elif world_y == 0:
                grid[screen_row, world_x, self.screen_channel_index['ground']] = 1.0
            else:
                grid[screen_row, world_x, self.screen_channel_index['platform']] = 1.0

        goal_x = int(self.goal_x)
        if 0 <= goal_x < self.width:
            goal_idx = self.screen_channel_index['goal']
            grid[:, goal_x, goal_idx] = 1.0
            grid[:, goal_x, empty_idx] = 0.0

        self._static_screen_grid = grid
        self._cache_static_screen_views()

    def _build_input_feature_names(self) -> list[str]:
        if self.observation_mode == 'engineered':
            return list(self.ENGINEERED_FEATURE_NAMES)

        if self.observation_mode != 'full_screen':
            raise ValueError(f'Unknown observation_mode: {self.observation_mode}')

        names: list[str] = []
        for screen_row in range(self.height):
            for screen_col in range(self.camera_width):
                for channel_name in self.SCREEN_CHANNEL_NAMES:
                    names.append(f'screen_r{screen_row}_c{screen_col}_{channel_name}')
        if self.include_state_features:
            names.extend(self.STATE_FEATURE_NAMES)
        return names

    def _engineered_obs(self) -> np.ndarray:
        p = self.player
        enemy_dx, enemy_dy, enemy_vx = self._nearest_enemy_features()
        coin_dx, coin_dy = self._nearest_collectible_features(self.coins)
        item_dx, item_dy = self._nearest_collectible_features(self.items)
        block_dx, block_dy = self._nearest_block_features()
        platform_dx, platform_dy, platform_height = self._nearest_platform_features()
        gap_start_dx, gap_end_dx, bridge_ahead = self._gap_features()

        return np.array(
            [
                p.x / self.width,
                p.y / max(1, self.height),
                p.vx,
                p.vy,
                1.0 if p.on_ground else 0.0,
                1.0 if p.vy < 0.0 else 0.0,
                p.collected_coins / max(1, self.num_coins),
                p.collected_items / max(1, max(1, self.num_question_blocks)),
                enemy_dx,
                enemy_dy,
                enemy_vx,
                coin_dx,
                coin_dy,
                item_dx,
                item_dy,
                block_dx,
                block_dy,
                platform_dx,
                platform_dy,
                platform_height,
                gap_start_dx,
                gap_end_dx,
                bridge_ahead,
                (self.goal_x - p.x) / self.width,
                self.t / self.max_steps,
                *self._lookahead_features(),
            ],
            dtype=np.float32,
        )

    def _screen_obs(self) -> np.ndarray:
        left, right = self.get_camera_bounds()
        channel_index = self.screen_channel_index
        screen = self._screen_obs_buffer[: self.screen_dim]
        screen[:] = self._static_screen_views[left]

        def set_dynamic_cell(world_x: int, world_y: int, channel: str) -> None:
            if left <= world_x < right and 0 <= world_y < self.height:
                screen_col = world_x - left
                screen_row = self.height - 1 - world_y
                base = (screen_row * self.camera_width + screen_col) * self.num_screen_channels
                screen[base + channel_index[channel]] = 1.0
                screen[base + channel_index['empty']] = 0.0

        for enemy in self.enemies:
            ex = int(round(enemy.x))
            ey = int(round(enemy.y))
            set_dynamic_cell(ex, ey, 'enemy')

        for coin in self.coins:
            if coin.collected:
                continue
            cx = int(round(coin.x))
            cy = int(round(coin.y))
            set_dynamic_cell(cx, cy, 'coin')

        for item in self.items:
            if item.collected:
                continue
            ix = int(round(item.x))
            iy = int(round(item.y))
            set_dynamic_cell(ix, iy, 'item')

        px = int(round(self.player.x))
        py = int(round(self.player.y))
        set_dynamic_cell(px, py, 'player')

        if not self.include_state_features:
            return screen

        player_screen_x = (self.player.x - left) / max(1, self.camera_width - 1)
        gap_start_dx, gap_end_dx, bridge_ahead = self._gap_features()
        state = self._screen_obs_buffer[self.screen_dim :]
        state[0] = player_screen_x
        state[1] = self.player.y / max(1, self.height)
        state[2] = self.player.vx
        state[3] = self.player.vy
        state[4] = 1.0 if self.player.on_ground else 0.0
        state[5] = min(1.0, self.player.score / 2000.0)
        state[6] = (self.goal_x - self.player.x) / self.width
        state[7] = self.t / self.max_steps
        state[8] = gap_start_dx
        state[9] = gap_end_dx
        state[10] = bridge_ahead
        return self._screen_obs_buffer

    def _obs(self) -> np.ndarray:
        if self.observation_mode == 'engineered':
            return self._engineered_obs()
        if self.observation_mode == 'full_screen':
            return self._screen_obs()
        raise ValueError(f'Unknown observation_mode: {self.observation_mode}')

    def get_input_feature_names(self) -> list[str]:
        return list(self.input_feature_names)

    def get_camera_bounds(self) -> tuple[int, int]:
        half = self.camera_width // 2
        center = int(round(self.player.x))
        left = center - half
        right = left + self.camera_width
        if left < 0:
            left = 0
            right = self.camera_width
        if right > self.width:
            right = self.width
            left = self.width - self.camera_width
        return max(0, left), min(self.width, right)

    def render(self):
        import cv2

        scale = self.render_scale
        left, right = self.get_camera_bounds()
        view_width = right - left
        img = np.zeros((self.height * scale, view_width * scale, 3), dtype=np.uint8)
        img[:, :, :] = np.array([140, 208, 244], dtype=np.uint8)

        for row in range(self.height):
            if row % 4 == 0:
                y0 = (self.height - row - 1) * scale
                img[y0 : y0 + 1, :, :] = np.array([170, 225, 250], dtype=np.uint8)

        for x in range(left, right):
            local_x = x - left
            for y in range(self.height):
                if not self._is_solid_tile(x, y):
                    continue
                ry = self.height - 1 - y
                color = np.array([115, 98, 62], dtype=np.uint8) if y > 0 else np.array([90, 170, 86], dtype=np.uint8)
                if any(block.x == x and block.y == y for block in self.question_blocks):
                    block = next(block for block in self.question_blocks if block.x == x and block.y == y)
                    color = np.array([218, 176, 55], dtype=np.uint8) if not block.used else np.array([155, 140, 110], dtype=np.uint8)
                img[ry * scale : (ry + 1) * scale, local_x * scale : (local_x + 1) * scale] = color

        for coin in self.coins:
            if coin.collected:
                continue
            cx = int(round(coin.x))
            cy = int(round(coin.y))
            if left <= cx < right and 0 <= cy < self.height:
                local_x = cx - left
                ry = self.height - 1 - cy
                cv2.circle(img, (local_x * scale + scale // 2, ry * scale + scale // 2), max(2, scale // 4), (245, 210, 40), -1)

        for item in self.items:
            if item.collected:
                continue
            ix = int(round(item.x))
            iy = int(round(item.y))
            if left <= ix < right and 0 <= iy < self.height:
                local_x = ix - left
                ry = self.height - 1 - iy
                color = (210, 50, 50) if item.kind == 'mushroom' else (240, 220, 80)
                img[ry * scale : (ry + 1) * scale, local_x * scale : (local_x + 1) * scale] = color

        for enemy in self.enemies:
            ex = int(round(enemy.x))
            ey = int(round(enemy.y))
            if left <= ex < right and 0 <= ey < self.height:
                local_x = ex - left
                ry = self.height - 1 - ey
                img[ry * scale : (ry + 1) * scale, local_x * scale : (local_x + 1) * scale] = [196, 58, 48]
                eye_y = ry * scale + max(2, scale // 4)
                cv2.circle(img, (local_x * scale + max(2, scale // 3), eye_y), max(1, scale // 10), (255, 255, 255), -1)

        gx = int(self.goal_x)
        if left <= gx < right:
            local_gx = gx - left
            img[:, local_gx * scale : (local_gx + 1) * scale] = [245, 226, 74]

        px = int(round(self.player.x))
        py = int(round(self.player.y))
        if left <= px < right and 0 <= py < self.height:
            local_px = px - left
            ry = self.height - 1 - py
            img[ry * scale : (ry + 1) * scale, local_px * scale : (local_px + 1) * scale] = [56, 72, 230]

        cv2.putText(
            img,
            f'x={self.player.x:.1f}/{self.goal_x:.0f} score={self.player.score} coins={self.player.collected_coins} items={self.player.collected_items}',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            2,
        )
        return img
