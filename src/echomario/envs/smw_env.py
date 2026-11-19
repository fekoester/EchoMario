"""
Super Mario World (SNES) through stable-retro, seen by the reservoir as a tile grid read from RAM.

No ROM is included. Import your own dump once with ``python -m stable_retro.import <dir>``
(the integration expects the headerless US ROM, SHA-1 6b47bb75...). See docs/real_mario.md.

What the agent sees (``observation``): an egocentric grid of ROWS x COLS 16-pixel tiles around
Mario, one-hot over CHANNELS, read from the level's Map16 tile table and the sprite tables,
followed by four scalars (horizontal and vertical speed, on ground, powered up). Layout is
(row, col, channel) row-major, the same as the toy platformer's screen, so the reservoir's local
receptive-field input works unchanged.

One env step is FRAME_SKIP emulator frames with the same buttons held (30 decisions per second,
like the toy platformer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GAME = 'SuperMarioWorld-Snes-v0'
WRAM_OFFSET = 10240  # stable-retro concatenates the 0x000000 (8 KB) and 0x700000 (2 KB) blocks before WRAM
FRAME_SKIP = 2

ACTION_NAMES = ('idle', 'right', 'right_run', 'jump', 'right_jump', 'right_run_jump', 'left', 'left_jump')
_ACTION_BUTTONS = (
    (),
    ('RIGHT',),
    ('RIGHT', 'Y'),
    ('B',),
    ('RIGHT', 'B'),
    ('RIGHT', 'Y', 'B'),
    ('LEFT',),
    ('LEFT', 'B'),
)

# Horizontal, non-water levels among the bundled save-states.
TRAIN_LEVELS = ('YoshiIsland1', 'YoshiIsland3', 'YoshiIsland4', 'DonutPlains1', 'DonutPlains4', 'DonutPlains5',
                'Forest1', 'Forest3', 'VanillaDome1', 'VanillaDome2', 'VanillaDome4', 'ChocolateIsland1', 'Bridges1')
TEST_LEVELS = ('YoshiIsland2', 'Forest5', 'VanillaDome3', 'VanillaDome5', 'ChocolateIsland2', 'Bridges2')

ROWS, COLS = 16, 20
ROWS_ABOVE, COLS_BEHIND = 10, 4  # Mario's feet are on row ROWS_ABOVE, column COLS_BEHIND
CHANNEL_NAMES = ('solid', 'platform', 'coin', 'enemy', 'item')
N_SCALARS = 4
SCREEN_SHAPE = (ROWS, COLS, len(CHANNEL_NAMES))
OBS_DIM = ROWS * COLS * len(CHANNEL_NAMES) + N_SCALARS

ITEM_SPRITES = frozenset({0x21, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79})  # moving coin, mushroom, flower, star, feather, 1up
HARMLESS_SPRITES = frozenset({0x35, 0x7B, 0x7C})  # Yoshi, goal tape, ...


def tile_class(t: np.ndarray) -> np.ndarray:
    """Map16 tile number -> channel index (-1 = empty). Vanilla SMW tile behaviour:
    0x100-0x110 ledges you can stand on and jump through, 0x111-0x1FF solid blocks, slopes and
    pipes, 0x2B-0x2D coins and dragon coins, everything else on page 0 passable."""
    out = np.full(t.shape, -1, np.int8)
    page = t & 0x1FF
    out[(page >= 0x111)] = 0
    out[(page >= 0x100) & (page <= 0x110)] = 1
    out[(page >= 0x2B) & (page <= 0x2D)] = 2
    return out


@dataclass
class _Track:
    max_x: int = 0
    steps: int = 0
    since_progress: int = 0
    coins: int = 0
    stomps: int = 0
    hurt: int = 0
    events: list = field(default_factory=list)


class SMWEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, level: str = 'YoshiIsland2', max_steps: int = 3000, stagnation_steps: int = 300,
                 render_mode: str | None = 'rgb_array'):
        import stable_retro as retro

        self.level = level
        self.max_steps = int(max_steps)
        self.stagnation_steps = int(stagnation_steps)
        self.render_mode = render_mode
        self.retro = retro.make(GAME, state=level, render_mode=None, inttype=retro.data.Integrations.ALL)
        self.em = self.retro.unwrapped.em
        buttons = self.retro.unwrapped.buttons
        self._masks = []
        for names in _ACTION_BUTTONS:
            m = np.zeros(len(buttons), np.uint8)
            for n in names:
                m[buttons.index(n)] = 1
            self._masks.append(m)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-4.0, 4.0, shape=(OBS_DIM,), dtype=np.float32)
        self.screen_dim = ROWS * COLS * len(CHANNEL_NAMES)
        self.track = _Track()
        self.goal_x = 1
        self._frame = None

    # ----------------------------------------------------------------- RAM helpers
    def _wram(self) -> np.ndarray:
        return self.retro.unwrapped.get_ram()[WRAM_OFFSET:]

    @staticmethod
    def _u16(w: np.ndarray, a: int) -> int:
        return int(w[a]) | (int(w[a + 1]) << 8)

    @staticmethod
    def _s8(v) -> int:
        v = int(v)
        return v - 256 if v > 127 else v

    def mario(self, w: np.ndarray | None = None) -> dict:
        w = self._wram() if w is None else w
        return {
            'x': self._u16(w, 0x94), 'y': self._u16(w, 0x96),
            'vx': self._s8(w[0x7B]), 'vy': self._s8(w[0x7D]),
            'anim': int(w[0x71]), 'power': int(w[0x19]), 'on_ground': bool(w[0x13EF]) or bool(w[0x1471]),
            'goal': bool(w[0x1493]) or bool(w[0x13C6]), 'coins': int(w[0xDBF]), 'lives': int(w[0xDBE]),
            'screens': int(w[0x5D]),
        }

    def sprites(self, w: np.ndarray | None = None) -> list[tuple[int, int, int]]:
        """(number, x, y) of the active sprites."""
        w = self._wram() if w is None else w
        out = []
        for i in range(12):
            if int(w[0x14C8 + i]) >= 8:  # 8 = normal, 9-11 carried/kicked shells
                out.append((int(w[0x9E + i]), int(w[0xE4 + i]) | (int(w[0x14E0 + i]) << 8),
                            int(w[0xD8 + i]) | (int(w[0x14D4 + i]) << 8)))
        return out

    # ----------------------------------------------------------------- observation
    def grid(self, w: np.ndarray | None = None) -> np.ndarray:
        """(ROWS, COLS, CHANNELS) one-hot tile grid centred on Mario."""
        w = self._wram() if w is None else w
        m = self.mario(w)
        feet_ty = (m['y'] + 24) // 16
        mx_t = (m['x'] + 8) // 16
        tys = np.arange(feet_ty - ROWS_ABOVE, feet_ty - ROWS_ABOVE + ROWS)
        txs = np.arange(mx_t - COLS_BEHIND, mx_t - COLS_BEHIND + COLS)
        g = np.zeros(SCREEN_SHAPE, np.float32)
        ty, tx = np.meshgrid(tys, txs, indexing='ij')
        inside = (ty >= 0) & (ty < 27) & (tx >= 0) & (tx < 16 * max(1, m['screens']))
        idx = (tx // 16) * 0x1B0 + ty * 16 + (tx % 16)
        idx = np.where(inside, idx, 0)
        lo = w[0xC800 + idx].astype(np.int32)
        hi = w[0x1C800 + idx].astype(np.int32)
        cls = np.where(inside, tile_class(lo | (hi << 8)), -1)
        # below the level counts as a pit (empty), above as empty
        r, c = np.nonzero(cls >= 0)
        g[r, c, cls[r, c]] = 1.0
        for num, sx, sy in self.sprites(w):
            if num in HARMLESS_SPRITES:
                continue
            ch = 4 if num in ITEM_SPRITES else 3
            rr = (sy + 8) // 16 - (feet_ty - ROWS_ABOVE)
            cc = (sx + 8) // 16 - (mx_t - COLS_BEHIND)
            if 0 <= rr < ROWS and 0 <= cc < COLS:
                g[rr, cc, ch] = 1.0
        return g

    def _obs(self, w: np.ndarray) -> np.ndarray:
        m = self.mario(w)
        scalars = np.array([m['vx'] / 48.0, m['vy'] / 64.0, float(m['on_ground']), float(m['power'] > 0)], np.float32)
        return np.concatenate([self.grid(w).reshape(-1), scalars])

    # ----------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        level = (options or {}).get('level', self.level)
        if level != self.level:
            self.retro.load_state(level, inttype=__import__('stable_retro').data.Integrations.ALL)
            self.level = level
        self._frame, _ = self.retro.reset()
        w = self._wram()
        m = self.mario(w)
        self.goal_x = 256 * max(1, m['screens']) - 48
        self.track = _Track(max_x=m['x'], coins=m['coins'])
        self._power = m['power']
        return self._obs(w), self._info(m, False, False)

    def emulate(self, action: int) -> None:
        """FRAME_SKIP frames with the action's buttons, without building an observation."""
        mask = self._masks[int(action)]
        for _ in range(FRAME_SKIP):
            self.em.set_button_mask(mask, 0)
            self.em.step()
        self._close_message_box()

    def _close_message_box(self) -> None:
        """Info-block messages freeze the game until a button is pressed; press A to close them."""
        if not self._wram()[0x1426]:
            return
        idle = np.zeros_like(self._masks[0])
        press = idle.copy()
        press[self.retro.unwrapped.buttons.index('A')] = 1
        for _ in range(240):
            for mask in (press, press, idle, idle):
                self.em.set_button_mask(mask, 0)
                self.em.step()
            if not self._wram()[0x1426]:
                break
        for _ in range(4):
            self.em.set_button_mask(idle, 0)
            self.em.step()

    def step(self, action: int):
        self.emulate(action)
        return self._after_step(compute_obs=True)

    def _after_step(self, compute_obs: bool):
        w = self._wram()
        m = self.mario(w)
        t = self.track
        t.steps += 1
        reward = max(0, m['x'] - t.max_x) / 16.0
        if m['x'] > t.max_x + 1:
            t.since_progress = 0
        else:
            t.since_progress += 1
        t.max_x = max(t.max_x, m['x'])
        if m['coins'] != t.coins:
            t.coins = m['coins']
        hurt = m['power'] < self._power
        self._power = m['power']
        t.hurt += int(hurt)
        died = m['anim'] == 9 or m['y'] > 27 * 16 + 32
        goal = m['goal']
        terminated = bool(died or goal)
        truncated = bool(not terminated and (t.steps >= self.max_steps or t.since_progress >= self.stagnation_steps))
        if died:
            reward -= 20.0
        if goal:
            reward += 50.0
        obs = self._obs(w) if compute_obs else None
        return obs, reward, terminated, truncated, self._info(m, died, hurt)

    def _info(self, m: dict, died: bool, hurt: bool) -> dict:
        t = self.track
        return {
            'x': m['x'], 'max_x': t.max_x, 'progress': min(1.0, t.max_x / self.goal_x), 'reached_goal': m['goal'],
            'died': died, 'hurt': hurt, 'coins': m['coins'], 'power': m['power'], 'steps': t.steps,
            'stagnated': t.since_progress >= self.stagnation_steps, 'level': self.level,
        }

    def render(self):
        return self.retro.unwrapped.em.get_screen()

    def close(self):
        # stable-retro allows one emulator per process; drop our reference so it is really freed
        self.em = None
        self.retro.close()

    # ----------------------------------------------------------------- planner support
    def get_sim_state(self) -> tuple:
        t = self.track
        return (self.em.get_state(), _Track(t.max_x, t.steps, t.since_progress, t.coins, t.stomps, t.hurt), self._power)

    def set_sim_state(self, s: tuple) -> None:
        self.em.set_state(s[0])
        t = s[1]
        self.track = _Track(t.max_x, t.steps, t.since_progress, t.coins, t.stomps, t.hurt)
        self._power = s[2]


def smw_screen_shape() -> tuple[int, int, int]:
    return SCREEN_SHAPE
