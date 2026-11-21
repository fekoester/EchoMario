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

import ctypes
import os
import sys
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GAME = 'SuperMarioWorld-Snes-v0'
# Optional faster emulator core: stable-retro's snes9x with a switch to skip drawing (scripts/build_snes_core.sh).
# stable-retro fixes its core directory on import, so this has to happen before stable_retro is imported.
FAST_CORE_DIR = os.environ.get('ECHOMARIO_SNES_CORE', os.path.expanduser('~/.cache/echomario/snes_core'))


LEVEL_STATE_DIR = os.environ.get('ECHOMARIO_LEVEL_STATES', os.path.expanduser('~/.cache/echomario/levelstates'))


def catalog_state(level: str) -> bytes | None:
    """Start state of a catalogue level ('L2A': loaded via the $0109 level override, see
    scripts/smw_level_catalog.py), or None for the save states shipped with stable-retro."""
    if not (len(level) == 3 and level[0] == 'L'):
        return None
    import gzip  # noqa: PLC0415

    path = os.path.join(LEVEL_STATE_DIR, f'{level}.state')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing: run scripts/smw_level_catalog.py first')
    with gzip.open(path, 'rb') as fh:
        return fh.read()


def _prefer_fast_core() -> None:
    if os.environ.get('ECHOMARIO_SNES_CORE') == '':  # explicitly disabled
        return
    if os.path.exists(os.path.join(FAST_CORE_DIR, 'snes9x_libretro.so')) and 'stable_retro' not in sys.modules:
        os.environ.setdefault('RETRO_CORE_PATH', FAST_CORE_DIR)


_prefer_fast_core()
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
# The same 19 levels split by difficulty: sorted by bunny-hop progress, every third level held out, so both
# sets are equally hard (bunny-hop 0.18 train / 0.20 held out). A gap between them is then overfitting.
BALANCED_TRAIN_LEVELS = ('YoshiIsland2', 'YoshiIsland4', 'DonutPlains1', 'VanillaDome3', 'DonutPlains5', 'Forest3',
                         'ChocolateIsland2', 'DonutPlains4', 'YoshiIsland3', 'Forest1', 'Bridges1', 'VanillaDome4',
                         'Bridges2')
BALANCED_TEST_LEVELS = ('YoshiIsland1', 'Forest5', 'VanillaDome2', 'VanillaDome5', 'ChocolateIsland1', 'VanillaDome1')
# All 24 horizontal levels with a bundled save-state (water and layer-2 levels included since observation v2).
ALL_LEVELS = ('YoshiIsland1', 'YoshiIsland2', 'YoshiIsland3', 'YoshiIsland4', 'DonutPlains1', 'DonutPlains2',
              'DonutPlains3', 'DonutPlains4', 'DonutPlains5', 'Forest1', 'Forest2', 'Forest3', 'Forest4', 'Forest5',
              'VanillaDome1', 'VanillaDome2', 'VanillaDome3', 'VanillaDome4', 'VanillaDome5', 'ChocolateIsland1',
              'ChocolateIsland2', 'ChocolateIsland3', 'Bridges1', 'Bridges2')

ROWS, COLS = 16, 20
ROWS_ABOVE, COLS_BEHIND = 10, 4  # Mario's feet are on row ROWS_ABOVE, column COLS_BEHIND
CHANNEL_NAMES = ('solid', 'platform', 'coin', 'enemy', 'item')
N_SCALARS = 4
SCREEN_SHAPE = (ROWS, COLS, len(CHANNEL_NAMES))
OBS_DIM = ROWS * COLS * len(CHANNEL_NAMES) + N_SCALARS

# Observation v2 (see docs/real_mario.md): tiles by their measured behaviour (smw_tiles.py) on layer 1 and on
# an interactive layer 2 (water in Yoshi's Island 4); the void below the level counts as hazard; enemies split
# by the game's own "can be jumped on" flag; sprite platforms, items, projectiles (extended sprites), cluster
# sprites (Boo rings) and the goal tape; every sprite covers its hitbox from the ROM's clipping table and
# carries its speed in two value channels; 19 scalars of Mario's state.
CHANNEL_NAMES_V2 = ('solid', 'platform', 'slope', 'water', 'climb', 'hazard', 'coin', 'enemy_stompable',
                    'enemy_harmful', 'platform_sprite', 'item', 'projectile', 'goal', 'sprite_vx', 'sprite_vy')
BINARY_CHANNELS_V2 = 13  # the first 13 are 0/1, sprite_vx/vy are speeds scaled to about -1..1
SCALAR_NAMES_V2 = ('vx', 'vy', 'on_ground', 'sub_x', 'sub_y', 'big', 'cape', 'fire', 'in_water', 'climbing', 'ducking',
                   'spin_jump', 'carrying', 'star', 'invulnerable', 'on_yoshi', 'facing_right', 'p_meter', 'on_sprite')
N_SCALARS_V2 = len(SCALAR_NAMES_V2)
# tile class (smw_tiles.TILE_CLASS_NAMES) -> v2 channel
_TILE_CHANNEL_V2 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
LAYER2_Y_OFFSET = 128  # px; fitted: layer-2 water matches the game's in-water flag in 99.8 % of 3129 frames (YI4)

ITEM_SPRITES = frozenset({0x21, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x81})  # moving coin, power-ups, 1up, changing item
HARMLESS_SPRITES = frozenset({0x35, 0x7B, 0x7C})  # Yoshi, goal tape, ...
# sprites Mario can stand on (vanilla sprite numbers): moving/falling/line-guided/chained platforms, skull raft,
# turn-block bridges, springboards, flying ? blocks, castle blocks, Lakitu's cloud, info box
PLATFORM_SPRITES = frozenset({0x2F, 0x52, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61,
                              0x62, 0x63, 0x64, 0x6B, 0x6C, 0x83, 0x84, 0x87, 0xA3, 0xB9, 0xBA, 0xBB, 0xC0, 0xC1, 0xC4})
# sprites whose interaction ignores the clipping table (custom platform code): assume a 48 x 16 box
PLATFORM_DEFAULT_BOX = (-16, 0, 48, 16)
# extended sprites that hurt Mario: Reznor fireball, hopping flame, hammer, bone, Torpedo Ted arm,
# piranha fireball, volcano lotus fire, baseball (Mario's own fireballs, smoke, splashes are ignored)
PROJECTILE_EXT_SPRITES = frozenset({0x02, 0x03, 0x04, 0x06, 0x08, 0x0B, 0x0C, 0x0D})
HARMLESS_SPRITES_V2 = frozenset({0x7C, 0xC7})  # (v1 also ignored Yoshi 0x35 and the goal tape 0x7B)
GOAL_SPRITES = frozenset({0x7B})
YOSHI_SPRITES = frozenset({0x35})


def channel_names(obs: str = 'v1') -> tuple[str, ...]:
    return CHANNEL_NAMES_V2 if obs == 'v2' else CHANNEL_NAMES


def n_scalars(obs: str = 'v1') -> int:
    return N_SCALARS_V2 if obs == 'v2' else N_SCALARS


_CLIPPING = None


def sprite_clipping() -> dict[str, np.ndarray]:
    """The ROM's sprite hitbox table (x/y offset, width, height per clipping index; 60 entries at
    $03:B56C..). Read from the imported ROM file."""
    global _CLIPPING
    if _CLIPPING is None:
        import stable_retro as retro

        rom = open(retro.data.get_romfile_path(GAME, retro.data.Integrations.ALL), 'rb').read()

        def table(addr):  # LoROM bank:address -> file offset (headerless ROM)
            off = ((addr >> 16) & 0x7F) * 0x8000 + ((addr & 0xFFFF) - 0x8000)
            return np.frombuffer(rom[off:off + 0x3C], np.uint8)

        _CLIPPING = {'dx': table(0x03B56C).astype(np.int8).astype(int), 'w': table(0x03B5A8).astype(int),
                     'dy': table(0x03B5E4).astype(np.int8).astype(int), 'h': table(0x03B620).astype(int)}
    return _CLIPPING


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
                 render_mode: str | None = 'rgb_array', rows: int = ROWS, cols: int = COLS,
                 rows_above: int = ROWS_ABOVE, cols_behind: int = COLS_BEHIND, obs: str = 'v1'):
        import stable_retro as retro

        self.level = level
        self.max_steps = int(max_steps)
        self.stagnation_steps = int(stagnation_steps)
        self.render_mode = render_mode
        self.rows, self.cols, self.rows_above, self.cols_behind = int(rows), int(cols), int(rows_above), int(cols_behind)
        if obs not in ('v1', 'v2'):
            raise ValueError(f'obs must be v1 or v2, not {obs!r}')
        self.obs_version = obs
        self.channel_names = channel_names(obs)
        self.screen_shape = (self.rows, self.cols, len(self.channel_names))
        self.obs_dim = self.rows * self.cols * len(self.channel_names) + n_scalars(obs)
        # the patched core exports echomario_set_av(render, audio); the stock core does not. The switch is a
        # global of the loaded core, so a previous env in this process may have left drawing off: turn it on
        # (stable-retro needs a frame when it creates the env).
        core = ctypes.CDLL(os.path.join(retro.core_path(), 'snes9x_libretro.so'))
        self._set_av = getattr(core, 'echomario_set_av', None)
        if self._set_av is not None:
            self._set_av(1, 1)
        self.rendering = True
        custom = catalog_state(level)
        self.retro = retro.make(GAME, state='YoshiIsland2' if custom else level, render_mode=None,
                                inttype=retro.data.Integrations.ALL)
        if custom:
            self.retro.unwrapped.initial_state = custom
        self.em = self.retro.unwrapped.em
        buttons = self.retro.unwrapped.buttons
        self._masks = []
        for names in _ACTION_BUTTONS:
            m = np.zeros(len(buttons), np.uint8)
            for n in names:
                m[buttons.index(n)] = 1
            self._masks.append(m)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-4.0, 4.0, shape=(self.obs_dim,), dtype=np.float32)
        self.screen_dim = self.rows * self.cols * len(self.channel_names)
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
        """(rows, cols, channels) one-hot tile grid around Mario."""
        w = self._wram() if w is None else w
        m = self.mario(w)
        feet_ty = (m['y'] + 24) // 16
        mx_t = (m['x'] + 8) // 16
        top, left = feet_ty - self.rows_above, mx_t - self.cols_behind
        tys = np.arange(top, top + self.rows)
        txs = np.arange(left, left + self.cols)
        g = np.zeros(self.screen_shape, np.float32)
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
            rr = (sy + 8) // 16 - top
            cc = (sx + 8) // 16 - left
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                g[rr, cc, ch] = 1.0
        return g

    def _tiles(self, w: np.ndarray, ty: np.ndarray, tx: np.ndarray, layer2: bool) -> np.ndarray:
        """Map16 tile numbers at level tile coordinates (ty, tx); -1 outside the level."""
        n_screens = max(1, int(w[0x5D]))
        inside = (ty >= 0) & (ty < 27) & (tx >= 0) & (tx < 16 * (16 if layer2 else n_screens))
        base = 0x1B00 if layer2 else 0
        idx = np.where(inside, base + (tx // 16) * 0x1B0 + ty * 16 + (tx % 16), 0)
        t = w[0xC800 + idx].astype(np.int32) | (w[0x1C800 + idx].astype(np.int32) << 8)
        return np.where(inside, t & 0x1FF, -1)

    def ground_below(self, w: np.ndarray | None = None) -> bool:
        """Is there anything to land on below Mario, anywhere down to the bottom of the level? Ground, ledges,
        slopes or water (measured tile classes, layer 1 and an interactive layer 2), level-wide water, or a
        platform sprite. Independent of the observation window (the teacher's pit check uses it)."""
        from echomario.envs.smw_tiles import TILE_CLASS  # noqa: PLC0415

        w = self._wram() if w is None else w
        if int(w[0x85]):
            return True
        m = self.mario(w)
        cx = (m['x'] + 8) // 16
        rows = np.arange((m['y'] + 24) // 16, 27)
        if len(rows) == 0:
            return False
        ty, tx = np.meshgrid(rows, np.arange(cx - 1, cx + 2), indexing='ij')
        layers = [self._tiles(w, ty, tx, False)]
        if int(w[0x5B]) & 0x80:
            dx = (self._u16(w, 0x1466) - self._u16(w, 0x1462)) // 16
            dy = (self._u16(w, 0x1468) - self._u16(w, 0x1464) - LAYER2_Y_OFFSET) // 16
            layers.append(self._tiles(w, ty + dy, tx + dx, True))
        table = np.asarray(TILE_CLASS, np.int8)
        for t in layers:
            cls = np.where(t >= 0, table[np.maximum(t, 0)], 0)
            if np.isin(cls, (1, 2, 3, 4)).any():
                return True
        clip = sprite_clipping()
        for i in range(12):
            if int(w[0x14C8 + i]) < 8 or int(w[0x9E + i]) not in PLATFORM_SPRITES:
                continue
            sx = int(w[0xE4 + i]) | (int(w[0x14E0 + i]) << 8)
            sy = int(w[0xD8 + i]) | (int(w[0x14D4 + i]) << 8)
            sx, sy = sx - 0x10000 if sx >= 0x8000 else sx, sy - 0x10000 if sy >= 0x8000 else sy
            k = int(w[0x1662 + i]) & 0x3F
            dx, dy, wd, _ht = ((int(clip['dx'][k]), int(clip['dy'][k]), int(clip['w'][k]), int(clip['h'][k])) if k
                               else PLATFORM_DEFAULT_BOX)
            if sx + dx - 16 <= m['x'] + 8 <= sx + dx + wd + 16 and sy + dy >= m['y']:
                return True
        return False

    def grid_v2(self, w: np.ndarray | None = None) -> np.ndarray:
        """(rows, cols, 15) grid of observation v2 (see CHANNEL_NAMES_V2)."""
        from echomario.envs.smw_tiles import TILE_CLASS  # noqa: PLC0415

        w = self._wram() if w is None else w
        m = self.mario(w)
        feet_ty = (m['y'] + 24) // 16
        mx_t = (m['x'] + 8) // 16
        top, left = feet_ty - self.rows_above, mx_t - self.cols_behind
        ty, tx = np.meshgrid(np.arange(top, top + self.rows), np.arange(left, left + self.cols), indexing='ij')
        g = np.zeros((self.rows, self.cols, len(CHANNEL_NAMES_V2)), np.float32)
        table = np.asarray(TILE_CLASS, np.int8)
        layers = [self._tiles(w, ty, tx, False)]
        if int(w[0x5B]) & 0x80:  # layer 2 interacts with Mario (offset by its scroll relative to layer 1)
            dx = (self._u16(w, 0x1466) - self._u16(w, 0x1462)) // 16
            dy = (self._u16(w, 0x1468) - self._u16(w, 0x1464) - LAYER2_Y_OFFSET) // 16
            layers.append(self._tiles(w, ty + dy, tx + dx, True))
        for t in layers:
            cls = np.where(t >= 0, table[np.maximum(t, 0)], 0)
            for c, ch in _TILE_CHANNEL_V2.items():
                g[..., ch] = np.maximum(g[..., ch], cls == c)
        if int(w[0x85]):  # level-wide water (Forest 2): Mario swims wherever there is no wall
            g[..., 3] = np.maximum(g[..., 3], (g[..., 0] == 0) & (ty >= 0) & (ty < 27))
        g[..., 5] = np.maximum(g[..., 5], ty >= 27)  # below the level: falling there is death
        g[..., 0] = np.maximum(g[..., 0], tx < 0)    # left of the level start: a wall
        clip = sprite_clipping()

        def stamp(ch, x0, y0, wd, ht, vx=0.0, vy=0.0):
            r0, r1 = y0 // 16 - top, (y0 + max(1, ht) - 1) // 16 - top
            c0, c1 = x0 // 16 - left, (x0 + max(1, wd) - 1) // 16 - left
            r0, r1, c0, c1 = max(r0, 0), min(r1, self.rows - 1), max(c0, 0), min(c1, self.cols - 1)
            if r0 > r1 or c0 > c1:
                return
            g[r0:r1 + 1, c0:c1 + 1, ch] = 1.0
            g[r0:r1 + 1, c0:c1 + 1, 13] = vx
            g[r0:r1 + 1, c0:c1 + 1, 14] = vy

        def s16(v):
            return v - 0x10000 if v >= 0x8000 else v

        for i in range(12):
            if int(w[0x14C8 + i]) < 8:
                continue
            num = int(w[0x9E + i])
            if num in HARMLESS_SPRITES_V2:
                continue
            sx = s16(int(w[0xE4 + i]) | (int(w[0x14E0 + i]) << 8))
            sy = s16(int(w[0xD8 + i]) | (int(w[0x14D4 + i]) << 8))
            k = int(w[0x1662 + i]) & 0x3F
            dx, dy, wd, ht = int(clip['dx'][k]), int(clip['dy'][k]), int(clip['w'][k]), int(clip['h'][k])
            if num in PLATFORM_SPRITES:
                ch = 9
                if k == 0:
                    dx, dy, wd, ht = PLATFORM_DEFAULT_BOX
            elif num in GOAL_SPRITES:
                ch = 12
            elif num in ITEM_SPRITES or num in YOSHI_SPRITES or int(w[0x14C8 + i]) in (9, 11):
                ch = 10  # status 9 / 11: lying still / carried (P-switch, stunned shell, key...): safe to touch
            else:
                ch = 7 if int(w[0x1656 + i]) & 0x10 else 8  # tweaker $1656 bit 4: can be jumped on
            stamp(ch, sx + dx, sy + dy, wd, ht, self._s8(w[0xB6 + i]) / 48.0, self._s8(w[0xAA + i]) / 64.0)
        for i in range(10):  # extended sprites: projectiles
            if int(w[0x170B + i]) not in PROJECTILE_EXT_SPRITES:
                continue
            ex = s16(int(w[0x171F + i]) | (int(w[0x1733 + i]) << 8))
            ey = s16(int(w[0x1715 + i]) | (int(w[0x1729 + i]) << 8))
            stamp(11, ex, ey, 8, 8, self._s8(w[0x1747 + i]) / 48.0, self._s8(w[0x173D + i]) / 64.0)
        for i in range(20):  # cluster sprites (Boo rings, ...); type 1 is the bonus-game 1-up
            kind = int(w[0x1892 + i])
            if kind in (0, 1):
                continue
            cx_ = s16(int(w[0x1E16 + i]) | (int(w[0x1E3E + i]) << 8))
            cy_ = s16(int(w[0x1E02 + i]) | (int(w[0x1E2A + i]) << 8))
            stamp(8, cx_, cy_, 16, 16)
        return g

    def _obs(self, w: np.ndarray) -> np.ndarray:
        m = self.mario(w)
        if self.obs_version == 'v2':
            p = m['power']
            scalars = np.array([
                m['vx'] / 48.0, m['vy'] / 64.0, float(m['on_ground']), ((m['x'] + 8) % 16) / 16.0, ((m['y'] + 24) % 16) / 16.0,
                float(p == 1), float(p == 2), float(p == 3), float(bool(w[0x75])), float(bool(w[0x74])), float(bool(w[0x73])),
                float(bool(w[0x140D])), float(bool(w[0x148F])), float(bool(w[0x1490])), float(bool(w[0x1497])),
                float(bool(w[0x187A])), float(bool(w[0x76])), int(w[0x13E4]) / 112.0, float(bool(w[0x1471]))], np.float32)
            return np.concatenate([self.grid_v2(w).reshape(-1), scalars])
        scalars = np.array([m['vx'] / 48.0, m['vy'] / 64.0, float(m['on_ground']), float(m['power'] > 0)], np.float32)
        return np.concatenate([self.grid(w).reshape(-1), scalars])

    # ----------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        """options: 'level'; 'prefix' (actions played first, e.g. to a mid-level start; progress is then measured
        from there: 'progress' = share of the remaining way to the goal); 'end_fraction' (the episode ends,
        truncated with info['reached_end'], once Mario has covered this share of the level, e.g. 0.5)."""
        super().reset(seed=seed)
        options = options or {}
        level = options.get('level', self.level)
        if level != self.level:
            custom = catalog_state(level)
            if custom:
                self.retro.unwrapped.initial_state = custom
            else:
                self.retro.load_state(level, inttype=__import__('stable_retro').data.Integrations.ALL)
            self.level = level
        self._frame, _ = self.retro.reset()
        w = self._wram()
        m = self.mario(w)
        self.goal_x = 256 * max(1, m['screens']) - 48
        self.end_fraction = options.get('end_fraction')
        self.start_x = 0
        self.track = _Track(max_x=m['x'], coins=m['coins'])
        self._power = m['power']
        self._spawns, self._spawn_next = [], 0
        if options.get('mutate') is not None and int(options['mutate']) >= 0:
            self._spawns = mutate_level(self, int(options['mutate']))
            w = self._wram()
        for a in options.get('prefix', ()):
            self.emulate(int(a))
        if options.get('prefix'):
            w = self._wram()
            m = self.mario(w)
            self.start_x = m['x']
            self.track = _Track(max_x=m['x'], coins=m['coins'])
            self._power = m['power']
        return self._obs(w), self._info(m, False, False)

    def emulate(self, action: int) -> None:
        """FRAME_SKIP frames with the action's buttons, without building an observation."""
        mask = self._masks[int(action)]
        spawns = getattr(self, '_spawns', None)
        if spawns and self._spawn_next < len(spawns):
            self._fire_spawns()
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
        end = getattr(self, 'end_fraction', None)
        truncated = bool(not terminated and (t.steps >= self.max_steps or t.since_progress >= self.stagnation_steps
                                             or (end and t.max_x / self.goal_x >= end)))
        if died:
            reward -= 20.0
        if goal:
            reward += 50.0
        obs = self._obs(w) if compute_obs else None
        return obs, reward, terminated, truncated, self._info(m, died, hurt)

    def _info(self, m: dict, died: bool, hurt: bool) -> dict:
        t = self.track
        start = getattr(self, 'start_x', 0)
        progress = min(1.0, max(0.0, (t.max_x - start) / max(1, self.goal_x - start)))
        return {
            'x': m['x'], 'max_x': t.max_x, 'progress': progress, 'level_progress': min(1.0, t.max_x / self.goal_x),
            'reached_goal': m['goal'], 'reached_end': bool(getattr(self, 'end_fraction', None)
                                                           and t.max_x / self.goal_x >= self.end_fraction),
            'died': died, 'hurt': hurt, 'coins': m['coins'], 'power': m['power'], 'steps': t.steps,
            'stagnated': t.since_progress >= self.stagnation_steps, 'level': self.level,
        }

    @property
    def fast_core(self) -> bool:
        return self._set_av is not None

    def set_render(self, on: bool) -> None:
        """Draw the picture (and mix audio) or not. Game logic and RAM are identical either way;
        not drawing is ~1.5x faster. A no-op with the stock core."""
        if self._set_av is not None and bool(on) != self.rendering:
            self._set_av(int(bool(on)), int(bool(on)))
        self.rendering = bool(on)

    def render(self):
        return self.retro.unwrapped.em.get_screen()

    def close(self):
        self.set_render(True)  # the switch outlives this env (see __init__)
        # stable-retro allows one emulator per process; drop our reference so it is really freed
        self.em = None
        self.retro.close()

    # ----------------------------------------------------------------- planner support
    def _fire_spawns(self) -> None:
        """Spawns the mutation's enemies (see mutate_level) once Mario comes within reach of them."""
        w = self._wram()
        mx = self._u16(w, 0x94)
        mem = self.retro.unwrapped.data.memory
        while self._spawn_next < len(self._spawns) and mx >= self._spawns[self._spawn_next][0]:
            _, num, x, y = self._spawns[self._spawn_next]
            self._spawn_next += 1
            free = [i for i in range(10) if int(w[0x14C8 + i]) == 0]
            if not free:
                continue
            i = free[-1]
            for a, v in ((0x9E, num), (0x14C8, 1), (0xE4, x & 0xFF), (0x14E0, x >> 8), (0xD8, y & 0xFF),
                         (0x14D4, y >> 8), (0x161A, 0xFF)):
                mem.assign(0x7E0000 + a + i, '|u1', v)

    def get_sim_state(self) -> tuple:
        t = self.track
        return (self.em.get_state(), _Track(t.max_x, t.steps, t.since_progress, t.coins, t.stomps, t.hurt), self._power,
                getattr(self, '_spawn_next', 0))

    def set_sim_state(self, s: tuple) -> None:
        self.em.set_state(s[0])
        t = s[1]
        self.track = _Track(t.max_x, t.steps, t.since_progress, t.coins, t.stomps, t.hurt)
        self._power = s[2]
        if len(s) > 3:
            self._spawn_next = s[3]


def grid_kwargs(config: dict) -> dict:
    """SMWEnv observation arguments stored in a run config (``env.grid``, ``env.obs``); defaults for older runs."""
    g = dict(config.get('env', {}).get('grid', {}))
    return {'rows': int(g.get('rows', ROWS)), 'cols': int(g.get('cols', COLS)),
            'rows_above': int(g.get('rows_above', ROWS_ABOVE)), 'cols_behind': int(g.get('cols_behind', COLS_BEHIND)),
            'obs': str(config.get('env', {}).get('obs', 'v1'))}


def smw_screen_shape(config: dict | None = None) -> tuple[int, int, int]:
    g = grid_kwargs(config or {})
    return g['rows'], g['cols'], len(channel_names(g['obs']))


def smw_obs_dim(config: dict | None = None) -> int:
    r, c, ch = smw_screen_shape(config)
    return r * c * ch + n_scalars(grid_kwargs(config or {})['obs'])


MUTATION_ENEMIES = (0x0F, 0x04, 0x05, 0x0D, 0x11, 0x13)  # Galoomba, green / red Koopa, Bob-omb, Buzzy Beetle, Spiny
EMPTY_TILE, BLOCK_TILE = 0x025, 0x130


def mutate_level(env: 'SMWEnv', seed: int) -> list[tuple[int, int, int, int]]:
    """Automatic level variation: a seeded set of edits of the loaded level's tile map in RAM (pits cut into flat
    ground, block walls and steps), plus enemies that spawn when Mario comes near (returned as (trigger x, sprite,
    x, y) for SMWEnv._fire_spawns). The game's own physics handles everything; the start (first 12 tile columns
    after Mario) and the last 16 columns are left alone. Horizontal levels only (vertical ones are not changed)."""
    from echomario.envs.smw_tiles import TILE_CLASS  # noqa: PLC0415

    w = env._wram()
    if int(w[0x5B]) & 1:
        return []
    rng = np.random.default_rng(seed)
    screens = max(1, int(w[0x5D]))
    width = 16 * screens
    mem = env.retro.unwrapped.data.memory
    table = np.asarray(TILE_CLASS, np.int8)
    ty, tx = np.meshgrid(np.arange(27), np.arange(width), indexing='ij')
    cls = table[np.maximum(env._tiles(w, ty, tx, False), 0)]
    standable = (cls == 1) | (cls == 2)
    ground = np.where(standable.any(0), standable.argmax(0), -1)  # topmost standable row per column
    air_free = np.array([ground[c] >= 4 and not (cls[ground[c] - 4:ground[c], c] != 0).any() for c in range(width)])
    start = (env.mario(w)['x'] + 8) // 16 + 12
    flat = [c for c in range(start, width - 16) if ground[c] >= 8 and air_free[c]]
    flat_set = set(flat)
    used: set[int] = set()

    def put(c, r, t):
        idx = (c // 16) * 0x1B0 + r * 16 + (c % 16)
        mem.assign(0x7EC800 + idx, '|u1', t & 0xFF)
        mem.assign(0x7FC800 + idx, '|u1', t >> 8)

    def stretch(c, n):
        cols = range(c - 1, c + n + 1)
        return all(x in flat_set and x not in used and ground[x] == ground[c] for x in cols)

    for kind, count in (('pit', int(rng.integers(screens // 5, screens // 3 + 2))),
                        ('wall', int(rng.integers(screens // 5, screens // 3 + 2)))):
        for _ in range(count):
            for _try in range(20):
                if not flat:
                    break
                c = int(rng.choice(flat))
                n = int(rng.integers(2, 5)) if kind == 'pit' else int(rng.integers(1, 3))
                if not stretch(c, n):
                    continue
                g = int(ground[c])
                if kind == 'pit':
                    for x in range(c, c + n):
                        for r in range(g, 27):
                            put(x, r, EMPTY_TILE)
                else:
                    h = int(rng.integers(1, 4))
                    for x in range(c, c + n):
                        for r in range(g - h, g):
                            put(x, r, BLOCK_TILE)
                used.update(range(c - 3, c + n + 3))
                break
    spawns = []
    for _ in range(int(rng.integers(screens // 3, screens // 2 + 2))):
        cand = [c for c in flat if c not in used]
        if not cand:
            break
        c = int(rng.choice(cand))
        spawns.append(((c - 11) * 16, int(rng.choice(MUTATION_ENEMIES)), c * 16, (int(ground[c]) - 1) * 16))
        used.update(range(c - 2, c + 3))
    return sorted(spawns)


def replay_observations(log: dict, **env_kwargs) -> np.ndarray:
    """Observations of a logged episode (smw_imitation.py --save-data), one per executed step, in any encoding
    (``obs``, grid size). The emulator is deterministic, so replaying the logged actions reproduces every state;
    a rewind (DAgger rescue) shows up as the step number jumping back and restores the state saved there."""
    steps, actions = np.asarray(log['steps']), np.asarray(log['actions'])
    jumps = {int(steps[p]) for p in range(1, len(steps)) if steps[p] != steps[p - 1] + 1}
    env = SMWEnv(str(log['level']), max_steps=10 ** 9, stagnation_steps=10 ** 9, **env_kwargs)
    try:
        start = {}
        if len(log.get('prefix', ())):
            start['prefix'] = [int(a) for a in log['prefix']]
        if float(log.get('end_fraction', 0.0) or 0.0):
            start['end_fraction'] = float(log['end_fraction'])
        if int(log.get('mutate', -1)) >= 0:
            start['mutate'] = int(log['mutate'])
        obs, _ = env.reset(options=start)
        env.set_render(False)
        for _ in range(int(log['noop'])):
            obs, *_ = env.step(0)
        saved: dict[int, tuple] = {}
        out = np.zeros((len(steps), env.obs_dim), np.float32)
        for p, (k, a) in enumerate(zip(steps.tolist(), actions.tolist())):
            if p and k != steps[p - 1] + 1:
                state, obs = saved[k]
                env.set_sim_state(state)
            elif k in jumps and k not in saved:
                saved[k] = (env.get_sim_state(), obs.copy())
            out[p] = obs
            obs, *_ = env.step(a)
        return out
    finally:
        env.close()
