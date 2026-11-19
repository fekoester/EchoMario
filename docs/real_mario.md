# Real Super Mario World (SNES)

Can the frozen reservoir play the real game, not just the toy platformer? This page covers
running Super Mario World in an emulator, turning it into input for the reservoir, a look-ahead
teacher built on emulator save-states, and what a reservoir student learned.

## Setup

No ROM is included or downloaded. You need a dump of a cartridge you own.

```bash
pip install -e ".[smw]"                     # stable-retro (emulator + game integrations)
# headerless US ROM, SHA-1 6b47bb75d16514b6a476aa0c73a683a2a4c18765.
# A .smc with a 512-byte copier header: tail -c 524288 game.smc > smw.sfc
python -m stable_retro.import <dir containing smw.sfc>
pytest tests/test_smw.py                    # skipped when the ROM is missing
```

stable-retro ships save-states at the start of 25 levels. The 19 horizontal, non-water levels
are used: 13 for training (`TRAIN_LEVELS`) and 6 held out (`TEST_LEVELS`) in
`src/echomario/envs/smw_env.py`.

## From the game to reservoir input

`SMWEnv` reads the game's RAM after every step (2 frames, 30 decisions per second):

- **Tile grid around Mario** (16 rows × 20 columns of 16-pixel tiles, Mario's feet at row 10,
  column 4) from the level's Map16 tile table (`$7EC800`/`$7FC800`, screen-major layout). Tiles map
  to channels using vanilla SMW tile behaviour. 0x100–0x110 are ledges you can stand on and jump
  through from below (*platform*). 0x111–0x1FF are blocks, pipes and slopes (*solid*). 0x2B–0x2D
  are coins.
- **Sprites** from the 12 sprite slots (status `$14C8`, number `$9E`, position `$E4/$14E0`,
  `$D8/$14D4`). Power-ups count as *item*, everything else (except Yoshi and the goal tape) as
  *enemy*.
- **Four scalars**: horizontal and vertical speed, on ground, powered up. 1604 inputs in total,
  laid out like the toy platformer's screen, so the reservoir's local receptive fields work unchanged.

Progress is Mario's x position (`$94`). Death is the death animation (`$71 == 9`) or falling
below the level. The goal is the end-of-level timer (`$1493`). Message boxes (`$1426`) freeze the
game until a button is pressed, so the env closes them itself.

Things that cost time to find:
- stable-retro's `get_ram()` puts the 8 KB `$0000` mirror and 2 KB SRAM before the 128 KB WRAM, so
  WRAM starts at offset 10240.
- Only one emulator per process: training workers use `maxtasksperchild=1`, and `close()` must drop
  every reference to the emulator.
- Restoring a save-state blanks the frame buffer until the next frame. The video grabs the screen
  before the teacher plans.

## The teacher

`SMWPlanner` (`src/echomario/experts/smw_planner.py`) saves the emulator state, tries each of 6
first actions for 4 steps, then 5 continuation macros for 30 steps: run, hop (press jump,
release, press again; SMW only starts a jump on a fresh press), long jump, wait-then-hop and
run-up (back up, then run and jump for wide gaps). It picks the action with the best continuation.
The score is pixels of new progress, −2000 for death, −300 for losing a power-up and +3000 for
the goal. A plan that ends falling below its start height with no ground in view counts as death.
Without that check the teacher happily ran into pits whose bottom was beyond the look-ahead.

It plays at about 3 game steps per second per CPU core (≈600 emulated frames per decision).

| level | teacher | hold run | bunny-hop |
|---|---:|---:|---:|
| Yoshi's Island 1 | finished | 2 % | 62 % |
| Yoshi's Island 2 | finished | 26 % | finished |
| Donut Plains 1 | finished | 8 % | 28 % |
| Forest of Illusion 1 | finished | 4 % | 8 % |
| Donut Plains 5 | 68 % | 6 % | 14 % |
| Yoshi's Island 4 | 60 % | 13 % | 28 % |
| Donut Plains 4 | 48 % | 6 % | 9 % |
| Vanilla Dome 1 | 14 % (stuck at a wall of blocks) | 4 % | 4 % |
| Yoshi's Island 3 | 9 % (moving platforms) | 7 % | 8 % |

Its limits are those of a 1-second look-ahead with fixed macros. Moving platforms, walls of
turn blocks and anything needing a detour defeat it. Letting rollouts run on until Mario lands
(`settle_steps`) made it twice as slow and no better, so that option is off.

## The student

`scripts/smw_imitation.py` uses the same recipe as on the toy game:
- A frozen 3000-neuron reservoir with local receptive fields and multi-timescale leaks.
- A ridge-regression readout on the teacher's q targets (`q_clip` 32 pixels).
- DAgger rounds.

The student decides every 4 steps and holds its choice, like the teacher. Switching every step
cut jumps short, because SMW's jump height depends on how long the button is held.

**DAgger rescues.** In plain DAgger the student dies early on most levels, so each round adds
labels only from the first tenth of every level (~500 new samples per round). With `--rescues N`,
a student death in a labelled episode rewinds 2 s using a save-state (reservoir state included).
The teacher then drives for 80 steps and hands control back. One episode now covers most of
the level.

RESULTS_PLACEHOLDER

## Videos

```bash
python scripts/smw_video.py --teacher --level YoshiIsland2 --out outputs/smw_teacher.mp4   # writes .actions.json
python scripts/smw_video.py --run runs/smw/rescue --level DonutPlains1 --out outputs/smw_student.mp4
```

Each video shows the game, the tile grid the reservoir reads, all reservoir neurons as a heat map,
a 5-second trace of the 48 neurons with the largest readout weights, and the readout's score for
every button combination.
