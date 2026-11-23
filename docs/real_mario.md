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

### Results (7 training levels, 7 held-out levels, best of 8 DAgger rounds)

Mean progress (fraction of the level) on the training levels and on held-out levels:

| | training levels | held-out levels | levels finished |
|---|---:|---:|---:|
| teacher (look-ahead planner) | ~0.80 | – | 4 of 7 training |
| reservoir student, plain DAgger (3 rounds) | 0.28 | 0.09 | Yoshi's Island 2 |
| reservoir student, DAgger with rescues (best round) | 0.40 | 0.09 | Yoshi's Island 2 |
| same, taller 22 × 24 view (see below) | **0.48** | 0.09 | Yoshi's Island 2 |
| bunny-hop (constant run + jump) | 0.35 | 0.12 | Yoshi's Island 2 |
| hold right + run | 0.09 | 0.08 | none |

Per training level (student with rescues vs bunny-hop): Yoshi's Island 1 0.63 / 0.62,
Yoshi's Island 2 finished / finished, Donut Plains 1 0.32 / 0.28, Forest of Illusion 1 0.08 / 0.08,
Donut Plains 5 0.36 / 0.14, Yoshi's Island 4 0.20 / 0.28, Donut Plains 4 0.26 / 0.09.

What this says:

- **The reservoir does play the real game**, from RAM-derived tiles, with only a linear readout
  trained and no reward. It runs, times jumps over gaps and enemies, and finishes Yoshi's Island 2.
- **But on real SMW it is only a little better than the bunny-hop floor**, and only on levels it
  was trained on. It clearly beats bunny-hopping on Donut Plains 4 and 5, ties on the rest, and does
  not transfer to unseen levels. On the toy game the same recipe reached 55 % of unseen levels
  against a 40 % floor.
- **Data, not just the readout, was the first bottleneck.** Plain DAgger plateaued at 0.28 because
  the student never got deep into a level. Rescues tripled the labels per round (16k after 8 rounds
  vs ~4k) and lifted training progress. The agreement with the teacher's choice stays near 0.45,
  and progress moves up and down between rounds by ±0.1 (7 levels, one episode each).
- **The teacher is the second bottleneck.** It finishes 4 of the 14 levels it was run on. Its
  1-second look-ahead cannot plan around moving platforms or walls of turn blocks.

**A taller view helps.** In a high jump Mario rises out of the 16-row window and the reservoir
loses sight of the ground it has to land on. With a 22 × 24 window (12 rows above the feet, 6
columns behind; `--grid 22,24,12,6`, 2644 inputs) and everything else unchanged:

| | best round | mean of last 3 rounds | held-out |
|---|---:|---:|---:|
| 16 × 20 window, rescues | 0.40 | 0.32 | 0.09 |
| 22 × 24 window, rescues | **0.48** | **0.44** | 0.09 |

Per training level in the best round: Yoshi's Island 1 0.43, Yoshi's Island 2 finished, Donut Plains 1
0.60, Forest of Illusion 1 0.30, Donut Plains 5 0.31, Yoshi's Island 4 0.61, Donut Plains 4 0.17.
That is now clearly above bunny-hopping (0.35) on the training levels. Held-out levels do not move:
the student learns these levels, not yet the game.

## Version 2: corrected input, better teacher, a cleaner test

The results above use observation v1 and teacher v1. Checking them turned up enough problems that the setup
was rebuilt (`--obs v2 --teacher v2`); v1 stays available so the runs above reproduce.

**Observation v2** (`SMWEnv(obs='v2')`, 22 × 28 window, 9259 inputs):
- Tiles are classed by their *measured* behaviour. `scripts/smw_tile_probe.py` wrote each of the 512 Map16
  tiles into a running level next to Mario and read the game's own flags (`src/echomario/envs/smw_tiles.py`).
  The v1 rule (0x111 and up = solid) was wrong for 2689 cells of the 24 levels: 1070 passable cells shown as
  walls, 774 water cells missing, 637 lava/muncher cells shown as ground, 128 slopes shown as walls, 76 coins.
- Water on an interactive layer 2 (Yoshi's Island 4) and level-wide water (Forest 2) are read too: the water
  channel matches the game's in-water flag in 99.6 % (YI4) and 98.2 % (VD2) of frames.
- 15 channels: solid, ledge, slope, water, climbable, deadly (incl. the void below the level), coin, stompable
  enemy, harmful enemy (incl. cluster sprites such as Boo rings), platform sprite, item (incl. carryable
  sprites), projectile, goal tape, sprite x / y speed. Sprites cover their hitbox from the ROM's clipping table.
- 19 numbers of Mario's state (speed, on ground, position inside his tile, power-up, swimming, climbing,
  ducking, spin jump, carrying, star, invulnerable, Yoshi, facing, P-meter, standing on a sprite).

**Teacher v2**: v1's 1-second scores plus a 3-second safety check that vetoes an action when every 3-second
plan from it dies. Ranking actions by 3-second scores directly made the teacher procrastinate. Its pit check
looks straight down to the level bottom (`SMWEnv.ground_below`); v1 looked into the student's observation
window, so the teacher's decisions depended on the window size. A patience rule stops it from waiting forever.
Alone on 7 levels it finishes 5 (v1: 4).

**Speed**: `scripts/build_snes_core.sh` builds stable-retro's snes9x with a switch to skip drawing; rollouts
are 1.55× faster with identical RAM, and macros share their common prefixes: 1.65× per teacher decision.

**Replay logs**: `--save-data` stores every labelled episode as actions plus rewinds; any observation
encoding is rebuilt from them without the teacher (`replay_observations`, verified bit-exact).

**A/B/C split** (`--split-file`, `runs/smw/make_split.py`): all 24 levels, sorted by bunny-hop progress and
dealt into three equally hard groups. A (9 levels) trained in full; B (7) first half trained, second half tested
from a mid-level state the teacher reached (`scripts/smw_midpoints.py`); C (8) never seen.

### Results (6 DAgger rounds, 20 k labels, 3 start delays per level)

| | best round | final round | bunny-hop |
|---|---:|---:|---:|
| A: full levels, trained | 0.29 ± 0.06 | 0.22 ± 0.05 | 0.22 |
| B: first halves, trained | 0.20 ± 0.03 | 0.18 ± 0.04 | 0.17 |
| B: second halves, not trained | 0.20 ± 0.02 | 0.21 ± 0.04 | 0.20 |
| C: unseen levels | 0.09 ± 0.01 | 0.08 ± 0.01 | 0.13 |

The student is at best a little above bunny-hopping on what it was trained on, at the bunny-hop level on the
untrained second halves, and below it on unseen levels: it learns these levels, not the game.

**Which reservoir generalizes?** `scripts/smw_cv_lab.py` fits readouts on 3 of 4 folds of levels and scores the
held-out levels against the teacher's labels (quality of the chosen action on the 0..1 target scale).

| features | held-out | fitted levels |
|---|---:|---:|
| constant action (bias only) | 0.823 | 0.823 |
| reservoir 500 / 1000 / 3000 / 6000 | 0.839–0.840 | 0.843–0.851 |
| reservoir + raw input | 0.839 | 0.855 |
| raw input, no reservoir | 0.836 | 0.871 |

Every configuration gains under 0.02 over a constant action on held-out levels, and the best regularization is
very strong. Size, seeds and raw input change nothing there. More capacity fits the training levels better
without carrying over, so the limit is 16 training levels for a linear readout, not the reservoir.

## Version 3: more levels, honest labels, teacher v3 (2026-11-22)

**More levels.** `scripts/smw_level_catalog.py` loads every level number through the RAM override
(`$0109` + game mode `0x0F`) and keeps the distinct layouts: 40 new playable levels (`L01` … `LFF`, states in
`~/.cache/echomario/levelstates`, not in git). `SMWEnv` also builds variants (`Level#m<seed>`: pits, block walls
and extra enemies cut into flat ground).

**The teacher's "deadly" labels were half wrong.** `scripts/smw_label_check.py` / `smw_relabel_deadly.py`
retry every deadly (moment, button) with 100 random sticky button sequences for 3 s: 45 % of v2's 15,818 deadly
labels are survivable. The fixed macros missed the escape, so "deadly" partly meant "this level confused the
search". With corrected labels, predicting danger on unseen levels improves (AUC: local features 0.60 → 0.72,
reservoir ridge 0.53 → 0.61, MLP 0.61 → 0.69; within levels 0.72–0.83 → 0.84–0.90).

**Teacher v3** (`teacher_config('v3')`): v2 plus (1) the random rescue search for every action scored deadly
(it corrects the scores only; the teacher acts on rescued actions only when nothing else survives), and
(2) plan follow-through: after 3 decisions without progress it keeps executing its last plan, which stops v2's
endless idling ("wait, then back up" always outscored "back up now"). On 93 levels both played: mean progress
0.56 → 0.62. Data: `runs/smw/v3_all`, 64 levels + 48 variants, 24 k moments (`scripts/smw_collect.py`).

**Offline transfer appears** (`scripts/smw_scaling.py`, 4 folds of 16 held-out levels, variants with their
level; q = teacher rating of the chosen button):

| training levels | constant button | reservoir ridge | gain |
|---:|---:|---:|---:|
| 1 | 0.765 | 0.738 | −0.027 |
| 4 | 0.744 | 0.799 | +0.055 |
| 16 | 0.757 | 0.826 | +0.069 |
| 48 | 0.759 | 0.846 | +0.087 |

Positive in every fold at 48 levels (+0.066 to +0.106; v2 data, 12 levels: +0.014). The MLP overfits (final
+0.017). Numbers after the variant replay fix (see the DAgger section; before: +0.072 at 48 levels).

**…but not in play** (`scripts/smw_closed_loop.py`: the 48-level ridge student plays the 16 held-out levels
alone, 2 start delays):

| policy | mean progress | died |
|---|---:|---:|
| ridge student (alpha 10 / alpha 1) | 0.171 / 0.175 | 72 % / 70 % |
| constant button (4-step cadence) | 0.185 | 70 % |
| bunny-hop | 0.239 | 87 % |

Behavioural cloning on the teacher's states does not carry over to the student's own states: one wrong
decision puts it where no label exists. The next step is DAgger with teacher v3 on many levels.

**The reservoir as a critic**: a second readout fitted to the teacher's raw scores per action explains 44 % of
their variance (state value: 50 %); the raw input explains 60 % (67 %). With the rich v2 input the random
3000-neuron projection loses information. Memory settings barely matter.

## DAgger with teacher v3, and why the student still dies (2026-11-23)

`scripts/smw_dagger.py`: level fold 0 of the scaling / closed-loop tests (16 unseen levels, 2 start delays);
the student trains on the other 48 base levels and their variants (84 episodes per round). Round 0 is the
offline student (teacher data only); each round the student plays every training level while teacher v3
labels every decision (teacher share 0.3 in round 1, then 0; up to 3 rescue rewinds after a death; random start
delay 0..40), and the ridge readout is refitted on all labels. Alpha is chosen each round by closed-loop
progress on the training levels; the unseen levels never choose anything. About 3 h per round on 7 cores.

| round | labels | alpha | unseen levels: progress | died | training levels: progress | died |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (offline) | 18 963 | 0.01 | 0.132 ± 0.021 | 94 % | 0.209 | 86 % |
| 1 | 34 049 | 0.001 | 0.169 ± 0.035 | 91 % | 0.228 | 95 % |
| 2 | 48 189 | 0.001 | 0.137 ± 0.022 | 94 % | 0.233 | 86 % |
| 3 | 61 886 | 0.001 | 0.156 ± 0.028 | 91 % | 0.237 | 83 % |
| 4 | 75 782 | 0.001 | 0.153 ± 0.031 | 91 % | 0.256 | 87 % |

Fold 0 references (`smw_closed_loop.py`): constant button 0.134, bunny-hop **0.217**. (Round 1 was collected
by an alpha-10 student; at alpha 10 the readout barely reacts to new labels: only 6 of 84 training levels
changed, so alpha selection was added and the run restarted from round 1's data.)

DAgger runs as intended (collection progress 0.41-0.45 with rescues, 14-15 k new labels per round), but the
student improves only slowly even on the levels it is labelled on (0.209 -> 0.256 after 4 rounds, 57 k DAgger
labels, ~9 h) and stays flat and below bunny-hop on unseen levels (0.15 ± 0.03 from round 1 on; the best
unseen number in the per-alpha log, 0.191, is a different alpha and was not selected). Per-round results:
`runs/smw/dagger_v3_f0_results.json`.

**Why** (`scripts/smw_dagger_probes.py`, `smw_label_duplicates.py`, `smw_label_noise.py`):

- *Critical states decide.* In ~10 % of decisions one button is deadly and another is safe. There the student
  picks a deadly one 12-20 % of the time; over ~150 decisions per level that is almost always a death.
- *Not capacity.* An MLP readout fits the labels far better (q 0.978 vs 0.885) and plays no better (training
  levels 0.236, unseen 0.143). Weighting critical states 10-50x drives the in-sample unsafe rate to 2-4 %
  without changing play.
- *Not the reservoir.* On new states of the same training levels (round 2, trained on everything before), every
  learner picks a deadly button on 27-35 % of critical states, the same as the safest single button (30.5 %):
  ridge or MLP, on reservoir features or on the raw tile input, weighted or not. Safety does not generalize
  beyond the exact states that were labelled.
- *The labels at borderline states are partly random.* Decisions with exactly identical input on the same level
  (`smw_label_duplicates.py`) disagree on whether a button is deadly in 20 % of button verdicts (47 % of
  repeated critical observations; mostly Mario standing still, consecutive decisions of one episode). Replaying
  both copies (`smw_label_ram_diff.py`, every replayed input checked against the stored one) shows the same
  moment with almost identical RAM, and fresh planners give both copies the same verdict (1 % differ,
  `smw_label_rescore.py`). The same state rescored with another rescue seed differs in 8.5 % of verdicts here
  (2-3 % on random critical states, `smw_label_noise.py`). So the disagreement is the teacher's random rescue
  search (and its internal history during collection), not hidden game state.

**A replay bug, found on the way:** `attach_streams` (smw_readout_lab.py) did not pass `mutate` to the replay, so
the 48 variant episodes of `v3_all` (~11 k of 24 k moments) were replayed on the unmodified base level: their
input streams and reservoir features did not match their labels (83 % of their feature rows changed after the
fix; base levels unchanged). Live DAgger labels (rounds 1+) were not affected; the offline data of round 0
and the training data of every round, the v3 scaling and closed-loop results above, and the first version of
the label checks were. Fixed in fdbefbe; the label checks above are on the fixed data, the scaling and
closed-loop tables of Version 3 are corrected (offline gain +0.087 instead of +0.072; play unchanged).

So more DAgger labels do not teach safety on new states, and the teacher's own verdicts at borderline states
are noisy. Candidate next steps: make the teacher's verdicts consistent (rescue search seeded by the state, or
more rollouts), and rerun DAgger on the fixed features.

## Videos

```bash
python scripts/smw_video.py --teacher --level YoshiIsland2 --out outputs/smw_teacher.mp4   # writes .actions.json
python scripts/smw_video.py --run runs/smw/rescue_tall --level DonutPlains1 --out outputs/smw_student.mp4
```

Each video shows the game, the tile grid the reservoir reads, all reservoir neurons as a heat map,
a 5-second trace of the 48 neurons with the largest readout weights, and the readout's score for
every button combination.
