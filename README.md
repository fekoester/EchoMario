# EchoMario

EchoMario is a research-oriented platformer RL project built around a simple question:
can a fixed recurrent reservoir support competitive control when only a lightweight policy/value head is trained?

The repository includes:
- a progressively richer Mario-like toy platformer with procedural level generation
- fixed-reservoir + PPO baselines
- direct linear-observation PPO baseline
- CNN actor-critic baseline
- evaluation, playback, plotting, and training-evolution video tooling

The current environment is an original toy world implemented in this repository. No ROMs, Nintendo assets, emulator binaries, or copyrighted game data are included.

## Project Goals

- study frozen-reservoir control in platformer-like domains
- compare reservoirs against simpler and stronger practical baselines
- make experiments reproducible with deterministic eval seeds, checkpoint sweeps, and videos
- provide a clean path toward optional external Mario backends later

## Repository Layout

```text
configs/                 Training and evaluation configs
src/echomario/           Core package
src/echomario/envs/      Toy platformer and optional Mario wrappers
src/echomario/agents/    Readout and CNN policies
src/echomario/training/  PPO, rollout, evaluation
scripts/                 Train, play, evaluate, record, compare
tests/                   Automated tests
docs/                    Architecture, legal, and visualization notes
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

For manual play with the pygame explorer:

```bash
pip install pygame
```

## Quick Start

Manual exploration of the toy world:

```bash
.venv/bin/python scripts/explore_toy_world.py --config configs/toy_platformer_continuous.yaml
```

Train the reservoir baseline:

```bash
.venv/bin/python scripts/train.py --config configs/toy_platformer_continuous.yaml
```

Train the linear direct-observation baseline:

```bash
.venv/bin/python scripts/train.py --config configs/toy_platformer_continuous_linear.yaml
```

Train the parallel CNN baseline:

```bash
.venv/bin/python scripts/train.py --config configs/toy_platformer_continuous_cnn_parallel.yaml
```

Play a trained checkpoint:

```bash
.venv/bin/python scripts/play.py --checkpoint runs/toy_platformer_continuous_cnn_parallel/best.pt
```

Evaluate checkpoints from a run:

```bash
.venv/bin/python scripts/evaluate_checkpoints.py   --checkpoint-dir runs/toy_platformer_continuous_cnn_parallel/snapshots   --csv-out outputs/cnn_parallel_eval.csv   --plot-out outputs/cnn_parallel_eval.png
```

Compare run curves:

```bash
.venv/bin/python scripts/plot_run_comparison.py   --run-dirs runs/toy_platformer_continuous_stable runs/toy_platformer_continuous_linear runs/toy_platformer_continuous_cnn_parallel   --labels reservoir linear cnn_parallel   --out outputs/run_comparison.png
```

## Research Setup

The main experimental variants currently supported are:
- `reservoir.type: sparse_esn` with a trainable readout policy/value head
- `reservoir.type: none` for direct-observation baselines
- `agent.model: cnn` for a spatial actor-critic baseline over the visible tile screen

The toy platformer supports:
- randomized gaps and bridge gaps
- multi-floor platforms
- moving enemies
- question blocks and spawned items
- coins and score events
- seeded train/eval generation
- full-screen tile observations and engineered-feature observations

## Reproducibility

- configs define deterministic eval seed ranges
- runs write `eval_history.csv` and `eval_history.jsonl`
- best checkpoints are saved automatically
- tests cover env behavior, rollout/runtime paths, policy shapes, and reservoir basics

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

## Documentation

- [Architecture](docs/architecture.md)
- [Visualization](docs/visualization.md)
- [Legal / Asset Policy](docs/legal.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
