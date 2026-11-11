#!/usr/bin/env python
"""
Grid sweep over config values (e.g. reservoir hyperparameters) with multiple seeds.

Every run is a normal ``scripts/train.py`` run in its own directory, so runs can be resumed: finished
runs (with a ``latest.pt``) are skipped. Afterwards the eval histories are aggregated into
``results.csv`` (one row per run) and ``summary.md`` (mean ± std over seeds per grid point).

Example::

    python scripts/sweep.py --config configs/toy_platformer_curriculum_stage1.yaml \\
        --grid reservoir.size=200,400,800 --grid reservoir.spectral_radius=0.8,0.95 \\
        --seeds 0,1,2 --total-steps 200000 --out-dir sweeps/stage1_reservoir

    python scripts/sweep.py ... --summarize-only   # just rebuild the tables
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def parse_grid(items: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for item in items:
        key, _, values = item.partition("=")
        if not key or not values:
            raise ValueError(f"--grid expects key=v1,v2,..., got {item!r}")
        grid[key.strip()] = [yaml.safe_load(v) for v in values.split(",")]
    return grid


def set_dotted(config: dict, dotted_key: str, value) -> None:
    node = config
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def run_name(params: dict, seed: int) -> str:
    parts = [f"{key.split('.')[-1]}={value}" for key, value in params.items()]
    return "__".join(parts + [f"seed={seed}"]).replace("/", "_")


def read_eval_history(run_dir: Path) -> list[dict]:
    path = run_dir / "eval_history.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def summarize(runs: list[tuple[dict, int, Path]], out_dir: Path, grid_keys: list[str]) -> None:
    rows = []
    for params, seed, run_dir in runs:
        history = read_eval_history(run_dir)
        if not history:
            continue
        rows.append({
            **params,
            "seed": seed,
            "steps": int(history[-1]["step"]),
            "best_eval_return": max(h["eval_return_mean"] for h in history),
            "final_eval_return": history[-1]["eval_return_mean"],
            "best_success_rate": max(h.get("eval_success_rate", 0.0) for h in history),
            "final_max_x": history[-1].get("eval_max_x_mean", 0.0),
            "run_dir": str(run_dir),
        })
    if not rows:
        print("no finished runs to summarize yet")
        return

    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["best_eval_return", "final_eval_return", "best_success_rate", "final_max_x"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in grid_keys), []).append(row)

    def fmt(values: list[float]) -> str:
        if len(values) == 1:
            return f"{values[0]:.2f}"
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

    header = [k.split(".")[-1] for k in grid_keys] + ["seeds"] + metrics
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    ranked = sorted(groups.items(), key=lambda kv: -statistics.mean(r["best_eval_return"] for r in kv[1]))
    for key, group in ranked:
        cells = [str(v) for v in key] + [str(len(group))]
        cells += [fmt([r[m] for r in group]) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", action="append", default=[], help="dotted.key=v1,v2,... (repeatable)")
    parser.add_argument("--seeds", default="0", help="comma-separated project seeds")
    parser.add_argument("--total-steps", type=int, default=None, help="override training.total_steps")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only print the planned runs")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    grid = parse_grid(args.grid)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[dict, int, Path]] = []
    for values in itertools.product(*grid.values()) if grid else [()]:
        params = dict(zip(grid.keys(), values, strict=True))
        for seed in seeds:
            runs.append((params, seed, out_dir / run_name(params, seed)))

    print(f"{len(runs)} runs ({len(runs) // len(seeds)} grid points x {len(seeds)} seeds)")
    if not args.summarize_only:
        for i, (params, seed, run_dir) in enumerate(runs, 1):
            if (run_dir / "latest.pt").exists():
                print(f"[{i}/{len(runs)}] skip (done): {run_dir.name}")
                continue
            print(f"[{i}/{len(runs)}] {run_dir.name}")
            if args.dry_run:
                continue
            config = copy.deepcopy(base)
            for key, value in params.items():
                set_dotted(config, key, value)
            set_dotted(config, "project.seed", seed)
            set_dotted(config, "logging.run_dir", str(run_dir))
            if args.total_steps is not None:
                set_dotted(config, "training.total_steps", args.total_steps)
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)], check=True)

    summarize(runs, out_dir, list(grid.keys()))


if __name__ == "__main__":
    main()
