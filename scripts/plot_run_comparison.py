#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ('eval_return_mean', 'Mean Return'),
    ('eval_max_x_mean', 'Mean Max X'),
    ('eval_success_rate', 'Success Rate'),
    ('eval_stagnation_rate', 'Stagnation Rate'),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='label=path/to/eval_history.csv')
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()


def load_run(spec: str) -> tuple[str, list[dict[str, float]]]:
    if '=' not in spec:
        raise ValueError(f'Invalid --run value: {spec}. Expected label=path')
    label, path_str = spec.split('=', 1)
    path = Path(path_str)
    rows: list[dict[str, float]] = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return label, rows


def main() -> None:
    args = parse_args()
    runs = [load_run(spec) for spec in args.run]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 12), sharex=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for label, rows in runs:
            steps = [row['step'] for row in rows]
            values = [row[metric] for row in rows]
            ax.plot(steps, values, label=label, linewidth=2)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)

    axes[0].legend()
    axes[-1].set_xlabel('Training Step')
    fig.suptitle('EchoMario Architecture Comparison')
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'saved comparison plot: {out_path}')


if __name__ == '__main__':
    main()
