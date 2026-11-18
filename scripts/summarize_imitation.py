#!/usr/bin/env python
"""
Summarises imitation runs (``results.json`` from train_imitation.py) as a markdown table and a plot.

    python scripts/summarize_imitation.py runs/imitation/* --out docs/media/imitation_results.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', default=None, help='optional PNG with test success per DAgger iteration')
    args = p.parse_args()

    rows = []
    for run in args.runs:
        path = Path(run) / 'results.json'
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        rows.append((Path(run).name, res))

    print('| run | features | teacher | best test success | best test progress | samples | alpha |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for name, res in rows:
        best = res.get('best', {})
        test = best.get('test', {})
        its = res.get('iterations', [])
        teacher = res.get('teacher_test', {}).get('success_rate')
        print(
            f"| {name} | {res['mode']} | {'' if teacher is None else f'{teacher:.0%}'} "
            f"| {test.get('success_rate', float('nan')):.0%} | {test.get('progress_mean', float('nan')):.2f} "
            f"| {its[-1]['samples'] if its else 0} | {best.get('alpha', '')} |"
        )

    if args.out:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, res in rows:
            its = res.get('iterations', [])
            xs = [r['iteration'] for r in its]
            axes[0].plot(xs, [r['test']['success_rate'] for r in its], marker='o', label=name)
            axes[1].plot(xs, [r['test']['progress_mean'] for r in its], marker='o', label=name)
        axes[0].set_title('test levels finished')
        axes[1].set_title('mean progress through level')
        for ax in axes:
            ax.set_xlabel('DAgger iteration')
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7, loc='lower right')
        fig.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
