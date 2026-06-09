#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from echomario.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, default='configs/curriculum_laptop')
    parser.add_argument('--start-checkpoint', type=str, default=None)
    parser.add_argument('--start-stage', type=str, default=None)
    parser.add_argument('--keep-going-on-fail', action='store_true')
    return parser.parse_args()


def best_success_checkpoint(eval_csv_path: Path, snapshots_dir: Path) -> tuple[float, Path | None]:
    if not eval_csv_path.exists():
        return 0.0, None
    best_success = 0.0
    best_return = float('-inf')
    best_step = None
    with eval_csv_path.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            success_rate = float(row.get('eval_success_rate', 0.0))
            return_mean = float(row.get('eval_return_mean', 0.0))
            if success_rate > best_success or (
                success_rate == best_success and return_mean > best_return
            ):
                best_success = success_rate
                best_return = return_mean
                best_step = int(float(row['step']))
    if best_step is None:
        return 0.0, None
    return best_success, snapshots_dir / f'step_{best_step:09d}.pt'


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    stage_configs = sorted(config_dir.glob('stage*.yaml'))
    if not stage_configs:
        raise FileNotFoundError(f'No stage*.yaml configs found in {config_dir}')
    if args.start_stage is not None:
        stage_configs = [path for path in stage_configs if path.name >= args.start_stage]
        if not stage_configs:
            raise FileNotFoundError(f'No stage configs at or after {args.start_stage!r} in {config_dir}')

    checkpoint = args.start_checkpoint
    for stage_config in stage_configs:
        config = load_config(stage_config)
        run_dir = Path(config['logging']['run_dir'])
        threshold = float(config.get('curriculum', {}).get('success_threshold', 0.0))

        command = [sys.executable, 'scripts/train.py', '--config', str(stage_config)]
        if checkpoint is not None:
            command.extend(['--checkpoint', checkpoint])

        print(f'\n=== curriculum stage: {stage_config} ===')
        if checkpoint is not None:
            print(f'warm-start checkpoint: {checkpoint}')
        print(f'success threshold: {threshold:.3f}')
        subprocess.run(command, check=True)

        success_rate, stage_checkpoint = best_success_checkpoint(
            run_dir / 'eval_history.csv',
            run_dir / 'snapshots',
        )
        checkpoint = str(stage_checkpoint or run_dir / 'best.pt')
        print(
            f'stage complete: best_success_rate={success_rate:.3f}, '
            f'next_checkpoint={checkpoint}'
        )

        if success_rate < threshold:
            message = (
                f'Stage {stage_config.name} did not meet threshold '
                f'{threshold:.3f}; best was {success_rate:.3f}.'
            )
            if args.keep_going_on_fail:
                print(f'WARNING: {message}')
            else:
                raise SystemExit(message)


if __name__ == '__main__':
    main()
