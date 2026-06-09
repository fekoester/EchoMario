#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from echomario.agents.factory import build_policy as build_policy_from_config
from echomario.envs.make_env import make_env
from echomario.envs.reset import EpisodeSeedManager
from echomario.reservoirs.esn import load_reservoir_from_state_dict
from echomario.training.evaluate import evaluate_policy
from echomario.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--glob', type=str, default='step_*.pt')
    parser.add_argument('--include-best', action='store_true')
    parser.add_argument('--include-latest', action='store_true')
    parser.add_argument('--include-initial', action='store_true')
    parser.add_argument('--csv-out', type=str, default='')
    parser.add_argument('--jsonl-out', type=str, default='')
    parser.add_argument('--plot-out', type=str, default='')
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r'step_(\d+)\.pt', path.name)
    if match is not None:
        return int(match.group(1))
    return -1


def collect_checkpoints(args) -> list[Path]:
    paths: list[Path] = [Path(p) for p in args.checkpoint]
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        paths.extend(sorted(ckpt_dir.glob(args.glob), key=extract_step))
        if args.include_best:
            paths.append(ckpt_dir / 'best.pt')
        if args.include_latest:
            paths.append(ckpt_dir / 'latest.pt')
        if args.include_initial:
            paths.append(ckpt_dir / 'initial.pt')

    deduped = []
    seen = set()
    for path in paths:
        if path.exists() and path not in seen:
            deduped.append(path)
            seen.add(path)
    if not deduped:
        raise FileNotFoundError('No checkpoints found')
    return deduped


def load_policy_for_checkpoint(env, ckpt):
    reservoir = load_reservoir_from_state_dict(ckpt['reservoir_state'])
    policy = build_policy_from_config(ckpt['config'], env, reservoir.cfg.size)
    policy.load_state_dict(ckpt['policy_state'])
    policy.eval()
    return reservoir, policy


def write_rows_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_jsonl(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def save_plot(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    numeric_rows = [row for row in rows if int(row['global_step']) >= 0]
    if not numeric_rows:
        return

    steps = [int(row['global_step']) for row in numeric_rows]
    returns = [float(row['mean_return']) for row in numeric_rows]
    max_x = [float(row['mean_max_x']) for row in numeric_rows]
    success = [float(row['success_rate']) for row in numeric_rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(steps, returns, marker='o')
    axes[0].set_ylabel('mean return')
    axes[1].plot(steps, max_x, marker='o')
    axes[1].set_ylabel('mean max_x')
    axes[2].plot(steps, success, marker='o')
    axes[2].set_ylabel('success rate')
    axes[2].set_xlabel('global step')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    headers = ['checkpoint', 'global_step', 'mean_return', 'std_return', 'mean_max_x', 'success_rate']
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = '  '.join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print('  '.join('-' * widths[header] for header in headers))
    for row in rows:
        print('  '.join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> None:
    args = parse_args()
    checkpoints = collect_checkpoints(args)
    rows = []

    for checkpoint_path in checkpoints:
        ckpt = load_checkpoint(checkpoint_path)
        config = ckpt['config']
        env = make_env(config)
        seed_manager = EpisodeSeedManager(config)
        reset_specs = seed_manager.evaluation_reset_specs()
        reservoir, policy = load_policy_for_checkpoint(env, ckpt)

        stats = evaluate_policy(
            env=env,
            reservoir=reservoir,
            policy=policy,
            episodes=len(reset_specs),
            reset_specs=reset_specs,
            device=torch.device(config['project'].get('device', 'cpu')),
        )
        env.close()

        rows.append(
            {
                'checkpoint': str(checkpoint_path),
                'global_step': int(ckpt.get('global_step', extract_step(checkpoint_path))),
                'mean_return': round(stats['eval_return_mean'], 6),
                'std_return': round(stats['eval_return_std'], 6),
                'mean_max_x': round(stats['eval_max_x_mean'], 6),
                'success_rate': round(stats['eval_success_rate'], 6),
                'fell_rate': round(stats['eval_fell_rate'], 6),
                'enemy_hit_rate': round(stats['eval_enemy_hit_rate'], 6),
                'stagnation_rate': round(stats['eval_stagnation_rate'], 6),
                'truncation_rate': round(stats['eval_truncation_rate'], 6),
            }
        )

    print_table(rows)

    if args.csv_out:
        write_rows_csv(Path(args.csv_out), rows)
    if args.jsonl_out:
        write_rows_jsonl(Path(args.jsonl_out), rows)
    if args.plot_out:
        save_plot(Path(args.plot_out), rows)


if __name__ == '__main__':
    main()
