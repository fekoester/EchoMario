#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from echomario.agents.factory import PolicyLike, build_policy
from echomario.envs.make_env import make_env
from echomario.envs.reset import EpisodeSeedManager
from echomario.reservoirs.esn import ReservoirLike, make_reservoir
from echomario.training.evaluate import evaluate_policy
from echomario.training.ppo import ppo_update
from echomario.training.rollout import collect_rollout_parallel
from echomario.utils.checkpoint import save_checkpoint
from echomario.utils.config import load_config
from echomario.utils.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def save_policy_checkpoint(
    *,
    path: Path,
    config: dict,
    policy: PolicyLike,
    reservoir: ReservoirLike,
    global_step: int,
    stats: dict,
) -> None:
    save_checkpoint(
        path,
        config=config,
        policy_state=policy.state_dict(),
        reservoir_state=reservoir.state_dict(),
        global_step=global_step,
        stats=stats,
    )


def append_eval_record(path_jsonl: Path, path_csv: Path, record: dict[str, float | int]) -> None:
    path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with path_jsonl.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    write_header = not path_csv.exists()
    with path_csv.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config['project'].get('seed', 42))
    set_seed(seed)

    device = torch.device(config['project'].get('device', 'cpu'))
    seed_manager = EpisodeSeedManager(config)

    num_envs = int(config['training'].get('num_envs', 1))
    if num_envs < 1:
        raise ValueError('training.num_envs must be at least 1')

    envs = [make_env(config) for _ in range(num_envs)]
    env = envs[0]
    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    reservoir.reset(batch_size=num_envs)
    policy = build_policy(config, env, reservoir.cfg.size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=float(config['agent']['lr']))

    run_dir = Path(config['logging']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshots_dir = run_dir / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    eval_jsonl_path = run_dir / 'eval_history.jsonl'
    eval_csv_path = run_dir / 'eval_history.csv'

    total_steps = int(config['training']['total_steps'])
    rollout_steps = int(config['training']['rollout_steps'])
    save_every = int(config['training'].get('save_every_steps', 25000))
    eval_every = int(config['training'].get('eval_every_steps', 25000))

    def reset_training_episode(env_idx: int) -> np.ndarray:
        reset_spec = seed_manager.training_reset_spec()
        obs, _ = envs[env_idx].reset(**reset_spec.as_kwargs())
        return obs

    obs_batch = np.stack([reset_training_episode(env_idx) for env_idx in range(num_envs)], axis=0)
    rollout_state = None

    eval_env = make_env(config)
    eval_reservoir = make_reservoir(config, input_dim=int(eval_env.observation_space.shape[0]))
    eval_reset_specs = seed_manager.evaluation_reset_specs()

    global_step = 0
    recent_returns: list[float] = []
    best_eval_return = float('-inf')
    eval_history: list[dict[str, float | int]] = []

    initial_stats = {
        'kind': 'initial_untrained',
        'best_eval_return': best_eval_return,
        'eval_history': eval_history,
    }

    save_policy_checkpoint(
        path=snapshots_dir / 'step_000000000.pt',
        config=config,
        policy=policy,
        reservoir=reservoir,
        global_step=0,
        stats=initial_stats,
    )
    save_policy_checkpoint(
        path=run_dir / 'initial.pt',
        config=config,
        policy=policy,
        reservoir=reservoir,
        global_step=0,
        stats=initial_stats,
    )

    print(f"saved initial untrained checkpoint: {snapshots_dir / 'step_000000000.pt'}")

    pbar = tqdm(total=total_steps, desc='training')
    next_eval_step = eval_every
    next_save_step = save_every

    while global_step < total_steps:
        rollout = collect_rollout_parallel(
            envs=envs,
            reservoir=reservoir,
            policy=policy,
            obs_batch=obs_batch,
            state=rollout_state,
            rollout_steps=rollout_steps,
            device=device,
            reset_fn=reset_training_episode,
        )
        obs_batch = rollout.last_obs
        rollout_state = rollout.last_state

        step_increment = int(rollout.rewards.numel())
        previous_step = global_step
        global_step += step_increment
        pbar.update(min(step_increment, total_steps - previous_step))

        stats = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            gamma=float(config['training']['gamma']),
            gae_lambda=float(config['training']['gae_lambda']),
            update_epochs=int(config['training']['update_epochs']),
            minibatch_size=int(config['training']['minibatch_size']),
            clip_eps=float(config['agent']['clip_eps']),
            value_coef=float(config['agent']['value_coef']),
            entropy_coef=float(config['agent']['entropy_coef']),
            max_grad_norm=float(config['agent']['max_grad_norm']),
        )

        recent_returns.extend(rollout.episode_returns)
        if len(recent_returns) > 50:
            recent_returns = recent_returns[-50:]

        mean_return = sum(recent_returns) / max(1, len(recent_returns))
        pbar.set_postfix(
            {
                'ret50': f'{mean_return:.2f}',
                'loss': f'{stats.loss:.3f}',
                'ent': f'{stats.entropy:.3f}',
                'best': f'{best_eval_return:.2f}',
            }
        )

        while global_step >= next_eval_step:
            eval_stats = evaluate_policy(
                env=eval_env,
                reservoir=eval_reservoir,
                policy=policy,
                episodes=len(eval_reset_specs),
                reset_specs=eval_reset_specs,
                device=device,
            )

            print(f"\nstep={global_step} eval={eval_stats}")

            eval_record = {'step': global_step, **eval_stats}
            eval_history.append(eval_record)
            append_eval_record(eval_jsonl_path, eval_csv_path, eval_record)

            snapshot_stats = {
                'kind': 'eval_snapshot',
                'recent_return_mean': mean_return,
                'best_eval_return': best_eval_return,
                'eval_stats': eval_stats,
                'eval_history': eval_history,
            }

            snapshot_path = snapshots_dir / f'step_{global_step:09d}.pt'
            save_policy_checkpoint(
                path=snapshot_path,
                config=config,
                policy=policy,
                reservoir=reservoir,
                global_step=global_step,
                stats=snapshot_stats,
            )
            print(f'saved evolution snapshot: {snapshot_path}')

            if eval_stats['eval_return_mean'] > best_eval_return:
                best_eval_return = eval_stats['eval_return_mean']
                save_policy_checkpoint(
                    path=run_dir / 'best.pt',
                    config=config,
                    policy=policy,
                    reservoir=reservoir,
                    global_step=global_step,
                    stats={
                        'kind': 'best',
                        'best_eval_return': best_eval_return,
                        'eval_stats': eval_stats,
                        'eval_history': eval_history,
                    },
                )
                print(f"new best checkpoint saved: {run_dir / 'best.pt'}")
            next_eval_step += eval_every

        while global_step >= next_save_step:
            save_policy_checkpoint(
                path=run_dir / f'step_{global_step}.pt',
                config=config,
                policy=policy,
                reservoir=reservoir,
                global_step=global_step,
                stats={
                    'kind': 'periodic',
                    'recent_return_mean': mean_return,
                    'best_eval_return': best_eval_return,
                    'eval_history': eval_history,
                },
            )
            save_policy_checkpoint(
                path=run_dir / 'latest.pt',
                config=config,
                policy=policy,
                reservoir=reservoir,
                global_step=global_step,
                stats={
                    'kind': 'latest',
                    'recent_return_mean': mean_return,
                    'best_eval_return': best_eval_return,
                    'eval_history': eval_history,
                },
            )
            next_save_step += save_every

    save_policy_checkpoint(
        path=run_dir / 'latest.pt',
        config=config,
        policy=policy,
        reservoir=reservoir,
        global_step=global_step,
        stats={
            'kind': 'latest_final',
            'recent_return_mean': sum(recent_returns) / max(1, len(recent_returns)),
            'best_eval_return': best_eval_return,
            'eval_history': eval_history,
        },
    )

    pbar.close()
    eval_env.close()
    for env in envs:
        env.close()


if __name__ == '__main__':
    main()
