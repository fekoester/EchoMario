#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from echomario.agents.factory import PolicyLike, build_policy
from echomario.envs.make_env import make_env
from echomario.envs.reset import EpisodeSeedManager
from echomario.reservoirs.esn import ReservoirLike, make_reservoir
from echomario.training.evaluate import evaluate_policy, evaluate_policy_env_pool
from echomario.training.env_pool import SubprocEnvPool
from echomario.training.ppo import ppo_update
from echomario.training.rollout import (
    collect_rollout_env_pool,
    collect_rollout_parallel,
    env_is_continuous,
    reservoir_is_identity,
)
from echomario.utils.checkpoint import load_checkpoint, save_checkpoint
from echomario.utils.config import load_config
from echomario.utils.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
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
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    seed_manager = EpisodeSeedManager(config)

    num_envs = int(config['training'].get('num_envs', 1))
    if num_envs < 1:
        raise ValueError('training.num_envs must be at least 1')

    rollout_workers = int(config['training'].get('rollout_workers', 0))
    if rollout_workers not in {0, num_envs}:
        raise ValueError('training.rollout_workers must be 0 or equal to training.num_envs')

    env_pool = None
    envs = [] if rollout_workers > 0 else [make_env(config) for _ in range(num_envs)]
    env = make_env(config) if rollout_workers > 0 else envs[0]
    if rollout_workers > 0:
        start_method = str(config['training'].get('env_start_method', 'spawn'))
        env_pool = SubprocEnvPool(config, num_envs=num_envs, start_method=start_method)

    reservoir = make_reservoir(config, input_dim=int(env.observation_space.shape[0]))
    reservoir.reset(batch_size=num_envs)
    policy = build_policy(config, env, reservoir.cfg.size).to(device)
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
        policy.load_state_dict(checkpoint['policy_state'])
        print(f"loaded policy weights from checkpoint: {args.checkpoint}")
    optimizer = optim.Adam(policy.parameters(), lr=float(config['agent']['lr']))

    run_dir = Path(config['logging']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshots_dir = run_dir / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    eval_jsonl_path = run_dir / 'eval_history.jsonl'
    eval_csv_path = run_dir / 'eval_history.csv'
    eval_jsonl_path.unlink(missing_ok=True)
    eval_csv_path.unlink(missing_ok=True)

    total_steps = int(config['training']['total_steps'])
    rollout_steps = int(config['training']['rollout_steps'])
    save_every = int(config['training'].get('save_every_steps', 25000))
    eval_every = int(config['training'].get('eval_every_steps', 25000))
    eval_workers = int(config['training'].get('eval_workers', 0))
    rollout_progress_interval = int(config['training'].get('rollout_progress_interval', 16))
    minibatch_size = int(config['training']['minibatch_size'])
    update_epochs = int(config['training']['update_epochs'])
    value_clip_eps = float(config['training'].get('value_clip_eps', 0.0))
    target_kl = float(config['training'].get('target_kl', 0.0))
    use_amp = bool(config['training'].get('use_amp', False))
    amp_dtype_name = str(config['training'].get('amp_dtype', 'bfloat16')).lower()
    amp_dtype = torch.float16 if amp_dtype_name in {'float16', 'fp16'} else torch.bfloat16
    entropy_coef_start = float(config['agent']['entropy_coef'])
    entropy_coef_end = float(config['agent'].get('entropy_coef_end', entropy_coef_start))
    entropy_anneal_steps = int(config['agent'].get('entropy_anneal_steps', total_steps))
    success_threshold = float(config.get('curriculum', {}).get('success_threshold', 0.0))
    stop_on_success = bool(config.get('curriculum', {}).get('stop_on_success', True))

    print(
        'training setup: '
        f'device={device}, num_envs={num_envs}, rollout_steps={rollout_steps}, '
        f'batch={num_envs * rollout_steps}, minibatch_size={minibatch_size}, '
        f'update_epochs={update_epochs}, rollout_workers={rollout_workers}, '
        f'amp={use_amp}:{amp_dtype_name}'
    )

    def reset_training_spec():
        return seed_manager.training_reset_spec()

    def reset_training_episode(env_idx: int) -> np.ndarray:
        reset_spec = reset_training_spec()
        obs, _ = envs[env_idx].reset(**reset_spec.as_kwargs())
        return obs

    if env_pool is None:
        obs_batch = np.stack([reset_training_episode(env_idx) for env_idx in range(num_envs)], axis=0)
    else:
        obs_batch = env_pool.reset_all([reset_training_spec() for _ in range(num_envs)])
    rollout_state = None

    eval_env = make_env(config)
    eval_reservoir = make_reservoir(config, input_dim=int(eval_env.observation_space.shape[0]))
    eval_reset_specs = seed_manager.evaluation_reset_specs()
    if eval_workers <= 0 and reservoir_is_identity(eval_reservoir):
        eval_workers = len(eval_reset_specs)
    if eval_workers not in {0, len(eval_reset_specs)}:
        raise ValueError('training.eval_workers must be 0 or equal to env.num_eval_seeds')
    eval_env_pool = None
    if eval_workers > 0:
        start_method = str(config['training'].get('env_start_method', 'spawn'))
        eval_env_pool = SubprocEnvPool(config, num_envs=eval_workers, start_method=start_method)

    global_step = 0
    stop_training = False
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
        iter_start_time = time.perf_counter()
        rollout_start_time = time.perf_counter()
        progress_during_rollout = 0

        def update_rollout_progress(step_count: int) -> None:
            nonlocal progress_during_rollout
            step_count = min(step_count, total_steps - (global_step + progress_during_rollout))
            if step_count <= 0:
                return
            progress_during_rollout += step_count
            pbar.update(step_count)

        if env_pool is None:
            rollout = collect_rollout_parallel(
                envs=envs,
                reservoir=reservoir,
                policy=policy,
                obs_batch=obs_batch,
                state=rollout_state,
                rollout_steps=rollout_steps,
                device=device,
                reset_fn=reset_training_episode,
                progress_fn=update_rollout_progress,
                progress_interval=rollout_progress_interval,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        else:
            rollout = collect_rollout_env_pool(
                env_pool=env_pool,
                continuous=env_is_continuous(env),
                reservoir=reservoir,
                policy=policy,
                obs_batch=obs_batch,
                state=rollout_state,
                rollout_steps=rollout_steps,
                device=device,
                reset_spec_fn=lambda _env_idx: reset_training_spec(),
                progress_fn=update_rollout_progress,
                progress_interval=rollout_progress_interval,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - rollout_start_time
        obs_batch = rollout.last_obs
        rollout_state = rollout.last_state

        step_increment = int(rollout.rewards.numel())
        previous_step = global_step
        global_step += step_increment
        remaining_progress = min(step_increment, total_steps - previous_step) - progress_during_rollout
        if remaining_progress > 0:
            pbar.update(remaining_progress)

        entropy_progress = min(1.0, global_step / max(1, entropy_anneal_steps))
        entropy_coef = entropy_coef_start + entropy_progress * (entropy_coef_end - entropy_coef_start)

        update_start_time = time.perf_counter()
        stats = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            gamma=float(config['training']['gamma']),
            gae_lambda=float(config['training']['gae_lambda']),
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            clip_eps=float(config['agent']['clip_eps']),
            value_coef=float(config['agent']['value_coef']),
            entropy_coef=entropy_coef,
            max_grad_norm=float(config['agent']['max_grad_norm']),
            value_clip_eps=value_clip_eps,
            target_kl=target_kl,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_start_time
        iter_seconds = time.perf_counter() - iter_start_time
        steps_per_second = step_increment / max(1e-9, iter_seconds)

        recent_returns.extend(rollout.episode_returns)
        if len(recent_returns) > 50:
            recent_returns = recent_returns[-50:]

        mean_return = sum(recent_returns) / max(1, len(recent_returns))
        rollout_timings = rollout.timings
        pbar.set_postfix(
            {
                'ret50': f'{mean_return:.2f}',
                'loss': f'{stats.loss:.3f}',
                'ent': f'{stats.entropy:.3f}',
                'entc': f'{entropy_coef:.4f}',
                'kl': f'{stats.approx_kl:.4f}',
                'clip': f'{stats.clip_fraction:.2f}',
                'best': f'{best_eval_return:.2f}',
                'sps': f'{steps_per_second:.0f}',
                'roll': f'{rollout_seconds:.2f}s',
                'upd': f'{update_seconds:.2f}s',
                'pol': f"{rollout_timings.get('policy', 0.0):.2f}s",
                'env': f"{rollout_timings.get('env', 0.0):.2f}s",
                'store': f"{rollout_timings.get('store', 0.0):.2f}s",
            }
        )
        del rollout

        while global_step >= next_eval_step:
            eval_start_time = time.perf_counter()
            if eval_env_pool is not None:
                eval_stats = evaluate_policy_env_pool(
                    env_pool=eval_env_pool,
                    continuous=env_is_continuous(eval_env),
                    reservoir=eval_reservoir,
                    policy=policy,
                    reset_specs=eval_reset_specs,
                    device=device,
                )
            else:
                eval_stats = evaluate_policy(
                    env=eval_env,
                    reservoir=eval_reservoir,
                    policy=policy,
                    episodes=len(eval_reset_specs),
                    reset_specs=eval_reset_specs,
                    device=device,
                )
            eval_seconds = time.perf_counter() - eval_start_time

            print(f"\nstep={global_step} eval_seconds={eval_seconds:.2f} eval={eval_stats}")

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
            if (
                stop_on_success
                and success_threshold > 0.0
                and float(eval_stats.get('eval_success_rate', 0.0)) >= success_threshold
            ):
                stop_training = True
                print(
                    f"curriculum threshold reached: "
                    f"{eval_stats['eval_success_rate']:.3f} >= {success_threshold:.3f}"
                )
                break
            next_eval_step += eval_every

        if stop_training:
            break

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
    if eval_env_pool is not None:
        eval_env_pool.close()
    if env_pool is not None:
        env_pool.close()
        env.close()
    else:
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
