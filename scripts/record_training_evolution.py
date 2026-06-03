#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from gymnasium import spaces

from echomario.agents.factory import build_policy, policy_contribution
from echomario.envs.make_env import make_env
from echomario.reservoirs.esn import load_reservoir_from_state_dict
from echomario.utils.checkpoint import load_checkpoint
from echomario.viz.panels import compose_frame


DISCRETE_ACTION_LABELS_TOY = [
    "noop",
    "right",
    "jump",
    "right+jump",
    "left",
    "sprint",
    "sprint+jump",
]

CONTINUOUS_ACTION_LABELS = [
    "move_axis",
    "jump_signal",
    "run_signal",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-steps-per-policy", type=int, default=350)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-title-frames", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--step-stride", type=int, default=50000)
    return parser.parse_args()


def extract_step(path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def filter_checkpoints_by_stride(checkpoints: list[Path], step_stride: int) -> list[Path]:
    if step_stride <= 0 or len(checkpoints) <= 2:
        return checkpoints

    filtered = [checkpoints[0]]
    last_step = extract_step(checkpoints[0])

    for ckpt in checkpoints[1:-1]:
        step = extract_step(ckpt)
        if step - last_step >= step_stride:
            filtered.append(ckpt)
            last_step = step

    if checkpoints[-1] not in filtered:
        filtered.append(checkpoints[-1])

    return filtered


def make_title_frame(
    width: int,
    height: int,
    title: str,
    subtitle: str,
) -> np.ndarray:
    frame = np.full((height, width, 3), 245, dtype=np.uint8)

    cv2.putText(
        frame,
        title,
        (40, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (20, 20, 20),
        3,
    )

    cv2.putText(
        frame,
        subtitle,
        (40, height // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (60, 60, 60),
        2,
    )

    return frame


def load_policy_from_checkpoint(path: Path):
    ckpt = load_checkpoint(path)
    config = ckpt["config"]

    env = make_env(config)

    continuous = isinstance(env.action_space, spaces.Box)

    if continuous:
        num_actions = int(env.action_space.shape[0])
    else:
        num_actions = int(env.action_space.n)

    input_labels = None
    if hasattr(env, "get_input_feature_names"):
        input_labels = env.get_input_feature_names()

    reservoir = load_reservoir_from_state_dict(ckpt["reservoir_state"])

    policy = build_policy(config, env, reservoir.cfg.size)

    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    return ckpt, config, env, reservoir, policy, input_labels, continuous


def run_episode_frames(
    *,
    ckpt_path: Path,
    max_steps: int,
) -> tuple[list[np.ndarray], float, float]:
    (
        ckpt,
        config,
        env,
        reservoir,
        policy,
        input_labels,
        continuous,
    ) = load_policy_from_checkpoint(ckpt_path)

    obs, _ = env.reset(seed=int(config["project"].get("seed", 42)))
    reservoir.reset()

    frames = []
    done = False
    ep_return = 0.0
    max_x = 0.0

    for step in range(max_steps):
        if done:
            break

        obs_for_panel = obs.copy()

        x_np = reservoir.step(obs)
        x = torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if continuous:
                action, _, _, _ = policy.get_action_and_value(x, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

                env_action = action_np
                action_values = action_np
                selected_action = int(np.argmax(np.abs(action_np)))
                action_labels = CONTINUOUS_ACTION_LABELS
            else:
                logits, _value = policy.forward(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                env_action = int(torch.argmax(logits, dim=-1).item())
                action_values = probs
                selected_action = env_action
                action_labels = DISCRETE_ACTION_LABELS_TOY[: env.action_space.n]

        contribution = policy_contribution(policy, x_np, selected_action)

        obs, reward, terminated, truncated, info = env.step(env_action)
        done = bool(terminated or truncated)
        ep_return += float(reward)
        max_x = max(max_x, float(info.get("max_x", 0.0)))

        game_frame = env.render()

        frame_rgb = compose_frame(
            game_frame=game_frame,
            reservoir_state=x_np,
            action_probs=action_values,
            action_labels=action_labels,
            contribution=contribution,
            selected_action=selected_action,
            step=step,
            ep_return=ep_return,
            observation=obs_for_panel,
            input_labels=input_labels,
        )

        cv2.putText(
            frame_rgb,
            f"checkpoint step: {int(ckpt.get('global_step', extract_step(ckpt_path)))}",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        frames.append(frame_rgb)

    env.close()
    return frames, ep_return, max_x


def main() -> None:
    args = parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(snapshot_dir.glob("step_*.pt"), key=extract_step)

    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    checkpoints = filter_checkpoints_by_stride(checkpoints, args.step_stride)

    if not checkpoints:
        raise FileNotFoundError(f"No step_*.pt snapshots found in {snapshot_dir}")

    print("snapshots:")
    for p in checkpoints:
        print(f"  {p}")

    writer = None
    video_width = None
    video_height = None

    for idx, ckpt_path in enumerate(checkpoints):
        step = extract_step(ckpt_path)
        print(f"recording snapshot {idx + 1}/{len(checkpoints)}: {ckpt_path}")

        frames, ep_return, max_x = run_episode_frames(
            ckpt_path=ckpt_path,
            max_steps=args.max_steps_per_policy,
        )

        if not frames:
            continue

        if writer is None:
            first = frames[0]
            video_height, video_width = first.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (video_width, video_height),
            )

        title = f"Training snapshot {idx + 1}/{len(checkpoints)}"

        if step == 0:
            subtitle = f"step 0 | untrained policy | return={ep_return:.2f} | max_x={max_x:.2f}"
        else:
            subtitle = f"step {step} | return={ep_return:.2f} | max_x={max_x:.2f}"

        title_frame = make_title_frame(
            width=video_width,
            height=video_height,
            title=title,
            subtitle=subtitle,
        )

        title_bgr = cv2.cvtColor(title_frame, cv2.COLOR_RGB2BGR)
        for _ in range(args.hold_title_frames):
            writer.write(title_bgr)

        for frame_rgb in frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    print(f"saved evolution video: {out_path}")


if __name__ == "__main__":
    main()