#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)

    config = ckpt["config"]
    env = make_env(config)

    continuous = isinstance(env.action_space, spaces.Box)

    input_labels = None
    if hasattr(env, "get_input_feature_names"):
        input_labels = env.get_input_feature_names()

    reservoir = load_reservoir_from_state_dict(ckpt["reservoir_state"])

    policy = build_policy(config, env, reservoir.cfg.size)

    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    obs, _ = env.reset()
    reservoir.reset()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    done = False
    ep_return = 0.0

    for step in range(args.max_steps):
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
                labels = CONTINUOUS_ACTION_LABELS
            else:
                logits, _value = policy.forward(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                env_action = int(torch.argmax(logits, dim=-1).item())
                action_values = probs
                selected_action = env_action
                labels = DISCRETE_ACTION_LABELS_TOY[: env.action_space.n]

        contribution = policy_contribution(policy, x_np, selected_action)

        obs, reward, terminated, truncated, _info = env.step(env_action)
        done = bool(terminated or truncated)
        ep_return += float(reward)

        game_frame = env.render()

        frame_rgb = compose_frame(
            game_frame=game_frame,
            reservoir_state=x_np,
            action_probs=action_values,
            action_labels=labels,
            contribution=contribution,
            selected_action=selected_action,
            step=step,
            ep_return=ep_return,
            observation=obs_for_panel,
            input_labels=input_labels,
        )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if writer is None:
            h, w = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (w, h),
            )

        writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    env.close()
    print(f"saved video: {out_path}")


if __name__ == "__main__":
    main()