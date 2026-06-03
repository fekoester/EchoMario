#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import cv2
import torch
from gymnasium import spaces

from echomario.agents.factory import build_policy
from echomario.envs.make_env import make_env
from echomario.reservoirs.esn import load_reservoir_from_state_dict
from echomario.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--delay', type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt['config']
    env = make_env(config)

    continuous = isinstance(env.action_space, spaces.Box)
    reservoir = load_reservoir_from_state_dict(ckpt['reservoir_state'])
    policy = build_policy(config, env, reservoir.cfg.size)
    policy.load_state_dict(ckpt['policy_state'])
    policy.eval()

    obs, _ = env.reset()
    reservoir.reset()
    done = False
    ep_return = 0.0

    while not done:
        x_np = reservoir.step(obs)
        x = torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if continuous:
                action, _, _, _ = policy.get_action_and_value(x, deterministic=True)
                env_action = action.squeeze(0).cpu().numpy()
            else:
                logits, _ = policy.forward(x)
                env_action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, _info = env.step(env_action)
        done = bool(terminated or truncated)
        ep_return += float(reward)

        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('EchoMario', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(args.delay)

    print(f'episode return: {ep_return:.3f}')
    cv2.destroyAllWindows()
    env.close()


if __name__ == '__main__':
    main()
