#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from echomario.envs.make_env import make_env
from echomario.utils.config import load_config


HELP_TEXT = [
    'A/D or arrows: move left/right',
    'J / shift / Z: run (Mario B)',
    'K / space / X: jump (Mario A)',
    'R: reset same level',
    'N: next random level',
    'P: toggle random autoplay',
    'Esc/Q: quit',
]


@dataclass
class ManualStatus:
    move_dir: float = 0.0
    run_pressed: bool = False
    jump_pressed: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/toy_platformer_continuous.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--level-seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=60)
    return parser.parse_args()


def reset_env(env, seed: int, level_seed: int, randomize: bool):
    options = {'level_seed': level_seed} if randomize else None
    return env.reset(seed=seed, options=options)


def action_from_keys(pygame, keys) -> tuple[np.ndarray, ManualStatus]:
    left = bool(keys[pygame.K_a] or keys[pygame.K_LEFT])
    right = bool(keys[pygame.K_d] or keys[pygame.K_RIGHT])
    run = bool(keys[pygame.K_j] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_z])
    jump = bool(keys[pygame.K_k] or keys[pygame.K_SPACE] or keys[pygame.K_x])

    move_dir = 0.0
    if left and not right:
        move_dir = -1.0
    elif right and not left:
        move_dir = 1.0

    action = np.array(
        [move_dir, 1.0 if jump else -1.0, 1.0 if run else -1.0],
        dtype=np.float32,
    )
    return action, ManualStatus(move_dir=move_dir, run_pressed=run, jump_pressed=jump)


def draw_overlay(pygame, screen, font, mode: str, ep_return: float, level_seed: int, status: ManualStatus | None) -> None:
    lines = [f'mode={mode} return={ep_return:.2f} level_seed={level_seed}']
    if status is not None:
        lines.append(
            f'move={status.move_dir:+.0f} run={"on" if status.run_pressed else "off"} jump={"on" if status.jump_pressed else "off"}'
        )
    lines.extend(HELP_TEXT)

    padding = 8
    line_h = 18
    panel_h = padding * 2 + line_h * len(lines)
    panel = pygame.Surface((520, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    screen.blit(panel, (0, 0))

    for idx, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, padding + idx * line_h))


def main() -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit('pygame is required for Mario-style manual controls. Install dependencies and rerun.') from exc

    args = parse_args()
    config = load_config(args.config)
    config.setdefault('project', {})['seed'] = args.seed
    env = make_env(config)

    randomize = bool(config['env'].get('randomize', False))
    level_seed = args.level_seed if args.level_seed > 0 else args.seed
    _obs, _ = reset_env(env, args.seed, level_seed, randomize)

    pygame.init()
    pygame.display.set_caption('EchoMario Toy World Explorer')
    first_frame = env.render()
    height, width = first_frame.shape[:2]
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    autoplay = False
    ep_return = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    autoplay = not autoplay
                elif event.key == pygame.K_r:
                    ep_return = 0.0
                    _obs, _ = reset_env(env, args.seed, level_seed, randomize)
                elif event.key == pygame.K_n:
                    level_seed += 1
                    ep_return = 0.0
                    _obs, _ = reset_env(env, args.seed, level_seed, randomize)

        if autoplay:
            action = env.action_space.sample().astype(np.float32)
            status = None
        else:
            keys = pygame.key.get_pressed()
            action, status = action_from_keys(pygame, keys)

        _obs, reward, terminated, truncated, _info = env.step(action)
        ep_return += float(reward)

        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        draw_overlay(pygame, screen, font, 'autoplay' if autoplay else 'manual', ep_return, level_seed, status)
        pygame.display.flip()

        if terminated or truncated:
            level_seed += 1 if randomize else 0
            ep_return = 0.0
            _obs, _ = reset_env(env, args.seed, level_seed, randomize)

        clock.tick(args.fps)

    pygame.quit()
    env.close()


if __name__ == '__main__':
    main()
