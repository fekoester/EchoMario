"""
Gymnasium registration for the toy platformer, so it can be used outside this repository::

    import gymnasium as gym
    import echomario  # registers the environments

    env = gym.make("EchoMario/ToyPlatformer-v0")
    # or, without the explicit import:
    env = gym.make("echomario:EchoMario/ToyPlatformer-v0")

Any ``ToyPlatformerEnv`` keyword argument can be passed through ``gym.make`` (e.g. ``num_enemies=0``).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from echomario.envs.make_env import DiscreteToyActionWrapper, FullScreenFrameStack
from echomario.envs.toy_platformer_env import ToyPlatformerEnv

# Easy level for quick experiments and demos (no gaps or enemies, short level).
EASY_KWARGS: dict[str, Any] = dict(
    width=90, camera_width=30, max_steps=700, num_gaps=0, num_bridge_gaps=0, num_enemies=0,
    stagnation_timeout=100,
)


def make_toy_platformer(
    discrete: bool = False,
    frame_stack: int = 1,
    render_mode: str | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Builds a ``ToyPlatformerEnv`` with optional discrete actions and frame stacking."""
    kwargs.setdefault("randomize", True)
    env: gym.Env = ToyPlatformerEnv(**kwargs)
    env.render_mode = render_mode
    if discrete:
        env = DiscreteToyActionWrapper(env)
    if frame_stack > 1:
        env = FullScreenFrameStack(env, frame_stack)
    return env


ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "EchoMario/ToyPlatformer-v0": {},
    "EchoMario/ToyPlatformerDiscrete-v0": {"discrete": True},
    "EchoMario/ToyPlatformerEasy-v0": dict(EASY_KWARGS),
    "EchoMario/ToyPlatformerEasyDiscrete-v0": {**EASY_KWARGS, "discrete": True},
}


def register_envs() -> None:
    for env_id, kwargs in ENVIRONMENTS.items():
        if env_id not in gym.registry:
            gym.register(
                id=env_id,
                entry_point="echomario.envs.registration:make_toy_platformer",
                kwargs=kwargs,
                disable_env_checker=True,
            )
