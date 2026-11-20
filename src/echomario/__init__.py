"""EchoMario: frozen-reservoir vs. trained-recurrent control in a procedurally generated platformer."""

__version__ = "0.3.0"

from echomario.envs.registration import register_envs as _register_envs

_register_envs()
