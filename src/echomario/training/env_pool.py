from __future__ import annotations

from multiprocessing import connection, get_context
from multiprocessing.context import BaseContext, Process
from typing import Any

import numpy as np

from echomario.envs.make_env import make_env
from echomario.envs.reset import ResetSpec


def _env_worker(config: dict[str, Any], conn: connection.Connection) -> None:
    env = make_env(config)
    try:
        while True:
            command, payload = conn.recv()
            if command == 'reset':
                obs, info = env.reset(**payload)
                conn.send((obs, info))
            elif command == 'step':
                conn.send(env.step(payload))
            elif command == 'close':
                conn.close()
                return
            else:
                raise ValueError(f'Unknown env worker command: {command}')
    finally:
        env.close()


class SubprocEnvPool:
    def __init__(self, config: dict[str, Any], num_envs: int, start_method: str = 'spawn'):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError('num_envs must be at least 1')

        self.ctx: BaseContext = get_context(start_method)
        self.parent_conns: list[connection.Connection] = []
        self.processes: list[Process] = []
        self.closed = False

        for _ in range(self.num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(target=_env_worker, args=(config, child_conn))
            process.daemon = True
            process.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(process)

    def reset_all(self, specs: list[ResetSpec]) -> np.ndarray:
        if len(specs) != self.num_envs:
            raise ValueError(f'Expected {self.num_envs} reset specs, got {len(specs)}')
        for conn, spec in zip(self.parent_conns, specs, strict=True):
            conn.send(('reset', spec.as_kwargs()))
        return np.stack([conn.recv()[0] for conn in self.parent_conns], axis=0).astype(np.float32)

    def reset_at(self, env_idx: int, spec: ResetSpec) -> np.ndarray:
        conn = self.parent_conns[int(env_idx)]
        conn.send(('reset', spec.as_kwargs()))
        obs, _info = conn.recv()
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: np.ndarray) -> list[tuple[np.ndarray, float, bool, bool, dict]]:
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f'Expected actions for {self.num_envs} envs, got shape {actions.shape}')
        for conn, action in zip(self.parent_conns, actions, strict=True):
            conn.send(('step', action))
        return [conn.recv() for conn in self.parent_conns]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for conn in self.parent_conns:
            conn.close()

    def __enter__(self) -> SubprocEnvPool:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
