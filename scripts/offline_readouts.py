#!/usr/bin/env python
"""
Compares readout feature maps offline on a recorded teacher dataset (see record_teacher.py).

    python scripts/offline_readouts.py --config configs/imitation/full.yaml --data runs/teacher/full.npz \
        --workers 3 --out runs/teacher/full_offline.json

Each variant is ``name|mode|key=value,...`` where keys override ``reservoir.*`` (plus the special
``screen_only=1`` which hides the 11 state features, and ``critical_weight=k``).
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path

import yaml

from echomario.envs.toy_platformer_env import ToyPlatformerEnv
from echomario.teacher_data import evaluate_feature_map, load_levels
from echomario.utils.config import load_config

DEFAULT_VARIANTS = [
    'frame-linear|input|',
    'res-dense|reservoir|',
    'res-dense+frame|reservoir+input|',
    'res-local|reservoir|input_structure=local',
    'res-local+frame|reservoir+input|input_structure=local',
    'res-local-multi|reservoir|input_structure=local,leak_rate_min=0.05,leak_rate_max=1.0',
    'res-local-3000|reservoir|input_structure=local,size=3000',
    'screen:frame-linear|input|screen_only=1',
    'screen:res-dense|reservoir|screen_only=1',
    'screen:res-local|reservoir|screen_only=1,input_structure=local',
    'screen:res-local-multi|reservoir|screen_only=1,input_structure=local,leak_rate_min=0.05,leak_rate_max=1.0',
]

_LEVELS: list[dict] = []


def run_variant(args: tuple) -> dict:
    config, spec = args
    name, mode, overrides = spec.split('|')
    cfg = copy.deepcopy(config)
    drop_extra = 0
    critical_weight = 0.0
    fit = 'ridge'
    device = 'cpu'
    for item in filter(None, overrides.split(',')):
        key, value = item.split('=', 1)
        value = yaml.safe_load(value)
        if key == 'screen_only':
            if value:
                cfg['env']['include_state_features'] = False
                drop_extra = len(ToyPlatformerEnv.STATE_FEATURE_NAMES)
        elif key == 'critical_weight':
            critical_weight = float(value)
        elif key == 'fit':
            fit = str(value)
        elif key == 'device':
            device = str(value)
        else:
            cfg['reservoir'][key] = value
    result = evaluate_feature_map(cfg, mode, _LEVELS, drop_extra=drop_extra, critical_weight=critical_weight,
                                  fit=fit, device=device)
    result['name'] = name
    result['spec'] = spec
    best = result['best']
    print(f"{name:28s} dim={result['dim']:5d} train safe={best['train']['safe_rate']:.3f} "
          f"crit={best['train']['critical_safe_rate']:.3f} | test safe={best['test']['safe_rate']:.3f} "
          f"crit={best['test']['critical_safe_rate']:.3f} alpha={result['best_alpha']:g} ({result['seconds']}s)",
          flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--variant', action='append', default=None)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    config = load_config(args.config)
    _LEVELS.extend(load_levels(Path(args.data)))  # loaded before forking: shared copy-on-write
    variants = args.variant or DEFAULT_VARIANTS
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(run_variant, [(config, v) for v in variants], chunksize=1)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
