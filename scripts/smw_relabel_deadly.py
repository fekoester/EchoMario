#!/usr/bin/env python
"""
Corrects the teacher's "deadly" labels of a whole dataset with the stronger survival search of
smw_label_check.py: every deadly (moment, button) gets random sticky rollouts; if one survives 3 s, the label is
flipped to survivable. Writes <out>/<episode>.npy with a (moments, 8) bool mask "really deadly" (True only where
the teacher said deadly and no rollout survived). Resumable: finished episodes are skipped.

    PYTHONPATH=scripts .venv/bin/python scripts/smw_relabel_deadly.py --rollouts 100 --workers 4
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from smw_label_check import check_episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/abc_v2/data')
    p.add_argument('--rollouts', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='runs/smw/deadly_checked')
    args = p.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((data.parent / 'results.json').read_text())['config']
    jobs = []
    for n, f in enumerate(sorted(data.glob('iter*_*.npz'))):
        if (out / f'{f.stem}.npy').exists():
            continue
        with np.load(f) as z:
            s = z['scores'].astype(np.float64)
        pos = np.nonzero((np.isfinite(s) & (s <= -1000)).any(1))[0].tolist()
        if not pos:
            np.save(out / f'{f.stem}.npy', np.zeros(s.shape, bool))
            continue
        jobs.append((str(f), pos, config, args.rollouts, n))
    print(f'{len(jobs)} episodes to check', flush=True)
    done = 0
    with get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(check_episode, jobs):
            if not res:
                continue
            stem = res[0]['file']
            with np.load(data / f'{stem}.npz') as z:
                s = z['scores'].astype(np.float64)
            mask = np.isfinite(s) & (s <= -1000)
            for x in res:
                if x['rescued']:
                    mask[x['moment'], x['button']] = False
            np.save(out / f'{stem}.npy', mask)
            done += 1
            print(f"{done}/{len(jobs)} {stem}: {len(res)} deadly labels, {np.mean([x['rescued'] for x in res]):.0%} "
                  f'survivable', flush=True)


if __name__ == '__main__':
    main()
