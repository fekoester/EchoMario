"""Play whole levels with the A* teacher under a given config and compare with teacher v3's run.

   PYTHONPATH=scripts .venv/bin/python scripts/dev/astar/astar_eval.py --levels Forest1,VanillaDome3 \
       --config '{"deep_children": ["right_run", "right_run_jump", "idle", "left"]}' --workers 3 --out /tmp/a.json

Same episode limits as the collection (2400 steps, 300 steps without progress). Plain level names only."""
import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

from smw_imitation import grid_kwargs  # noqa: E402

from echomario.envs.smw_env import SMWEnv  # noqa: E402
from echomario.experts.smw_astar import AStarConfig, SMWAStar  # noqa: E402

CFG = json.load(open('runs/smw/v3_all/results.json'))['config']


def run(job):
    level, extra = job
    if 'deep_children' in extra and extra['deep_children'] is not None:
        extra = {**extra, 'deep_children': tuple(extra['deep_children'])}
    env = SMWEnv(level, max_steps=2400, **grid_kwargs(CFG))
    env.reset()
    env.set_render(False)
    T = SMWAStar(AStarConfig(**extra))
    t0, d, info = time.time(), 0, {}
    done = False
    while not done:
        a = T.act(env)
        d += 1
        for _ in range(4):
            _, _, te, tr, info = env.step(a)
            if te or tr:
                done = True
                break
    return {'level': level, 'progress': round(float(info['progress']), 3), 'goal': bool(info['reached_goal']),
            'died': bool(info['died']), 'decisions': d, 'sec_per_decision': round((time.time() - t0) / d, 2),
            'stall_decisions': T.stall_decisions, 'rescued': T.rescued, 'final_x': env.mario()['x']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--levels', required=True)
    p.add_argument('--config', default='{}')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    extra = json.loads(args.config)
    v3 = {}
    for e in json.load(open('runs/smw/v3_all/collect_summary.json')):
        v3.setdefault(e['level'], e)
    levels = args.levels.split(',')
    rows = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for r in pool.imap_unordered(run, [(lv, extra) for lv in levels]):
            r['v3'] = v3.get(r['level'], {}).get('progress')
            rows.append(r)
            print(json.dumps(r), flush=True)
            json.dump({'config': extra, 'rows': rows}, open(args.out, 'w'), indent=1)
    a = sum(r['progress'] for r in rows) / len(rows)
    b = sum(r['v3'] or 0 for r in rows) / len(rows)
    print(f'mean A* {a:.3f}  v3 {b:.3f}  ({len(rows)} levels)', flush=True)


if __name__ == '__main__':
    main()
