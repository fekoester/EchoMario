"""Relabels saved v2 episodes with teacher v1 at exactly the same moments (replaying the logs), so the
learnability of v1 and v2 labels can be compared on identical states."""
import json, sys
import numpy as np
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, 'scripts')
from smw_imitation import parse_spec
from echomario.envs.smw_env import SMWEnv
from echomario.experts.smw_planner import SMWPlanner, teacher_config

CLEAN = ['YoshiIsland2', 'YoshiIsland4', 'Forest4', 'DonutPlains5', 'ChocolateIsland1@first', 'ChocolateIsland3@first',
         'DonutPlains1@first', 'Forest1@first']
data = Path('runs/smw/abc_v2/data')
out = Path('runs/smw/relabel_v1')
out.mkdir(exist_ok=True)


def relabel(f):
    with np.load(f) as z:
        log = {k: z[k] for k in z.files}
    level, start = parse_spec(str(log['spec']))
    env = SMWEnv(level, max_steps=10 ** 9, stagnation_steps=10 ** 9)
    env.reset(options=start)
    env.set_render(False)
    for _ in range(int(log['noop'])):
        env.step(0)
    planner = SMWPlanner(teacher_config('v1'))
    steps, actions = log['steps'], log['actions']
    jumps = {int(steps[p]) for p in range(1, len(steps)) if steps[p] != steps[p - 1] + 1}
    saved, want = {}, set(log['sample_pos'].tolist())
    new = {}
    for p, (k, a) in enumerate(zip(steps.tolist(), actions.tolist())):
        if p and k != steps[p - 1] + 1:
            env.set_sim_state(saved[k])
        elif k in jumps and k not in saved:
            saved[k] = env.get_sim_state()
        if p in want:
            new[p] = planner.scores(env)
        env.step(a)
    env.close()
    scores_v1 = np.stack([new[p] for p in log['sample_pos'].tolist()]).astype(np.float32)
    np.save(out / f'{f.stem}.npy', scores_v1)
    return f.stem, len(scores_v1)


files = [f for f in sorted(data.glob('iter*_*.npz')) if any(f.stem.endswith('_' + s) for s in CLEAN)]
print(len(files), 'episodes to relabel', flush=True)
with get_context('fork').Pool(6, maxtasksperchild=1) as pool:
    for stem, n in pool.imap_unordered(relabel, files):
        print(stem, n, flush=True)
