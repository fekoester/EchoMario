import json
import sys

from smw_imitation import grid_kwargs

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
A = {n: i for i, n in enumerate(ACTION_NAMES)}
level = sys.argv[1]
env = SMWEnv(level, max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg))
env.reset(); env.set_render(False)
xs, vs, ps = [], [], []
for k in range(120):
    env.emulate(A['right_run']); env._after_step(compute_obs=False)
    m = env.mario(); xs.append(m['x']); vs.append(m['vx']); ps.append(int(env._wram()[0x13E4]))
print(level, 'vx', vs[::3], '\npmeter', ps[::3])
