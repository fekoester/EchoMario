import json
import time

from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
env = SMWEnv('YoshiIsland2', max_steps=10**9, stagnation_steps=10**9, render_mode=None, **grid_kwargs(cfg))
env.reset(); print('fast core', env.fast_core, 'state bytes', len(env.em.get_state()))
env.set_render(False)
t=time.time(); n=4000
for i in range(n): env.emulate(5 if i%20<8 else 2); env._after_step(compute_obs=False)
print(f'render off: {n/(time.time()-t):.0f} steps/s')
