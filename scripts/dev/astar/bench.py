import json
import time

from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
env = SMWEnv('YoshiIsland2', max_steps=10**9, stagnation_steps=10**9, render_mode=None, **grid_kwargs(cfg))
env.reset()
for _ in range(100): env.step(2)
t=time.time(); n=3000
for i in range(n): env.emulate(5 if i%20<8 else 2); env._after_step(compute_obs=False)
dt=time.time()-t; print(f'emulate+track: {n/dt:.0f} steps/s')
t=time.time()
for i in range(500): s=env.get_sim_state()
print(f'get_state: {(time.time()-t)/500*1e3:.3f} ms')
t=time.time()
for i in range(500): env.set_sim_state(s)
print(f'set_state: {(time.time()-t)/500*1e3:.3f} ms')
t=time.time()
for i in range(300): env.step(2)
print(f'step with obs: {300/(time.time()-t):.0f} steps/s')
