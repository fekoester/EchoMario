import cProfile
import json
import pstats

from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv
from echomario.experts.smw_astar import AStarConfig, SMWAStar

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
env = SMWEnv('YoshiIsland2', max_steps=2400, **grid_kwargs(cfg)); env.reset(); env.set_render(False)
for _ in range(40): env.step(2)
T = SMWAStar(AStarConfig(budget_steps=3000))
cProfile.run('T.scores(env)', '/tmp/astar_prof.out')
pstats.Stats('/tmp/astar_prof.out').sort_stats('tottime').print_stats(14)
