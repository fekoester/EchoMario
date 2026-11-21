"""A/B/C split: levels sorted by bunny-hop progress, taken in groups of three; per group one level with a
midpoint goes to 'half' (first half trained, second half tested), one to 'full', one to 'unseen'."""
import json
from pathlib import Path
HOP = ['YoshiIsland2', 'DonutPlains3', 'YoshiIsland1', 'YoshiIsland4', 'DonutPlains1', 'Forest5', 'Forest4', 'DonutPlains2',
       'VanillaDome3', 'DonutPlains5', 'VanillaDome2', 'ChocolateIsland3', 'Forest3', 'ChocolateIsland2', 'VanillaDome5',
       'DonutPlains4', 'YoshiIsland3', 'ChocolateIsland1', 'Forest2', 'Forest1', 'Bridges1', 'VanillaDome1', 'VanillaDome4',
       'Bridges2']  # bunny-hop progress, best first
mid = {lv for lv in HOP if Path(f'runs/smw/midpoints/{lv}.state').exists()}
split = {'full': [], 'half': [], 'unseen': []}
for i in range(0, len(HOP), 3):
    group = HOP[i:i + 3]
    half = next((lv for lv in group[1:] + group[:1] if lv in mid), None)
    rest = [lv for lv in group if lv != half]
    if half:
        split['half'].append(half)
    split['full'].append(rest[0])
    if len(rest) > 1:
        split['unseen'].append(rest[1])
    if len(rest) > 2:
        split['full'].append(rest[2])
Path('runs/smw/split_abc.json').write_text(json.dumps(split, indent=1))
print(json.dumps(split), 'midpoints:', len(mid))
