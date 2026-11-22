#!/usr/bin/env python
"""
Why does the DAgger student still die? Probes on a smw_dagger.py run (training fold data + DAgger rounds).

  --probe mlp     an MLP readout on the same reservoir features (CPU): fits the labels far better than ridge;
                  does it play the training / held-out levels better (closed loop)?
  --probe weight  ridge with states that have a deadly AND a safe button weighted 1 + k: unsafe-choice rate on
                  those critical states (in-sample) and closed-loop play
  --probe oos     the same weighted ridge trained on everything before round --upto, tested on round --upto's
                  states (new states of the same training levels): how often it picks a deadly button there,
                  compared with the safest single button

    PYTHONPATH=scripts .venv/bin/python scripts/smw_dagger_probes.py --probe oos --run runs/smw/dagger_v3_f0 --upto 2
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '3')

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from echomario.imitation import make_targets
from smw_imitation import job, play, summarize
from smw_readout_lab import load_episodes


def load_run(data: Path, run: Path, fold: int, rounds: list[int]):
    eps = load_episodes(data)
    S0 = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    lab = np.concatenate([e['labels'] for e in eps])
    Y0 = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(S0, lab)]).astype(np.float32)
    lv0 = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    F0 = np.load(data.parent / 'feats_reservoir_baseline.npy')
    uniq = sorted(set(lv0.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    m = np.array([fold_of[lv] != fold for lv in lv0])
    parts = [(F0[m], Y0[m], S0[m])]
    for it in rounds:
        z = np.load(run / f'iter{it}.npz')
        coll = json.loads(str(z['collect']))
        S = np.concatenate([np.load(run / 'data' / f"iter{it}_{ep['level']}.npz")['scores'] for ep in coll])
        parts.append((z['feats'], z['targets'], S.astype(np.float64)))
    specs = sorted({p.stem.split('_', 1)[1] for p in data.glob('*.npz')})
    train_specs = [s for s in specs if fold_of[s.split('#m')[0]] != fold]
    test = [lv for lv in uniq if fold_of[lv] == fold]
    return parts, train_specs, test


def masks(S: np.ndarray):
    fin = np.isfinite(S)
    safe = fin & (S > -1000)
    return safe, safe.any(1) & (fin & (S <= -1000)).any(1)


def weighted_ridge(F, Y, w, alpha):
    Fw = F * np.sqrt(w)[:, None]
    G = Fw.T.astype(np.float64) @ Fw / w.sum()
    B = Fw.T.astype(np.float64) @ (Y * np.sqrt(w)[:, None]) / w.sum()
    r = alpha * np.trace(G) / G.shape[0]
    G[np.diag_indices_from(G)] += r
    G[-1, -1] -= r
    return np.linalg.solve(G, B).astype(np.float32)


def train_mlp(F, Y, width, steps, seed=0):
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    Ft, Yt = torch.from_numpy(F), torch.from_numpy(Y)
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-3
    mu[-1], sd[-1] = 0, 1
    Xn = (Ft - mu) / sd
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(F), (512,), generator=g)
        loss = ((net(Xn[b]) - Yt[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # fold the input normalization into the first layer: a plain numpy MLP for smw_imitation.student_values
    return {'W1': (net[0].weight.T / sd[:, None]).detach().numpy().astype(np.float32),
            'b1': (net[0].bias - (mu / sd) @ net[0].weight.T).detach().numpy().astype(np.float32),
            'W2': net[2].weight.T.detach().numpy().astype(np.float32),
            'b2': net[2].bias.detach().numpy().astype(np.float32)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--probe', required=True, choices=('mlp', 'weight', 'oos'))
    p.add_argument('--data', default='runs/smw/v3_all/data')
    p.add_argument('--run', default='runs/smw/dagger_v3_f0')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--upto', type=int, default=2, help='last DAgger round to use (oos: the round tested on)')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    data, run = Path(args.data), Path(args.run)
    rounds = list(range(1, args.upto + 1))
    parts, train_specs, test = load_run(data, run, args.fold, rounds)
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    if args.probe == 'oos':
        F, Y, S = (np.concatenate(x) for x in zip(*parts[:-1]))
        F2, Y2, S2 = parts[-1]
        safe, crit = masks(S)
        safe2, crit2 = masks(S2)
        n2 = np.arange(len(F2))
        print(f'test: round {args.upto} states ({len(F2)}, critical {crit2.mean():.3f}); safest single button is unsafe '
              f'on {1 - max(float(safe2[:, a][crit2].mean()) for a in range(S2.shape[1])):.3f} of them')
        for k in (0, 10, 50):
            for alpha in (0.001, 0.01, 0.1):
                W = weighted_ridge(F, Y, (1 + k * crit).astype(np.float64), alpha)
                a, a2 = (F @ W).argmax(1), (F2 @ W).argmax(1)
                print(f'k {k:2d} alpha {alpha}: unsafe on critical in-sample {(~safe[np.arange(len(F)), a])[crit].mean():.3f}'
                      f' | on round-{args.upto} states {(~safe2[n2, a2])[crit2].mean():.3f} | q there {Y2[n2, a2].mean():.3f}',
                      flush=True)
        return
    F, Y, S = (np.concatenate(x) for x in zip(*parts))
    safe, crit = masks(S)
    n = np.arange(len(F))
    if args.probe == 'mlp':
        cands = [(f'MLP {w}', (w, s)) for w, s in ((256, 6000), (512, 15000))]
    else:
        cands = [(f'k {k} alpha {a}', (k, a)) for k in (0, 10, 50) for a in (0.001, 0.01)]
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=4)
    rows = []
    for name, c in cands:
        W = train_mlp(F, Y, *c) if args.probe == 'mlp' else weighted_ridge(F, Y, (1 + c[0] * crit).astype(np.float64), c[1])
        from smw_imitation import student_values
        a = student_values(F, W).argmax(1)
        et = pool.map(play, [job(s, cfg, 'reservoir', W, noop=0) for s in train_specs], chunksize=1)
        ev = pool.map(play, [job(lv, cfg, 'reservoir', W, noop=nn) for lv in test for nn in (0, 20)], chunksize=1)
        row = {'name': name, 'fit_q': float(Y[n, a].mean()), 'unsafe_on_critical': float((~safe[n, a])[crit].mean()),
               'train': summarize(et), 'test': summarize(ev)}
        rows.append(row)
        print(f"{name}: fit q {row['fit_q']:.3f} unsafe on critical (in-sample) {row['unsafe_on_critical']:.3f} | training "
              f"levels {row['train']['progress']} died {row['train']['died']} | unseen {row['test']['progress']} "
              f"+- {row['test']['se']} died {row['test']['died']}", flush=True)
        (run / f'probe_{args.probe}_upto{args.upto}.json').write_text(json.dumps(rows, indent=1))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
