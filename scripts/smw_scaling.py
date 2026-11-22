#!/usr/bin/env python
"""
How does generalization to unseen levels grow with the number of training levels?

For each of 4 folds of held-out levels, an MLP (on the GPU) and the reservoir ridge readout are trained on
k of the remaining levels (several random subsets per k) and tested on the held-out levels. A control trains
on the same number of moments drawn from all remaining levels, which separates "more different levels"
(diversity) from "more data" (quantity).

    env -u HSA_OVERRIDE_GFX_VERSION PYTHONPATH=scripts .venv-gfx1010/bin/python scripts/smw_scaling.py

q = teacher's 0..1 rating of the chosen button; gain = q minus the best constant button (picked on the
training data). MLP: 'final' after a fixed number of steps, 'peak' = best held-out value seen during
training (optimistic, it peeks at the test set). Ridge: best of a fixed alpha grid (also optimistic).
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from echomario.imitation import make_targets
from smw_readout_lab import attach_streams, load_episodes, replay_features


def mlp(d: int, width: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                               torch.nn.ReLU(), torch.nn.Linear(width, 8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/abc_v2/data')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--subsets', type=int, default=3)
    p.add_argument('--out', default='runs/smw/scaling')
    args = p.parse_args()
    dev = __import__('os').environ.get('DEVICE', 'cuda')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    attach_streams(eps, cfg, data.parent / 'streams', 6)
    X = sparse.vstack([e['stream'][e['sample_pos']] for e in eps]).tocsr().astype(np.float32)
    scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)]).astype(np.float32)
    deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    fpath = data.parent / 'feats_reservoir_baseline.npy'
    if not fpath.exists():  # the 3000-neuron baseline reservoir of the config, plus a bias column
        print('computing reservoir features ...', flush=True)
        np.save(fpath, np.concatenate(replay_features(eps, cfg, 'reservoir')).astype(np.float32))
    F = np.load(fpath)
    assert len(F) == len(y), (F.shape, y.shape)
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    print(f'{len(y)} moments, {len(uniq)} levels: ' + ', '.join(f'{lv} ({np.sum(levels == lv)})' for lv in uniq), flush=True)

    Xg = torch.from_numpy(X.toarray()).to(dev)
    yg = torch.from_numpy(y).to(dev)
    Fg = torch.from_numpy(F).to(dev)
    Yd = torch.from_numpy(y.astype(np.float64))

    def quality(pred_actions, idx):
        return float(y[idx, pred_actions].mean()), float(deadly[idx, pred_actions].mean())

    def run_mlp(tr, te, seed):
        torch.manual_seed(seed)
        model = mlp(X.shape[1], args.width).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        g = torch.Generator(device='cpu').manual_seed(seed)
        tr_t = torch.from_numpy(tr).to(dev)
        te_t = torch.from_numpy(te).to(dev)
        curve = []
        for step in range(1, args.steps + 1):
            b = tr_t[torch.randint(0, len(tr), (min(args.batch, len(tr)),), generator=g).to(dev)]
            loss = torch.mean((model(Xg[b]) - yg[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                with torch.no_grad():
                    a = model(Xg[te_t]).argmax(1).cpu().numpy()
                curve.append((step, *quality(a, te)))
        return curve

    def run_ridge(tr, te):
        Ft = Fg[torch.from_numpy(tr).to(dev)]
        G = (Ft.T @ Ft).double().cpu() / len(tr)
        B = (Ft.T @ yg[torch.from_numpy(tr).to(dev)]).double().cpu() / len(tr)
        best = None
        for alpha in (0.01, 0.1, 1.0, 10.0):
            r = alpha * torch.trace(G) / G.shape[0]
            Gr = G + r * torch.eye(G.shape[0], dtype=G.dtype)
            Gr[-1, -1] -= r  # bias column is not regularized
            W = torch.linalg.solve(Gr, B)
            a = (torch.from_numpy(F[te]).double() @ W).argmax(1).numpy()
            q, d = quality(a, te)
            if best is None or q > best[0]:
                best = (q, d, alpha)
        return best

    rows = []
    t_start = time.time()
    for fold in range(4):
        test_lv = [lv for lv in uniq if fold_of[lv] == fold]
        pool = [lv for lv in uniq if fold_of[lv] != fold]
        te = np.nonzero(np.isin(levels, test_lv))[0]
        ks = sorted({k for k in (1, 2, 4, 8, 16, 32, len(pool)) if k <= len(pool)})
        print(f'\nfold {fold}: test {test_lv} ({len(te)} moments), pool of {len(pool)} levels', flush=True)
        rng = np.random.default_rng(100 + fold)
        for k in ks:
            n_sub = 1 if k == len(pool) else args.subsets
            for s in range(n_sub):
                chosen = sorted(rng.choice(pool, k, replace=False).tolist())
                tr = np.nonzero(np.isin(levels, chosen))[0]
                pool_idx = np.nonzero(np.isin(levels, pool))[0]
                ctrl = np.sort(rng.choice(pool_idx, len(tr), replace=False))
                for kind, idx in (('levels', tr), ('control', ctrl)):
                    if kind == 'control' and k == len(pool):
                        continue  # identical to the full pool
                    const_a = int(np.argmax(y[idx].mean(0)))
                    cq, cd = quality(np.full(len(te), const_a), te)
                    curve = run_mlp(idx, te, seed=1000 * fold + 10 * k + s)
                    peak = max(curve, key=lambda c: c[1])
                    rq, rd, ra = run_ridge(idx, te)
                    row = {'fold': fold, 'k': k, 'subset': s, 'kind': kind, 'levels': chosen, 'n': len(idx),
                           'const_q': cq, 'mlp_final_q': curve[-1][1], 'mlp_final_dead': curve[-1][2],
                           'mlp_peak_q': peak[1], 'mlp_peak_step': peak[0], 'ridge_q': rq, 'ridge_dead': rd,
                           'ridge_alpha': ra, 'curve': curve}
                    rows.append(row)
                    print(f'  k={k:2d} {kind:7s} n={len(idx):5d} const {cq:.3f} | MLP final {curve[-1][1]:.3f} '
                          f'({curve[-1][1] - cq:+.3f}) peak {peak[1]:.3f} ({peak[1] - cq:+.3f}) @ {peak[0]} | '
                          f'reservoir ridge {rq:.3f} ({rq - cq:+.3f}) | {time.time() - t_start:.0f}s', flush=True)
                    (out / 'results.json').write_text(json.dumps(rows))

    print('\nSUMMARY (mean gain over constant across folds and subsets)')
    print(f"{'k':>3} {'kind':8s} {'n':>6} {'MLP final':>10} {'MLP peak':>9} {'ridge':>8}")
    for k in sorted({r['k'] for r in rows}):
        for kind in ('levels', 'control'):
            rs = [r for r in rows if r['k'] == k and r['kind'] == kind]
            if not rs:
                continue
            g = lambda key: np.mean([r[key] - r['const_q'] for r in rs])
            print(f"{k:3d} {kind:8s} {np.mean([r['n'] for r in rs]):6.0f} {g('mlp_final_q'):+10.4f} "
                  f"{g('mlp_peak_q'):+9.4f} {g('ridge_q'):+8.4f}")


if __name__ == '__main__':
    main()
