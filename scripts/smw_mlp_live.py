#!/usr/bin/env python
"""
Live training curves on the saved DAgger data: two MLPs train side by side on the GPU, one tested on held-out
levels ("across"), one on held-out DAgger rounds of the same levels ("within"). Every epoch prints the training
error and both test errors; nothing stops early, so overfitting (or a double descent) shows up.

    PYTHONPATH=scripts .venv-gfx1010/bin/python scripts/smw_mlp_live.py --epochs 2000

.venv-gfx1010 holds AMD's PyTorch built for the RX 5700 (gfx1010); do not set HSA_OVERRIDE_GFX_VERSION.

Columns: mse = squared error on the 8 button targets; q = teacher's 0..1 rating of the chosen button (constant
action ~0.823, teacher 1.0); dead = share of moments where the chosen button is deadly.
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from echomario.imitation import make_targets
from smw_readout_lab import attach_streams, load_episodes


class SparseMLP(torch.nn.Module):
    """MLP whose first layer takes a sparse batch (the observation has ~270 non-zeros out of 9259)."""

    def __init__(self, d: int, width: int, depth: int, dropout: float):
        super().__init__()
        first = torch.nn.Linear(d, width)  # same initialisation as a dense layer
        self.w1 = torch.nn.Parameter(first.weight.detach().t().contiguous())
        self.b1 = torch.nn.Parameter(first.bias.detach().clone())
        layers = [torch.nn.ReLU()] + ([torch.nn.Dropout(dropout)] if dropout else [])
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.ReLU()] + ([torch.nn.Dropout(dropout)] if dropout else [])
        layers.append(torch.nn.Linear(width, 8))
        self.rest = torch.nn.Sequential(*layers)

    def forward(self, x):
        h = torch.sparse.mm(x, self.w1) if x.is_sparse else x @ self.w1
        return self.rest(h + self.b1)


def to_torch(X: sparse.csr_matrix, idx: np.ndarray, dev: str):
    c = X[idx].tocoo()
    t = torch.sparse_coo_tensor(np.vstack([c.row, c.col]), c.data, c.shape)
    return t.to(dev) if dev == 'cpu' else t.to_dense().to(dev)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/abc_v2/data')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=5e-4)  # 1e-3 killed the net in epoch 1 without regularization
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--fold', type=int, default=0, help='which of 4 folds is held out')
    p.add_argument('--out', default='runs/smw/mlp_live.csv')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='cuda (use .venv-gfx1010: AMD gfx1010 build, no override) or cpu (sparse first layer)')
    args = p.parse_args()
    dev = args.device

    data = Path(args.data)
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    attach_streams(eps, cfg, data.parent / 'streams', 2)
    X = sparse.vstack([e['stream'][e['sample_pos']] for e in eps]).tocsr().astype(np.float32)
    scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)]).astype(np.float32)
    deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    rounds = np.concatenate([np.full(len(e['labels']), e['iter']) for e in eps])
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    held = {'across': np.array([fold_of[lv] for lv in levels]) == args.fold, 'within': rounds % 4 == args.fold}
    print(f"{len(y)} moments, {X.shape[1]} inputs on {dev}. across: held-out levels "
          f"{', '.join(lv for lv in uniq if fold_of[lv] == args.fold)}; within: held-out DAgger rounds "
          f"{sorted(set(rounds[held['within']].tolist()))}. constant action q = 0.823", flush=True)

    yt = torch.from_numpy(y).to(dev)
    Xg = torch.from_numpy(X.toarray()).to(dev) if dev != 'cpu' else None  # ~740 MB, batches are then just indexing

    def batch_x(idx):
        return Xg[torch.from_numpy(idx).to(dev)] if Xg is not None else to_torch(X, idx, dev)
    yn, dn = y, deadly
    runs = {}
    for name, te in held.items():
        torch.manual_seed(0)
        model = SparseMLP(X.shape[1], args.width, args.depth, args.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        tr_np = np.nonzero(~te)[0]
        runs[name] = {'model': model, 'opt': opt, 'te': np.nonzero(te)[0], 'tr_np': tr_np,
                      'tr_sample': np.sort(np.random.default_rng(1).choice(tr_np, min(4000, len(tr_np)), replace=False))}
    n_params = sum(p_.numel() for p_ in runs['within']['model'].parameters())
    print(f'MLP {X.shape[1]} -> ' + ' -> '.join([str(args.width)] * args.depth) + f' -> 8, {n_params / 1e6:.2f} M parameters, '
          f'dropout {args.dropout}, weight decay {args.weight_decay}, lr {args.lr}, batch {args.batch}', flush=True)

    def measure(model, idx):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(batch_x(idx[i:i + 4096])) for i in range(0, len(idx), 4096)]).cpu().numpy()
        model.train()
        a = out.argmax(1)
        return float(np.mean((out - yn[idx]) ** 2)), float(yn[idx, a].mean()), float(dn[idx, a].mean())

    head = (f"{'epoch':>5} | {'ACROSS levels: train mse   q   dead':^36} | {'test mse   q    dead':^22} || "
            f"{'WITHIN levels: train mse   q   dead':^36} | {'test mse   q    dead':^22}")
    tqdm.write(head)
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['epoch'] + [f'{s}_{part}_{m}' for s in ('across', 'within') for part in ('train', 'test') for m in ('mse', 'q', 'dead')])
        bar = tqdm(range(1, args.epochs + 1), unit='epoch', dynamic_ncols=True)
        for ep in bar:
            t0 = time.time()
            row = [ep]
            line = f'{ep:5d} |'
            for name in ('across', 'within'):
                r = runs[name]
                perm = np.random.permutation(r['tr_np'])
                for i in range(0, len(perm), args.batch):
                    b = perm[i:i + args.batch]
                    loss = torch.mean((r['model'](batch_x(b)) - yt[torch.from_numpy(b).to(dev)]) ** 2)
                    r['opt'].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(r['model'].parameters(), 5.0)
                    r['opt'].step()
                tr_m = measure(r['model'], r['tr_sample'])  # training error on a fixed 4000-moment sample
                te_m = measure(r['model'], r['te'])
                row += list(tr_m) + list(te_m)
                line += (f'  {tr_m[0]:.4f} {tr_m[1]:.4f} {tr_m[2]:.3f}            | {te_m[0]:.4f} {te_m[1]:.4f} {te_m[2]:.3f} '
                         + ('||' if name == 'across' else ''))
            w.writerow(row)
            fh.flush()
            tqdm.write(line)
            bar.set_postfix(across_test_q=f'{row[5]:.4f}', within_test_q=f'{row[11]:.4f}', s_per_epoch=f'{time.time() - t0:.1f}')


if __name__ == '__main__':
    main()
