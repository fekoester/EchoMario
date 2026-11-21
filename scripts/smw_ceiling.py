#!/usr/bin/env python
"""
What limits generalization? Fits models of increasing power to the saved DAgger data (raw observation v2 ->
teacher's 0..1 action targets) and scores them on held-out levels and, for comparison, on held-out episodes of
the same levels. If strong nonlinear models generalize across levels where the linear readout does not, the
reservoir + linear readout is the limit; if they do not either, the data / labels are.

    python scripts/smw_ceiling.py --data runs/smw/abc_v2/data --out runs/smw/ceiling
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '6')

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from echomario.envs.smw_env import grid_kwargs, smw_screen_shape
from echomario.imitation import make_targets
from smw_readout_lab import attach_streams, load_episodes, replay_features


def sample_matrix(eps) -> sparse.csr_matrix:
    rows = []
    for e in eps:
        st = e['stream']
        r = st[e['sample_pos']]
        rows.append(r if sparse.issparse(r) else sparse.csr_matrix(np.asarray(r, np.float32)))
    return sparse.vstack(rows).tocsr().astype(np.float32)


def score(pred_q: np.ndarray, y: np.ndarray, deadly: np.ndarray) -> dict:
    a = np.argmax(pred_q, axis=1)
    i = np.arange(len(a))
    return {'q_chosen': round(float(y[i, a].mean()), 4), 'deadly': round(float(deadly[i, a].mean()), 4)}


def ridge(Xtr, ytr, Xte, alpha):
    Xtr = np.hstack([Xtr.toarray(), np.ones((Xtr.shape[0], 1), np.float32)]) if sparse.issparse(Xtr) else Xtr
    Xte = np.hstack([Xte.toarray(), np.ones((Xte.shape[0], 1), np.float32)]) if sparse.issparse(Xte) else Xte
    g = Xtr.T.astype(np.float64) @ Xtr / len(Xtr)
    r = alpha * float(np.trace(g)) / g.shape[0]
    g[np.diag_indices(g.shape[0])] += r
    g[-1, -1] -= r
    W = np.linalg.solve(g, Xtr.T.astype(np.float64) @ ytr / len(Xtr))
    return Xte @ W, Xtr @ W


class CNN(torch.nn.Module):
    def __init__(self, shape, n_scalar):
        super().__init__()
        r, c, ch = shape
        self.shape = shape
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(ch, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1), torch.nn.ReLU())
        with torch.no_grad():
            n = self.conv(torch.zeros(1, ch, r, c)).numel()
        self.head = torch.nn.Sequential(torch.nn.Linear(n + n_scalar, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                        torch.nn.Linear(256, 8))

    def forward(self, x):
        r, c, ch = self.shape
        g = x[:, :r * c * ch].reshape(-1, r, c, ch).permute(0, 3, 1, 2)
        return self.head(torch.cat([self.conv(g).flatten(1), x[:, r * c * ch:]], 1))


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                       torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 8))

    def forward(self, x):
        return self.net(x)


def train_net(kind, Xtr, ytr, Xte, shape, n_scalar, epochs=(5, 10, 20, 40), seed=0, device='cpu'):
    torch.manual_seed(seed)
    model = (CNN(shape, n_scalar) if kind == 'cnn' else MLP(Xtr.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr_t = (torch.from_numpy(Xtr.toarray()) if sparse.issparse(Xtr) else torch.from_numpy(Xtr)).to(device)
    Xte_t = (torch.from_numpy(Xte.toarray()) if sparse.issparse(Xte) else torch.from_numpy(Xte)).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    out = {}
    for ep in range(1, max(epochs) + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = torch.mean((model(Xtr_t[b]) - ytr_t[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in epochs:
            model.eval()
            with torch.no_grad():
                out[ep] = (model(Xte_t).cpu().numpy(),
                           torch.cat([model(Xtr_t[i:i + 2048]) for i in range(0, len(Xtr_t), 2048)]).cpu().numpy())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--models', default='constant,ridge,mlp,cnn')
    p.add_argument('--device', default='cpu')
    p.add_argument('--splits', default='within,across')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '6')))
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((data.parent / 'results.json').read_text())
    cfg = run['config']
    eps = load_episodes(data)
    attach_streams(eps, cfg, data.parent / 'streams', 4)
    X = sample_matrix(eps)
    R = None
    if 'res_ridge' in args.models:
        R = np.concatenate(replay_features(eps, cfg, 'reservoir'))
    scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)]).astype(np.float32)
    deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    iters = np.concatenate([np.full(len(e['labels']), e['iter']) for e in eps])
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % args.folds) for i, lv in zip(order, uniq)}
    splits = {'within levels (by round)': iters % args.folds, 'across levels': np.array([fold_of[lv] for lv in levels])}
    shape = smw_screen_shape(cfg)
    n_scalar = X.shape[1] - int(np.prod(shape))
    print(f'{X.shape[0]} samples, {X.shape[1]} inputs, {X.nnz / X.shape[0]:.0f} non-zero per sample', flush=True)
    results = {}
    splits = {k: v for k, v in splits.items() if k.split()[0] in args.splits.split(',')}
    for split, fold in splits.items():
        for model in args.models.split(','):
            preds = {}
            for k in range(args.folds):
                te, tr = fold == k, fold != k
                if model == 'constant':
                    preds.setdefault('', [np.zeros_like(y), []])
                    preds[''][0][te] = y[tr].mean(0)
                    preds[''][1].append(score(np.tile(y[tr].mean(0), (tr.sum(), 1)), y[tr], deadly[tr]))
                elif model in ('ridge', 'res_ridge'):
                    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                        if model == 'ridge':
                            pt, pi = ridge(X[tr], y[tr], X[te], alpha)
                        else:  # reservoir features already carry the bias column
                            pt, pi = ridge(R[tr][:, :-1], y[tr], R[te][:, :-1], alpha)
                        preds.setdefault(f'alpha {alpha}', [np.zeros_like(y), []])
                        preds[f'alpha {alpha}'][0][te] = pt
                        preds[f'alpha {alpha}'][1].append(score(pi, y[tr], deadly[tr]))
                else:
                    res = train_net(model, X[tr], y[tr], X[te], shape, n_scalar, device=args.device)
                    for ep, (pt, pi) in res.items():
                        preds.setdefault(f'{ep} epochs', [np.zeros_like(y), []])
                        preds[f'{ep} epochs'][0][te] = pt
                        preds[f'{ep} epochs'][1].append(score(pi, y[tr], deadly[tr]))
            for variant, (pt, ins) in preds.items():
                s = score(pt, y, deadly)
                s['q_in_sample'] = round(float(np.mean([r['q_chosen'] for r in ins])), 4)
                s['deadly_in_sample'] = round(float(np.mean([r['deadly'] for r in ins])), 4)
                results[f'{split} | {model} {variant}'.strip()] = s
                print(f'{split:26s} {model:8s} {variant:12s} held-out q {s["q_chosen"]:.4f} deadly {s["deadly"]:.4f} | '
                      f'fitted q {s["q_in_sample"]:.4f} deadly {s["deadly_in_sample"]:.4f}', flush=True)
            (out / 'ceiling.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
