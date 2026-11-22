#!/usr/bin/env python
"""
Does "which buttons are deadly here" transfer to unseen levels better than "which button did the teacher pick"?

Targets: deadly[t, a] = the teacher's look-ahead found action a deadly at moment t (only for tried actions).
Models: per-action ridge on the reservoir features, and an MLP on the raw observation (GPU) with a masked
binary cross-entropy. Splits: across levels (4 folds of held-out levels) and within levels (held-out DAgger
rounds). Reports the danger AUC and a "go right unless deadly" policy: take the first button of the constant
preference order (from the training data) whose predicted danger is below a threshold.

    env -u HSA_OVERRIDE_GFX_VERSION PYTHONPATH=scripts .venv-gfx1010/bin/python scripts/smw_danger.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from echomario.imitation import make_targets
from smw_readout_lab import attach_streams, load_episodes


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


DEADLY_DIR = __import__('os').environ.get('DEADLY_DIR', '')  # corrected deadly labels, if set


def main() -> None:
    dev = __import__('os').environ.get('DEVICE', 'cuda')
    data = Path('runs/smw/abc_v2/data')
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    attach_streams(eps, cfg, data.parent / 'streams', 2)
    X = sparse.vstack([e['stream'][e['sample_pos']] for e in eps]).tocsr().astype(np.float32)
    S = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    Y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(S, labels)])
    tried = np.isfinite(S)  # untried actions are stored as -inf
    D = np.where(tried, S, 0.0) <= -1000  # deadly among tried actions (training target)
    if DEADLY_DIR:  # corrected labels (smw_relabel_deadly.py): survivable buttons are not deadly
        D = np.concatenate([np.load(Path(DEADLY_DIR) / f"{e['file']}.npy") for e in eps])
    D_eval = np.nan_to_num(S, neginf=-1e9) <= -1000  # as in the other labs: an untried choice counts as deadly
    F = np.load(data.parent / 'feats_reservoir_baseline.npy').astype(np.float64)
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    rounds = np.concatenate([np.full(len(e['labels']), e['iter']) for e in eps])
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    lvfold = np.array([fold_of[lv] for lv in levels])
    print(f'{len(Y)} moments; tried share per action {tried.mean(0).round(2)}; deadly share among tried '
          f'{(D & tried).sum() / tried.sum():.3f}', flush=True)
    Xg = torch.from_numpy(X.toarray()).to(dev)

    def ridge_danger(tr, te, alpha):
        P = np.zeros((len(te), 8))
        for a in range(8):
            r = tr[tried[tr, a]]
            Fa = F[r]
            G = Fa.T @ Fa / len(r)
            reg = alpha * np.trace(G) / G.shape[0]
            G[np.diag_indices_from(G)] += reg
            G[-1, -1] -= reg
            w = np.linalg.solve(G, Fa.T @ D[r, a].astype(np.float64) / len(r))
            P[:, a] = F[te] @ w
        return P

    def mlp_danger(tr, te, steps=600, seed=0):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.ReLU(), torch.nn.Linear(512, 512),
                                  torch.nn.ReLU(), torch.nn.Linear(512, 8)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
        Dg = torch.from_numpy(D.astype(np.float32)).to(dev)
        Mg = torch.from_numpy(tried.astype(np.float32)).to(dev)
        trg = torch.from_numpy(tr).to(dev)
        g = torch.Generator().manual_seed(seed)
        out = {}
        for step in range(1, steps + 1):
            b = trg[torch.randint(0, len(tr), (512,), generator=g).to(dev)]
            l = torch.nn.functional.binary_cross_entropy_with_logits(net(Xg[b]), Dg[b], weight=Mg[b], reduction='sum')
            opt.zero_grad()
            (l / Mg[b].sum()).backward()
            opt.step()
            if step in (100, 200, 400, 600):
                with torch.no_grad():
                    out[step] = torch.sigmoid(net(Xg[torch.from_numpy(te).to(dev)])).cpu().numpy()
        return out

    def policy_eval(P, tr, te, thresholds):
        pref = np.argsort(-Y[tr].mean(0))  # constant preference order from the training data
        res = {}
        for th in thresholds:
            ok = P[:, pref] < th
            first = np.where(ok.any(1), ok.argmax(1), 0)
            a = pref[first]
            res[th] = (float(Y[te, a].mean()), float(D_eval[te, a].mean()))
        return res, pref[0]

    for split, fold_ids in (('across levels', lvfold), ('within levels', rounds % 4)):
        coll = {}
        for k in range(4):
            tr, te = np.nonzero(fold_ids != k)[0], np.nonzero(fold_ids == k)[0]
            const = int(np.argmax(Y[tr].mean(0)))
            coll.setdefault('const', []).append((Y[te, const].mean(), D_eval[te, const].mean(), len(te)))
            preds = {f'ridge a={al}': ridge_danger(tr, te, al) for al in (0.01, 0.1, 1.0)}
            for st, P in mlp_danger(tr, te, seed=k).items():
                preds[f'mlp {st} steps'] = P
            for name, P in preds.items():
                m = tried[te]
                a_all = auc(P[m], D[te][m])
                pol, _ = policy_eval(P, tr, te, (0.05, 0.1, 0.2, 0.3, 0.5))
                coll.setdefault(name, []).append((a_all, pol, len(te)))
        print(f'\n== {split} ==', flush=True)
        n = np.array([c[2] for c in coll['const']], float)
        cq = np.average([c[0] for c in coll['const']], weights=n)
        cd = np.average([c[1] for c in coll['const']], weights=n)
        print(f'constant button: q {cq:.4f}, deadly {cd:.3f}')
        for name, v in coll.items():
            if name == 'const':
                continue
            a_ = np.nanmean([c[0] for c in v])
            line = f'{name:18s} danger AUC {a_:.3f} | go-right-unless-deadly policy:'
            for th in v[0][1]:
                q = np.average([c[1][th][0] for c in v], weights=n)
                d = np.average([c[1][th][1] for c in v], weights=n)
                line += f'  th {th}: q {q:.4f} ({q - cq:+.4f}) dead {d:.3f}'
            print(line, flush=True)


if __name__ == '__main__':
    main()
