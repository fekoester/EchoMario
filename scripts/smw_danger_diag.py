#!/usr/bin/env python
"""
Is the teacher's "this button is deadly" explained by what is visible near Mario?

Builds ~70 hand-made local features from the v2 observation (pits ahead, walls ahead, enemies / projectiles /
hazards in boxes around Mario, Mario's speed and state) and fits a per-button logistic regression. If these
simple features predict danger on unseen levels clearly better than the reservoir / MLP did (AUC 0.53 / 0.61),
the learners are memorizing levels; if they are also near 0.5, the label itself does not follow from the local
picture (long look-ahead, pixel timing).

    .venv-gfx1010/bin/python scripts/smw_danger_diag.py     (CPU is enough)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from smw_danger import auc
from smw_readout_lab import attach_streams, load_episodes

R, C, CH = 22, 28, 15
FEET_R, MARIO_C = 12, 6  # rows_above, cols_behind of the run's grid


def local_features(X: sparse.csr_matrix) -> tuple[np.ndarray, list[str]]:
    G = X[:, :R * C * CH].toarray().reshape(-1, R, C, CH)
    sc = X[:, R * C * CH:].toarray()
    stand = G[..., 0:3].max(-1) + G[..., 3] + G[..., 9]  # solid, platform, slope, water, platform sprite
    feats, names = [], []
    below = stand[:, FEET_R:, :] > 0  # rows from the feet row down
    hazard_below = G[:, FEET_R:, :, 5].max(1)
    for dx in range(-1, 11):
        c = MARIO_C + dx
        feats.append(~below[:, :, c].any(1)); names.append(f'pit@{dx}')
        feats.append(hazard_below[:, c] > 0); names.append(f'hazard_below@{dx}')
        feats.append(G[:, FEET_R - 2:FEET_R, c, 0].max(1) > 0); names.append(f'wall@{dx}')
    boxes = {'near': (-2, 2, -4, 2), 'ahead': (3, 6, -5, 2), 'far': (7, 12, -6, 3), 'above': (-2, 3, -8, -3),
             'below': (-2, 4, 3, 8)}
    for ch, chn in ((7, 'stomp'), (8, 'harm'), (11, 'proj'), (5, 'hazard'), (9, 'platspr')):
        for bn, (c0, c1, r0, r1) in boxes.items():
            sub = G[:, max(0, FEET_R + r0):FEET_R + r1 + 1, MARIO_C + c0:MARIO_C + c1 + 1, ch]
            feats.append(sub.max((1, 2)) > 0); names.append(f'{chn}_{bn}')
    for i in range(sc.shape[1]):
        feats.append(sc[:, i]); names.append(f'scalar{i}')
        feats.append(sc[:, i] ** 2); names.append(f'scalar{i}^2')
    return np.stack(feats, 1).astype(np.float32), names


def logistic(Ftr, ytr, wtr, Fte, l2=1e-3, steps=300):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.from_numpy((Ftr - mu) / sd)
    B = torch.from_numpy((Fte - mu) / sd)
    y = torch.from_numpy(ytr.astype(np.float32))
    wt = torch.from_numpy(wtr.astype(np.float32))
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y, weight=wt, reduction='sum')
                / wt.sum() + l2 * (w ** 2).sum())
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (B @ w + b).numpy()


DEADLY_DIR = __import__('os').environ.get('DEADLY_DIR', '')  # corrected deadly labels, if set


def main() -> None:
    data = Path('runs/smw/abc_v2/data')
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    attach_streams(eps, cfg, data.parent / 'streams', 2)
    X = sparse.vstack([e['stream'][e['sample_pos']] for e in eps]).tocsr().astype(np.float32)
    S = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    tried = np.isfinite(S)
    D = np.where(tried, S, 0.0) <= -1000
    if DEADLY_DIR:
        D = np.concatenate([np.load(Path(DEADLY_DIR) / f"{e['file']}.npy") for e in eps])
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    rounds = np.concatenate([np.full(len(e['labels']), e['iter']) for e in eps])
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    lvfold = np.array([fold_of[lv] for lv in levels])
    Fh, names = local_features(X)
    print(f'{len(Fh)} moments, {Fh.shape[1]} local features', flush=True)
    nd, nt = (D & tried).sum(1), tried.sum(1)
    mixed = (nd > 0) & (nd < nt)
    for split, fid in (('across levels', lvfold), ('within levels', rounds % 4)):
        P = np.zeros(D.shape)
        for k in range(4):
            tr, te = fid != k, fid == k
            for a in range(8):
                if tried[:, a].sum() == 0:
                    continue
                P[te, a] = logistic(Fh[tr], D[tr, a], tried[tr, a].astype(float), Fh[te])
        m = tried
        a_all = auc(P[m], D[m])
        # discrimination inside moments where the button matters: rank the buttons of one moment
        Pc = P - np.nanmean(np.where(tried, P, np.nan), 1, keepdims=True)
        a_mixed = auc(Pc[mixed][tried[mixed]], D[mixed][tried[mixed]])
        any_d = D.any(1)
        a_moment = auc(np.where(tried, P, -np.inf).max(1), any_d)
        print(f'{split:14s} local-feature logistic: danger AUC {a_all:.3f} | "is this moment dangerous" AUC '
              f'{a_moment:.3f} | "which button" AUC in mixed moments {a_mixed:.3f}', flush=True)


if __name__ == '__main__':
    main()
