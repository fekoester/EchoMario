#!/usr/bin/env python
"""
Correctness and stability check for the GPU: trains the same MLP / CNN on GPU and CPU from identical
initial weights and identical batches and compares the losses step by step; repeats GPU runs to check
reproducibility; runs a long GPU loop to catch memory faults or hangs.

    .venv-gfx1010/bin/python scripts/gpu_stress.py
"""

from __future__ import annotations

import copy
import time

import torch


def mlp(d=9259):
    return torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(),
                               torch.nn.Linear(512, 8))


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(15, 32, 3, padding=1), torch.nn.ReLU(),
                                        torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
                                        torch.nn.Conv2d(64, 64, 3, stride=2, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(64 * 6 * 7, 256), torch.nn.ReLU(), torch.nn.Linear(256, 8))

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


def data(kind, n, gen):
    if kind == 'mlp':
        return (torch.rand(n, 9259, generator=gen) < 0.03).float(), torch.rand(n, 8, generator=gen)
    return (torch.rand(n, 15, 22, 28, generator=gen) < 0.05).float(), torch.rand(n, 8, generator=gen)


def train(model, X, Y, device, steps, batch, lr, seed):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    Xd, Yd = X.to(device), Y.to(device)
    losses = []
    for _ in range(steps):
        b = torch.randint(0, len(X), (batch,), generator=g).to(device)
        loss = torch.mean((model(Xd[b]) - Yd[b]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def main() -> None:
    print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    gen = torch.Generator().manual_seed(0)
    ok = True
    for kind, batch, lr in (('mlp', 512, 1e-3), ('mlp', 256, 3e-4), ('cnn', 256, 1e-3)):
        X, Y = data(kind, 20000, gen)
        torch.manual_seed(0)
        base = mlp() if kind == 'mlp' else CNN()
        t0 = time.time()
        cpu = train(copy.deepcopy(base), X, Y, 'cpu', 200, batch, lr, 1)
        t_cpu = time.time() - t0
        runs = []
        for rep in range(3):
            t0 = time.time()
            runs.append(train(copy.deepcopy(base), X, Y, 'cuda', 200, batch, lr, 1))
            torch.cuda.synchronize()
        t_gpu = time.time() - t0
        dev_cpu = max(abs(a - b) / max(abs(b), 1e-6) for a, b in zip(runs[0], cpu))
        dev_rep = max(abs(a - b) for r in runs[1:] for a, b in zip(r, runs[0]))
        finite = all(torch.isfinite(torch.tensor(r)).all() for r in runs)
        good = finite and dev_cpu < 0.05 and runs[0][-1] < runs[0][0]
        ok &= good
        print(f'{kind} batch {batch} lr {lr}: loss {runs[0][0]:.4f} -> {runs[0][-1]:.4f} (cpu {cpu[-1]:.4f}); max rel diff to CPU '
              f'{dev_cpu:.2e}; max diff between GPU repeats {dev_rep:.2e}; 200 steps cpu {t_cpu:.1f}s gpu {t_gpu:.2f}s -> '
              f'{"OK" if good else "PROBLEM"}', flush=True)
    # long loop: memory faults / hangs
    X, Y = data('mlp', 20000, gen)
    t0 = time.time()
    torch.manual_seed(0)
    losses = train(mlp(), X, Y, 'cuda', 5000, 256, 3e-4, 2)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(torch.tensor(losses)).all())
    ok &= finite
    print(f'long GPU run: 5000 MLP steps in {time.time() - t0:.1f}s, all losses finite: {finite}, final loss {losses[-1]:.4f}')
    print('ALL OK' if ok else 'PROBLEMS FOUND')


if __name__ == '__main__':
    main()
