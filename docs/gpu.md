# GPU: AMD Radeon RX 5700 (gfx1010)

The RX 5700 is RDNA1 (gfx1010). The usual ROCm PyTorch wheels contain no gfx1010 code, and the common workaround,
`HSA_OVERRIDE_GFX_VERSION=10.3.0` (run gfx1030 code), is not safe on this card: training runs sometimes
diverged after one epoch, crashed with "Memory access fault by GPU node-1" or hung at 99 % load.

## Working setup

AMD's TheRock builds publish PyTorch with native gfx1010 kernels (`[device-gfx1010]` extra). They bundle their own
ROCm user space, so the system ROCm (5.3 here) is not used or touched.

```bash
uv venv --python 3.13 .venv-gfx1010
uv pip install --python .venv-gfx1010/bin/python --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1010]"
uv pip install --python .venv-gfx1010/bin/python numpy scipy tqdm gymnasium
uv pip install --python .venv-gfx1010/bin/python --no-deps -e .
```

Installed: torch 2.15.0+rocm10.2.0, HIP 7.17, `amd-torch-device-gfx1010`, `rocm-sdk-device-gfx1010`.
Do **not** set `HSA_OVERRIDE_GFX_VERSION` with this environment.

## Checks (`scripts/gpu_stress.py`)

| test | result |
|---|---|
| matmul 4096², float32 | 4.2 TFLOPS, max relative difference to CPU 3e-6 |
| MLP 9259-512-512-8, 200 Adam steps, batch 512, lr 1e-3 (diverged with the override) | same losses as CPU (< 1 %), GPU repeats bit-identical, ~70x faster than CPU |
| CNN (MIOpen convolutions), 200 steps | same losses as CPU (3e-6), ~15x faster |
| 5000 MLP steps | no fault, no hang, all losses finite |

`scripts/smw_mlp_live.py` (two MLPs on the SMW data, train / held-out curves every epoch): 0.3 s per epoch on
the GPU against ~9 s on the CPU.
