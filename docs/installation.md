# Installation

## Requirements

* Python 3.10 or newer
* PyTorch 2.5 or newer
* numpy, ninja
* Optional, for `torchwarp.plot` and `Alignment.plot`: matplotlib 3.7 or
  newer (`pip install -e ".[plot]"`)
* For the CUDA kernels: an NVIDIA GPU and a CUDA toolkit (`nvcc`) whose
  major version matches the CUDA version of your PyTorch build. On ROCm
  builds of PyTorch the kernels are hipified automatically.

Without a GPU or without `nvcc`, torchwarp runs on its PyTorch backend.

## pip

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
pip install -r requirements.txt -e .
```

`requirements.txt` includes matplotlib, used by `torchwarp.plot` and the
experiments.

To choose the CUDA version of PyTorch, install it first, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## uv

Pick the PyTorch build with one extra: `cu126`, `cu128` or `cpu`.

```bash
git clone https://github.com/hibana2077/torchwarp && cd torchwarp
uv sync --extra cu126 --extra plot
```

Without an extra, PyTorch comes from PyPI, which on Linux is a CUDA 13
build; the kernels then need `nvcc` 13.x.

## The CUDA extensions

The two extensions (`udtw` and `jeanie`) are compiled the first time
they are used, with `torch.utils.cpp_extension`, and cached under
`~/.cache/torch_extensions`. The first call therefore takes a while
(about a minute per extension); later calls load the cached build.

If an extension cannot be built, torchwarp prints one warning with the
reason and uses the PyTorch backend, which gives the same results but
is much slower on the GPU.

To check that the kernels are used:

```python
from torchwarp import _cuda

print(_cuda.available("udtw"), _cuda.available("jeanie"))
```

## Environment variables

| variable | effect |
| --- | --- |
| `TORCHWARP_VERBOSE=1` | show the compiler output while building the extensions |
| `TORCHWARP_NO_CUDA=1` | never build the extensions; always use the PyTorch backend |
| `TORCHWARP_FAST_MATH=1` | build a variant whose float32 exp/log use the hardware approximations (float64 is unaffected) |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (PyTorch's own variable), for example `"7.5;8.6"` |

## Troubleshooting

| message or symptom | cause and fix |
| --- | --- |
| `Ninja is required to load C++ extensions` | `ninja` is not on `PATH`. Install it (`pip install ninja`) in the same environment, or activate the environment so its `bin/` is on `PATH`. |
| `could not build the ... CUDA extension` | Run again with `TORCHWARP_VERBOSE=1` to see the compiler error. The most common cause is an `nvcc` whose major version differs from `torch.version.cuda`. |
| First call is slow | The extension is being compiled. This happens once per machine and per GPU architecture. |
| Slow build or a build for the wrong GPU on a shared cache | Set `TORCH_CUDA_ARCH_LIST` to the architectures you use. |
| `CUDA backend unavailable` | `backend="cuda"` was requested but the inputs are not float32/float64 CUDA tensors, or the extension could not be built. Use `backend="auto"`. |

## Running the tests

```bash
pip install pytest scipy
python -m pytest
```

The tests compare torchwarp with the outputs of the official uDTW and
JEANIE code (values and gradients, float64), stored in
`tests/golden/reference.pt`, and check both backends on the CPU and, if
available, on the GPU. They do not need the official code or network
access. After adding or changing a test case, regenerate the stored
outputs (downloads the official code once):

```bash
python tests/golden/generate.py
```
