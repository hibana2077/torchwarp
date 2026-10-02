"""JIT loader for the CUDA/HIP extensions.

Each extension ("udtw", "jeanie") is compiled on first use with
``torch.utils.cpp_extension`` and cached under ``~/.cache/torch_extensions``.
On ROCm builds of PyTorch the same sources are hipified automatically.

Environment variables:
  TORCHWARP_FAST_MATH=1  build a variant whose float32 exp/log use the
                         hardware approximations (__expf/__logf); float64 is
                         unaffected.
  TORCHWARP_VERBOSE=1    show the compiler output.
  TORCHWARP_NO_CUDA=1    never build the extensions (always use the
                         portable PyTorch backend).

If an extension cannot be built, torchwarp falls back to the portable
PyTorch backend and emits a single warning saying why.
"""

import os
import warnings

import torch

_SOURCES = {"udtw": "udtw_cuda.cu", "jeanie": "jeanie_cuda.cu"}
_EXT = {}
_ERROR = {}


def load(name):
    """Return the compiled extension ``name``, or None if it cannot be built."""
    if name in _EXT:
        return _EXT[name]
    if name in _ERROR:
        return None
    if os.environ.get("TORCHWARP_NO_CUDA") == "1":
        _ERROR[name] = RuntimeError("disabled by TORCHWARP_NO_CUDA=1")
        return None
    if not torch.cuda.is_available():
        _ERROR[name] = RuntimeError("no CUDA/ROCm device available")
        return None
    try:
        from torch.utils.cpp_extension import load as _load

        fast_math = os.environ.get("TORCHWARP_FAST_MATH") == "1"
        src = os.path.join(os.path.dirname(__file__), "csrc", _SOURCES[name])
        _EXT[name] = _load(
            name="torchwarp_{}_cuda{}".format(name, "_fastmath" if fast_math else ""),
            sources=[src],
            extra_cuda_cflags=["-O3"] + (["-DDTW_FAST_MATH"] if fast_math else []),
            verbose=os.environ.get("TORCHWARP_VERBOSE") == "1",
        )
        return _EXT[name]
    except Exception as exc:  # build failure -> portable backend
        _ERROR[name] = exc
        warnings.warn(
            "torchwarp: could not build the {} CUDA extension ({}: {}). Falling back to "
            "the portable PyTorch backend, which is much slower on GPU. Set "
            "TORCHWARP_VERBOSE=1 to see the build log.".format(
                name, type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""),
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def build_error(name):
    return _ERROR.get(name)


def available(name):
    """True if the CUDA extension ``name`` is (or can be) built on this machine."""
    return load(name) is not None
