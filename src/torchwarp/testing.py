"""Official implementations, used by the test suite as a correctness check.

The official uDTW and JEANIE code is at github.com/LeiWangR/uDTW and
github.com/LeiWangR/JEANIE and is not redistributed with torchwarp.
``load_reference`` downloads it at a pinned commit into
``~/.cache/torchwarp/reference`` (override with ``TORCHWARP_REFERENCE_DIR``)
and imports it.
"""

import importlib
import os
import sys
import urllib.request
from pathlib import Path

_UPSTREAM = {
    "udtw": dict(
        repo="LeiWangR/uDTW",
        commit="608c3abbb6b238d2520a18ac4595813aa3c22dd1",
        files=["udtw/__init__.py", "udtw/core.py"],
    ),
    "jeanie": dict(
        repo="LeiWangR/JEANIE",
        commit="9c275db0be83e2ed9e979f573ac280133ca389d9",
        files=["jeanie/__init__.py", "jeanie/alignment.py", "jeanie/distances.py",
               "jeanie/fvm.py"],
    ),
}


def _cache_root():
    env = os.environ.get("TORCHWARP_REFERENCE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "torchwarp" / "reference"


def load_reference(name, timeout=30):
    """Return the official package ``"udtw"`` or ``"jeanie"`` (downloaded once).

    Raises RuntimeError if it is not cached and cannot be downloaded.
    """
    spec = _UPSTREAM[name]
    root = _cache_root() / "{}-{}".format(spec["repo"].replace("/", "_"), spec["commit"][:12])
    for rel in spec["files"]:
        path = root / rel
        if path.exists():
            continue
        url = "https://raw.githubusercontent.com/{}/{}/{}".format(
            spec["repo"], spec["commit"], rel)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
        except Exception as exc:
            raise RuntimeError("could not download reference {} from {}: {}".format(
                name, url, exc)) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(name)
