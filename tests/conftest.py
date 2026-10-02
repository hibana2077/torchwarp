import sys
from pathlib import Path

import pytest

# make `from conftest import reference` work from the test modules
sys.path.insert(0, str(Path(__file__).resolve().parent))


def reference(name):
    """Upstream reference package ("udtw" or "jeanie"), or skip the module.

    The reference code is downloaded once from github.com/LeiWangR at a
    pinned commit (see torchwarp.testing).
    """
    from torchwarp.testing import load_reference

    try:
        return load_reference(name)
    except RuntimeError as exc:
        pytest.skip("reference implementation unavailable: {}".format(exc),
                    allow_module_level=True)
