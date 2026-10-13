"""Every name of the official uDTW and JEANIE packages exists in torchwarp,
with the same positional arguments and defaults; added options are keyword-only."""

import inspect

import pytest
import torch

import torchwarp

from conftest import reference

OFFICIAL = {"udtw": reference("udtw"), "jeanie": reference("jeanie")}
NAMES = [(pkg, name) for pkg, mod in OFFICIAL.items() for name in mod.__all__]
# the official uDTW and JEANIE code name the first argument of softmin differently
RENAMED = {("udtw", "softmin"): {"values": "x"}}


def _check(ref_fn, fn, renamed):
    ref = [p for p in inspect.signature(ref_fn).parameters.values() if p.name != "self"]
    got = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
    head, tail = got[:len(ref)], got[len(ref):]
    assert [renamed.get(p.name, p.name) for p in ref] == [p.name for p in head]
    assert [(p.kind, p.default) for p in ref] == [(p.kind, p.default) for p in head]
    assert all(p.kind is p.KEYWORD_ONLY for p in tail)


@pytest.mark.parametrize("pkg,name", NAMES, ids=["{}.{}".format(*n) for n in NAMES])
def test_official_signature(pkg, name):
    ref = getattr(OFFICIAL[pkg], name)
    got = getattr(torchwarp, name)
    renamed = RENAMED.get((pkg, name), {})
    if inspect.isclass(ref):
        _check(ref.__init__, got.__init__, renamed)
        _check(ref.forward, got.forward, renamed)
    else:
        _check(ref, got, renamed)


def test_softmin_matches_official():
    v = torch.randn(4, 5, dtype=torch.float64)
    for ref in (OFFICIAL["udtw"], OFFICIAL["jeanie"]):
        torch.testing.assert_close(torchwarp.softmin(v, 0.3, dim=1), ref.softmin(v, 0.3, dim=1))
    torch.testing.assert_close(torchwarp.softmin_weights(v, 0.3, dim=1),
                               OFFICIAL["udtw"].softmin_weights(v, 0.3, dim=1))
