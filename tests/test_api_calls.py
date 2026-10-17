"""The README, docs, examples and experiments match the API.

Each Python file and each ```python block of a Markdown file is parsed (not
run). Every torchwarp name it uses must exist, and every call of a torchwarp
function or class must bind to its signature (positional count, keywords).
The signatures in docs/api.md must equal those in the code, and every public
function and module has one.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest
import torch

import torchwarp
import torchwarp.functional as TF
from torchwarp import paths, plot

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md"),
                  *(ROOT / "examples").glob("*.py"), *(ROOT / "experiments").glob("*.py")])


API = ROOT / "docs" / "api.md"
# a signature block: torchwarp.Name(...) or TF.name(...) (TF = torchwarp.functional)
SIGNATURE = re.compile(r"^((?:torchwarp|TF)(?:\.\w+)+)\((.*)\)\s*$", flags=re.S)
ROOTS = {"torchwarp": torchwarp, "TF": TF}


def _blocks(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [text]
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    if path == API:                                 # signature blocks are checked below
        blocks = [b for b in blocks if not SIGNATURE.match(b.strip())]
    return blocks


def _resolve(node, names):
    """The torchwarp object an expression refers to, or None if not torchwarp."""
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, names)
        if base is None or not inspect.ismodule(base):
            return None
        if not hasattr(base, node.attr):
            raise AttributeError("{}.{} does not exist".format(base.__name__, node.attr))
        return getattr(base, node.attr)
    return None


def _check_call(call, obj):
    if not callable(obj):
        return
    sig = inspect.signature(obj)
    if any(isinstance(a, ast.Starred) for a in call.args):
        args = []                                   # positional count unknown
    else:
        args = [None] * len(call.args)
    kwargs = {k.arg: None for k in call.keywords if k.arg is not None}
    sig.bind_partial(*args, **kwargs)


def _problems(path):
    names = dict(ROOTS)                             # docs use torchwarp.* / TF.* without importing
    out = []
    for i, code in enumerate(_blocks(path)):
        tree = ast.parse(code)
        for node in ast.walk(tree):                 # imports first: blocks use names freely
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "torchwarp" or a.name.startswith("torchwarp."):
                        if a.asname:                # import torchwarp.functional as TF
                            names[a.asname] = importlib.import_module(a.name)
                        else:
                            names["torchwarp"] = torchwarp
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torchwarp"):
                module = importlib.import_module(node.module)
                for a in node.names:
                    try:                            # a name, or a submodule (torchwarp.plot)
                        obj = getattr(module, a.name, None) or importlib.import_module(
                            "{}.{}".format(node.module, a.name))
                    except ImportError:
                        out.append("block {}: {} has no {}".format(i, node.module, a.name))
                        continue
                    names[a.asname or a.name] = obj
        for node in ast.walk(tree):
            try:
                obj = _resolve(node.func if isinstance(node, ast.Call) else node, names)
                if isinstance(node, ast.Call) and obj is not None:
                    _check_call(node, obj)
            except (AttributeError, TypeError) as exc:
                out.append("block {} line {}: {}".format(i, getattr(node, "lineno", "?"), exc))
    return out


@pytest.mark.parametrize("path", SOURCES, ids=[str(p.relative_to(ROOT)) for p in SOURCES])
def test_api_calls(path):
    assert _problems(path) == []


def _params(sig):
    """[(name, kind, default)] with defaults compared by value."""
    out = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        kind = "kw" if p.kind is p.KEYWORD_ONLY else "pos"
        out.append((p.name, kind, None if p.default is p.empty else repr(p.default)))
    return out


def _documented():
    """{object: problem or None} for each signature block of docs/api.md."""
    out = {}
    for block in re.findall(r"```python\n(.*?)```", API.read_text(encoding="utf-8"), flags=re.S):
        m = SIGNATURE.match(block.strip())
        if m is None:
            continue
        root, *attrs = m.group(1).split(".")
        obj = ROOTS[root]
        for attr in attrs:
            obj = getattr(obj, attr)
        ns = {}
        exec("def f({}):\n    pass".format(m.group(2)), ns)
        doc = inspect.signature(ns["f"])
        got = _params(inspect.signature(obj))
        out[obj] = None if _params(doc) == got else "{}{} != code {}".format(
            m.group(1), doc, inspect.signature(obj))
    return out


def test_api_md_signatures():
    documented = _documented()
    assert [p for p in documented.values() if p] == []
    public = [getattr(torchwarp, n) for n in torchwarp.__all__]
    public += [getattr(TF, n) for n in TF.__all__]
    public += [getattr(paths, n) for n in paths.__all__] + [getattr(plot, n) for n in plot.__all__]
    public = [o for o in public if inspect.isfunction(o)
              or (inspect.isclass(o) and issubclass(o, torch.nn.Module))]
    assert [o.__name__ for o in public if o not in documented] == []
