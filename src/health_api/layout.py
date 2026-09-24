"""Load `shared` under both package layouts this module is imported in.

Lambda sets `CodeUri` to `src/` and the handler to `health_api.ingest`.
`health_api` is then the top-level package, and `from ..shared` raises
`ImportError: attempted relative import beyond top-level package` — the
function never starts.

Tests import `src.health_api`, where `shared` really is the sibling
`src.shared` and the relative import is the one that resolves. Both have to
keep working: a fix that only satisfies the test layout is how this shipped.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def shared_module(name: str) -> ModuleType:
    package = __package__ or ""
    if "." in package:
        parent = package.rsplit(".", 1)[0]
        return importlib.import_module(f"{parent}.shared.{name}")
    return importlib.import_module(f"shared.{name}")
