#!/usr/bin/env python3
"""Run the pinned 217 owner with a narrow D696 typed-ancestry loader adapter."""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HANDOFF = HERE / "prepare_tetgen_handoff_697_typedD.py"
P = HERE.parents[1]
BUILD = P / "volume-worker/native-domain-217/build_native_domain_217.py"
BUILD_SHA = "bd266b63967f8120b95f2df817704defe9608dec011eed3013f510517c597b66"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_typed_owner():
    if sha(BUILD) != BUILD_SHA:
        raise ValueError("pinned native-domain-217 owner SHA changed")
    owner = import_path("native_domain_217_typedD_697_owner", BUILD)
    adapter = import_path("native_domain_217_typedD_697_adapter", HANDOFF)
    base_loader = owner.load_surface

    def load_surface(path, layer, collar=None):
        return adapter.load_surface_217(owner, path, layer, collar,
                                        legacy_loader=base_loader)

    owner.load_surface = load_surface
    return owner

def main() -> int:
    return int(load_typed_owner().main())

if __name__ == "__main__":
    raise SystemExit(main())
