#!/usr/bin/env python3
"""030d reader bridge for frozen recorded-flow contract 009 and receipt builder 013.

The bridge preserves 009/029 behavior. For the new 030d source-control it
revalidates the complete versioned 038 control and closure, then constructs the
narrow frozen-013 compatibility view. That view retains the actual 030d source,
binary, and support header. The older compiler contract is used only because
013's schema is frozen; the actual 030d compile command and all dependencies
are independently checked by the 038 source-control/closure before projection.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

BASE_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-007-recorded-flow-scope/recorded_flow_contract_009_p029_source_controls.py")
BASE_SHA256 = "948165fd3c503c40d36855a4e424d9f6b3cd11cea404b3abce29578e2945e1b1"
BUILDER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/postpass-030d-canonical-038/build_source_control_038.py")
BUILDER_SHA256 = "4b1280a4de45b0ecbc2d6b8e33665e4644d77235c6b50b797c2797525cfe9ec1"
VARIANT_030D = "030d-contiguous-preference-lookup"
STATUS_030D = "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN"

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _import_pinned(path: Path, expected_sha: str, name: str):
    path = path.resolve(strict=True)
    if sha_file(path) != expected_sha:
        raise ValueError(f"pinned {name} source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = _import_pinned(BASE_PATH, BASE_SHA256, "recorded_flow_contract_009_pinned")
BUILDER = _import_pinned(BUILDER_PATH, BUILDER_SHA256, "source_control_builder_038_pinned")
_ORIGINAL_NESTED = BASE.verify_nested_control_refs
_ORIGINAL_PROJECTION = BASE._control_projection_for_013

def _validate_030d_control(control: dict[str, Any]) -> None:
    if not isinstance(control, dict) or control.get("variant") != VARIANT_030D:
        raise ValueError("030d source-control variant mismatch")
    if control.get("status") != STATUS_030D:
        raise ValueError("030d source-control must retain full allocator NOT_RUN status")
    with tempfile.TemporaryDirectory(prefix="h706-038-control-replay-") as temporary:
        expected, _ = BUILDER.build(Path(temporary) / "derived")
    if control != expected:
        raise ValueError("030d source-control differs from independent pinned-input reconstruction")

def verify_nested_control_refs(control: dict[str, Any]) -> None:
    if isinstance(control, dict) and control.get("variant") == VARIANT_030D:
        _validate_030d_control(control)
        view = copy.deepcopy(control)
        view["solver_contract"].pop("status")
        BASE.verify_nested_artifact_refs(view, "030d source control")
        return
    _ORIGINAL_NESTED(control)

def _projection_030d(
    control: dict[str, Any], *,
    rejected_parent_exact_parity: bool = False,
    mass_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_030d_control(control)
    if rejected_parent_exact_parity:
        raise ValueError("030d does not project a rejected-parent parity claim")
    if mass_receipt is None:
        raise ValueError("030d receipt projection requires the actual source-only mass receipt")
    parents = control.get("parent_manifests")
    if not isinstance(parents, list) or not parents:
        raise ValueError("030d source-control is missing its pinned P029 compatibility parent")
    parent_path, _ = BASE._ref_with_optional_bytes(parents[0], "030d P029 compatibility parent")
    parent = BASE.read_json(parent_path)
    if parent.get("variant") != "029-canonical-source-tet-sites":
        raise ValueError("030d compatibility parent is not canonical P029")
    projection = copy.deepcopy(_ORIGINAL_PROJECTION(parent, mass_receipt=mass_receipt))
    for key in ("source", "binary", "support_header"):
        value = control.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"030d source-control lacks {key}")
        projection[key] = BASE._strip_verified_bytes(value)
    rows = control.get("focused_controls")
    if not isinstance(rows, list):
        raise ValueError("030d focused controls are malformed")
    pass_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "PASS"]
    if len(pass_rows) != 4:
        raise ValueError("030d compatibility projection requires the four exact parity PASS controls")
    projection["focused_controls"] = [
        {"source": BASE._strip_verified_bytes(row["source"]),
         "binary": BASE._strip_verified_bytes(row["binary"]), "status": "PASS"}
        for row in pass_rows
    ]
    return projection

def _control_projection_for_013(
    control: dict[str, Any], *,
    rejected_parent_exact_parity: bool = False,
    mass_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(control, dict) and control.get("variant") == VARIANT_030D:
        return _projection_030d(
            control,
            rejected_parent_exact_parity=rejected_parent_exact_parity,
            mass_receipt=mass_receipt,
        )
    return _ORIGINAL_PROJECTION(
        control,
        rejected_parent_exact_parity=rejected_parent_exact_parity,
        mass_receipt=mass_receipt,
    )

# verify_source_control resolves these names from BASE globals. Only dispatch
# points change; the pinned 009 run, bundle, and closure validation stays intact.
BASE.verify_nested_control_refs = verify_nested_control_refs
BASE._control_projection_for_013 = _control_projection_for_013

def verify_source_control(
    run: dict[str, Any], bundle: dict[str, Any],
    mass_receipt: dict[str, Any] | None = None,
) -> None:
    BASE.verify_source_control(run, bundle, mass_receipt)

def legacy_projection(control: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _ORIGINAL_PROJECTION(control, **kwargs)

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-control", required=True, type=Path)
    parser.add_argument("--source-closure", required=True, type=Path)
    args = parser.parse_args()
    try:
        control = json.loads(args.source_control.read_text(encoding="utf-8"))
        verify_nested_control_refs(control)
        BASE.verify_source_closure(
            args.source_closure.resolve(), sha_file(args.source_closure),
            args.source_control.resolve(), sha_file(args.source_control), control,
        )
        print(json.dumps({
            "schema": "numi.h706.recorded-flow-030d-source-preflight.v1",
            "status": STATUS_030D,
            "source_control_sha256": sha_file(args.source_control),
            "source_closure_sha256": sha_file(args.source_closure),
            "variant": VARIANT_030D,
        }, sort_keys=True))
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())

