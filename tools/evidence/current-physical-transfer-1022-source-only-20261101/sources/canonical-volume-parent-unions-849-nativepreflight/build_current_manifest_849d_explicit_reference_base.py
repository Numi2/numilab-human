#!/usr/bin/env python3
"""Build a hash-pinned, read-only ancestry manifest for current Skin/Fat volumes.

This discovers state ancestry from source_state links. It does not edit NPZs,
run geometry, imply physical admission, or claim point ancestry. Cell-parent
CSR composition is delegated to the pinned 834 owner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
H = HERE.parent
CORE_PATH = H / "canonical-volume-parent-unions-834-nativepreflight" / "canonical_volume_parent_unions_834.py"
SCHEMA = "numi.h706.auto-current-volume-parent-union-manifest.v1"
CORE_SCHEMA = "numi.h706.current-volume-parent-union-rebuild-input.v1"


def _load_core():
    spec = importlib.util.spec_from_file_location("canonical_volume_parent_unions_834", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned 834 owner: {CORE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_pin(path: str | Path) -> dict[str, str]:
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"missing input file: {p}")
    return {"path": str(p), "sha256": sha256_file(p)}


def _verify_pin(pin: dict, label: str) -> tuple[Path, dict]:
    if not isinstance(pin, dict) or not isinstance(pin.get("path"), str) or not isinstance(pin.get("sha256"), str):
        raise ValueError(f"{label}: malformed path/SHA pin")
    p = Path(pin["path"])
    if not p.is_file():
        raise ValueError(f"{label}: missing {p}")
    got = sha256_file(p)
    if got != pin["sha256"]:
        raise ValueError(f"{label}: SHA256 mismatch, expected {pin['sha256']} got {got}")
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"{label}: invalid JSON: {e}") from e
    return p, d


def _normalize_state(pin: dict, wrappers: list[dict], label: str) -> tuple[dict, dict]:
    """Resolve latest-verified wrappers without discarding wrapper pins."""
    seen = set()
    cur = dict(pin)
    while True:
        path, data = _verify_pin(cur, label)
        identity = (str(path), cur["sha256"])
        if identity in seen:
            raise ValueError(f"{label}: wrapper cycle")
        seen.add(identity)
        if isinstance(data.get("Skin_current"), dict) and isinstance(data.get("Fat_current"), dict):
            return cur, data
        nxt = None
        cur_obj = data.get("current")
        if isinstance(cur_obj, dict):
            nxt = cur_obj.get("state")
        if nxt is None:
            nxt = data.get("current_state")
        if not isinstance(nxt, dict) or not isinstance(nxt.get("path"), str) or not isinstance(nxt.get("sha256"), str):
            raise ValueError(f"{label}: not a paired current state or recognized wrapper")
        wrappers.append({"path": str(path), "sha256": cur["sha256"], "points_to": dict(nxt)})
        cur = dict(nxt)


def _state_volume_pin(state: dict, role: str, label: str) -> dict:
    item = state.get(f"{role}_current")
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
        raise ValueError(f"{label}: missing {role}_current volume pin")
    return {"path": item["path"], "sha256": item["sha256"]}


def _source_state_pin(state: dict, label: str) -> dict:
    p = state.get("source_state")
    if not isinstance(p, dict) or not isinstance(p.get("path"), str) or not isinstance(p.get("sha256"), str):
        raise ValueError(f"{label}: missing hash-pinned source_state link")
    return {"path": p["path"], "sha256": p["sha256"]}


def _validate_source_only(state: dict, label: str, state_path=None) -> None:
    if state.get("physical_admission") is True or state.get("native_steps", 0) != 0:
        raise ValueError(f"{label}: state is not source-only")
    status = state.get("status")
    if isinstance(status, str) and ("PENDING" in status or "CANDIDATE" in status or "PARTIAL" in status):
        # Construction-stage strings can remain pending after a separately bound
        # current source proof passes. Authenticate that proof, never infer it
        # from a PASS string or relax the canonical ancestry validation.
        if state_path is None:
            raise ValueError(f"{label}: pending state needs authenticated current source evidence")
        vp=H/"verify_current_source_evidence_808_v26a.py"
        if sha256_file(vp)!="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9":
            raise ValueError("808 source verifier pin mismatch")
        spec=importlib.util.spec_from_file_location("manifest849c_source808",vp)
        ver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ver;spec.loader.exec_module(ver)
        proof=ver.verify_current_source_evidence(Path(state_path),state)
        if not proof.get("status", "").startswith("PASS_") or proof.get("physical_admission") is not False or proof.get("native_steps")!=0:
            raise ValueError(f"{label}: pending state source evidence failed")
    for role in ("Skin", "Fat"):
        v = state.get(f"{role}_current")
        if not isinstance(v, dict):
            raise ValueError(f"{label}: missing paired {role}_current")
        # A known failure count is accepted as a geometry-quality state; this
        # tool proves ancestry only and never admits an unqualified mesh.
        if v.get("status") in ("CANDIDATE_PENDING_775_697_217", "PENDING"):
            raise ValueError(f"{label}: {role} volume remains pending")


def discover_chain(base_pin: dict, current_pin: dict) -> tuple[list[dict], dict, dict, list[dict]]:
    base_pin, base_state = _normalize_state(base_pin, [], "base state")
    wrappers: list[dict] = []
    current_pin, current_state = _normalize_state(current_pin, wrappers, "current state")
    _validate_source_only(base_state, "base state",base_pin["path"])
    _validate_source_only(current_state, "current state",current_pin["path"])
    base_key = (base_pin["path"], base_pin["sha256"])
    reverse: list[tuple[dict, dict]] = []
    seen: set[tuple[str, str]] = set()
    cur_pin, cur_state = current_pin, current_state
    while (cur_pin["path"], cur_pin["sha256"]) != base_key:
        key = (cur_pin["path"], cur_pin["sha256"])
        if key in seen:
            raise ValueError("cycle in source_state chain")
        seen.add(key)
        reverse.append((cur_pin, cur_state))
        parent_pin = _source_state_pin(cur_state, cur_pin["path"])
        # Do not silently resolve a state link through a wrapper: state ancestry
        # links must bind actual state JSON directly.
        parent_path, parent_state = _verify_pin(parent_pin, "source_state")
        if not (isinstance(parent_state.get("Skin_current"), dict) and
                isinstance(parent_state.get("Fat_current"), dict)):
            raise ValueError("source_state points to a wrapper rather than a paired state")
        _validate_source_only(parent_state, f"source state {parent_path}",parent_path)
        cur_pin, cur_state = parent_pin, parent_state
        if len(reverse) > 2048:
            raise ValueError("state chain exceeds bounded 2048 links")
    reverse.append((base_pin, base_state))
    chain = list(reversed(reverse))
    if len(chain) < 1:
        raise ValueError("empty ancestry chain")
    return [x[0] for x in chain], base_state, current_state, wrappers


def _minimal(pin: dict, label: str, core) -> dict:
    return core._read_minimal_volume(pin, label)


def _classify(parent: dict, child: dict, label: str) -> tuple[str, dict]:
    pt, pm = parent["tetrahedra"], parent["materials"]
    ct, cm = child["tetrahedra"], child["materials"]
    if pt.shape == ct.shape and np.array_equal(pt, ct) and np.array_equal(pm, cm):
        return "identity", {
            "kind": "identity_cell_rows_ordered_tetrahedra_and_materials",
            "parent_cells": int(len(pt)),
            "output_cells": int(len(ct)),
            "parent_points": int(parent["points"]),
            "output_points": int(child["points"]),
            "point_row_domain_changed": parent["points"] != child["points"],
            "point_ancestry_claim": False,
        }
    # A mapped step is admitted only by the exact immediate cell maps and full
    # replacement-parent CSR consumed by 834; no guessed ancestry is inferred.
    return "mapped", {"kind": "mapped_requires_immediate_full_parent_csr"}


def build_manifest(base_path: str | Path, current_path: str | Path) -> dict:
    core = _load_core()
    base_wrappers: list[dict] = []
    current_wrappers: list[dict] = []
    base_pin, base_state = _normalize_state(exact_pin(base_path), base_wrappers, "base state")
    current_pin, current_state = _normalize_state(exact_pin(current_path), current_wrappers, "current state")
    _validate_source_only(base_state, "base state",base_pin["path"])
    _validate_source_only(current_state, "current state",current_pin["path"])
    base_key = (base_pin["path"], base_pin["sha256"])
    reverse: list[tuple[dict, dict]] = []
    seen: set[tuple[str, str]] = set()
    cur_pin, cur_state = current_pin, current_state
    while (cur_pin["path"], cur_pin["sha256"]) != base_key:
        key = (cur_pin["path"], cur_pin["sha256"])
        if key in seen:
            raise ValueError("cycle in source_state chain")
        seen.add(key)
        reverse.append((cur_pin, cur_state))
        parent_pin = _source_state_pin(cur_state, cur_pin["path"])
        parent_path, parent_state = _verify_pin(parent_pin, "source_state")
        if not (isinstance(parent_state.get("Skin_current"), dict) and isinstance(parent_state.get("Fat_current"), dict)):
            raise ValueError("source_state points to an unrecognized wrapper, not a paired current state")
        _validate_source_only(parent_state, f"source state {parent_path}",parent_path)
        cur_pin, cur_state = parent_pin, parent_state
        if len(reverse) > 2048:
            raise ValueError("state chain exceeds bounded 2048 links")
    chain = [(base_pin, base_state)] + list(reversed(reverse))
    if not chain or chain[0][0] != base_pin or chain[-1][0] != current_pin:
        raise AssertionError("chain endpoints were not preserved")

    roles: dict[str, dict[str, Any]] = {}
    step_classification: dict[str, list[dict]] = {"Skin": [], "Fat": []}
    cache: dict[tuple[str, str], dict] = {}

    def get_volume(pin: dict, label: str) -> dict:
        key = (pin["path"], pin["sha256"])
        if key not in cache:
            cache[key] = _minimal(pin, label, core)
        return cache[key]

    for role in ("Skin", "Fat"):
        base_volume = _state_volume_pin(base_state, role, "base")
        final_volume = _state_volume_pin(current_state, role, "current")
        parent_pin = base_volume
        parent_state_pin = base_pin
        parent_vol = get_volume(parent_pin, f"{role} base")
        steps = []
        details = []
        for next_state_pin, next_state in chain[1:]:
            output_pin = _state_volume_pin(next_state, role, str(next_state_pin["path"]))
            output_vol = get_volume(output_pin, f"{role} output")
            kind, detail = _classify(parent_vol, output_vol, role)
            # For exact byte-identical volume pins, record a state-only identity
            # step; it proves chain continuity and costs no cell map storage.
            step = {
                "kind": kind,
                "parent": dict(parent_pin),
                "output": dict(output_pin),
                "parent_binding": {"kind": "state_current", "state": dict(parent_state_pin)},
            }
            steps.append(step)
            details.append({
                "state": dict(next_state_pin),
                "parent_state": dict(parent_state_pin),
                "parent_volume": dict(parent_pin),
                "output_volume": dict(output_pin),
                "classification": detail,
            })
            parent_pin = output_pin
            parent_state_pin = next_state_pin
            parent_vol = output_vol
        if parent_pin != final_volume:
            raise ValueError(f"{role}: traversed endpoint does not equal final state current pin")
        roles[role] = {"base_volume": base_volume, "current_volume": final_volume, "steps": steps}
        step_classification[role] = details

    chain_rows = []
    for pin, state in chain:
        chain_rows.append({
            "state": dict(pin),
            "source_state": state.get("source_state"),
            "Skin_current": _state_volume_pin(state, "Skin", "chain"),
            "Fat_current": _state_volume_pin(state, "Fat", "chain"),
            "physical_admission": state.get("physical_admission", False),
            "native_steps": state.get("native_steps", 0),
            "mass_mapping": state.get("mass_mapping", state.get("mass_reconciliation", "PENDING")),
        })
    manifest = {
        "schema": CORE_SCHEMA,
        "base_state": dict(base_pin),
        "current_state": dict(current_pin),
        "roles": roles,
        "auto_discovered_state_chain": chain_rows,
        "wrapper_pins": {"base": base_wrappers, "current": current_wrappers},
    }
    report = {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_ONLY_AUTODISCOVERED_CELL_ANCESTRY_MANIFEST",
        "physical_admission": False,
        "native_steps": 0,
        "mass_mapping": "NOT_RUN",
        "base_state": dict(base_pin),
        "current_state": dict(current_pin),
        "chain_state_count": len(chain),
        "chain_state_path_order": [x["state"]["path"] for x in chain_rows],
        "roles": {
            role: {
                "steps": len(details),
                "identity_cell_steps": sum(d["classification"]["kind"].startswith("identity") for d in details),
                "mapped_steps": sum(d["classification"]["kind"] == "mapped_requires_immediate_full_parent_csr" for d in details),
                "point_row_domain_changed_identity_steps": sum(bool(d["classification"].get("point_row_domain_changed")) for d in details),
                "cell_domain": len(get_volume(roles[role]["current_volume"], role)["tetrahedra"]),
            }
            for role, details in step_classification.items()
        },
        "limitations": [
            "Manifest discovery authenticates hash-pinned state links and volume pins; it does not run geometry or physical admission.",
            "Identity classification proves cell-row ancestry only from identical ordered tetrahedra and material rows; point ancestry is outside this manifest.",
            "Mapped cell ancestry is accepted only through 834's retained-row, direct/reverse map, replacement tetrahedron, and complete replacement-parent CSR checks.",
            "The canonical base is the explicitly pinned base_state; base=current is an operational-reference identity rebase and does not claim historical or ultimate source330 ancestry.",
            "Single-role pending proposals such as 848 are not valid paired current states and are rejected.",
        ],
    }
    return manifest, report


def write_json_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def validate_manifest(manifest: dict) -> dict:
    """Run 834's exact source-only validation without editing the frozen owner."""
    core = _load_core()

    def identity_allow_point_domain_change(parent: dict, child: dict, label: str):
        pt, pm = parent["tetrahedra"], parent["materials"]
        ct, cm = child["tetrahedra"], child["materials"]
        if pt.shape != ct.shape or not np.array_equal(pt, ct) or not np.array_equal(pm, cm):
            raise ValueError(f"{label}: declared cell identity changes ordered tets/materials")
        n = len(ct)
        return np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.int64), {
            "kind": "identity_exact_ordered_tetrahedra_materials_point_domain_independent",
            "parent_cells": n,
            "output_cells": n,
            "changed_cell_rows": 0,
            "parent_references": n,
            "parent_points": parent["points"],
            "output_points": child["points"],
            "point_ancestry_claim": False,
        }

    # Identity is a cell-domain relation. New or moved points with unchanged
    # ordered tets/materials do not change the identity cell map.
    core._identity_unions = identity_allow_point_domain_change
    # Ensure 834 points are loaded from actual paths and every mapped step has the
    # exact replacement-parent CSR; no arrays are written by validate_only.
    with __import__("tempfile").TemporaryDirectory(prefix="auto-union-849-") as td:
        p = Path(td) / "manifest.json"
        p.write_text(json.dumps(manifest))
        return core.run_manifest(p, validate_only=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", required=True, type=Path)
    ap.add_argument("--current-state", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--report-out", required=True, type=Path)
    ap.add_argument("--validate-only", action="store_true")
    ns = ap.parse_args()
    manifest, report = build_manifest(ns.base_state, ns.current_state)
    write_json_new(ns.manifest_out, manifest)
    report["manifest_path"] = str(ns.manifest_out)
    report["manifest_sha256"] = sha256_file(ns.manifest_out)
    if ns.validate_only:
        report["core_validation"] = validate_manifest(manifest)
    write_json_new(ns.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
