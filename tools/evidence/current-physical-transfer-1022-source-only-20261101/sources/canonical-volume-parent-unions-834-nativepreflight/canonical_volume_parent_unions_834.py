#!/usr/bin/env python3
"""Rebuild canonical per-cell ancestry CSR across pinned current volume steps.

The CSR rows point into an explicitly pinned base volume cell-row domain. This
utility changes no geometry and makes no mass/native/physical-admission claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

SCHEMA = "numi.run055.current-immediate-parent-unions.v1"
MANIFEST_SCHEMA = "numi.h706.current-volume-parent-union-rebuild-input.v1"
REPORT_SCHEMA = "numi.h706.current-volume-parent-union-rebuild.v1"
CANON_FIELDS = (
    "canonical_parent_union_offsets",
    "canonical_parent_union_parent_rows",
)
REPLACED_METADATA = {
    "canonical_parent_union_operation_order",
    "canonical_parent_union_physical_status",
}
ADDED_METADATA = {
    "canonical_parent_union_base_state_path",
    "canonical_parent_union_base_state_sha256",
    "canonical_parent_union_base_volume_path",
    "canonical_parent_union_base_volume_sha256",
    "canonical_parent_union_current_state_path",
    "canonical_parent_union_current_state_sha256",
    "canonical_parent_union_current_volume_path",
    "canonical_parent_union_current_volume_sha256",
    "canonical_parent_union_parent_row_domain",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin_matches(pin: dict, label: str) -> Path:
    p = Path(pin["path"])
    if not p.is_file():
        raise ValueError(f"{label}: missing file {p}")
    got = sha256_file(p)
    if got != pin["sha256"]:
        raise ValueError(f"{label}: SHA256 mismatch: expected {pin['sha256']}, got {got}")
    return p


def read_json_pin(pin: dict, label: str) -> dict:
    p = pin_matches(pin, label)
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"{label}: invalid JSON: {e}") from e


def state_role_pin(state: dict, role: str, label: str) -> dict:
    key = f"{role}_current"
    v = state.get(key)
    if not isinstance(v, dict) or not isinstance(v.get("path"), str) or not isinstance(v.get("sha256"), str):
        raise ValueError(f"{label}: missing {key} volume pin")
    return {"path": v["path"], "sha256": v["sha256"]}


def verify_state_volume(state_pin: dict, state: dict, role: str, volume_pin: dict, label: str) -> None:
    role_pin = state_role_pin(state, role, label)
    if Path(role_pin["path"]) != Path(volume_pin["path"]) or role_pin["sha256"] != volume_pin["sha256"]:
        raise ValueError(f"{label}: state {role}_current does not bind listed volume")
    pin_matches(volume_pin, f"{label} {role} volume")


def _read_minimal_volume(pin: dict, label: str) -> dict:
    path = pin_matches(pin, label)
    try:
        with np.load(path, allow_pickle=False) as z:
            required = ("tetrahedra", "material_indices")
            missing = [k for k in required if k not in z.files]
            if missing:
                raise ValueError(f"{label}: missing geometry arrays {missing}")
            t = np.asarray(z["tetrahedra"])
            m = np.asarray(z["material_indices"])
            pkey = "points_world_m" if "points_world_m" in z.files else "points_world_m_f32"
            if pkey not in z.files:
                raise ValueError(f"{label}: missing point coordinates")
            p = np.asarray(z[pkey])
            if p.ndim != 2 or p.shape[1] != 3 or p.dtype.kind != "f":
                raise ValueError(f"{label}: malformed {pkey}")
            if t.ndim != 2 or t.shape[1] != 4 or t.dtype.kind not in "iu":
                raise ValueError(f"{label}: malformed tetrahedra")
            if m.shape != (len(t),) or m.dtype.kind not in "iu":
                raise ValueError(f"{label}: material domain differs from tetrahedra")
            if len(t) and (int(t.min()) < 0 or int(t.max()) >= len(p)):
                raise ValueError(f"{label}: tetrahedron point index outside point domain")
            if "points_world_m" in z.files and "points_world_m_f32" in z.files:
                p32 = np.asarray(z["points_world_m_f32"])
                if p32.shape != p.shape or p32.dtype.kind != "f":
                    raise ValueError(f"{label}: points_world_m_f32 domain mismatch")
                if not np.array_equal(np.asarray(p, dtype=np.float32), np.asarray(p32, dtype=np.float32)):
                    raise ValueError(f"{label}: F32 coordinate aliases disagree")
            for key in ("boundary_faces", "desired_boundary_faces"):
                if key in z.files:
                    f = np.asarray(z[key])
                    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
                        raise ValueError(f"{label}: malformed {key}")
                    if len(f) and (int(f.min()) < 0 or int(f.max()) >= len(p)):
                        raise ValueError(f"{label}: {key} point index outside point domain")
            # Only current cell arrays used by this protocol are loaded. Historical
            # point/Steiner/source lineage remains opaque and byte-preserved.
            return {
                "path": str(path),
                "sha256": pin["sha256"],
                "points": len(p),
                "tetrahedra": t.astype(np.int64, copy=False),
                "materials": m.astype(np.int64, copy=False),
                "point_shape": list(p.shape),
            }
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"{label}: cannot load NPZ: {e}") from e


def _validate_csr(offsets: np.ndarray, rows: np.ndarray, nrows: int, row_limit: int, label: str) -> None:
    if offsets.ndim != 1 or offsets.shape != (nrows + 1,) or offsets.dtype.kind not in "iu":
        raise ValueError(f"{label}: CSR offsets must have nrows+1 entries")
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError(f"{label}: CSR rows must be a one-dimensional integer array")
    off = offsets.astype(np.int64, copy=False)
    rr = rows.astype(np.int64, copy=False)
    if int(off[0]) != 0 or int(off[-1]) != len(rr) or np.any(off[1:] < off[:-1]):
        raise ValueError(f"{label}: malformed CSR offsets")
    if len(rr) and (int(rr.min()) < 0 or int(rr.max()) >= row_limit):
        raise ValueError(f"{label}: CSR parent row outside immediate parent domain")
    for i in range(nrows):
        s, e = int(off[i]), int(off[i + 1])
        if e > s and np.any(rr[s + 1:e] <= rr[s:e - 1]):
            raise ValueError(f"{label}: row {i} must be sorted and unique")


def _identity_unions(parent: dict, child: dict, label: str) -> tuple[np.ndarray, np.ndarray, dict]:
    pt, pm = parent["tetrahedra"], parent["materials"]
    ct, cm = child["tetrahedra"], child["materials"]
    if pt.shape != ct.shape or not np.array_equal(pt, ct) or not np.array_equal(pm, cm):
        raise ValueError(f"{label}: declared identity step changes ordered tetrahedra/materials")
    if parent["points"] != child["points"]:
        raise ValueError(f"{label}: identity cell step changes point-row domain")
    n = len(ct)
    offsets = np.arange(n + 1, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)
    return offsets, rows, {
        "kind": "identity_exact_ordered_tetrahedra_materials",
        "parent_cells": n,
        "output_cells": n,
        "changed_cell_rows": 0,
        "parent_references": n,
    }


def _mapped_step_unions(parent: dict, child: dict, output_pin: dict, label: str, step_arrays: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    pt, pm = parent["tetrahedra"], parent["materials"]
    ct, cm = child["tetrahedra"], child["materials"]
    nparent, nchild = len(pt), len(ct)
    common = (
        "retained_input_cell_rows",
        "replacement_tetrahedra",
        "replacement_parent_offsets",
        "replacement_parent_cell_rows",
    )
    map_lanes = (
        ("output_cell_to_input_cell_rows", "input_cell_to_output_cell_rows"),
        ("output_cell_to_input_cell", "input_cell_to_output_cell"),
    )
    path = Path(output_pin["path"])
    optional_point_keys = (
        "input_node_to_output_node_rows",
        "output_node_to_input_node_rows",
        "output_point_to_input_point_rows",
        "output_point_to_input_point_row",
        "output_point_parent_offsets",
        "output_point_parent_input_cell_rows",
    )
    if step_arrays is None:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in common if k not in z.files]
            if missing:
                raise ValueError(f"{label}: missing immediate lineage arrays {missing}")
            data = {k: np.asarray(z[k]) for k in common}
            for out_key, in_key in map_lanes:
                if out_key in z.files and in_key in z.files:
                    data[out_key] = np.asarray(z[out_key])
                    data[in_key] = np.asarray(z[in_key])
            for key in optional_point_keys:
                if key in z.files:
                    data[key] = np.asarray(z[key])
    else:
        missing = [k for k in common if k not in step_arrays]
        if missing:
            raise ValueError(f"{label}: missing immediate lineage arrays {missing}")
        data = step_arrays
    for key in common:
        if np.asarray(data[key]).dtype.kind not in "iu":
            raise ValueError(f"{label}: {key} must use integer row IDs")
    matching_lanes = []
    for out_key, in_key in map_lanes:
        if out_key in data and in_key in data:
            if np.asarray(data[out_key]).shape == (nchild,) and np.asarray(data[in_key]).shape == (nparent,):
                matching_lanes.append((out_key, in_key))
    if not matching_lanes:
        observed = {k: list(np.asarray(data[k]).shape) for pair in map_lanes for k in pair if k in data}
        raise ValueError(f"{label}: no direct/reverse map lane matches immediate cell domains: {observed}")
    lane_values = []
    for out_key, in_key in matching_lanes:
        lane_values.append((
            out_key + "/" + in_key,
            np.asarray(data[out_key], dtype=np.int64),
            np.asarray(data[in_key], dtype=np.int64),
        ))
    direct, inverse = lane_values[0][1], lane_values[0][2]
    for lane, other_direct, other_inverse in lane_values[1:]:
        if not np.array_equal(direct, other_direct) or not np.array_equal(inverse, other_inverse):
            raise ValueError(f"{label}: multiple immediate map lanes match domains but disagree")
    retained = np.asarray(data["retained_input_cell_rows"], dtype=np.int64)
    replacement_tets = np.asarray(data["replacement_tetrahedra"], dtype=np.int64)
    rep_offsets = np.asarray(data["replacement_parent_offsets"], dtype=np.int64)
    rep_rows = np.asarray(data["replacement_parent_cell_rows"], dtype=np.int64)

    if direct.shape != (nchild,) or inverse.shape != (nparent,):
        raise ValueError(f"{label}: direct/reverse cell-map domains mismatch")
    if retained.ndim != 1 or retained.dtype.kind not in "iu":
        raise ValueError(f"{label}: retained_input_cell_rows malformed")
    retained = retained.astype(np.int64, copy=False)
    if len(retained) and (int(retained.min()) < 0 or int(retained.max()) >= nparent):
        raise ValueError(f"{label}: retained row outside immediate parent")
    if len(np.unique(retained)) != len(retained) or (len(retained) > 1 and np.any(retained[1:] <= retained[:-1])):
        raise ValueError(f"{label}: retained rows must be sorted and unique")
    nreplacement = nchild - len(retained)
    if nreplacement < 0 or replacement_tets.shape != (nreplacement, 4):
        raise ValueError(f"{label}: replacement tetrahedra do not cover output tail")
    if not np.array_equal(ct[:len(retained)], pt[retained]) or not np.array_equal(cm[:len(retained)], pm[retained]):
        raise ValueError(f"{label}: retained output prefix differs from immediate parent rows")
    if not np.array_equal(ct[len(retained):], replacement_tets):
        raise ValueError(f"{label}: replacement tetrahedra differ from output tail")
    if direct[:len(retained)].shape != retained.shape or not np.array_equal(direct[:len(retained)], retained):
        raise ValueError(f"{label}: retained prefix direct map mismatch")
    if direct.shape and (int(direct.min()) < 0 or int(direct.max()) >= nparent):
        raise ValueError(f"{label}: direct representative outside immediate parent domain")
    expected_inverse = np.full(nparent, -1, np.int64)
    expected_inverse[retained] = np.arange(len(retained), dtype=np.int64)
    if not np.array_equal(inverse, expected_inverse):
        raise ValueError(f"{label}: inverse map does not exactly encode retained cells")
    _validate_csr(rep_offsets, rep_rows, nreplacement, nparent, f"{label} replacement-parent")
    for i in range(nreplacement):
        s, e = int(rep_offsets[i]), int(rep_offsets[i + 1])
        if e == s:
            raise ValueError(f"{label}: replacement output {i} has empty parent union")
        representative = int(direct[len(retained) + i])
        if representative not in rep_rows[s:e]:
            raise ValueError(f"{label}: direct representative omitted from complete replacement union")

    # The parent CSR maps each immediate-parent row to canonical base rows.
    parent_offsets = parent["union_offsets"]
    parent_rows = parent["union_rows"]
    parent_lengths = np.diff(parent_offsets)
    # Prefix outputs retain the same ordering as sorted retained input rows.
    retained_lengths = parent_lengths[retained]
    owner = np.repeat(np.arange(nparent, dtype=np.int64), parent_lengths)
    retained_mask = np.zeros(nparent, dtype=bool)
    retained_mask[retained] = True
    retained_flat = parent_rows[retained_mask[owner]]
    if len(retained_flat) != int(retained_lengths.sum()):
        raise ValueError(f"{label}: retained ancestry compaction mismatch")

    replacement_unions = []
    replacement_lengths = np.empty(nreplacement, np.int64)
    for i in range(nreplacement):
        s, e = int(rep_offsets[i]), int(rep_offsets[i + 1])
        src_parents = rep_rows[s:e]
        pieces = [parent_rows[int(parent_offsets[q]):int(parent_offsets[q + 1])] for q in src_parents]
        expanded = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64)
        if len(expanded) == 0:
            raise ValueError(f"{label}: replacement output {i} composes to empty base ancestry")
        replacement_unions.append(expanded)
        replacement_lengths[i] = len(expanded)
    lengths = np.concatenate((retained_lengths, replacement_lengths))
    offsets = np.empty(nchild + 1, np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    rows = np.concatenate((retained_flat, *replacement_unions)) if nreplacement else retained_flat.copy()
    if int(offsets[-1]) != len(rows):
        raise ValueError(f"{label}: composed CSR length mismatch")
    # Validate optional immediate node/point maps only by their declared domains.
    # These are never used to infer cell ancestry.
    point_checks = {
            "input_node_to_output_node_rows": (parent["points"], child["points"], True),
            "output_node_to_input_node_rows": (child["points"], parent["points"], False),
            "output_point_to_input_point_rows": (child["points"], parent["points"], False),
            "output_point_to_input_point_row": (child["points"], parent["points"], False),
        }
    for key, (expected_len, target_len, source_to_target) in point_checks.items():
        if key not in data:
            continue
        a = np.asarray(data[key])
        if a.dtype.kind not in "iu" or a.shape != (expected_len,):
            raise ValueError(f"{label}: {key} point-row domain mismatch")
        lo = 0 if source_to_target else -1
        if len(a) and (int(a.min()) < lo or int(a.max()) >= target_len):
            raise ValueError(f"{label}: {key} references outside point-row domain")
    if "output_point_parent_offsets" in data or "output_point_parent_input_cell_rows" in data:
        if "output_point_parent_offsets" not in data or "output_point_parent_input_cell_rows" not in data:
            raise ValueError(f"{label}: incomplete output-point parent CSR")
        po = np.asarray(data["output_point_parent_offsets"])
        pr = np.asarray(data["output_point_parent_input_cell_rows"])
        if po.dtype.kind not in "iu" or pr.dtype.kind not in "iu":
            raise ValueError(f"{label}: output-point parent CSR must use integer rows")
        _validate_csr(po, pr, child["points"], nparent, f"{label} output-point parent")
    return offsets, rows, {
        "kind": "full_replacement_parent_csr_composed",
        "selected_map_lane": lane_values[0][0],
        "parent_cells": nparent,
        "output_cells": nchild,
        "retained_cells": len(retained),
        "replacement_cells": nreplacement,
        "immediate_replacement_parent_references": len(rep_rows),
        "canonical_base_parent_references": len(rows),
        "maximum_union_per_cell": int(lengths.max(initial=0)),
    }


def compose_steps(base_volume: dict, steps: list[dict], role: str) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    base = _read_minimal_volume(base_volume, f"{role} canonical base")
    nbase = len(base["tetrahedra"])
    offsets = np.arange(nbase + 1, dtype=np.int64)
    rows = np.arange(nbase, dtype=np.int64)
    current = base
    reports = []
    for i, step in enumerate(steps):
        label = f"{role} step {i}"
        output_pin = step["output"]
        if Path(step["parent"]["path"]) != Path(current["path"]) or step["parent"]["sha256"] != current["sha256"]:
            raise ValueError(f"{label}: explicit parent pin does not match preceding chain output")
        child = _read_minimal_volume(output_pin, label)
        if step["kind"] == "identity":
            child_offsets, child_rows, report = _identity_unions(current, child, label)
            # The identity operation leaves the current CSR untouched. Its identity
            # CSR is cell-domain proof only; existing composed ancestry may be wider.
            current = child
        elif step["kind"] == "mapped":
            current["union_offsets"], current["union_rows"] = offsets, rows
            child_offsets, child_rows, report = _mapped_step_unions(current, child, output_pin, label)
            offsets, rows = child_offsets, child_rows
            child["union_offsets"], child["union_rows"] = offsets, rows
            current = child
        else:
            raise ValueError(f"{label}: kind must be identity or mapped")
        if step["kind"] == "identity":
            # Store the same lineage under the identity output domain.
            child["union_offsets"], child["union_rows"] = offsets, rows
        reports.append({
            "index": i,
            "kind": report["kind"],
            "input": {"path": step["parent"]["path"], "sha256": step["parent"]["sha256"]},
            "output": {"path": child["path"], "sha256": child["sha256"]},
            "point_rows": child["points"],
            "cell_rows": len(child["tetrahedra"]),
            **{k: v for k, v in report.items() if k != "kind"},
        })
    if steps:
        final = _read_minimal_volume(steps[-1]["output"], f"{role} final volume")
    else:
        final = base
    if not np.array_equal(final["tetrahedra"], current["tetrahedra"]) or not np.array_equal(final["materials"], current["materials"]):
        raise ValueError(f"{role}: final chain geometry differs from composed current domain")
    if len(offsets) != len(final["tetrahedra"]) + 1 or int(offsets[-1]) != len(rows):
        raise ValueError(f"{role}: final canonical CSR domain mismatch")
    # Validate each canonical-base row list sorted and unique.
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s or (e - s > 1 and np.any(rows[s + 1:e] <= rows[s:e - 1])):
            raise ValueError(f"{role}: final cell {i} has empty or noncanonical base ancestry")
    return offsets, rows, reports, {
        "canonical_base_cells": nbase,
        "current_cells": len(final["tetrahedra"]),
        "canonical_parent_references": len(rows),
        "maximum_base_union_per_cell": int(np.diff(offsets).max(initial=0)),
        "current_point_rows": final["points"],
    }


def _safe_np_scalar(value: str) -> bytes:
    b = io.BytesIO()
    np.lib.format.write_array(b, np.asarray(value), allow_pickle=False)
    return b.getvalue()


def write_npz_with_unions(source: Path, destination: Path, offsets: np.ndarray, rows: np.ndarray, metadata: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    tmp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    metadata_arrays = {
        "canonical_parent_union_operation_order": metadata["operation_order"],
        "canonical_parent_union_physical_status": "SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE",
        "canonical_parent_union_base_state_path": metadata["base_state_path"],
        "canonical_parent_union_base_state_sha256": metadata["base_state_sha256"],
        "canonical_parent_union_base_volume_path": metadata["base_volume_path"],
        "canonical_parent_union_base_volume_sha256": metadata["base_volume_sha256"],
        "canonical_parent_union_current_state_path": metadata["current_state_path"],
        "canonical_parent_union_current_state_sha256": metadata["current_state_sha256"],
        "canonical_parent_union_current_volume_path": metadata["current_volume_path"],
        "canonical_parent_union_current_volume_sha256": metadata["current_volume_sha256"],
        "canonical_parent_union_parent_row_domain": "canonical_base_volume_cell_rows",
    }
    added = {
        "canonical_parent_union_offsets": np.asarray(offsets, np.int64),
        "canonical_parent_union_parent_rows": np.asarray(rows, np.int64),
        **{k: np.asarray(v) for k, v in metadata_arrays.items()},
    }
    try:
        with zipfile.ZipFile(source, "r") as zin:
            existing = {n[:-4] for n in zin.namelist() if n.endswith(".npy")}
            if set(CANON_FIELDS) & existing:
                raise ValueError("input already contains canonical parent CSR; refusing to overwrite")
            replace = REPLACED_METADATA & existing
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zout:
                for name in zin.namelist():
                    key = name[:-4] if name.endswith(".npy") else None
                    if key in replace:
                        continue
                    with zin.open(name, "r") as src, zout.open(name, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                for key, value in added.items():
                    if key in existing and key not in replace:
                        raise ValueError(f"input already has output metadata field {key}; refusing duplicate")
                    if key == "canonical_parent_union_offsets":
                        payload = io.BytesIO()
                        np.lib.format.write_array(payload, value, allow_pickle=False)
                        data = payload.getvalue()
                    elif key == "canonical_parent_union_parent_rows":
                        payload = io.BytesIO()
                        np.lib.format.write_array(payload, value, allow_pickle=False)
                        data = payload.getvalue()
                    else:
                        data = _safe_np_scalar(str(value))
                    zout.writestr(key + ".npy", data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        os.replace(tmp, destination)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

    # Readback verifies the stored CSR and that every historical/non-allowlisted
    # .npy member remained byte-identical after archive rewriting.
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(destination, "r") as zout:
        src_names, dst_names = set(zin.namelist()), set(zout.namelist())
        expected_dst = (src_names - {k + ".npy" for k in replace}) | {k + ".npy" for k in added}
        if dst_names != expected_dst:
            raise ValueError("archive member set changed unexpectedly")
        preserved = 0
        for name in sorted(src_names - {k + ".npy" for k in replace}):
            h1, h2 = hashlib.sha256(), hashlib.sha256()
            with zin.open(name) as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h1.update(block)
            with zout.open(name) as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h2.update(block)
            if h1.digest() != h2.digest():
                raise ValueError(f"existing array payload changed: {name}")
            preserved += 1
    with np.load(destination, allow_pickle=False) as z:
        if not np.array_equal(z["canonical_parent_union_offsets"], offsets):
            raise ValueError("written canonical offsets differ")
        if not np.array_equal(z["canonical_parent_union_parent_rows"], rows):
            raise ValueError("written canonical parent rows differ")
        for k, v in metadata_arrays.items():
            if str(z[k].item()) != str(v):
                raise ValueError(f"written metadata differs: {k}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "preserved_existing_npy_members": preserved,
        "rewritten_metadata_members": sorted(replace),
        "added_members": sorted(added),
    }


def run_manifest(manifest_path: Path, output_dir: Path | None = None, validate_only: bool = False) -> dict:
    msha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported manifest schema")
    base_state_pin = manifest["base_state"]
    current_state_pin = manifest["current_state"]
    base_state = read_json_pin(base_state_pin, "base state")
    current_state = read_json_pin(current_state_pin, "current state")
    if base_state.get("physical_admission") is True or base_state.get("native_steps", 0) != 0:
        raise ValueError("base state is not source-only")
    if current_state.get("physical_admission") is True or current_state.get("native_steps", 0) != 0:
        raise ValueError("current state is not source-only")
    mass_status = current_state.get("mass_mapping", current_state.get("mass_reconciliation", "PENDING"))
    if mass_status not in ("PENDING", "NOT_RUN", "PENDING_VOLUME_CHANGE_RECONCILIATION") and not (
        isinstance(mass_status, dict) and mass_status.get("status", "").startswith("PENDING")
    ):
        raise ValueError("current state has unexpected mass status")
    results = {}
    for role in ("Skin", "Fat"):
        role_cfg = manifest["roles"][role]
        base_pin, final_pin = role_cfg["base_volume"], role_cfg["current_volume"]
        verify_state_volume(base_state_pin, base_state, role, base_pin, f"base {role}")
        verify_state_volume(current_state_pin, current_state, role, final_pin, f"current {role}")
        steps = role_cfg.get("steps", [])
        # Authenticate each explicit mapping context. A context may be a state-current
        # pin or a run-config volume input; it must bind this exact immediate parent.
        for i, step in enumerate(steps):
            binding = step.get("parent_binding")
            if not binding:
                continue
            if binding["kind"] == "state_current":
                st = read_json_pin(binding["state"], f"{role} step {i} parent state")
                rp = state_role_pin(st, role, f"{role} step {i} parent state")
                if rp["path"] != step["parent"]["path"] or rp["sha256"] != step["parent"]["sha256"]:
                    raise ValueError(f"{role} step {i}: parent state does not bind immediate input")
            elif binding["kind"] == "run_config_volume":
                cfg = read_json_pin(binding["run_config"], f"{role} step {i} run config")
                rp = cfg.get("volumes", {}).get(role)
                if not isinstance(rp, dict) or rp.get("path") != step["parent"]["path"] or rp.get("sha256") != step["parent"]["sha256"]:
                    raise ValueError(f"{role} step {i}: run config does not bind immediate input")
            else:
                raise ValueError(f"{role} step {i}: unknown parent binding kind")
        offsets, rows, step_reports, stats = compose_steps(base_pin, steps, role)
        if steps and (Path(steps[-1]["output"]["path"]) != Path(final_pin["path"]) or steps[-1]["output"]["sha256"] != final_pin["sha256"]):
            raise ValueError(f"{role}: final chain step differs from current state pin")
        out_info = None
        if not validate_only:
            if output_dir is None:
                raise ValueError("output directory required unless validate-only")
            suffix = f"{role.lower()}-canonical-parent-union.npz"
            source = Path(final_pin["path"])
            destination = output_dir / suffix
            order = "->".join([Path(base_pin["path"]).name] + [Path(x["output"]["path"]).name for x in steps])
            out_info = write_npz_with_unions(source, destination, offsets, rows, {
                "operation_order": order,
                "base_state_path": str(base_state_pin["path"]),
                "base_state_sha256": base_state_pin["sha256"],
                "base_volume_path": str(base_pin["path"]),
                "base_volume_sha256": base_pin["sha256"],
                "current_state_path": str(current_state_pin["path"]),
                "current_state_sha256": current_state_pin["sha256"],
                "current_volume_path": str(final_pin["path"]),
                "current_volume_sha256": final_pin["sha256"],
            })
        results[role] = {
            **stats,
            "base_volume": base_pin,
            "current_volume": final_pin,
            "steps": step_reports,
            "canonical_parent_offsets_entries": len(offsets),
            "canonical_parent_references": len(rows),
            "canonical_parent_rows_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
            "canonical_parent_offsets_sha256": hashlib.sha256(np.asarray(offsets, dtype="<i8").tobytes()).hexdigest(),
            "output": out_info,
            "current_quality_failures": current_state.get(f"{role}_current", {}).get("native_quality_failures"),
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS_SOURCE_ONLY_CANONICAL_CURRENT_VOLUME_PARENT_UNIONS",
        "physical_admission": False,
        "native_steps": 0,
        "mass_mapping": "NOT_RUN",
        "manifest": {"path": str(manifest_path), "sha256": msha},
        "base_state": base_state_pin,
        "current_state": current_state_pin,
        "roles": results,
        "limitations": "Canonical ancestry rows refer to the explicitly pinned base volume cell-row domain. They do not claim ultimate source330/original-cell ancestry, mass mapping, native execution, or physical admission.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--report", type=Path)
    ns = ap.parse_args()
    started = time.monotonic()
    result = run_manifest(ns.manifest, ns.output_dir, ns.validate_only)
    result["elapsed_s"] = time.monotonic() - started
    if ns.report:
        ns.report.parent.mkdir(parents=True, exist_ok=True)
        ns.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
