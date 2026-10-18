#!/usr/bin/env python3
"""Rebind an existing cardiac common-field map after one unused vertex retires.

The source and compact candidate must have identical emitted triangles after
index remapping. This copies every affine record except the retired slot,
updates the row ranges and active receipt bindings, and proves referenced map
records are byte-exact. It does not create or admit geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_pleura_proxy import _parse_payload  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("common-field vertex retirement: " + message)


def remap_vertex_indices_after_retirement(indices: np.ndarray, retired_vertex: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices == retired_vertex):
        raise ValueError("retired or invalid vertex index cannot be remapped")
    remapped = indices.copy()
    remapped[remapped > retired_vertex] -= 1
    return remapped


def retire_affine_map_record(source_map: np.ndarray, record_index: int) -> np.ndarray:
    source_map = np.asarray(source_map)
    if source_map.ndim != 3 or source_map.shape[1:] != (7, 4):
        raise ValueError("common-field map must have shape (records, 7, 4)")
    if not 0 <= record_index < len(source_map):
        raise ValueError("retired common-field map record is out of range")
    return np.delete(source_map, record_index, axis=0)


def validate_vertex_retirement_header(old_header, new_header) -> None:
    """Require only the global vertex allocation count to decrease by one."""
    require(len(old_header) == len(new_header) and len(old_header) >= 6,
            "payload header layouts differ")
    require(old_header[0:3] == new_header[0:3]
            and old_header[4] == new_header[4]
            and old_header[5:] == new_header[5:]
            and old_header[3] == new_header[3] + 1,
            "payload header differs beyond retiring one vertex allocation")


def update_matching_active_hashes(value, old_sha: str, new_sha: str, prefix: str) -> list[str]:
    updated = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "anatomy_payload_sha256" and child == old_sha:
                value[key] = new_sha
                updated.append(prefix + "." + key)
            else:
                updated.extend(update_matching_active_hashes(child, old_sha, new_sha,
                                                              prefix + "." + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            updated.extend(update_matching_active_hashes(child, old_sha, new_sha,
                                                          prefix + f"[{index}]"))
    return updated


def rebind_common_field(
    base_path: Path,
    candidate_path: Path,
    receipt_path: Path,
    output_dir: Path,
    stable_id: int,
    retired_vertex: int,
    overwrite: bool = False,
) -> dict:
    base_path = Path(base_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_raw = base_path.read_bytes()
    candidate_raw = candidate_path.read_bytes()
    old_header, _, old_rows = _parse_payload(base_raw)
    new_header, _, new_rows = _parse_payload(candidate_raw)
    require(old_rows.keys() == new_rows.keys(), "stable surface row IDs changed")
    require(stable_id in old_rows, f"stable ID {stable_id} is absent from the base payload")
    validate_vertex_retirement_header(old_header, new_header)

    old_vertex_count = len(old_rows[stable_id]["vertices6"])
    require(0 <= retired_vertex < old_vertex_count, "retired vertex index is outside the source row")
    old_target, new_target = old_rows[stable_id], new_rows[stable_id]
    require(not np.any(old_target["faces"] == retired_vertex),
            "retired source vertex is still referenced by a source triangle")
    expected_vertices = np.delete(old_target["vertices6"], retired_vertex, axis=0)
    require(new_target["vertices6"].shape == expected_vertices.shape
            and new_target["vertices6"].tobytes() == expected_vertices.tobytes(),
            "candidate surviving vertex attributes differ from exact source-slot retirement")
    expected_faces = remap_vertex_indices_after_retirement(
        old_target["faces"], retired_vertex
    )
    require(np.array_equal(new_target["faces"], expected_faces),
            "candidate triangles differ from source triangles after the index remap")
    for sid in old_rows:
        if sid == stable_id:
            continue
        old, new = old_rows[sid], new_rows[sid]
        require(old["body_index"] == new["body_index"]
                and old["layer"] == new["layer"] and old["flags"] == new["flags"],
                f"unselected stable row {sid} identity changed")
        require(old["vertices6"].tobytes() == new["vertices6"].tobytes()
                and np.array_equal(old["faces"], new["faces"]),
                f"unselected stable row {sid} geometry changed")

    old_sha = sha256_bytes(base_raw)
    new_sha = sha256_bytes(candidate_raw)
    receipt = json.loads(receipt_path.read_text())
    rebound = copy.deepcopy(receipt)
    common = rebound["provenance"]["cardiac_geometry_binding"]["common_field"]
    require(common.get("anatomy_payload_sha256") == old_sha,
            "source common-field receipt is not bound to the base anatomy payload")
    map_descriptor = common["map"]
    map_relative = Path(map_descriptor["path"])
    require(not map_relative.is_absolute() and ".." not in map_relative.parts,
            "common-field map path must be a safe relative path")
    map_source = receipt_path.parent / map_relative
    source_map_sha = map_descriptor["sha256"]
    require(sha256_file(map_source) == source_map_sha,
            "source common-field map hash differs")
    stride = int(map_descriptor.get("record_stride_bytes", 0))
    require(stride == 112, "unsupported common-field map record stride")
    source_map = np.fromfile(map_source, dtype="<f4")
    require(source_map.size % 28 == 0, "source common-field map has a malformed stride")
    source_map = source_map.reshape(-1, 7, 4)
    require(len(source_map) == int(map_descriptor["record_count"]),
            "source common-field map record count differs")
    target_range = next(
        (item for item in common["vertex_ranges"] if int(item["stable_id"]) == stable_id), None
    )
    require(target_range is not None, "common-field map lacks the selected stable row")
    require(int(target_range["vertex_count"]) == old_vertex_count,
            "source common-field range differs from the anatomy allocation")
    retired_record = int(target_range["first_vertex"]) + retired_vertex
    require(0 <= retired_record < len(source_map), "retired map record is out of range")
    for item in common["vertex_ranges"]:
        start = int(item["first_vertex"])
        count = int(item["vertex_count"])
        if int(item["stable_id"]) == stable_id:
            require(start <= retired_record < start + count,
                    "retired map record is outside the selected row range")
        else:
            require(not (start <= retired_record < start + count),
                    "retired map record overlaps a different stable row")

    compact_map = retire_affine_map_record(source_map, retired_record)
    map_output = output_dir / map_relative
    require(map_output.resolve() != map_source.resolve(),
            "map output must not overwrite the source common-field map")
    map_output.parent.mkdir(parents=True, exist_ok=True)
    if map_output.exists() and not overwrite:
        require(map_output.is_file() and sha256_file(map_output) ==
                sha256_bytes(compact_map.tobytes()), "map output exists with different bytes")
    else:
        map_output.write_bytes(compact_map.tobytes())
    new_map_sha = sha256_file(map_output)

    for group in ("vertex_ranges", "passive_attachment_ranges"):
        for item in common.get(group, []):
            start = int(item["first_vertex"])
            if group == "vertex_ranges" and int(item["stable_id"]) == stable_id:
                item["vertex_count"] = int(item["vertex_count"]) - 1
            elif start > retired_record:
                item["first_vertex"] = start - 1
    map_descriptor["record_count"] = int(map_descriptor["record_count"]) - 1
    map_descriptor["sha256"] = new_map_sha
    common["anatomy_payload_sha256"] = new_sha
    require(common.get("anatomy_payload_sha256") == new_sha,
            "rebound common-field binding is not bound to the candidate anatomy payload")

    # Non-map common-field sidecars retain their original bytes and are linked
    # beside the rebound receipt so its relative paths remain valid.
    sidecar_paths = {}
    for key in ("domain_boxes", "polynomials"):
        descriptor = common.get(key)
        if not isinstance(descriptor, dict) or "path" not in descriptor:
            continue
        relative = Path(descriptor["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"common-field {key} path must be relative")
        source = receipt_path.parent / relative
        require(source.is_file(), f"common-field {key} sidecar is missing")
        require(isinstance(descriptor.get("sha256"), str)
                and sha256_file(source) == descriptor["sha256"],
                f"common-field {key} source hash differs from its receipt")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            require(destination.resolve() == source.resolve(),
                    f"common-field {key} output already points elsewhere")
        else:
            destination.symlink_to(source.resolve())
        sidecar_paths[key] = str(destination)

    active_hash_paths = [
        "provenance.cardiac_geometry_binding.common_field.anatomy_payload_sha256"
    ]
    active_hash_paths.extend(update_matching_active_hashes(
        rebound.get("functional_bindings", {}), old_sha, new_sha, "functional_bindings"
    ))
    active_hash_paths.extend(update_matching_active_hashes(
        rebound.get("passive_tissue_parent_bindings", {}), old_sha, new_sha,
        "passive_tissue_parent_bindings"
    ))
    payload = rebound.get("payload", {})
    if "sha256" in payload:
        payload["sha256"] = new_sha
    if "path" in payload:
        payload["path"] = str(candidate_path)
    if "bytes" in payload:
        payload["bytes"] = len(candidate_raw)
    if "vertex_count" in payload:
        payload["vertex_count"] = int(new_header[3])

    map_out_bytes = map_output.read_bytes()
    map_mapped = np.frombuffer(map_out_bytes, dtype="<f4").reshape(-1, 7, 4)
    source_ranges = {int(item["stable_id"]): item for item in common["vertex_ranges"]}
    for sid, row in new_rows.items():
        item = source_ranges.get(int(sid))
        if item is None:
            continue
        first = int(item["first_vertex"])
        require(first + len(row["vertices6"]) <= len(map_mapped),
                f"rebound map range for stable row {sid} is out of bounds")
        source_item = next(x for x in receipt["provenance"]["cardiac_geometry_binding"]["common_field"]["vertex_ranges"]
                           if int(x["stable_id"]) == int(sid))
        source_first = int(source_item["first_vertex"])
        for local_index in np.unique(row["faces"]):
            old_local = int(local_index) + (1 if int(sid) == stable_id and int(local_index) >= retired_vertex else 0)
            require(np.array_equal(map_mapped[first + int(local_index)],
                                   source_map[source_first + old_local]),
                    f"affine map record changed for referenced stable row {sid} vertex {local_index}")

    report = {
        "schema": "numi.human.common-field-vertex-retirement-rebind.v1",
        "source_payload_sha256": old_sha,
        "candidate_payload_sha256": new_sha,
        "source_map_sha256": source_map_sha,
        "candidate_map_sha256": new_map_sha,
        "retired_stable_id": int(stable_id),
        "retired_source_vertex": int(retired_vertex),
        "retired_common_map_record": int(retired_record),
        "source_allocated_vertices_before": int(old_vertex_count),
        "candidate_allocated_vertices_after": int(len(new_target["vertices6"])),
        "source_face_count": int(len(old_target["faces"])),
        "candidate_face_count": int(len(new_target["faces"])),
        "all_emitted_triangle_vertex_attributes_identical_after_index_remap": True,
        "all_referenced_affine_displacement_records_byte_exact_after_index_remap": True,
        "all_other_common_map_records_byte_exact": bool(
            np.array_equal(np.delete(source_map, retired_record, axis=0), map_mapped)
        ),
        "all_unselected_surface_rows_byte_exact": True,
        "active_anatomy_hash_bindings_updated": active_hash_paths,
        "preserved_common_field_sidecars": sidecar_paths,
        "no_force_mass_controller_or_solver_tolerance_change": True,
        "native_admission": False,
    }
    require(report["all_other_common_map_records_byte_exact"],
            "map bytes changed beyond the retired record")
    output_receipt = output_dir / "resting-anatomy-receipt.json"
    output_report = output_dir / "allocation-retirement.json"
    require(output_receipt.resolve() != receipt_path,
            "rebound receipt must not overwrite the source receipt")
    for path in (output_receipt, output_report):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {path}")
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    report["report_sha256"] = sha256_file(output_report)
    provenance = rebound.setdefault("provenance", {})
    history = provenance.get("common_field_vertex_allocation_retirement_chain")
    if not isinstance(history, list):
        previous = provenance.get("common_field_vertex_allocation_retirement")
        history = [previous] if isinstance(previous, dict) else []
    else:
        history = list(history)
        previous = provenance.get("common_field_vertex_allocation_retirement")
        if isinstance(previous, dict) and (not history or history[-1] != previous):
            history.append(previous)
    history.append(report)
    provenance["common_field_vertex_allocation_retirement_chain"] = history
    provenance["common_field_vertex_allocation_retirement"] = report
    output_receipt.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="uncompacted NHANATOMY payload")
    parser.add_argument("--candidate", required=True, type=Path, help="compacted NHANATOMY payload")
    parser.add_argument("--receipt", required=True, type=Path, help="source common-field receipt")
    parser.add_argument("--output-dir", required=True, type=Path, help="rebound receipt/map output directory")
    parser.add_argument("--stable-id", required=True, type=int)
    parser.add_argument("--retired-vertex", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = rebind_common_field(
        args.base, args.candidate, args.receipt, args.output_dir,
        args.stable_id, args.retired_vertex, args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
