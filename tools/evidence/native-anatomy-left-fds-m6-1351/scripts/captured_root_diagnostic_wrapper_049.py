"""Local, in-memory adapter for candidate-reference clearance rows.

It preserves the frozen owner's SAT, QP, Float32, and displacement checks.
The local adapter allows a current candidate triangle to be clear or intersecting
while every constraint remains tied to an exact, tagged, pose-mapped reference
triangle and fixed target triangle. Reference geometry is used only as a numeric
repair reference; it does not promote a candidate to accepted physical state.
The frozen module file is never modified.

A caller must construct its geometry through tag_reference_geometry and
bind_external_row, retain the returned row tags, then publish receipts only
through retag_external_receipts.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import numpy as np

FROZEN_MODULE_SHA256 = "fccc7f29d421c9ad8e7da40d61d16cc8633244a16829dc7a7cd9d83ff109cfae"
OLD = '''            if not rejected_points:
                raise human.ImportError(
                    f"local self-separation pose {pose_index} external rejected face {candidate_face} "
                    f"does not exactly intersect target {target_key}"
                )'''
NEW = '''            # Candidate-reference preservation may constrain an already-clear
            # current candidate. Exact clear-reference and target checks above
            # remain mandatory; the caller supplies exact current status tags.
            if not rejected_points:
                pass'''
OLD_APPEND = '''            external_rows.append({
                **values,
                "target_triangle": target_triangle,
                "parent_triangle": parent_triangle,
                "rejected_triangle": rejected_triangle,
            })'''
NEW_APPEND = '''            external_rows.append({
                **values,
                "target_triangle": target_triangle,
                "parent_triangle": parent_triangle,
                "rejected_triangle": rejected_triangle,
                "reference_adapter_current_exactly_intersects": bool(rejected_points),
            })'''
OLD_FLOOR = '            floor_mm = min(parent_gap_mm, _LOCAL_SELF_SEPARATION_EXTERNAL_GAP_CAP_MM)\n            current_gap_mm = float((target_tri @ axis).min() - (trial_tri @ axis).max()) * 1000.0'
NEW_FLOOR = OLD_FLOOR


def _array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_reference_clearance_proposer(ca, expected_module_sha256: str = FROZEN_MODULE_SHA256, *, max_pose_samples: int = 17):
    """Return an in-memory proposer with the exact frozen-helper call signature."""
    if isinstance(max_pose_samples, bool) or not isinstance(max_pose_samples, int) or not 17 <= max_pose_samples <= 32:
        raise ValueError("max_pose_samples must be an integer in [17, 32]")
    module_path = Path(ca.__file__)
    actual = _file_sha(module_path)
    if actual != expected_module_sha256:
        raise RuntimeError(f"frozen helper module hash mismatch: {actual}")
    original = inspect.getsource(ca._propose_local_self_separation_increment)
    if (original.count(OLD) != 1 or original.count(OLD_APPEND) != 1
            or original.count(OLD_FLOOR) != 1
            or original.count("one to seventeen failing pose samples") != 1):
        raise RuntimeError("frozen helper acceptance blocks differ from the pinned wrapper source")
    adapted = original.replace(OLD, NEW, 1).replace(OLD_APPEND, NEW_APPEND, 1).replace(
        OLD_FLOOR, NEW_FLOOR, 1).replace(
        "def _propose_local_self_separation_increment(",
        "def propose_local_reference_clearance_increment(", 1,
    )
    dump = '''    _diag_base = "/Users/n/numi-human-fds132-m6-1350/attempt-048/diagnostics"
    import os as _diag_os, json as _diag_json
    _diag_os.makedirs(_diag_base, exist_ok=True)
    np.savez_compressed(_diag_base + "/constraint-system.npz", matrix=matrix, rhs=rhs,
                        solver_rhs=solver_rhs, solver_cap=solver_cap, movable=movable)
    with open(_diag_base + "/constraint-receipts.json", "w") as _diag_file:
        _diag_json.dump({"constraint_receipts": constraint_receipts, "rows_by_pose": rows_by_pose,
                         "nvar": nvar, "pose_count": len(pose_samples)}, _diag_file,
                        allow_nan=False)
    '''
    marker = "    feasible = linprog("
    if original.count(marker) != 1:
        raise RuntimeError("diagnostic insertion point changed")
    adapted = adapted.replace(marker, dump + marker, 1)
    namespace = dict(ca.__dict__)
    namespace["_LOCAL_SELF_SEPARATION_MAX_POSES"] = max_pose_samples
    adapted = adapted.replace("one to seventeen failing pose samples", f"one to {max_pose_samples} failing pose samples", 1)
    exec(compile(adapted, f"<{module_path}:local-reference-clearance-adapter>", "exec"), namespace)
    return namespace["propose_local_reference_clearance_increment"]


def tag_reference_geometry(*, kind: str, path, expected_sha256: str, pose_step: int,
                           world_positions, faces):
    """Pin a named source/candidate and its exact mapped geometry for one pose."""
    p = Path(path)
    actual = _file_sha(p)
    if actual != expected_sha256:
        raise RuntimeError(f"reference file hash mismatch: {p}: {actual}")
    positions = np.ascontiguousarray(world_positions, dtype="<f4")
    face_rows = np.ascontiguousarray(faces, dtype=np.int64)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not positions.size
            or not np.isfinite(positions).all()):
        raise ValueError("tagged reference positions must be finite Float32 Nx3")
    if (face_rows.ndim != 2 or face_rows.shape[1] != 3 or not face_rows.size
            or face_rows.min() < 0 or face_rows.max() >= len(positions)):
        raise ValueError("tagged reference faces must be valid integer Fx3")
    if not isinstance(kind, str) or not kind.strip() or not isinstance(pose_step, int):
        raise ValueError("reference kind and exact pose step are required")
    return {
        "kind": kind, "path": str(p), "sha256": actual, "pose_step": pose_step,
        "positions": positions, "faces": face_rows,
        "positions_sha256": _array_sha(positions), "faces_sha256": _array_sha(face_rows),
    }


def bind_external_row(*, sample_index: int, sample_pose_step: int, candidate_face_row: int,
                      target_semantic: int, target_stable_id: int, target_face_id: int,
                      target_triangle_world_positions, current_candidate_triangle,
                      reference_geometry: dict, reference_face_row: int, ci):
    """Return owner-schema row plus its mandatory provenance/exact-status tag."""
    if int(reference_geometry["pose_step"]) != int(sample_pose_step):
        raise RuntimeError("reference geometry was not mapped at the sample's exact accepted pose")
    target = np.ascontiguousarray(target_triangle_world_positions, dtype="<f4")
    current = np.ascontiguousarray(current_candidate_triangle, dtype="<f4")
    if target.shape != (3, 3) or current.shape != (3, 3):
        raise ValueError("target and current candidate triangles must be Float32 3x3")
    reference = reference_geometry["positions"][reference_geometry["faces"][int(reference_face_row)]]
    ref_points = ci.triangle_intersection_points(
        tuple(ci.float32_point_lattice_key(p) for p in reference),
        tuple(ci.float32_point_lattice_key(p) for p in target),
    )
    if ref_points:
        raise RuntimeError("tagged reference triangle exactly intersects its target")
    current_points = ci.triangle_intersection_points(
        tuple(ci.float32_point_lattice_key(p) for p in current),
        tuple(ci.float32_point_lattice_key(p) for p in target),
    )
    packed_ref_face_row = int(reference_face_row)
    row = {
        "candidate_face_row": int(candidate_face_row),
        "parent_face_row": packed_ref_face_row,
        "target_semantic": int(target_semantic),
        "target_stable_id": int(target_stable_id),
        "target_face_id": int(target_face_id),
        "target_triangle_world_positions": target,
    }
    tag = {
        "sample_index": int(sample_index), "sample_pose_step": int(sample_pose_step),
        "reference_kind": reference_geometry["kind"],
        "reference_path": reference_geometry["path"],
        "reference_sha256": reference_geometry["sha256"],
        "reference_pose_step": reference_geometry["pose_step"],
        "reference_positions_sha256": reference_geometry["positions_sha256"],
        "reference_faces_sha256": reference_geometry["faces_sha256"],
        "reference_face_row": int(reference_face_row),
        "packed_reference_face_row": packed_ref_face_row,
        "current_exactly_intersects": bool(current_points),
    }
    return row, tag


def retag_external_receipts(proposal: dict, row_tags: dict[tuple, dict]):
    """Retag owner rows and fail if provenance or exact status is not bound.

    row_tags key is (sample_index, candidate_face_row, packed_reference_face_row,
    target_semantic, target_stable_id, target_face_id), matching bind_external_row.
    """
    import copy
    out = copy.deepcopy(proposal.get("external_clearance_pair_receipts", []))
    required = {"sample_index", "sample_pose_step", "reference_kind", "reference_path",
                "reference_sha256", "reference_pose_step", "reference_positions_sha256",
                "reference_faces_sha256", "reference_face_row", "packed_reference_face_row",
                "current_exactly_intersects"}
    for row in out:
        old_parent_face = int(row["accepted_parent_face_row"])
        key = (int(row["pose_index"]), int(row["candidate_face_row"]), old_parent_face,
               int(row["target_semantic"]), int(row["target_stable_id"]), int(row["target_face_id"]))
        if key not in row_tags or not required.issubset(row_tags[key]):
            raise RuntimeError(f"missing complete tagged-reference row for receipt {key}")
        tag = dict(row_tags[key])
        if int(tag["sample_index"]) != key[0] or int(tag["packed_reference_face_row"]) != old_parent_face:
            raise RuntimeError(f"receipt does not match tagged reference array row: {key}")
        if int(tag["reference_pose_step"]) != int(tag["sample_pose_step"]):
            raise RuntimeError(f"reference pose does not match exact sample pose: {key}")
        if not isinstance(tag["current_exactly_intersects"], bool):
            raise RuntimeError(f"current exact-intersection status is not boolean: {key}")
        if not isinstance(tag["reference_face_row"], int) or tag["reference_face_row"] < 0:
            raise RuntimeError(f"reference face row is invalid: {key}")
        if _file_sha(tag["reference_path"]) != tag["reference_sha256"]:
            raise RuntimeError(f"reference file hash mismatch for receipt {key}")
        for field in ("reference_positions_sha256", "reference_faces_sha256"):
            value = tag[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError(f"reference array hash is invalid for {key}: {field}")
        if not isinstance(tag["reference_kind"], str) or not tag["reference_kind"].strip():
            raise RuntimeError(f"reference kind is missing for receipt {key}")
        row["reference_kind"] = tag["reference_kind"]
        row["reference_path"] = tag["reference_path"]
        row["reference_sha256"] = tag["reference_sha256"]
        row["reference_pose_step"] = int(tag["reference_pose_step"])
        row["reference_positions_sha256"] = tag["reference_positions_sha256"]
        row["reference_faces_sha256"] = tag["reference_faces_sha256"]
        row["reference_face_row"] = int(tag["reference_face_row"])
        row["reference_triangle_exactly_clear"] = bool(row.pop("accepted_parent_exactly_clear"))
        row["current_candidate_exactly_intersects"] = tag["current_exactly_intersects"]
        row.pop("rejected_candidate_exactly_intersects", None)
        row["reference_sat_gap_mm"] = row.pop("accepted_parent_sat_gap_mm")
        row["reference_clearance_loss_mm"] = row.pop("accepted_parent_clearance_loss_mm")
        row["sat_axis_selection"] = "maximum_positive_gap_from_tagged_reference_and_fixed_target_triangles"
        row.pop("accepted_parent_face_row", None)
        if any("accepted_parent" in k or "rejected_candidate" in k for k in row):
            raise RuntimeError(f"misleading owner receipt wording remains after retagging: {key}")
    return out
