"""Exact TetGen warning-row rebasing for role-local surface candidates.

This module performs no geometry repair. It maps warning endpoint rows through
an authenticated row map (or an explicitly hash-matched identity map), checks
their emitted Float32 coordinates bitwise, and classifies whether the edge is
still present in the selected candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _edge_keys(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    e.sort(axis=1)
    if np.any(e < 0) or np.any(e >= int(vertex_count)):
        raise ValueError("candidate face references a point outside its candidate domain")
    encoded = e[:, 0] * np.int64(vertex_count) + e[:, 1]
    return np.unique(encoded)


def rebase_segments(
    warning_report: dict,
    candidate_by_role: dict[str, dict],
    row_map_by_role: dict[str, np.ndarray | None],
) -> dict:
    """Map TetGen warning endpoints and classify exact current-edge status.

    `candidate_by_role` values contain `points_f32`, `faces`, and pinned
    candidate path/hash metadata. `row_map_by_role[role][warning_row]` maps an
    old warning-domain point row to the current candidate row; `None` is only
    permitted when the caller has already authenticated an identity domain.
    """
    roles = {}
    for role, candidate in candidate_by_role.items():
        p = np.asarray(candidate["points_f32"], dtype=np.float32)
        f = np.asarray(candidate["faces"], dtype=np.int64).reshape((-1, 3))
        m = row_map_by_role.get(role)
        if m is not None:
            m = np.asarray(m, dtype=np.int64).reshape(-1)
        roles[role] = {
            "points": p,
            "faces": f,
            "edge_keys": _edge_keys(f, len(p)),
            "row_map": m,
            "candidate": candidate,
        }

    mapped = []
    counts: dict[str, int] = {}
    for role_block in warning_report.get("roles", []):
        warning_role = str(role_block.get("role", ""))
        for warning in role_block.get("warnings", []):
            for segment_index, segment in enumerate(warning.get("segments", [])):
                role_rows = segment.get("role_rows", [])
                coords = np.asarray(segment.get("coordinates_f32_m", []), dtype=np.float32)
                item = {
                    "warning_role": warning_role,
                    "warning_index": int(warning.get("warning_index", -1)),
                    "segment_index": int(segment_index),
                    "role_rows": role_rows,
                    "coordinates_f32_m": coords.tolist(),
                }
                if len(role_rows) != 2 or len(coords) != 2 or coords.shape != (2, 3):
                    item["status"] = "MALFORMED_SEGMENT"
                else:
                    endpoint_roles = [str(x[0]) for x in role_rows]
                    old_rows = [int(x[1]) for x in role_rows]
                    if endpoint_roles[0] != endpoint_roles[1]:
                        item["status"] = "CROSS_ROLE_SEGMENT"
                    else:
                        role = endpoint_roles[0]
                        if role not in roles:
                            item["status"] = "ROLE_NOT_IN_CANDIDATE_SET"
                        else:
                            rec = roles[role]
                            row_map = rec["row_map"]
                            if row_map is None:
                                candidate_row_count = len(rec["points"])
                                current_rows = old_rows if all(0 <= r < candidate_row_count for r in old_rows) else [-1, -1]
                            else:
                                current_rows = [int(row_map[r]) if 0 <= r < len(row_map) else -1 for r in old_rows]
                            item["role"] = role
                            item["current_rows"] = current_rows
                            if any(r < 0 or r >= len(rec["points"]) for r in current_rows):
                                item["status"] = "ENDPOINT_REMOVED_OR_OUT_OF_RANGE"
                            elif current_rows[0] == current_rows[1]:
                                item["status"] = "ENDPOINTS_COLLAPSED"
                            elif not np.array_equal(rec["points"][current_rows], coords):
                                item["status"] = "FLOAT32_COORDINATE_MISMATCH"
                            else:
                                a, b = sorted(current_rows)
                                key = np.int64(a) * np.int64(len(rec["points"])) + np.int64(b)
                                edge_index = int(np.searchsorted(rec["edge_keys"], key))
                                if edge_index >= len(rec["edge_keys"]) or rec["edge_keys"][edge_index] != key:
                                    item["status"] = "EDGE_REMOVED"
                                else:
                                    item["status"] = "UNRESOLVED_CURRENT_EDGE"
                counts[item["status"]] = counts.get(item["status"], 0) + 1
                mapped.append(item)
    return {
        "schema": "numi.exact-warning-row-rebase.v1",
        "status": "MAPPED_WITH_UNRESOLVED_EDGES" if counts.get("UNRESOLVED_CURRENT_EDGE", 0) else "MAPPED_NO_UNRESOLVED_EDGES",
        "segment_count": len(mapped),
        "status_counts": counts,
        "segments": mapped,
        "geometry_changed": False,
        "physical_admission": False,
        "runtime": 0,
        "accepted_steps": 0,
    }


def run_manifest(manifest_path: str | Path, output_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "numi.exact-warning-row-rebase-inputs.v1":
        raise ValueError("unsupported manifest schema")
    warning_pin = manifest["warning_report"]
    if sha256_file(warning_pin["path"]) != warning_pin["sha256"]:
        raise ValueError("warning report SHA mismatch")
    warning_report = json.loads(Path(warning_pin["path"]).read_text())
    candidate_by_role = {}
    row_map_by_role = {}
    input_pins = {"warning_report": warning_pin, "roles": {}}
    for role, spec in manifest["roles"].items():
        candidate_path = Path(spec["candidate_path"])
        candidate_sha = sha256_file(candidate_path)
        if candidate_sha != spec["candidate_sha256"]:
            raise ValueError(f"{role} candidate SHA mismatch")
        with np.load(candidate_path, allow_pickle=False) as z:
            points_key = spec.get("points_key", "points_world_m")
            f32_key = spec.get("points_f32_key")
            points = np.array(z[f32_key] if f32_key and f32_key in z else z[points_key], copy=True).astype(np.float32)
            faces = np.array(z[spec.get("faces_key", "faces")], copy=True).astype(np.int64)
        row_spec = spec.get("warning_to_candidate_row_map")
        if row_spec is None:
            if spec.get("identity_domain_sha256") != candidate_sha:
                raise ValueError(f"{role} identity map is not bound to candidate SHA")
            row_map = None
            map_pin = {"kind": "identity", "identity_domain_sha256": candidate_sha}
        else:
            map_path = Path(row_spec["path"])
            if sha256_file(map_path) != row_spec["sha256"]:
                raise ValueError(f"{role} row-map SHA mismatch")
            with np.load(map_path, allow_pickle=False) as z:
                row_map = np.array(z[row_spec["key"]], copy=True).astype(np.int64).reshape(-1)
            map_pin = {"kind": "explicit_map", **row_spec}
        candidate_by_role[role] = {
            "points_f32": points,
            "faces": faces,
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha,
        }
        row_map_by_role[role] = row_map
        input_pins["roles"][role] = {
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha,
            "points_key": points_key,
            "faces_key": spec.get("faces_key", "faces"),
            "row_map": map_pin,
        }
    result = rebase_segments(warning_report, candidate_by_role, row_map_by_role)
    result["inputs"] = input_pins
    result["manifest_path"] = str(manifest_path)
    result["manifest_sha256"] = sha256_file(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_manifest(args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ("status", "segment_count", "status_counts", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
