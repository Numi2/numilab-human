#!/usr/bin/env python3
"""Prepare one registered interior-vertex collapse and retire its source slot.

The existing respiratory mesh owner validates the directed collapse. This CLI
serializes the compact result through the existing NHANATOMY ABI5 encoder; it
preserves every surviving source vertex attribute and writes face ancestry.
It prepares geometry only and does not admit a native runtime candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_intercostal_mesh_arrangement import (  # noqa: E402
    encode_payload_with_row_replacements,
)
from numilab_human.resting_pleura_proxy import _parse_payload  # noqa: E402
from numilab_human.resting_respiratory_mesh_quality import (  # noqa: E402
    collapse_registered_interior_vertex,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_point_owners(rows: dict, point: np.ndarray) -> list[int]:
    point = np.asarray(point, dtype=np.float32)
    owners = []
    for stable_id, row in rows.items():
        indices = np.flatnonzero(np.all(row["vertices6"][:, :3] == point, axis=1))
        if len(indices) > 1:
            raise ValueError(f"stable row {stable_id} duplicates a collapse endpoint")
        if len(indices) == 1:
            owners.append(int(stable_id))
    return sorted(owners)


def compact_faces_after_collapse(
    source_faces: np.ndarray, keep_vertex: int, drop_vertex: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a directed edge collapse and retire the dropped allocation slot."""
    source_faces = np.asarray(source_faces, dtype=np.int64)
    edge_mask = np.sum(np.isin(source_faces, (keep_vertex, drop_vertex)), axis=1) == 2
    edge_faces = np.flatnonzero(edge_mask)
    if len(edge_faces) != 2:
        raise ValueError(f"collapse edge must have two incident faces, found {len(edge_faces)}")
    compact = source_faces[~edge_mask].copy()
    compact[compact == drop_vertex] = keep_vertex
    compact[compact > drop_vertex] -= 1
    if np.any(compact[:, 0] == compact[:, 1]) or np.any(compact[:, 1] == compact[:, 2]) \
            or np.any(compact[:, 2] == compact[:, 0]):
        raise ValueError("collapse creates a repeated-index output face")
    return compact, edge_faces


def prepare_collapse(
    base_path: Path,
    candidate_path: Path,
    preparation_path: Path,
    ancestry_path: Path,
    stable_id: int,
    keep_vertex: int,
    drop_vertex: int,
    maximum_source_displacement_m: float = 1e-6,
    minimum_altitude_m: float = 2.5e-7,
    maximum_volume_change_m3: float = 1e-12,
) -> dict:
    base_path = Path(base_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    preparation_path = Path(preparation_path).resolve()
    ancestry_path = Path(ancestry_path).resolve()
    outputs = (candidate_path, preparation_path, ancestry_path)
    if base_path in outputs or len(set(outputs)) != len(outputs):
        raise ValueError("base and output paths must be distinct")
    raw = base_path.read_bytes()
    header, _, rows = _parse_payload(raw)
    if stable_id not in rows:
        raise ValueError(f"stable ID {stable_id} is absent from the base payload")
    row = rows[stable_id]
    vertices = row["vertices6"]
    faces = row["faces"]
    if not (0 <= keep_vertex < len(vertices) and 0 <= drop_vertex < len(vertices)):
        raise ValueError("collapse vertex index is outside the selected stable row")
    if keep_vertex == drop_vertex:
        raise ValueError("keep and drop vertices must differ")
    keep_point = tuple(map(float, vertices[keep_vertex, :3]))
    drop_point = tuple(map(float, vertices[drop_vertex, :3]))
    keep_owners = exact_point_owners(rows, vertices[keep_vertex, :3])
    drop_owners = exact_point_owners(rows, vertices[drop_vertex, :3])
    expected_owners = [int(stable_id)]
    if keep_owners != expected_owners or drop_owners != expected_owners:
        raise ValueError(
            "this CLI handles one source row only; both exact endpoints must be owned "
            f"only by stable ID {stable_id} (keep={keep_owners}, drop={drop_owners})"
        )

    prepared = {
        int(stable_id): (
            vertices.copy(),
            faces.copy(),
            {index: [index] for index in range(len(faces))},
            {},
        )
    }
    operation = collapse_registered_interior_vertex(
        prepared,
        keep_point,
        drop_point,
        maximum_source_displacement_m=maximum_source_displacement_m,
        minimum_altitude_m=minimum_altitude_m,
        maximum_volume_change_m3=maximum_volume_change_m3,
        volume_bounded_surface_ids=(stable_id,),
        complete_point_owners={keep_point: keep_owners, drop_point: drop_owners},
        allow_equal_owner_sets=True,
    )
    compacted_vertices, compacted_faces, parent_faces, _ = prepared[int(stable_id)]
    expected_faces, removed_face_indices = compact_faces_after_collapse(
        faces, keep_vertex, drop_vertex
    )
    if not np.array_equal(compacted_faces, expected_faces):
        raise ValueError("existing collapse owner output differs from the compact remap recipe")
    candidate_vertices = np.delete(vertices, drop_vertex, axis=0).copy()
    if len(candidate_vertices) != len(compacted_vertices):
        raise ValueError("retired source allocation count differs from owner output")
    if not np.array_equal(candidate_vertices[:, :3], compacted_vertices[:, :3]):
        raise ValueError("owner output positions differ from the exact compact source positions")

    face_ancestry = {}
    mapped_children = []
    removed = set(map(int, removed_face_indices))
    for source_face_index, source_face in enumerate(faces):
        children = list(map(int, parent_faces.get(source_face_index, [])))
        if source_face_index in removed:
            if children:
                raise ValueError("removed edge face unexpectedly has candidate ancestry")
            face_ancestry[str(source_face_index)] = []
            continue
        if len(children) != 1:
            raise ValueError("each surviving source face must map to exactly one candidate face")
        child_index = children[0]
        if child_index < 0 or child_index >= len(compacted_faces):
            raise ValueError("collapse owner returned an out-of-range face child")
        mapped_children.append(child_index)
        restored = compacted_faces[child_index].copy()
        restored[restored >= drop_vertex] += 1
        expected_source = source_face.copy()
        expected_source[expected_source == drop_vertex] = keep_vertex
        if not np.array_equal(restored, expected_source):
            raise ValueError("candidate face differs from the registered directed-collapse mapping")
        if not np.array_equal(restored, source_face):
            face_ancestry[str(source_face_index)] = [child_index]
    if sorted(mapped_children) != list(range(len(compacted_faces))):
        raise ValueError("candidate face ancestry is incomplete or duplicated")

    replacement = {
        **row,
        "vertices6": candidate_vertices,
        "faces": compacted_faces,
    }
    payload, abi = encode_payload_with_row_replacements(raw, {stable_id: replacement})
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(payload)
    ancestry_path.parent.mkdir(parents=True, exist_ok=True)
    ancestry_path.write_text(json.dumps(face_ancestry, indent=2) + "\n")
    ancestry_raw = ancestry_path.read_bytes()
    report = {
        "schema": "numi.human.registered-source-vertex-collapse-preparation.v1",
        "status": "experimental_geometry_candidate_prepared",
        "base": {"path": str(base_path), "sha256": sha256_bytes(raw)},
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256_bytes(payload),
            "abi": abi,
        },
        "stable_id": int(stable_id),
        "operation": operation,
        "allocation_retirement": {
            "policy": "retire_collapsed_source_slot",
            "keep_source_vertex": int(keep_vertex),
            "retired_source_vertex": int(drop_vertex),
            "source_vertex_count": int(len(vertices)),
            "candidate_vertex_count": int(len(candidate_vertices)),
            "candidate_index_remap": "source i maps to candidate i for i<drop; to i-1 for i>drop; dropped index is unused after collapse",
            "source_face_count": int(len(faces)),
            "candidate_face_count": int(len(compacted_faces)),
            "removed_source_face_indices": removed_face_indices.astype(int).tolist(),
            "surviving_vertex_attributes_preserved_byte_exact": True,
        },
        "face_ancestry": {
            "path": str(ancestry_path),
            "sha256": sha256_bytes(ancestry_raw),
            "changed_source_face_count": len(face_ancestry),
        },
        "native_admission": False,
        "scope": "Registered local source topology preparation only; runtime, common-field receipt rebinding, and anatomy admission are separate checks.",
    }
    preparation_path.parent.mkdir(parents=True, exist_ok=True)
    preparation_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="source NHANATOMY payload")
    parser.add_argument("--candidate", required=True, type=Path, help="compacted candidate NHANATOMY output")
    parser.add_argument("--preparation", required=True, type=Path, help="preparation receipt JSON output")
    parser.add_argument("--face-ancestry", required=True, type=Path, help="source-face ancestry JSON output")
    parser.add_argument("--stable-id", required=True, type=int)
    parser.add_argument("--keep-vertex", required=True, type=int)
    parser.add_argument("--drop-vertex", required=True, type=int)
    parser.add_argument("--maximum-source-displacement-m", type=float, default=1e-6)
    parser.add_argument("--minimum-altitude-m", type=float, default=2.5e-7)
    parser.add_argument("--maximum-volume-change-m3", type=float, default=1e-12)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing outputs")
    args = parser.parse_args()
    outputs = (args.candidate, args.preparation, args.face_ancestry)
    if args.base.resolve() in tuple(path.resolve() for path in outputs):
        parser.error("--base must differ from every output path")
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("candidate, preparation, and ancestry outputs must differ")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("an output already exists; choose new paths or pass --overwrite")
    result = prepare_collapse(
        args.base,
        args.candidate,
        args.preparation,
        args.face_ancestry,
        args.stable_id,
        args.keep_vertex,
        args.drop_vertex,
        args.maximum_source_displacement_m,
        args.minimum_altitude_m,
        args.maximum_volume_change_m3,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
