"""Independently check the bounded, unadopted left-ACL initial-state candidate."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from numilab_human.open_knee import (
    HEADER_STRUCT, NODE_SET_STRUCT, NODE_STRUCT, REGION_STRUCT,
    SURFACE_PAIR_STRUCT, SURFACE_STRUCT, TETRAHEDRON_STRUCT,
)
from tools.audit_open_knee_cruciate_source_contact import (
    decode, strictly_crosses,
)
from tools.audit_patellofemoral_tetrahedral_separation import (
    SCALE, aabb_candidates, exact_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Docs/media/cruciate-source-contact-20260930/initialization-candidate.json"
SOURCE_AUDIT = OUTPUT.with_name("receipt.json")
PAIRS = (
    ("PCL_@_ACL_ContactFaces", "ACL_@_PCL_ContactFaces"),
    ("ACL_@_FMB_ContactFaces", "FMB_@_ACL_ContactFaces"),
    ("ACL_@_TBB_ContactFaces", "TBB_@_ACL_ContactFaces"),
)
CENTER = np.array((0.05168572, 0.15708857, 0.50562678))
PLATEAU_M = 0.003
OUTER_M = 0.008
SHIFT_M = 0.0005


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("cruciate initialization: " + message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_crossings(first_name: str, second_name: str,
                    surfaces: dict, faces: np.ndarray,
                    positions: np.ndarray) -> dict:
    first_start, first_count = surfaces[first_name]
    second_start, second_count = surfaces[second_name]
    first_points = positions[faces[first_start:first_start + first_count]]
    second_points = positions[faces[second_start:second_start + second_count]]

    @lru_cache(maxsize=8192)
    def first(index: int):
        return tuple(tuple(int(x) for x in vertex)
                     for vertex in first_points[index])

    candidate_count = crossing_count = 0
    for i, j in aabb_candidates(first_points.min(axis=1),
                                 first_points.max(axis=1),
                                 second_points.min(axis=1),
                                 second_points.max(axis=1)):
        candidate_count += 1
        other = tuple(tuple(int(x) for x in vertex)
                      for vertex in second_points[j])
        crossing_count += strictly_crosses(first(i), other)
    return {"aabb_candidates": candidate_count,
            "strict_crossings": crossing_count}


def run() -> dict:
    raw, regions, surfaces, _, source, faces = decode()
    source_audit_bytes = SOURCE_AUDIT.read_bytes()
    source_audit = json.loads(source_audit_bytes)
    require(source_audit["payload_sha256"] == sha(raw) and
            source_audit["exact_strict_crossing_face_pairs"] == 84 and
            source_audit["source_contact_pairs_with_initial_crossings"] == 8,
            "pinned source contact audit changed")
    h = HEADER_STRUCT.unpack_from(raw)
    acl_index = [r[0] for r in regions].index("ACL")
    _, first, count = regions[acl_index]
    require(count == 15792, "source ACL node count changed")
    node_offset = (HEADER_STRUCT.size + h[3] * REGION_STRUCT.size +
                   h[6] * SURFACE_STRUCT.size +
                   h[8] * NODE_SET_STRUCT.size +
                   h[10] * SURFACE_PAIR_STRUCT.size)
    anchored = np.array([
        bool(NODE_STRUCT.unpack_from(
            raw, node_offset + index * NODE_STRUCT.size)[-1] & 1)
        for index in range(first, first + count)], dtype=bool)
    original = source.astype(np.float64) / SCALE
    candidate = original.copy()
    radius = np.linalg.norm(original[first:first + count] - CENTER, axis=1)
    u = np.clip((radius - PLATEAU_M) / (OUTER_M - PLATEAU_M), 0, 1)
    weight = 1.0 - u * u * (3.0 - 2.0 * u)
    weight[anchored] = 0.0
    candidate[first:first + count, 0] += SHIFT_M * weight
    candidate = candidate.astype(np.float32)
    candidate_exact = exact_lattice(candidate)
    moved = np.linalg.norm(
        candidate[first:first + count].astype(np.float64) -
        original[first:first + count], axis=1)
    require(np.count_nonzero(weight) == 4448 and
            not np.any(candidate_exact[first:first + count][anchored] !=
                       source[first:first + count][anchored]) and
            0.000499 < moved.max() <= 0.000501,
            "candidate free-node correction or rigid ties drifted")

    pair_rows = {}
    for left, right in PAIRS:
        pair_rows[left + "/" + right] = {
            "source": count_crossings(left, right, surfaces, faces, source),
            "candidate": count_crossings(
                left, right, surfaces, faces, candidate_exact),
        }
    require(pair_rows[PAIRS[0][0] + "/" + PAIRS[0][1]]
            ["source"]["strict_crossings"] == 84 and
            all(row["candidate"]["strict_crossings"] == 0
                for row in pair_rows.values()),
            "candidate retains a named ACL contact crossing")

    region = REGION_STRUCT.unpack_from(
        raw, HEADER_STRUCT.size + acl_index * REGION_STRUCT.size)
    tet_offset = node_offset + h[4] * NODE_STRUCT.size
    tetrahedra = np.frombuffer(raw, dtype="<u4", count=h[5] * 4,
                               offset=tet_offset).reshape(-1, 4)
    acl_cells = tetrahedra[region[5]:region[5] + region[6]]

    def volumes(points: np.ndarray) -> np.ndarray:
        xyz = points[acl_cells].astype(np.float64) / SCALE
        return np.einsum(
            "ij,ij->i", np.cross(xyz[:, 1] - xyz[:, 0],
                                  xyz[:, 2] - xyz[:, 0]),
            xyz[:, 3] - xyz[:, 0]) / 6.0

    source_volume = volumes(source)
    candidate_volume = volumes(candidate_exact)
    require(np.all(source_volume != 0.0),
            "source ACL has a degenerate tetrahedron")
    ratios = candidate_volume / source_volume
    require(np.all(ratios > 0.0) and ratios.min() >= 0.84 and
            ratios.max() <= 1.05,
            "candidate inverts or excessively distorts ACL tetrahedra")

    return {
        "schema": "numi.human.cruciate-initialization-candidate.v1",
        "status": "unadopted_local_acl_separation_candidate",
        "source_contact_audit_sha256": sha(source_audit_bytes),
        "source_payload_sha256": sha(raw),
        "center_source_world_m": CENTER.tolist(),
        "plateau_radius_m": PLATEAU_M,
        "outer_radius_m": OUTER_M,
        "maximum_authored_shift_m": SHIFT_M,
        "shift_axis": "source_world_positive_x",
        "shifted_free_acl_nodes": int(np.count_nonzero(weight)),
        "anchored_acl_nodes_changed": 0,
        "maximum_actual_shift_m": float(moved.max()),
        "acl_tetrahedra": len(acl_cells),
        "minimum_source_to_candidate_tetra_volume_ratio": float(ratios.min()),
        "maximum_source_to_candidate_tetra_volume_ratio": float(ratios.max()),
        "inverted_acl_tetrahedra": 0,
        "named_acl_contact_pairs": pair_rows,
        "whole_knee_source_contact_pairs_with_crossings":
            source_audit["source_contact_pairs_with_initial_crossings"],
        "loaded_knee_qualified": False,
        "boundary": "This free-node ACL correction is a bounded diagnostic, not a source-authored unloaded configuration. It clears named ACL contact-surface crossings and preserves ACL volume orientation, but seven other source contact pairs retain initial crossings and native patellar-tendon force transfer still fails its physical gate.",
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("status", "shifted_free_acl_nodes",
                       "minimum_source_to_candidate_tetra_volume_ratio")},
                     sort_keys=True))
