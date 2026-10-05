"""Build the existing passive liver exterior guide for one pinned aggregate mesh.

The output remains the repository's existing NPZ plus JSON receipt. The eight
labels are nearest registered source-segment candidates on one aggregate
surface; they do not define internal partitions or segment volumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import resting_anatomy_interface_patch as anatomy

STABLE_IDS = tuple(range(14, 22))
PINNED_CANDIDATE_SHA256 = "14d70c30802a87e9c80084bd11c8bf260571c1d97e4fed583fcc8769f83df379"
PINNED_CANDIDATE_REPORT_SHA256 = "8284e97e910527246669ddcfe5e6268d74d3daaa1d3c91b3c9b955a27a6c72b8"
PINNED_SOURCE_INVENTORY_SHA256 = "0e70241d420205c2d50cf7f745ef241a48e437b9cdb42e5cdbc38d625aebb8b0"
PINNED_PRIOR_GUIDE_RECEIPT_SHA256 = "b98cdfe8ae751938262777ada741b82e70f19a261d36a50feb4393dd4a172446"
PINNED_ORIGINAL_LIVER_SHA256 = "97d273693bea34dfa30e5269756ae17e7a89bf16b1be591fbb1ac3088597f933"
PINNED_IMPRESSION_INPUT_PAYLOAD_SHA256 = "105addebae2533038d671967bd51a24b4d667d13bfea2ab1b13ec3e0481dc518"
PINNED_SOURCE_OWNER_VALUES = (0, 2, 4, 398, 454, 458, 460, 461)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_triangle_distances(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Exact float64 point-to-triangle distances for paired point/candidate rows."""
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 4 or t.shape[0] != len(p) or t.shape[2:] != (3, 3):
        raise ValueError("point/triangle candidate arrays have inconsistent shapes")
    a, b, c = t[:, :, 0, :], t[:, :, 1, :], t[:, :, 2, :]
    ab, ac, ap = b - a, c - a, p[:, None, :] - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum("nki,nki->nk", normal, normal)
    if np.any(normal_sq <= 0) or not np.isfinite(normal_sq).all():
        raise ValueError("registered segment source contains a degenerate candidate triangle")
    signed = np.einsum("nki,nki->nk", ap, normal) / np.sqrt(normal_sq)
    projected = p[:, None, :] - (signed / np.sqrt(normal_sq))[:, :, None] * normal
    v0, v1, v2 = ab, ac, projected - a
    d00 = np.einsum("nki,nki->nk", v0, v0)
    d01 = np.einsum("nki,nki->nk", v0, v1)
    d11 = np.einsum("nki,nki->nk", v1, v1)
    d20 = np.einsum("nki,nki->nk", v2, v0)
    d21 = np.einsum("nki,nki->nk", v2, v1)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= 0):
        raise ValueError("registered segment source contains a numerically degenerate triangle")
    u = (d11 * d20 - d01 * d21) / denominator
    v = (d00 * d21 - d01 * d20) / denominator
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    broadcast_points = np.broadcast_to(p[:, None, :], a.shape)

    def segment_distance(x: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        edge = end - start
        edge_sq = np.einsum("nki,nki->nk", edge, edge)
        if np.any(edge_sq <= 0):
            raise ValueError("registered segment source contains a collapsed triangle edge")
        tval = np.einsum("nki,nki->nk", x - start, edge) / edge_sq
        tval = np.clip(tval, 0.0, 1.0)
        closest = start + tval[:, :, None] * edge
        return np.linalg.norm(x - closest, axis=2)

    edge_distance = np.minimum(
        np.minimum(segment_distance(broadcast_points, a, b), segment_distance(broadcast_points, b, c)),
        segment_distance(broadcast_points, c, a),
    )
    result = np.where(inside, np.abs(signed), edge_distance)
    if not np.isfinite(result).all():
        raise ValueError("point-to-triangle candidate distance is nonfinite")
    return result


def _nearest_candidates(points: np.ndarray, objects: list[dict[str, Any]], k: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    from scipy.spatial import cKDTree

    distances, triangle_indices = [], []
    for obj in objects:
        source_vertices = np.asarray(obj["body20_vertices_m"], dtype=np.float64).copy()
        source_vertices[:, 1] -= 0.006
        source_faces = np.asarray(obj["triangles"], dtype=np.int64)
        if source_vertices.ndim != 2 or source_vertices.shape[1] != 3 or source_faces.ndim != 2 or source_faces.shape[1] != 3:
            raise ValueError("registered segment source mesh is malformed")
        if source_faces.size == 0 or source_faces.min() < 0 or source_faces.max() >= len(source_vertices):
            raise ValueError("registered segment source face indices are invalid")
        source_triangles = source_vertices[source_faces]
        tree = cKDTree(source_triangles.mean(axis=1))
        per_object_distance, per_object_index = [], []
        candidate_count = min(k, len(source_triangles))
        for start in range(0, len(points), 192):
            batch = points[start : start + 192]
            _, candidate_indices = tree.query(batch, k=candidate_count, workers=1)
            if candidate_indices.ndim == 1:
                candidate_indices = candidate_indices[:, None]
            candidate_triangles = source_triangles[candidate_indices]
            candidate_distances = point_triangle_distances(batch, candidate_triangles)
            nearest = candidate_distances.argmin(axis=1)
            row_indices = np.arange(len(batch))
            per_object_distance.append(candidate_distances[row_indices, nearest])
            per_object_index.append(candidate_indices[row_indices, nearest])
        distances.append(np.concatenate(per_object_distance))
        triangle_indices.append(np.concatenate(per_object_index))
    return np.stack(distances, axis=1), np.stack(triangle_indices, axis=1)


def build_guide(*, aggregate_npz: Path, registered_segments_json: Path,
                prior_guide_receipt: Path, candidate_report: Path,
                output_dir: Path) -> dict[str, Any]:
    aggregate_npz = Path(aggregate_npz)
    registered_segments_json = Path(registered_segments_json)
    prior_guide_receipt = Path(prior_guide_receipt)
    candidate_report = Path(candidate_report)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for label, path in (("aggregate liver", aggregate_npz), ("registered segments", registered_segments_json),
                        ("prior guide receipt", prior_guide_receipt), ("candidate report", candidate_report)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    exact_inputs = (
        ("aggregate liver candidate", aggregate_npz, PINNED_CANDIDATE_SHA256),
        ("candidate report", candidate_report, PINNED_CANDIDATE_REPORT_SHA256),
        ("registered source-segment inventory", registered_segments_json, PINNED_SOURCE_INVENTORY_SHA256),
        ("prior source guide receipt", prior_guide_receipt, PINNED_PRIOR_GUIDE_RECEIPT_SHA256),
    )
    for label, path, expected in exact_inputs:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA mismatch: {actual}")
    candidate_meta = json.loads(candidate_report.read_text())
    if (candidate_meta.get("candidate_sha256") != sha256(aggregate_npz)
            or candidate_meta.get("input_payload_sha256") != PINNED_IMPRESSION_INPUT_PAYLOAD_SHA256
            or candidate_meta.get("liver_source_sha256") != PINNED_ORIGINAL_LIVER_SHA256
            or candidate_meta.get("original_diaphragm_interface_faces") != 849
            or candidate_meta.get("preserved_diaphragm_interface_faces") != 849
            or candidate_meta.get("lost_diaphragm_interface_faces") != 0):
        raise ValueError("visceral-impression report does not bind the pinned source and all 849 diaphragm contacts")
    prior = json.loads(prior_guide_receipt.read_text())
    source = json.loads(registered_segments_json.read_text())
    if prior.get("source_inputs", {}).get("registered_segment_export_sha256") != sha256(registered_segments_json):
        raise ValueError("prior guide receipt does not bind the registered source-segment inventory")
    objects = source.get("objects")
    if not isinstance(objects, list) or len(objects) != 9:
        raise ValueError("registered inventory must contain the torso object and exactly eight liver segment meshes")
    objects = objects[1:]
    prior_names = [row.get("zanatomy_object") for row in prior.get("segment_crosswalk", []) if "zanatomy_object" in row]
    names = [obj.get("object_name") for obj in objects]
    if len(names) != 8 or prior_names != names:
        raise ValueError("registered segment mesh identities differ from the retained source crosswalk")

    aggregate = np.load(aggregate_npz)
    required = {"vertices", "source_owner"}
    face_key = "faces" if "faces" in aggregate.files else "triangles" if "triangles" in aggregate.files else None
    if not required.issubset(aggregate.files) or face_key is None:
        raise ValueError("aggregate candidate lacks its registered vertices/faces/source-owner arrays")
    vertices = np.asarray(aggregate["vertices"], dtype=np.float32)
    faces = np.asarray(aggregate[face_key], dtype=np.int64)
    source_owner = np.asarray(aggregate["source_owner"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("aggregate candidate vertices are malformed")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("aggregate candidate triangles are malformed")
    if source_owner.shape != (len(faces),):
        raise ValueError("aggregate source-owner lineage must have one value per face")
    if tuple(sorted(map(int, np.unique(source_owner)))) != PINNED_SOURCE_OWNER_VALUES:
        raise ValueError("visceral-impression source-owner lineage differs from its retained organ-side set")
    topology = anatomy.topology_report(faces)
    topology_summary = {key: value for key, value in topology.items() if key != "boundary_edges"}
    if any(topology[key] for key in ("boundary_edge_count", "nonmanifold_edge_count", "orientation_error_edge_count", "boundary_branch_vertex_count")):
        raise ValueError(f"aggregate candidate is not closed and oriented: {topology_summary}")
    volume_m3 = anatomy.signed_volume(vertices.astype(np.float64), faces)
    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("aggregate candidate has nonpositive signed volume")
    triangles = vertices[faces].astype(np.float64)
    centroids = triangles.mean(axis=1)
    face_area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(face_area <= 0) or not np.isfinite(face_area).all():
        raise ValueError("aggregate candidate contains an exactly degenerate triangle")

    distance32, triangle32 = _nearest_candidates(centroids, objects, 32)
    distance128, triangle128 = _nearest_candidates(centroids, objects, 128)
    choice = distance128.argmin(axis=1)
    stable_ids = np.asarray(STABLE_IDS, dtype=np.uint32)[choice]
    sorted_distance = np.sort(distance128, axis=1)
    margin = sorted_distance[:, 1] - sorted_distance[:, 0]
    changed = distance32.argmin(axis=1) != choice
    source_owner_values, owner_counts = np.unique(source_owner, return_counts=True)
    assignment_by_owner = {
        str(int(owner)): {str(sid): int(np.count_nonzero((source_owner == owner) & (stable_ids == sid))) for sid in STABLE_IDS}
        for owner in source_owner_values
    }
    summary = []
    for column, (sid, obj) in enumerate(zip(STABLE_IDS, objects)):
        mask = choice == column
        distances = distance128[mask, column]
        if not len(distances):
            raise ValueError(f"nearest-source candidate assigned no aggregate faces to stable ID {sid}")
        summary.append({
            "stable_id": sid,
            "registered_z_source_object": obj["object_name"],
            "source_triangles": len(obj["triangles"]),
            "aggregate_boundary_faces_assigned": int(mask.sum()),
            "boundary_area_m2": float(face_area[mask].sum()),
            "distance_to_source_triangle_candidate_m": {
                "median": float(np.median(distances)), "p95": float(np.percentile(distances, 95)),
                "p99": float(np.percentile(distances, 99)), "max": float(distances.max()),
            },
            "faces_over_2mm": int(np.count_nonzero(distances > 0.002)),
            "faces_over_5mm": int(np.count_nonzero(distances > 0.005)),
        })

    output_dir.mkdir(parents=True)
    guide_path = output_dir / "aggregate-liver-exterior-segment-guide.npz"
    receipt_path = output_dir / "aggregate-liver-exterior-segment-guide.receipt.json"
    np.savez_compressed(
        guide_path,
        vertices=vertices.astype("<f4"), triangles=faces.astype("<u4"),
        stable_id_per_triangle=stable_ids,
        source_segment_distance_candidate_m=np.min(distance128, axis=1).astype("<f4"),
        segment_choice_margin_candidate_m=margin.astype("<f4"),
        nearest_candidate_triangle_index_per_segment=triangle128.astype("<u4"),
        source_owner_per_triangle=source_owner.astype("<i4"),
    )
    prior_source_inputs = prior["source_inputs"]
    receipt = {
        "schema": "numi.human.aggregate-liver-exterior-segment-guides.v1",
        "classification": "Each triangle of one closed aggregate liver exterior is labeled by nearest registered Z-Anatomy segment-boundary candidates. The eight labels are passive display identities only; no internal Couinaud planes or independent segment volumes are defined.",
        "source_inputs": {
            **prior_source_inputs,
            "registered_segment_export_sha256": sha256(registered_segments_json),
            "aggregate_liver_clipped_candidate_sha256": sha256(aggregate_npz),
            "prior_guide_receipt_sha256": sha256(prior_guide_receipt),
            "visceral_impression_candidate_report_sha256": sha256(candidate_report),
            "visceral_impression_driver_sha256": candidate_meta.get("driver_sha256"),
            "original_aggregate_liver_source_sha256": PINNED_ORIGINAL_LIVER_SHA256,
            "source_candidate_face_owner_values": {str(int(owner)): int(count) for owner, count in zip(source_owner_values, owner_counts)},
            "transformation": "Retained registered segment meshes in torso20 frame with the existing inferred caudal adjustment (0,-0.006,0) m; original source meshes and aggregate candidate remain separately identified.",
        },
        "assignment_method": {
            "target": "centroid of each aggregate boundary triangle",
            "source_distance": "Euclidean point-to-triangle distance among 128 source triangles whose centroids are nearest by cKDTree",
            "candidate_completeness": "upper-bound nearest surface candidates, not certified global closest-point search",
            "k_sensitivity": {"label_changed_between_32_and_128": int(np.count_nonzero(changed)), "fraction_changed": float(np.mean(changed))},
            "ambiguous_nearest_label_margin_under_0_25mm": int(np.count_nonzero(margin < 0.00025)),
        },
        "aggregate_surface": {
            "source_path": str(aggregate_npz), "sha256": sha256(aggregate_npz),
            "vertex_count": int(len(vertices)), "triangle_count": int(len(faces)),
            "signed_volume_m3": float(volume_m3), "closed_oriented": True,
            "exact_topology": topology_summary,
            "source_owner_semantics": "CSG face-lineage owner values identify the organ-side source used to form the aggregate difference; they are not liver-segment labels.",
            "source_owner_face_counts": {str(int(owner)): int(count) for owner, count in zip(source_owner_values, owner_counts)},
            "max_nearest_source_triangle_candidate_distance_m": float(np.min(distance128, axis=1).max()),
            "distance_quantiles_m": np.percentile(np.min(distance128, axis=1), [0, 25, 50, 75, 90, 95, 99, 100]).tolist(),
        },
        "segment_crosswalk": prior["segment_crosswalk"],
        "segment_assignment_summary": summary,
        "assignment_by_source_face_owner": assignment_by_owner,
        "output": {"path": str(guide_path), "sha256": sha256(guide_path)},
        "qualification": {
            "source_identity": "source mesh names, archive, registration, and candidate hashes are retained",
            "label_interpretation": "inferred exterior display attribution only",
            "segment_volume_accounting": "one aggregate exterior/volume owner; no separate segment volumes",
            "full_surface_closest_point": "not certified; candidate-triangle upper-bound method retained from guide v1",
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {
        "guide_path": str(guide_path), "guide_sha256": sha256(guide_path),
        "receipt_path": str(receipt_path), "receipt_sha256": sha256(receipt_path),
        "candidate_sha256": sha256(aggregate_npz), "candidate_volume_m3": float(volume_m3),
        "topology": topology_summary, "k_sensitivity": receipt["assignment_method"]["k_sensitivity"],
        "segment_assignment_summary": summary, "source_owner_values": receipt["source_inputs"]["source_candidate_face_owner_values"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-npz", type=Path, required=True)
    parser.add_argument("--registered-segments-json", type=Path, required=True)
    parser.add_argument("--prior-guide-receipt", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_guide(aggregate_npz=args.aggregate_npz,
        registered_segments_json=args.registered_segments_json,
        prior_guide_receipt=args.prior_guide_receipt, candidate_report=args.candidate_report,
        output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
