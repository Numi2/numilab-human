#!/usr/bin/env python3
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
ROOT = BASE / "fine-physical-volume-706/current-native-1024-root"
CURRENT_NPZ = ROOT / "current-433-1037/candidate-current-typed-volume.npz"
CURRENT_RECEIPT = ROOT / "current-433-1044/volume-export/receipt.json"
NODE_MAP = ROOT / "current-433-1044/volume-export/volume/combined-node-map.csv"
CURRENT_NODE = ROOT / "current-433-1044/volume-export/volume/mesh.node"
CURRENT_ELE = ROOT / "current-433-1044/volume-export/volume/mesh.ele"
SOURCE002_NPZ = BASE / "reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
PARENT330_PACKET = BASE / "native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
PULLBACK514 = BASE / "operational-reference-map-514/run-005-owner-authenticated-import/actual_import_check.cpp"
PULLBACK514_STDOUT = BASE / "operational-reference-map-514/run-005-owner-authenticated-import/actual-result.stdout"
PULLBACK514_MANIFEST = BASE / "operational-reference-map-514/run-005-owner-authenticated-import/launch.json"
OUT_DIR = ROOT / "source002-composed-preimage-audit-003/run-001"
OUT = OUT_DIR / "report.json"
TOL = 2.0e-12

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def det6(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    a = points[tets[:, 1]] - p0
    b = points[tets[:, 2]] - p0
    c = points[tets[:, 3]] - p0
    return np.einsum("ij,ij->i", a, np.cross(b, c))

def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.median(values)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }

if OUT_DIR.exists():
    raise SystemExit(f"refusing existing evidence directory: {OUT_DIR}")
OUT_DIR.mkdir(parents=True)
started = time.monotonic()

expected = {
    str(CURRENT_NPZ): "5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372",
    str(NODE_MAP): "457cdd42274199f51d5df247cb79e998c319dc49ec5b67820e8e78e09e4caee6",
    str(SOURCE002_NPZ): "2f547b21c6be0f3e21a021d80dad6c63c3a0e6af31d5ebe1780ca1b32c237a54",
    str(PARENT330_PACKET): "c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679",
    str(PULLBACK514): "2565fe01931e16c6b37f018160b45f9bbe9cf784f8141f69a68f1249825f0675",
}
actual_hashes = {p: sha(Path(p)) for p in expected}
assert actual_hashes == expected, (actual_hashes, expected)
assert "source330_to_source002_max_m=0.000351375" in PULLBACK514_STDOUT.read_text()
pullback_manifest = json.loads(PULLBACK514_MANIFEST.read_text())
assert pullback_manifest["status"] == "PASS"

source = np.load(SOURCE002_NPZ, allow_pickle=False)
source_points = np.asarray(source["points_reference_world_m_f64"], dtype=np.float64)
source_tets = np.asarray(source["volume_tetrahedra"], dtype=np.int64)
source_mat = np.asarray(source["material_index"], dtype=np.int64)
source_donor = np.asarray(source["geometric_donor_body"], dtype=np.int64)
source_region = np.asarray(source["anatomical_region_id"], dtype=np.int64)
source_face = np.asarray(source["stable_source_face_id"], dtype=np.int64)
assert source_points.shape == (7233, 3)
assert source_tets.shape == (28632, 4)

with np.load(CURRENT_NPZ, allow_pickle=False) as current:
    current_points_f64 = np.asarray(current["points_reference_world_m_f64"], dtype=np.float64)
    current_tets = np.asarray(current["volume_tetrahedra"], dtype=np.int64)
    child_parent = np.asarray(current["child_parent_tet_row"], dtype=np.int64)
    current_mat = np.asarray(current["material_index"], dtype=np.int64)
    current_donor = np.asarray(current["geometric_donor_body"], dtype=np.int64)
    current_region = np.asarray(current["anatomical_region_id"], dtype=np.int64)
    current_face = np.asarray(current["stable_source_face_id"], dtype=np.int64)

node_file = np.loadtxt(CURRENT_NODE, skiprows=1, dtype=np.float64)
assert current_points_f64.shape == (311768, 3)
assert current_tets.shape == (1495294, 4)
assert child_parent.shape == current_mat.shape == current_donor.shape == current_region.shape == current_face.shape == (len(current_tets),)
assert node_file.shape == (len(current_points_f64), 4)
current_points_f32 = np.asarray(current_points_f64, dtype=np.float32)
node_points_f32 = np.asarray(node_file[:, 1:], dtype=np.float32)
assert np.array_equal(current_points_f32.view(np.uint32), node_points_f32.view(np.uint32))
assert np.array_equal(node_file[:, 0].astype(np.int64), np.arange(1, len(current_points_f64) + 1))
assert current_tets.min() >= 0 and current_tets.max() < len(current_points_f64)
assert child_parent.min() >= 0 and child_parent.max() < len(source_tets)

parent_tet_for_node = np.full(len(current_points_f64), -1, dtype=np.int64)
parent_bary_for_node = np.full((len(current_points_f64), 4), np.nan, dtype=np.float64)
node_role = np.empty(len(current_points_f64), dtype="U16")
node_donor = np.full(len(current_points_f64), -1, dtype=np.int64)
node_region = np.full(len(current_points_f64), -1, dtype=np.int64)
row_multiplicity = np.zeros(len(current_points_f64), dtype=np.uint8)
duplicate_conflicts = 0
kind_counts: collections.Counter[str] = collections.Counter()
with NODE_MAP.open(newline="") as f:
    for row in csv.DictReader(f):
        node_id = int(row["joint_node_id"])
        i = node_id - 1
        assert 0 <= i < len(current_points_f64)
        p = int(row["parent_tet_row"])
        b = np.asarray([float(row[f"parent_bary{k}"]) for k in range(4)], dtype=np.float64)
        role = row["role"]
        kind_counts[row["parent_registration_kind"]] += 1
        if row_multiplicity[i] == 0:
            parent_tet_for_node[i] = p
            parent_bary_for_node[i] = b
            node_role[i] = role
            node_donor[i] = int(row["source_parent_geometric_donor_body"])
            node_region[i] = int(row["source_parent_anatomical_region_id"])
        elif (
            parent_tet_for_node[i] != p
            or not np.array_equal(parent_bary_for_node[i], b)
            or node_donor[i] != int(row["source_parent_geometric_donor_body"])
            or node_region[i] != int(row["source_parent_anatomical_region_id"])
            or node_role[i] != role
        ):
            duplicate_conflicts += 1
        row_multiplicity[i] += 1

assert np.all(row_multiplicity > 0)
assert duplicate_conflicts == 0
assert parent_tet_for_node.min() >= 0 and parent_tet_for_node.max() < len(source_tets)
assert np.isfinite(parent_bary_for_node).all()
assert float(np.max(np.abs(parent_bary_for_node.sum(axis=1) - 1.0))) <= 1e-12
assert float(np.min(parent_bary_for_node)) >= -TOL
assert float(np.max(parent_bary_for_node)) <= 1.0 + TOL
mapped_node_points = np.einsum(
    "ni,nij->nj",
    parent_bary_for_node,
    source_points[source_tets[parent_tet_for_node]],
)
source_map_residual = np.linalg.norm(current_points_f64 - mapped_node_points, axis=1)

source_parent_det6 = det6(source_points, source_tets)
current_det6 = det6(current_points_f64, current_tets)
preimage_det6 = det6(mapped_node_points, current_tets)
assert np.all(np.isfinite(source_parent_det6))
assert np.all(np.isfinite(current_det6))
assert np.all(np.isfinite(preimage_det6))
source_parent_volume = source_parent_det6 / 6.0
current_volume = current_det6 / 6.0
preimage_volume = preimage_det6 / 6.0
safe_current = current_det6 != 0.0
jacobian = np.full(len(current_det6), np.nan, dtype=np.float64)
jacobian[safe_current] = preimage_det6[safe_current] / current_det6[safe_current]

# Re-express each preimage vertex in the child row's declared parent tetrahedron.
parent_origins = source_points[source_tets[:, 0]]
parent_edges = np.swapaxes(
    source_points[source_tets[:, 1:]] - parent_origins[:, None, :], 1, 2
)
parent_det = np.linalg.det(parent_edges)
assert np.all(np.isfinite(parent_det)) and np.all(parent_det != 0.0)
parent_inverse = np.linalg.inv(parent_edges)
cell_parent_edges = parent_inverse[child_parent]
cell_origin = parent_origins[child_parent]
cell_points = mapped_node_points[current_tets]
delta = cell_points - cell_origin[:, None, :]
lambda123 = np.einsum("tij,tkj->tki", cell_parent_edges, delta)
lambda0 = 1.0 - np.sum(lambda123, axis=2)
cell_parent_bary = np.concatenate([lambda0[:, :, None], lambda123], axis=2)
cell_parent_min = np.min(cell_parent_bary, axis=(1, 2))
cell_parent_max = np.max(cell_parent_bary, axis=(1, 2))
outside_cell_parent = np.any(
    (cell_parent_bary < -TOL) | (cell_parent_bary > 1.0 + TOL), axis=(1, 2)
)
node_parent_for_cells = parent_tet_for_node[current_tets]
node_parent_mismatch = node_parent_for_cells != child_parent[:, None]
mismatch_cells = np.any(node_parent_mismatch, axis=1)
mismatch_incidence_count = int(np.count_nonzero(node_parent_mismatch))
mismatch_on_parent_simplex = np.any(np.abs(cell_parent_bary) <= TOL, axis=(1, 2))
mismatch_interior = mismatch_cells & ~mismatch_on_parent_simplex

parent_child_count = np.bincount(child_parent, minlength=len(source_tets))
parent_preimage_volume = np.bincount(child_parent, weights=preimage_volume, minlength=len(source_tets))
parent_volume_ratio = parent_preimage_volume / source_parent_volume
assert len(parent_volume_ratio) == len(source_tets)

def per_material_summary(labels: np.ndarray, volume: np.ndarray) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for material in sorted(int(x) for x in np.unique(labels)):
        mask = labels == material
        out[str(material)] = {
            "tet_count": int(mask.sum()),
            "signed_volume_m3": float(np.sum(volume[mask])),
            "absolute_volume_m3": float(np.sum(np.abs(volume[mask]))),
            "negative_det_count": int(np.count_nonzero(volume[mask] < 0.0)),
            "zero_det_count": int(np.count_nonzero(volume[mask] == 0.0)),
        }
    return out

label_mismatch = {
    "material": int(np.count_nonzero(current_mat != source_mat[child_parent])),
    "geometric_donor": int(np.count_nonzero(current_donor != source_donor[child_parent])),
    "region": int(np.count_nonzero(current_region != source_region[child_parent])),
    "stable_source_face": int(np.count_nonzero(current_face != source_face[child_parent])),
}

def witnesses(mask: np.ndarray, n: int = 12) -> list[dict[str, object]]:
    ix = np.flatnonzero(mask)[:n]
    out: list[dict[str, object]] = []
    for cell in ix:
        out.append({
            "child_tet_row": int(cell),
            "declared_parent_tet_row": int(child_parent[cell]),
            "node_rows": [int(x) for x in current_tets[cell]],
            "node_parent_tet_rows": [int(x) for x in node_parent_for_cells[cell]],
            "node_parent_barycentric": [
                [float(x) for x in parent_bary_for_node[int(node)]]
                for node in current_tets[cell]
            ],
            "reexpressed_parent_barycentric": [
                [float(x) for x in xrow] for xrow in cell_parent_bary[cell]
            ],
            "current_det6_m3": float(current_det6[cell]),
            "preimage_det6_m3": float(preimage_det6[cell]),
            "jacobian": float(jacobian[cell]) if np.isfinite(jacobian[cell]) else None,
        })
    return out

label_cross_cell_counts = {}
for key, arr in [("material", current_mat), ("geometric_donor", current_donor), ("region", current_region)]:
    mismatch_count = np.count_nonzero(arr != ({"material": source_mat, "geometric_donor": source_donor, "region": source_region}[key])[child_parent])
    label_cross_cell_counts[key] = int(mismatch_count)

parent_coverage = {
    "parent_rows": int(len(source_tets)),
    "without_child": int(np.count_nonzero(parent_child_count == 0)),
    "child_count_per_parent": {
        "min": int(parent_child_count.min()), "p50": float(np.median(parent_child_count)),
        "p99": float(np.quantile(parent_child_count, 0.99)), "max": int(parent_child_count.max())
    },
    "signed_preimage_vs_parent_volume_ratio": stats(parent_volume_ratio),
    "parents_abs_rel_volume_error_gt_1e-6": int(np.count_nonzero(np.abs(parent_volume_ratio - 1.0) > 1e-6)),
    "parents_abs_rel_volume_error_gt_1e-3": int(np.count_nonzero(np.abs(parent_volume_ratio - 1.0) > 1e-3)),
    "max_abs_parent_volume_delta_m3": float(np.max(np.abs(parent_preimage_volume - source_parent_volume))),
}

report = {
    "schema": "numi.h706.current1020.source002.composed-preimage-audit.v1",
    "status": "COMPLETED_READ_ONLY_GEOMETRY_DIAGNOSTIC",
    "physical_admission": False,
    "native_steps": 0,
    "geometry_execution": False,
    "source_registration_note": "The 514 source330->source002 proof is a same-index, same-tet/labels mapping, with measured source330-to-source002 max node residual 0.351375 mm. That threshold applies to that leg; current candidate-to-source002 displacement below is reported as data, not a rejection threshold.",
    "method": "Map each unique current 433 node through pinned source330 parent_tet_row+bary into source002 points; compute signed current/preimage det6 and per-material/per-parent volume. Re-express each mapped cell vertex in its declared source002 parent tet to test single-parent containment.",
    "inputs": {
        "current_candidate_npz": {"path": str(CURRENT_NPZ), "sha256": sha(CURRENT_NPZ)},
        "current_433_1044_receipt": {"path": str(CURRENT_RECEIPT), "sha256": sha(CURRENT_RECEIPT)},
        "current_node_parent_map": {"path": str(NODE_MAP), "sha256": sha(NODE_MAP), "rows": int(row_multiplicity.sum()), "unique_nodes": int(len(row_multiplicity))},
        "current_native_node": {"path": str(CURRENT_NODE), "sha256": sha(CURRENT_NODE)},
        "current_native_elements": {"path": str(CURRENT_ELE), "sha256": sha(CURRENT_ELE)},
        "source002_volume_npz": {"path": str(SOURCE002_NPZ), "sha256": sha(SOURCE002_NPZ)},
        "immutable_parent330_packet": {"path": str(PARENT330_PACKET), "sha256": sha(PARENT330_PACKET)},
        "514_source330_to_source002_cpu_check_source": {"path": str(PULLBACK514), "sha256": sha(PULLBACK514)},
        "514_cpu_chain_launch": {"path": str(PULLBACK514_MANIFEST), "sha256": sha(PULLBACK514_MANIFEST)},
        "514_cpu_chain_output": {"path": str(PULLBACK514_STDOUT), "sha256": sha(PULLBACK514_STDOUT)},
        "514_source330_to_source002_output_line": PULLBACK514_STDOUT.read_text().strip(),
        "514_source330_to_source002_source_max_m": 0.000351375,
    },
    "array_shapes": {
        "current_points": list(current_points_f64.shape), "current_tets": list(current_tets.shape),
        "source002_points": list(source_points.shape), "source002_tets": list(source_tets.shape),
    },
    "float32_native_node_parity": {
        "passed": True,
        "meaning": "current NPZ reference points cast to F32 exactly equal current mesh.node point coordinates",
    },
    "node_ancestry": {
        "map_rows": int(row_multiplicity.sum()), "unique_nodes": int(len(row_multiplicity)),
        "duplicate_row_multiplicity": {str(int(k)): int(v) for k, v in sorted(collections.Counter(row_multiplicity.tolist()).items())},
        "duplicate_ancestry_conflicts": duplicate_conflicts,
        "parent_registration_kind_counts": dict(kind_counts),
        "parent_tet_rows_used": int(len(np.unique(parent_tet_for_node))),
        "parent_bary_sum_error_max": float(np.max(np.abs(parent_bary_for_node.sum(axis=1) - 1.0))),
        "parent_bary_min": float(np.min(parent_bary_for_node)),
        "current_to_source002_point_residual_m": stats(source_map_residual),
    },
    "labels_vs_declared_parent": label_mismatch,
    "oriented_determinants": {
        "source002_parent_det6": {
            "count": int(len(source_parent_det6)), "negative": int(np.count_nonzero(source_parent_det6 < 0.0)),
            "zero": int(np.count_nonzero(source_parent_det6 == 0.0)), "min": float(source_parent_det6.min()),
            "max": float(source_parent_det6.max()),
        },
        "current_433_det6": {
            "count": int(len(current_det6)), "negative": int(np.count_nonzero(current_det6 < 0.0)),
            "zero": int(np.count_nonzero(current_det6 == 0.0)), "min": float(current_det6.min()),
            "max": float(current_det6.max()),
        },
        "source002_preimage_det6": {
            "count": int(len(preimage_det6)), "negative": int(np.count_nonzero(preimage_det6 < 0.0)),
            "zero": int(np.count_nonzero(preimage_det6 == 0.0)), "nonpositive": int(np.count_nonzero(preimage_det6 <= 0.0)),
            "min": float(preimage_det6.min()), "max": float(preimage_det6.max()),
            "negative_witnesses": witnesses(preimage_det6 < 0.0),
            "zero_witnesses": witnesses(preimage_det6 == 0.0),
        },
        "preimage_over_current_jacobian": {
            "valid_current_det_count": int(np.count_nonzero(safe_current)),
            "nonfinite_or_undefined_count": int(np.count_nonzero(~np.isfinite(jacobian))),
            "negative_count": int(np.count_nonzero(jacobian[safe_current] < 0.0)),
            "zero_count": int(np.count_nonzero(jacobian[safe_current] == 0.0)),
            "stats_positive_current_only": stats(jacobian[safe_current]),
        },
    },
    "volume_by_material_m3": {
        "source002_parent": per_material_summary(source_mat, source_parent_volume),
        "current_433": per_material_summary(current_mat, current_volume),
        "source002_preimage": per_material_summary(current_mat, preimage_volume),
    },
    "single_parent_refinement_replay": {
        "tolerance_barycentric": TOL,
        "child_cells": int(len(current_tets)),
        "cells_with_any_node_parent_tet_different_from_declared_child_parent": int(np.count_nonzero(mismatch_cells)),
        "mismatch_node_incidences": mismatch_incidence_count,
        "mismatch_cells_with_vertex_on_declared_parent_simplex": int(np.count_nonzero(mismatch_cells & mismatch_on_parent_simplex)),
        "mismatch_cells_with_no_vertex_on_declared_parent_simplex": int(np.count_nonzero(mismatch_interior)),
        "cells_with_preimage_vertex_outside_declared_parent_tet": int(np.count_nonzero(outside_cell_parent)),
        "reexpressed_parent_bary_min": float(cell_parent_min.min()),
        "reexpressed_parent_bary_max": float(cell_parent_max.max()),
        "outside_parent_witnesses": witnesses(outside_cell_parent),
    },
    "parent_volume_coverage": parent_coverage,
    "interpretation": "Diagnostic only. Positive preimage determinants and parent containment would establish a source-mapped refinement candidate; they do not by themselves establish surface, exact source-registration, mass transport, native or physical admission.",
    "elapsed_s": time.monotonic() - started,
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": report["status"], "report": str(OUT), "report_sha256": sha(OUT),
    "candidate_tets": len(current_tets), "preimage_negative": int(np.count_nonzero(preimage_det6 < 0.0)),
    "preimage_zero": int(np.count_nonzero(preimage_det6 == 0.0)),
    "outside_parent_cells": int(np.count_nonzero(outside_cell_parent)),
    "mismatch_cells": int(np.count_nonzero(mismatch_cells)),
    "parent_coverage_max_abs_delta_m3": parent_coverage["max_abs_parent_volume_delta_m3"],
    "elapsed_s": report["elapsed_s"],
}, sort_keys=True), flush=True)
