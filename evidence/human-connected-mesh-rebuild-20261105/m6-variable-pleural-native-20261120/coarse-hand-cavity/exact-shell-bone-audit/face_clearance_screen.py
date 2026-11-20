"""Screen joint skin/fat column corrections on the native crossing witnesses."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.sparse import coo_matrix

from audit import TRIANGLE_ROW
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_bed_layer_surfaces import role_surfaces
from compare_bank_trials import emitted
from fit_bank_group_partitions import ROLE_COUNT
from fit_hand_cavity import PARENT, determinant, quality, table


def quantiles(values):
    return np.quantile(values, [0., .5, .9, .99, 1.]).tolist()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = monotonic()
    points, masses = emitted(args.run)
    assert points.shape == (3 * ROLE_COUNT, 3)
    dtype_points = points.dtype
    layered = points.reshape(3, ROLE_COUNT, 3).astype(np.float64)
    directions = layered[0] - layered[2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rows = table(PARENT.with_name(PARENT.name + ".tets.csv"))
    tets = np.array([[int(row[f"parent_node{i}"]) for i in range(4)]
                     for row in rows], dtype=np.int32)
    material = np.array([int(row["material_index"]) for row in rows], dtype=np.int32)
    volumes = determinant(points.astype(np.float64), tets)
    shapes = quality(points.astype(np.float64), tets)
    assert np.min(volumes) > 0 and np.min(shapes) > 0
    topology = role_surfaces(rows)
    all_faces = np.asarray([face for bank in topology.values() for face in bank], dtype=np.int32)
    face_columns = all_faces % ROLE_COUNT
    direction_by_face = directions[face_columns].mean(axis=1)
    direction_by_face /= np.linalg.norm(direction_by_face, axis=1, keepdims=True)
    face_axis_projection = np.einsum("fd,fkd->fk", direction_by_face, points[all_faces])
    corner_scale = np.einsum("fd,fkd->fk", direction_by_face, directions[face_columns])

    proof = json.loads(args.audit.read_text())
    expected = proof["crossing_pairs"]
    data = np.memmap(args.run / "physical-bone-initial.bin", dtype=TRIANGLE_ROW, mode="r")
    identity = {(int(row["body"]), int(row["bone"]), int(row["triangle"])): index
                for index, row in enumerate(data)}
    assert len(identity) == len(data)
    crossed = np.array([r[1] for r in proof["witnesses"]], dtype=np.int32)
    obstacle = np.array([identity[tuple(r[-3:])] for r in proof["witnesses"]], dtype=np.int32)
    assert len(crossed) == len(obstacle) == expected
    obstacle_support = np.einsum("nd,nkd->nk", direction_by_face[crossed],
                                data["vertices"][obstacle]).max(axis=1)
    required_projection = np.full(len(all_faces), -np.inf)
    np.maximum.at(required_projection, crossed, obstacle_support + 0.00005)
    active_faces = np.unique(crossed)
    admissible = np.all(corner_scale[active_faces] > 0.25, axis=1)
    local = active_faces[admissible]
    minimum_shift = (required_projection[local, None] - face_axis_projection[local]) / corner_scale[local]
    targets = np.zeros(ROLE_COUNT)
    np.maximum.at(targets, face_columns[local].ravel(), minimum_shift.ravel())
    fixed_support_rows = np.flatnonzero(layered[0, :, 2] <= 1.0e-5)
    assert len(fixed_support_rows) == 3

    # Harmonic extension of the conservative lower bounds on coupled columns
    # spreads the displacement through the mesh next to verified witnesses.
    local_triangles = topology["outer"]
    neighbors = {(i, j) for a in local_triangles for i in a for j in a if i != j}
    centers, neighbors = np.transpose(sorted(neighbors))
    strengths = coo_matrix((np.ones(len(centers)), (centers, neighbors)),
                           shape=(ROLE_COUNT, ROLE_COUNT)).tocsr()
    degree = np.maximum(1., np.asarray(strengths.sum(axis=1)).ravel())
    field = np.minimum(targets, 0.003)
    field[fixed_support_rows] = 0.
    for _ in range(18):
        field = np.maximum(np.minimum(targets, 0.003), strengths @ field / degree)
        field[fixed_support_rows] = 0.
    shifted = field[:, None] * directions
    samples = []
    for scale in (0.125, 0.25, 0.5, 1.0):
        moved = (layered + scale * shifted).reshape(points.shape).astype(np.float32).astype(np.float64)
        trial_volume = determinant(moved, tets)
        trial_quality = quality(moved, tets)
        ratio = trial_quality / shapes
        joint_gap = np.einsum("fd,fkd->fk", direction_by_face[crossed], moved[all_faces[crossed]])
        plane_cleared = joint_gap.min(axis=1) > obstacle_support + 0.00001
        layer_delta = {str(k): float(np.sum((trial_volume - volumes)[material == k]) / 6.)
                       for k in np.unique(material)}
        samples.append({"scale": scale, "displaced_columns": int(np.count_nonzero(field)),
                        "maximum_displacement_m": float((scale * field).max()),
                        "removed_initial_pairs_by_sufficient_plane": int(plane_cleared.sum()),
                        "inverted_cells": int(np.count_nonzero(trial_volume <= 0)),
                        "cells_below_88_percent_baseline_shape": int(np.count_nonzero(ratio < .88)),
                        "minimum_shape_ratio": float(ratio.min()),
                        "layer_volume_delta_m3": layer_delta,
                        "maximum_bed_contact_node_motion_m": float(np.linalg.norm(
                            moved[fixed_support_rows] - points[fixed_support_rows], axis=1).max())})
    result = {
        "scope": "Conservative plane-separation and physical volume screen; proposals only, no native replay and no new pairs assessed",
        "crossed_faces": len(active_faces),
        "faces_missing_coherent_column_direction": int(np.count_nonzero(~admissible)),
        "positive_target_shift_m_quantiles": quantiles(targets[targets > 0.]),
        "columns_with_required_shift_above_three_mm": int(np.count_nonzero(targets > .003)),
        "pinned_support_rows": fixed_support_rows.tolist(),
        "model_native_shell_mass_kg": float(masses.sum()),
        "original_pose_coordinate_dtype": str(dtype_points),
        "banks_with_clearance_witnesses": Counter(row[3] for row in proof["witnesses"]),
        "candidates": samples,
        "wall_seconds": monotonic() - start,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "banks_with_clearance_witnesses"}))


if __name__ == "__main__":
    main()
