"""Screen bounded, collar-anchored registration on the dominant muscle bank."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.with_name("resting-clearance-and-atrium-007")
TOOLS = HERE.with_name("resting-clearance-and-atrium-006")
sys.path[:0] = [str(TOOLS), str(PREVIOUS)]

from fit_edge_dependency_repair import capture_in_memory
from fit_skin_tissue_depth import nearest_faces
from register_tissue_bellies import (
    MUSCLE, READER, SKIN, basis, blend, clear, exact, exact_skin_pair,
    install_for_float32, quality, source, terms,
)

BANK = (MUSCLE, 138)


def muscle_matching_faces(points, triangles, indices, directions):
    """Only original positive witnesses drive the registration objective."""
    vertices = np.unique(indices)
    tree = cKDTree(triangles.mean(axis=1))
    _, _, normal, gap, confident = nearest_faces(points[vertices], tree, triangles)
    return (
        np.einsum("ri,rik->rk", normal[confident], directions[vertices[confident]]),
        gap[confident] + .035,
    )


def fit(matrix, rhs, smoothing):
    def objective(coefficients):
        violation = np.maximum(matrix @ coefficients + rhs, 0.)
        curvature = smoothing @ coefficients
        value = (violation @ violation + .5 * (coefficients @ coefficients)
                 + 2. * (curvature @ curvature))
        gradient = 2. * matrix.T @ violation + coefficients + 4. * smoothing.T @ curvature
        return value, gradient

    result = minimize(objective, np.zeros(matrix.shape[1]), jac=True,
                      method="L-BFGS-B", bounds=[(-8., 8.)] * matrix.shape[1],
                      options={"maxiter": 120, "ftol": 1.e-11})
    assert np.isfinite(result.x).all()
    return result.x


def neighbors(surfaces, low, high):
    for key, (points, faces) in surfaces.items():
        if key in (BANK, SKIN) or (points.max(axis=0) < low).any() or (points.min(axis=0) > high).any():
            continue
        xyz = points[faces]
        relevant = ((xyz.max(axis=1) >= low) & (xyz.min(axis=1) <= high)).all(axis=1)
        if np.any(relevant):
            yield key, clear._exact_surface_records(points, faces[np.flatnonzero(relevant)])


def intersections(points, faces, surfaces, lo, hi):
    own_records = clear._exact_surface_records(points, faces)
    own_aabb = exact._prepare_surface_aabb(own_records)
    reports = {
        key: set(map(tuple, exact._audit_pair_prepared_first(own_aabb, other)["triangle_pairs"]))
        for key, other in neighbors(surfaces, lo, hi)
    }
    reports[BANK] = set(map(tuple, exact._audit_pair_prepared_first(
        own_aabb, own_records, same_surface=True)["triangle_pairs"]))
    return reports


def main():
    native = PREVIOUS / "recomputed-support-native"
    launch = json.loads((native / "launch.json").read_text())
    tissue_path = Path(launch["argv"][launch["argv"].index("--soft-tissue-payload") + 1])
    with tissue_path.open("rb") as stream:
        tissue_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert tissue_hash == launch["pins"][str(tissue_path)]
    muscle = source(tissue_path)[BANK[1]]
    local = muscle["points"]
    directions, aspect, pinned = basis(local)
    faces = muscle["source_faces"]
    install_for_float32()

    frame = native / "native/accepted-geometry/step-8000.mrvpack"
    report = json.loads((native / "step-8000-skin-exact.json").read_text())
    expected = next(r for r in report["surfaces"] if (r["semantic"], r["stable_id"]) == BANK)
    expected_pairs = set(map(tuple, expected["triangle_pairs"]))
    with capture_in_memory(frame) as (data, bodies, receipt):
        assert receipt["pack_file_sha256"] == report["accepted_pack_sha256"]
        surfaces = {key: (points, tri) for key, _, points, tri in READER["primitives"](data, bodies)}
        predicted, jacobians = blend(muscle, bodies)
    points, mesh = surfaces[BANK]
    assert np.array_equal(mesh, faces)
    pose_error = float(np.linalg.norm(predicted - points, axis=1).max())
    assert pose_error < .00012, pose_error
    skin, skin_faces = surfaces[SKIN]
    skin_records = clear._exact_surface_records(skin, skin_faces)
    skin_tree = exact._prepare_surface_aabb(skin_records)
    skin_float = clear._float32_skin_aabb_index(skin_records)
    assert exact_skin_pair(skin_tree, skin_float, points, faces) == expected_pairs
    pairs = np.asarray(sorted(expected_pairs))
    world_directions = np.einsum("vij,vjk->vik", jacobians, directions)

    signed_a, signed_d = muscle_matching_faces(
        1000. * points, 1000. * skin[skin_faces], faces[pairs[:, 1]], world_directions)
    pair_a, pair_d = terms((skin, skin_faces), points, faces, world_directions, pairs)
    smooth = np.kron(np.eye(2), np.diff(np.eye(directions.shape[-1] // 2), n=2, axis=0))
    proposals = {
        "unique_signed": fit(signed_a, signed_d, smooth),
        "signed_and_triangle": fit(np.concatenate((signed_a, .2 * pair_a)),
                                  np.concatenate((signed_d, .2 * pair_d)), smooth),
    }
    margin = .015
    lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
    original = intersections(points, faces, surfaces, lo, hi)
    results = []
    for name, coefficients in proposals.items():
        local_change = directions @ coefficients * .001
        world_change = world_directions @ coefficients * .001
        for scale in (1., .5, .25):
            shift = scale * local_change
            orientation, volume_error = quality(local, (local + shift).astype("<f4"), faces)
            row = dict(fit=name, scale=scale, source_minimum_area_ratio=orientation,
                       source_relative_signed_volume_change=volume_error,
                       max_shift_mm=float(np.linalg.norm(shift, axis=1).max()) * 1000.)
            if orientation >= .85 and volume_error <= .005:
                trial = (points + scale * world_change).astype("<f4")
                hits = exact_skin_pair(skin_tree, skin_float, trial, faces)
                row.update(after_skin_pairs=len(hits), new_skin_pairs=len(hits-expected_pairs))
                if len(hits) < len(expected_pairs):
                    other = intersections(trial, faces, surfaces, lo, hi)
                    row["introduced_neighbor_or_self_pairs"] = {
                        str(key): len(hits - original.get(key, set()))
                        for key, hits in other.items() if hits - original.get(key, set())
                    }
            results.append(row)
            print(json.dumps(row), flush=True)
    output = dict(bank=list(BANK), source_sha256=tissue_hash, native_sha256=receipt["pack_file_sha256"],
                  exact_skin_pairs=len(expected_pairs), posed_max_error_m=pose_error,
                  pinned_source_vertices=pinned, aspect=aspect,
                  native_neighbor_pair_counts={str(k): len(v) for k, v in original.items() if v},
                  results=results, promoted=False,
                  scope="One retained native pose. Source signed surface-volume proxy. Existing crossings and neighbors use exact Float32 predicates.")
    (HERE / "dominant-muscle-screen.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
