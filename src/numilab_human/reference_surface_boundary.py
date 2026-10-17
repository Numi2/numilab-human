"""Offline source-face boundary emission through the existing NHTISS owner."""
from collections import defaultdict
from fractions import Fraction as Q
import numpy as np
from . import passive_attachment_composition as pc
from .cardiac_cavity_partition import encode_rational


def _require(condition, message):
    if not condition:
        raise ValueError("reference surface boundary: " + message)


def _barycentric(point, triangle):
    a, b, c = triangle
    sub = lambda u, v: tuple(x-y for x, y in zip(u, v))
    dot = lambda u, v: sum((x*y for x, y in zip(u, v)), Q(0))
    ab, ac, ap = sub(b, a), sub(c, a), sub(point, a)
    cross = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0])
    denominator = dot(ab, ab)*dot(ac, ac)-dot(ab, ac)**2
    _require(denominator > 0 and dot(ap, cross) == 0,
             "construction corner is not on a nondegenerate source face")
    v = (dot(ac, ac)*dot(ap, ab)-dot(ab, ac)*dot(ap, ac))/denominator
    w = (dot(ab, ab)*dot(ap, ac)-dot(ab, ac)*dot(ap, ab))/denominator
    result = (1-v-w, v, w)
    _require(min(result) >= 0 and max(result) <= 1,
             "construction corner is outside its source face")
    return result


def _weights(indices, weights):
    result = defaultdict(Q)
    for index, weight in zip(indices, weights):
        if weight > 0:
            result[int(index)] += Q.from_float(float(weight))
    return dict(result)


def emit_source_face_boundary(row, poses, boundary, *, inferred_covers=(),
                              maximum_cover_radius_m=Q(1, 5000)):
    """Emit the decoded existing positive-winding result with source ancestry.

    Original allocations, coefficients and body palette remain unchanged.
    Birth coefficients use positive source-face barycentric interpolation,
    averaged across incident parents without truncating lanes. Explicit inferred covers bind to an original source vertex and retain
    its attachment owner; their ancestry remains distinct from measurements.
    Float32 self/neighbor/motion and native gates remain mandatory.
    This prepares assets and never evaluates simulation steps or adds state.
    """
    positions = np.array(row["positions"], dtype="<f4", copy=True)
    faces = np.asarray(row["faces"], dtype=np.int64)
    indices = np.array(row["binding_indices"], dtype="<u4", copy=True)
    weights = np.array(row["weights"], dtype="<f4", copy=True)
    world = pc._native_blended_rest_positions(row, positions, poses)
    original_count = len(positions)
    _require(np.isfinite(world).all() and np.isfinite(weights).all()
             and np.all(weights >= 0), "nonfinite source coordinates or coefficients")
    exact = [tuple(Q.from_float(float(x)) for x in p) for p in world]
    original_exact = tuple(exact)
    construction_positions, construction_indices, construction_weights = positions, indices, weights
    cover_proofs = []
    for cover in inferred_covers:
        center = tuple(cover["exact_center"])
        _require(center in original_exact, "inferred cover lacks exact original source vertex ancestry")
        radius = Q(str(cover["coordinate_radius_m"]))
        _require(0 < radius <= maximum_cover_radius_m, "inferred cover exceeds its declared construction bound")
        center_slot = original_exact.index(center)
        start_vertex, start_face = len(exact), len(faces)
        vertices = [tuple(center[a]+radius*signs[a] for a in range(3))
                    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
        local_faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        volume6 = sum(sum(vertices[a][k]*(vertices[b][(k+1)%3]*vertices[c][(k+2)%3]
                                         - vertices[b][(k+2)%3]*vertices[c][(k+1)%3])
                          for k in range(3)) for a, b, c in local_faces)
        if volume6 < 0:
            local_faces = [(a, c, b) for a, b, c in local_faces]
        exact.extend(vertices)
        faces = np.concatenate((faces, np.asarray(local_faces, dtype=np.int64)+start_vertex))
        construction_positions = np.concatenate((construction_positions, np.tile(positions[center_slot], (4, 1))))
        construction_indices = np.concatenate((construction_indices, np.tile(indices[center_slot], (4, 1))))
        construction_weights = np.concatenate((construction_weights, np.tile(weights[center_slot], (4, 1))))
        cover_proofs.append({
            "first_augmented_source_face": start_face, "face_count": 4,
            "original_source_vertex_ancestry": center_slot,
            "exact_world_center": encode_rational(center),
            "coordinate_radius_m": float(radius), "inferred_local_reference_material": True,
            "source_measurement": False})
    mesh = boundary["output_mesh"]
    points = []
    for point in mesh["vertices_m"]:
        _require(len(point) == 3 and all(isinstance(x, (Q, int)) for x in point),
                 "construction coordinates require exact rational metres")
        points.append(tuple(Q(x) for x in point))
    output_faces = np.asarray(mesh["triangles"])
    _require(output_faces.ndim == 2 and output_faces.shape[1] == 3
             and output_faces.dtype.kind in "iu" and len(output_faces)
             and output_faces.min() >= 0 and output_faces.max() < len(points),
             "construction triangle inventory")
    lineage = boundary["output_face_ancestry"]
    _require(len(lineage) == len(output_faces), "construction face ancestry coverage")
    parents = [x["source_face"] for x in lineage]
    _require(all(type(x) is int and 0 <= x < len(faces) for x in parents),
             "construction source face identity")
    parent_triangles = [[exact[int(j)] for j in face] for face in faces]
    incident = [set() for _ in points]
    coverage = defaultdict(Q)
    for output, parent in zip(output_faces, parents):
        bary = [_barycentric(points[int(k)], parent_triangles[parent]) for k in output]
        u, v, w = bary
        ratio = ((v[1]-u[1])*(w[2]-u[2])-(v[2]-u[2])*(w[1]-u[1]))
        _require(ratio > 0, "construction face reverses source orientation or collapses")
        coverage[parent] += ratio
        _require(coverage[parent] <= 1, "construction duplicates source face area")
        for k in output:
            incident[int(k)].add(parent)
    original_slots = {}
    for slot, point in enumerate(original_exact):
        if point in original_slots:
            first = original_slots[point]
            _require(_weights(indices[first], weights[first])
                     == _weights(indices[slot], weights[slot]),
                     "coincident source slots have different attachment owners")
        else:
            original_slots[point] = slot
    slots, births, new_positions, new_indices, new_weights, targets = {}, [], [], [], [], []
    for point_index, point in enumerate(points):
        if point in original_slots:
            slots[point_index] = original_slots[point]
            continue
        source_parents = sorted(incident[point_index])
        _require(source_parents, "unreferenced construction birth")
        interpolations, guesses = [], []
        for parent in source_parents:
            lam = _barycentric(point, parent_triangles[parent])
            interpolated = defaultdict(Q)
            guess = np.zeros(3)
            for coefficient, vertex in zip(lam, faces[parent]):
                for index, weight in _weights(construction_indices[vertex], construction_weights[vertex]).items():
                    interpolated[index] += coefficient * weight
                guess += float(coefficient) * construction_positions[vertex].astype(float)
            interpolations.append(interpolated)
            guesses.append(guess)
        combined = {index: sum((x.get(index, Q(0)) for x in interpolations), Q(0))
                    / len(interpolations) for index in set().union(*interpolations)}
        combined = {k: v for k, v in combined.items() if v}
        _require(0 < len(combined) <= 4 and all(v > 0 for v in combined.values()),
                 "birth exceeds the existing four-lane owner; coefficients are never truncated")
        _require(abs(float(sum(combined.values()))-1) <= 2e-6,
                 "birth attachment coefficients do not sum to one")
        ii = np.full(4, 0xffffffff, dtype="<u4")
        ww = np.zeros(4, dtype="<f4")
        for lane, (index, weight) in enumerate(sorted(combined.items())):
            ii[lane], ww[lane] = index, float(weight)
        slot = original_count + len(new_positions)
        slots[point_index] = slot
        new_positions.append(np.mean(guesses, axis=0))
        new_indices.append(ii)
        new_weights.append(ww)
        targets.append([float(x) for x in point])
        births.append({"source_slot": slot, "parent_source_faces": source_parents,
                       "exact_world_coordinate": encode_rational(point), "positive_ancestry": True})
    emitted_faces = np.asarray([[slots[int(k)] for k in face] for face in output_faces], dtype="<u4")
    positions = np.concatenate((positions, np.asarray(new_positions, dtype="<f4").reshape(-1, 3)))
    indices = np.concatenate((indices, np.asarray(new_indices, dtype="<u4").reshape(-1, 4)))
    weights = np.concatenate((weights, np.asarray(new_weights, dtype="<f4").reshape(-1, 4)))
    emitted = row | dict(positions=positions, binding_indices=indices, weights=weights,
                        faces=emitted_faces, vertex_count=len(positions))
    new_slots = np.arange(original_count, len(positions))
    if len(new_slots):
        targets = np.asarray(targets, dtype="<f4")
        solved, _ = pc._solve_sparse_rest_positions(
            emitted, targets, positions[new_slots], poses, vertex_indices=new_slots)
        positions[new_slots] = solved
        affine, _ = pc._child_rest_affine(emitted, poses)
        for _ in range(4):
            actual = pc._native_blended_rest_positions(emitted, positions, poses)
            delta = targets.astype(float) - actual[new_slots].astype(float)
            positions[new_slots] = (positions[new_slots].astype(float)
                                   + np.linalg.solve(affine[new_slots], delta[..., None])[..., 0]).astype("<f4")
    world = pc._native_blended_rest_positions(emitted, positions, poses)
    _require(np.array_equal(positions[:original_count], row["positions"])
             and np.array_equal(indices[:original_count], row["binding_indices"])
             and np.array_equal(weights[:original_count], row["weights"]),
             "original source allocations or attachment coefficients changed")
    normals = np.concatenate((row["normals"], np.tile(
        np.asarray([0, 0, 1], dtype="<f4"), (len(new_slots), 1))))
    normals, _, normal_report = pc._recompute_rest_conforming_normals(
        emitted, normals, world, np.unique(emitted_faces), poses)
    return {
        "patch_arrays": {"vertices6": np.column_stack((positions, normals)).astype("<f4"),
                         "binding_indices": indices, "weights": weights, "faces": emitted_faces},
        "world_positions_f32": world,
        "ancestry": {"positive_birth_ancestry": births, "output_face_augmented_source_ancestry": parents,
                     "inferred_repair_material": cover_proofs,
                     "original_source_slot_count": original_count,
                     "original_source_positions_bindings_weights_byte_exact": True,
                     "normal_reconstruction": normal_report, "inferred_reference_not_measurement": True,
                     "anatomical_admission": False, "native_admission": False},
    }
