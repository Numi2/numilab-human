from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import BFGS, LinearConstraint, NonlinearConstraint, minimize
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import norm as sparse_norm


@dataclass
class LocalProblem:
    base_mm: np.ndarray
    faces: np.ndarray
    patch_vertices: np.ndarray
    free_vertices: np.ndarray
    fixed_delta_mm: dict[int, np.ndarray]
    fixed_collar_by_side: dict[str, list[int]]
    radial_rows: list[tuple[dict[int, float], np.ndarray, float, str]]
    area_faces: np.ndarray
    H: csr_matrix
    linear_constraints: LinearConstraint | None
    nonlinear_radial_boundary_constraint: NonlinearConstraint | None
    nonlinear_area_constraint: NonlinearConstraint
    initial_x: np.ndarray
    plan: dict[str, Any]
    free_index: dict[int, int]
    edge_pairs: np.ndarray
    edge_weights: np.ndarray
    smoothness_lambda: float


def _edge_adjacency(vertex_count: int, faces: np.ndarray) -> list[set[int]]:
    adjacency = [set() for _ in range(vertex_count)]
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, tri)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def ray_outer_radius_mm(point_mm: np.ndarray, center_mm: np.ndarray, triangles_mm: np.ndarray) -> float | None:
    point = np.asarray(point_mm, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64)
    triangles = np.asarray(triangles_mm, dtype=np.float64)
    direction = point - center
    radius = float(np.linalg.norm(direction))
    if radius <= 1.0e-12 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        return None
    direction /= radius
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1.0e-9
    inv = np.zeros_like(det)
    inv[valid] = 1.0 / det[valid]
    s = center[None, :] - v0
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", edge2, q)
    hit = valid & (u >= -1.0e-8) & (v >= -1.0e-8) & (u + v <= 1.0 + 1.0e-8) & (t > 1.0e-9)
    if not np.any(hit):
        return None
    return float(np.max(t[hit]))


def radial_clearance_and_gradient_mm(
    point_mm: np.ndarray, center_mm: np.ndarray, triangles_mm: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return exact radial shell clearance and its active-facet local gradient."""
    point = np.asarray(point_mm, dtype=np.float64)
    center = np.asarray(center_mm, dtype=np.float64)
    triangles = np.asarray(triangles_mm, dtype=np.float64)
    radial = point - center
    radius = float(np.linalg.norm(radial))
    if radius <= 1.0e-12 or triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("radial clearance needs a nonzero ray and triangle shell")
    direction = radial / radius
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    normals = np.cross(edge1, edge2)
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1.0e-9
    inv = np.zeros_like(det)
    inv[valid] = 1.0 / det[valid]
    offset = center[None, :] - v0
    u = inv * np.einsum("ij,ij->i", offset, h)
    q = np.cross(offset, edge1)
    v = inv * (q @ direction)
    t = inv * np.einsum("ij,ij->i", edge2, q)
    hit = valid & (u >= -1.0e-8) & (v >= -1.0e-8) & (u + v <= 1.0 + 1.0e-8) & (t > 1.0e-9)
    if not np.any(hit):
        raise ValueError("radial shell ray has no positive triangle hit")
    winner = int(np.argmax(np.where(hit, t, -np.inf)))
    normal = normals[winner]
    numerator = float(np.dot(normal, v0[winner] - center))
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1.0e-12:
        raise ValueError("active radial shell facet is tangent to the ray")
    outer_radius = float(t[winner])
    grad_outer = -numerator / (denominator * denominator * radius) * (
        normal - denominator * direction
    )
    return radius - outer_radius, direction - grad_outer


def _target_ring_deltas(rings: list[dict[str, Any]]) -> tuple[dict[int, np.ndarray], dict[str, list[int]]]:
    fixed: dict[int, np.ndarray] = {}
    boundary_by_side: dict[str, list[int]] = {}
    for ring in rings:
        side = str(ring["name"])
        ids = list(map(int, ring["vertex_ids"]))
        center = np.asarray(ring["center_world_m"], dtype=np.float64) * 1000.0
        distances = np.asarray(ring["radial_displacement_m"], dtype=np.float64) * 1000.0
        if len(ids) != len(distances):
            raise ValueError(f"{side} ring target shape mismatch")
        movable = set(map(int, ring.get("movable_vertex_ids", [])))
        if not movable.issubset(set(ids)):
            raise ValueError(f"{side} movable boundary IDs are outside the boundary ring")
        boundary_by_side[side] = ids
        for vertex, distance in zip(ids, distances):
            if int(vertex) in movable:
                continue
            point = np.asarray(ring["_captured_world_m"][str(vertex)], dtype=np.float64) * 1000.0
            direction = point - center
            length = float(np.linalg.norm(direction))
            if length <= 1.0e-9:
                raise ValueError(f"{side} ring vertex {vertex} has zero radial direction")
            delta = direction / length * float(distance)
            prior = fixed.get(vertex)
            if prior is not None and not np.allclose(prior, delta, atol=1.0e-9, rtol=0.0):
                raise ValueError(f"conflicting boundary targets for skin vertex {vertex}")
            fixed[vertex] = delta
    return fixed, boundary_by_side


def _make_area_callbacks(
    base_mm: np.ndarray,
    faces: np.ndarray,
    area_faces: np.ndarray,
    free_index: dict[int, int],
    fixed_delta_mm: dict[int, np.ndarray],
):
    face_vertices = faces[area_faces].astype(np.int64)
    base_tri = base_mm[face_vertices]
    base_cross = np.cross(base_tri[:, 1] - base_tri[:, 0], base_tri[:, 2] - base_tri[:, 0])
    base_norm = np.linalg.norm(base_cross, axis=1)
    if np.any(base_norm <= 1.0e-14):
        raise ValueError("original skin has zero-area face in constrained patch")
    base_normal = base_cross / base_norm[:, None]
    face_local = np.full(face_vertices.shape, -1, dtype=np.int64)
    for row in range(len(face_vertices)):
        for col in range(3):
            face_local[row, col] = free_index.get(int(face_vertices[row, col]), -1)
    fixed = np.zeros_like(base_mm)
    for vertex, value in fixed_delta_mm.items():
        fixed[int(vertex)] = np.asarray(value, dtype=np.float64)
    const_tri = base_tri + fixed[face_vertices]

    def displacement(x):
        d = fixed[face_vertices].copy()
        for row in range(len(face_vertices)):
            for col in range(3):
                idx = int(face_local[row, col])
                if idx >= 0:
                    d[row, col] += x[3 * idx:3 * idx + 3]
        return d

    def fun(x):
        tri = base_tri + displacement(x)
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        return np.einsum("ij,ij->i", np.cross(a, b), base_normal) / base_norm

    def jac(x):
        tri = base_tri + displacement(x)
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        g0 = -(np.cross(b, base_normal) + np.cross(base_normal, a)) / base_norm[:, None]
        g1 = np.cross(b, base_normal) / base_norm[:, None]
        g2 = np.cross(base_normal, a) / base_norm[:, None]
        rows, cols, data = [], [], []
        grads = (g0, g1, g2)
        for row in range(len(face_vertices)):
            for corner in range(3):
                idx = int(face_local[row, corner])
                if idx < 0:
                    continue
                for axis in range(3):
                    value = float(grads[corner][row, axis])
                    if value:
                        rows.append(row)
                        cols.append(3 * idx + axis)
                        data.append(value)
        return coo_matrix((data, (rows, cols)), shape=(len(face_vertices), 3 * len(free_index))).tocsr()

    return fun, jac


def build_local_problem(
    *,
    base_world_m: np.ndarray,
    faces: np.ndarray,
    seed_faces_by_side: dict[str, set[int]],
    rings: list[dict[str, Any]],
    sample_faces_by_side: dict[str, set[int]] | None = None,
    centers_mm: dict[str, np.ndarray],
    outer_shell_triangles_mm: dict[str, np.ndarray],
    collar_hops: int = 3,
    maximum_hops: int = 12,
    smoothness_lambda: float = 1.0,
    area_floor_ratio: float = 0.05,
    margin_mm: float = 0.25,
) -> LocalProblem:
    base = np.asarray(base_world_m, dtype=np.float64) * 1000.0
    tri = np.asarray(faces, dtype=np.int64)
    adjacency = _edge_adjacency(len(base), tri)
    boundary_fixed, boundary_by_side = _target_ring_deltas(rings)
    ring_captured = {
        int(v): np.asarray(ring["_captured_world_m"][str(v)], dtype=np.float64) * 1000.0
        for ring in rings for v in ring["vertex_ids"]
    }
    side_data = {}
    all_patch: set[int] = set()
    collar_by_side: dict[str, list[int]] = {}
    radial_all: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    expansions: dict[str, list[int]] = {}
    for side in ("right", "left"):
        seed_faces = {int(f) for f in seed_faces_by_side.get(side, set())}
        if not seed_faces:
            raise ValueError(f"{side} has no exact witness seed faces")
        if min(seed_faces) < 0 or max(seed_faces) >= len(tri):
            raise ValueError(f"{side} seed face outside source topology")
        seeds = set(map(int, tri[sorted(seed_faces)].ravel()))
        seeds.update(boundary_by_side[side])
        layers = [seeds]
        seen = set(seeds)
        clearances: dict[int, tuple[np.ndarray, float]] = {}
        def clearance(vertex: int):
            if vertex in clearances:
                return clearances[vertex]
            p = base[vertex]
            center = np.asarray(centers_mm[side], dtype=np.float64)
            radial = p - center
            radius = float(np.linalg.norm(radial))
            direction = radial / radius if radius > 1.0e-12 else np.zeros(3)
            outer = ray_outer_radius_mm(p, center, outer_shell_triangles_mm[side])
            gap = float("inf") if outer is None else radius - outer
            clearances[vertex] = (direction, gap)
            radial_all[(side, vertex)] = (direction, gap)
            return clearances[vertex]
        collar = None
        for hop in range(1, maximum_hops + 1):
            frontier = set()
            for vertex in layers[-1]:
                frontier.update(adjacency[vertex])
            frontier -= seen
            if not frontier:
                break
            layers.append(frontier)
            seen.update(frontier)
            ok = True
            for vertex in frontier:
                _, gap = clearance(vertex)
                if np.isfinite(gap) and gap < margin_mm - 1.0e-6:
                    ok = False
            if hop >= collar_hops and ok:
                collar = frontier
                break
        if collar is None:
            raise ValueError(f"{side} did not reach an outer-shell-clearance zero collar within {maximum_hops} edge hops")
        side_patch = set().union(*layers)
        for vertex in side_patch:
            clearance(int(vertex))
        collar_by_side[side] = sorted(collar)
        expansions[side] = [len(layer) for layer in layers]
        side_data[side] = {
            "patch": side_patch,
            "seed_faces": sorted(seed_faces),
            "seed_vertex_count": len(seeds),
            "layers": layers,
            "collar": collar,
            "clearance": clearances,
        }
        all_patch |= side_patch
    fixed = dict(boundary_fixed)
    active_union = set().union(*(side_data[side]["patch"] - side_data[side]["collar"] for side in side_data))
    collar_union = set().union(*(set(collar) for collar in collar_by_side.values()))
    all_boundary = set(v for values in boundary_by_side.values() for v in values)
    # A vertex that is active in either eye patch remains optimizable even when
    # it is an outer collar vertex for the other patch.
    for vertex in sorted(collar_union - active_union):
        if vertex in all_boundary:
            continue
        fixed[vertex] = np.zeros(3, dtype=np.float64)
    # Radial point constraints cover both vertices and face-interior samples.
    # Exact binary32 triangle-pair auditing remains the acceptance authority;
    # these samples only guide the local minimum-change solve.
    radial_rows: list[tuple[dict[int, float], np.ndarray, float, str]] = []
    vertex_constraint_count = 0
    movable_by_side = {
        str(ring["name"]): set(map(int, ring.get("movable_vertex_ids", [])))
        for ring in rings
    }
    anchor_by_side = {
        side: set(boundary_by_side[side]) - movable_by_side.get(side, set())
        for side in ("right", "left")
    }
    for vertex in sorted(all_patch):
        for side in ("right", "left"):
            if vertex not in side_data[side]["patch"]:
                continue
            direction, gap = radial_all[(side, vertex)]
            if not np.isfinite(gap) or gap >= margin_mm - 1.0e-6 or vertex in anchor_by_side[side]:
                continue
            if vertex in fixed:
                projected = float(np.dot(direction, fixed[vertex]))
                if gap + projected < margin_mm - 0.05:
                    raise ValueError(f"fixed skin vertex {vertex} violates {side} outer-shell radial margin")
                continue
            radial_rows.append(({int(vertex): 1.0}, direction, margin_mm - gap, "vertex"))
            vertex_constraint_count += 1
    barycentric_samples = (
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    )
    face_sample_constraint_count = 0
    sample_scope = sample_faces_by_side or {
        side: set(side_data[side]["seed_faces"]) for side in ("right", "left")
    }
    for side in ("right", "left"):
        center = np.asarray(centers_mm[side], dtype=np.float64)
        for face_id in sorted(map(int, sample_scope.get(side, set()))):
            face_vertices = tri[int(face_id)]
            for bary in barycentric_samples:
                bary = np.asarray(bary, dtype=np.float64)
                point = bary @ base[face_vertices]
                radial = point - center
                radius = float(np.linalg.norm(radial))
                if radius <= 1.0e-12:
                    continue
                direction = radial / radius
                outer = ray_outer_radius_mm(point, center, outer_shell_triangles_mm[side])
                if outer is None:
                    continue
                gap = radius - outer
                if gap >= margin_mm - 1.0e-6:
                    continue
                fixed_projection = 0.0
                free_coefficients: dict[int, float] = {}
                for vertex, weight in zip(face_vertices, bary):
                    vertex = int(vertex)
                    fixed_projection += float(weight) * float(np.dot(direction, fixed.get(vertex, np.zeros(3))))
                    if vertex in all_patch and vertex not in fixed and weight > 0.0:
                        free_coefficients[vertex] = free_coefficients.get(vertex, 0.0) + float(weight)
                need = margin_mm - gap - fixed_projection
                if need <= 1.0e-8:
                    continue
                if not free_coefficients:
                    statuses = [
                        (int(vertex), int(int(vertex) in all_patch), int(int(vertex) in fixed),
                         np.asarray(fixed.get(int(vertex), np.zeros(3)), dtype=np.float64).tolist())
                        for vertex in face_vertices
                    ]
                    raise ValueError(
                        f"fixed witness-face sample on {side} face {face_id} cannot reach radial margin; "
                        f"bary={bary.tolist()} gap_mm={gap:.9g} fixed_projection_mm={fixed_projection:.9g} "
                        f"need_mm={need:.9g} vertex_statuses={statuses}"
                    )
                radial_rows.append((free_coefficients, direction, need, "face_sample"))
                face_sample_constraint_count += 1

    patch_vertices = np.array(sorted(all_patch), dtype=np.int64)
    free_vertices = np.array(sorted(all_patch - set(fixed)), dtype=np.int64)
    free_index = {int(v): i for i, v in enumerate(free_vertices)}
    patch_set = set(map(int, patch_vertices))
    touched_face_mask = np.any(np.isin(tri, patch_vertices), axis=1)
    area_faces = np.flatnonzero(touched_face_mask)
    movable_boundary_rows = [
        (side, int(vertex))
        for side in ("right", "left")
        for vertex in sorted(movable_by_side.get(side, set()))
    ]
    for side, vertex in movable_boundary_rows:
        if vertex not in free_index:
            raise ValueError(f"movable {side} boundary vertex {vertex} is not an optimization variable")

    def radial_boundary_fun(x):
        values = np.empty(len(movable_boundary_rows), dtype=np.float64)
        for row, (side, vertex) in enumerate(movable_boundary_rows):
            idx = free_index[vertex]
            point = base[vertex] + np.asarray(x[3*idx:3*idx+3], dtype=np.float64)
            values[row] = radial_clearance_and_gradient_mm(
                point, np.asarray(centers_mm[side], dtype=np.float64),
                np.asarray(outer_shell_triangles_mm[side], dtype=np.float64),
            )[0]
        return values

    def radial_boundary_jac(x):
        rows, cols, data = [], [], []
        for row, (side, vertex) in enumerate(movable_boundary_rows):
            idx = free_index[vertex]
            point = base[vertex] + np.asarray(x[3*idx:3*idx+3], dtype=np.float64)
            _gap, gradient = radial_clearance_and_gradient_mm(
                point, np.asarray(centers_mm[side], dtype=np.float64),
                np.asarray(outer_shell_triangles_mm[side], dtype=np.float64),
            )
            for axis, value in enumerate(gradient):
                if value:
                    rows.append(row); cols.append(3 * idx + axis); data.append(float(value))
        return coo_matrix(
            (data, (rows, cols)), shape=(len(movable_boundary_rows), nvar),
        ).tocsr()

    radial_boundary_constraint = None
    if movable_boundary_rows:
        radial_boundary_constraint = NonlinearConstraint(
            radial_boundary_fun, float(margin_mm), np.inf,
            jac=radial_boundary_jac, hess=BFGS(), keep_feasible=False,
        )

    # The objective is squared displacement (mm^2) plus dimensionless
    # edge-gradient energy, normalized by the median positive patch-edge length.
    edges = set()
    lengths = []
    for face in tri[area_faces]:
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = (a, b) if a < b else (b, a)
            if key in edges:
                continue
            edges.add(key)
            length = float(np.linalg.norm(base[a] - base[b]))
            if length > 1.0e-9:
                lengths.append(length)
    median_edge = float(np.median(lengths))
    nvar = 3 * len(free_vertices)
    rows, cols, data = [], [], []
    for i in range(nvar):
        rows.append(i); cols.append(i); data.append(1.0)
    # Add lambda * sum w_uv ||d_u-d_v||^2, with w=(median_edge/edge_length)^2.
    for a, b in sorted(edges):
        ia, ib = free_index.get(a), free_index.get(b)
        if ia is None and ib is None:
            continue
        length = float(np.linalg.norm(base[a] - base[b]))
        weight = float(smoothness_lambda) * (median_edge / length) ** 2
        if ia is not None:
            for axis in range(3):
                rows.append(3 * ia + axis); cols.append(3 * ia + axis); data.append(weight)
        if ib is not None:
            for axis in range(3):
                rows.append(3 * ib + axis); cols.append(3 * ib + axis); data.append(weight)
        if ia is not None and ib is not None:
            for axis in range(3):
                rows.extend((3 * ia + axis, 3 * ib + axis))
                cols.extend((3 * ib + axis, 3 * ia + axis))
                data.extend((-weight, -weight))
    H_allfree = coo_matrix((data, (rows, cols)), shape=(nvar, nvar)).tocsr()
    edge_pairs = np.asarray(sorted(edges), dtype=np.int64)
    edge_weights = np.asarray([
        float(smoothness_lambda) * (median_edge / np.linalg.norm(base[a] - base[b])) ** 2
        for a, b in edge_pairs
    ], dtype=np.float64)
    # Fixed nonzero ring targets contribute a linear term to the reduced objective.
    gradient_const = np.zeros(nvar, dtype=np.float64)
    for a, b in sorted(edges):
        ia, ib = free_index.get(a), free_index.get(b)
        da = fixed.get(a, np.zeros(3))
        db = fixed.get(b, np.zeros(3))
        if ia is not None and ib is None:
            weight = float(smoothness_lambda) * (median_edge / np.linalg.norm(base[a] - base[b])) ** 2
            gradient_const[3*ia:3*ia+3] -= weight * db
        elif ib is not None and ia is None:
            weight = float(smoothness_lambda) * (median_edge / np.linalg.norm(base[a] - base[b])) ** 2
            gradient_const[3*ib:3*ib+3] -= weight * da
    fixed_vertices = set(fixed)
    # Remove variables with any fixed constraint; collar and aperture arcs are exact.
    area_fun, area_jac = _make_area_callbacks(base, tri, area_faces, free_index, fixed)
    area_constraint = NonlinearConstraint(
        area_fun, float(area_floor_ratio), np.inf,
        jac=area_jac, hess=BFGS(), keep_feasible=False,
    )
    if radial_rows:
        ar, ac, av, lower = [], [], [], []
        for row, (coefficients, direction, need, _kind) in enumerate(radial_rows):
            for vertex, bary_weight in coefficients.items():
                idx = free_index[vertex]
                for axis in range(3):
                    value = float(bary_weight) * float(direction[axis])
                    if value:
                        ar.append(row); ac.append(3 * idx + axis); av.append(value)
            lower.append(float(need))
        A = coo_matrix((av, (ar, ac)), shape=(len(radial_rows), nvar)).tocsr()
        linear_constraint = LinearConstraint(A, np.asarray(lower), np.full(len(lower), np.inf))
    else:
        linear_constraint = None
    # Report and optimize the local problem. Start from zero free displacement;
    # outside end-arc anchors are fixed, while inward-ring vertices are free
    # under the same +margin radial lower bound as the local support.
    plan = {
        "side_seed_face_counts": {s: len(side_data[s]["seed_faces"]) for s in side_data},
        "side_seed_vertex_counts": {s: side_data[s]["seed_vertex_count"] for s in side_data},
        "dilation_layer_vertex_counts": expansions,
        "collar_hops": collar_hops,
        "patch_vertex_count": int(len(patch_vertices)),
        "free_vertex_count": int(len(free_vertices)),
        "fixed_vertex_count": int(len(fixed_vertices)),
        "free_component_count": int(nvar),
        "radial_clearance_inequality_count": int(len(radial_rows)),
        "radial_vertex_inequality_count": int(vertex_constraint_count),
        "radial_face_sample_inequality_count": int(face_sample_constraint_count),
        "radial_sample_face_counts": {side: len(set(sample_scope.get(side, set()))) for side in ("right", "left")},
        "face_sample_barycentric_locations": [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5],[1.0/3.0,1.0/3.0,1.0/3.0]],
        "area_constraint_face_count": int(len(area_faces)),
        "area_constraints_with_any_free_vertex": int(sum(
            any(int(v) in free_index for v in tri[f]) for f in area_faces
        )),
        "unique_local_edge_count": int(len(edges)),
        "median_edge_mm": median_edge,
        "objective": {
            "formula": "0.5*sum_v ||d_v||^2 + lambda/2*sum_edges w_uv||d_u-d_v||^2",
            "units": "mm^2; lambda and edge weights dimensionless",
            "smoothness_lambda": float(smoothness_lambda),
            "edge_weight": "w_uv=(median_positive_patch_edge_length/edge_length)^2",
        },
        "constraints": {
            "boundary_targets": "inside-ring vertices are free and constrained by the updated point's exact outer-shell ray clearance >=+0.25 mm; fixed-ray vertex and edge/centroid sample inequalities additionally guide the sparse solve; outside end-arc anchors exactly zero",
            "actual_radial_clearance_constraint_count": int(len(movable_boundary_rows)),
            "outer_collar": "the first geodesic edge layer whose vertices all have ray clearance >=0.25 mm (or no outer-shell ray hit) is fixed to zero; no source positions outside patch change",
            "radial_inequalities": "all free patch vertices and edge-midpoint/centroid samples of exact witness faces with base radial clearance <0.25 mm must satisfy outward fixed-ray projection to at least +0.25 mm; exact triangle audit remains final authority",
            "oriented_area": f"exact quadratic signed projected area ratio >= {area_floor_ratio}; SciPy trust-constr sparse local NLP",
        },
        "area_floor_ratio": float(area_floor_ratio),
        "margin_mm": float(margin_mm),
        "median_edge_length_mm": median_edge,
        "hessian_shape": list(H_allfree.shape),
        "hessian_nnz": int(H_allfree.nnz),
        "linear_constraint_shape": None if linear_constraint is None else list(linear_constraint.A.shape),
        "area_constraint_shape": [int(len(area_faces)), int(nvar)],
        "fixed_boundary_anchor_count": int(len(boundary_fixed)),
        "movable_inward_boundary_vertex_count": int(sum(len(v) for v in movable_by_side.values())),
        "outside_anchor_count": int(sum(1 for r in rings for d in r["radial_displacement_m"] if float(d) == 0.0)),
    }
    return LocalProblem(
        base_mm=base, faces=tri, patch_vertices=patch_vertices, free_vertices=free_vertices,
        fixed_delta_mm=fixed, fixed_collar_by_side=collar_by_side, radial_rows=radial_rows,
        area_faces=area_faces, H=H_allfree, linear_constraints=linear_constraint,
        nonlinear_radial_boundary_constraint=radial_boundary_constraint,
        nonlinear_area_constraint=area_constraint, initial_x=np.zeros(nvar,dtype=np.float64),
        plan=plan, free_index=free_index, edge_pairs=edge_pairs,
        edge_weights=edge_weights, smoothness_lambda=float(smoothness_lambda),
    ), gradient_const


def solve_local_problem(problem: LocalProblem, gradient_const: np.ndarray, *, maxiter: int = 400):
    H = problem.H
    g = np.asarray(gradient_const, dtype=np.float64)
    def fun(x):
        return 0.5 * float(x @ (H @ x)) + float(g @ x)
    def jac(x):
        return H @ x + g
    constraints = [problem.nonlinear_area_constraint]
    if problem.nonlinear_radial_boundary_constraint is not None:
        constraints.append(problem.nonlinear_radial_boundary_constraint)
    if problem.linear_constraints is not None:
        constraints.append(problem.linear_constraints)
    result = minimize(
        fun, problem.initial_x, method="trust-constr", jac=jac, hess=lambda _x: H,
        constraints=constraints,
        options={
            "maxiter": int(maxiter), "gtol": 1.0e-8, "xtol": 1.0e-10,
            "barrier_tol": 1.0e-10, "sparse_jacobian": True,
            "verbose": 0,
        },
    )
    disp = {int(v): np.asarray(problem.fixed_delta_mm.get(int(v), np.zeros(3)), dtype=np.float64)
            for v in problem.patch_vertices}
    for vertex, index in problem.free_index.items():
        disp[int(vertex)] = np.asarray(result.x[3*index:3*index+3], dtype=np.float64)
    full_delta = np.zeros_like(problem.base_mm, dtype=np.float64)
    for vertex, value in disp.items():
        full_delta[vertex] = value / 1000.0
    actual_area = np.asarray(problem.nonlinear_area_constraint.fun(result.x), dtype=np.float64)
    node_energy = 0.5 * sum(float(np.dot(value, value)) for value in disp.values())
    edge_energy = 0.0
    for (a, b), weight in zip(problem.edge_pairs, problem.edge_weights):
        da = disp.get(int(a), np.zeros(3, dtype=np.float64))
        db = disp.get(int(b), np.zeros(3, dtype=np.float64))
        diff = da - db
        edge_energy += 0.5 * float(weight) * float(np.dot(diff, diff))
    true_objective = node_energy + edge_energy
    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.niter),
        "objective_value_mm2": float(true_objective),
        "reduced_free_variable_objective_mm2": float(result.fun),
        "objective_node_term_mm2": float(node_energy),
        "objective_smoothness_term_mm2": float(edge_energy),
        "optimality": float(result.optimality),
        "constraint_violation": float(result.constr_violation),
        "lagrangian_gradient_inf_norm": float(np.max(np.abs(np.asarray(getattr(result, "lagrangian_grad", jac(result.x)))))),
        "minimum_oriented_area_ratio": float(actual_area.min(initial=np.inf)),
        "area_floor_ratio": float(problem.plan["area_floor_ratio"]),
        "radial_clearance_inequality_count": len(problem.radial_rows),
        "full_delta_max_mm": float(max((np.linalg.norm(v) for v in disp.values()), default=0.0)),
    }
    if not result.success or report["constraint_violation"] > 1.0e-6 or report["minimum_oriented_area_ratio"] < problem.plan["area_floor_ratio"] - 1.0e-6:
        raise RuntimeError("constrained local eye-registration solve failed declared convergence/area gates: " + str(report))
    return full_delta, report
