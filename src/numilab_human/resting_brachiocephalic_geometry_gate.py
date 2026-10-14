"""Captured-pose surface-intersection stage for appended brachiocephalic veins.

This wrapper uses the existing resting-anatomy exact packed-Float32 verifier.
It checks each predicted bridge against captured anatomy through the frozen
P6 exact predicates, then exposes a separate source-bound solid-continuity and
local-calibre screen. It is not clinical or physiological qualification.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

BRIDGE_KEYS = ((51011, 525), (51011, 526))
PORT_TARGET_IDS = {525: (10, 469, 473), 526: (10, 468, 472)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("brachiocephalic captured-pose geometry: " + message)


def _identity(key: object) -> tuple[int, int]:
    _require(isinstance(key, (tuple, list)) and len(key) == 2
             and all(type(value) is int for value in key),
             "surface keys must be (semantic, stable_id) integer pairs")
    semantic, stable_id = key
    _require(semantic > 0 and stable_id > 0,
             "surface key must identify positive semantic and stable IDs")
    return semantic, stable_id


def _faces(value: object, *, name: str, vertex_count: int) -> np.ndarray:
    faces = np.asarray(value)
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
             and faces.dtype.kind in "iu",
             name + " faces must be nonempty integer triangles")
    _require(int(faces.min()) >= 0 and int(faces.max()) < vertex_count,
             name + " face index is outside its vertex array")
    return np.ascontiguousarray(faces, dtype="<u4")


def _float32_xyz(value: object, *, name: str) -> np.ndarray:
    positions = np.asarray(value)
    _require(positions.dtype == np.dtype("<f4") and positions.ndim == 2
             and positions.shape[1] == 3 and len(positions) > 0
             and np.isfinite(positions).all(),
             name + " positions must be nonempty finite emitted Float32 XYZ")
    return np.ascontiguousarray(positions)


def _pair_rows(value: object, *, name: str, bridge_face_count: int,
               target_face_count: int) -> tuple[tuple[int, int], ...]:
    _require(isinstance(value, (list, tuple)) and len(value) > 0,
             name + " must declare at least one exact face pair")
    pairs = []
    for pair in value:
        _require(isinstance(pair, (list, tuple)) and len(pair) == 2
                 and all(type(index) is int and index >= 0 for index in pair),
                 name + " contains an invalid face pair")
        bridge_face, target_face = pair
        _require(bridge_face < bridge_face_count and target_face < target_face_count,
                 name + " face pair is outside current topology")
        pairs.append((bridge_face, target_face))
    _require(len(set(pairs)) == len(pairs), name + " contains duplicate face pairs")
    return tuple(sorted(pairs))


def audit_captured_pose_surface_intersections(
    verifier,
    captured_surfaces: Mapping[tuple[int, int], Sequence[object]],
    bridge_positions: Mapping[tuple[int, int], np.ndarray],
    bridge_faces: Mapping[tuple[int, int], np.ndarray],
    port_face_pairs: Mapping[tuple[tuple[int, int], tuple[int, int]], Sequence[Sequence[int]]],
    *,
    target_keys: Sequence[tuple[int, int]] | None = None,
) -> dict:
    """Audit predicted bridges against every anatomical surface decoded from a pack.

    ``captured_surfaces`` values use ``load_capture`` layout:
    ``(positions, faces, normals, source_vertex_ids)``. Each declared port
    pair is ordered ``(bridge_face_id, captured_target_face_id)`` and freezes
    the allowed source collar face patches. Pose-specific pair loci may move
    inside each declared patch cross-product; any observed pair outside it
    rejects this stage. Solid overlap and section continuity are checked by
    the separate source-geometry gate. Inputs are not reconstructed or mutated.
    """
    _require(hasattr(verifier, "audit_changed_surfaces"),
             "the frozen resting-anatomy verifier API is required")
    _require(set(bridge_positions) == set(BRIDGE_KEYS)
             and set(bridge_faces) == set(BRIDGE_KEYS),
             "both bridge identities 525 and 526 are required")
    _require(isinstance(captured_surfaces, Mapping) and bool(captured_surfaces),
             "captured anatomy surface mapping is empty")

    targets = tuple(sorted(_identity(key) for key in
                           (captured_surfaces.keys() if target_keys is None else target_keys)))
    _require(len(set(targets)) == len(targets), "duplicate captured target identity")
    _require(set(targets) == set(captured_surfaces),
             "target list must cover every decoded anatomical surface exactly")
    _require(set(BRIDGE_KEYS).isdisjoint(targets),
             "captured target identities may not shadow appended bridges")

    surfaces = dict(captured_surfaces)
    prepared_bridge = {}
    for key in BRIDGE_KEYS:
        positions = _float32_xyz(bridge_positions[key], name=f"bridge {key[1]}")
        faces = _faces(bridge_faces[key], name=f"bridge {key[1]}",
                       vertex_count=len(positions))
        prepared_bridge[key] = (positions, faces)
        surfaces[key] = (positions, faces, None,
                          np.arange(len(positions), dtype=np.uint64))

    frozen_ports = {}
    interface_allowlist = {}
    for bridge_key in BRIDGE_KEYS:
        expected = {(51011, stable_id) for stable_id in PORT_TARGET_IDS[bridge_key[1]]}
        _require(expected.issubset(set(targets)),
                 f"captured pack is missing a declared port for bridge {bridge_key[1]}")
        for target_key in sorted(expected):
            target_value = captured_surfaces[target_key]
            _require(isinstance(target_value, (tuple, list)) and len(target_value) >= 2,
                     f"captured port {target_key} has no geometry")
            target_positions = _float32_xyz(target_value[0], name=f"target {target_key[1]}")
            target_faces = _faces(target_value[1], name=f"target {target_key[1]}",
                                  vertex_count=len(target_positions))
            pair_key = (bridge_key, target_key)
            _require(pair_key in port_face_pairs,
                     f"missing frozen face-pair witness for port {pair_key}")
            frozen = _pair_rows(
                port_face_pairs[pair_key], name=f"port {pair_key}",
                bridge_face_count=len(prepared_bridge[bridge_key][1]),
                target_face_count=len(target_faces))
            frozen_ports[pair_key] = frozen
            # Frozen source pairs establish face-patch ancestry. Pose-specific
            # pair loci may move within the patch, so do not require exact
            # per-pose equality with the source pair list.
            bridge_patch = sorted({pair[0] for pair in frozen})
            target_patch = sorted({pair[1] for pair in frozen})
            interface_allowlist[pair_key] = tuple(
                (bridge_face, target_face)
                for bridge_face in bridge_patch for target_face in target_patch)

    _require(set(port_face_pairs) == set(frozen_ports),
             "port witness map must contain exactly the six declared interfaces")
    skin_targets = tuple(key for key in targets if key[0] == 51007)
    all_targets = tuple(sorted((*targets, *BRIDGE_KEYS)))
    enclosure_pairs = [(bridge_key, skin_key)
                       for bridge_key in BRIDGE_KEYS for skin_key in skin_targets]
    import inspect
    parameters = inspect.signature(verifier.audit_changed_surfaces).parameters
    if enclosure_pairs:
        _require("intended_enclosures" in parameters,
                 "central P6 verifier with role-aware intended_enclosures is required")
    audit_kwargs = {
        "targets": all_targets, "interfaces": interface_allowlist,
        "required_closed": BRIDGE_KEYS,
    }
    if "intended_enclosures" in parameters:
        audit_kwargs["intended_enclosures"] = enclosure_pairs
    result = verifier.audit_changed_surfaces(
        surfaces, changed=list(BRIDGE_KEYS), **audit_kwargs)

    observed_ports = {}
    port_unexpected = {}
    for row in result.get("rows", []):
        if row.get("stage") != "nearby_pair":
            continue
        identities = row.get("identities", [])
        if len(identities) != 2:
            continue
        pair_key = (tuple(identities[0]), tuple(identities[1]))
        if pair_key in frozen_ports:
            observed_ports[pair_key] = tuple(sorted(
                tuple(pair) for pair in row.get("declared_interface_triangle_pairs", [])))
            port_unexpected[pair_key] = tuple(sorted(
                tuple(pair) for pair in row.get("unexpected_triangle_pairs", [])))
    port_mismatches = []
    port_observations = []
    for pair_key, frozen in frozen_ports.items():
        observed = observed_ports.get(pair_key)
        unexpected = port_unexpected.get(pair_key, ())
        allowed = set(interface_allowlist[pair_key])
        outside_patch = tuple(pair for pair in (observed or ()) if pair not in allowed)
        port_observations.append({
            "bridge": list(pair_key[0]), "target": list(pair_key[1]),
            "frozen_source_face_pairs": [list(pair) for pair in frozen],
            "frozen_bridge_patch_faces": sorted({pair[0] for pair in frozen}),
            "frozen_target_patch_faces": sorted({pair[1] for pair in frozen}),
            "allowed_patch_cross_product_pair_count": len(allowed),
            "observed_pose_face_pairs": [list(pair) for pair in (observed or ())],
            "observed_exactly_matches_frozen_pairs": observed == frozen,
            "observed_pose_pairs_within_frozen_patch": not outside_patch,
        })
        if unexpected or outside_patch:
            port_mismatches.append({
                "bridge": list(pair_key[0]), "target": list(pair_key[1]),
                "frozen_source_face_pairs": [list(pair) for pair in frozen],
                "observed_pose_face_pairs": None if observed is None else
                    [list(pair) for pair in observed],
                "unexpected_actual_face_pairs_outside_frozen_patch":
                    [list(pair) for pair in sorted(set((*unexpected, *outside_patch)))],
            })

    skin_boundary_checks = []
    for row in result.get("rows", []):
        identities = row.get("identities", [])
        if row.get("stage") != "nearby_pair" or len(identities) != 2:
            continue
        bridge, container = tuple(identities[0]), tuple(identities[1])
        if bridge in BRIDGE_KEYS and container in skin_targets:
            skin_boundary_checks.append({
                "bridge": list(bridge), "skin": list(container),
                "triangle_pair_count": int(row.get("triangle_pair_count", 0)),
                "declared_intended_enclosure": [list(bridge), list(container)]
                    in row.get("intended_enclosures", []),
                "central_verifier_pair_passed": bool(row.get("passed")),
                "containment": row.get("containment", []),
            })
    surface_passed = bool(result.get("passed")) and not port_mismatches
    return {
        "schema": "numi.human.brachiocephalic-captured-pose-surface-audit.v1",
        "status": ("CAPTURED_POSE_SURFACE_INTERSECTION_STAGE_PASSED_CONTINUITY_UNRESOLVED"
                   if surface_passed else "CAPTURED_POSE_SURFACE_INTERSECTION_STAGE_REJECTED"),
        "surface_intersection_stage_passed": surface_passed,
        "candidate_bridge_keys": [list(key) for key in BRIDGE_KEYS],
        "captured_target_keys": [list(key) for key in targets],
        "captured_target_identity_sha256": hashlib.sha256(
            b"".join(np.asarray(key, dtype="<u4").tobytes() for key in targets)
        ).hexdigest(),
        "port_observations": port_observations,
        "port_pair_mismatches": port_mismatches,
        "skin_boundary_checks": skin_boundary_checks,
        "verifier_result": result,
        "limits": {
            "target_universe": "all anatomical surfaces decoded from this accepted pack; skin boundary crossings are checked while intended enclosure is allowed; not compared here to undrawn NHA rows",
            "interface_measure": "exact captured-pose triangle pairs must remain inside the frozen source collar face-patch cross-product; pair identity does not bound the intersection locus",
            "skin_enclosure": "delegated to the central P6 verifier intended_enclosures role; containment is declared and every boundary crossing remains checked",
            "solid_containment_at_contacting_ports": "not_evaluated",
            "lumen_continuity": "not_evaluated",
            "calibre_continuity": "not_evaluated",
            "clearance_margin": "not_evaluated",
            "native_common_field_confirmation": "required",
            "anatomical_admission": False,
            "all_pose_admission": False,
        },
    }


def _surface_area(positions, faces):
    triangles = np.asarray(positions, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() * 0.5)


def _signed_volume(positions, faces):
    triangles = np.asarray(positions, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", triangles[:, 0],
                           np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def _prepare_vessel(verifier, key, value):
    _require(isinstance(value, (tuple, list)) and len(value) >= 2,
             f"surface {key} must contain positions and faces")
    positions = _float32_xyz(value[0], name=f"surface {key}")
    faces = _faces(value[1], name=f"surface {key}", vertex_count=len(positions))
    normals = value[2] if len(value) > 2 else None
    source_ids = value[3] if len(value) > 3 else np.arange(len(positions), dtype=np.uint64)
    prepared = verifier.prepare_surface(
        key, positions, faces, normals=normals, source_vertex_ids=source_ids)
    _require(prepared.closed and prepared.self_audit()["count"] == 0,
             f"surface {key} must be closed and exact-self-clear")
    return prepared


def _make_manifold(prepared, manifold3d):
    positions = np.ascontiguousarray(prepared.quotient_positions, dtype=np.float32)
    faces = np.ascontiguousarray(prepared.quotient_faces, dtype=np.uint32)
    signed = _signed_volume(positions, faces)
    reversed_for_query = signed < 0.0
    query_faces = faces[:, [0, 2, 1]].copy() if reversed_for_query else faces
    mesh = manifold3d.Mesh(
        vert_properties=positions, tri_verts=query_faces,
        run_index=np.asarray([0, 3 * len(query_faces)], dtype=np.uint32),
        run_original_id=np.asarray([prepared.key[1]], dtype=np.uint32),
        face_id=np.arange(len(query_faces), dtype=np.uint32), tolerance=0)
    solid = manifold3d.Manifold(mesh)
    return solid, {
        "identity": list(prepared.key),
        "source_signed_volume_m3": signed,
        "manifold_query_reversed_all_faces": bool(reversed_for_query),
        "source_position_sha256": hashlib.sha256(prepared.positions.tobytes()).hexdigest(),
        "source_face_sha256": hashlib.sha256(prepared.faces.tobytes()).hexdigest(),
        "source_vertex_ancestry_sha256": hashlib.sha256(
            np.asarray(prepared.source_vertex_ids, dtype="<u8").tobytes()).hexdigest(),
    }


def _manifold_mesh(solid):
    """Read a floating Boolean result at native Mesh64 precision for analysis.

    The returned Float64 mesh is derived CSG only; it is never an emitted
    anatomy asset and its precision does not make Manifold's Boolean exact.
    """
    _require(callable(getattr(solid, "to_mesh64", None)),
             "installed Manifold3D must expose to_mesh64 for derived CSG analysis")
    mesh = solid.to_mesh64()
    positions = np.asarray(mesh.vert_properties)
    raw_faces = np.asarray(mesh.tri_verts)
    _require(positions.dtype == np.dtype("<f8") and positions.ndim == 2
             and positions.shape[1] >= 3 and len(positions) > 0
             and np.isfinite(positions[:, :3]).all(),
             "Manifold Mesh64 must provide finite Float64 XYZ")
    _require(raw_faces.ndim == 2 and raw_faces.shape[1] == 3
             and raw_faces.dtype.kind in "iu" and len(raw_faces) > 0
             and int(raw_faces.min()) >= 0 and int(raw_faces.max()) < len(positions)
             and int(raw_faces.max()) <= np.iinfo(np.uint32).max,
             "Manifold Mesh64 has invalid triangle indices")
    positions = np.ascontiguousarray(positions[:, :3], dtype="<f8")
    faces = np.ascontiguousarray(raw_faces, dtype="<u4")
    return positions, faces, mesh


def _exact_inside_witness(verifier, first, second, intersection):
    positions, faces, _ = _manifold_mesh(intersection)
    if len(faces) == 0:
        return None
    tri = positions[faces].astype(np.float64)
    signed_tet = np.einsum("ij,ij->i", tri[:, 0],
                           np.cross(tri[:, 1], tri[:, 2])) / 6.0
    total = float(signed_tet.sum())
    candidates = []
    if abs(total) > 0.0:
        centroid = np.sum(signed_tet[:, None] *
                          (tri[:, 0] + tri[:, 1] + tri[:, 2]) / 4.0, axis=0) / total
        candidates.append(("intersection_volume_centroid", centroid))
    candidates.append(("intersection_vertex_centroid",
                       positions.astype(np.float64).mean(axis=0)))
    diag = float(np.linalg.norm(positions.astype(np.float64).max(axis=0)
                                - positions.astype(np.float64).min(axis=0)))
    for face_id in np.linspace(0, len(tri) - 1, min(48, len(tri)), dtype=np.int64):
        triangle = tri[int(face_id)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            continue
        normal /= norm
        center = triangle.mean(axis=0)
        for power in (5, 6, 7, 8, 9):
            epsilon = max(diag * (10.0 ** -power),
                          float(np.max(np.abs(np.spacing(positions)))) * 8.0)
            candidates.append((f"intersection_face_inward_{power}",
                               center - normal * epsilon))
    first_meter = first.meter_index()
    second_meter = second.meter_index()
    for method, point in candidates:
        exact = tuple(Fraction.from_float(float(component)) for component in point)
        first_location = verifier.ci.point_location_prepared_indexed(exact, first_meter)
        second_location = verifier.ci.point_location_prepared_indexed(exact, second_meter)
        if first_location["location"] == "inside" and second_location["location"] == "inside":
            return {
                "method": method,
                "point_world_m": [float(component) for component in point],
                "first_location": first_location["location"],
                "second_location": second_location["location"],
                "predicate": "P6 exact rational point location for the exact binary64 candidate on the original Float32 source surfaces",
                "candidate_source": "derived floating Manifold Boolean Mesh64; candidate construction is not an exact CSG witness",
            }
    return None


def _material_vertex_ancestry(surface, tangent, collar_face_ids):
    faces = surface.faces
    collar_face_ids = np.asarray(collar_face_ids, dtype=np.int64)
    collar_vertex_ids = np.unique(faces[collar_face_ids].reshape(-1))
    _require(len(collar_vertex_ids) > 0, "collar has no source vertex ancestry")
    anchor_ids = collar_vertex_ids.astype(np.int64)
    anchor_weights = np.full(len(anchor_ids), 1.0 / len(anchor_ids), dtype=np.float64)
    positions = surface.positions.astype(np.float64)
    projections = positions @ tangent
    low_id, high_id = int(np.argmin(projections)), int(np.argmax(projections))
    _require(low_id != high_id, "source segment has no tangent extent")
    return {
        "material_anchor_local_vertex_ids": anchor_ids.tolist(),
        "material_anchor_source_vertex_ids": [
            int(surface.source_vertex_ids[index]) for index in anchor_ids],
        "material_anchor_weights": anchor_weights.tolist(),
        "tangent_support_local_vertex_ids": [low_id, high_id],
        "tangent_support_source_vertex_ids": [
            int(surface.source_vertex_ids[low_id]),
            int(surface.source_vertex_ids[high_id])],
        "frozen_collar_face_ids": collar_face_ids.tolist(),
    }


def _transport_material_station(surface, spec):
    positions = surface.positions.astype(np.float64)
    anchor_ids = np.asarray(spec["material_anchor_local_vertex_ids"], dtype=np.int64)
    weights = np.asarray(spec["material_anchor_weights"], dtype=np.float64)
    tangent_ids = np.asarray(spec["tangent_support_local_vertex_ids"], dtype=np.int64)
    collar_face_ids = np.asarray(spec["frozen_collar_face_ids"], dtype=np.int64)
    anchor = np.sum(positions[anchor_ids] * weights[:, None], axis=0)
    tangent = positions[int(tangent_ids[1])] - positions[int(tangent_ids[0])]
    norm = float(np.linalg.norm(tangent))
    _require(norm > 1e-15, "transported source tangent collapsed")
    tangent /= norm
    reference = np.asarray(spec["tangent_world_unit"], dtype=np.float64)
    if float(np.dot(tangent, reference)) < 0.0:
        tangent = -tangent
    patch = positions[surface.faces[collar_face_ids]].reshape(-1, 3)
    patch_extent = float(np.max((patch - anchor) @ tangent))
    margin = max(float(np.max(np.abs(np.spacing(surface.positions)))) * 8.0,
                 float(np.finfo(np.float32).tiny))
    offset = max(float(spec["station_offset_from_anchor_m"]),
                 patch_extent + margin)
    center = anchor + tangent * offset
    projection = (positions - center) @ tangent
    _require(projection.min() < 0.0 < projection.max(),
             "transported material station plane leaves source segment bounds")
    return {
        **spec,
        "section_center_world_m": center.tolist(),
        "tangent_world_unit": tangent.tolist(),
        "material_anchor_world_m": anchor.tolist(),
        "collar_projection_extent_m": patch_extent,
        "station_offset_from_anchor_m": offset,
        "numerical_margin_m": margin,
        "transported_from_source_vertex_ancestry": True,
    }


def _port_station_spec(surface, witness_points, collar_face_ids, *, route_anchor=None):
    positions = surface.positions.astype(np.float64)
    faces = surface.faces
    witness = np.asarray(witness_points, dtype=np.float64)
    _require(witness.ndim == 2 and witness.shape[1] == 3 and len(witness) > 0,
             "station inference requires exact interface witness points")
    junction = witness.mean(axis=0)
    if route_anchor is not None:
        anchor = np.asarray(route_anchor, dtype=np.float64)
        _require(anchor.shape == (3,) and np.isfinite(anchor).all(),
                 "route anchor must be finite XYZ")
        direction = anchor - junction
        method = "source-graph route anchor from frozen neighbor/port station"
    else:
        center = positions.mean(axis=0)
        cloud = positions - center
        covariance = cloud.T @ cloud / max(1, len(cloud))
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        method = "inferred PCA axis oriented toward source-surface interior centroid"
        if float(np.dot(direction, center - junction)) < 0.0:
            direction = -direction
        if abs(float(np.dot(direction, center - junction))) <= 1e-15:
            component = int(np.argmax(np.abs(direction)))
            if direction[component] < 0.0:
                direction = -direction
            method += " with deterministic positive dominant coordinate fallback"
    norm = float(np.linalg.norm(direction))
    _require(norm > 1e-15, "cannot infer a stable port tangent")
    tangent = direction / norm
    ancestry = _material_vertex_ancestry(surface, tangent, collar_face_ids)
    anchor_ids = np.asarray(ancestry["material_anchor_local_vertex_ids"], dtype=np.int64)
    weights = np.asarray(ancestry["material_anchor_weights"], dtype=np.float64)
    material_anchor = np.sum(positions[anchor_ids] * weights[:, None], axis=0)
    collar_points = positions[faces[np.asarray(collar_face_ids, dtype=np.int64)]].reshape(-1, 3)
    patch_extent = float(np.max((collar_points - material_anchor) @ tangent))
    margin = max(float(np.max(np.abs(np.spacing(surface.positions))) * 8.0),
                 float(np.finfo(np.float32).tiny))
    spec = {
        "tangent_world_unit": tangent.tolist(),
        "provenance": method,
        "inferred": True,
        "station_offset_from_anchor_m": patch_extent + margin,
        **ancestry,
    }
    return _transport_material_station(surface, spec)


def _validated_station_spec(surface, value, *, witness_points, collar_face_ids,
                            route_anchor=None):
    if value is None:
        return _port_station_spec(surface, witness_points, collar_face_ids,
                                  route_anchor=route_anchor)
    _require(isinstance(value, Mapping), "station specification must be a mapping")
    center = np.asarray(value.get("section_center_world_m"), dtype=np.float64)
    tangent = np.asarray(value.get("tangent_world_unit"), dtype=np.float64)
    _require(center.shape == (3,) and tangent.shape == (3,)
             and np.isfinite(center).all() and np.isfinite(tangent).all(),
             "station center and tangent must be finite XYZ")
    norm = float(np.linalg.norm(tangent))
    _require(abs(norm - 1.0) <= 2e-5, "station tangent must be unit length")
    tangent /= norm
    _require(isinstance(value.get("provenance"), str) and value["provenance"].strip(),
             "explicit station specification needs provenance")
    ancestry = _material_vertex_ancestry(surface, tangent, collar_face_ids)
    anchor_ids = np.asarray(ancestry["material_anchor_local_vertex_ids"], dtype=np.int64)
    weights = np.asarray(ancestry["material_anchor_weights"], dtype=np.float64)
    anchor = np.sum(surface.positions[anchor_ids].astype(np.float64) * weights[:, None], axis=0)
    offset = float(np.dot(center - anchor, tangent))
    collar_points = surface.positions[surface.faces[
        np.asarray(collar_face_ids, dtype=np.int64)]].astype(np.float64).reshape(-1, 3)
    patch_extent = float(np.max((collar_points - anchor) @ tangent))
    margin = max(float(np.max(np.abs(np.spacing(surface.positions)))) * 4.0,
                 float(np.finfo(np.float32).tiny))
    _require(offset > patch_extent + margin,
             "explicit section station must be beyond the frozen collar patch")
    spec = {
        "tangent_world_unit": tangent.tolist(),
        "provenance": value["provenance"],
        "inferred": bool(value.get("inferred", False)),
        "station_offset_from_anchor_m": offset,
        **ancestry,
    }
    return _transport_material_station(surface, spec)


def _convex_hull_2d(points):
    ordered = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(ordered) < 3:
        return ordered
    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))
    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _minimum_feret_width(hull):
    if len(hull) < 3:
        return 0.0
    points = np.asarray(hull, dtype=np.float64)
    widths = []
    for index in range(len(points)):
        edge = points[(index + 1) % len(points)] - points[index]
        length = float(np.linalg.norm(edge))
        if length == 0.0:
            continue
        normal = np.asarray([-edge[1], edge[0]]) / length
        projections = points @ normal
        widths.append(float(projections.max() - projections.min()))
    return min(widths, default=0.0)


def _section_loops(positions, faces, center, tangent, *, derived_csg=False):
    points = np.asarray(positions)
    expected_dtype = np.dtype("<f8") if derived_csg else np.dtype("<f4")
    precision_label = "derived Manifold Mesh64" if derived_csg else "emitted Float32"
    _require(points.dtype == expected_dtype and points.ndim == 2
             and points.shape[1] == 3 and len(points) > 0
             and np.isfinite(points).all(),
             f"{precision_label} section positions have invalid dtype or coordinates")
    points = np.ascontiguousarray(points)
    triangles = _faces(faces, name="section surface", vertex_count=len(points))
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    helper = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    u = np.cross(tangent, helper)
    u /= np.linalg.norm(u)
    v = np.cross(tangent, u)
    world_scale = max(1.0, float(np.max(np.abs(points))))
    coordinate_precision = np.finfo(points.dtype)
    ulp = max(float(np.max(np.abs(np.spacing(points)))),
              float(coordinate_precision.eps) * world_scale)
    tolerance = max(ulp * 2.0, float(coordinate_precision.tiny))
    segments = set()
    projected_points = {}
    for face in triangles:
        tri = points[face].astype(np.float64)
        distances = (tri - center) @ tangent
        hits = []
        for edge in range(3):
            a, b = edge, (edge + 1) % 3
            da, db = float(distances[a]), float(distances[b])
            if abs(da) <= tolerance:
                hits.append(tri[a])
            if (da < -tolerance and db > tolerance) or (da > tolerance and db < -tolerance):
                fraction = da / (da - db)
                hits.append(tri[a] + fraction * (tri[b] - tri[a]))
        unique = []
        for hit in hits:
            if not any(float(np.linalg.norm(hit - prior)) <= tolerance for prior in unique):
                unique.append(hit)
        if len(unique) != 2:
            continue
        nodes = []
        for hit in unique:
            xy = (float(np.dot(hit - center, u)), float(np.dot(hit - center, v)))
            key = (int(round(xy[0] / tolerance)), int(round(xy[1] / tolerance)))
            projected_points.setdefault(key, np.asarray(xy, dtype=np.float64))
            nodes.append(key)
        if nodes[0] != nodes[1]:
            segments.add(tuple(sorted(nodes)))
    adjacency = {}
    for first, second in segments:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return {"passed": False, "loop_count": 0, "reason": "section graph is open or branched",
                "segment_count": len(segments), "tolerance_m": tolerance}
    unused = set(segments)
    loops = []
    while unused:
        start, current = min(unused)
        loop = [start, current]
        previous = start
        unused.discard(tuple(sorted((start, current))))
        closed = False
        while True:
            options = sorted(adjacency[current] - {previous})
            if not options:
                break
            following = options[0]
            unused.discard(tuple(sorted((current, following))))
            if following == start:
                closed = True
                break
            if following in loop:
                break
            loop.append(following)
            previous, current = current, following
            if len(loop) > len(adjacency):
                break
        if not closed:
            return {"passed": False, "loop_count": len(loops),
                    "reason": "section cycle did not close",
                    "segment_count": len(segments), "tolerance_m": tolerance}
        if len(loop) >= 3 and len(set(loop)) == len(loop):
            loops.append(np.asarray([projected_points[key] for key in loop],
                                    dtype=np.float64))
        else:
            return {"passed": False, "loop_count": len(loops),
                    "reason": "section cycle did not close",
                    "segment_count": len(segments), "tolerance_m": tolerance}
    if len(loops) != 1:
        return {"passed": False, "loop_count": len(loops),
                "reason": "section must have exactly one closed loop",
                "segment_count": len(segments), "tolerance_m": tolerance}
    loop = loops[0]
    area = abs(float(np.dot(loop[:, 0], np.roll(loop[:, 1], -1))
                     - np.dot(loop[:, 1], np.roll(loop[:, 0], -1)))) / 2.0
    hull = _convex_hull_2d(loop)
    width = _minimum_feret_width(hull)
    positive_floor = tolerance * 64.0
    passed = area > positive_floor * positive_floor and width > positive_floor
    return {
        "passed": passed,
        "loop_count": 1,
        "vertex_count": int(len(loop)),
        "area_m2": area,
        "minimum_feret_width_m": width,
        "area_mm2": area * 1e6,
        "minimum_feret_width_mm": width * 1e3,
        "coordinate_dtype": str(points.dtype),
        "coordinate_ulp_tolerance_m": tolerance,
        "positive_measure_floor_m": positive_floor,
        "reason": None if passed else "cross-section area or minimum Feret width collapsed",
    }


def _section_comparison(candidate, baseline):
    _require(candidate["passed"] and baseline["passed"],
             "candidate and baseline sections must have positive closed-loop measures")
    ratios = {
        "area": candidate["area_m2"] / baseline["area_m2"],
        "minimum_feret_width": (candidate["minimum_feret_width_m"]
                                / baseline["minimum_feret_width_m"]),
    }
    intervals = {"area": [0.8, 1.25], "minimum_feret_width": [0.9, 1.1]}
    passed = all(intervals[name][0] <= value <= intervals[name][1]
                 for name, value in ratios.items())
    sensitivity = {}
    for lower in (0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9):
        for upper in (1.1, 1.25, 1.5, 2.0, 2.5):
            if lower >= upper:
                continue
            sensitivity[f"{lower:g}_to_{upper:g}"] = {
                name: lower <= value <= upper for name, value in ratios.items()
            }
    return {
        "candidate_to_baseline_ratios": ratios,
        "declared_preservation_intervals": intervals,
        "passed": passed,
        "sensitivity_diagnostics_only": sensitivity,
        "wide_bounds_are_not_preservation_success": True,
    }


def audit_brachiocephalic_port_pair_continuity(
    verifier, manifold3d, *, bridge_key, bridge_surface, target_key, target_surface,
    frozen_face_pairs, baseline_bridge_surface, baseline_target_surface,
    bridge_station=None, target_station=None, bridge_route_anchor_world_m=None,
    target_route_anchor_world_m=None,
):
    """Check one declared bridge port as closed solids and local section loops.

    This is a numerical geometry gate only. It does not establish measured
    physiology, native acceptance, or clinical calibre.
    """
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    bridge = _prepare_vessel(verifier, bridge_key, bridge_surface)
    target = _prepare_vessel(verifier, target_key, target_surface)
    base_bridge = _prepare_vessel(verifier, bridge_key, baseline_bridge_surface)
    base_target = _prepare_vessel(verifier, target_key, baseline_target_surface)
    source_preservation = []
    for label, candidate, baseline in (
        ("bridge", bridge, base_bridge), ("target", target, base_target)):
        same_faces = (candidate.faces.shape == baseline.faces.shape
                      and np.array_equal(candidate.faces, baseline.faces))
        same_vertices = (candidate.source_vertex_ids.shape == baseline.source_vertex_ids.shape
                         and np.array_equal(candidate.source_vertex_ids,
                                            baseline.source_vertex_ids))
        _require(same_faces, f"{label} candidate changed frozen source face ancestry")
        _require(same_vertices, f"{label} candidate changed frozen source vertex ancestry")
        candidate_volume = _signed_volume(candidate.positions, candidate.faces)
        baseline_volume = _signed_volume(baseline.positions, baseline.faces)
        same_orientation = candidate_volume * baseline_volume > 0.0
        _require(same_orientation,
                 f"{label} candidate reversed source branch orientation")
        source_preservation.append({
            "role": label,
            "source_face_ancestry_sha256": hashlib.sha256(
                candidate.faces.tobytes()).hexdigest(),
            "source_vertex_ancestry_sha256": hashlib.sha256(
                np.asarray(candidate.source_vertex_ids, dtype="<u8").tobytes()).hexdigest(),
            "baseline_face_ancestry_sha256": hashlib.sha256(
                baseline.faces.tobytes()).hexdigest(),
            "baseline_vertex_ancestry_sha256": hashlib.sha256(
                np.asarray(baseline.source_vertex_ids, dtype="<u8").tobytes()).hexdigest(),
            "candidate_signed_volume_m3": candidate_volume,
            "baseline_signed_volume_m3": baseline_volume,
            "branch_orientation_preserved": same_orientation,
        })
    pairs = _pair_rows(frozen_face_pairs, name="frozen port face-pair ancestry",
                       bridge_face_count=len(bridge.faces),
                       target_face_count=len(target.faces))
    bridge_patch = sorted({pair[0] for pair in pairs})
    target_patch = sorted({pair[1] for pair in pairs})
    allowed = tuple((a, b) for a in bridge_patch for b in target_patch)
    port_audit = verifier.audit_surface_pair(
        bridge, target, intended_face_pairs=allowed)
    observed = tuple(tuple(map(int, pair))
                     for pair in port_audit.get("declared_interface_triangle_pairs", []))
    exact_points = []
    for bridge_face, target_face in observed:
        bridge_triangle = [[Fraction.from_float(float(component)) for component in point]
                           for point in bridge.quotient_positions[
                               bridge.quotient_faces[bridge_face]].tolist()]
        target_triangle = [[Fraction.from_float(float(component)) for component in point]
                           for point in target.quotient_positions[
                               target.quotient_faces[target_face]].tolist()]
        exact_points.extend(verifier.ci.triangle_intersection_points(
            bridge_triangle, target_triangle))
    exact_points = sorted(set(tuple(float(component) for component in point)
                              for point in exact_points))
    port_pair_passed = bool(port_audit.get("passed")) and bool(exact_points)

    first, first_info = _make_manifold(bridge, manifold3d)
    second, second_info = _make_manifold(target, manifold3d)
    first_status, second_status = str(first.status()), str(second.status())
    manifold_inputs_passed = (first_status.endswith("NoError")
                              and second_status.endswith("NoError"))
    if manifold_inputs_passed:
        intersection = manifold3d.Manifold.batch_boolean(
            [first, second], manifold3d.OpType.Intersect)
        union = manifold3d.Manifold.batch_boolean(
            [first, second], manifold3d.OpType.Add)
        intersection_status, union_status = str(intersection.status()), str(union.status())
        intersection_volume = float(intersection.volume())
        intersection_components = len(intersection.decompose())
        union_components = len(union.decompose())
        union_volume = float(union.volume())
        union_positions, union_faces, union_mesh = _manifold_mesh(union)
        union_topology = {
            "solid_status": union_status,
            "component_count": union_components,
            "triangle_count": int(union.num_tri()),
            "vertex_count": int(union.num_vert()),
            "genus": int(union.genus()),
        }
        # The Boolean output remains numerical CSG. Mesh64 avoids introducing
        # another Float32 quantization before section analysis, but does not
        # establish exact Boolean topology or independently exact self-closure.
        union_self_intersections = None
        intersection_witness = _exact_inside_witness(
            verifier, bridge, target, intersection) if intersection_volume > 0.0 else None
    else:
        intersection = union = None
        intersection_status = union_status = "NOT_RUN_INPUT_MANIFOLD_ERROR"
        intersection_volume = union_volume = 0.0
        intersection_components = union_components = 0
        union_positions = np.empty((0, 3), dtype="<f4")
        union_faces = np.empty((0, 3), dtype="<u4")
        union_mesh = None
        union_topology = {}
        union_self_intersections = None
        intersection_witness = None
    positive_volume = intersection_volume > 0.0
    overlap_scale = min(_surface_area(bridge.quotient_positions, bridge.quotient_faces),
                        _surface_area(target.quotient_positions, target.quotient_faces))
    ulp = max(float(np.max(np.abs(np.spacing(bridge.quotient_positions)))),
              float(np.max(np.abs(np.spacing(target.quotient_positions)))))
    threshold_base = 64.0 * ulp * overlap_scale
    overlap_sensitivity = {
        str(multiplier): {
            "minimum_intersection_volume_m3": threshold_base * (multiplier / 64.0),
            "passes": intersection_volume > threshold_base * (multiplier / 64.0),
        } for multiplier in (0, 16, 32, 64, 128)
    }
    union_passed = bool(
        manifold_inputs_passed and union_status.endswith("NoError")
        and union_components == 1 and union_volume > 0.0
    )
    solid_passed = bool(positive_volume and intersection_witness is not None
                        and union_passed)
    _require(port_pair_passed,
             f"port {bridge_key}/{target_key} lacks a frozen-patch exact interface witness")
    _require(solid_passed,
             f"port {bridge_key}/{target_key} fails overlap, exact inside witness, or closed connected union")

    bridge_station = _validated_station_spec(
        bridge, bridge_station, witness_points=exact_points, collar_face_ids=bridge_patch,
        route_anchor=bridge_route_anchor_world_m)
    target_station = _validated_station_spec(
        target, target_station, witness_points=exact_points, collar_face_ids=target_patch,
        route_anchor=target_route_anchor_world_m)
    station_results = {}
    for role, surface, base_surface, spec in (
        ("bridge", bridge, base_bridge, bridge_station),
        ("target", target, base_target, target_station),
    ):
        center = spec["section_center_world_m"]
        tangent = spec["tangent_world_unit"]
        candidate_section = _section_loops(surface.positions, surface.faces, center, tangent)
        baseline_spec = _transport_material_station(base_surface, spec)
        baseline_section = _section_loops(
            base_surface.positions, base_surface.faces,
            baseline_spec["section_center_world_m"],
            baseline_spec["tangent_world_unit"])
        station_results[role] = {
            "station": spec,
            "baseline_station_transported_by_ancestry": baseline_spec,
            "candidate": candidate_section,
            "baseline": baseline_section,
            "candidate_vs_baseline": _section_comparison(
                candidate_section, baseline_section),
        }
        _require(station_results[role]["candidate_vs_baseline"]["passed"],
                 f"{role} section calibre ratio is outside the declared local screen")
    union_candidate_section = _section_loops(
        union_positions, union_faces,
        bridge_station["section_center_world_m"], bridge_station["tangent_world_unit"],
        derived_csg=True)
    base_first, _ = _make_manifold(base_bridge, manifold3d)
    base_second, _ = _make_manifold(base_target, manifold3d)
    base_union = manifold3d.Manifold.batch_boolean(
        [base_first, base_second], manifold3d.OpType.Add)
    base_union_positions, base_union_faces, _ = _manifold_mesh(base_union)
    base_bridge_station = _transport_material_station(base_bridge, bridge_station)
    base_target_station = _transport_material_station(base_target, target_station)
    baseline_union_bridge_section = _section_loops(
        base_union_positions, base_union_faces,
        base_bridge_station["section_center_world_m"],
        base_bridge_station["tangent_world_unit"], derived_csg=True)
    station_results["union_at_bridge_station"] = {
        "station": bridge_station,
        "candidate": union_candidate_section,
        "baseline_station_transported_by_ancestry": base_bridge_station,
        "baseline": baseline_union_bridge_section,
        "candidate_vs_baseline": _section_comparison(
            union_candidate_section, baseline_union_bridge_section),
    }
    _require(station_results["union_at_bridge_station"]["candidate_vs_baseline"]["passed"],
             "union section calibre ratio is outside the declared local screen")
    union_target_section = _section_loops(
        union_positions, union_faces,
        target_station["section_center_world_m"], target_station["tangent_world_unit"],
        derived_csg=True)
    baseline_union_target_section = _section_loops(
        base_union_positions, base_union_faces,
        base_target_station["section_center_world_m"],
        base_target_station["tangent_world_unit"], derived_csg=True)
    station_results["union_at_target_station"] = {
        "station": target_station,
        "baseline_station_transported_by_ancestry": base_target_station,
        "candidate": union_target_section,
        "baseline": baseline_union_target_section,
        "candidate_vs_baseline": _section_comparison(
            union_target_section, baseline_union_target_section),
    }
    _require(station_results["union_at_target_station"]["candidate_vs_baseline"]["passed"],
             "union target-station calibre ratio is outside the declared local screen")
    return {
        "schema": "numi.human.brachiocephalic-port-solid-continuity.v1",
        "bridge": list(bridge_key),
        "target": list(target_key),
        "passed": True,
        "source_preservation": source_preservation,
        "port_pair_gate": {
            "passed": port_pair_passed,
            "frozen_bridge_patch_faces": bridge_patch,
            "frozen_target_patch_faces": target_patch,
            "allowed_patch_cross_product_pair_count": len(allowed),
            "frozen_source_face_pair_count": len(pairs),
            "observed_pose_face_pair_count": len(observed),
            "observed_pose_face_pairs": [list(pair) for pair in observed],
            "exact_interface_point_count": len(exact_points),
            "exact_interface_points_world_m": [list(point) for point in exact_points],
        },
        "solid_gate": {
            "passed": solid_passed,
            "manifold_input_statuses": [first_status, second_status],
            "manifold_inputs": [first_info, second_info],
            "intersection_status": intersection_status,
            "intersection_volume_m3": intersection_volume,
            "intersection_component_count": intersection_components,
            "strict_positive_volume": positive_volume,
            "float32_resolution_sensitivity_diagnostic_only": overlap_sensitivity,
            "intersection_threshold_is_not_used_for_acceptance": True,
            "exact_inside_both_witness": intersection_witness,
            "union_status": union_status,
            "union_volume_m3": union_volume,
            "union_component_count": union_components,
            "union_numerical_solid_contract_passed": bool(
                manifold_inputs_passed and union_status.endswith("NoError")
                and union_components == 1 and union_volume > 0.0),
            "union_csg_analysis_precision": "Manifold Mesh64 Float64; floating Boolean analysis only, no emitted asset",
            "union_boolean_exact_topology_proven": False,
            "union_self_intersection_count": union_self_intersections,
            "union_self_intersections": "not independently exact-audited; Manifold NoError/decompose/volume are numerical CSG diagnostics, not exact topology proof",
            "union_topology_diagnostics": union_topology,
            "union_generated_face_count": int(np.count_nonzero(
                np.asarray(union_mesh.face_id) == np.iinfo(np.uint32).max))
                if union_mesh is not None else 0,
            "union_section_mesh_position_dtype": str(union_positions.dtype),
            "union_lineage_original_ids": sorted(set(
                np.asarray(union_mesh.run_original_id).astype(int).tolist()))
                if union_mesh is not None else [],
        },
        "calibre_gate": {
            "passed": True,
            "method": "closed section loop area and minimum Feret width at stations beyond frozen collar patches; candidate and baseline stations are transported from the same source vertex ancestry",
            "declared_preservation_intervals": {
                "area": [0.8, 1.25], "minimum_feret_width": [0.9, 1.1]},
            "station_results": station_results,
            "sensitivity_is_diagnostic_only": True,
            "clinical_calibre_claim": False,
        },
        "source_geometry_mutated": False,
    }


def audit_brachiocephalic_six_port_continuity(
    verifier, manifold3d, candidate_surfaces, baseline_surfaces, port_face_pairs,
    *, station_specs=None, route_anchors=None,
):
    """Run source-ordered bridge port checks and stop at the first failure."""
    station_specs = {} if station_specs is None else station_specs
    route_anchors = {} if route_anchors is None else route_anchors
    reports = []
    expected = []
    for bridge_key in BRIDGE_KEYS:
        for stable_id in PORT_TARGET_IDS[bridge_key[1]]:
            expected.append((bridge_key, (51011, stable_id)))
    _require(set(port_face_pairs) == set(expected),
             "six-port continuity requires exactly the frozen six interface witnesses")
    _require(set(candidate_surfaces) >= {key for pair in expected for key in pair},
             "candidate geometry must contain both bridges and all six port targets")
    _require(set(baseline_surfaces) >= {key for pair in expected for key in pair},
             "baseline geometry must contain all six port targets")
    for pair in expected:
        stations = station_specs.get(pair, {})
        anchors = route_anchors.get(pair, {})
        try:
            report = audit_brachiocephalic_port_pair_continuity(
                verifier, manifold3d,
                bridge_key=pair[0], bridge_surface=candidate_surfaces[pair[0]],
                target_key=pair[1], target_surface=candidate_surfaces[pair[1]],
                frozen_face_pairs=port_face_pairs[pair],
                baseline_bridge_surface=baseline_surfaces[pair[0]],
                baseline_target_surface=baseline_surfaces[pair[1]],
                bridge_station=stations.get("bridge"),
                target_station=stations.get("target"),
                bridge_route_anchor_world_m=anchors.get("bridge"),
                target_route_anchor_world_m=anchors.get("target"),
            )
        except ValueError as error:
            report = {
                "schema": "numi.human.brachiocephalic-port-solid-continuity.v1",
                "bridge": list(pair[0]), "target": list(pair[1]),
                "passed": False, "failure": str(error),
                "source_geometry_mutated": False,
            }
        reports.append(report)
        if not report.get("passed"):
            break
    all_passed = len(reports) == len(expected) and all(row["passed"] for row in reports)
    first_failed = next((row for row in reports if not row.get("passed")), None)
    return {
        "schema": "numi.human.brachiocephalic-six-port-solid-continuity.v1",
        "status": ("SIX_PORT_SOLID_AND_LOCAL_CALIBRE_SCREEN_PASSED"
                   if all_passed else "SIX_PORT_SOLID_AND_LOCAL_CALIBRE_SCREEN_REJECTED"),
        "passed": all_passed,
        "port_count": len(reports),
        "evaluated_port_count": len(reports),
        "requested_port_count": len(expected),
        "first_failed_port": (None if first_failed is None else {
            "bridge": first_failed.get("bridge"),
            "target": first_failed.get("target"),
        }),
        "ports_not_evaluated_after_first_failure": [
            {"bridge": list(pair[0]), "target": list(pair[1])}
            for pair in expected[len(reports):]
        ],
        "ports": reports,
        "limits": {
            "native_gpu_confirmation": "required",
            "all_pose_qualification": "required",
            "clinical_calibre": "not_established",
            "anatomical_admission": False,
        },
    }
