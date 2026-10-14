"""Captured-pose surface-intersection stage for appended brachiocephalic veins.

This wrapper uses the existing resting-anatomy exact packed-Float32 verifier.
It checks each predicted bridge against captured anatomy through the frozen
P6 exact predicates, then exposes a separate source-bound solid-continuity and
local-calibre screen. It is not clinical or physiological qualification.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from fractions import Fraction

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
                    [list(pair) for pair in sorted({*unexpected, *outside_patch})],
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


def _source_section_lineage(surface):
    """Bind each unchanged prepared source face to its stable surface identity."""
    return {
        "face_owner_ids": np.full(
            len(surface.faces), surface.key[1], dtype="<u4"),
        "face_source_ids": np.arange(len(surface.faces), dtype="<u8"),
        "face_generated_mask": np.zeros(len(surface.faces), dtype=np.bool_),
    }


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


def _mesh_face_lineage(mesh, source_face_counts):
    """Resolve each floating CSG face to its Manifold run and input face.

    Manifold's run_index addresses flattened triangle-corner ranges. Each
    triangle must therefore lie wholly in one run, and its face_id must index
    a source face in that run's original stable surface. Any generated,
    unmapped, or malformed face is retained as an explicit sentinel so a
    junction-section proof can fail closed when the lineage is ambiguous.
    """
    triangles = np.asarray(mesh.tri_verts)
    raw_run_index = np.asarray(mesh.run_index, dtype=np.int64)
    raw_owner_ids = np.asarray(mesh.run_original_id, dtype=np.uint64)
    raw_face_ids = np.asarray(mesh.face_id, dtype=np.uint64)
    _require(triangles.ndim == 2 and triangles.shape[1] == 3
             and len(triangles) > 0,
             "Manifold face lineage requires a nonempty triangle mesh")
    _require(raw_run_index.shape == (len(raw_owner_ids) + 1,)
             and len(raw_owner_ids) > 0 and raw_run_index[0] == 0
             and raw_run_index[-1] == 3 * len(triangles)
             and np.all(np.diff(raw_run_index) > 0)
             and np.all(raw_run_index % 3 == 0),
             "Manifold face lineage has invalid triangle-run boundaries")
    _require(raw_face_ids.shape == (len(triangles),),
             "Manifold face lineage face_id count differs from triangles")
    _require(isinstance(source_face_counts, Mapping)
             and all(type(key) is int and key > 0
                     and type(value) is int and value > 0
                     for key, value in source_face_counts.items()),
             "Manifold face lineage needs positive stable-ID source face counts")

    face_corner_offsets = np.arange(len(triangles), dtype=np.int64) * 3
    face_runs = np.searchsorted(raw_run_index[1:], face_corner_offsets, side="right")
    owner_sentinel = np.iinfo(np.uint32).max
    source_sentinel = np.iinfo(np.uint64).max
    owner_ids = np.full(len(triangles), owner_sentinel, dtype="<u4")
    source_face_ids = np.full(len(triangles), source_sentinel, dtype="<u8")
    generated_faces = np.zeros(len(triangles), dtype=np.bool_)
    unmapped_count = 0
    generated_markers = {int(np.iinfo(np.uint32).max), int(source_sentinel)}
    for face_index, run_index in enumerate(face_runs):
        owner = int(raw_owner_ids[int(run_index)])
        source_face = int(raw_face_ids[face_index])
        if owner not in source_face_counts:
            unmapped_count += 1
            continue
        owner_ids[face_index] = owner
        if source_face in generated_markers:
            generated_faces[face_index] = True
            continue
        if source_face < source_face_counts[owner]:
            source_face_ids[face_index] = source_face
        else:
            unmapped_count += 1
    return {
        "face_owner_ids": owner_ids,
        "source_face_ids": source_face_ids,
        "generated_face_mask": generated_faces,
        "report": {
            "face_count": len(triangles),
            "run_original_ids": sorted(set(map(int, raw_owner_ids.tolist()))),
            "source_face_counts_by_owner": {
                str(key): int(value) for key, value in sorted(source_face_counts.items())},
            "generated_face_count": int(np.count_nonzero(generated_faces)),
            "unmapped_face_count": int(unmapped_count),
            "lineage_source": "Manifold Mesh64 run_original_id plus face_id and triangle-aligned run_index",
        },
    }


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


def _explicit_material_vertex_ancestry(surface, value, collar_face_ids):
    fields = (
        "material_anchor_local_vertex_ids",
        "material_anchor_source_vertex_ids",
        "material_anchor_weights",
        "tangent_support_local_vertex_ids",
        "tangent_support_source_vertex_ids",
        "frozen_collar_face_ids",
    )
    present = [field in value for field in fields]
    if not any(present):
        return None
    _require(all(present),
             "explicit material station ancestry must provide every vertex and collar field")
    source_ids = np.asarray(surface.source_vertex_ids)
    _require(source_ids.shape == (len(surface.positions),)
             and source_ids.dtype.kind in "iu",
             "material station requires per-vertex source identity")
    anchor_local = value["material_anchor_local_vertex_ids"]
    anchor_source = value["material_anchor_source_vertex_ids"]
    anchor_weights = value["material_anchor_weights"]
    tangent_local = value["tangent_support_local_vertex_ids"]
    tangent_source = value["tangent_support_source_vertex_ids"]
    frozen_collar = value["frozen_collar_face_ids"]
    _require(isinstance(anchor_local, (list, tuple)) and len(anchor_local) > 0
             and all(type(index) is int and 0 <= index < len(source_ids)
                     for index in anchor_local),
             "material station anchor local vertex IDs are invalid")
    _require(isinstance(anchor_source, (list, tuple))
             and len(anchor_source) == len(anchor_local)
             and all(type(index) is int and index >= 0 for index in anchor_source)
             and np.array_equal(source_ids[np.asarray(anchor_local, dtype=np.int64)],
                                np.asarray(anchor_source, dtype=source_ids.dtype)),
             "material station anchor source vertex IDs do not match current topology")
    weights = np.asarray(anchor_weights, dtype=np.float64)
    _require(weights.shape == (len(anchor_local),) and np.isfinite(weights).all()
             and np.all(weights > 0.0)
             and abs(float(np.sum(weights)) - 1.0) <= 1e-12,
             "material station anchor weights must be positive and sum to one")
    _require(isinstance(tangent_local, (list, tuple)) and len(tangent_local) == 2
             and all(type(index) is int and 0 <= index < len(source_ids)
                     for index in tangent_local)
             and tangent_local[0] != tangent_local[1],
             "material station tangent support local IDs must be two distinct vertices")
    _require(isinstance(tangent_source, (list, tuple)) and len(tangent_source) == 2
             and all(type(index) is int and index >= 0 for index in tangent_source)
             and np.array_equal(source_ids[np.asarray(tangent_local, dtype=np.int64)],
                                np.asarray(tangent_source, dtype=source_ids.dtype)),
             "material station tangent source IDs do not match current topology")
    collar_ids = np.asarray(collar_face_ids, dtype=np.int64)
    _require(isinstance(frozen_collar, (list, tuple)) and len(frozen_collar) > 0
             and all(type(index) is int and 0 <= index < len(surface.faces)
                     for index in frozen_collar),
             "material station frozen collar face IDs are invalid")
    frozen_collar_ids = np.asarray(frozen_collar, dtype=np.int64)
    _require(collar_ids.ndim == 1 and len(collar_ids) > 0
             and np.all(np.isin(collar_ids, frozen_collar_ids)),
             "observed station collar faces leave the frozen interface patch")
    return {
        "material_anchor_local_vertex_ids": [int(index) for index in anchor_local],
        "material_anchor_source_vertex_ids": [int(index) for index in anchor_source],
        "material_anchor_weights": weights.tolist(),
        "tangent_support_local_vertex_ids": [int(index) for index in tangent_local],
        "tangent_support_source_vertex_ids": [int(index) for index in tangent_source],
        "frozen_collar_face_ids": [int(index) for index in frozen_collar],
    }


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
    retained_ancestry = _explicit_material_vertex_ancestry(
        surface, value, collar_face_ids)
    if retained_ancestry is not None:
        offset = float(value.get("station_offset_from_anchor_m", 0.0))
        _require(np.isfinite(offset) and offset > 0.0,
                 "explicit material station needs a positive ancestry-relative offset")
        spec = {
            "tangent_world_unit": tangent.tolist(),
            "provenance": value["provenance"],
            "inferred": bool(value.get("inferred", False)),
            "station_offset_from_anchor_m": offset,
            **retained_ancestry,
        }
        return _transport_material_station(surface, spec)
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
    ordered = sorted({(float(point[0]), float(point[1])) for point in points})
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


def _exact_interface_points(verifier, first, second, face_pairs):
    points = []
    for first_face, second_face in face_pairs:
        first_triangle = [[Fraction.from_float(float(component)) for component in point]
                          for point in first.quotient_positions[
                              first.quotient_faces[first_face]].tolist()]
        second_triangle = [[Fraction.from_float(float(component)) for component in point]
                           for point in second.quotient_positions[
                               second.quotient_faces[second_face]].tolist()]
        points.extend(verifier.ci.triangle_intersection_points(
            first_triangle, second_triangle))
    return sorted({tuple(float(component) for component in point)
                    for point in points})


def _section_loop_lineage(face_indices, face_owner_ids, face_source_ids,
                          face_generated_mask, *, lineage_available):
    if not lineage_available:
        return {
            "lineage_available": False, "lineage_complete": False,
            "section_face_ids": sorted(set(map(int, face_indices))),
            "owner_ids": [], "source_face_ancestor_ids": [],
            "source_face_ancestor_ids_sha256": None,
            "generated_face_count": None, "unmapped_face_count": None,
        }
    owner_sentinel = int(np.iinfo(np.uint32).max)
    source_sentinel = int(np.iinfo(np.uint64).max)
    face_ids = sorted(set(map(int, face_indices)))
    owners = sorted({int(face_owner_ids[index]) for index in face_ids
                     if int(face_owner_ids[index]) != owner_sentinel})
    ancestors = sorted({int(face_source_ids[index]) for index in face_ids
                        if int(face_source_ids[index]) != source_sentinel})
    generated_ids = [index for index in face_ids
                     if bool(face_generated_mask[index])]
    unmapped_ids = [
        index for index in face_ids
        if (int(face_owner_ids[index]) == owner_sentinel
            or (not bool(face_generated_mask[index])
                and int(face_source_ids[index]) == source_sentinel))
    ]
    ancestor_bytes = np.asarray(ancestors, dtype="<u8").tobytes()
    return {
        "lineage_available": True,
        "lineage_complete": bool(face_ids and not generated_ids and not unmapped_ids),
        "section_face_ids": face_ids,
        "owner_ids": owners,
        "source_face_ancestor_ids": ancestors,
        "source_face_ancestor_ids_sha256": hashlib.sha256(ancestor_bytes).hexdigest(),
        "generated_face_count": len(generated_ids),
        "generated_face_ids": generated_ids,
        "unmapped_face_count": len(unmapped_ids),
        "unmapped_face_ids": unmapped_ids,
    }


def _section_loops(positions, faces, center, tangent, *, derived_csg=False,
                   face_owner_ids=None, face_source_ids=None,
                   face_generated_mask=None):
    points = np.asarray(positions)
    expected_dtype = np.dtype("<f8") if derived_csg else np.dtype("<f4")
    precision_label = "derived Manifold Mesh64" if derived_csg else "emitted Float32"
    _require(points.dtype == expected_dtype and points.ndim == 2
             and points.shape[1] == 3 and len(points) > 0
             and np.isfinite(points).all(),
             f"{precision_label} section positions have invalid dtype or coordinates")
    points = np.ascontiguousarray(points)
    triangles = _faces(faces, name="section surface", vertex_count=len(points))
    lineage_available = face_owner_ids is not None or face_source_ids is not None
    if lineage_available:
        _require(face_owner_ids is not None and face_source_ids is not None,
                 "section face lineage requires owner and source-face arrays")
        face_owner_ids = np.asarray(face_owner_ids)
        face_source_ids = np.asarray(face_source_ids)
        _require(face_owner_ids.shape == (len(triangles),)
                 and face_owner_ids.dtype.kind in "iu"
                 and face_source_ids.shape == (len(triangles),)
                 and face_source_ids.dtype.kind in "iu",
                 "section face lineage arrays must match the triangle count")
        _require(not (face_owner_ids.dtype.kind == "i"
                      and np.any(face_owner_ids < 0))
                 and not (face_source_ids.dtype.kind == "i"
                          and np.any(face_source_ids < 0)),
                 "section face lineage IDs must be nonnegative")
        face_owner_ids = np.asarray(face_owner_ids, dtype="<u4")
        face_source_ids = np.asarray(face_source_ids, dtype="<u8")
        if face_generated_mask is None:
            face_generated_mask = np.zeros(len(triangles), dtype=np.bool_)
        else:
            face_generated_mask = np.asarray(face_generated_mask, dtype=np.bool_)
        _require(face_generated_mask.shape == (len(triangles),),
                 "section generated-face mask must match the triangle count")
    else:
        _require(face_generated_mask is None,
                 "section generated-face mask requires face lineage arrays")
        face_generated_mask = np.zeros(len(triangles), dtype=np.bool_)
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    _require(center.shape == (3,) and np.isfinite(center).all()
             and tangent.shape == (3,) and np.isfinite(tangent).all()
             and float(np.linalg.norm(tangent)) > 0.0,
             "section plane center and tangent must be finite XYZ values")
    tangent = tangent / np.linalg.norm(tangent)
    helper = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    u = np.cross(tangent, helper)
    u /= np.linalg.norm(u)
    v = np.cross(tangent, u)

    # Quotient only bitwise-equal source coordinates before deriving topological
    # section identities. This rejoins duplicated seam vertices without changing
    # the input arrays or merging merely-nearby anatomical points.
    canonical_by_coordinate = {}
    canonical_source_ids = np.empty(len(points), dtype=np.int64)
    canonical_points = []
    for source_id, point in enumerate(points):
        coordinate = tuple(0.0 if float(value) == 0.0 else float(value)
                           for value in point)
        canonical_id = canonical_by_coordinate.get(coordinate)
        if canonical_id is None:
            canonical_id = len(canonical_points)
            canonical_by_coordinate[coordinate] = canonical_id
            canonical_points.append(np.asarray(point, dtype=np.float64))
        canonical_source_ids[source_id] = canonical_id
    canonical_points = np.ascontiguousarray(canonical_points, dtype=np.float64)
    canonical_triangles = canonical_source_ids[triangles]
    _require(not np.any(
        (canonical_triangles[:, 0] == canonical_triangles[:, 1])
        | (canonical_triangles[:, 1] == canonical_triangles[:, 2])
        | (canonical_triangles[:, 2] == canonical_triangles[:, 0])),
        "section surface has a triangle degenerate after exact-coordinate quotient")

    world_scale = max(1.0, float(np.max(np.abs(points))))
    coordinate_precision = np.finfo(points.dtype)
    ulp = max(float(np.max(np.abs(np.spacing(points)))),
              float(coordinate_precision.eps) * world_scale)
    tolerance = max(ulp * 2.0, float(coordinate_precision.tiny))
    signed_distances = (canonical_points - center) @ tangent
    signs = np.zeros(len(canonical_points), dtype=np.int8)
    signs[signed_distances > tolerance] = 1
    signs[signed_distances < -tolerance] = -1

    segments = set()
    segment_faces = {}
    node_positions = {}
    tangent_vertex_ids = set()
    coplanar_edge_incidence = {}
    coplanar_face_ids = []
    face_signs = signs[canonical_triangles]
    for face_index, (face, face_sign) in enumerate(zip(canonical_triangles, face_signs)):
        zero = np.flatnonzero(face_sign == 0)
        if len(zero) == 3:
            coplanar_face_ids.append(face_index)
        elif len(zero) == 2:
            zero_ids = tuple(sorted(int(face[index]) for index in zero))
            third = next(index for index in range(3) if index not in zero)
            coplanar_edge_incidence.setdefault(zero_ids, []).append(
                (face_index, int(face_sign[third])))
    if coplanar_face_ids:
        return {
            "passed": False, "loop_count": 0,
            "reason": "section plane contains a coplanar source face",
            "segment_count": 0, "section_node_count": 0,
            "coplanar_face_count": len(coplanar_face_ids),
            "coplanar_face_ids": coplanar_face_ids[:32],
            "coordinate_dtype": str(points.dtype),
            "coordinate_ulp_tolerance_m": tolerance,
        }
    for edge, incidents in sorted(coplanar_edge_incidence.items()):
        if len(incidents) != 2:
            return {
                "passed": False, "loop_count": 0,
                "reason": "coplanar section edge does not have exactly two incident faces",
                "segment_count": 0, "section_node_count": 0,
                "coplanar_edge_count": len(coplanar_edge_incidence),
                "coplanar_edge_vertex_ids": list(edge),
                "coplanar_edge_incident_face_count": len(incidents),
                "coordinate_dtype": str(points.dtype),
                "coordinate_ulp_tolerance_m": tolerance,
            }
        if incidents[0][1] == incidents[1][1]:
            return {
                "passed": False, "loop_count": 0,
                "reason": "section plane is tangent along a coplanar source edge",
                "segment_count": 0, "section_node_count": 0,
                "coplanar_edge_count": len(coplanar_edge_incidence),
                "coplanar_edge_vertex_ids": list(edge),
                "coplanar_edge_incident_face_ids": [item[0] for item in incidents],
                "incident_side_signs": [item[1] for item in incidents],
                "coordinate_dtype": str(points.dtype),
                "coordinate_ulp_tolerance_m": tolerance,
            }

    def vertex_node(vertex_id):
        key = ("v", int(vertex_id))
        node_positions.setdefault(key, canonical_points[int(vertex_id)])
        return key

    edge_intersections = {}

    def crossing_edge_node(first_id, second_id):
        edge = tuple(sorted((int(first_id), int(second_id))))
        cached = edge_intersections.get(edge)
        if cached is not None:
            return cached[0]
        first, second = edge
        first_sign, second_sign = int(signs[first]), int(signs[second])
        _require(first_sign * second_sign == -1,
                 "section edge intersection lacks opposite signed endpoints")
        first_distance = float(signed_distances[first])
        second_distance = float(signed_distances[second])
        fraction = first_distance / (first_distance - second_distance)
        point = canonical_points[first] + fraction * (
            canonical_points[second] - canonical_points[first])
        key = ("e", edge[0], edge[1])
        edge_intersections[edge] = (key, np.asarray(point, dtype=np.float64))
        node_positions.setdefault(key, edge_intersections[edge][1])
        return key

    def add_segment(first, second, face_index):
        # Distinct source edges stay distinct even when their geometric
        # intersections are closer than the Float32 coordinate tolerance.
        if first != second:
            edge = tuple(sorted((first, second)))
            segments.add(edge)
            segment_faces.setdefault(edge, set()).add(int(face_index))

    for face_index, (face, face_sign) in enumerate(
            zip(canonical_triangles, face_signs)):
        zero = np.flatnonzero(face_sign == 0)
        if len(zero) == 2:
            first, second = (vertex_node(int(face[index])) for index in zero)
            add_segment(first, second, face_index)
        elif len(zero) == 1:
            on_plane_index = int(zero[0])
            others = [index for index in range(3) if index != on_plane_index]
            first_index, second_index = others
            first_sign, second_sign = int(face_sign[first_index]), int(face_sign[second_index])
            if first_sign == second_sign:
                tangent_vertex_ids.add(int(face[on_plane_index]))
            else:
                first = vertex_node(int(face[on_plane_index]))
                second = crossing_edge_node(
                    int(face[first_index]), int(face[second_index]))
                add_segment(first, second, face_index)
        else:
            crossing_edges = []
            for edge_index in range(3):
                first_index, second_index = edge_index, (edge_index + 1) % 3
                if int(face_sign[first_index]) * int(face_sign[second_index]) == -1:
                    crossing_edges.append(crossing_edge_node(
                        int(face[first_index]), int(face[second_index])))
            if len(crossing_edges) == 2:
                add_segment(crossing_edges[0], crossing_edges[1], face_index)
            else:
                _require(len(crossing_edges) == 0,
                         "section triangle has an unsupported sign arrangement")

    adjacency = {}
    for first, second in segments:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    isolated_tangent_vertices = sorted(
        vertex for vertex in tangent_vertex_ids if ("v", vertex) not in adjacency)

    def graph_details():
        degree_histogram = {}
        for neighbors in adjacency.values():
            degree = str(len(neighbors))
            degree_histogram[degree] = degree_histogram.get(degree, 0) + 1
        node_kind_counts = {}
        for key in adjacency:
            node_kind_counts[key[0]] = node_kind_counts.get(key[0], 0) + 1
        segment_lengths = [
            float(np.linalg.norm(node_positions[first] - node_positions[second]))
            for first, second in segments
        ]
        return {
            "section_node_count": len(adjacency),
            "minimum_section_segment_length_m": min(segment_lengths, default=None),
            "section_degree_histogram": degree_histogram,
            "section_node_kind_counts": node_kind_counts,
            "section_edge_intersection_count": len(edge_intersections),
            "exact_coordinate_quotient_vertex_count": len(canonical_points),
            "exact_coordinate_duplicate_vertex_count": len(points) - len(canonical_points),
            "coplanar_edge_count": len(coplanar_edge_incidence),
            "isolated_tangent_vertex_ids": isolated_tangent_vertices[:32],
            "coordinate_dtype": str(points.dtype),
            "coordinate_ulp_tolerance_m": tolerance,
            "section_center_world_m": center.tolist(),
            "section_tangent_world_unit": tangent.tolist(),
        }

    details = graph_details()
    if isolated_tangent_vertices:
        return {
            "passed": False, "loop_count": 0,
            "reason": "section plane touches an isolated tangent source vertex",
            "segment_count": len(segments), **details,
        }
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return {
            "passed": False, "loop_count": 0,
            "reason": "section graph is open or branched",
            "segment_count": len(segments), **details,
        }

    projected_points = {}
    for key, point in node_positions.items():
        projected_points[key] = np.asarray([
            float(np.dot(point - center, u)),
            float(np.dot(point - center, v)),
        ], dtype=np.float64)
    unused = set(segments)
    loops = []
    loop_records = []
    while unused:
        start, current = min(unused)
        loop = [start, current]
        loop_edges = [tuple(sorted((start, current)))]
        previous = start
        unused.discard(tuple(sorted((start, current))))
        closed = False
        while True:
            options = sorted(adjacency[current] - {previous})
            if not options:
                break
            following = options[0]
            edge = tuple(sorted((current, following)))
            loop_edges.append(edge)
            unused.discard(edge)
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
            return {
                "passed": False, "loop_count": len(loops),
                "reason": "section cycle did not close",
                "segment_count": len(segments), **details,
            }
        if len(loop) >= 3 and len(set(loop)) == len(loop):
            loops.append(np.asarray([projected_points[key] for key in loop],
                                    dtype=np.float64))
            loop_records.append((loops[-1], tuple(loop_edges)))
        else:
            return {
                "passed": False, "loop_count": len(loops),
                "reason": "section cycle did not close",
                "segment_count": len(segments), **details,
            }
    positive_floor = tolerance * 64.0
    loop_metrics = []
    for loop, loop_edges in loop_records:
        area = abs(float(np.dot(loop[:, 0], np.roll(loop[:, 1], -1))
                         - np.dot(loop[:, 1], np.roll(loop[:, 0], -1)))) / 2.0
        hull = _convex_hull_2d(loop)
        width = _minimum_feret_width(hull)
        loop_passed = area > positive_floor * positive_floor and width > positive_floor
        face_ids = sorted({
            face_id for edge in loop_edges for face_id in segment_faces[edge]
        })
        lineage = _section_loop_lineage(
            face_ids, face_owner_ids, face_source_ids, face_generated_mask,
            lineage_available=lineage_available)
        loop_metrics.append({
            "passed": loop_passed,
            "vertex_count": len(loop),
            "area_m2": area,
            "minimum_feret_width_m": width,
            "area_mm2": area * 1e6,
            "minimum_feret_width_mm": width * 1e3,
            "coordinate_dtype": str(points.dtype),
            "coordinate_ulp_tolerance_m": tolerance,
            "positive_measure_floor_m": positive_floor,
            "reason": None if loop_passed
                else "cross-section area or minimum Feret width collapsed",
            **lineage,
        })
    if len(loops) != 1:
        return {
            "passed": False, "loop_count": len(loops),
            "reason": "section must have exactly one closed loop",
            "loop_metrics": loop_metrics,
            "segment_count": len(segments), **details,
        }
    return {
        **loop_metrics[0],
        "passed": loop_metrics[0]["passed"],
        "loop_count": 1,
        "loop_metrics": loop_metrics,
        "segment_count": len(segments),
        **details,
    }


def _source_surface_section(surface, station):
    return _section_loops(
        surface.positions, surface.faces,
        station["section_center_world_m"], station["tangent_world_unit"],
        **_source_section_lineage(surface))


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


def _section_metric_for_owner(section, owner_id, *, label):
    if (section.get("loop_count") != 1 or section.get("passed") is not True
            or len(section.get("loop_metrics", [])) != 1):
        return None, f"{label} must have one positive closed section loop"
    metric = section["loop_metrics"][0]
    if (metric.get("passed") is not True
            or metric.get("lineage_available") is not True
            or metric.get("lineage_complete") is not True
            or metric.get("owner_ids") != [owner_id]
            or not metric.get("source_face_ancestor_ids")
            or metric.get("generated_face_count") != 0
            or metric.get("unmapped_face_count") != 0):
        return None, f"{label} source-face lineage is incomplete or belongs to another owner"
    return metric, None


def _two_loop_metrics_by_owner(section, expected_owner_ids, *, label):
    if (section.get("loop_count") != 2
            or section.get("reason") != "section must have exactly one closed loop"
            or len(section.get("loop_metrics", [])) != 2):
        return None, f"{label} must contain exactly two closed loops at this junction station"
    by_owner = {}
    for metric in section["loop_metrics"]:
        if (metric.get("passed") is not True
                or metric.get("lineage_available") is not True
                or metric.get("lineage_complete") is not True
                or len(metric.get("owner_ids", [])) != 1
                or not metric.get("source_face_ancestor_ids")
                or metric.get("generated_face_count") != 0
                or metric.get("unmapped_face_count") != 0):
            return None, f"{label} has a collapsed, generated, mixed, or unmapped loop"
        owner_id = int(metric["owner_ids"][0])
        if owner_id not in expected_owner_ids or owner_id in by_owner:
            return None, f"{label} loop ownership is not a one-to-one vessel partition"
        by_owner[owner_id] = metric
    if set(by_owner) != set(expected_owner_ids):
        return None, f"{label} source-face lineage omits a declared vessel owner"
    return by_owner, None


def _same_section_plane(first, second):
    first_center = np.asarray(first.get("section_center_world_m"), dtype=np.float64)
    second_center = np.asarray(second.get("section_center_world_m"), dtype=np.float64)
    first_tangent = np.asarray(first.get("section_tangent_world_unit"), dtype=np.float64)
    second_tangent = np.asarray(second.get("section_tangent_world_unit"), dtype=np.float64)
    return bool(
        first_center.shape == (3,) and second_center.shape == (3,)
        and first_tangent.shape == (3,) and second_tangent.shape == (3,)
        and np.isfinite(first_center).all() and np.isfinite(second_center).all()
        and np.isfinite(first_tangent).all() and np.isfinite(second_tangent).all()
        and np.array_equal(first_center, second_center)
        and np.allclose(first_tangent, second_tangent, rtol=0.0, atol=2e-15))


def _junction_station_evidence(declaration, expected_owner_ids,
                               candidate_union_section, baseline_union_section,
                               candidate_branch_sections, baseline_branch_sections):
    issues = []
    if not isinstance(declaration, Mapping):
        return {
            "declared": False,
            "status": "MISSING_EXPLICIT_JUNCTION_STATION_DECLARATION",
            "rejections": ["two-loop exception needs an explicit source station declaration"],
        }
    declaration_id = declaration.get("declaration_id")
    role = declaration.get("station_role")
    declared_owners = declaration.get("expected_owner_ids")
    anchor_owner = declaration.get("anchor_owner_id")
    ancestry = declaration.get("source_ancestry")
    if not isinstance(declaration_id, str) or not declaration_id.strip():
        issues.append("junction station declaration_id must be a nonempty string")
    if role not in ("bridge", "target"):
        issues.append("junction station role must be bridge or target")
    if (not isinstance(declared_owners, (list, tuple))
            or not all(type(value) is int for value in declared_owners)
            or tuple(sorted(declared_owners)) != tuple(expected_owner_ids)
            or len(set(declared_owners)) != 2):
        issues.append("junction station declaration must name exactly both stable owners")
    bridge_owner = declaration.get("bridge_owner_id")
    target_owner = declaration.get("target_owner_id")
    if (bridge_owner not in expected_owner_ids or target_owner not in expected_owner_ids
            or bridge_owner == target_owner
            or {bridge_owner, target_owner} != set(expected_owner_ids)):
        issues.append("junction station declaration must identify both branch roles")
    if anchor_owner not in expected_owner_ids or anchor_owner != (
            bridge_owner if role == "bridge" else
            target_owner if role == "target" else None):
        issues.append("junction station anchor must identify its declared source branch")
    ancestry_keys = (
        "material_anchor_source_vertex_ids",
        "tangent_support_source_vertex_ids",
        "frozen_collar_face_ids",
    )
    if not isinstance(ancestry, Mapping):
        issues.append("junction station declaration needs frozen material source ancestry")
        ancestry = {}
    normalized_ancestry = {}
    for key in ancestry_keys:
        values = ancestry.get(key)
        if (not isinstance(values, (list, tuple)) or not values
                or not all(type(value) is int and value >= 0 for value in values)):
            issues.append(f"junction station source ancestry {key} must be nonempty nonnegative IDs")
            normalized_ancestry[key] = []
        else:
            normalized_ancestry[key] = [int(value) for value in values]

    station_evidence = {}
    for station_name, section, branch_sections in (
        ("candidate", candidate_union_section, candidate_branch_sections),
        ("baseline", baseline_union_section, baseline_branch_sections),
    ):
        spec = declaration.get(f"{station_name}_station")
        if not isinstance(spec, Mapping):
            issues.append(f"{station_name} junction station material spec is missing")
            continue
        for key in ancestry_keys:
            station_key = key
            value = spec.get(station_key)
            if value != normalized_ancestry[key]:
                issues.append(
                    f"{station_name} junction station {key} differs from frozen ancestry")
        expected_branch_owner = anchor_owner
        branch_section = branch_sections.get(expected_branch_owner)
        if branch_section is None or not _same_section_plane(section, branch_section):
            issues.append(
                f"{station_name} selected junction plane is not shared by its anchor branch section")
        center = np.asarray(spec.get("section_center_world_m"), dtype=np.float64)
        tangent = np.asarray(spec.get("tangent_world_unit"), dtype=np.float64)
        if (center.shape != (3,) or tangent.shape != (3,)
                or not np.isfinite(center).all() or not np.isfinite(tangent).all()
                or not np.array_equal(center, section.get("section_center_world_m"))
                or not np.allclose(tangent, section.get("section_tangent_world_unit"),
                                   rtol=0.0, atol=2e-15)):
            issues.append(f"{station_name} junction plane differs from its material station")
        station_evidence[station_name] = {
            "section_center_world_m": center.tolist() if center.shape == (3,) else None,
            "section_center_sha256": hashlib.sha256(
                np.ascontiguousarray(center, dtype="<f8").tobytes()).hexdigest()
                if center.shape == (3,) else None,
            "tangent_world_unit": tangent.tolist() if tangent.shape == (3,) else None,
            "tangent_sha256": hashlib.sha256(
                np.ascontiguousarray(tangent, dtype="<f8").tobytes()).hexdigest()
                if tangent.shape == (3,) else None,
        }
    return {
        "declared": not issues,
        "status": ("EXPLICIT_MATERIAL_JUNCTION_STATION_BOUND"
                   if not issues else "REJECT_JUNCTION_STATION_DECLARATION"),
        "declaration_id": declaration_id,
        "station_role": role,
        "expected_owner_ids": declared_owners,
        "bridge_owner_id": bridge_owner,
        "target_owner_id": target_owner,
        "anchor_owner_id": anchor_owner,
        "source_ancestry": normalized_ancestry,
        "candidate_station": station_evidence.get("candidate"),
        "baseline_station": station_evidence.get("baseline"),
        "rejections": issues,
    }


def _junction_aware_two_loop_section_proof(
    *,
    expected_owner_ids,
    candidate_union_section,
    baseline_union_section,
    candidate_branch_sections,
    baseline_branch_sections,
    solid_evidence,
    station_declaration=None,
):
    """Verify one explicit two-vessel Y-junction section without generalizing it.

    A pair of disjoint contours on one selected plane is accepted only if their
    exact CSG face lineage partitions them one-to-one across the declared
    bridge and target, each branch has one closed source section at the same
    plane, the CSG unions remain single numerical components, and every
    candidate/baseline branch and lobe retains the declared local calibre.
    """
    expected_owner_ids = tuple(sorted(set(map(int, expected_owner_ids))))
    issues = []
    comparisons = {}
    station_evidence = _junction_station_evidence(
        station_declaration, expected_owner_ids,
        candidate_union_section, baseline_union_section,
        candidate_branch_sections, baseline_branch_sections)
    if not station_evidence["declared"]:
        issues.extend(station_evidence["rejections"])
    if len(expected_owner_ids) != 2 or any(owner <= 0 for owner in expected_owner_ids):
        issues.append("junction proof requires exactly two distinct positive stable owners")

    candidate_lobes, error = _two_loop_metrics_by_owner(
        candidate_union_section, expected_owner_ids, label="candidate union section")
    if error:
        issues.append(error)
        candidate_lobes = {}
    baseline_lobes, error = _two_loop_metrics_by_owner(
        baseline_union_section, expected_owner_ids, label="baseline union section")
    if error:
        issues.append(error)
        baseline_lobes = {}

    candidate_branches = {}
    baseline_branches = {}
    for owner_id in expected_owner_ids:
        candidate_section = candidate_branch_sections.get(owner_id)
        baseline_section = baseline_branch_sections.get(owner_id)
        if candidate_section is None:
            issues.append(f"candidate source section is missing for stable owner {owner_id}")
        else:
            metric, error = _section_metric_for_owner(
                candidate_section, owner_id,
                label=f"candidate source section {owner_id}")
            if error:
                issues.append(error)
            elif metric is not None:
                candidate_branches[owner_id] = metric
        if baseline_section is None:
            issues.append(f"baseline source section is missing for stable owner {owner_id}")
        else:
            metric, error = _section_metric_for_owner(
                baseline_section, owner_id,
                label=f"baseline source section {owner_id}")
            if error:
                issues.append(error)
            elif metric is not None:
                baseline_branches[owner_id] = metric

    for owner_id, section in candidate_branch_sections.items():
        if not _same_section_plane(candidate_union_section, section):
            issues.append(
                f"candidate branch {owner_id} section is not on the selected union plane")
    for owner_id, section in baseline_branch_sections.items():
        if not _same_section_plane(baseline_union_section, section):
            issues.append(
                f"baseline branch {owner_id} section is not on the selected union plane")

    required_positive_counts = (
        "candidate_port_exact_interface_point_count",
        "baseline_port_exact_interface_point_count",
    )
    for key in required_positive_counts:
        if type(solid_evidence.get(key)) is not int or solid_evidence[key] <= 0:
            issues.append(f"{key} must be positive")
    for key in ("candidate_intersection_volume_m3", "baseline_intersection_volume_m3",
                "candidate_union_volume_m3", "baseline_union_volume_m3"):
        value = solid_evidence.get(key)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0.0:
            issues.append(f"{key} must be strictly positive")
    for key in ("candidate_exact_inside_both_witness",
                "baseline_exact_inside_both_witness"):
        witness = solid_evidence.get(key)
        if (not isinstance(witness, Mapping)
                or witness.get("first_location") != "inside"
                or witness.get("second_location") != "inside"):
            issues.append(f"{key} must place one exact witness inside both branch sources")
    for key in ("candidate_union_component_count", "baseline_union_component_count"):
        if solid_evidence.get(key) != 1:
            issues.append(f"{key} must equal one")
    for key in ("candidate_port_audit_passed", "baseline_port_audit_passed"):
        if solid_evidence.get(key) is not True:
            issues.append(f"{key} must preserve the frozen exact face-patch interface")
    for key in ("candidate_union_status", "baseline_union_status",
                "candidate_intersection_status", "baseline_intersection_status",
                "candidate_bridge_status", "candidate_target_status",
                "baseline_bridge_status", "baseline_target_status"):
        value = solid_evidence.get(key)
        if not isinstance(value, str) or not value.endswith("NoError"):
            issues.append(f"{key} must report the numerical Manifold NoError status")
    for key in ("candidate_branch_component_counts",
                "baseline_branch_component_counts"):
        counts = solid_evidence.get(key)
        if (not isinstance(counts, Mapping)
                or {int(owner): count for owner, count in counts.items()}
                != {owner_id: 1 for owner_id in expected_owner_ids}):
            issues.append(f"{key} must show one connected closed source component per vessel")

    if (set(candidate_lobes) == set(expected_owner_ids)
            and set(candidate_branches) == set(expected_owner_ids)):
        for owner_id in expected_owner_ids:
            comparison = _section_comparison(
                candidate_lobes[owner_id], candidate_branches[owner_id])
            comparisons.setdefault(str(owner_id), {})[
                "candidate_union_lobe_vs_candidate_source_section"] = comparison
            if not comparison["passed"]:
                issues.append(
                    f"candidate union lobe differs from vessel {owner_id} source section calibre")
    if (set(baseline_lobes) == set(expected_owner_ids)
            and set(baseline_branches) == set(expected_owner_ids)):
        for owner_id in expected_owner_ids:
            comparison = _section_comparison(
                baseline_lobes[owner_id], baseline_branches[owner_id])
            comparisons.setdefault(str(owner_id), {})[
                "baseline_union_lobe_vs_baseline_source_section"] = comparison
            if not comparison["passed"]:
                issues.append(
                    f"baseline union lobe differs from vessel {owner_id} source section calibre")
    if set(candidate_lobes) == set(expected_owner_ids) and \
            set(baseline_lobes) == set(expected_owner_ids):
        for owner_id in expected_owner_ids:
            comparison = _section_comparison(
                candidate_lobes[owner_id], baseline_lobes[owner_id])
            comparisons.setdefault(str(owner_id), {})[
                "candidate_union_lobe_vs_baseline_union_lobe"] = comparison
            if not comparison["passed"]:
                issues.append(
                    f"candidate union lobe calibre changed for vessel {owner_id}")
    if set(candidate_branches) == set(expected_owner_ids) and \
            set(baseline_branches) == set(expected_owner_ids):
        for owner_id in expected_owner_ids:
            comparison = _section_comparison(
                candidate_branches[owner_id], baseline_branches[owner_id])
            comparisons.setdefault(str(owner_id), {})[
                "candidate_source_section_vs_baseline_source_section"] = comparison
            if not comparison["passed"]:
                issues.append(
                    f"candidate source-section calibre changed for vessel {owner_id}")

    return {
        "passed": not issues,
        "status": ("PASS_EXPLICIT_TWO_VESSEL_JUNCTION_SECTION"
                   if not issues else "REJECT_EXPLICIT_TWO_VESSEL_JUNCTION_SECTION"),
        "method": "per-face Mesh64 run_original_id/face_id ancestry plus one-loop source sections, single-component union, positive volumetric intersection, exact inside-both witness, exact interface witness, and per-vessel area/minimum-Feret comparisons",
        "scope": "only the declared bridge-target pair and this one selected junction station; does not relax the general exactly-one-loop section rule",
        "expected_owner_ids": list(expected_owner_ids),
        "junction_station": station_evidence,
        "candidate_union_section_loop_count": candidate_union_section.get("loop_count"),
        "baseline_union_section_loop_count": baseline_union_section.get("loop_count"),
        "candidate_union_loops_by_owner": {
            str(owner): metric for owner, metric in sorted(candidate_lobes.items())},
        "baseline_union_loops_by_owner": {
            str(owner): metric for owner, metric in sorted(baseline_lobes.items())},
        "candidate_source_sections_by_owner": {
            str(owner): metric for owner, metric in sorted(candidate_branches.items())},
        "baseline_source_sections_by_owner": {
            str(owner): metric for owner, metric in sorted(baseline_branches.items())},
        "solid_evidence": dict(solid_evidence),
        "per_vessel_section_comparisons": comparisons,
        "rejections": issues,
        "floating_csg_topology_limit": "Manifold NoError and decompose() are numerical CSG diagnostics, not exact Boolean topology proof",
        "calibre_scope": "source-geometry consistency against the same compiled branch lineage, not measured or clinical calibre",
    }


def audit_brachiocephalic_port_pair_continuity(
    verifier, manifold3d, *, bridge_key, bridge_surface, target_key, target_surface,
    frozen_face_pairs, baseline_bridge_surface, baseline_target_surface,
    bridge_station=None, target_station=None, bridge_route_anchor_world_m=None,
    target_route_anchor_world_m=None,
    junction_station_declarations=None,
):
    """Check one declared bridge port as closed solids and local section loops.

    This is a numerical geometry gate only. It does not establish measured
    physiology, native acceptance, or clinical calibre.
    """
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    if junction_station_declarations is None:
        junction_station_declarations = {}
    _require(isinstance(junction_station_declarations, Mapping),
             "junction station declarations must be a source-bound mapping")
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
    exact_points = _exact_interface_points(
        verifier, bridge, target, observed)
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
        candidate_section = _source_surface_section(surface, spec)
        baseline_spec = _transport_material_station(base_surface, spec)
        baseline_section = _source_surface_section(base_surface, baseline_spec)
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
    union_lineage = _mesh_face_lineage(
        union_mesh, {
            bridge_key[1]: len(bridge.quotient_faces),
            target_key[1]: len(target.quotient_faces),
        })
    union_candidate_section = _section_loops(
        union_positions, union_faces,
        bridge_station["section_center_world_m"], bridge_station["tangent_world_unit"],
        derived_csg=True,
        face_owner_ids=union_lineage["face_owner_ids"],
        face_source_ids=union_lineage["source_face_ids"],
        face_generated_mask=union_lineage["generated_face_mask"])
    base_first, _ = _make_manifold(base_bridge, manifold3d)
    base_second, _ = _make_manifold(base_target, manifold3d)
    base_first_status, base_second_status = str(base_first.status()), str(base_second.status())
    base_union = manifold3d.Manifold.batch_boolean(
        [base_first, base_second], manifold3d.OpType.Add)
    base_union_status = str(base_union.status())
    base_union_components = len(base_union.decompose())
    base_union_volume = float(base_union.volume())
    base_union_positions, base_union_faces, base_union_mesh = _manifold_mesh(base_union)
    base_union_lineage = _mesh_face_lineage(
        base_union_mesh, {
            bridge_key[1]: len(base_bridge.quotient_faces),
            target_key[1]: len(base_target.quotient_faces),
        })
    base_bridge_station = _transport_material_station(base_bridge, bridge_station)
    base_target_station = _transport_material_station(base_target, target_station)
    baseline_union_bridge_section = _section_loops(
        base_union_positions, base_union_faces,
        base_bridge_station["section_center_world_m"],
        base_bridge_station["tangent_world_unit"], derived_csg=True,
        face_owner_ids=base_union_lineage["face_owner_ids"],
        face_source_ids=base_union_lineage["source_face_ids"],
        face_generated_mask=base_union_lineage["generated_face_mask"])
    union_target_section = _section_loops(
        union_positions, union_faces,
        target_station["section_center_world_m"], target_station["tangent_world_unit"],
        derived_csg=True,
        face_owner_ids=union_lineage["face_owner_ids"],
        face_source_ids=union_lineage["source_face_ids"],
        face_generated_mask=union_lineage["generated_face_mask"])
    baseline_union_target_section = _section_loops(
        base_union_positions, base_union_faces,
        base_target_station["section_center_world_m"],
        base_target_station["tangent_world_unit"], derived_csg=True,
        face_owner_ids=base_union_lineage["face_owner_ids"],
        face_source_ids=base_union_lineage["source_face_ids"],
        face_generated_mask=base_union_lineage["generated_face_mask"])

    candidate_branch_components = {
        bridge_key[1]: len(first.decompose()),
        target_key[1]: len(second.decompose()),
    }
    baseline_branch_components = {
        bridge_key[1]: len(base_first.decompose()),
        target_key[1]: len(base_second.decompose()),
    }
    junction_evidence_cache = None

    def junction_solid_evidence():
        nonlocal junction_evidence_cache
        if junction_evidence_cache is not None:
            return junction_evidence_cache
        base_intersection = manifold3d.Manifold.batch_boolean(
            [base_first, base_second], manifold3d.OpType.Intersect)
        base_intersection_status = str(base_intersection.status())
        base_intersection_volume = float(base_intersection.volume())
        base_intersection_witness = (
            _exact_inside_witness(verifier, base_bridge, base_target, base_intersection)
            if base_intersection_volume > 0.0 else None)
        base_port_audit = verifier.audit_surface_pair(
            base_bridge, base_target, intended_face_pairs=allowed)
        base_observed = tuple(tuple(map(int, pair)) for pair in
                              base_port_audit.get("declared_interface_triangle_pairs", []))
        base_exact_points = _exact_interface_points(
            verifier, base_bridge, base_target, base_observed)
        junction_evidence_cache = {
            "candidate_bridge_status": first_status,
            "candidate_target_status": second_status,
            "baseline_bridge_status": base_first_status,
            "baseline_target_status": base_second_status,
            "candidate_union_status": union_status,
            "baseline_union_status": base_union_status,
            "candidate_intersection_status": intersection_status,
            "baseline_intersection_status": base_intersection_status,
            "candidate_branch_component_counts": {
                str(owner): count for owner, count in candidate_branch_components.items()},
            "baseline_branch_component_counts": {
                str(owner): count for owner, count in baseline_branch_components.items()},
            "candidate_union_component_count": union_components,
            "baseline_union_component_count": base_union_components,
            "candidate_union_volume_m3": union_volume,
            "baseline_union_volume_m3": base_union_volume,
            "candidate_intersection_volume_m3": intersection_volume,
            "baseline_intersection_volume_m3": base_intersection_volume,
            "candidate_exact_inside_both_witness": intersection_witness,
            "baseline_exact_inside_both_witness": base_intersection_witness,
            "candidate_port_audit_passed": bool(port_audit.get("passed")),
            "baseline_port_audit_passed": bool(base_port_audit.get("passed")),
            "candidate_port_exact_interface_point_count": len(exact_points),
            "baseline_port_exact_interface_point_count": len(base_exact_points),
            "baseline_observed_port_face_pairs": [list(pair) for pair in base_observed],
            "baseline_exact_interface_points_world_m": base_exact_points,
        }
        return junction_evidence_cache

    def compare_union_station(label, station_role, candidate_section, baseline_section,
                              candidate_station, baseline_station):
        if (candidate_section.get("loop_count") == 1
                and baseline_section.get("loop_count") == 1
                and candidate_section.get("passed") is True
                and baseline_section.get("passed") is True):
            comparison = _section_comparison(candidate_section, baseline_section)
            _require(comparison["passed"],
                     f"{label} calibre ratio is outside the declared local screen")
            return comparison
        if (candidate_section.get("loop_count") != 2
                or baseline_section.get("loop_count") != 2):
            _require(False,
                     f"{label} requires matching single loops or the explicit two-vessel junction proof")
        candidate_branch_sections = {
            bridge_key[1]: _source_surface_section(bridge, candidate_station),
            target_key[1]: _source_surface_section(target, candidate_station),
        }
        baseline_branch_sections = {
            bridge_key[1]: _source_surface_section(base_bridge, baseline_station),
            target_key[1]: _source_surface_section(base_target, baseline_station),
        }
        proof = _junction_aware_two_loop_section_proof(
            expected_owner_ids=(bridge_key[1], target_key[1]),
            candidate_union_section=candidate_section,
            baseline_union_section=baseline_section,
            candidate_branch_sections=candidate_branch_sections,
            baseline_branch_sections=baseline_branch_sections,
            solid_evidence=junction_solid_evidence(),
            station_declaration={
                **dict(junction_station_declarations[station_role]),
                "candidate_station": candidate_station,
                "baseline_station": baseline_station,
            } if isinstance(junction_station_declarations.get(station_role), Mapping)
            else None,
        )
        _require(proof["passed"],
                 f"{label} junction-aware section proof rejected: "
                 + "; ".join(proof["rejections"][:5]))
        return {
            "passed": True,
            "method": "explicit one-loop-per-vessel Y-junction proof",
            "junction_aware_proof": proof,
        }

    station_results["union_at_bridge_station"] = {
        "station": bridge_station,
        "candidate": union_candidate_section,
        "baseline_station_transported_by_ancestry": base_bridge_station,
        "baseline": baseline_union_bridge_section,
        "candidate_vs_baseline": compare_union_station(
            "union section at bridge station", "bridge", union_candidate_section,
            baseline_union_bridge_section, bridge_station, base_bridge_station),
    }
    station_results["union_at_target_station"] = {
        "station": target_station,
        "baseline_station_transported_by_ancestry": base_target_station,
        "candidate": union_target_section,
        "baseline": baseline_union_target_section,
        "candidate_vs_baseline": compare_union_station(
            "union section at target station", "target", union_target_section,
            baseline_union_target_section, target_station, base_target_station),
    }
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
            "union_generated_face_count": (
                union_lineage["report"]["generated_face_count"]
                if union_mesh is not None else 0),
            "union_section_mesh_position_dtype": str(union_positions.dtype),
            "union_lineage_original_ids": (
                union_lineage["report"]["run_original_ids"]
                if union_mesh is not None else []),
            "union_face_lineage": union_lineage["report"],
            "candidate_branch_component_counts": {
                str(owner): count for owner, count in candidate_branch_components.items()},
            "baseline_union_status": base_union_status,
            "baseline_union_volume_m3": base_union_volume,
            "baseline_union_component_count": base_union_components,
            "baseline_union_face_lineage": base_union_lineage["report"],
        },
        "calibre_gate": {
            "passed": True,
            "method": "strict one-loop area and minimum-Feret comparison, with a narrowly declared two-loop fallback requiring exact per-face vessel lineage and independent one-loop source sections for both branches",
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
    junction_station_declarations=None,
):
    """Run source-ordered bridge port checks and stop at the first failure."""
    station_specs = {} if station_specs is None else station_specs
    route_anchors = {} if route_anchors is None else route_anchors
    junction_station_declarations = (
        {} if junction_station_declarations is None else junction_station_declarations)
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
                junction_station_declarations=junction_station_declarations.get(pair, {}),
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
