import hashlib

import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as av
from numilab_human import resting_brachiocephalic_3d_junction as junction3d
from numilab_human import resting_brachiocephalic_geometry_gate as bg
from numilab_human import resting_brachiocephalic_port_diagnostic as portdiag
from numilab_human import resting_brachiocephalic_source007 as source007
from numilab_human.resting_brachiocephalic_geometry_gate import (
    BRIDGE_KEYS,
    PORT_TARGET_IDS,
    _manifold_mesh,
    _section_loops,
    audit_brachiocephalic_port_pair_continuity,
    audit_brachiocephalic_six_port_continuity,
    audit_captured_pose_surface_intersections,
)


def _tetra(apex_z, offset_x=0.0):
    points = np.asarray([
        [offset_x, 0, 0], [offset_x + 1, 0, 0],
        [offset_x, 1, 0], [offset_x, 0, apex_z],
    ], dtype="<f4")
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    return points, faces


def _pack_surface(points, faces):
    return (points, faces, None, np.arange(len(points), dtype=np.uint64))


def _inputs():
    left, left_faces = _tetra(1.0)
    right, right_faces = _tetra(1.0, 10.0)
    targets = {}
    left_svc, svc_faces = _tetra(-1.0)
    right_svc, right_svc_faces = _tetra(-1.0, 10.0)
    svc_positions = np.concatenate((left_svc, right_svc), axis=0)
    svc_faces = np.concatenate((svc_faces, right_svc_faces + 4), axis=0)
    targets[(51011, 10)] = _pack_surface(svc_positions, svc_faces)
    for stable_id in (469, 473):
        targets[(51011, stable_id)] = _pack_surface(*_tetra(-1.0))
    for stable_id in (468, 472):
        targets[(51011, stable_id)] = _pack_surface(*_tetra(-1.0, 10.0))
    targets[(51005, 900)] = _pack_surface(*_tetra(1.0, 100.0))
    positions = {BRIDGE_KEYS[0]: left, BRIDGE_KEYS[1]: right}
    faces = {BRIDGE_KEYS[0]: left_faces, BRIDGE_KEYS[1]: right_faces}
    all_pairs = [[a, b] for a in range(4) for b in range(4)]
    ports = {}
    for stable_id in PORT_TARGET_IDS[525]:
        ports[(BRIDGE_KEYS[0], (51011, stable_id))] = all_pairs
    for stable_id in PORT_TARGET_IDS[526]:
        target_pairs = ([[a, b + 4] for a in range(4) for b in range(4)]
                        if stable_id == 10 else all_pairs)
        ports[(BRIDGE_KEYS[1], (51011, stable_id))] = target_pairs
    return targets, positions, faces, ports


def test_all_six_declared_ports_and_all_decoded_targets_are_checked_but_not_admitted():
    targets, positions, faces, ports = _inputs()
    result = audit_captured_pose_surface_intersections(
        av, targets, positions, faces, ports)

    assert result["surface_intersection_stage_passed"]
    assert result["status"].endswith("CONTINUITY_UNRESOLVED")
    assert len(result["captured_target_keys"]) == 6
    assert [51005, 900] in result["captured_target_keys"]
    assert len(result["verifier_result"]["rows"]) >= 2
    assert result["limits"]["lumen_continuity"] == "not_evaluated"
    assert result["limits"]["calibre_continuity"] == "not_evaluated"
    assert result["limits"]["anatomical_admission"] is False


def test_missing_port_pair_witness_fails_closed():
    targets, positions, faces, ports = _inputs()
    del ports[(BRIDGE_KEYS[0], (51011, 469))]
    with pytest.raises(ValueError, match="missing frozen face-pair witness"):
        audit_captured_pose_surface_intersections(av, targets, positions, faces, ports)


def test_pose_pair_locus_may_change_inside_frozen_patch_cross_product():
    targets, positions, faces, ports = _inputs()
    pair_key = (BRIDGE_KEYS[0], (51011, 469))
    ports[pair_key] = ports[pair_key][:-1]

    result = audit_captured_pose_surface_intersections(
        av, targets, positions, faces, ports)

    assert result["surface_intersection_stage_passed"]
    observation = next(row for row in result["port_observations"]
                       if row["target"] == [51011, 469]
                       and row["bridge"] == [51011, 525])
    assert observation["observed_pose_pairs_within_frozen_patch"]
    assert not observation["observed_exactly_matches_frozen_pairs"]
    assert not result["port_pair_mismatches"]


def test_pose_pair_outside_frozen_patch_cross_product_rejects():
    targets, positions, faces, ports = _inputs()
    pair_key = (BRIDGE_KEYS[0], (51011, 469))
    ports[pair_key] = [[0, 0]]

    result = audit_captured_pose_surface_intersections(
        av, targets, positions, faces, ports)

    assert not result["surface_intersection_stage_passed"]
    assert any(row["target"] == [51011, 469]
               and row["unexpected_actual_face_pairs_outside_frozen_patch"]
               for row in result["port_pair_mismatches"])


def test_input_geometry_is_not_mutated():
    targets, positions, faces, ports = _inputs()
    before = {key: value[0].tobytes() for key, value in targets.items()}
    before.update({key: value.tobytes() for key, value in positions.items()})

    audit_captured_pose_surface_intersections(av, targets, positions, faces, ports)

    after = {key: value[0].tobytes() for key, value in targets.items()}
    after.update({key: value.tobytes() for key, value in positions.items()})
    assert after == before


def _closed_cylinder(z_min, z_max, radius=0.2, *, center_x=0.0, sectors=12, axial=16):
    rings = []
    for z in np.linspace(z_min, z_max, axial + 1):
        for sector in range(sectors):
            angle = 2.0 * np.pi * sector / sectors
            rings.append([center_x + radius * np.cos(angle),
                          radius * np.sin(angle), z])
    bottom_center = len(rings)
    rings.append([center_x, 0.0, z_min])
    top_center = len(rings)
    rings.append([center_x, 0.0, z_max])
    faces = []
    for level in range(axial):
        for sector in range(sectors):
            following = (sector + 1) % sectors
            a = level * sectors + sector
            b = level * sectors + following
            c = (level + 1) * sectors + following
            d = (level + 1) * sectors + sector
            faces.extend(([a, b, c], [a, c, d]))
    for sector in range(sectors):
        following = (sector + 1) % sectors
        faces.extend(([bottom_center, following, sector],
                      [top_center, axial * sectors + sector,
                       axial * sectors + following]))
    return (np.asarray(rings, dtype="<f4"),
            np.asarray(faces, dtype="<u4"))


def _profiled_closed_cylinder(stations, *, sectors=12):
    rings = []
    for z, radius in stations:
        for sector in range(sectors):
            angle = 2.0 * np.pi * sector / sectors
            rings.append([radius * np.cos(angle), radius * np.sin(angle), z])
    bottom_center = len(rings)
    rings.append([0.0, 0.0, stations[0][0]])
    top_center = len(rings)
    rings.append([0.0, 0.0, stations[-1][0]])
    faces = []
    for level in range(len(stations) - 1):
        for sector in range(sectors):
            following = (sector + 1) % sectors
            a = level * sectors + sector
            b = level * sectors + following
            c = (level + 1) * sectors + following
            d = (level + 1) * sectors + sector
            faces.extend(([a, b, c], [a, c, d]))
    for sector in range(sectors):
        following = (sector + 1) % sectors
        faces.extend(([bottom_center, following, sector],
                      [top_center, (len(stations) - 1) * sectors + sector,
                       (len(stations) - 1) * sectors + following]))
    return (np.asarray(rings, dtype="<f4"),
            np.asarray(faces, dtype="<u4"))


def _manifold3d():
    import importlib
    import sys
    try:
        return importlib.import_module("manifold3d")
    except ImportError:
        from pathlib import Path
        dependency = Path(
            "/Users/n/numi-human-retained-delivery-20261009/"
            "anatomy-completion-1276/retained-cpu-reference-deps-1384/"
            "manifold3d-3.5.4")
        if not dependency.is_dir():
            pytest.skip("pinned Manifold3D CPU reference dependency is unavailable")
        sys.path.insert(0, str(dependency))
        return importlib.import_module("manifold3d")


def _port_pairs(bridge, bridge_faces, target, target_faces):
    first = av.prepare_surface((51011, 525), bridge, bridge_faces)
    second = av.prepare_surface((51011, 10), target, target_faces)
    return av.audit_surface_pair(first, second)["unexpected_triangle_pairs"]


def _source_y_target(*, radius=0.2, radial_ring_scales=None):
    positions, faces = _closed_cylinder(
        -1.0, 1.0, radius=radius, sectors=12, axial=8)
    if radial_ring_scales is not None:
        for level, scale in radial_ring_scales.items():
            start, end = level * 12, (level + 1) * 12
            positions[start:end, :2] *= np.float32(scale)
    angle = np.arctan(0.7)
    cosine, sine = np.cos(angle), np.sin(angle)
    local_x = positions[:, 0].copy()
    local_z = positions[:, 2].copy()
    positions[:, 0] = cosine * local_x + sine * local_z + np.float32(0.55)
    positions[:, 2] = -sine * local_x + cosine * local_z
    return np.ascontiguousarray(positions, dtype="<f4"), faces


def _run_cylinder_port(candidate_target, baseline_target, baseline_bridge=None):
    """Exercise the already-scoped 526/472 matched-collar 3D owner."""
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    bridge_baseline = bridge if baseline_bridge is None else baseline_bridge[0]
    bridge_baseline_faces = bridge_faces if baseline_bridge is None else baseline_bridge[1]
    target_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)[1]
    bridge_source = av.prepare_surface(
        (51011, 526), bridge_baseline, bridge_baseline_faces,
        source_vertex_ids=np.arange(len(bridge_baseline), dtype=np.uint64))
    target_source = av.prepare_surface(
        (51011, 472), baseline_target, target_faces,
        source_vertex_ids=np.arange(len(baseline_target), dtype=np.uint64))
    # The frozen patch is derived only from the registered baseline source pair;
    # candidate contacts must remain inside that collar cross-product.
    pairs = av.audit_surface_pair(bridge_source, target_source)[
        "unexpected_triangle_pairs"]
    return junction3d.audit_brachiocephalic_3d_junction(
        av, manifold3d,
        bridge_key=(51011, 526), bridge_surface=(bridge, bridge_faces),
        target_key=(51011, 472), target_surface=(candidate_target, target_faces),
        frozen_face_pairs=pairs,
        baseline_bridge_surface=(bridge_baseline, bridge_baseline_faces),
        baseline_target_surface=(baseline_target, target_faces))


def test_overlapping_closed_cylinders_pass_solid_continuity_and_calibre_screen():
    target, _ = _source_y_target()
    report = _run_cylinder_port(target, target)

    assert report["passed"]
    solid = report["solid_gates"]["candidate"]
    assert solid["positive_intersection_volume"]
    assert solid["intersection_volume_m3"] > 0.0
    assert solid["exact_inside_both_witness"]["first_location"] == "inside"
    assert solid["exact_inside_both_witness"]["second_location"] == "inside"
    assert solid["union_component_count"] == 1
    assert solid["union_connected_numerical_solid"]
    assert solid["csg_analysis_precision"].startswith("Manifold Mesh64 Float64")
    assert solid["floating_boolean_exact_topology_proven"] is False
    for role in ("bridge", "target"):
        calibre = report["independent_collar_calibre"][role]
        assert calibre["passed"]
        assert calibre["candidate_to_baseline_comparison"]["passed"]
        assert calibre["declared_area_ratio"] == [0.8, 1.25]
        assert calibre["declared_minimum_feret_ratio"] == [0.9, 1.1]


def test_declared_partial_collar_continues_only_after_solid_and_calibre_proofs():
    bridge, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    target, target_faces = _profiled_closed_cylinder(
        ((0.8, 0.18), (1.0, 0.2), (1.05, 0.2), (2.0, 0.18)))
    pairs = _port_pairs(bridge, bridge_faces, target, target_faces)

    report = audit_brachiocephalic_port_pair_continuity(
        av, _manifold3d(),
        bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
        target_key=(51011, 10), target_surface=(target, target_faces),
        frozen_face_pairs=pairs,
        baseline_bridge_surface=(bridge, bridge_faces),
        baseline_target_surface=(target, target_faces))

    assert report["passed"]
    boundary = report["port_pair_gate"]["declared_partial_overlap_boundary_only"]
    assert report["port_pair_gate"]["central_p6_audit_passed"] is False
    assert boundary["passed"]
    assert boundary["requires_separate_solid_and_calibre_proofs"]
    assert report["solid_gate"]["passed"]
    assert report["solid_gate"]["strict_positive_volume"]
    assert report["solid_gate"]["exact_inside_both_witness"] is not None
    assert report["solid_gate"]["union_component_count"] == 1
    assert report["calibre_gate"]["passed"]


def test_partial_overlap_cannot_use_unrelated_typed_surfaces_or_unmatched_pairs():
    bridge, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    target, target_faces = _profiled_closed_cylinder(
        ((0.8, 0.18), (1.0, 0.2), (1.05, 0.2), (2.0, 0.18)))
    prepared_bridge = av.prepare_surface((51011, 525), bridge, bridge_faces)
    prepared_target = av.prepare_surface((51011, 10), target, target_faces)
    raw = av.audit_surface_pair(prepared_bridge, prepared_target)
    pairs = raw["unexpected_triangle_pairs"]
    bridge_patch = {first for first, _ in pairs}
    target_patch = {second for _, second in pairs}
    allowed = {(first, second) for first in bridge_patch for second in target_patch}
    audit = av.audit_surface_pair(
        prepared_bridge, prepared_target, intended_face_pairs=allowed)
    observed = audit["declared_interface_triangle_pairs"]
    exact_points = bg._exact_interface_points(
        av, prepared_bridge, prepared_target, observed)

    valid = bg._declared_partial_vessel_port_overlap(
        (51011, 525), (51011, 10), audit, observed, (), exact_points, allowed)
    unrelated = bg._declared_partial_vessel_port_overlap(
        (51005, 525), (51005, 10), audit, observed, (), exact_points, allowed)
    outside = bg._declared_partial_vessel_port_overlap(
        (51011, 525), (51011, 10), audit, observed,
        ((max(first for first, _ in pairs) + 1, max(second for _, second in pairs) + 1),),
        exact_points, allowed)
    wrong_identity = bg._declared_partial_vessel_port_overlap(
        (51011, 525), (51011, 10),
        {**audit, "identities": [[51011, 526], [51011, 472]]},
        observed, (), exact_points, allowed)
    malformed = bg._declared_partial_vessel_port_overlap(
        (51011, 525), (51011, 10), {**audit, "containment": None},
        observed, (), exact_points, allowed)

    assert valid["passed"]
    assert not unrelated["passed"]
    assert not unrelated["conditions"]["declared_typed_vessel_port"]
    assert not outside["passed"]
    assert not outside["conditions"]["no_unexpected_pairs_outside_frozen_patch"]
    assert not wrong_identity["passed"]
    assert not wrong_identity["conditions"]["surface_pair_audit_identity_matches"]
    assert not malformed["passed"]
    assert not bg._declared_partial_vessel_port_overlap(
        (51011, 525), (51011, 10), audit, observed, (), (), allowed)["passed"]


def test_tangent_closed_cylinders_reject_zero_volume_contact():
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0)
    target, target_faces = _closed_cylinder(1.0, 2.0)
    pairs = _port_pairs(bridge, bridge_faces, target, target_faces)

    with pytest.raises(ValueError, match="overlap, exact inside witness, or closed connected union"):
        audit_brachiocephalic_port_pair_continuity(
            av, manifold3d,
            bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
            target_key=(51011, 10), target_surface=(target, target_faces),
            frozen_face_pairs=pairs,
            baseline_bridge_surface=(bridge, bridge_faces),
            baseline_target_surface=(target, target_faces))


def test_disconnected_closed_cylinders_reject_missing_port_contact():
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0)
    target, target_faces = _closed_cylinder(1.1, 2.1)
    tangent_target, tangent_faces = _closed_cylinder(1.0, 2.0)
    pairs = _port_pairs(bridge, bridge_faces, tangent_target, tangent_faces)

    with pytest.raises(ValueError, match="lacks a frozen-patch exact interface witness"):
        audit_brachiocephalic_port_pair_continuity(
            av, manifold3d,
            bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
            target_key=(51011, 10), target_surface=(target, target_faces),
            frozen_face_pairs=pairs,
            baseline_bridge_surface=(bridge, bridge_faces),
            baseline_target_surface=(target, target_faces))


def test_pinch_beyond_declared_local_calibre_range_rejects():
    baseline_target, target_faces = _source_y_target()
    bridge, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    bridge_source = av.prepare_surface((51011, 526), bridge, bridge_faces)
    target_source = av.prepare_surface((51011, 472), baseline_target, target_faces)
    frozen_pairs = av.audit_surface_pair(bridge_source, target_source)[
        "unexpected_triangle_pairs"]
    frozen_target_faces = sorted({int(pair[1]) for pair in frozen_pairs})
    collar_vertices = set(map(int, np.unique(target_faces[frozen_target_faces])))

    # Keep every source vertex used by the frozen exact interface unchanged.
    # The first material section just beyond that collar is locally pinched.
    candidate_local, _ = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    for vertex_id in range(36, 48):
        if vertex_id not in collar_vertices:
            candidate_local[vertex_id, :2] *= np.float32(0.75)
    for level in range(4, 9):
        candidate_local[level * 12:(level + 1) * 12, :2] *= np.float32(0.8)
    # Apply the local source-profile edit before the registered rigid map.
    angle = np.arctan(0.7)
    cosine, sine = np.cos(angle), np.sin(angle)
    local_x = candidate_local[:, 0].copy()
    local_z = candidate_local[:, 2].copy()
    candidate_local[:, 0] = cosine * local_x + sine * local_z + np.float32(0.55)
    candidate_local[:, 2] = -sine * local_x + cosine * local_z
    candidate_target = np.ascontiguousarray(candidate_local, dtype="<f4")

    report = _run_cylinder_port(candidate_target, baseline_target)
    target_calibre = report["independent_collar_calibre"]["target"]
    assert not report["passed"]
    assert not target_calibre["passed"]
    ratios = target_calibre["candidate_to_baseline_comparison"][
        "candidate_to_baseline_ratios"]
    assert ratios["area"] < 0.8 or ratios["minimum_feret_width"] < 0.9
    assert target_calibre["declared_area_ratio"] == [0.8, 1.25]
    assert target_calibre["declared_minimum_feret_ratio"] == [0.9, 1.1]


def test_baseline_section_is_transported_with_source_vertex_ancestry():
    baseline_target, _ = _source_y_target()
    candidate_target = np.ascontiguousarray(
        baseline_target + np.asarray([0.005, 0.0, 0.0], dtype="<f4"), dtype="<f4")
    report = _run_cylinder_port(candidate_target, baseline_target)
    calibre = report["independent_collar_calibre"]["target"]
    candidate_station = np.asarray(calibre["station"]["section_center_world_m"])
    baseline_station = np.asarray(
        calibre["baseline_station_transported_by_source_ancestry"]
        ["section_center_world_m"])

    assert report["passed"]
    assert np.linalg.norm(candidate_station - baseline_station) > 0.004
    assert calibre["candidate_to_baseline_comparison"]["passed"]
    assert calibre["station"]["material_anchor_source_vertex_ids"] == (
        calibre["baseline_station_transported_by_source_ancestry"]
        ["material_anchor_source_vertex_ids"])



def _closed_extruded_polygon_section(points_xy):
    points_xy = np.asarray(points_xy, dtype="<f4")
    count = len(points_xy)
    bottom = np.column_stack((points_xy, np.full(count, -1.0, dtype="<f4")))
    top = np.column_stack((points_xy, np.full(count, 1.0, dtype="<f4")))
    positions = np.concatenate((bottom, top), axis=0)
    faces = []
    for index in range(count):
        following = (index + 1) % count
        a, b = index, following
        c, d = following + count, index + count
        faces.extend(([a, b, c], [a, c, d]))
    bottom_center = len(positions)
    top_center = bottom_center + 1
    positions = np.concatenate((
        positions,
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype="<f4"),
    ), axis=0)
    for index in range(count):
        following = (index + 1) % count
        faces.extend(([bottom_center, following, index],
                      [top_center, count + index, count + following]))
    return positions, np.asarray(faces, dtype="<u4")


def test_section_topology_keeps_distinct_sub_tolerance_hits_in_open_triangle():
    positions = np.asarray(
        [[0.0, 0.0, 3.2e-7], [0.0, 0.0, -1.0], [0.2, 0.0, -1.0]],
        dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")

    report = _section_loops(
        positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    assert not report["passed"]
    assert report["reason"] == "section graph is open or branched"
    assert report["segment_count"] == 1
    assert report["section_node_count"] == 2
    assert report["section_edge_intersection_count"] == 2
    assert report["minimum_section_segment_length_m"] < report["coordinate_ulp_tolerance_m"]


def test_section_topology_keeps_distinct_nearby_edge_hits_and_tiny_valid_segment():
    delta = float(np.float32(-1.0 + 1.0e-7) - np.float32(-1.0))
    polygon = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1 + 1.0e-7)]
    positions, faces = _closed_extruded_polygon_section(polygon)

    report = _section_loops(
        positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    assert report["passed"]
    assert report["section_node_count"] == 2 * len(polygon)
    assert report["section_edge_intersection_count"] == 2 * len(polygon)
    assert report["minimum_section_segment_length_m"] == pytest.approx(delta / 2.0, abs=1e-12)
    assert report["minimum_section_segment_length_m"] < report["coordinate_ulp_tolerance_m"]
    assert report["minimum_feret_width_m"] > 1.0


def test_section_topology_quotients_exact_coordinate_seam_duplicates():
    positions, faces = _closed_cylinder(-1.0, 1.0, sectors=8, axial=4)
    duplicate_id = len(positions)
    duplicate_source_id = 0
    positions = np.concatenate((positions, positions[duplicate_source_id:duplicate_source_id + 1]))
    faces = faces.copy()
    # Rewire a single topological incident face to an exactly equal seam copy.
    incident = int(np.flatnonzero(np.any(faces == duplicate_source_id, axis=1))[0])
    faces[incident, np.flatnonzero(faces[incident] == duplicate_source_id)[0]] = duplicate_id

    report = _section_loops(
        positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    assert report["passed"]
    assert report["exact_coordinate_duplicate_vertex_count"] == 1
    assert report["coplanar_edge_count"] == 8


def test_section_topology_accepts_coplanar_ring_edges_crossed_from_both_sides():
    positions, faces = _closed_cylinder(-1.0, 1.0, sectors=8, axial=4)

    report = _section_loops(
        positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

    assert report["passed"]
    assert report["coplanar_edge_count"] == 8
    assert report["loop_count"] == 1


def test_section_topology_reports_coplanar_face_and_tangent_degeneracies():
    coplanar_positions = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
    tetra_faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    coplanar = _section_loops(
        coplanar_positions, tetra_faces, [0, 0, 0], [0, 0, 1])
    assert not coplanar["passed"]
    assert coplanar["reason"] == "section plane contains a coplanar source face"
    assert coplanar["coplanar_face_count"] == 1

    tangent_vertex_positions = np.asarray(
        [[0, 0, 0], [1, 0, -1], [0, 1, -1], [0, 0, -1]], dtype="<f4")
    tangent_vertex = _section_loops(
        tangent_vertex_positions, tetra_faces, [0, 0, 0], [0, 0, 1])
    assert not tangent_vertex["passed"]
    assert tangent_vertex["reason"] == "section plane touches an isolated tangent source vertex"
    assert tangent_vertex["isolated_tangent_vertex_ids"] == [0]

    tangent_edge_positions = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, -1], [0, 0, -1]], dtype="<f4")
    tangent_edge = _section_loops(
        tangent_edge_positions, tetra_faces, [0, 0, 0], [0, 0, 1])
    assert not tangent_edge["passed"]
    assert tangent_edge["reason"] == "section plane is tangent along a coplanar source edge"
    assert tangent_edge["incident_side_signs"] == [-1, -1]


def test_section_topology_is_translation_stable_for_binary_float32_offsets():
    positions, faces = _closed_cylinder(-1.0, 1.0, sectors=8, axial=8)
    offset = np.asarray([2.0 ** -10, -(2.0 ** -11), 2.0 ** -10], dtype="<f4")
    translated = np.ascontiguousarray(positions + offset, dtype="<f4")

    report = _section_loops(
        translated, faces, [float(offset[0]), float(offset[1]), float(offset[2])],
        [0.0, 0.0, 1.0])

    assert report["passed"]
    assert report["loop_count"] == 1
    assert report["section_degree_histogram"] == {"2": report["section_node_count"]}


def test_derived_boolean_sections_use_mesh64_without_weakening_source_float32_contract():
    manifold3d = _manifold3d()
    assert hasattr(manifold3d, "Mesh64")
    assert hasattr(manifold3d.Manifold, "to_mesh64")
    solid = manifold3d.Manifold.cylinder(0.2, 1.0, 1.0, 16)
    positions, faces, mesh = _manifold_mesh(solid)

    assert isinstance(mesh, manifold3d.Mesh64)
    assert positions.dtype == np.dtype("<f8")
    assert _section_loops(
        positions, faces, np.asarray([0.0, 0.0, 0.1]),
        np.asarray([0.0, 0.0, 1.0]), derived_csg=True)["passed"]
    with pytest.raises(ValueError, match="emitted Float32 section positions"):
        _section_loops(
            positions, faces, np.asarray([0.0, 0.0, 0.1]),
            np.asarray([0.0, 0.0, 1.0]))




def _synthetic_two_branch_section_proof(
    *, disconnected=False, corrupt_lineage=False, declared=True,
    include_internal=False):
    manifold3d = _manifold3d()
    bridge_positions, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    baseline_target_positions, target_faces = _source_y_target(radius=0.2)
    candidate_target_positions = baseline_target_positions.copy()
    if disconnected:
        candidate_target_positions[:, 0] += np.float32(0.45)
    bridge_positions = np.ascontiguousarray(bridge_positions, dtype="<f4")
    baseline_target_positions = np.ascontiguousarray(
        baseline_target_positions, dtype="<f4")
    candidate_target_positions = np.ascontiguousarray(
        candidate_target_positions, dtype="<f4")
    bridge = av.prepare_surface(
        (51011, 526), bridge_positions, bridge_faces,
        source_vertex_ids=np.arange(len(bridge_positions), dtype=np.uint64))
    target = av.prepare_surface(
        (51011, 472), candidate_target_positions, target_faces,
        source_vertex_ids=np.arange(len(candidate_target_positions), dtype=np.uint64))
    baseline_target = av.prepare_surface(
        (51011, 472), baseline_target_positions, target_faces,
        source_vertex_ids=np.arange(len(baseline_target_positions), dtype=np.uint64))

    def solids(second):
        bridge_solid, _ = bg._make_manifold(bridge, manifold3d)
        second_solid, _ = bg._make_manifold(second, manifold3d)
        intersection = manifold3d.Manifold.batch_boolean(
            [bridge_solid, second_solid], manifold3d.OpType.Intersect)
        union = manifold3d.Manifold.batch_boolean(
            [bridge_solid, second_solid], manifold3d.OpType.Add)
        positions, faces, mesh = bg._manifold_mesh(union)
        lineage = bg._mesh_face_lineage(mesh, {
            526: len(bridge.quotient_faces), 472: len(second.quotient_faces)})
        owners = lineage["face_owner_ids"].copy()
        if corrupt_lineage:
            owners[owners == 472] = 526
        section = bg._section_loops(
            positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            derived_csg=True,
            face_owner_ids=owners,
            face_source_ids=lineage["source_face_ids"],
            face_generated_mask=lineage["generated_face_mask"],
        )
        inside = (bg._exact_inside_witness(av, bridge, second, intersection)
                  if float(intersection.volume()) > 0.0 else None)
        return bridge_solid, second_solid, intersection, union, section, inside

    candidate = solids(target)
    baseline = solids(baseline_target)
    raw_port = av.audit_surface_pair(bridge, baseline_target)
    frozen_pairs = raw_port.get("unexpected_triangle_pairs", [])
    baseline_interface = junction3d._interface_gate(
        av, bridge, baseline_target, frozen_pairs)
    candidate_interface = junction3d._interface_gate(
        av, bridge, target, frozen_pairs)
    bridge_patch = sorted({int(pair[0]) for pair in frozen_pairs})
    target_patch = sorted({int(pair[1]) for pair in frozen_pairs})

    def source_station(surface, collar_faces, *, level=4):
        source_ids = surface.source_vertex_ids.astype(int)
        ring_start = level * 12
        return {
            "section_center_world_m": [0.0, 0.0, 0.0],
            "tangent_world_unit": [0.0, 0.0, 1.0],
            "material_anchor_source_vertex_ids": source_ids[
                ring_start:ring_start + 12].tolist(),
            "tangent_support_source_vertex_ids": [
                int(source_ids[0]), int(source_ids[96])],
            "frozen_collar_face_ids": sorted(set(map(int, collar_faces))),
        }

    bridge_station = source_station(bridge, bridge_patch)
    candidate_target_station = source_station(target, target_patch)
    baseline_target_station = source_station(baseline_target, target_patch)
    candidate_branch_sections = {
        526: bg._source_surface_section(bridge, bridge_station),
        472: bg._source_surface_section(target, candidate_target_station),
    }
    baseline_branch_sections = {
        526: bg._source_surface_section(bridge, bridge_station),
        472: bg._source_surface_section(baseline_target, baseline_target_station),
    }

    def status(solid):
        return str(solid.status())

    def component_count(solid):
        return len(solid.decompose())

    def volume(solid):
        return float(solid.volume())

    candidate_bridge_solid, candidate_target_solid, candidate_intersection, candidate_union, candidate_section, candidate_inside = candidate
    baseline_bridge_solid, baseline_target_solid, baseline_intersection, baseline_union, baseline_section, baseline_inside = baseline
    evidence = {
        "candidate_bridge_status": status(candidate_bridge_solid),
        "candidate_target_status": status(candidate_target_solid),
        "baseline_bridge_status": status(baseline_bridge_solid),
        "baseline_target_status": status(baseline_target_solid),
        "candidate_union_status": status(candidate_union),
        "baseline_union_status": status(baseline_union),
        "candidate_intersection_status": status(candidate_intersection),
        "baseline_intersection_status": status(baseline_intersection),
        "candidate_branch_component_counts": {
            "526": component_count(candidate_bridge_solid),
            "472": component_count(candidate_target_solid),
        },
        "baseline_branch_component_counts": {
            "526": component_count(baseline_bridge_solid),
            "472": component_count(baseline_target_solid),
        },
        "candidate_union_component_count": component_count(candidate_union),
        "baseline_union_component_count": component_count(baseline_union),
        "candidate_union_volume_m3": volume(candidate_union),
        "baseline_union_volume_m3": volume(baseline_union),
        "candidate_intersection_volume_m3": volume(candidate_intersection),
        "baseline_intersection_volume_m3": volume(baseline_intersection),
        "candidate_exact_inside_both_witness": candidate_inside,
        "baseline_exact_inside_both_witness": baseline_inside,
        "candidate_port_audit_passed": bool(candidate_interface.get("passed")),
        "baseline_port_audit_passed": bool(baseline_interface.get("passed")),
        "candidate_port_exact_interface_point_count": len(
            candidate_interface.get("exact_interface_points_world_m", [])),
        "baseline_port_exact_interface_point_count": len(
            baseline_interface.get("exact_interface_points_world_m", [])),
    }
    station_ancestry = {
        key: bridge_station[key] for key in (
            "material_anchor_source_vertex_ids",
            "tangent_support_source_vertex_ids",
            "frozen_collar_face_ids",
        )
    }
    station_declaration = ({
        "declaration_id": "synthetic-source-bound-y-junction-526-472",
        "station_role": "bridge",
        "expected_owner_ids": [472, 526],
        "bridge_owner_id": 526,
        "target_owner_id": 472,
        "anchor_owner_id": 526,
        "source_ancestry": station_ancestry,
        "candidate_station": bridge_station,
        "baseline_station": bridge_station,
    } if declared else None)
    junction_call = {
        "expected_owner_ids": (526, 472),
        "candidate_union_section": candidate_section,
        "baseline_union_section": baseline_section,
        "candidate_branch_sections": candidate_branch_sections,
        "baseline_branch_sections": baseline_branch_sections,
        "solid_evidence": evidence,
        "station_declaration": station_declaration,
    }
    proof = bg._junction_aware_two_loop_section_proof(**junction_call)
    if include_internal:
        internal = {
            "section_positions": candidate[4].get("_positions"),
            "section_faces": candidate[4].get("_faces"),
            "section_center": [0.0, 0.0, 0.0],
            "section_tangent": [0.0, 0.0, 1.0],
            "section_options": {},
            "junction_call": junction_call,
        }
        # Keep the actual mesh inputs for the recording-wrapper regression.
        candidate_lineage_mesh = bg._manifold_mesh(candidate_union)[2]
        internal_positions, internal_faces, _ = bg._manifold_mesh(candidate_union)
        candidate_lineage = bg._mesh_face_lineage(candidate_lineage_mesh, {
            526: len(bridge.quotient_faces), 472: len(target.quotient_faces)})
        internal["section_positions"] = internal_positions
        internal["section_faces"] = internal_faces
        internal["section_options"] = {
            "derived_csg": True,
            "face_owner_ids": candidate_lineage["face_owner_ids"].copy(),
            "face_source_ids": candidate_lineage["source_face_ids"],
            "face_generated_mask": candidate_lineage["generated_face_mask"],
        }
        return proof, candidate_section, evidence, internal
    return proof, candidate_section, evidence

def test_target_collar_station_candidates_use_target_source_ancestry_without_moving_mesh():
    from types import SimpleNamespace

    positions, faces = _closed_cylinder(-1.0, 1.0, sectors=12, axial=16)
    surface = SimpleNamespace(
        positions=positions,
        faces=faces,
        source_vertex_ids=np.arange(1000, 1000 + len(positions), dtype=np.uint64),
    )
    collar_faces = np.arange(24, dtype=np.int64)
    witness = np.asarray([
        [0.18, 0.0, -0.93], [0.0, 0.18, -0.93], [-0.18, 0.0, -0.93],
    ], dtype=np.float64)
    original_bytes = positions.tobytes()

    candidates = bg._target_collar_material_station_candidates(
        surface, witness, collar_faces)

    assert len(candidates) == 4
    assert [row["station_candidate_fraction_of_target_collar_extent"]
            for row in candidates] == [0.2, 0.4, 0.6, 0.8]
    assert all(row["station_plane_normal_is_target_source_tangent"] for row in candidates)
    assert all(not row["station_plane_inferred_from_bridge_tangent"] for row in candidates)
    assert all(row["mesh_coordinates_modified"] is False for row in candidates)
    assert all(row["station_scope"] == "TARGET_COLLAR_LOCAL_SWEEP_V1"
               for row in candidates)
    assert all(row["transported_from_source_vertex_ancestry"] for row in candidates)
    assert all(row["material_anchor_source_vertex_ids"]
               and row["tangent_support_source_vertex_ids"]
               and row["frozen_collar_face_ids"] == collar_faces.tolist()
               and row["target_collar_source_face_ids"] == collar_faces.tolist()
               and row["target_collar_source_face_ids_sha256"]
               for row in candidates)
    tangent = np.asarray(candidates[0]["tangent_world_unit"])
    assert tangent[2] > 0.99
    offsets = [row["station_offset_from_anchor_m"] for row in candidates]
    assert offsets == sorted(offsets)
    assert offsets[-1] < candidates[-1]["collar_projection_extent_m"]
    assert positions.tobytes() == original_bytes


def test_target_collar_station_candidate_selector_is_hard_bounded_at_four():
    from types import SimpleNamespace

    positions, faces = _closed_cylinder(-1.0, 1.0, sectors=12, axial=16)
    surface = SimpleNamespace(
        positions=positions,
        faces=faces,
        source_vertex_ids=np.arange(len(positions), dtype=np.uint64),
    )
    with pytest.raises(ValueError, match="one to four frozen fractions"):
        bg._target_collar_material_station_candidates(
            surface, [[0.0, 0.0, -1.0]], np.arange(24),
            fractions=(0.1, 0.2, 0.3, 0.4, 0.5))


def test_explicit_junction_proof_still_rejects_two_loops_from_one_source_vessel():
    proof, _, _, internal = _synthetic_two_branch_section_proof(include_internal=True)
    assert proof["passed"]
    call = dict(internal["junction_call"])
    candidate = dict(call["candidate_branch_sections"])
    baseline = dict(call["baseline_branch_sections"])
    for collection in (candidate, baseline):
        one_loop = collection[472]
        duplicate = dict(one_loop)
        duplicate["loop_count"] = 2
        duplicate["loop_metrics"] = [
            dict(one_loop["loop_metrics"][0]),
            dict(one_loop["loop_metrics"][0]),
        ]
        collection[472] = duplicate
    call["candidate_branch_sections"] = candidate
    call["baseline_branch_sections"] = baseline

    rejected = bg._junction_aware_two_loop_section_proof(**call)

    assert not rejected["passed"]
    assert any("candidate source section 472 must have one positive closed section loop"
               in item for item in rejected["rejections"])
    assert any("baseline source section 472 must have one positive closed section loop"
               in item for item in rejected["rejections"])


def test_y_junction_two_contours_pass_only_with_per_vessel_lineage_and_solid_witnesses():
    proof, section, evidence = _synthetic_two_branch_section_proof()

    assert section["loop_count"] == 2
    assert not section["passed"]
    assert {tuple(loop["owner_ids"]) for loop in section["loop_metrics"]} == {
        (472,), (526,)}
    assert all(loop["lineage_complete"] for loop in section["loop_metrics"])
    assert evidence["candidate_union_component_count"] == 1
    assert evidence["candidate_intersection_volume_m3"] > 0.0
    assert evidence["candidate_exact_inside_both_witness"] is not None
    assert proof["passed"]
    assert proof["status"] == "PASS_EXPLICIT_TWO_VESSEL_JUNCTION_SECTION"
    assert set(proof["per_vessel_section_comparisons"]) == {"472", "526"}


def test_recording_wrappers_forward_lineage_into_real_junction_section():
    from numilab_human.resting_brachiocephalic_port_diagnostic import Recorder

    _, _, _, internal = _synthetic_two_branch_section_proof(include_internal=True)
    recorder = Recorder()
    restore = portdiag.install_recording_wrappers(bg, av, recorder)
    try:
        section = bg._section_loops(
            internal["section_positions"],
            internal["section_faces"],
            internal["section_center"],
            internal["section_tangent"],
            **internal["section_options"])
        call = dict(internal["junction_call"])
        call["candidate_union_section"] = section
        proof = bg._junction_aware_two_loop_section_proof(**call)
    finally:
        restore()

    assert proof["passed"]
    section_record = recorder.data["section_calls"][0]
    assert section_record["section_metrics"]["loop_count"] == 2
    assert section_record["section_options"] == [
        "derived_csg", "face_generated_mask", "face_owner_ids", "face_source_ids"]
    for name in ("face_owner_ids", "face_source_ids", "face_generated_mask"):
        assert section_record[name + "_sha256"] == portdiag._array_hash(
            internal["section_options"][name])
        assert section_record[name + "_shape"] == [len(internal["section_faces"])]
    junction_record = recorder.data["junction_section_proofs"][0]
    assert junction_record["passed"] is True
    assert junction_record["proof"]["status"] == (
        "PASS_EXPLICIT_TWO_VESSEL_JUNCTION_SECTION")


def test_nearby_disconnected_section_loops_do_not_pass_as_a_junction():
    proof, section, evidence = _synthetic_two_branch_section_proof(disconnected=True)

    assert section["loop_count"] == 2
    assert not section["passed"]
    assert evidence["candidate_union_component_count"] == 2
    assert evidence["candidate_intersection_volume_m3"] == 0.0
    assert not proof["passed"]
    assert any("candidate_union_component_count" in item
               for item in proof["rejections"])
    assert any("candidate_intersection_volume_m3" in item
               for item in proof["rejections"])


def test_two_loop_junction_requires_an_explicit_source_material_station():
    proof, section, _ = _synthetic_two_branch_section_proof(declared=False)

    assert section["loop_count"] == 2
    assert not proof["passed"]
    assert proof["junction_station"]["status"] == (
        "MISSING_EXPLICIT_JUNCTION_STATION_DECLARATION")
    assert any("explicit source station declaration" in item
               for item in proof["rejections"])


def test_mixed_mesh64_loop_ancestry_rejects_the_junction_exception():
    proof, section, _ = _synthetic_two_branch_section_proof(corrupt_lineage=True)

    assert section["loop_count"] == 2
    assert not proof["passed"]
    assert any("ownership" in item or "lineage" in item
               for item in proof["rejections"])


def test_mesh64_face_lineage_requires_triangle_aligned_runs_and_explicit_generated_faces():
    class MeshLineage:
        tri_verts = np.asarray([[0, 1, 2], [2, 3, 0],
                                [4, 5, 6], [6, 7, 4]], dtype="<u8")
        run_index = (0, 6, 12)
        run_original_id = (526, 472)
        face_id = (0, 1, 0, np.iinfo(np.uint32).max)

    lineage = bg._mesh_face_lineage(
        MeshLineage(), {526: 2, 472: 1})

    assert lineage["face_owner_ids"].tolist() == [526, 526, 472, 472]
    assert lineage["source_face_ids"][:3].tolist() == [0, 1, 0]
    assert lineage["generated_face_mask"].tolist() == [False, False, False, True]
    assert lineage["report"]["generated_face_count"] == 1
    assert lineage["report"]["unmapped_face_count"] == 0

def test_six_port_continuity_stops_at_first_failed_port(monkeypatch):
    expected = [
        (BRIDGE_KEYS[0], (51011, 10)),
        (BRIDGE_KEYS[0], (51011, 469)),
        (BRIDGE_KEYS[0], (51011, 473)),
        (BRIDGE_KEYS[1], (51011, 10)),
        (BRIDGE_KEYS[1], (51011, 468)),
        (BRIDGE_KEYS[1], (51011, 472)),
    ]
    surfaces = {key: object() for pair in expected for key in pair}
    pairs = {pair: [[0, 0]] for pair in expected}
    visited = []

    def one_port(_verifier, _manifold, *, bridge_key, target_key, **_kwargs):
        pair = (bridge_key, target_key)
        visited.append(pair)
        return {
            "schema": "numi.human.brachiocephalic-port-solid-continuity.v1",
            "bridge": list(bridge_key),
            "target": list(target_key),
            "passed": pair != expected[1],
            "failure": None if pair != expected[1] else "synthetic first failure",
        }

    monkeypatch.setattr(bg, "audit_brachiocephalic_port_pair_continuity", one_port)
    result = audit_brachiocephalic_six_port_continuity(
        object(), object(), surfaces, surfaces, pairs)

    assert visited == expected[:2]
    assert not result["passed"]
    assert result["requested_port_count"] == 6
    assert result["evaluated_port_count"] == 2
    assert result["first_failed_port"] == {
        "bridge": list(expected[1][0]), "target": list(expected[1][1])}
    assert result["ports_not_evaluated_after_first_failure"] == [
        {"bridge": list(pair[0]), "target": list(pair[1])} for pair in expected[2:]]



def test_source007_pins_existing_rows_and_exact_six_port_order():
    assert source007.BRIDGE_KEYS == ((51011, 525), (51011, 526))
    assert source007.PORT_KEYS == (
        ((51011, 525), (51011, 10)),
        ((51011, 525), (51011, 469)),
        ((51011, 525), (51011, 473)),
        ((51011, 526), (51011, 10)),
        ((51011, 526), (51011, 468)),
        ((51011, 526), (51011, 472)),
    )
    assert source007.REQUIRED_SOURCE_TARGETS[0] == (51011, 7)
    assert source007.REQUIRED_SKIN_KEY == (51007, 1)
    assert source007.N27_RECEIPT_SHA == "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
    assert source007.N27_PAYLOAD_SHA == "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"


def test_source007_same_source_calibre_is_not_physical_claim():
    report = source007._decorate_same_source_calibre({
        "ports": [{"calibre_gate": {"passed": True}}]})
    assert "not independent" in report["calibre_interpretation"]
    assert report["ports"][0]["calibre_gate"]["same_source_geometry_sanity_only"]
    assert not report["ports"][0]["calibre_gate"]["independent_physical_calibration"]


def test_source007_step0_collision_geometry_uses_captured_svc_prediction():
    svc = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    svc_faces = np.asarray([[0, 1, 2]], dtype="<u4")
    captured_svc = svc + np.asarray([[0, 0, 0.2]] * 3, dtype="<f4")
    row10 = {"vertices6": np.column_stack([svc, np.zeros((3, 3), dtype="<f4")]),
             "faces": svc_faces}
    rows = {10: row10}
    rigid_bridges = {}
    preparation = {"bridges": {}}
    for stable_id, member in ((525, "FJ3482"), (526, "FJ3583")):
        bridge = np.asarray([[2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype="<f4")
        rows[stable_id] = {
            "vertices6": np.column_stack([bridge, np.zeros((3, 3), dtype="<f4")]),
            "faces": svc_faces,
        }
        rigid_bridges[(51011, stable_id)] = (
            bridge.copy(), svc_faces.copy(), None, np.arange(3, dtype=np.uint64))
        preparation["bridges"][member] = {"vertex_ancestry": [
            {"svc_local_vertices": [0, 1, 2], "barycentric_weights": [1, 0, 0],
             "coefficient_scale": 1.0},
            {"svc_local_vertices": [0, 1, 2], "barycentric_weights": [0, 1, 0],
             "coefficient_scale": 0.5},
            {"svc_local_vertices": [0, 1, 2], "barycentric_weights": [0, 0, 1],
             "coefficient_scale": 0.0, "distal_zero_field_vertex": True},
        ]}
    captured = {(51011, 10): (captured_svc, svc_faces, None,
                                np.arange(3, dtype=np.uint64))}
    body20_pose = {"quaternion_xyzw": [0, 0, 0, 1], "position_m": [0, 0, 0]}

    predicted, details = source007._predict_step0_bridges(
        rows, rigid_bridges, captured, body20_pose, preparation,
        lambda _q, _t, points: np.ascontiguousarray(points, dtype="<f4"))

    for stable_id, member in ((525, "FJ3482"), (526, "FJ3583")):
        positions = predicted[(51011, stable_id)][0]
        np.testing.assert_allclose(positions[:, 2], [0.2, 0.1, 0.0],
                                   rtol=0, atol=2e-7)
        np.testing.assert_array_equal(positions[2], rigid_bridges[(51011, stable_id)][0][2])
        assert details[member]["prediction_status"] == (
            "OFFLINE_CAPTURED_SVC_DISPLACEMENT_PREDICTION_ONLY")
        assert details[member]["native_field_error_bound"] == "not_available"
        assert details[member]["source_geometry_mutated"] is False


def test_port_replay_preflight_is_pinned_single_port_and_keeps_topology_metrics():
    assert portdiag.DIAGNOSTIC_DIR.name == "port-525-469-diagnostic-005"
    assert portdiag.PREFLIGHT_PATH.name == "port-525-469-diagnostic-preflight-005.json"
    assert portdiag.REPLAY_SCOPE == {
        "bridge": [51011, 525],
        "target": [51011, 469],
        "port_count": 1,
        "source_candidate_compiler_invoked": False,
        "all_target_checks_invoked": False,
        "stop_after_single_port_result": True,
    }
    assert portdiag.RUNTIME_CODE_EXPECTED_SHA256[
        "resting_brachiocephalic_geometry_gate.py"] == (
            "35d2df7a6f982c086c9442438f815fe39e91270d601ae92a9b71bd60a1365661")
    assert portdiag.SIX_JUNCTION_SOURCE_REPORT_SHA256 == (
        "40d01f68619153ef94f09e79bf83f160c2c83ed695899c1713693f9a2a88fe5f")
    assert portdiag.SIX_JUNCTION_SOURCE_PREFLIGHT_SHA256 == (
        "feedad4833ec03331554b0b733a1a9764ac13d556017a22302ba9293005f1995")
    station_source = portdiag.SIX_REPLAY_SCOPE["junction_station_source"]
    assert station_source["station_method"].startswith("inferred PCA")
    assert station_source["measured_station"] is False
    assert "captured-pose admission not evaluated" in station_source["source0_scope"]
    summary = portdiag._section_summary({
        "passed": False,
        "section_node_count": 39,
        "section_degree_histogram": {"1": 2, "2": 37},
        "loop_metrics": [{"owner_ids": [526], "source_face_ancestor_ids": [4, 7]}],
        "lineage_available": True,
        "lineage_complete": True,
        "owner_ids": [526],
        "source_face_ancestor_ids": [4, 7],
        "minimum_section_segment_length_m": 1.3e-7,
        "exact_coordinate_duplicate_vertex_count": 0,
        "coplanar_edge_count": 0,
    })
    assert summary["section_node_count"] == 39
    assert summary["section_degree_histogram"] == {"1": 2, "2": 37}
    assert summary["minimum_section_segment_length_m"] == 1.3e-7
    assert summary["exact_coordinate_duplicate_vertex_count"] == 0
    assert summary["coplanar_edge_count"] == 0
    assert summary["loop_metrics"] == [
        {"owner_ids": [526], "source_face_ancestor_ids": [4, 7]}]
    assert summary["lineage_complete"]


def test_six_port_diagnostic_scope_matches_source007_and_is_separate():
    assert portdiag.SIX_PORT_KEYS == source007.PORT_KEYS
    assert portdiag.SIX_REPLAY_SCOPE["port_count"] == 6
    assert portdiag.SIX_REPLAY_SCOPE["ports"] == [
        {"bridge": list(bridge), "target": list(target)}
        for bridge, target in source007.PORT_KEYS
    ]
    assert portdiag.SIX_REPLAY_SCOPE["source_candidate_compiler_invoked"] is False
    assert portdiag.SIX_REPLAY_SCOPE["all_nearby_collision_checks_invoked"] is False
    assert portdiag.SIX_REPLAY_SCOPE["stop_after_first_failed_port"] is True
    assert portdiag.SIX_DIAGNOSTIC_DIR.name == (
        "port-525-526-target-collar-diagnostic-004")
    assert portdiag.SIX_PREFLIGHT_PATH.name == (
        "port-525-526-target-collar-preflight-004.json")
    station_selector = portdiag.SIX_REPLAY_SCOPE["target_collar_station_selector"]
    assert station_selector["pair"] == {
        "bridge": [51011, 526], "target": [51011, 472]}
    assert station_selector["candidate_fractions"] == [0.2, 0.4, 0.6, 0.8]
    assert station_selector["maximum_candidates"] == 4
    assert station_selector["normal_from_bridge_tangent"] is False
    assert station_selector["post_failure_expansion"] is False
    assert station_selector["requires_one_positive_source_loop_per_owner_at_same_plane"]
    assert portdiag.SIX_REPLAY_SCOPE["retained_prior_attempt"]["report_sha256"] == (
        "c916c16230114ae679dc7d3145ea76b9433b223aefe38bd6472fe1adf601d054")
    assert portdiag.SIX_PREVIOUS_REPORT_SHA256 == (
        "5b0d95318a7eba859d316736a8e046320327209f98c7eb541f39212b55c5826a")
    assert portdiag.SIX_DIAGNOSTIC_DIR != portdiag.DIAGNOSTIC_DIR
    assert portdiag.SIX_PREFLIGHT_PATH != portdiag.PREFLIGHT_PATH
    policy = portdiag.SIX_REPLAY_SCOPE["junction_section_policy"]
    assert policy["ordinary_union_sections_require_exactly_one_closed_loop"]
    assert "one complete Mesh64 source-face lineage" in policy["two_loop_exception"]
    assert policy["mixed_generated_or_unmapped_loop_lineage"] == "reject"
    assert not policy["clinical_calibre_claim"]
    assert not policy["floating_boolean_exact_topology_claim"]


def test_port_recorder_resets_stage_indices_and_tags_each_port_event():
    recorder = portdiag.Recorder()
    left_bridge, left_target = source007.PORT_KEYS[0]
    left_failed_bridge, left_failed_target = source007.PORT_KEYS[1]
    recorder.set_port(left_bridge, left_target)
    assert recorder.next_index("section_calls") == 0
    assert recorder.next_index("section_calls") == 1
    recorder.record("port_results", {"passed": True})
    recorder.set_port(left_failed_bridge, left_failed_target)
    assert recorder.next_index("section_calls") == 0
    recorder.record("port_results", {"passed": False})
    assert recorder.data["port_results"][0]["port_identity"] == {
        "bridge": list(left_bridge), "target": list(left_target)}
    assert recorder.data["port_results"][1]["port_identity"] == {
        "bridge": list(left_failed_bridge), "target": list(left_failed_target)}


def test_recording_wrappers_restart_stage_names_for_each_frozen_port():
    from types import SimpleNamespace

    def section(_positions, _faces, _center, _tangent, *, derived_csg=False):
        return {"passed": True, "loop_count": 1, "derived_csg": derived_csg}

    gate = SimpleNamespace(
        _section_loops=section,
        _section_comparison=lambda candidate, baseline: {
            "passed": candidate["passed"] and baseline["passed"]},
        _validated_station_spec=lambda *args, **kwargs: {},
        _transport_material_station=lambda *args, **kwargs: {},
        _make_manifold=lambda *args, **kwargs: (object(), {}),
        _exact_inside_witness=lambda *args, **kwargs: {},
        audit_brachiocephalic_port_pair_continuity=None,
    )
    verifier = SimpleNamespace(audit_surface_pair=lambda *args, **kwargs: {})
    def one_port(_verifier, _manifold, *, bridge_key, target_key, **_kwargs):
        gate._section_loops(
            np.zeros((1, 3), dtype="<f4"),
            np.zeros((1, 3), dtype="<u4"),
            np.zeros(3), np.asarray([1.0, 0.0, 0.0]),
        )
        return {"bridge": list(bridge_key), "target": list(target_key),
                "passed": True}
    gate.audit_brachiocephalic_port_pair_continuity = one_port

    recorder = portdiag.Recorder()
    restore = portdiag.install_recording_wrappers(gate, verifier, recorder)
    try:
        for bridge_key, target_key in source007.PORT_KEYS[:2]:
            gate.audit_brachiocephalic_port_pair_continuity(
                verifier, object(), bridge_key=bridge_key, target_key=target_key)
    finally:
        restore()

    assert [entry["stage"] for entry in recorder.data["section_calls"]] == [
        "bridge_candidate", "bridge_candidate"]
    assert [entry["port_identity"] for entry in recorder.data["section_calls"]] == [
        {"bridge": list(pair[0]), "target": list(pair[1])}
        for pair in source007.PORT_KEYS[:2]
    ]
    assert len(recorder.data["port_results"]) == 2
    assert [entry["result"]["passed"] for entry in recorder.data["port_results"]] == [
        True, True]


def test_port_diagnostic_records_solid_boolean_metrics_before_return_or_error():
    from types import SimpleNamespace

    class FakeSolid:
        def status(self):
            return "NoError"

        def volume(self):
            return 0.25

        def decompose(self):
            return [object()]

        def num_tri(self):
            return 12

        def num_vert(self):
            return 8

        def genus(self):
            return 0

    class FakeManifold:
        def __new__(cls, _mesh):
            return FakeSolid()

        @staticmethod
        def batch_boolean(solids, operation):
            assert len(solids) == 2
            assert operation == "add"
            return FakeSolid()

    recorder = portdiag.Recorder()
    proxy = portdiag._recording_manifold_proxy(
        SimpleNamespace(Manifold=FakeManifold, Mesh=object, OpType=object), recorder)
    first = proxy.Manifold(object())
    result = proxy.Manifold.batch_boolean([first, FakeSolid()], "add")

    assert isinstance(result, FakeSolid)
    event = recorder.data["solid_boolean_operations"][0]
    assert event["input_count"] == 2
    assert event["operation"] == "add"
    assert event["inputs"][0]["volume_m3"] == 0.25
    assert event["inputs"][0]["component_count"] == 1
    assert event["result"]["status"] == "NoError"
    assert event["result"]["triangle_count"] == 12
    assert event["result"]["genus"] == 0


def test_port_diagnostic_wrappers_preserve_failed_section_and_comparison():
    from types import SimpleNamespace

    def section(_positions, _faces, _center, _tangent, *, derived_csg=False):
        return {"passed": False, "loop_count": 0, "reason": "section graph is open",
                "derived_csg": derived_csg}

    def compare(_candidate, _baseline):
        raise ValueError("candidate and baseline sections must have positive closed-loop measures")

    gate = SimpleNamespace(
        _section_loops=section,
        _section_comparison=compare,
        _validated_station_spec=lambda *args, **kwargs: {},
        _transport_material_station=lambda *args, **kwargs: {},
        _make_manifold=lambda *args, **kwargs: None,
        _exact_inside_witness=lambda *args, **kwargs: None,
        audit_brachiocephalic_port_pair_continuity=lambda *args, **kwargs: {},
    )
    verifier = SimpleNamespace(audit_surface_pair=lambda *args, **kwargs: {})
    recorder = portdiag.Recorder()
    restore = portdiag.install_recording_wrappers(gate, verifier, recorder)
    positions = np.asarray([[0, 0, -1], [1, 0, 1], [0, 1, 1]], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    try:
        metrics = gate._section_loops(
            positions, faces, [0, 0, 0], [0, 0, 1])
        with pytest.raises(ValueError, match="positive closed-loop measures"):
            gate._section_comparison(metrics, metrics)
    finally:
        restore()

    assert recorder.data["section_calls"][0]["stage"] == "bridge_candidate"
    assert recorder.data["section_calls"][0]["section_metrics"]["reason"] == "section graph is open"
    comparison = recorder.data["section_comparisons"][0]
    assert comparison["stage"] == "bridge_source_section"
    assert comparison["candidate_section"]["passed"] is False
    assert comparison["baseline_section"]["passed"] is False
    assert comparison["error"]["message"].endswith("positive closed-loop measures")

def test_material_station_transports_frozen_vertex_ancestry_across_a_pose():
    positions, faces = _closed_cylinder(
        -1.0, 1.0, radius=0.25, sectors=12, axial=8)
    source_ids = np.arange(len(positions), dtype=np.uint64)
    surface = av.prepare_surface(
        (51011, 526), np.ascontiguousarray(positions, dtype="<f4"),
        faces, source_vertex_ids=source_ids)
    collar_faces = [0, 1, 2, 3]
    original_station = bg._validated_station_spec(
        surface, None, witness_points=positions[:4],
        collar_face_ids=collar_faces)

    angle = 0.63
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])
    rotated_positions = np.ascontiguousarray(
        positions @ rotation.T, dtype="<f4")
    rotated = av.prepare_surface(
        (51011, 526), rotated_positions, faces,
        source_vertex_ids=source_ids)
    transported = bg._validated_station_spec(
        rotated, original_station,
        witness_points=rotated_positions[:4],
        collar_face_ids=[1, 3])

    for field in (
        "material_anchor_local_vertex_ids",
        "material_anchor_source_vertex_ids",
        "material_anchor_weights",
        "tangent_support_local_vertex_ids",
        "tangent_support_source_vertex_ids",
        "frozen_collar_face_ids",
    ):
        assert transported[field] == original_station[field]
    assert not np.allclose(
        transported["section_center_world_m"],
        original_station["section_center_world_m"])
    support = np.asarray(transported["tangent_support_local_vertex_ids"])
    expected_tangent = (
        rotated_positions[support[1]].astype(np.float64)
        - rotated_positions[support[0]].astype(np.float64))
    expected_tangent /= np.linalg.norm(expected_tangent)
    assert np.allclose(
        transported["tangent_world_unit"], expected_tangent,
        rtol=0.0, atol=2e-7)

    with pytest.raises(ValueError, match="leave the frozen interface patch"):
        bg._validated_station_spec(
            rotated, original_station,
            witness_points=rotated_positions[:4],
            collar_face_ids=[4])

    mismatched = dict(original_station)
    mismatched["material_anchor_source_vertex_ids"] = list(
        original_station["material_anchor_source_vertex_ids"])
    mismatched["material_anchor_source_vertex_ids"][0] += 1
    with pytest.raises(ValueError, match="anchor source vertex IDs"):
        bg._validated_station_spec(
            rotated, mismatched,
            witness_points=rotated_positions[:4],
            collar_face_ids=collar_faces)



def test_network_all_target_screen_uses_only_frozen_edges_and_role_aware_skin_enclosures():
    from types import SimpleNamespace

    keys = ((51011, 479), (51011, 487), (51011, 481), (51011, 489),
            (51007, 1), (51001, 20))
    surfaces = {key: _pack_surface(*_tetra(1.0, float(index * 3)))
                for index, key in enumerate(keys)}
    edges = (((51011, 479), (51011, 487)),
             ((51011, 481), (51011, 489)))
    source_pairs = {edge: [[0, 0], [1, 1]] for edge in edges}
    observed = {}

    def audit_changed_surfaces(received, *, changed, targets, interfaces,
                               required_closed, intended_enclosures):
        observed.update({
            "surface_keys": set(received),
            "changed": tuple(changed),
            "targets": tuple(targets),
            "interfaces": interfaces,
            "required_closed": set(required_closed),
            "intended_enclosures": tuple(intended_enclosures),
        })
        return {"passed": True, "rows": []}

    result = bg.audit_network_rows_against_all_targets(
        SimpleNamespace(audit_changed_surfaces=audit_changed_surfaces),
        surfaces, edges, source_pairs)

    assert result["passed"]
    assert result["anatomical_admission"] is False
    assert result["network_edge_count"] == 2
    assert result["network_surface_count"] == 4
    assert observed["changed"] == tuple(sorted(key for edge in edges for key in edge))
    assert observed["targets"] == tuple(sorted(surfaces))
    assert set(observed["required_closed"]) == {key for edge in edges for key in edge}
    assert observed["interfaces"][((51011, 479), (51011, 487))] == (
        (0, 0), (0, 1), (1, 0), (1, 1))
    assert observed["interfaces"][((51011, 487), (51011, 479))] == (
        (0, 0), (1, 0), (0, 1), (1, 1))
    assert observed["intended_enclosures"] == (
        ((51011, 479), (51007, 1)), ((51011, 481), (51007, 1)),
        ((51011, 487), (51007, 1)), ((51011, 489), (51007, 1)))
    assert ((51011, 469), (51011, 473)) not in observed["interfaces"]


def test_network_all_target_screen_rejects_missing_edge_ancestry_and_bad_face_ids():
    keys = ((51011, 479), (51011, 487))
    surfaces = {key: _pack_surface(*_tetra(1.0, float(index * 3)))
                for index, key in enumerate(keys)}
    edge = ((51011, 479), (51011, 487))
    verifier = type("Verifier", (), {
        "audit_changed_surfaces": staticmethod(
            lambda *args, **kwargs: {"passed": True, "rows": []})
    })()
    with pytest.raises(ValueError, match="cover precisely"):
        bg.audit_network_rows_against_all_targets(
            verifier, surfaces, [edge], {})
    with pytest.raises(ValueError, match="outside current topology"):
        bg.audit_network_rows_against_all_targets(
            verifier, surfaces, [edge], {edge: [[99, 0]]})


class _HandoffPreparedSurface:
    def __init__(self, key, positions, faces, source_vertex_ids):
        self.key = key
        self.positions = np.asarray(positions, dtype="<f4")
        self.faces = np.asarray(faces, dtype="<u4")
        self.source_vertex_ids = np.asarray(source_vertex_ids, dtype="<u8")


class _HandoffVerifier:
    PreparedSurface = _HandoffPreparedSurface

    def __init__(self, *, nonnetwork_pass=True, failed_network_pair=None):
        self.nonnetwork_pass = nonnetwork_pass
        self.failed_network_pair = failed_network_pair
        self.shared_calls = []
        self.prepare_calls = []
        self.pair_calls = []

    def audit_changed_surfaces(self, surfaces, *, changed, targets, interfaces,
                               required_closed, intended_enclosures):
        self.shared_calls.append({
            "changed": tuple(changed),
            "targets": tuple(targets),
            "interfaces": interfaces,
            "required_closed": tuple(required_closed),
            "intended_enclosures": tuple(intended_enclosures),
        })
        return {
            "passed": self.nonnetwork_pass,
            "failed_stage": None if self.nonnetwork_pass else "nearby_pair",
            "rows": [] if self.nonnetwork_pass else [{
                "stage": "nearby_pair",
                "identities": [list(changed[0]), [51005, 900]],
                "passed": False,
                "unexpected_triangle_pairs": [[0, 0]],
            }],
        }

    def prepare_surface(self, key, positions, faces, *, normals=None,
                        source_vertex_ids=None):
        self.prepare_calls.append(tuple(key))
        return _HandoffPreparedSurface(key, positions, faces, source_vertex_ids)

    def audit_surface_pair(self, first, second):
        pair = frozenset((tuple(first.key), tuple(second.key)))
        self.pair_calls.append(pair)
        passed = pair != self.failed_network_pair
        return {
            "passed": passed,
            "identities": [list(first.key), list(second.key)],
            "unexpected_triangle_pairs": [] if passed else [[0, 0]],
        }


def _handoff_hash(value, dtype):
    packed = np.ascontiguousarray(value, dtype=dtype)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _handoff_surface_row(key, surface, *, positions=None):
    points, faces, _normals, source_ids = surface
    if positions is not None:
        points = np.asarray(positions, dtype="<f4")
    return {
        "identity": list(key),
        "closed_self_free_single_component": True,
        "exact_self_audit_count": 0,
        "referenced_face_component_count": 1,
        "source_vertex_ids_sha256": _handoff_hash(source_ids, "<u8"),
        "prepared_surface": {
            "identity": list(key),
            "positions_sha256": _handoff_hash(points, "<f4"),
            "faces_sha256": _handoff_hash(faces, "<u4"),
            "source_vertex_ids_sha256": None,
        },
    }


def _handoff_solid(edge, surfaces, *, source_kind):
    inputs = {}
    for role, key in (("bridge", edge[0]), ("target", edge[1])):
        points, faces, _normals, source_ids = surfaces[key]
        inputs[f"{role}_manifold_input"] = {
            "source_position_sha256": _handoff_hash(points, "<f4"),
            "source_face_sha256": _handoff_hash(faces, "<u4"),
            "source_vertex_ancestry_sha256": _handoff_hash(source_ids, "<u8"),
        }
    return {
        "passed": True,
        "manifold_inputs_passed": True,
        "positive_intersection_volume": True,
        "intersection_volume_m3": 1e-8,
        "union_connected_numerical_solid": True,
        "union_component_count": 1,
        "union_lineage_has_both_sources_without_generated_or_unmapped_faces": True,
        "union_float64_mesh64_lineage": {
            "generated_face_count": 0,
            "unmapped_face_count": 0,
            "run_original_ids": [edge[0][1], edge[1][1]],
        },
        "floating_boolean_exact_topology_proven": False,
        "exact_inside_both_witness": {
            "first_location": "inside",
            "second_location": "inside",
            "predicate": "P6 exact rational point location on original Float32 surfaces",
        },
        "volume_accounting": {"passed": True},
        **inputs,
        "test_source_kind": source_kind,
    }


def _handoff_interface(edge, pair_sha, *, source_kind):
    return {
        "passed": True,
        "typed_port": [list(edge[0]), list(edge[1])],
        "observed_pairs_within_frozen_face_patch_cross_product": True,
        "unexpected_face_pairs": [],
        "exact_interface_point_count": 3,
        "exact_interface_points_sha256": "a" * 64,
        "frozen_face_pairs_sha256": pair_sha,
        "documented_vessel_junction_scope": {"passed": True},
        "baseline_verified_declared_source_additions": {
            "added_to_frozen_patch_cross_product": False,
            "candidate_and_baseline_still_require_exact_points": True,
            "pairs": [],
            "used": False,
        },
        "test_source_kind": source_kind,
    }


def _handoff_calibre(key, role):
    station = {
        "transported_from_source_vertex_ancestry": True,
        "exact_interface_witness_points_sha256": "b" * 64,
        "frozen_collar_face_ids": [0, 1],
    }
    baseline_station = {
        "transported_from_source_vertex_ancestry": True,
        "exact_interface_witness_points_sha256": "b" * 64,
        "frozen_collar_face_ids": [0, 1],
    }
    section = {
        "passed": True,
        "loop_count": 1,
        "lineage_complete": True,
        "generated_face_count": 0,
        "unmapped_face_count": 0,
        "area_m2": 2e-5,
        "minimum_feret_width_m": 0.004,
    }
    return {
        "passed": True,
        "owner_id": key[1],
        "role": role,
        "baseline_station_witness_point_count": 3,
        "baseline_interface_point_sha256": "b" * 64,
        "station": station,
        "baseline_station_transported_by_source_ancestry": baseline_station,
        "candidate_section": dict(section),
        "baseline_section": dict(section),
        "candidate_to_baseline_comparison": {
            "passed": True,
            "wide_bounds_are_not_preservation_success": True,
            "candidate_to_baseline_ratios": {
                "area": 1.0, "minimum_feret_width": 1.0},
            "declared_preservation_intervals": {
                "area": [0.8, 1.25],
                "minimum_feret_width": [0.9, 1.1],
            },
        },
    }


def _handoff_fixture():
    edges = (
        ((51011, 7), (51011, 471)),
        ((51011, 473), (51011, 477)),
    )
    all_network = tuple(key for edge in edges for key in edge)
    surfaces = {}
    for index, key in enumerate(all_network):
        points, faces = _tetra(1.0, float(index * 2))
        source_ids = np.arange(len(points), dtype="<u8") + np.uint64(index * 100)
        surfaces[key] = (points, faces, None, source_ids)
    surfaces[(51005, 900)] = _pack_surface(*_tetra(1.0, 30.0))
    surfaces[(51007, 1)] = _pack_surface(*_tetra(1.0, 40.0))

    links = []
    source_pairs = {}
    for edge in edges:
        pairs = ((0, 0), (1, 1))
        source_pairs[edge] = pairs
        pair_sha = _handoff_hash(pairs, "<i8")
        identities = {}
        for key in edge:
            identities[str(key[1])] = {
                "typed_identity": list(key),
                "canonical_source_identity": {
                    "stable_id": key[1],
                    "receipt_identity_matches_abi5_owner": True,
                },
            }
        ancestry = {
            "typed_link": [list(edge[0]), list(edge[1])],
            "source_local_face_indices_bound": True,
            "clinical_or_anatomical_admission": False,
            "source_evidence_path": "/frozen/source-witnesses.json",
            "source_evidence_sha256": "c" * 64,
            "source_parent_faces_sha256": _handoff_hash(
                surfaces[edge[0]][1], "<u4"),
            "source_child_faces_sha256": _handoff_hash(
                surfaces[edge[1]][1], "<u4"),
            "source_face_pair_count": len(pairs),
            "source_face_pair_ancestry": [list(pair) for pair in pairs],
            "source_face_pair_ancestry_sha256": pair_sha,
            "source_identities": identities,
        }
        topology_surfaces = {}
        ancestry_rows = {}
        for role, key in (("bridge", edge[0]), ("target", edge[1])):
            baseline_positions = np.asarray(surfaces[key][0] + np.float32(0.01),
                                            dtype="<f4")
            topology_surfaces[f"candidate_{role}"] = _handoff_surface_row(
                key, surfaces[key])
            topology_surfaces[f"baseline_{role}"] = _handoff_surface_row(
                key, surfaces[key], positions=baseline_positions)
            face_sha = _handoff_hash(surfaces[key][1], "<u4")
            ids_sha = _handoff_hash(surfaces[key][3], "<u8")
            ancestry_rows[role] = {
                "passed": True,
                "same_ordered_source_face_ancestry": True,
                "same_source_vertex_ancestry": True,
                "source_orientation_preserved": True,
                "source_volumes_finite_and_positive": True,
                "candidate_face_ancestry_sha256": face_sha,
                "baseline_face_ancestry_sha256": face_sha,
                "candidate_source_vertex_ids_sha256": ids_sha,
                "baseline_source_vertex_ids_sha256": ids_sha,
            }
        baseline_surfaces = {}
        for role, key in (("bridge", edge[0]), ("target", edge[1])):
            baseline_surfaces[key] = (
                np.asarray(surfaces[key][0] + np.float32(0.01), dtype="<f4"),
                surfaces[key][1], None, surfaces[key][3])
        links.append({
            "bridge": list(edge[0]),
            "target": list(edge[1]),
            "passed": True,
            "declared_network_link_source_ancestry": ancestry,
            "source_topology_gate": {
                "passed": True, "surfaces": topology_surfaces},
            "candidate_baseline_ancestry_gate": {
                "passed": True, "vessels": ancestry_rows},
            "solid_gates": {
                "candidate": _handoff_solid(edge, surfaces, source_kind="candidate"),
                "baseline": _handoff_solid(edge, baseline_surfaces,
                                           source_kind="baseline"),
            },
            "port_interfaces": {
                kind: _handoff_interface(edge, pair_sha, source_kind=kind)
                for kind in ("candidate", "baseline")},
            "independent_collar_calibre": {
                role: _handoff_calibre(key, role)
                for role, key in (("bridge", edge[0]), ("target", edge[1]))},
        })
    continuity = {
        "status": "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED",
        "passed": True,
        "requested_link_count": len(edges),
        "evaluated_link_count": len(edges),
        "links": links,
    }
    return surfaces, edges, source_pairs, continuity


def test_verified_handoff_skips_only_proven_edges_and_checks_every_other_pair():
    surfaces, edges, source_pairs, continuity = _handoff_fixture()
    verifier = _HandoffVerifier()
    progress = []

    result = bg.audit_network_rows_against_all_targets(
        verifier, surfaces, edges, source_pairs,
        verified_network_continuity=continuity,
        progress_callback=progress.append)

    network_keys = {key for edge in edges for key in edge}
    expected_nonedges = {
        frozenset((first, second))
        for index, first in enumerate(sorted(network_keys))
        for second in sorted(network_keys)[index + 1:]
        if (first, second) not in edges and (second, first) not in edges
    }
    assert result["passed"] is True
    assert result["anatomical_admission"] is False
    assert verifier.shared_calls[0]["targets"] == (
        (51005, 900), (51007, 1))
    assert verifier.shared_calls[0]["interfaces"] == {}
    assert set(verifier.pair_calls) == expected_nonedges
    assert len(result["verified_junction_handoff"]["links"]) == 2
    assert result["verified_junction_handoff"]["links"][0][
        "continuity_report_sha256"]
    assert "STARTING_NONNETWORK_TARGET_SCREEN" in progress
    assert len([event for event in progress
                if event.startswith("CHECKING_UNDECLARED_NETWORK_PAIR_")]) == len(
                    expected_nonedges)


def test_verified_handoff_rejects_muscle_contact_and_stops_at_first_nonedge_failure():
    surfaces, edges, source_pairs, continuity = _handoff_fixture()
    muscle_rejector = _HandoffVerifier(nonnetwork_pass=False)
    muscle_result = bg.audit_network_rows_against_all_targets(
        muscle_rejector, surfaces, edges, source_pairs,
        verified_network_continuity=continuity)
    assert muscle_result["passed"] is False
    assert muscle_result["failed_stage"] == "network_surface_against_nonnetwork_target"
    assert muscle_result["shared_verifier_result"]["rows"][0]["identities"][1] == [
        51005, 900]
    assert muscle_rejector.pair_calls == []

    network_rejector = _HandoffVerifier(
        failed_network_pair=frozenset(((51011, 7), (51011, 473))))
    network_result = bg.audit_network_rows_against_all_targets(
        network_rejector, surfaces, edges, source_pairs,
        verified_network_continuity=continuity)
    assert network_result["passed"] is False
    assert network_result["failed_stage"] == "undeclared_network_pair"
    assert len(network_result["nonedge_network_pair_checks"]) == 1
    assert network_result["nonedge_network_pair_checks"][0]["identities"] == [
        [51011, 7], [51011, 473]]


def test_verified_handoff_rejects_changed_candidate_source_bytes_before_oracle():
    import copy

    surfaces, edges, source_pairs, continuity = _handoff_fixture()
    tampered = copy.deepcopy(continuity)
    tampered["links"][0]["source_topology_gate"]["surfaces"][
        "candidate_bridge"]["prepared_surface"]["positions_sha256"] = "f" * 64
    verifier = _HandoffVerifier()
    with pytest.raises(ValueError, match="candidate or independent baseline bytes"):
        bg.audit_network_rows_against_all_targets(
            verifier, surfaces, edges, source_pairs,
            verified_network_continuity=tampered)
    assert verifier.shared_calls == []
    assert verifier.pair_calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross_link", "undeclared or duplicated"),
        ("partial", "complete passing fail-first link report"),
        ("failed", "failed or malformed link proof"),
    ],
)
def test_verified_handoff_rejects_cross_linked_partial_or_failed_network_report(
    mutation, message,
):
    import copy

    surfaces, edges, source_pairs, continuity = _handoff_fixture()
    tampered = copy.deepcopy(continuity)
    if mutation == "cross_link":
        tampered["links"][1]["target"] = tampered["links"][0]["target"]
    elif mutation == "partial":
        tampered["links"].pop()
        tampered["requested_link_count"] -= 1
        tampered["evaluated_link_count"] -= 1
    else:
        tampered["links"][0]["passed"] = False
    verifier = _HandoffVerifier()

    with pytest.raises(ValueError, match=message):
        bg.audit_network_rows_against_all_targets(
            verifier, surfaces, edges, source_pairs,
            verified_network_continuity=tampered)
    assert verifier.shared_calls == []
    assert verifier.pair_calls == []


def _synthetic_source_addendum_case(edge):
    import copy

    bridge_key, target_key = edge
    bridge_points, faces = _tetra(1.0)
    target_points, target_faces = _tetra(1.0, 2.0)
    source_pairs = ((0, 0), (1, 1))
    pair_sha = _handoff_hash(source_pairs, "<i8")
    evidence_path = "/frozen/source-collar.json"
    evidence_sha = "c" * 64
    members = {
        475: (91, "FJ2216"), 479: (91, "FJ2219"),
        525: (20, "FJ3482"), 469: (22, "FJ3485"),
    }
    source_faces = {
        bridge_key[1]: _handoff_hash(faces, "<u4"),
        target_key[1]: _handoff_hash(target_faces, "<u4"),
    }
    source_ids = {
        bridge_key: np.arange(len(bridge_points), dtype="<u8"),
        target_key: np.arange(len(target_points), dtype="<u8") + np.uint64(10),
    }
    baseline_points = {
        bridge_key: np.asarray(bridge_points + np.float32(0.01), dtype="<f4"),
        target_key: np.asarray(target_points + np.float32(0.01), dtype="<f4"),
    }
    row_fingerprints = {}
    ancestry_identities = {}
    addendum_identities = {}
    for role, key, row_faces in (
        ("bridge", bridge_key, faces), ("target", target_key, target_faces)):
        body, member = members[key[1]]
        identity = {
            "typed_identity": list(key),
            "body_index": body,
            "layer": 2,
            "source_member": member,
            "canonical_source_identity": {
                "stable_id": key[1],
                "receipt_identity_matches_abi5_owner": True,
            },
        }
        ancestry_identities[str(key[1])] = identity
        addendum_identities[str(key[1])] = {
            "stable_id": key[1],
            "abi5_body_index": body,
            "abi5_layer": 2,
            "canonical_source_member": member,
            "source_face_order_preserved": True,
            "receipt_identity_matches_abi5_owner": True,
            "source_faces_u32_sha256": source_faces[key[1]],
            "source_sha256": "d" * 64,
            "source_vertices6_f32_sha256": "e" * 64,
            "source_face_to_baseline_vertex_map_sha256": "f" * 64,
        }
        prefix = "parent" if role == "bridge" else "target"
        row_fingerprints[f"{role}_independent_source_baseline"] = {
            "identity": list(key),
            "positions_f32_sha256": _handoff_hash(baseline_points[key], "<f4"),
            "faces_u32_sha256": _handoff_hash(row_faces, "<u4"),
            "source_vertex_ids_u64_sha256": _handoff_hash(source_ids[key], "<u8"),
        }
    ancestry = {
        "source_face_pair_ancestry_sha256": pair_sha,
        "source_evidence_path": evidence_path,
        "source_evidence_sha256": evidence_sha,
        "source_parent_faces_sha256": source_faces[bridge_key[1]],
        "source_child_faces_sha256": source_faces[target_key[1]],
        "source_identities": ancestry_identities,
    }
    if bridge_key[1] == 475:
        additions = []
        extra_pairs = [[1, 1]]
        status = "VERIFIED_INDEPENDENT_BASELINE_EXACT_ONE_FACE_COLLAR_EXTENSION"
        bounded = {
            "no_cross_product_expansion": True,
            "only_additional_allowed_face_pair": [1, 1],
            "all_contact_points_and_segment_inside_parent_face_and_bounded_target_face": True,
        }
        exact = {
            "coordinate_units": "exact rational metres over emitted Float32 input coordinates",
            "source_face_pair": [1, 1],
            "contact_topology": "one positive-length segment, exactly two unique endpoints",
            "exact_endpoint_count": 2,
            "endpoints": [{}, {}],
        }
    else:
        extra_pairs = [[0, 0], [1, 1], [2, 2]]
        additions = extra_pairs
        status = "VERIFIED_INDEPENDENT_BASELINE_DECLARED_COLLAR_ADDITIONS"
        bounded = {
            "no_cross_product_expansion": True,
            "additional_allowed_face_pairs": extra_pairs,
            "pair_regions": [
                {
                    "face_pair": pair,
                    "bounded_contact_connection": {"passed": True},
                    "endpoint_rule": "one_shared_edge_endpoint_or_exact_indexed_contact_chain",
                } for pair in extra_pairs
            ],
        }
        exact = {
            "coordinate_units": "exact rational metres over emitted Float32 input coordinates",
            "per_pair_proofs": [
                {
                    "face_pair": pair,
                    "contact_topology": "one positive-length segment",
                    "exact_contact_point_count": 2,
                } for pair in extra_pairs
            ],
        }
    baseline_identity = {
        "accepted_capture_receipt_sha256": "1" * 64,
        "accepted_pack_sha256": "2" * 64,
        "accepted_step": 0,
        "accepted_time_s": 0.0,
        "candidate_geometry_used": False,
        "invoked_anatomy_receipt_sha256": "3" * 64,
        "source_nha_payload_sha256": "4" * 64,
    }
    lineage = {"independent_from_actual_candidate_geometry": True}
    for role, key in (("bridge", bridge_key), ("target", target_key)):
        prefix = "parent" if role == "bridge" else "target"
        row = row_fingerprints[f"{role}_independent_source_baseline"]
        lineage[f"{prefix}_position_f32_sha256"] = row["positions_f32_sha256"]
        lineage[f"{prefix}_face_u32_sha256"] = row["faces_u32_sha256"]
        lineage[f"{prefix}_gids_u64_sha256"] = row["source_vertex_ids_u64_sha256"]
    addendum = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "typed_link": [list(bridge_key), list(target_key)],
        "status": status,
        "passed": True,
        "candidate_geometry_used": False,
        "clinical_or_anatomical_admission": False,
        "source_geometry_modified": False,
        "candidate_gate_still_requires_exact_pair_audit_and_solid_calibre_proof": True,
        "baseline_identity": baseline_identity,
        "source_evidence": {
            "path": evidence_path,
            "sha256": evidence_sha,
            "frozen_source_pairs_unchanged": True,
            "frozen_source_pair_count": len(source_pairs),
            "frozen_source_pairs_sha256": pair_sha,
        },
        "baseline_surface_lineage": lineage,
        "source_identities": addendum_identities,
        "native_nha_gid_base_offset": 767009,
        "bounded_port_region": bounded,
        "exact_p6_rat_contact": exact,
    }
    addition_record = {
        "added_to_frozen_patch_cross_product": False,
        "candidate_and_baseline_still_require_exact_points": True,
        "pairs": additions,
        "used": bool(additions),
    }
    link = {
        "port_interfaces": {
            role: {"baseline_verified_declared_source_additions": copy.deepcopy(addition_record)}
            for role in ("baseline", "candidate")
        },
        "verified_baseline_collar_addendum": copy.deepcopy(addendum),
    }
    source_inventory = {
        "native_source_slot_identity": {
            "bound_to_validated_actual_capture_receipt": True,
            "native_nha_gid_base_offset": 767009,
        },
        ("baseline_matched_collar_addendum" if bridge_key[1] == 475
         else "source_baseline_matched_collar_addendum"): addendum,
    }
    return link, ancestry, row_fingerprints, {edge: source_pairs}, source_inventory


@pytest.mark.parametrize(
    ("edge", "expected_pairs"),
    [
        (((51011, 475), (51011, 479)), [[1, 1]]),
        (((51011, 525), (51011, 469)), [[0, 0], [1, 1], [2, 2]]),
    ],
)
def test_verified_handoff_binds_independent_source_collar_addenda(edge, expected_pairs):
    link, ancestry, rows, source_pairs, inventory = _synthetic_source_addendum_case(edge)
    proof = bg._validate_network_link_addendum(
        edge, link, ancestry, rows, source_pairs, inventory,
        lambda value: isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))

    assert proof["baseline_surface_lineage_bound"]
    assert proof["candidate_geometry_used_by_addendum"] is False
    assert proof["independent_additional_source_face_pairs"] == expected_pairs


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stale", "stale against the current inventory"),
        ("cross_link", "stale against the current inventory"),
        ("partial", "cross-linked"),
        ("failed", "missing, failed, or cross-linked"),
    ],
)
def test_verified_handoff_rejects_stale_cross_linked_partial_or_failed_addenda(
    mutation, message,
):
    import copy

    edge = ((51011, 525), (51011, 469))
    link, ancestry, rows, source_pairs, inventory = _synthetic_source_addendum_case(edge)
    if mutation in ("stale", "cross_link"):
        if mutation == "stale":
            link["verified_baseline_collar_addendum"]["baseline_identity"][
                "accepted_pack_sha256"] = "9" * 64
        else:
            link["verified_baseline_collar_addendum"]["typed_link"] = [
                [51011, 475], [51011, 479]]
    else:
        link_proof = link["verified_baseline_collar_addendum"]
        if mutation == "partial":
            link_proof["exact_p6_rat_contact"]["per_pair_proofs"].pop()
            inventory["source_baseline_matched_collar_addendum"] = copy.deepcopy(
                link_proof)
        else:
            link_proof["passed"] = False
            inventory["source_baseline_matched_collar_addendum"] = copy.deepcopy(
                link_proof)
    with pytest.raises(ValueError, match=message):
        bg._validate_network_link_addendum(
            edge, link, ancestry, rows, source_pairs, inventory,
            lambda value: isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))
