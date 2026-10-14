import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as av
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


def _run_cylinder_port(candidate_target, baseline_target, baseline_bridge=None):
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0)
    bridge_baseline = bridge if baseline_bridge is None else baseline_bridge[0]
    bridge_baseline_faces = bridge_faces if baseline_bridge is None else baseline_bridge[1]
    target_faces = _closed_cylinder(0.8, 2.0, radius=0.18)[1]
    baseline_target_faces = target_faces
    pairs = _port_pairs(bridge, bridge_faces, candidate_target, target_faces)
    return audit_brachiocephalic_port_pair_continuity(
        av, manifold3d,
        bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
        target_key=(51011, 10), target_surface=(candidate_target, target_faces),
        frozen_face_pairs=pairs,
        baseline_bridge_surface=(bridge_baseline, bridge_baseline_faces),
        baseline_target_surface=(baseline_target, baseline_target_faces))


def test_overlapping_closed_cylinders_pass_solid_continuity_and_calibre_screen():
    target, _ = _closed_cylinder(0.8, 2.0, radius=0.18)
    report = _run_cylinder_port(target, target)

    assert report["passed"]
    assert report["solid_gate"]["strict_positive_volume"]
    assert report["solid_gate"]["intersection_volume_m3"] > 0.0
    assert report["solid_gate"]["exact_inside_both_witness"]["first_location"] == "inside"
    assert report["solid_gate"]["exact_inside_both_witness"]["second_location"] == "inside"
    assert report["solid_gate"]["union_component_count"] == 1
    assert report["solid_gate"]["union_numerical_solid_contract_passed"]
    assert report["solid_gate"]["union_csg_analysis_precision"].startswith(
        "Manifold Mesh64 Float64")
    assert report["solid_gate"]["union_section_mesh_position_dtype"] == "float64"
    assert report["solid_gate"]["union_boolean_exact_topology_proven"] is False
    assert report["calibre_gate"]["station_results"]["bridge"]["candidate_vs_baseline"]["passed"]
    assert report["calibre_gate"]["station_results"]["target"]["candidate_vs_baseline"]["passed"]
    assert report["calibre_gate"]["declared_preservation_intervals"] == {
        "area": [0.8, 1.25], "minimum_feret_width": [0.9, 1.1]}


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
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0)
    baseline_target, target_faces = _closed_cylinder(0.8, 2.0, radius=0.18)
    candidate_target, _ = _closed_cylinder(0.8, 2.0, radius=0.08)
    pairs = _port_pairs(bridge, bridge_faces, candidate_target, target_faces)

    with pytest.raises(ValueError, match="target section calibre ratio"):
        audit_brachiocephalic_port_pair_continuity(
            av, manifold3d,
            bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
            target_key=(51011, 10), target_surface=(candidate_target, target_faces),
            frozen_face_pairs=pairs,
            baseline_bridge_surface=(bridge, bridge_faces),
            baseline_target_surface=(baseline_target, target_faces))


def test_baseline_section_is_transported_with_source_vertex_ancestry():
    manifold3d = _manifold3d()
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0)
    candidate_target, target_faces = _closed_cylinder(0.84, 2.04, radius=0.18)
    baseline_target, _ = _closed_cylinder(0.8, 2.0, radius=0.18)
    pairs = _port_pairs(bridge, bridge_faces, candidate_target, target_faces)

    report = audit_brachiocephalic_port_pair_continuity(
        av, manifold3d,
        bridge_key=(51011, 525), bridge_surface=(bridge, bridge_faces),
        target_key=(51011, 10), target_surface=(candidate_target, target_faces),
        frozen_face_pairs=pairs,
        baseline_bridge_surface=(bridge, bridge_faces),
        baseline_target_surface=(baseline_target, target_faces))
    candidate_station = np.asarray(
        report["calibre_gate"]["station_results"]["target"]["station"]
        ["section_center_world_m"])
    baseline_station = np.asarray(
        report["calibre_gate"]["station_results"]["target"]
        ["baseline_station_transported_by_ancestry"]["section_center_world_m"])

    assert report["passed"]
    assert np.linalg.norm(candidate_station - baseline_station) > 0.03
    assert (report["calibre_gate"]["station_results"]["target"]
            ["candidate_vs_baseline"]["passed"])



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
    *, disconnected=False, corrupt_lineage=False, declared=True):
    manifold3d = _manifold3d()
    bridge_positions, bridge_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    target_positions, target_faces = _closed_cylinder(
        -1.0, 1.0, radius=0.2, sectors=12, axial=8)
    if disconnected:
        target_positions[:, 0] += np.float32(0.45)
    else:
        angle = np.arctan(0.7)
        cosine, sine = np.cos(angle), np.sin(angle)
        x = target_positions[:, 0].copy()
        z = target_positions[:, 2].copy()
        target_positions[:, 0] = cosine * x + sine * z + np.float32(0.55)
        target_positions[:, 2] = -sine * x + cosine * z
    bridge_positions = np.ascontiguousarray(bridge_positions, dtype="<f4")
    target_positions = np.ascontiguousarray(target_positions, dtype="<f4")
    bridge = av.prepare_surface(
        (51011, 526), bridge_positions, bridge_faces,
        source_vertex_ids=np.arange(len(bridge_positions), dtype=np.uint64))
    target = av.prepare_surface(
        (51011, 472), target_positions, target_faces,
        source_vertex_ids=np.arange(len(target_positions), dtype=np.uint64))
    bridge_solid, _ = bg._make_manifold(bridge, manifold3d)
    target_solid, _ = bg._make_manifold(target, manifold3d)
    intersection = manifold3d.Manifold.batch_boolean(
        [bridge_solid, target_solid], manifold3d.OpType.Intersect)
    union = manifold3d.Manifold.batch_boolean(
        [bridge_solid, target_solid], manifold3d.OpType.Add)
    positions, faces, mesh = bg._manifold_mesh(union)
    lineage = bg._mesh_face_lineage(mesh, {
        526: len(bridge.quotient_faces), 472: len(target.quotient_faces)})
    owners = lineage["face_owner_ids"].copy()
    if corrupt_lineage:
        owners[owners == 472] = 526
    candidate_union_section = bg._section_loops(
        positions, faces, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        derived_csg=True, face_owner_ids=owners,
        face_source_ids=lineage["source_face_ids"],
        face_generated_mask=lineage["generated_face_mask"])
    baseline_union_section = candidate_union_section
    def source_station(surface):
        source_ids = surface.source_vertex_ids.astype(int).tolist()
        return {
            "section_center_world_m": [0.0, 0.0, 0.0],
            "tangent_world_unit": [0.0, 0.0, 1.0],
            "material_anchor_source_vertex_ids": source_ids[:4],
            "tangent_support_source_vertex_ids": [source_ids[0], source_ids[-1]],
            "frozen_collar_face_ids": [0, 1],
        }

    bridge_station = source_station(bridge)
    candidate_branch_sections = {
        526: bg._source_surface_section(bridge, bridge_station),
        472: bg._source_surface_section(target, bridge_station),
    }
    baseline_branch_sections = candidate_branch_sections
    intersection_volume = float(intersection.volume())
    inside_witness = (
        bg._exact_inside_witness(av, bridge, target, intersection)
        if intersection_volume > 0.0 else None)
    raw_port = av.audit_surface_pair(bridge, target)
    allowed_pairs = tuple(tuple(map(int, pair))
                          for pair in raw_port.get("unexpected_triangle_pairs", []))
    port = av.audit_surface_pair(
        bridge, target, intended_face_pairs=allowed_pairs)
    exact_points = bg._exact_interface_points(
        av, bridge, target,
        port.get("declared_interface_triangle_pairs", []))
    evidence = {
        "candidate_bridge_status": str(bridge_solid.status()),
        "candidate_target_status": str(target_solid.status()),
        "baseline_bridge_status": str(bridge_solid.status()),
        "baseline_target_status": str(target_solid.status()),
        "candidate_union_status": str(union.status()),
        "baseline_union_status": str(union.status()),
        "candidate_intersection_status": str(intersection.status()),
        "baseline_intersection_status": str(intersection.status()),
        "candidate_branch_component_counts": {
            "526": len(bridge_solid.decompose()),
            "472": len(target_solid.decompose()),
        },
        "baseline_branch_component_counts": {
            "526": len(bridge_solid.decompose()),
            "472": len(target_solid.decompose()),
        },
        "candidate_union_component_count": len(union.decompose()),
        "baseline_union_component_count": len(union.decompose()),
        "candidate_union_volume_m3": float(union.volume()),
        "baseline_union_volume_m3": float(union.volume()),
        "candidate_intersection_volume_m3": intersection_volume,
        "baseline_intersection_volume_m3": intersection_volume,
        "candidate_exact_inside_both_witness": inside_witness,
        "baseline_exact_inside_both_witness": inside_witness,
        "candidate_port_audit_passed": bool(port.get("passed")),
        "baseline_port_audit_passed": bool(port.get("passed")),
        "candidate_port_exact_interface_point_count": len(exact_points),
        "baseline_port_exact_interface_point_count": len(exact_points),
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
    proof = bg._junction_aware_two_loop_section_proof(
        expected_owner_ids=(526, 472),
        candidate_union_section=candidate_union_section,
        baseline_union_section=baseline_union_section,
        candidate_branch_sections=candidate_branch_sections,
        baseline_branch_sections=baseline_branch_sections,
        solid_evidence=evidence,
        station_declaration=station_declaration,
    )
    return proof, candidate_union_section, evidence


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
            "f80c2128420620b4670e3ea6f672d2c56ca077ceac25c7e7b6819fdb8608dfad")
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
    assert portdiag.SIX_DIAGNOSTIC_DIR.name == "port-525-526-six-port-diagnostic-002"
    assert portdiag.SIX_PREFLIGHT_PATH.name == "port-525-526-six-port-preflight-002.json"
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
