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
