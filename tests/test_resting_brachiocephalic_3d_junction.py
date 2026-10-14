import json

import numpy as np
from test_resting_brachiocephalic_geometry_gate import (
    _closed_cylinder,
    _manifold3d,
)

from numilab_human import resting_brachiocephalic_source007 as source007

av = source007._load_central_p6()
from numilab_human import resting_brachiocephalic_3d_junction as junction


def _surface(points, faces):
    return (points, faces, None, np.arange(len(points), dtype=np.uint64))


def _frozen_pairs(bridge, bridge_faces, target, target_faces):
    first = av.prepare_surface(
        (51011, 526), bridge, bridge_faces,
        source_vertex_ids=np.arange(len(bridge), dtype=np.uint64))
    second = av.prepare_surface(
        (51011, 472), target, target_faces,
        source_vertex_ids=np.arange(len(target), dtype=np.uint64))
    return av.audit_surface_pair(first, second)["unexpected_triangle_pairs"]


def _proof(bridge, bridge_faces, target, target_faces, *,
           baseline_bridge=None, baseline_target=None, pairs=None):
    if pairs is None:
        pairs = _frozen_pairs(bridge, bridge_faces, target, target_faces)
    return junction.audit_brachiocephalic_3d_junction(
        av, _manifold3d(),
        bridge_key=(51011, 526), bridge_surface=_surface(bridge, bridge_faces),
        target_key=(51011, 472), target_surface=_surface(target, target_faces),
        frozen_face_pairs=pairs,
        baseline_bridge_surface=_surface(
            bridge if baseline_bridge is None else baseline_bridge[0],
            bridge_faces if baseline_bridge is None else baseline_bridge[1]),
        baseline_target_surface=_surface(
            target if baseline_target is None else baseline_target[0],
            target_faces if baseline_target is None else baseline_target[1]),
    )


def test_overlapping_vessels_pass_with_independent_collar_sections_and_3d_witnesses():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _closed_cylinder(0.8, 2.0, radius=0.18)

    result = _proof(bridge, bridge_faces, target, target_faces)

    assert result["passed"]
    assert result["source_topology_gate"]["passed"]
    assert result["candidate_baseline_ancestry_gate"]["passed"]
    assert result["port_interfaces"]["candidate"]["exact_interface_point_count"] > 0
    assert result["solid_gates"]["candidate"]["positive_intersection_volume"]
    assert result["solid_gates"]["candidate"]["exact_inside_both_witness"][
        "first_location"] == "inside"
    assert result["solid_gates"]["candidate"]["exact_inside_both_witness"][
        "second_location"] == "inside"
    assert result["solid_gates"]["candidate"]["union_component_count"] == 1
    assert result["solid_gates"]["candidate"]["volume_accounting"]["passed"]
    assert result["solid_gates"]["candidate"][
        "union_lineage_has_both_sources_without_generated_or_unmapped_faces"]
    assert result["independent_collar_calibre"]["bridge"]["passed"]
    assert result["independent_collar_calibre"]["target"]["passed"]
    bridge_station = result["independent_collar_calibre"]["bridge"]["station"]
    target_station = result["independent_collar_calibre"]["target"]["station"]
    assert bridge_station["own_collar_station_policy"] == "OWN_COLLAR_OUTSIDE_PATCH_V1"
    assert target_station["own_collar_station_policy"] == "OWN_COLLAR_OUTSIDE_PATCH_V1"
    assert bridge_station["section_center_world_m"] != target_station["section_center_world_m"]
    assert not result["limitations"]["shared_planar_junction_section_required"]
    assert not result["limitations"]["anatomical_admission"]
    assert not result["limitations"]["existing_blood_state_owner_mutated"]


def test_tangent_vessels_reject_zero_volume_even_if_closed():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _closed_cylinder(1.0, 2.0, radius=0.18)

    result = _proof(bridge, bridge_faces, target, target_faces)

    assert not result["passed"]
    assert result["source_topology_gate"]["passed"]
    assert result["solid_gates"]["candidate"]["intersection_volume_m3"] == 0.0
    assert not result["solid_gates"]["candidate"]["passed"]


def test_disconnected_extra_closed_component_is_reported_and_rejected():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    extra, extra_faces = _closed_cylinder(
        3.0, 4.0, radius=0.05, center_x=2.0, sectors=8, axial=4)
    bridge_with_extra = np.concatenate((bridge, extra), axis=0)
    bridge_with_extra_faces = np.concatenate(
        (bridge_faces, extra_faces + len(bridge)), axis=0)
    target, target_faces = _closed_cylinder(0.8, 2.0, radius=0.18)

    result = _proof(
        bridge_with_extra, bridge_with_extra_faces, target, target_faces)

    assert not result["passed"]
    assert not result["source_topology_gate"]["passed"]
    evidence = result["source_topology_gate"]["surfaces"]["candidate_bridge"]
    assert evidence["prepared_surface"]["exact_coordinate_quotient_topology"][
        "face_component_count"] == 2
    assert evidence["referenced_face_component_count"] == 2
    assert evidence["exact_self_audit_count"] == 0
    assert "candidate_bridge" in result["rejections"][0]


def test_local_collar_calibre_rejects_a_pinch_after_global_solid_checks():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    baseline_target, target_faces = _closed_cylinder(0.8, 2.0, radius=0.18)
    candidate_target, candidate_faces = _closed_cylinder(0.8, 2.0, radius=0.08)
    pairs = _frozen_pairs(bridge, bridge_faces, candidate_target, candidate_faces)

    result = _proof(
        bridge, bridge_faces, candidate_target, candidate_faces,
        baseline_target=(baseline_target, target_faces), pairs=pairs)

    assert not result["passed"]
    assert result["solid_gates"]["candidate"]["passed"]
    assert result["independent_collar_calibre"]["bridge"]["passed"]
    target_calibre = result["independent_collar_calibre"]["target"]
    assert not target_calibre["passed"]
    ratios = target_calibre["candidate_to_baseline_comparison"][
        "candidate_to_baseline_ratios"]
    assert ratios["area"] < 0.8
    assert ratios["minimum_feret_width"] < 0.9


def test_volume_accounting_rejects_union_that_drops_input_volume():
    result = junction._volume_accounting(
        first_volume=2.0, second_volume=3.0, intersection_volume=0.5,
        union_volume=4.0)

    assert not result["passed"]
    assert result["expected_union_volume_m3"] == 4.5
    assert result["union_volume_not_less_than_either_source"]
    assert result["relative_volume_balance_error"] > result["declared_relative_tolerance"]



def test_source004_plane_failure_is_multiloop_then_missing_not_branching():
    assert junction.sha_file(junction.SOURCE004_REPORT) == (
        junction.SOURCE004_REPORT_SHA256)
    report = json.loads(junction.SOURCE004_REPORT.read_text())

    diagnosis = junction._source004_plane_failure(report)

    assert diagnosis["diagnosis"] == "target_tangent_plane_multi_loop_then_bridge_absent"
    assert diagnosis["source_section_graph_branching_proven"] is False
    assert [item["candidate_fraction_from_pinned_004_policy"]
            for item in diagnosis["candidate_planes"]] == [0.2, 0.4, 0.6, 0.8]
    assert [item["bridge526_source_section"]["loop_count"]
            for item in diagnosis["candidate_planes"]] == [2, 0, 0, 0]
    assert all(item["target472_source_section"]["loop_count"] == 1
               for item in diagnosis["candidate_planes"])

    first = diagnosis["candidate_planes"][0]["bridge526_source_section"]
    assert first["degree_histogram"] == {"2": 36}
    assert first["degree_not_two_node_count"] == 0
    assert [loop["source_face_ancestor_ids"]
            for loop in first["loop_face_ancestry"]] == [
        [1, 2, 3, 4, 6, 7, 8, 17, 18, 40, 41, 65, 67],
        [112, 128, 129, 130, 131, 133, 134, 135, 136, 159, 168, 169,
         198, 199, 200, 201, 287, 288, 289, 306, 307, 308, 316],
    ]
    assert first["edge_ids_retained_in_source004"] is False


def test_source007_compiler_owner_hashes_resolve_to_frozen_emission_revision():
    compiler_preflight = json.loads(source007.PREFLIGHT_PATH.read_text())

    verified = junction._verify_source007_compiler_code_pins(
        compiler_preflight["code_and_dependency_pins"])

    assert verified["passed"]
    assert verified["source_code_file_count"] == 242
    assert verified["frozen_source_repository"]["head"] == (
        "e921f79b9b078911c30f552890d22274fb8cf56e")
    changed_gate = verified["verified_source_code"][
        "/Users/numi/numi-human-vein-geometry-gate-20261014/src/"
        "numilab_human/resting_brachiocephalic_geometry_gate.py"]
    assert changed_gate["verification_method"] == "frozen_source_commit_git_object"
    assert changed_gate["verified_sha256"] == changed_gate["expected_sha256"]
