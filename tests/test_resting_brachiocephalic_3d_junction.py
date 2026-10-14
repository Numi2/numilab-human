import hashlib
import json
import subprocess

import numpy as np
import pytest

from numilab_human import resting_brachiocephalic_source007 as source007
from tests.test_resting_brachiocephalic_geometry_gate import (
    _closed_cylinder,
    _manifold3d,
)

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


def _interface_result(bridge_key, bridge, bridge_faces, target_key, target,
                      target_faces, frozen_pairs):
    first = av.prepare_surface(
        bridge_key, bridge, bridge_faces,
        source_vertex_ids=np.arange(len(bridge), dtype=np.uint64))
    second = av.prepare_surface(
        target_key, target, target_faces,
        source_vertex_ids=np.arange(len(target), dtype=np.uint64))
    return junction._interface_gate(av, first, second, frozen_pairs)


def _profiled_cylinder(stations, *, sectors=12):
    """Closed tapered source with an exact r=0.2, z=1 collar ring."""
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
    return np.asarray(rings, dtype="<f4"), np.asarray(faces, dtype="<u4")


def _calibre_preserving_target():
    # It matches the bridge top-ring exactly at z=1, then tapers away from that
    # junction. The matched source ring gives the interface gate real witnesses.
    return _profiled_cylinder(((0.8, 0.18), (1.0, 0.2),
                               (1.05, 0.2), (2.0, 0.18)))


def _pinched_target():
    # Keep the exact z=1 collar but collapse just distal to the matched ring.
    return _profiled_cylinder(((0.8, 0.18), (1.0, 0.2),
                               (1.05, 0.08), (2.0, 0.08)))


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
    target, target_faces = _calibre_preserving_target()

    result = _proof(bridge, bridge_faces, target, target_faces)

    assert result["passed"]
    assert result["source_topology_gate"]["passed"]
    assert result["candidate_baseline_ancestry_gate"]["passed"]
    candidate_port = result["port_interfaces"]["candidate"]
    assert candidate_port["exact_interface_point_count"] > 0
    assert not candidate_port["p6_audit_passed"]
    assert candidate_port["exact_partial_overlap_witnesses"]["passed"]
    assert candidate_port["exact_partial_overlap_witnesses"][
        "requires_separate_closed_solid_confirmation"]
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


def test_positive_bone_muscle_penetration_is_outside_vessel_junction_scope():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _calibre_preserving_target()
    frozen_pairs = _frozen_pairs(bridge, bridge_faces, target, target_faces)

    # 51004 and 51005 are the existing bone and muscle semantic classes. The
    # fixture intentionally reuses penetrating vessel-shaped geometry so only
    # typed ownership distinguishes it from the declared vein junction.
    result = _interface_result(
        (51004, 526), bridge, bridge_faces,
        (51005, 472), target, target_faces, frozen_pairs)

    assert result["exact_interface_point_count"] > 0
    assert len(result["exact_partial_overlap_witnesses"]["containment"]) == 2
    assert not result["documented_vessel_junction_scope"]["passed"]
    assert not result["exact_partial_overlap_witnesses"]["passed"]
    assert not result["passed"]


def test_vessel_crossing_outside_frozen_collar_patch_is_rejected():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _calibre_preserving_target()
    frozen_pairs = _frozen_pairs(bridge, bridge_faces, target, target_faces)
    moved_target = target.copy()
    moved_target[:, 0] += 0.05

    result = _interface_result(
        (51011, 526), bridge, bridge_faces,
        (51011, 472), moved_target, target_faces, frozen_pairs)

    assert result["documented_vessel_junction_scope"]["passed"]
    assert result["exact_interface_point_count"] > 0
    assert result["unexpected_face_pairs"]
    # Some observed pairs still lie in the collar patch; the additional
    # crossing pairs outside that patch are independently disqualifying.
    assert result["observed_pairs_within_frozen_face_patch_cross_product"]
    assert not result["exact_partial_overlap_witnesses"]["passed"]
    assert not result["passed"]


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
    baseline_target, target_faces = _calibre_preserving_target()
    candidate_target, candidate_faces = _pinched_target()
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


def test_candidate_decoder_calls_the_existing_abi5_codec_owner(monkeypatch):
    calls = []
    source_map = {"472": {}, "525": {}, "526": {}}
    decoded_rows = {472: {}, 525: {}, 526: {}}

    def parse_with_existing_owner(raw):
        calls.append(raw)
        return ("NHANAT-header", ["record"], decoded_rows)

    monkeypatch.setattr(junction.nhanat_codec, "_parse_payload", parse_with_existing_owner)
    header, records, rows = junction._decode_candidate_payload(b"pinned-payload", source_map)

    assert calls == [b"pinned-payload"]
    assert header == "NHANAT-header"
    assert records == ["record"]
    assert rows is decoded_rows


def test_source007_compiler_pins_resolve_immutable_git_objects(tmp_path, monkeypatch):
    repo = tmp_path / "frozen-source"
    source = repo / "src" / "owner.py"
    source.parent.mkdir(parents=True)
    original = b"OWNER_REVISION = 'frozen'" + bytes([10])
    source.write_bytes(original)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "src/owner.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "freeze owner"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    source.write_bytes(b"OWNER_REVISION = 'modified-after-run'" + bytes([10]))
    monkeypatch.setattr(
        source007, "_code_and_dependency_pins",
        lambda: {"manifold3d_retained_wheel": {}})
    pins = {
        "source_repository": {"head": revision, "dirty": False},
        "source_code_sha256": {
            str(source): hashlib.sha256(original).hexdigest()},
        "manifold3d_retained_wheel": {},
    }

    verified = junction._verify_source007_compiler_code_pins(pins)

    assert verified["passed"]
    assert verified["source_code_file_count"] == 1
    owner = verified["verified_source_code"][str(source)]
    assert owner["verification_method"] == "frozen_source_commit_git_object"
    assert owner["verified_sha256"] == owner["expected_sha256"]
    assert owner["current_sha256"] != owner["expected_sha256"]

    pins["source_code_sha256"][str(source)] = "0" * 64
    with pytest.raises(ValueError, match="historical compiler code hash mismatch"):
        junction._verify_source007_compiler_code_pins(pins)


def test_material_branch_scope_names_one_frozen_branch_and_limits_coverage():
    source_map = {
        "526": {
            "source_member": "FJ3583",
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
        },
        "472": {
            "source_member": "FJ3587",
            "name": "right subclavian vein",
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
        },
    }
    pairs = {key: [[0, 0]] for key in source007.PORT_KEYS}
    pairs[((51011, 526), (51011, 472))] = [[0, 0] for _ in range(63)]

    scope = junction._material_branch_scope(source_map, pairs)

    assert scope["selected_material_branch"] == (
        "FJ3583 right brachiocephalic vein to FJ3587 right subclavian vein")
    assert scope["all_six_expected_ports_present"]
    assert scope["selected_frozen_interface"]["face_pair_count"] == 63
    assert "does not establish the complete right venous tree" in scope["coverage_limit"]
    assert scope["existing_cvsim_compartment_id"] == 4
    assert scope["added_physical_mass_kg"] == 0
    assert not scope["added_blood_state_or_force"]
