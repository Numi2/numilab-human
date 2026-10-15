import hashlib
import json
import subprocess

import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as av
from numilab_human import resting_brachiocephalic_3d_junction as junction
from numilab_human import resting_brachiocephalic_source007 as source007
from tests.test_resting_brachiocephalic_geometry_gate import (
    _closed_cylinder,
    _manifold3d,
)


def _surface(points, faces):
    return (points, faces, None, np.arange(len(points), dtype=np.uint64))


def _frozen_pairs(bridge, bridge_faces, target, target_faces, *,
                  bridge_key=(51011, 526), target_key=(51011, 472)):
    first = av.prepare_surface(
        bridge_key, bridge, bridge_faces,
        source_vertex_ids=np.arange(len(bridge), dtype=np.uint64))
    second = av.prepare_surface(
        target_key, target, target_faces,
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


def test_new_junction_gate_uses_current_package_verifier(monkeypatch):
    def historical_loader_must_not_run():
        raise AssertionError("new gate must not reload historical HP98 verifier")

    monkeypatch.setattr(source007, "_load_central_p6",
                        historical_loader_must_not_run)
    assert junction._execution_verifier() is av
    assert junction.current_shared_verifier is av
    runtime_pins = junction._runtime_code_pins()
    assert runtime_pins["current_shared_verifier"]["path"] == str(
        junction.Path(av.__file__).resolve())
    assert "hp98_verifier" not in runtime_pins
    assert junction.PREFLIGHT_PATH.name == (
        "port-526-472-3d-junction-preflight-004.json")
    assert junction.OUTPUT_DIR.name == "port-526-472-3d-junction-diagnostic-003"


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


def _synthetic_network_link_inputs(parent_key, child_key, bridge, bridge_faces,
                                   target, target_faces):
    pairs = _frozen_pairs(
        bridge, bridge_faces, target, target_faces,
        bridge_key=parent_key, target_key=child_key)
    parent_faces = np.ascontiguousarray(bridge_faces, dtype="<u4")
    child_faces = np.ascontiguousarray(target_faces, dtype="<u4")
    pair_array = np.ascontiguousarray(pairs, dtype="<i8")
    parent_id, child_id = parent_key[1], child_key[1]
    evidence = {
        "bridge_stable_id": parent_id,
        "target_stable_id": child_id,
        "pairs": pair_array,
        "pairs_sha256": hashlib.sha256(pair_array.tobytes()).hexdigest(),
        "source_parent_faces_sha256": hashlib.sha256(
            parent_faces.tobytes()).hexdigest(),
        "source_child_faces_sha256": hashlib.sha256(
            child_faces.tobytes()).hexdigest(),
        "path": "/retained/source-collar-report.json",
        "sha256": "a" * 64,
        "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
    }
    source_rows = {
        parent_id: {
            "record": (20, 0, len(bridge), 0, len(parent_faces) * 3,
                       parent_id, 2, 0),
            "faces": parent_faces, "vertices6": np.asarray(bridge, dtype="<f4"),
            "body_index": 20, "layer": 2,
        },
        child_id: {
            "record": (20, 0, len(target), 0, len(child_faces) * 3,
                       child_id, 2, 0),
            "faces": child_faces, "vertices6": np.asarray(target, dtype="<f4"),
            "body_index": 20, "layer": 2,
        },
    }
    source_map = {
        str(parent_id): {
            "source_member": junction.network_preparation.SOURCE_MEMBER_BY_STABLE_ID[parent_id],
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
        },
        str(child_id): {
            "source_member": junction.network_preparation.SOURCE_MEMBER_BY_STABLE_ID[child_id],
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": None if child_id == 10 else 4,
        },
    }
    return evidence, source_rows, source_map



def _registered_svc_source_inputs():
    network = junction.network_preparation
    raw = source007._read_port_pairs()[0][((51011, 525), (51011, 10))]
    pair_array = np.ascontiguousarray(raw, dtype="<i8")
    counts = {525: int(pair_array[:, 0].max()) + 1,
              10: int(pair_array[:, 1].max()) + 1}
    rows = {
        stable_id: {
            "faces": np.tile(np.asarray([[0, 1, 2]], dtype="<u4"),
                             (counts[stable_id], 1)),
            "body_index": 20,
            "layer": 2,
        }
        for stable_id in (525, 10)
    }
    source_map = {
        "525": {"source_member": "FJ3482", "body_index": 20,
                "layer": 2, "vascular_compartment_id": 4},
        "10": {"source_member": "FJ3645", "body_index": 20,
               "layer": 2, "vascular_compartment_id": None},
    }
    evidence = network._read_registered_source_link_evidence(
        525, 10, source_rows=rows, source_map=source_map)
    return evidence, rows, source_map





def _exact_prepared_surface_for_source_test(key, points, faces, source_vertex_ids):
    """Build only the PreparedSurface fields used by the exact pair primitive."""
    points = np.ascontiguousarray(points, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    surface = object.__new__(av.PreparedSurface)
    surface.key = key
    surface.positions = points
    surface.faces = faces
    surface.source_vertex_ids = np.asarray(source_vertex_ids, dtype="<u8")
    lattice = [av.ci.float32_point_lattice_key(row) for row in points.tolist()]
    surface.records = tuple(
        (tuple(tuple(int(value) for value in lattice[int(vertex)])
               for vertex in face), None, None, face_id)
        for face_id, face in enumerate(faces))
    surface._face_records = {
        int(record[3]): record for record in surface.records}
    return surface

def _declared_source_baseline_addendum_fixture(
        monkeypatch, tmp_path, *, disconnect_bridge_extension=False,
        target_edge_contact_chain=False):
    parent_key, child_key = (51011, 525), (51011, 469)
    if target_edge_contact_chain:
        parent_points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0], [0.5, -0.5, 0.0], [0.5, -0.2, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, -1.0, 0.0],
            [3.0, 1.0, 0.0], [5.0, -1.0, 0.0],
        ], dtype="<f4")
        target_points = np.asarray([
            [0.5, -0.5, 1.0], [0.5, -0.5, -1.0],
            [0.5, -0.2, 0.0], [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
        ], dtype="<f4")
    else:
        parent_points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [4.0, -1.0, 0.0], [3.0, 1.0, 0.0], [5.0, -1.0, 0.0],
        ], dtype="<f4")
        target_points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
        ], dtype="<f4")
    parent_faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (748, 1))
    target_faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (280, 1))
    parent_faces[704] = [0, 1, 3]
    if disconnect_bridge_extension:
        parent_faces[704] = [0, 3, 7]
    if target_edge_contact_chain:
        parent_faces[0] = [3, 4, 5]
        parent_faces[733] = [6, 7, 9]
        parent_faces[737] = [6, 7, 8]
        parent_faces[741] = [7, 8, 10]
        baseline_pairs = np.asarray(
            [[697, 185], [717, 184], [733, 176], [741, 176]], dtype="<i8")
    else:
        parent_faces[733] = [4, 5, 7]
        parent_faces[737] = [4, 5, 6]
        parent_faces[741] = [5, 6, 8]
        baseline_pairs = np.asarray(
            [[697, 184], [697, 185], [733, 176], [741, 176]], dtype="<i8")
    target_faces[176] = [4, 5, 6]
    target_faces[185] = [0, 1, 3]
    evidence = {
        "parent_stable_id": 525,
        "child_stable_id": 469,
        "path": "/frozen/bridge-collar-ancestry-107-001/report.json",
        "sha256": "e" * 64,
        "pairs": baseline_pairs,
        "pairs_sha256": hashlib.sha256(baseline_pairs.tobytes()).hexdigest(),
        "source_parent_faces_sha256": hashlib.sha256(
            parent_faces.tobytes()).hexdigest(),
        "source_child_faces_sha256": hashlib.sha256(
            target_faces.tobytes()).hexdigest(),
        "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
    }
    source_rows = {
        525: {
            "record": (20, 100, len(parent_points), 0,
                       len(parent_faces) * 3, 525, 2, 0),
            "faces": parent_faces, "vertices6": parent_points,
            "body_index": 20, "layer": 2,
        },
        469: {
            "record": (22, 200, len(target_points), 0,
                       len(target_faces) * 3, 469, 2, 0),
            "faces": target_faces, "vertices6": target_points,
            "body_index": 22, "layer": 2,
        },
    }
    source_map = {
        "525": {"source_member": "FJ3482", "body_index": 20,
                "layer": 2, "vascular_compartment_id": 4},
        "469": {"source_member": "FJ3485", "body_index": 22,
                "layer": 2, "vascular_compartment_id": 4},
    }
    parent_surface = _exact_prepared_surface_for_source_test(
        parent_key, parent_points, parent_faces,
        np.arange(10100, 10100 + len(parent_points), dtype="<u8"))
    target_surface = _exact_prepared_surface_for_source_test(
        child_key, target_points, target_faces,
        np.arange(10200, 10200 + len(target_points), dtype="<u8"))
    monkeypatch.setattr(
        junction.network_preparation, "_read_registered_source_link_evidence",
        lambda _parent, _child, **_kwargs: evidence)
    capture = {
        "pack_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "capture_receipt_sha256": "2" * 64,
        "invoked_anatomy_receipt_sha256": "3" * 64,
        "common_field_source_anatomy_payload_sha256": "4" * 64,
        "accepted_step": 0,
        "accepted_time_s": 0.0,
    }
    baseline_binding = {
        "actual_accepted_capture": capture,
        "source_bound_rigid_rest_baseline": {
            "source_nha_payload_sha256": "4" * 64,
            "payload_is_current_capture_source": True,
            "method": (
                "apply the existing pinned transform_f32 to exact NHA vertices6 XYZ "
                "using the actual accepted rigid pose; cardiac common-field "
                "displacement is excluded"),
        },
    }
    preflight = tmp_path / "source-preflight.json"
    preflight.write_text(json.dumps({"actual_accepted_capture": capture}))
    preflight_sha = hashlib.sha256(preflight.read_bytes()).hexdigest()
    source_pairs_sha = evidence["pairs_sha256"]
    parent_faces_sha = evidence["source_parent_faces_sha256"]
    child_faces_sha = evidence["source_child_faces_sha256"]
    declarations = junction.SOURCE_BASELINE_ADDITIONS
    baseline_observed_pairs = set(map(tuple, baseline_pairs.tolist()))
    if target_edge_contact_chain:
        baseline_observed_pairs.discard((717, 184))
    declared_pairs = set(baseline_observed_pairs)
    for bridge_face in range(len(parent_faces)):
        if len(declared_pairs) == 80:
            break
        declared_pairs.add((bridge_face, 250))
    assert len(declared_pairs) == 80
    unexpected_pairs = {
        tuple(item["face_pair"]) for item in declarations}
    observed_pairs = sorted(declared_pairs | unexpected_pairs)
    frozen_surface_lineage = {
        role: {
            "position_f32_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.positions, dtype="<f4").tobytes()
            ).hexdigest(),
            "face_u32_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.faces, dtype="<u4").tobytes()
            ).hexdigest(),
            "gids_u64_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.source_vertex_ids, dtype="<u8").tobytes()
            ).hexdigest(),
        }
        for role, surface in (
            ("parent", parent_surface), ("target", target_surface))
    }
    report = {
        "status": "STOPPED_AT_FIRST_FAILED_NETWORK_LINK",
        "anatomical_admission": False,
        "native_or_gpu_work_performed": False,
        "source_preflight_path": str(preflight.resolve()),
        "source_preflight_sha256": preflight_sha,
        "candidate_and_source_baseline_inventory": {
            "baseline_matched_collar_addendum": {
                "typed_link": [[51011, 475], [51011, 479]],
                "baseline_surface_lineage": {
                    "parent_position_f32_sha256": "a" * 64,
                    "parent_face_u32_sha256": "b" * 64,
                    "parent_gids_u64_sha256": "c" * 64,
                    "target_position_f32_sha256": "d" * 64,
                    "target_face_u32_sha256": "e" * 64,
                    "target_gids_u64_sha256": "f" * 64,
                },
            },
        },
        "network_source_continuity": {
            "first_failed_link": [list(parent_key), list(child_key)],
            "links": [{
                "bridge": list(parent_key),
                "target": list(child_key),
                "declared_network_link_source_ancestry": {
                    "typed_link": [list(parent_key), list(child_key)],
                    "source_evidence_sha256": evidence["sha256"],
                    "source_face_pair_ancestry_sha256": source_pairs_sha,
                    "source_parent_faces_sha256": parent_faces_sha,
                    "source_child_faces_sha256": child_faces_sha,
                },
                "source_topology_gate": {
                    "surfaces": {
                        "baseline_bridge": {
                            "identity": list(parent_key),
                            "source_vertex_ids_sha256": frozen_surface_lineage[
                                "parent"]["gids_u64_sha256"],
                            "prepared_surface": {
                                "identity": list(parent_key),
                                "positions_sha256": frozen_surface_lineage[
                                    "parent"]["position_f32_sha256"],
                                "faces_sha256": frozen_surface_lineage[
                                    "parent"]["face_u32_sha256"],
                            },
                        },
                        "baseline_target": {
                            "identity": list(child_key),
                            "source_vertex_ids_sha256": frozen_surface_lineage[
                                "target"]["gids_u64_sha256"],
                            "prepared_surface": {
                                "identity": list(child_key),
                                "positions_sha256": frozen_surface_lineage[
                                    "target"]["position_f32_sha256"],
                                "faces_sha256": frozen_surface_lineage[
                                    "target"]["face_u32_sha256"],
                            },
                        },
                    },
                },
                "port_interfaces": {"baseline": {
                    "observed_face_pairs": [list(pair)
                                             for pair in sorted(declared_pairs)],
                    "observed_pair_count": len(declared_pairs),
                    "unexpected_face_pairs": [list(pair)
                                              for pair in sorted(unexpected_pairs)],
                    "passed": False,
                    "audit_result": {
                        "declared_interface_triangle_pairs": [list(pair)
                                                              for pair in sorted(declared_pairs)],
                        "unexpected_triangle_pairs": [list(pair)
                                                      for pair in sorted(unexpected_pairs)],
                        "triangle_pair_count": len(observed_pairs),
                        "passed": False,
                    },
                }},
                "abi5_face_corner_ancestry": {
                    role: {
                        "typed_identity": [51011, stable_id],
                        "native_nha_gid_base_offset": 10000,
                        "nha_source_row_first_vertex": first_vertex,
                        "capture_gid_formula": (
                            "native NHA base offset + ABI5 record firstVertex + source-local vertex index"),
                        "native_source_slot_identity_is_contiguous": True,
                        "observed_gid_minus_nha_global_vertex_index_min": 10000,
                        "observed_gid_minus_nha_global_vertex_index_max": 10000,
                        "first_used_source_vertex_local_index": 0,
                        "first_used_source_vertex_capture_gid": 10000 + first_vertex,
                        "capture_face_count": len(row["faces"]),
                        "capture_faces_u32_sha256": hashlib.sha256(
                            row["faces"].tobytes()).hexdigest(),
                        "source_abi5_face_array_sha256": hashlib.sha256(
                            row["faces"].tobytes()).hexdigest(),
                    }
                    for role, stable_id, first_vertex, row in (
                        ("baseline_parent", 525, 100, source_rows[525]),
                        ("baseline_child", 469, 200, source_rows[469]),
                    )
                },
            }],
        },
    }
    report_path = tmp_path / "baseline-report.json"
    report_path.write_text(json.dumps(report))
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    monkeypatch.setattr(junction, "SOURCE_BASELINE_OBSERVATION_REPORT", report_path)
    monkeypatch.setattr(
        junction, "SOURCE_BASELINE_OBSERVATION_REPORT_SHA256", report_sha)
    monkeypatch.setattr(
        junction, "SOURCE_BASELINE_OBSERVATION_PREFLIGHT_SHA256", preflight_sha)
    return {
        "bridge_key": parent_key,
        "target_key": child_key,
        "source_link_evidence": evidence,
        "source_rows": source_rows,
        "source_map": source_map,
        "bridge": parent_surface,
        "target": target_surface,
        "baseline_binding": baseline_binding,
        "gid_base_offset": 10000,
        "declarations": declarations,
        "baseline_report_path": report_path,
        "baseline_observation": {
            "report_path": str(report_path.resolve()),
            "report_sha256": report_sha,
        },
        "observed_pairs": observed_pairs,
    }


def _rebind_source_baseline_fixture_to_current_capture(inputs, *, gid_offset=20000):
    capture = inputs["baseline_binding"]["actual_accepted_capture"]
    capture["pack_sha256"] = "9" * 64
    capture["receipt_sha256"] = "8" * 64
    capture["capture_receipt_sha256"] = "8" * 64
    capture["invoked_anatomy_receipt_sha256"] = "7" * 64
    capture["common_field_source_anatomy_payload_sha256"] = "6" * 64
    baseline = inputs["baseline_binding"]["source_bound_rigid_rest_baseline"]
    baseline["source_nha_payload_sha256"] = "6" * 64
    for stable_id, surface in (
        (inputs["bridge_key"][1], inputs["bridge"]),
        (inputs["target_key"][1], inputs["target"]),
    ):
        row = inputs["source_rows"][stable_id]
        record = row["record"]
        current_first_vertex = record[1] + 8961
        row["record"] = (
            record[0], current_first_vertex, *record[2:])
        surface.source_vertex_ids = np.asarray(
            gid_offset + current_first_vertex
            + np.arange(len(surface.positions)),
            dtype="<u8")
    inputs["gid_base_offset"] = gid_offset
    inputs["baseline_observation"]["current_source_surface_replay"] = (
        junction.SOURCE_BASELINE_SURFACE_REPLAY_POLICY)
    return inputs


def _run_declared_source_baseline_addendum(inputs):
    return junction.verify_baseline_matched_collar_addendum(
        av,
        bridge_key=inputs["bridge_key"],
        target_key=inputs["target_key"],
        source_link_evidence=inputs["source_link_evidence"],
        source_rows=inputs["source_rows"],
        source_map=inputs["source_map"],
        independent_source_baseline_bridge_surface=inputs["bridge"],
        independent_source_baseline_target_surface=inputs["target"],
        native_nha_gid_base_offset=inputs["gid_base_offset"],
        baseline_binding=inputs["baseline_binding"],
        declarations=inputs["declarations"],
        baseline_observation=inputs["baseline_observation"],
    )


def test_current_capture_rebind_replays_only_exact_source_positions_and_faces(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    old_parent_gids = np.ascontiguousarray(
        inputs["bridge"].source_vertex_ids, dtype="<u8").tobytes()
    inputs = _rebind_source_baseline_fixture_to_current_capture(inputs)
    assert old_parent_gids != np.ascontiguousarray(
        inputs["bridge"].source_vertex_ids, dtype="<u8").tobytes()

    proof = _run_declared_source_baseline_addendum(inputs)
    observation = proof["source_baseline_observation"]
    replay = observation["capture_identity_replay"]
    assert replay["applied"] is True
    assert replay["policy"] == junction.SOURCE_BASELINE_SURFACE_REPLAY_POLICY
    assert replay["historical_capture_identity"]["accepted_pack_sha256"] == "1" * 64
    assert replay["current_capture_identity"]["accepted_pack_sha256"] == "9" * 64
    assert replay["historical_capture_identity"]["source_nha_payload_sha256"] == "4" * 64
    assert replay["current_capture_identity"]["source_nha_payload_sha256"] == "6" * 64
    assert replay["historical_nha_source_row_first_vertices"] == {
        "bridge": 100, "target": 200}
    assert replay["current_nha_source_row_first_vertices"] == {
        "bridge": 9061, "target": 9161}
    assert replay["exact_position_and_ordered_face_hashes_replayed"] is True
    assert replay["historical_gid_hashes_are_provenance_only_when_capture_differs"] is True
    assert replay["current_abi5_gid_slot_rebinding_required"] is True
    current_gid_binding = observation["current_abi5_gid_slot_rebinding"]
    assert current_gid_binding["passed"] is True
    assert set(current_gid_binding["typed_rows"]) == {"469", "525"}
    assert current_gid_binding["typed_rows"]["525"][
        "current_nha_source_row_first_vertex"] == 9061
    assert current_gid_binding["typed_rows"]["469"][
        "current_nha_source_row_first_vertex"] == 9161
    assert proof["source_baseline_observation"][
        "baseline_surfaces_reconstructed_from_current_pinned_source_and_capture"]
    assert proof["candidate_geometry_used"] is False
    assert proof["candidate_gate_still_requires_exact_pair_audit_and_solid_calibre_proof"]


def test_current_capture_rebind_requires_explicit_policy_and_exact_surfaces(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    _rebind_source_baseline_fixture_to_current_capture(inputs)
    inputs["baseline_observation"].pop("current_source_surface_replay")
    with pytest.raises(ValueError, match="another capture"):
        _run_declared_source_baseline_addendum(inputs)

    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    _rebind_source_baseline_fixture_to_current_capture(inputs)
    inputs["bridge"].positions[0, 0] = np.nextafter(
        inputs["bridge"].positions[0, 0], np.float32(np.inf), dtype=np.float32)
    with pytest.raises(ValueError, match="positions or ordered faces differ"):
        _run_declared_source_baseline_addendum(inputs)

    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    _rebind_source_baseline_fixture_to_current_capture(inputs)
    inputs["target"].faces[0, 0] = np.uint32(1)
    with pytest.raises(ValueError, match="current ABI5 parent/child face arrays"):
        _run_declared_source_baseline_addendum(inputs)

    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    _rebind_source_baseline_fixture_to_current_capture(inputs)
    inputs["bridge"].source_vertex_ids[0] += np.uint64(1)
    with pytest.raises(ValueError, match="GIDs do not bind exact NHA source slots"):
        _run_declared_source_baseline_addendum(inputs)


def test_declared_baseline_additions_require_exact_pinned_baseline_report(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    proof = junction.verify_baseline_matched_collar_addendum(
        av,
        bridge_key=inputs["bridge_key"],
        target_key=inputs["target_key"],
        source_link_evidence=inputs["source_link_evidence"],
        source_rows=inputs["source_rows"],
        source_map=inputs["source_map"],
        independent_source_baseline_bridge_surface=inputs["bridge"],
        independent_source_baseline_target_surface=inputs["target"],
        native_nha_gid_base_offset=inputs["gid_base_offset"],
        baseline_binding=inputs["baseline_binding"],
        declarations=inputs["declarations"],
        baseline_observation=inputs["baseline_observation"],
    )
    assert proof["passed"]
    assert proof["candidate_geometry_used"] is False
    assert proof["source_geometry_modified"] is False
    assert proof["bounded_port_region"]["additional_allowed_face_pairs"] == [
        [704, 184], [704, 185], [737, 176]]
    pair_reports = proof["bounded_port_region"]["pair_regions"]
    assert len(pair_reports) == 3
    assert all(item["bounded_contact_connection"]["passed"]
               for item in pair_reports)
    assert {item["face_pair"][0] for item in pair_reports} == {704, 737}
    assert proof["source_baseline_observation"]["unexpected_pairs"] == [
        [704, 184], [704, 185], [737, 176]]
    assert proof["source_baseline_observation"]["observed_pair_count"] == 83
    raw_report = json.loads(inputs["baseline_report_path"].read_text())
    baseline_port = raw_report["network_source_continuity"]["links"][0][
        "port_interfaces"]["baseline"]
    assert len(baseline_port["audit_result"][
        "declared_interface_triangle_pairs"]) == 80
    assert len(baseline_port["audit_result"]["unexpected_triangle_pairs"]) == 3
    assert baseline_port["audit_result"]["triangle_pair_count"] == 83
    assert proof["source_baseline_observation"][
        "baseline_surface_lineage"] == proof[
            "source_baseline_observation"]["frozen_inventory_surface_lineage"]
    assert proof["source_baseline_observation"][
        "frozen_inventory_surface_lineage"]["bridge"] == {
            "position_f32_sha256": hashlib.sha256(
                np.ascontiguousarray(inputs["bridge"].positions, dtype="<f4").tobytes()
            ).hexdigest(),
            "face_u32_sha256": hashlib.sha256(
                np.ascontiguousarray(inputs["bridge"].faces, dtype="<u4").tobytes()
            ).hexdigest(),
            "gids_u64_sha256": hashlib.sha256(
                np.ascontiguousarray(inputs["bridge"].source_vertex_ids, dtype="<u8").tobytes()
            ).hexdigest(),
        }
    assert proof["candidate_gate_still_requires_exact_pair_audit_and_solid_calibre_proof"]
    scope = junction.bind_network_link_source_ancestry(
        inputs["bridge_key"], inputs["target_key"],
        source_link_evidence=inputs["source_link_evidence"],
        source_rows=inputs["source_rows"], source_map=inputs["source_map"])
    replayed = junction._verify_declared_source_baseline_addendum_replay(
        proof, source_scope=scope, baseline_bridge=inputs["bridge"],
        baseline_target=inputs["target"])
    assert replayed == {(704, 184), (704, 185), (737, 176)}


def test_interface_gate_hands_off_only_verified_exact_source_addition_pairs():
    from types import SimpleNamespace
    captured = {}

    class PairVerifier:
        def audit_surface_pair(self, bridge, target, *, intended_face_pairs):
            captured["allowed"] = set(intended_face_pairs)
            return {
                "passed": True,
                "declared_interface_triangle_pairs": [],
                "unexpected_triangle_pairs": [],
                "containment": [],
            }

    bridge_key, target_key = (51011, 525), (51011, 469)
    bridge = SimpleNamespace(key=bridge_key, faces=np.zeros((748, 3), dtype="<u4"))
    target = SimpleNamespace(key=target_key, faces=np.zeros((280, 3), dtype="<u4"))
    addendum = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "passed": True,
        "typed_link": [list(bridge_key), list(target_key)],
        "candidate_geometry_used": False,
        "source_geometry_modified": False,
        "bounded_port_region": {
            "additional_allowed_face_pairs": [[704, 184], [704, 185], [737, 176]],
            "no_cross_product_expansion": True,
        },
        "exact_p6_rat_contact": {
            "source_baseline_observation": (
                junction.SOURCE_BASELINE_OBSERVATION_REPORT_SHA256),
        },
    }
    result = junction._interface_gate(
        PairVerifier(), bridge, target, ((0, 0),),
        declared_vessel_ports=junction.declared_network_links(),
        verified_collar_addendum=addendum)
    assert captured["allowed"] == {
        (0, 0), (704, 184), (704, 185), (737, 176)}
    assert result["baseline_verified_declared_source_additions"] == {
        "used": True,
        "pairs": [[704, 184], [704, 185], [737, 176]],
        "added_to_frozen_patch_cross_product": False,
        "candidate_and_baseline_still_require_exact_points": True,
    }


def test_declared_baseline_additions_reject_unbound_pairs_and_changed_report(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    wrong_declarations = (dict(inputs["declarations"][0],
                               face_pair=(704, 183)),
                          *inputs["declarations"][1:])
    kwargs = {
        "source_link_evidence": inputs["source_link_evidence"],
        "source_rows": inputs["source_rows"],
        "source_map": inputs["source_map"],
        "independent_source_baseline_bridge_surface": inputs["bridge"],
        "independent_source_baseline_target_surface": inputs["target"],
        "native_nha_gid_base_offset": inputs["gid_base_offset"],
        "baseline_binding": inputs["baseline_binding"],
        "baseline_observation": inputs["baseline_observation"],
    }
    with pytest.raises(ValueError, match="three pinned 525->469 source pairs"):
        junction.verify_baseline_matched_collar_addendum(
            av, bridge_key=inputs["bridge_key"], target_key=inputs["target_key"],
            declarations=wrong_declarations, **kwargs)

    report_path = inputs["baseline_report_path"]
    report_path.write_text(report_path.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        junction.verify_baseline_matched_collar_addendum(
            av, bridge_key=inputs["bridge_key"], target_key=inputs["target_key"],
            declarations=inputs["declarations"], **kwargs)


def test_declared_baseline_additions_bind_frozen_baseline_position_and_gid_hashes(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(monkeypatch, tmp_path)
    report_path = inputs["baseline_report_path"]
    report = json.loads(report_path.read_text())
    report["network_source_continuity"]["links"][0][
        "source_topology_gate"]["surfaces"]["baseline_bridge"][
            "prepared_surface"]["positions_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        junction, "SOURCE_BASELINE_OBSERVATION_REPORT_SHA256", report_sha)
    with pytest.raises(ValueError, match="per-link baseline_bridge position/GID/face hashes changed"):
        junction.verify_baseline_matched_collar_addendum(
            av,
            bridge_key=inputs["bridge_key"],
            target_key=inputs["target_key"],
            source_link_evidence=inputs["source_link_evidence"],
            source_rows=inputs["source_rows"],
            source_map=inputs["source_map"],
            independent_source_baseline_bridge_surface=inputs["bridge"],
            independent_source_baseline_target_surface=inputs["target"],
            native_nha_gid_base_offset=inputs["gid_base_offset"],
            baseline_binding=inputs["baseline_binding"],
            declarations=inputs["declarations"],
            baseline_observation={
                "report_path": str(report_path.resolve()),
                "report_sha256": report_sha,
            },
        )


def test_declared_baseline_additions_accept_exact_target_edge_contact_chain(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(
        monkeypatch, tmp_path, target_edge_contact_chain=True)
    proof = junction.verify_baseline_matched_collar_addendum(
        av,
        bridge_key=inputs["bridge_key"],
        target_key=inputs["target_key"],
        source_link_evidence=inputs["source_link_evidence"],
        source_rows=inputs["source_rows"],
        source_map=inputs["source_map"],
        independent_source_baseline_bridge_surface=inputs["bridge"],
        independent_source_baseline_target_surface=inputs["target"],
        native_nha_gid_base_offset=inputs["gid_base_offset"],
        baseline_binding=inputs["baseline_binding"],
        declarations=inputs["declarations"],
        baseline_observation=inputs["baseline_observation"],
    )
    first = next(item for item in proof["bounded_port_region"]["pair_regions"]
                 if item["face_pair"] == [704, 184])
    assert first["exact_endpoints_on_frozen_extension_edges"] == [[], []]
    assert first["bounded_contact_connection"]["kind"] == (
        "exact_indexed_contact_chain_to_original_frozen_contact")
    assert [item["to_face_pair"]
            for item in first["bounded_contact_connection"]["path"]] == [
                [704, 185], [697, 185]]
    assert proof["candidate_gate_still_requires_exact_pair_audit_and_solid_calibre_proof"]


def test_declared_baseline_additions_reject_face_adjacency_without_shared_edge(
        monkeypatch, tmp_path):
    inputs = _declared_source_baseline_addendum_fixture(
        monkeypatch, tmp_path, disconnect_bridge_extension=True)
    with pytest.raises(ValueError, match="not edge-adjacent to frozen face 697"):
        junction.verify_baseline_matched_collar_addendum(
            av,
            bridge_key=inputs["bridge_key"],
            target_key=inputs["target_key"],
            source_link_evidence=inputs["source_link_evidence"],
            source_rows=inputs["source_rows"],
            source_map=inputs["source_map"],
            independent_source_baseline_bridge_surface=inputs["bridge"],
            independent_source_baseline_target_surface=inputs["target"],
            native_nha_gid_base_offset=inputs["gid_base_offset"],
            baseline_binding=inputs["baseline_binding"],
            declarations=inputs["declarations"],
            baseline_observation=inputs["baseline_observation"],
        )


def _registered_baseline_addendum_fixture(monkeypatch, *, unbound_target_neighbor=False):
    parent_key, child_key = (51011, 475), (51011, 479)
    parent_points = np.asarray([
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [0.0, 0.0, 1.0],
    ], dtype="<f4")
    target_points = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ], dtype="<f4")
    parent_faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (1578, 1))
    target_faces = np.tile(np.asarray([[0, 1, 2]], dtype="<u4"), (395, 1))
    target_faces[392] = [2, 0, 3]
    target_faces[394] = [0, 1, 4]
    if unbound_target_neighbor:
        target_faces[394] = [1, 2, 4]
    parent_first, target_first, base = 100, 200, 10000
    rows = {
        475: {
            "record": (20, parent_first, len(parent_points), 0,
                       len(parent_faces) * 3, 475, 2, 0),
            "faces": parent_faces,
            "vertices6": parent_points,
            "body_index": 20,
            "layer": 2,
        },
        479: {
            "record": (20, target_first, len(target_points), 0,
                       len(target_faces) * 3, 479, 2, 0),
            "faces": target_faces,
            "vertices6": target_points,
            "body_index": 20,
            "layer": 2,
        },
    }
    parent_surface = _exact_prepared_surface_for_source_test(
        parent_key, parent_points, parent_faces,
        np.arange(base + parent_first, base + parent_first + len(parent_points),
                  dtype="<u8"))
    target_surface = _exact_prepared_surface_for_source_test(
        child_key, target_points, target_faces,
        np.arange(base + target_first, base + target_first + len(target_points),
                  dtype="<u8"))
    pair_array = np.asarray([[1577, 392], [1577, 394]], dtype="<i8")
    evidence = {
        "parent_stable_id": 475,
        "child_stable_id": 479,
        "path": "/frozen/arterial-venous-route-field-086/report.json",
        "sha256": "e" * 64,
        "pairs": pair_array,
        "pairs_sha256": hashlib.sha256(pair_array.tobytes()).hexdigest(),
        "source_parent_faces_sha256": hashlib.sha256(
            parent_faces.tobytes()).hexdigest(),
        "source_child_faces_sha256": hashlib.sha256(
            target_faces.tobytes()).hexdigest(),
        "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
    }
    source_map = {
        "475": {
            "source_member": "FJ2216", "body_index": 20, "layer": 2,
            "vascular_compartment_id": 4, "source_sha256": "a" * 64,
            "name": "left axillary artery",
        },
        "479": {
            "source_member": "FJ2219", "body_index": 20, "layer": 2,
            "vascular_compartment_id": 4, "source_sha256": "b" * 64,
            "name": "left brachial artery",
        },
    }
    monkeypatch.setattr(
        junction.network_preparation, "_read_registered_source_link_evidence",
        lambda _parent, _child, **_kwargs: evidence)
    baseline_binding = {
        "actual_accepted_capture": {
            "pack_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "capture_receipt_sha256": "2" * 64,
            "invoked_anatomy_receipt_sha256": "3" * 64,
            "accepted_step": 0,
            "accepted_time_s": 0.0,
            "common_field_source_anatomy_payload_sha256": "4" * 64,
        },
        "source_bound_rigid_rest_baseline": {
            "source_nha_payload_sha256": "4" * 64,
            "payload_is_current_capture_source": True,
            "method": (
                "apply the existing pinned transform_f32 to exact NHA vertices6 XYZ "
                "using the actual accepted rigid pose; cardiac common-field "
                "displacement is excluded"),
        },
    }
    return {
        "bridge_key": parent_key,
        "target_key": child_key,
        "source_link_evidence": evidence,
        "source_rows": rows,
        "source_map": source_map,
        "bridge": parent_surface,
        "target": target_surface,
        "baseline_binding": baseline_binding,
        "gid_base_offset": base,
    }


def test_baseline_collaredge_addendum_is_exact_and_one_face_bounded(monkeypatch):
    inputs = _registered_baseline_addendum_fixture(monkeypatch)

    proof = junction.verify_baseline_matched_collar_addendum(
        av,
        bridge_key=inputs["bridge_key"],
        target_key=inputs["target_key"],
        source_link_evidence=inputs["source_link_evidence"],
        source_rows=inputs["source_rows"],
        source_map=inputs["source_map"],
        independent_source_baseline_bridge_surface=inputs["bridge"],
        independent_source_baseline_target_surface=inputs["target"],
        native_nha_gid_base_offset=inputs["gid_base_offset"],
        baseline_binding=inputs["baseline_binding"],
    )

    assert proof["passed"]
    assert proof["candidate_geometry_used"] is False
    assert proof["source_evidence"]["frozen_source_pairs_unchanged"]
    assert proof["bounded_port_region"]["only_additional_allowed_face_pair"] == [1577, 391]
    assert proof["bounded_port_region"]["no_cross_product_expansion"]
    contact = proof["exact_p6_rat_contact"]
    assert contact["contact_topology"] == "one positive-length segment, exactly two unique endpoints"
    assert contact["exact_endpoint_count"] == 2
    assert {row["target_frozen_neighbor_face"] for row in contact["endpoints"]} == {392, 394}
    assert all(row["exact_point_on_target_neighbor_edge"] for row in contact["endpoints"])
    assert all(
        isinstance(component["numerator"], str)
        and isinstance(component["denominator"], str)
        for row in contact["endpoints"] for component in row["point_m"])
    assert proof["source_identities"]["475"]["canonical_source_member"] == "FJ2216"
    assert proof["source_identities"]["479"]["canonical_source_member"] == "FJ2219"
    assert proof["baseline_surface_lineage"]["independent_from_actual_candidate_geometry"]


def test_baseline_collaredge_addendum_rejects_wrong_link_or_wrong_source_region(monkeypatch):
    inputs = _registered_baseline_addendum_fixture(monkeypatch)
    kwargs = {
        "source_link_evidence": inputs["source_link_evidence"],
        "source_rows": inputs["source_rows"],
        "source_map": inputs["source_map"],
        "independent_source_baseline_bridge_surface": inputs["bridge"],
        "independent_source_baseline_target_surface": inputs["target"],
        "native_nha_gid_base_offset": inputs["gid_base_offset"],
        "baseline_binding": inputs["baseline_binding"],
    }
    with pytest.raises(ValueError, match="restricted to typed link 475->479"):
        junction.verify_baseline_matched_collar_addendum(
            av, bridge_key=(51011, 525), target_key=(51011, 479), **kwargs)

    unbound = _registered_baseline_addendum_fixture(
        monkeypatch, unbound_target_neighbor=True)
    with pytest.raises(ValueError, match="exact contact endpoint must land"):
        junction.verify_baseline_matched_collar_addendum(
            av,
            bridge_key=unbound["bridge_key"],
            target_key=unbound["target_key"],
            source_link_evidence=unbound["source_link_evidence"],
            source_rows=unbound["source_rows"],
            source_map=unbound["source_map"],
            independent_source_baseline_bridge_surface=unbound["bridge"],
            independent_source_baseline_target_surface=unbound["target"],
            native_nha_gid_base_offset=unbound["gid_base_offset"],
            baseline_binding=unbound["baseline_binding"],
        )

def test_network_registry_is_exactly_thirteen_edges_and_two_svc_ports():
    from numilab_human import (
        resting_brachiocephalic_current_capture_network_preparation as network,
    )

    expected = (*network.NETWORK_EDGES, *network.SVC_PORTS)
    assert junction.declared_network_links() == expected
    assert len(expected) == 15
    assert ((51011, 469), (51011, 473)) not in expected
    assert ((51011, 473), (51011, 469)) not in expected


def test_source_link_binding_rejects_wrong_branch_and_wrong_face_ancestry():
    parent_key, child_key = (51011, 525), (51011, 10)
    evidence, source_rows, source_map = _registered_svc_source_inputs()

    scope = junction.bind_network_link_source_ancestry(
        parent_key, child_key, source_link_evidence=evidence,
        source_rows=source_rows, source_map=source_map)
    assert scope["typed_link"] == [[51011, 525], [51011, 10]]
    assert scope["source_face_pair_count"] > 0
    assert scope["source_identities"]["10"]["vascular_compartment_id"] is None
    assert scope["source_identities"]["525"]["canonical_source_identity"][
        "canonical_source_member"] == "FJ3482"

    with pytest.raises(ValueError, match="undeclared vascular link"):
        junction.bind_network_link_source_ancestry(
            (51011, 469), (51011, 473), source_link_evidence=evidence,
            source_rows=source_rows, source_map=source_map)

    pending = dict(evidence)
    pending["evidence_status"] = "SOURCE_BOUND_COLLAR_DERIVATION_REQUIRED"
    with pytest.raises(ValueError, match="pending"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=pending,
            source_rows=source_rows, source_map=source_map)

    bad_pairs = dict(evidence)
    bad_pairs["pairs_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="face-pair array SHA-256"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=bad_pairs,
            source_rows=source_rows, source_map=source_map)

    bad_faces = dict(evidence)
    bad_faces["source_parent_faces_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not bound to the current ABI5"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=bad_faces,
            source_rows=source_rows, source_map=source_map)

    bad_path = dict(evidence)
    bad_path["path"] = "/retained/forged-report.json"
    with pytest.raises(ValueError, match="not the registered frozen report"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=bad_path,
            source_rows=source_rows, source_map=source_map)

    bad_report_sha = dict(evidence)
    bad_report_sha["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="report SHA-256 differs"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=bad_report_sha,
            source_rows=source_rows, source_map=source_map)

    tampered_pairs = dict(evidence)
    tampered_pairs["pairs"] = evidence["pairs"].copy()
    tampered_pairs["pairs"][0, 0] = int(tampered_pairs["pairs"][:, 0].max()) + 1
    tampered_pairs["pairs_sha256"] = hashlib.sha256(
        tampered_pairs["pairs"].tobytes()).hexdigest()
    with pytest.raises(ValueError, match="pairs parsed from the actual frozen report"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=tampered_pairs,
            source_rows=source_rows, source_map=source_map)

    forged_map = dict(source_map)
    forged_map["525"] = dict(source_map["525"], source_member="FJ3583")
    with pytest.raises(ValueError, match="not canonical FJ3482"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=evidence,
            source_rows=source_rows, source_map=forged_map)

    wrong_svc = dict(evidence)
    wrong_svc["target_stable_id"] = 526
    with pytest.raises(ValueError, match="disagree with the frozen"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=wrong_svc,
            source_rows=source_rows, source_map=source_map)


def test_svc_network_link_uses_solid_and_independent_calibre_not_pair_count(monkeypatch):
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _calibre_preserving_target()
    parent_key, child_key = (51011, 525), (51011, 10)
    evidence, source_rows, source_map = _synthetic_network_link_inputs(
        parent_key, child_key, bridge, bridge_faces, target, target_faces)
    monkeypatch.setattr(
        junction.network_preparation, "_read_registered_source_link_evidence",
        lambda _parent, _child, **_kwargs: evidence)

    result = junction.audit_brachiocephalic_network_link_3d(
        av, _manifold3d(),
        bridge_key=parent_key, bridge_surface=_surface(bridge, bridge_faces),
        target_key=child_key, target_surface=_surface(target, target_faces),
        source_link_evidence=evidence, source_rows=source_rows,
        source_map=source_map,
        baseline_bridge_surface=_surface(bridge, bridge_faces),
        baseline_target_surface=_surface(target, target_faces),
        expected_nha_gid_base_offset=0)

    assert result["passed"]
    metric = result["continuity_metric"]
    assert metric["collision_pair_counts_are_not_continuity_metrics"]
    assert metric["source_face_pair_count_is_patch_ancestry_only"] > 0
    assert metric["candidate_and_independent_baseline_solid_metrics"][
        "candidate"]["positive_intersection_volume"]
    assert metric["candidate_and_independent_baseline_solid_metrics"][
        "candidate"]["exact_inside_both_witness"]["first_location"] == "inside"
    assert metric["independent_material_station_calibre"]["bridge"][
        "declared_area_ratio"] == [0.8, 1.25]
    assert metric["independent_material_station_calibre"]["target"][
        "declared_minimum_feret_ratio"] == [0.9, 1.1]
    assert result["declared_network_link_source_ancestry"][
        "source_identities"]["10"]["vascular_compartment_id"] is None





def test_verified_collar_addendum_adds_only_single_face_pair(monkeypatch):
    captured = {}

    class PassingPairVerifier:
        def audit_surface_pair(self, bridge, target, *, intended_face_pairs):
            captured["allowed"] = tuple(intended_face_pairs)
            return {
                "passed": True,
                "declared_interface_triangle_pairs": [],
                "unexpected_triangle_pairs": [],
                "containment": [],
            }

    from types import SimpleNamespace
    bridge = SimpleNamespace(key=(51011, 475), faces=np.zeros((1578, 3), dtype=np.uint32))
    target = SimpleNamespace(key=(51011, 479), faces=np.zeros((392, 3), dtype=np.uint32))
    pairs = [[0, 0], [1, 1]]
    addendum = {
        "schema": "numi.human.registered-source-collar-addendum.v1",
        "passed": True,
        "typed_link": [[51011, 475], [51011, 479]],
        "bounded_port_region": {
            "only_additional_allowed_face_pair": [1577, 391],
            "no_cross_product_expansion": True,
        },
    }

    junction._interface_gate(
        PassingPairVerifier(), bridge, target, pairs,
        declared_vessel_ports=junction.declared_network_links(),
        verified_collar_addendum=addendum)

    assert set(captured["allowed"]) == {
        (0, 0), (0, 1), (1, 0), (1, 1), (1577, 391)}
    assert (1577, 0) not in captured["allowed"]
    assert (0, 391) not in captured["allowed"]


def test_verified_collar_addendum_rejects_wrong_link_before_pair_audit():
    from types import SimpleNamespace
    class NeverCalledVerifier:
        def audit_surface_pair(self, *_args, **_kwargs):
            raise AssertionError("wrong-link addendum must fail before geometry")

    bridge = SimpleNamespace(key=(51011, 475), faces=np.zeros((1578, 3), dtype=np.uint32))
    target = SimpleNamespace(key=(51011, 479), faces=np.zeros((392, 3), dtype=np.uint32))
    with pytest.raises(ValueError, match="invalid or belongs to another typed port"):
        junction._interface_gate(
            NeverCalledVerifier(), bridge, target, [[0, 0]],
            declared_vessel_ports=junction.declared_network_links(),
            verified_collar_addendum={
                "passed": True,
                "schema": "numi.human.registered-source-collar-addendum.v1",
                "typed_link": [[51011, 525], [51011, 469]],
                "bounded_port_region": {
                    "only_additional_allowed_face_pair": [1577, 391],
                    "no_cross_product_expansion": True,
                },
            })


def test_legacy_interface_scope_does_not_expand_with_network_registry(monkeypatch):
    other_network_link = ((51011, 525), (51011, 469))
    monkeypatch.setattr(
        junction.gate, "DECLARED_VESSEL_PORTS",
        frozenset({junction.PARTIAL_OVERLAP_PORT, other_network_link}))

    class PassingPairVerifier:
        def audit_surface_pair(self, *_args, **_kwargs):
            return {
                "passed": True,
                "declared_interface_triangle_pairs": [],
                "unexpected_triangle_pairs": [],
                "containment": [],
            }

    triangle = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    bridge = av.prepare_surface((51011, 525), triangle, faces)
    target = av.prepare_surface((51011, 469), triangle, faces)
    result = junction._interface_gate(
        PassingPairVerifier(), bridge, target, [[0, 0]])

    scope = result["documented_vessel_junction_scope"]
    assert not scope["passed"]
    assert scope["scope_kind"] == "legacy_single_port_entrypoint"



def test_source_preparation_binds_every_typed_link_and_source_face_array(monkeypatch):
    from numilab_human import (
        resting_brachiocephalic_current_capture_network_preparation as network,
    )

    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    rows = {
        stable_id: {"faces": faces.copy(), "body_index": 20, "layer": 2}
        for edge in junction.declared_network_links()
        for stable_id in (edge[0][1], edge[1][1])
    }
    source_map = {
        str(stable_id): {
            "source_member": network.SOURCE_MEMBER_BY_STABLE_ID[stable_id],
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": (
                4 if stable_id in (525, 526) else
                None if stable_id == 10 else 1),
        }
        for stable_id in rows
    }
    pairs = np.asarray([[0, 0]], dtype="<i8")
    evidence_by_edge = {}
    for first, second in network.NETWORK_EDGES:
        parent, child = first[1], second[1]
        evidence_by_edge[f"{parent}->{child}"] = {
            "parent_stable_id": parent,
            "child_stable_id": child,
            "pairs": pairs.copy(),
            "pairs_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
            "source_parent_faces_sha256": hashlib.sha256(
                rows[parent]["faces"].tobytes()).hexdigest(),
            "source_child_faces_sha256": hashlib.sha256(
                rows[child]["faces"].tobytes()).hexdigest(),
            "path": "/retained/route-source.json",
            "sha256": "b" * 64,
            "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
        }
    svc = {}
    for first, second in network.SVC_PORTS:
        parent, child = first[1], second[1]
        svc[f"{parent}->{child}"] = {
            "bridge_stable_id": parent,
            "target_stable_id": child,
            "pairs": pairs.copy(),
            "pairs_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
            "source_parent_faces_sha256": hashlib.sha256(
                rows[parent]["faces"].tobytes()).hexdigest(),
            "source_child_faces_sha256": hashlib.sha256(
                rows[child]["faces"].tobytes()).hexdigest(),
            "path": "/retained/svc-source.json",
            "sha256": "c" * 64,
            "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
        }
    keys = {key for edge in junction.declared_network_links() for key in edge}
    surfaces = {key: ("source", key) for key in keys}

    fake_evidence = {
        **evidence_by_edge,
        **svc,
    }
    monkeypatch.setattr(
        junction.network_preparation, "_read_registered_source_link_evidence",
        lambda parent, child, **_kwargs: fake_evidence[f"{parent}->{child}"])
    inputs = junction.prepare_network_link_audit_inputs(
        evidence_by_edge, {"svc_bridge_ports": svc},
        source_rows=rows, source_map=source_map,
        candidate_surfaces=surfaces, baseline_surfaces=surfaces)

    assert len(inputs) == 15
    assert tuple((item["bridge_key"], item["target_key"]) for item in inputs) == (
        junction.declared_network_links())
    assert all(
        item["source_link_evidence"]["source_parent_faces_sha256"]
        == hashlib.sha256(rows[item["bridge_key"][1]]["faces"].tobytes()).hexdigest()
        for item in inputs)
    assert all(
        item["source_link_evidence"]["source_child_faces_sha256"]
        == hashlib.sha256(rows[item["target_key"][1]]["faces"].tobytes()).hexdigest()
        for item in inputs)

    wrong = dict(evidence_by_edge)
    wrong["525->469"] = dict(wrong["525->469"])
    wrong["525->469"]["parent_stable_id"] = 469
    wrong["525->469"]["child_stable_id"] = 473
    with pytest.raises(ValueError, match="typed link IDs disagree"):
        junction.prepare_network_link_audit_inputs(
            wrong, {"svc_bridge_ports": svc},
            source_rows=rows, source_map=source_map,
            candidate_surfaces=surfaces, baseline_surfaces=surfaces)


def test_network_link_aggregator_requires_exact_graph_and_stops_at_first_failure(monkeypatch):
    links = junction.declared_network_links()
    records = [{"bridge_key": edge[0], "target_key": edge[1]} for edge in links]
    calls = []

    def fake_link_audit(_verifier, _manifold, **kwargs):
        calls.append((kwargs["bridge_key"], kwargs["target_key"]))
        return {"passed": len(calls) < 3}

    monkeypatch.setattr(junction, "audit_brachiocephalic_network_link_3d",
                        fake_link_audit)
    result = junction.audit_brachiocephalic_network_links_3d(
        av, object(), records, expected_nha_gid_base_offset=0)

    assert result["status"] == "REJECT_SOURCE_NETWORK_LINK_CONTINUITY"
    assert result["requested_link_count"] == 15
    assert result["evaluated_link_count"] == 3
    assert result["first_failed_link"] == [
        list(links[2][0]), list(links[2][1])]
    assert calls == list(links[:3])

    with pytest.raises(ValueError, match="exactly 15"):
        junction.audit_brachiocephalic_network_links_3d(
            av, object(), records[:-1], expected_nha_gid_base_offset=0)

    wrong = list(records)
    wrong[-1] = {"bridge_key": (51011, 469), "target_key": (51011, 473)}
    with pytest.raises(ValueError, match="undeclared or duplicate"):
        junction.audit_brachiocephalic_network_links_3d(
            av, object(), wrong, expected_nha_gid_base_offset=0)


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
