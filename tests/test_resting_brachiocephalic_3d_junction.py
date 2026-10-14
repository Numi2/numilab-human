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
        parent_id: {"faces": parent_faces},
        child_id: {"faces": child_faces},
    }
    source_map = {
        str(parent_id): {
            "source_member": f"SYNTHETIC-{parent_id}",
            "body_index": 20,
            "layer": 2,
            "vascular_compartment_id": 4,
        },
        str(child_id): {
            "source_member": f"SYNTHETIC-{child_id}",
            "body_index": 20,
            "layer": 2,
            # The SVC has no compartment annotation in the source receipt.
        },
    }
    return evidence, source_rows, source_map


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
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _calibre_preserving_target()
    parent_key, child_key = (51011, 525), (51011, 10)
    evidence, source_rows, source_map = _synthetic_network_link_inputs(
        parent_key, child_key, bridge, bridge_faces, target, target_faces)

    scope = junction.bind_network_link_source_ancestry(
        parent_key, child_key, source_link_evidence=evidence,
        source_rows=source_rows, source_map=source_map)
    assert scope["typed_link"] == [[51011, 525], [51011, 10]]
    assert scope["source_face_pair_count"] > 0
    assert scope["source_identities"]["10"]["vascular_compartment_id"] is None

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

    wrong_svc = dict(evidence)
    wrong_svc["target_stable_id"] = 526
    with pytest.raises(ValueError, match="disagree with the frozen"):
        junction.bind_network_link_source_ancestry(
            parent_key, child_key, source_link_evidence=wrong_svc,
            source_rows=source_rows, source_map=source_map)


def test_svc_network_link_uses_solid_and_independent_calibre_not_pair_count():
    bridge, bridge_faces = _closed_cylinder(-1.0, 1.0, radius=0.2)
    target, target_faces = _calibre_preserving_target()
    parent_key, child_key = (51011, 525), (51011, 10)
    evidence, source_rows, source_map = _synthetic_network_link_inputs(
        parent_key, child_key, bridge, bridge_faces, target, target_faces)

    result = junction.audit_brachiocephalic_network_link_3d(
        av, _manifold3d(),
        bridge_key=parent_key, bridge_surface=_surface(bridge, bridge_faces),
        target_key=child_key, target_surface=_surface(target, target_faces),
        source_link_evidence=evidence, source_rows=source_rows,
        source_map=source_map,
        baseline_bridge_surface=_surface(bridge, bridge_faces),
        baseline_target_surface=_surface(target, target_faces))

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


def test_source_preparation_binds_every_typed_link_and_source_face_array():
    from numilab_human import (
        resting_brachiocephalic_current_capture_network_preparation as network,
    )

    faces = np.asarray([[0, 1, 2]], dtype="<u4")
    rows = {stable_id: {"faces": faces.copy()}
            for edge in junction.declared_network_links()
            for stable_id in (edge[0][1], edge[1][1])}
    source_map = {
        str(stable_id): {
            "source_member": f"MEMBER-{stable_id}",
            "body_index": 20,
            "layer": 2,
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
    result = junction.audit_brachiocephalic_network_links_3d(av, object(), records)

    assert result["status"] == "REJECT_SOURCE_NETWORK_LINK_CONTINUITY"
    assert result["requested_link_count"] == 15
    assert result["evaluated_link_count"] == 3
    assert result["first_failed_link"] == [
        list(links[2][0]), list(links[2][1])]
    assert calls == list(links[:3])

    with pytest.raises(ValueError, match="exactly 15"):
        junction.audit_brachiocephalic_network_links_3d(
            av, object(), records[:-1])

    wrong = list(records)
    wrong[-1] = {"bridge_key": (51011, 469), "target_key": (51011, 473)}
    with pytest.raises(ValueError, match="undeclared or duplicate"):
        junction.audit_brachiocephalic_network_links_3d(av, object(), wrong)


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
