"""Captured-pose surface-intersection stage for appended brachiocephalic veins.

This wrapper uses the existing resting-anatomy exact packed-Float32 verifier.
It checks each predicted bridge against captured anatomy through the frozen
P6 exact predicates, then exposes a separate source-bound solid-continuity and
local-calibre screen. It is not clinical or physiological qualification.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction

import numpy as np

BRIDGE_KEYS = ((51011, 525), (51011, 526))
PORT_TARGET_IDS = {525: (10, 469, 473), 526: (10, 468, 472)}
BRIDGE_VESSEL_PORTS = frozenset(
    (bridge_key, (51011, stable_id))
    for bridge_key in BRIDGE_KEYS
    for stable_id in PORT_TARGET_IDS[bridge_key[1]])
# Two source-bound left ulnar branch links are explicit vascular interfaces.
# They still require the exact source-derived face patch and the full solid,
# inside-both, connected-union, and independent section/calibre proof below.
ULNAR_BRANCH_PORTS = frozenset({
    ((51011, 479), (51011, 487)),
    ((51011, 481), (51011, 489)),
})
DECLARED_VESSEL_PORTS = BRIDGE_VESSEL_PORTS | ULNAR_BRANCH_PORTS

# One separately proven non-flow interface in the exact accepted capture used
# by the 150 network screen. It is deliberately not added to the flow graph.
AORTIC_ARCH_SOURCE_INTERFACE = ((51011, 7), (51011, 6))
AORTIC_ARCH_INTERFACE_REPORT_SHA256 = (
    "6c8dc412b7fc00d2140d3d9c72120c7fc2c56cd86dc5087a0296702fd01fdf38")
AORTIC_ARCH_INTERFACE_PREFLIGHT_SHA256 = (
    "2ecfd157b0c8f92d47409a4177a7661864f62836f90aaac094b43fbebac75078")
AORTIC_ARCH_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "61f15af8dc20fc7534dc231a560649cb04e0dd60abd76d42f8e5b52a9670daa6")
AORTIC_ARCH_INTERFACE_PAIRSET_SHA256 = (
    "478c7ad156fddfe8166c1e49a63d74f5d2c625c4cde39a8d3ffe1fd9dda2d212")
AORTIC_ARCH_INTERFACE_PAIR_COUNT = 36
# Field-002 is a separately re-audited rest-source candidate. Its full row-7
# fingerprint changed, so it has an independent evidence profile; the original
# proof004 profile above remains frozen for its original capture.
AORTIC_ARCH_FIELD002_INTERFACE_SCHEMA = (
    "numi.human.verified-aortic-arch-field-002-interface-handoff.v1")
AORTIC_ARCH_FIELD002_INTERFACE_STATUS = (
    "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_FIELD_002_INTERFACE")
AORTIC_ARCH_FIELD002_INTERFACE_REPORT_SHA256 = (
    "4bfb2ffa7d7267ea6f971e32becddc103f908795d74884fc8279ba4f35d8a584")
AORTIC_ARCH_FIELD002_INTERFACE_PREFLIGHT_SHA256 = (
    "a1d886fdb2f0ebea25d881dd04777a7065c1538624d943979dd5689a005c69dc")
AORTIC_ARCH_FIELD002_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "81102e440325e89cfd4fbf74ee3fe9a45bdbeaadee1375757a14ebf6b8ee5501")
AORTIC_ARCH_FIELD002_INTERFACE_PAIRSET_SHA256 = (
    "478c7ad156fddfe8166c1e49a63d74f5d2c625c4cde39a8d3ffe1fd9dda2d212")
AORTIC_ARCH_FIELD002_INTERFACE_PAIR_COUNT = 36
AORTIC_ARCH_SOURCE_ROWS = {
    (51011, 7): {
        "source_member": "FJ3411", "fma": "FMA3768", "name": "arch of aorta",
        "body_index": 20, "layer": 2,
        "first_vertex": 110808, "vertex_count": 381, "face_count": 670,
        "faces_sha256": "bd1b68b7ee6da377fd5b6a7d67339a2f1063b6cbef866e24149e1dad66083da8",
    },
    (51011, 6): {
        "source_member": "FJ3413", "fma": "FMA3736", "name": "ascending aorta",
        "body_index": 20, "layer": 2,
        "first_vertex": 110579, "vertex_count": 229, "face_count": 396,
        "faces_sha256": "a6884a09ba8975813151a31d851b7f712a8e59a3299f135485f7d8822c9649af",
    },
}
AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE = ((51011, 7), (51011, 8))
AORTIC_ARCH_DESCENDING_SOURCE_ROWS = {
    (51011, 7): AORTIC_ARCH_SOURCE_ROWS[(51011, 7)],
    (51011, 8): {
        "source_member": "FJ3427", "fma": "FMA3784",
        "name": "descending aorta", "body_index": 20, "layer": 2,
        "first_vertex": 111189, "vertex_count": 2351, "face_count": 4562,
        "faces_sha256": "ec5cd5366cd434c0b167740d5276f84ac35045de56968c10aa7f2931758115c2",
    },
}
AORTIC_ARCH_DESCENDING_INTERFACE_REPORT_SHA256 = (
    "d4905ba93f1e11829ad38d076056fa434fa70d37b9269eb9f1ef231e10aefcdf")
AORTIC_ARCH_DESCENDING_INTERFACE_PREFLIGHT_SHA256 = (
    "e00854e2883508e92ae0234353ec94e9df82649834ad7b4169ffd83b98fd8d0c")
AORTIC_ARCH_DESCENDING_INTERFACE_HANDOFF_SHA256 = (
    "56922c77b63edcdc2ac8164f70cb47485312d558055cad95547754e183985ae5")
AORTIC_ARCH_DESCENDING_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "61f15af8dc20fc7534dc231a560649cb04e0dd60abd76d42f8e5b52a9670daa6")
AORTIC_ARCH_DESCENDING_INTERFACE_PAIRSET_SHA256 = (
    "2bdd574e91e91c209caae11f6370113df7612a6857aaecf40310ac142cac7d8a")
AORTIC_ARCH_DESCENDING_INTERFACE_PAIR_COUNT = 28

# Fresh 7↔8 source-interface proof for the field-002 arch geometry. The older
# constants above remain bound to the earlier source/capture and are retained
# for historical replay only.
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_SCHEMA = (
    "numi.human.verified-aortic-source-interface-field-002-handoff.v1")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_STATUS = (
    "VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE_FIELD_002")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_REPORT_SHA256 = (
    "8e7dbe300a93f4cf458b1286f31e9ae130044c857cf1e377356b43afbd6ce42c")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PREFLIGHT_SHA256 = (
    "8fef3b0134c745129089694ce0a29ccab425aef1cf7b2c7e08adc44c357975c3")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_HANDOFF_SHA256 = (
    "e7b2a0a3b10023e1eca75627d523b662b97903205ca5f50bda027b26d345e91d")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "81102e440325e89cfd4fbf74ee3fe9a45bdbeaadee1375757a14ebf6b8ee5501")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PAIRSET_SHA256 = (
    "2bdd574e91e91c209caae11f6370113df7612a6857aaecf40310ac142cac7d8a")
AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PAIR_COUNT = 28
AORTIC_ARCH_CAROTID_SOURCE_INTERFACE = ((51011, 7), (51011, 465))
AORTIC_ARCH_CAROTID_INTERFACE_REPORT_SHA256 = (
    "c59e4ff043b36c5db2fcf9499922710e784e177eb7cf0878ddaf9331cd3054ed")
AORTIC_ARCH_CAROTID_INTERFACE_PREFLIGHT_SHA256 = (
    "8e31c88cbe52be5a0c93ccc53bd340d1282a5748e732b32f6f909ea75740e960")
AORTIC_ARCH_CAROTID_INTERFACE_HANDOFF_SHA256 = (
    "b20b8ec29de9d1ae4483baa2ebc470405bc72d78a1686c0b0e3c1b267f575ab2")
AORTIC_ARCH_CAROTID_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "3d47e3920a4cb60242d07c2e01786bfb91035e68d5bf316dfc1919e8cbacae12")
AORTIC_ARCH_CAROTID_INTERFACE_PAIRSET_SHA256 = (
    "f408e4ad80d0be34e95c96dcf50834b45bb7894f748335e78f8128c5d1aac90a")
AORTIC_ARCH_CAROTID_INTERFACE_PAIR_COUNT = 30
AORTIC_ARCH_CAROTID_SOURCE_ROWS = {
    (51011, 7): {
        "typed_identity": [51011, 7],
        "source_member": "FJ3411",
        "name": "arch of aorta",
        "fma": "FMA3768",
        "body_index": 20,
        "layer": 2,
        "record": [20, 110808, 381, 664590, 2010, 7, 2, 0],
        "first_vertex": 110808,
        "vertex_count": 381,
        "face_count": 670,
        "source_vertices_f32_sha256": (
            "3cd9483fa2a7a17c21026f719a6f3635a865a2b5a4223d8c5068a9fd50ee614a"),
        "source_faces_u32_sha256": (
            "bd1b68b7ee6da377fd5b6a7d67339a2f1063b6cbef866e24149e1dad66083da8"),
    },
    (51011, 465): {
        "typed_identity": [51011, 465],
        "source_member": "FJ3483",
        "name": "left common carotid artery",
        "fma": "FMA4058",
        "body_index": 22,
        "layer": 2,
        "record": [22, 1952328, 126, 10322379, 510, 465, 2, 0],
        "first_vertex": 1952328,
        "vertex_count": 126,
        "face_count": 170,
        "source_vertices_f32_sha256": (
            "08a2e279435e652d2f3a9075d04ddb059bcabbb00e18702ddfc11604de31b59c"),
        "source_faces_u32_sha256": (
            "aaa1711508c7a41b015dffeda30d8f991b3e60853c6db9d3410ef99deda5305b"),
    },
}


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



def _continuity_surface_fingerprints(verifier, key, value):
    prepared_type = getattr(verifier, "PreparedSurface", ())
    if isinstance(value, prepared_type):
        positions, faces, source_ids = (
            value.positions, value.faces, value.source_vertex_ids)
    else:
        _require(isinstance(value, (tuple, list)) and len(value) >= 4,
                 f"verified continuity handoff lacks source IDs for {key}")
        positions, faces, source_ids = value[0], value[1], value[3]
    positions = _float32_xyz(positions, name=f"handoff surface {key}")
    faces = _faces(faces, name=f"handoff surface {key}",
                   vertex_count=len(positions))
    source_ids = np.asarray(source_ids)
    _require(source_ids.ndim == 1 and len(source_ids) == len(positions)
             and source_ids.dtype.kind in "iu"
             and (source_ids.dtype.kind != "i" or int(source_ids.min()) >= 0)
             and len(np.unique(source_ids)) == len(source_ids),
             f"verified continuity handoff source IDs are invalid for {key}")
    source_ids = np.ascontiguousarray(source_ids, dtype="<u8")
    return {
        "identity": list(key),
        "positions_f32_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "faces_u32_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        "source_vertex_ids_u64_sha256": hashlib.sha256(source_ids.tobytes()).hexdigest(),
    }


def _validate_network_link_addendum(
    edge, link, ancestry, row_fingerprints, source_pairs, source_inventory,
    valid_sha,
):
    """Validate the two frozen independent-baseline collar extensions."""
    specs = {
        ((51011, 475), (51011, 479)): (
            "VERIFIED_INDEPENDENT_BASELINE_EXACT_ONE_FACE_COLLAR_EXTENSION",
            "single_pair"),
        ((51011, 525), (51011, 469)): (
            "VERIFIED_INDEPENDENT_BASELINE_DECLARED_COLLAR_ADDITIONS",
            "declared_pairs"),
    }
    expected = specs.get(edge)
    addendum = link.get("verified_baseline_collar_addendum")
    interfaces = link["port_interfaces"]
    additions = {}
    for role in ("baseline", "candidate"):
        value = interfaces[role].get("baseline_verified_declared_source_additions")
        _require(isinstance(value, Mapping)
                 and isinstance(value.get("pairs"), list)
                 and value.get("added_to_frozen_patch_cross_product") is False
                 and value.get("candidate_and_baseline_still_require_exact_points") is True
                 and type(value.get("used")) is bool,
                 f"{role} collar-addition declaration is incomplete for {edge}")
        pairs = []
        for pair in value["pairs"]:
            _require(isinstance(pair, list) and len(pair) == 2
                     and all(type(index) is int and index >= 0 for index in pair),
                     f"{role} collar-addition pair is malformed for {edge}")
            pairs.append(tuple(pair))
        additions[role] = tuple(sorted(pairs))
        _require(value["used"] is bool(pairs),
                 f"{role} collar-addition state is partial for {edge}")
    _require(additions["baseline"] == additions["candidate"],
             f"candidate and baseline collar additions disagree for {edge}")
    if expected is None:
        _require(addendum is None and not additions["baseline"],
                 f"undeclared source collar addendum appeared for {edge}")
        return None

    expected_status, mode = expected
    inventory_field = (
        "baseline_matched_collar_addendum" if edge[0][1] == 475
        else "source_baseline_matched_collar_addendum")
    _require(isinstance(source_inventory, Mapping)
             and source_inventory.get(inventory_field) == addendum,
             f"source collar addendum is stale against the current inventory for {edge}")
    slot_identity = source_inventory.get("native_source_slot_identity")
    _require(isinstance(slot_identity, Mapping)
             and slot_identity.get("bound_to_validated_actual_capture_receipt") is True
             and type(slot_identity.get("native_nha_gid_base_offset")) is int
             and addendum.get("native_nha_gid_base_offset")
             == slot_identity["native_nha_gid_base_offset"],
             f"source collar addendum uses stale native source-slot identity for {edge}")
    _require(isinstance(addendum, Mapping)
             and addendum.get("typed_link") == [list(edge[0]), list(edge[1])]
             and addendum.get("schema") == "numi.human.registered-source-collar-addendum.v1"
             and addendum.get("status") == expected_status
             and addendum.get("passed") is True
             and addendum.get("candidate_geometry_used") is False
             and addendum.get("clinical_or_anatomical_admission") is False
             and addendum.get("source_geometry_modified") is False
             and addendum.get(
                 "candidate_gate_still_requires_exact_pair_audit_and_solid_calibre_proof"
             ) is True,
             f"independent source collar addendum is missing, failed, or cross-linked for {edge}")

    baseline_identity = addendum.get("baseline_identity")
    evidence = addendum.get("source_evidence")
    lineage = addendum.get("baseline_surface_lineage")
    pair_sha = ancestry.get("source_face_pair_ancestry_sha256")
    _require(isinstance(baseline_identity, Mapping)
             and valid_sha(baseline_identity.get("accepted_capture_receipt_sha256"))
             and valid_sha(baseline_identity.get("accepted_pack_sha256"))
             and type(baseline_identity.get("accepted_step")) is int
             and baseline_identity["accepted_step"] == 0
             and baseline_identity.get("accepted_time_s") == 0.0
             and baseline_identity.get("candidate_geometry_used") is False
             and valid_sha(baseline_identity.get("invoked_anatomy_receipt_sha256"))
             and valid_sha(baseline_identity.get("source_nha_payload_sha256"))
             and isinstance(evidence, Mapping)
             and evidence.get("path") == ancestry.get("source_evidence_path")
             and evidence.get("sha256") == ancestry.get("source_evidence_sha256")
             and valid_sha(evidence.get("sha256"))
             and evidence.get("frozen_source_pairs_unchanged") is True
             and evidence.get("frozen_source_pair_count") == len(source_pairs[edge])
             and evidence.get("frozen_source_pairs_sha256") == pair_sha
             and isinstance(lineage, Mapping)
             and lineage.get("independent_from_actual_candidate_geometry") is True,
             f"independent source collar baseline identity is stale for {edge}")

    identities = addendum.get("source_identities")
    _require(isinstance(identities, Mapping)
             and set(identities) == {str(edge[0][1]), str(edge[1][1])},
             f"independent source collar identity set is partial for {edge}")
    for role, key in (("bridge", edge[0]), ("target", edge[1])):
        prefix = "parent" if role == "bridge" else "target"
        baseline = row_fingerprints[f"{role}_independent_source_baseline"]
        source = identities[str(key[1])]
        original = ancestry.get("source_identities", {}).get(str(key[1]))
        source_faces_sha = ancestry.get(
            "source_parent_faces_sha256" if role == "bridge"
            else "source_child_faces_sha256")
        _require(
            lineage.get(f"{prefix}_position_f32_sha256")
            == baseline["positions_f32_sha256"]
            and lineage.get(f"{prefix}_face_u32_sha256")
            == baseline["faces_u32_sha256"]
            and lineage.get(f"{prefix}_gids_u64_sha256")
            == baseline["source_vertex_ids_u64_sha256"]
            and isinstance(source, Mapping)
            and isinstance(original, Mapping)
            and source.get("stable_id") == key[1]
            and source.get("abi5_body_index") == original.get("body_index")
            and source.get("abi5_layer") == original.get("layer")
            and source.get("canonical_source_member") == original.get("source_member")
            and source.get("receipt_identity_matches_abi5_owner") is True
            and source.get("source_face_order_preserved") is True
            and source.get("source_faces_u32_sha256") == source_faces_sha
            and valid_sha(source.get("source_sha256"))
            and valid_sha(source.get("source_vertices6_f32_sha256"))
            and valid_sha(source.get("source_face_to_baseline_vertex_map_sha256")),
            f"independent source collar lineage is stale or cross-linked for {key}")

    bounded = addendum.get("bounded_port_region")
    exact = addendum.get("exact_p6_rat_contact")
    _require(isinstance(bounded, Mapping)
             and bounded.get("no_cross_product_expansion") is True
             and isinstance(exact, Mapping)
             and exact.get("coordinate_units")
             == "exact rational metres over emitted Float32 input coordinates",
             f"source collar extension is not bounded by exact coordinates for {edge}")
    if mode == "single_pair":
        extra = bounded.get("only_additional_allowed_face_pair")
        _require(isinstance(extra, list) and len(extra) == 2
                 and all(type(index) is int and index >= 0 for index in extra)
                 and exact.get("source_face_pair") == extra
                 and exact.get("contact_topology")
                 == "one positive-length segment, exactly two unique endpoints"
                 and exact.get("exact_endpoint_count") == 2
                 and isinstance(exact.get("endpoints"), list)
                 and len(exact["endpoints"]) == 2
                 and bounded.get(
                     "all_contact_points_and_segment_inside_parent_face_and_bounded_target_face"
                 ) is True
                 and not additions["baseline"],
                 f"single-pair source collar proof is partial for {edge}")
        additional_pairs = (tuple(extra),)
    else:
        raw_pairs = bounded.get("additional_allowed_face_pairs")
        _require(isinstance(raw_pairs, list) and bool(raw_pairs)
                 and isinstance(exact.get("per_pair_proofs"), list)
                 and isinstance(bounded.get("pair_regions"), list),
                 f"declared source collar pair proof is partial for {edge}")
        additional_pairs = []
        for pair in raw_pairs:
            _require(isinstance(pair, list) and len(pair) == 2
                     and all(type(index) is int and index >= 0 for index in pair),
                     f"declared source collar pair is malformed for {edge}")
            additional_pairs.append(tuple(pair))
        additional_pairs = tuple(sorted(additional_pairs))
        _require(additions["baseline"] == additional_pairs,
                 f"source interface additions differ from exact addendum for {edge}")
        exact_pairs = []
        for proof in exact["per_pair_proofs"]:
            _require(isinstance(proof, Mapping)
                     and proof.get("contact_topology") == "one positive-length segment"
                     and proof.get("exact_contact_point_count") == 2,
                     f"declared source collar exact contact is incomplete for {edge}")
            exact_pairs.append(tuple(proof.get("face_pair", ())))
        _require(tuple(sorted(exact_pairs)) == additional_pairs
                 and len(exact_pairs) == len(additional_pairs),
                 f"declared source collar exact contacts are cross-linked for {edge}")
        region_pairs = []
        for region in bounded["pair_regions"]:
            connection = (region.get("bounded_contact_connection")
                          if isinstance(region, Mapping) else None)
            _require(isinstance(connection, Mapping)
                     and connection.get("passed") is True
                     and region.get("endpoint_rule")
                     == "one_shared_edge_endpoint_or_exact_indexed_contact_chain",
                     f"declared source collar region lacks an indexed connection for {edge}")
            region_pairs.append(tuple(region.get("face_pair", ())))
        _require(tuple(sorted(region_pairs)) == additional_pairs
                 and len(region_pairs) == len(additional_pairs),
                 f"declared source collar regions are partial or cross-linked for {edge}")

    return {
        "source_addendum_sha256": hashlib.sha256(
            json.dumps(addendum, sort_keys=True, separators=(",", ":"), allow_nan=False)
            .encode("utf-8")).hexdigest(),
        "source_evidence_path": evidence["path"],
        "source_evidence_sha256": evidence["sha256"],
        "independent_additional_source_face_pairs": [
            list(pair) for pair in additional_pairs],
        "baseline_surface_lineage_bound": True,
        "candidate_geometry_used_by_addendum": False,
    }


def _verified_network_link_handoff(
    verifier, surfaces, edges, source_pairs, continuity, source_inventory=None,
):
    """Bind proven vascular joins to exact current rows before skipping duplicate pair tests."""
    _require(isinstance(continuity, Mapping)
             and continuity.get("status") == "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED"
             and continuity.get("passed") is True
             and type(continuity.get("requested_link_count")) is int
             and type(continuity.get("evaluated_link_count")) is int
             and continuity["requested_link_count"] == len(edges)
             and continuity["evaluated_link_count"] == len(edges)
             and isinstance(continuity.get("links"), list)
             and len(continuity["links"]) == len(edges),
             "all-target handoff requires a complete passing fail-first link report")
    try:
        report_bytes = json.dumps(
            continuity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "brachiocephalic captured-pose geometry: continuity report is not canonical JSON"
        ) from exc
    continuity_sha = hashlib.sha256(report_bytes).hexdigest()

    def valid_sha(value):
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))

    addendum_edges = {
        ((51011, 475), (51011, 479)),
        ((51011, 525), (51011, 469)),
    }
    _require(source_inventory is None or (
                 isinstance(source_inventory, Mapping)
                 and source_inventory.get(
                     "candidate_gid_values_were_read_from_actual_capture_pack") is True
                 and source_inventory.get(
                     "baseline_coordinates_are_independently_cpu_reconstructed") is True
                 and source_inventory.get(
                     "baseline_gid_values_are_transported_current_nha_source_slots") is True
                 and source_inventory.get(
                     "source_face_arrays_bound_by_exact_ordered_abi5_corner_ancestry") is True
                 and isinstance(source_inventory.get("candidate_surfaces"), Mapping)
                 and isinstance(source_inventory.get("independent_source_baselines"), Mapping)
                 and isinstance(source_inventory.get(
                     "candidate_baseline_gid_identity_by_typed_row"), Mapping)
                 and isinstance(source_inventory.get(
                     "source_face_ancestry_by_typed_row"), Mapping)),
             "all-target handoff source/baseline inventory is malformed")
    _require(not (addendum_edges & set(edges)) or source_inventory is not None,
             "special source-collar links require the current source/baseline inventory")
    edge_set = set(edges)
    links_by_edge = {}
    for link in continuity["links"]:
        _require(isinstance(link, Mapping) and link.get("passed") is True,
                 "all-target handoff contains a failed or malformed link proof")
        first = _identity(link.get("bridge"))
        second = _identity(link.get("target"))
        edge = (first, second)
        _require(edge in edge_set and edge not in links_by_edge,
                 "all-target handoff link identity is undeclared or duplicated")
        links_by_edge[edge] = link
    _require(set(links_by_edge) == edge_set,
             "all-target handoff does not cover every declared network edge")

    handoffs = []
    for edge in edges:
        link = links_by_edge[edge]
        ancestry = link.get("declared_network_link_source_ancestry")
        pairs = source_pairs[edge]
        pair_sha = hashlib.sha256(
            np.ascontiguousarray(pairs, dtype="<i8").tobytes()).hexdigest()
        _require(isinstance(ancestry, Mapping)
                 and ancestry.get("typed_link") == [list(edge[0]), list(edge[1])]
                 and ancestry.get("source_local_face_indices_bound") is True
                 and ancestry.get("clinical_or_anatomical_admission") is False
                 and isinstance(ancestry.get("source_evidence_path"), str)
                 and valid_sha(ancestry.get("source_evidence_sha256"))
                 and valid_sha(ancestry.get("source_parent_faces_sha256"))
                 and valid_sha(ancestry.get("source_child_faces_sha256"))
                 and type(ancestry.get("source_face_pair_count")) is int
                 and ancestry["source_face_pair_count"] == len(pairs)
                 and ancestry.get("source_face_pair_ancestry")
                 == [list(pair) for pair in pairs]
                 and ancestry.get("source_face_pair_ancestry_sha256") == pair_sha,
                 f"network link source face ancestry is not bound for {edge}")

        topology = link.get("source_topology_gate")
        ancestry_gate = link.get("candidate_baseline_ancestry_gate")
        solid_gates = link.get("solid_gates")
        interfaces = link.get("port_interfaces")
        calibres = link.get("independent_collar_calibre")
        _require(isinstance(topology, Mapping) and topology.get("passed") is True
                 and isinstance(topology.get("surfaces"), Mapping)
                 and isinstance(ancestry_gate, Mapping)
                 and ancestry_gate.get("passed") is True
                 and isinstance(ancestry_gate.get("vessels"), Mapping)
                 and isinstance(solid_gates, Mapping)
                 and isinstance(interfaces, Mapping)
                 and isinstance(calibres, Mapping),
                 f"network link lacks complete source and candidate evidence for {edge}")
        topology_surfaces = topology["surfaces"]
        ancestry_vessels = ancestry_gate["vessels"]

        def validate_surface_row(row, key, *, name):
            _require(isinstance(row, Mapping) and row.get("identity") == list(key)
                     and row.get("closed_self_free_single_component") is True
                     and row.get("exact_self_audit_count") == 0
                     and row.get("referenced_face_component_count") == 1
                     and isinstance(row.get("prepared_surface"), Mapping),
                     f"{name} lacks a closed single-component proof for {key}")
            prepared = row["prepared_surface"]
            positions_sha = prepared.get("positions_sha256")
            faces_sha = prepared.get("faces_sha256")
            source_ids_sha = row.get("source_vertex_ids_sha256")
            _require(prepared.get("identity") == list(key)
                     and valid_sha(positions_sha) and valid_sha(faces_sha)
                     and valid_sha(source_ids_sha),
                     f"{name} lacks exact position, face, or source-slot hashes for {key}")
            return {
                "identity": list(key),
                "positions_f32_sha256": positions_sha,
                "faces_u32_sha256": faces_sha,
                "source_vertex_ids_u64_sha256": source_ids_sha,
                "closed_self_free_single_component": True,
                "exact_self_audit_count": 0,
                "referenced_face_component_count": 1,
            }

        def validate_solid(solid, edge, *, role):
            _require(isinstance(solid, Mapping)
                     and solid.get("passed") is True
                     and solid.get("manifold_inputs_passed") is True
                     and solid.get("positive_intersection_volume") is True
                     and isinstance(solid.get("intersection_volume_m3"), (int, float))
                     and np.isfinite(solid["intersection_volume_m3"])
                     and solid["intersection_volume_m3"] > 0.0
                     and solid.get("union_connected_numerical_solid") is True
                     and solid.get("union_component_count") == 1
                     and solid.get(
                         "union_lineage_has_both_sources_without_generated_or_unmapped_faces"
                     ) is True
                     and isinstance(solid.get("exact_inside_both_witness"), Mapping)
                     and solid["exact_inside_both_witness"].get("first_location") == "inside"
                     and solid["exact_inside_both_witness"].get("second_location") == "inside"
                     and "exact rational point location" in solid[
                         "exact_inside_both_witness"].get("predicate", "")
                     and isinstance(solid.get("volume_accounting"), Mapping)
                     and solid["volume_accounting"].get("passed") is True,
                     f"{role} link lacks positive overlap, exact inside-both, or connected-solid evidence for {edge}")
            lineage = solid.get("union_float64_mesh64_lineage")
            _require(isinstance(lineage, Mapping)
                     and lineage.get("generated_face_count") == 0
                     and lineage.get("unmapped_face_count") == 0
                     and set(lineage.get("run_original_ids", []))
                     == {edge[0][1], edge[1][1]}
                     and solid.get("floating_boolean_exact_topology_proven") is False,
                     f"{role} union lineage or numerical-topology limits are incomplete for {edge}")
            return {
                "intersection_volume_m3": solid["intersection_volume_m3"],
                "exact_inside_both": [
                    solid["exact_inside_both_witness"]["first_location"],
                    solid["exact_inside_both_witness"]["second_location"]],
                "numerical_union_component_count": solid["union_component_count"],
                "numerical_boolean_topology_is_not_exact": True,
                "volume_accounting_passed": True,
                "union_lineage": lineage,
            }

        def validate_interface(interface, edge, pair_sha, *, role):
            _require(isinstance(interface, Mapping)
                     and interface.get("passed") is True
                     and interface.get("typed_port") == [list(edge[0]), list(edge[1])]
                     and interface.get("observed_pairs_within_frozen_face_patch_cross_product") is True
                     and interface.get("unexpected_face_pairs") == []
                     and type(interface.get("exact_interface_point_count")) is int
                     and interface["exact_interface_point_count"] > 0
                     and valid_sha(interface.get("exact_interface_points_sha256"))
                     and interface.get("frozen_face_pairs_sha256") == pair_sha
                     and isinstance(interface.get("documented_vessel_junction_scope"), Mapping)
                     and interface["documented_vessel_junction_scope"].get("passed") is True,
                     f"{role} link lacks bounded matched-collar proof for {edge}")
            return {
                "exact_interface_point_count": interface["exact_interface_point_count"],
                "exact_interface_points_sha256": interface["exact_interface_points_sha256"],
                "frozen_face_pairs_sha256": pair_sha,
            }

        def validate_calibre(calibre, key, role, candidate_interface,
                             baseline_interface):
            _require(isinstance(calibre, Mapping)
                     and calibre.get("passed") is True
                     and calibre.get("owner_id") == key[1]
                     and calibre.get("role") == role
                     and type(calibre.get("baseline_station_witness_point_count")) is int
                     and calibre["baseline_station_witness_point_count"] > 0
                     and valid_sha(calibre.get("baseline_interface_point_sha256"))
                     and isinstance(candidate_interface, Mapping)
                     and isinstance(baseline_interface, Mapping)
                     and valid_sha(candidate_interface.get(
                         "exact_interface_points_sha256"))
                     and valid_sha(baseline_interface.get(
                         "exact_interface_points_sha256"))
                     and type(candidate_interface.get("exact_interface_point_count")) is int
                     and type(baseline_interface.get("exact_interface_point_count")) is int
                     and calibre["baseline_interface_point_sha256"]
                     == baseline_interface["exact_interface_points_sha256"]
                     and calibre["baseline_station_witness_point_count"]
                     == baseline_interface["exact_interface_point_count"],
                     f"{role} lacks an independent source station for {key}")
            station = calibre.get("station")
            baseline_station = calibre.get("baseline_station_transported_by_source_ancestry")
            ancestry_fields = (
                "material_anchor_local_vertex_ids",
                "material_anchor_source_vertex_ids",
                "material_anchor_weights",
                "frozen_collar_face_ids",
                "tangent_support_local_vertex_ids",
                "tangent_support_source_vertex_ids",
            )
            _require(isinstance(station, Mapping)
                     and station.get("transported_from_source_vertex_ancestry") is True
                     and station.get("exact_interface_witness_points_sha256")
                     == candidate_interface["exact_interface_points_sha256"]
                     and isinstance(baseline_station, Mapping)
                     and baseline_station.get("transported_from_source_vertex_ancestry") is True
                     and baseline_station.get("exact_interface_witness_points_sha256")
                     == station.get("exact_interface_witness_points_sha256")
                     and all(station.get(field) == baseline_station.get(field)
                             for field in ancestry_fields)
                     and station.get("inferred") is True
                     and baseline_station.get("inferred") is True,
                     f"{role} station is not transported from source ancestry for {key}")
            anchor_local = station.get("material_anchor_local_vertex_ids")
            anchor_source = station.get("material_anchor_source_vertex_ids")
            anchor_weights = station.get("material_anchor_weights")
            collar_faces = station.get("frozen_collar_face_ids")
            tangent_local = station.get("tangent_support_local_vertex_ids")
            tangent_source = station.get("tangent_support_source_vertex_ids")
            _require(
                isinstance(anchor_local, list) and bool(anchor_local)
                and all(type(value) is int and value >= 0 for value in anchor_local)
                and len(set(anchor_local)) == len(anchor_local)
                and isinstance(anchor_source, list)
                and len(anchor_source) == len(anchor_local)
                and all(type(value) is int and value >= 0 for value in anchor_source)
                and len(set(anchor_source)) == len(anchor_source)
                and isinstance(anchor_weights, list)
                and len(anchor_weights) == len(anchor_local)
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and np.isfinite(value) and value > 0.0 for value in anchor_weights)
                and abs(sum(anchor_weights) - 1.0) <= 1e-12
                and isinstance(collar_faces, list) and bool(collar_faces)
                and all(type(value) is int and value >= 0 for value in collar_faces)
                and len(set(collar_faces)) == len(collar_faces)
                and isinstance(tangent_local, list) and len(tangent_local) == 2
                and all(type(value) is int and value >= 0 for value in tangent_local)
                and isinstance(tangent_source, list) and len(tangent_source) == 2
                and all(type(value) is int and value >= 0 for value in tangent_source),
                f"{role} source station ancestry is incomplete for {key}")
            for section_name in ("candidate_section", "baseline_section"):
                section = calibre.get(section_name)
                _require(isinstance(section, Mapping)
                         and section.get("passed") is True
                         and section.get("loop_count") == 1
                         and section.get("lineage_complete") is True
                         and section.get("generated_face_count") == 0
                         and section.get("unmapped_face_count") == 0
                         and isinstance(section.get("area_m2"), (int, float))
                         and np.isfinite(section["area_m2"])
                         and section["area_m2"] > 0.0
                         and isinstance(section.get("minimum_feret_width_m"), (int, float))
                         and np.isfinite(section["minimum_feret_width_m"])
                         and section["minimum_feret_width_m"] > 0.0,
                         f"{role} {section_name} is incomplete for {key}")
            comparison = calibre.get("candidate_to_baseline_comparison")
            _require(isinstance(comparison, Mapping)
                     and comparison.get("passed") is True
                     and comparison.get("wide_bounds_are_not_preservation_success") is True,
                     f"{role} lacks independent source-calibre comparison for {key}")
            ratios = comparison.get("candidate_to_baseline_ratios")
            bounds = comparison.get("declared_preservation_intervals")
            _require(isinstance(ratios, Mapping) and isinstance(bounds, Mapping)
                     and bounds.get("area") == [0.8, 1.25]
                     and bounds.get("minimum_feret_width") == [0.9, 1.1]
                     and all(isinstance(ratios.get(metric), (int, float))
                             and np.isfinite(ratios[metric])
                             and bounds[metric][0] <= ratios[metric] <= bounds[metric][1]
                             for metric in ("area", "minimum_feret_width")),
                     f"{role} calibre ratios exceed the declared source bounds for {key}")
            return {
                "owner_id": key[1],
                "baseline_interface_point_sha256": calibre[
                    "baseline_interface_point_sha256"],
                "baseline_station_witness_point_count": calibre[
                    "baseline_station_witness_point_count"],
                "candidate_to_baseline_ratios": dict(ratios),
                "declared_preservation_intervals": dict(bounds),
            }

        row_fingerprints = {}
        solid_proofs = {}
        interface_proofs = {}
        calibre_proofs = {}
        for role, key in (("bridge", edge[0]), ("target", edge[1])):
            source_identities = ancestry.get("source_identities")
            source_identity = (
                source_identities.get(str(key[1]))
                if isinstance(source_identities, Mapping) else None)
            canonical_identity = (
                source_identity.get("canonical_source_identity")
                if isinstance(source_identity, Mapping) else None)
            _require(isinstance(source_identity, Mapping)
                     and source_identity.get("typed_identity") == list(key)
                     and isinstance(canonical_identity, Mapping)
                     and canonical_identity.get("stable_id") == key[1]
                     and canonical_identity.get("receipt_identity_matches_abi5_owner") is True,
                     f"network link source metadata is not bound to {key}")
            ancestry_row = ancestry_vessels.get(role)
            _require(isinstance(ancestry_row, Mapping)
                     and ancestry_row.get("passed") is True
                     and ancestry_row.get("same_ordered_source_face_ancestry") is True
                     and ancestry_row.get("same_source_vertex_ancestry") is True
                     and ancestry_row.get("source_orientation_preserved") is True
                     and ancestry_row.get("source_volumes_finite_and_positive") is True,
                     f"{role} lacks independent source face/slot ancestry for {key}")

            candidate_row = validate_surface_row(
                topology_surfaces.get(f"candidate_{role}"), key,
                name="candidate topology")
            baseline_row = validate_surface_row(
                topology_surfaces.get(f"baseline_{role}"), key,
                name="independent source-baseline topology")
            fingerprint = _continuity_surface_fingerprints(
                verifier, key, surfaces[key])
            _require(candidate_row["positions_f32_sha256"]
                     == fingerprint["positions_f32_sha256"]
                     and candidate_row["faces_u32_sha256"]
                     == fingerprint["faces_u32_sha256"]
                     and candidate_row["source_vertex_ids_u64_sha256"]
                     == fingerprint["source_vertex_ids_u64_sha256"]
                     and ancestry_row.get("candidate_face_ancestry_sha256")
                     == fingerprint["faces_u32_sha256"]
                     and ancestry_row.get("candidate_source_vertex_ids_sha256")
                     == fingerprint["source_vertex_ids_u64_sha256"]
                     and ancestry_row.get("baseline_face_ancestry_sha256")
                     == baseline_row["faces_u32_sha256"]
                     and ancestry_row.get("baseline_source_vertex_ids_sha256")
                     == baseline_row["source_vertex_ids_u64_sha256"]
                     and ancestry_row.get("candidate_face_ancestry_sha256")
                     == ancestry_row.get("baseline_face_ancestry_sha256")
                     and ancestry_row.get("candidate_source_vertex_ids_sha256")
                     == ancestry_row.get("baseline_source_vertex_ids_sha256"),
                     f"candidate or independent baseline bytes differ from link proof for {key}")
            source_face_sha = ancestry.get(
                "source_parent_faces_sha256" if role == "bridge"
                else "source_child_faces_sha256")
            if source_inventory is None:
                _require(source_face_sha == fingerprint["faces_u32_sha256"],
                         f"unmapped source face bytes differ from current row for {key}")
            else:
                ancestry_inventory = source_inventory[
                    "source_face_ancestry_by_typed_row"].get(
                        f"{key[0]}:{key[1]}")
                _require(
                    isinstance(ancestry_inventory, Mapping)
                    and ancestry_inventory.get("typed_identity") == list(key)
                    and ancestry_inventory.get("source_faces_u32_sha256") == source_face_sha
                    and ancestry_inventory.get("source_abi5_face_array_sha256")
                    == source_face_sha
                    and ancestry_inventory.get("capture_faces_u32_sha256")
                    == fingerprint["faces_u32_sha256"]
                    and ancestry_inventory.get("packed_face_array_sha256")
                    == fingerprint["faces_u32_sha256"]
                    and ancestry_inventory.get(
                        "ordered_source_face_ids_are_preserved") is True
                    and ancestry_inventory.get("status")
                    == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
                    and type(ancestry_inventory.get("source_face_count")) is int
                    and type(ancestry_inventory.get("capture_face_count")) is int
                    and ancestry_inventory["source_face_count"]
                    == ancestry_inventory["capture_face_count"]
                    and valid_sha(ancestry_inventory.get("map_sha256"))
                    and valid_sha(ancestry_inventory.get(
                        "source_local_vertex_to_capture_global_gid_sha256")),
                    f"source-to-capture face ancestry is incomplete for {key}")
            for source_kind, source_row in (
                    ("candidate", candidate_row), ("baseline", baseline_row)):
                solid = solid_gates.get(source_kind)
                solid_input = (solid.get(f"{role}_manifold_input")
                               if isinstance(solid, Mapping) else None)
                _require(isinstance(solid_input, Mapping)
                         and solid_input.get("source_position_sha256")
                         == source_row["positions_f32_sha256"]
                         and solid_input.get("source_face_sha256")
                         == source_row["faces_u32_sha256"]
                         and solid_input.get("source_vertex_ancestry_sha256")
                         == source_row["source_vertex_ids_u64_sha256"],
                         f"{source_kind} solid input hashes differ from {key}")
                solid_proofs[f"{source_kind}_{role}"] = validate_solid(
                    solid, edge, role=source_kind)
            row_fingerprints[f"{role}_candidate"] = candidate_row
            row_fingerprints[f"{role}_independent_source_baseline"] = baseline_row

            for source_kind in ("candidate", "baseline"):
                interface_proofs[f"{source_kind}_{role}"] = validate_interface(
                    interfaces.get(source_kind), edge, pair_sha, role=source_kind)
            calibre_proofs[role] = validate_calibre(
                calibres.get(role), key, role,
                interface_proofs[f"candidate_{role}"],
                interface_proofs[f"baseline_{role}"])

            if source_inventory is not None:
                inventory_key = f"{key[0]}:{key[1]}"
                candidate_inventory = source_inventory["candidate_surfaces"].get(
                    inventory_key)
                baseline_inventory = source_inventory[
                    "independent_source_baselines"].get(inventory_key)
                gid_inventory = source_inventory[
                    "candidate_baseline_gid_identity_by_typed_row"].get(inventory_key)
                _require(
                    isinstance(candidate_inventory, Mapping)
                    and candidate_inventory.get("typed_identity") == list(key)
                    and candidate_inventory.get("positions_f32_sha256")
                    == candidate_row["positions_f32_sha256"]
                    and candidate_inventory.get("faces_u32_sha256")
                    == candidate_row["faces_u32_sha256"]
                    and candidate_inventory.get("captured_global_vertex_ids_u64_sha256")
                    == candidate_row["source_vertex_ids_u64_sha256"]
                    and isinstance(baseline_inventory, Mapping)
                    and baseline_inventory.get("typed_identity") == list(key)
                    and baseline_inventory.get("positions_f32_sha256")
                    == baseline_row["positions_f32_sha256"]
                    and baseline_inventory.get("faces_u32_sha256")
                    == baseline_row["faces_u32_sha256"]
                    and baseline_inventory.get("captured_global_vertex_ids_u64_sha256")
                    == baseline_row["source_vertex_ids_u64_sha256"]
                    and isinstance(gid_inventory, Mapping)
                    and gid_inventory.get("typed_identity") == list(key)
                    and gid_inventory.get("numeric_gid_values_equal") is True
                    and gid_inventory.get("candidate_gid_values_sha256")
                    == candidate_row["source_vertex_ids_u64_sha256"]
                    and gid_inventory.get("baseline_gid_values_sha256")
                    == baseline_row["source_vertex_ids_u64_sha256"],
                    f"current source/baseline inventory is stale for {key}")

        addendum_proof = _validate_network_link_addendum(
            edge, link, ancestry, row_fingerprints, source_pairs,
            source_inventory, valid_sha)
        handoffs.append({
            "typed_link": [list(edge[0]), list(edge[1])],
            "continuity_report_sha256": continuity_sha,
            "source_face_pair_ancestry_sha256": pair_sha,
            "independent_source_collar_addendum": addendum_proof,
            "candidate_and_independent_source_rows": row_fingerprints,
            "candidate_and_independent_solid_proofs": solid_proofs,
            "candidate_and_independent_interface_proofs": interface_proofs,
            "independent_source_calibre_proofs": calibre_proofs,
        })
    return handoffs



def _aortic_arch_surface_fingerprint(value, *, key):
    """Hash one exact actual-capture surface for the source-bound 6↔7 handoff."""
    if hasattr(value, "positions") and hasattr(value, "faces"):
        positions, faces = value.positions, value.faces
        normals = getattr(value, "normals", None)
        source_ids = getattr(value, "source_vertex_ids", None)
    else:
        _require(isinstance(value, (tuple, list)) and len(value) >= 4,
                 f"aortic interface surface {key} lacks packed geometry/normals/GIDs")
        positions, faces, normals, source_ids = value[:4]
    positions = _float32_xyz(positions, name=f"aortic interface {key}")
    faces = _faces(faces, name=f"aortic interface {key}",
                   vertex_count=len(positions))
    _require(normals is not None,
             f"aortic interface surface {key} lacks actual packed normals")
    normals = np.asarray(normals)
    _require(normals.dtype == np.dtype("<f4")
             and normals.shape == positions.shape
             and np.isfinite(normals).all(),
             f"aortic interface surface {key} normals are not finite Float32 XYZ")
    normals = np.ascontiguousarray(normals, dtype="<f4")
    source_ids = np.asarray(source_ids)
    _require(source_ids.shape == (len(positions),)
             and source_ids.dtype.kind in "iu"
             and (source_ids.dtype.kind != "i" or int(source_ids.min()) >= 0)
             and len(np.unique(source_ids)) == len(source_ids),
             f"aortic interface surface {key} has invalid actual pack GIDs")
    source_ids = np.ascontiguousarray(source_ids, dtype="<u8")
    return {
        "typed_identity": list(key),
        "positions_f32_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "faces_u32_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        "normals_f32_sha256": hashlib.sha256(normals.tobytes()).hexdigest(),
        "source_vertex_ids_u64_sha256": hashlib.sha256(source_ids.tobytes()).hexdigest(),
        "vertex_count": len(positions),
        "face_count": len(faces),
    }


def _validate_source_interface_certificate(
    certificate, surfaces, network_keys, *, edge, schema, status,
    evidence_report_sha256, source_preflight_sha256, capture_input_bundle_sha256,
    exact_pair_count, exact_pairset_sha256, source_rows, pair_dtype="<u4",
):
    """Validate one frozen source-bound non-flow vessel interface and live arrays."""
    edge = (_identity(edge[0]), _identity(edge[1]))
    label = f"source interface {edge[0]}->{edge[1]}"
    _require(isinstance(certificate, dict)
             and certificate.get("schema") == schema
             and certificate.get("status") == status
             and certificate.get("passed") is True
             and certificate.get("typed_link") == [list(edge[0]), list(edge[1])],
             f"{label} certificate is missing or names another typed pair")
    _require(certificate.get("evidence_report_sha256") == evidence_report_sha256
             and certificate.get("source_preflight_sha256") == source_preflight_sha256
             and certificate.get("capture_input_bundle_sha256")
             == capture_input_bundle_sha256,
             f"{label} certificate is stale or belongs to another capture")
    _require(edge[0] in network_keys and edge[1] not in network_keys,
             f"{label} must be one exact network-to-nonnetwork handoff")
    _require(certificate.get("source_topology_passed") is True
             and certificate.get("interface_gate_passed") is True
             and certificate.get("solid_gate_passed") is True
             and certificate.get("independent_source_collar_calibre_passed") is True
             and certificate.get("candidate_pairs_equal_frozen_source_pairs") is True
             and certificate.get("baseline_pairs_equal_frozen_source_pairs") is True
             and certificate.get("candidate_baseline_exact_pairsets_equal") is True,
             f"{label} lacks topology/interface/solid/calibre proof")
    proof_rows = certificate.get("source_rows")
    expected_row_keys = {f"{key[0]}:{key[1]}" for key in edge}
    _require(isinstance(proof_rows, dict) and set(proof_rows) == expected_row_keys,
             f"{label} does not bind exactly its two source rows")
    _require(isinstance(source_rows, Mapping)
             and set(source_rows) == set(edge),
             f"{label} expected source identity map does not match the typed pair")
    for key, expected in source_rows.items():
        observed = proof_rows[f"{key[0]}:{key[1]}"]
        _require(observed == expected,
                 f"{label} source identity/face ancestry differs for {key}")
    raw_pairs = certificate.get("exact_face_pairs")
    _require(isinstance(raw_pairs, list) and len(raw_pairs) == exact_pair_count
             and all(isinstance(pair, list) and len(pair) == 2
                     and all(type(face) is int and face >= 0 for face in pair)
                     for pair in raw_pairs),
             f"{label} exact face-pair set is malformed")
    pairs = tuple((pair[0], pair[1]) for pair in raw_pairs)
    left_row = source_rows.get(edge[0])
    right_row = source_rows.get(edge[1])
    _require(isinstance(left_row, Mapping) and type(left_row.get("face_count")) is int
             and isinstance(right_row, Mapping) and type(right_row.get("face_count")) is int
             and all(first < left_row["face_count"] and second < right_row["face_count"]
                     for first, second in pairs),
             f"{label} face-pair indices exceed their bound source rows")
    _require(pair_dtype in ("<u4", "<i8")
             and pairs == tuple(sorted(set(pairs)))
             and hashlib.sha256(np.asarray(pairs, dtype=pair_dtype).tobytes()).hexdigest()
             == exact_pairset_sha256
             and certificate.get("exact_face_pairset_sha256")
             == exact_pairset_sha256,
             f"{label} exact source-interface pair set differs from its frozen hash")
    fingerprints = certificate.get("candidate_surface_bindings")
    labels = {f"{key[0]}:{key[1]}" for key in edge}
    _require(isinstance(fingerprints, dict) and set(fingerprints) == labels,
             f"{label} certificate does not bind both actual capture surfaces")
    for key in edge:
        key_label = f"{key[0]}:{key[1]}"
        observed = _aortic_arch_surface_fingerprint(surfaces.get(key), key=key)
        _require(fingerprints[key_label] == observed,
                 f"{label} certificate is stale for actual surface {key}")
    return {
        "schema": certificate["schema"],
        "status": certificate["status"],
        "passed": True,
        "typed_link": [list(edge[0]), list(edge[1])],
        "evidence_report_sha256": certificate["evidence_report_sha256"],
        "source_preflight_sha256": certificate["source_preflight_sha256"],
        "capture_input_bundle_sha256": certificate["capture_input_bundle_sha256"],
        "exact_face_pair_count": len(pairs),
        "exact_face_pairset_sha256": exact_pairset_sha256,
        "candidate_surface_bindings": fingerprints,
        "source_rows": proof_rows,
        "closed_connected_embedded_solid_proof": True,
        "exact_inside_both_witness_and_connected_union_proof": True,
        "independent_source_collar_calibre_proof": True,
        "skipped_duplicate_generic_pair_only": True,
        "anatomical_admission": False,
    }


def _validate_aortic_arch_interface_certificate(certificate, surfaces, network_keys):
    """Validate the frozen arch/ascending aorta handoff."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_SOURCE_INTERFACE,
        schema="numi.human.verified-aortic-arch-interface-handoff.v1",
        status="VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE",
        evidence_report_sha256=AORTIC_ARCH_INTERFACE_REPORT_SHA256,
        source_preflight_sha256=AORTIC_ARCH_INTERFACE_PREFLIGHT_SHA256,
        capture_input_bundle_sha256=AORTIC_ARCH_INTERFACE_CAPTURE_BUNDLE_SHA256,
        exact_pair_count=AORTIC_ARCH_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=AORTIC_ARCH_INTERFACE_PAIRSET_SHA256,
        source_rows=AORTIC_ARCH_SOURCE_ROWS,
        pair_dtype="<u4",
    )


def _validate_aortic_arch_field002_interface_certificate(
        certificate, surfaces, network_keys):
    """Validate the separately re-audited field-002 arch source surface."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_SOURCE_INTERFACE,
        schema=AORTIC_ARCH_FIELD002_INTERFACE_SCHEMA,
        status=AORTIC_ARCH_FIELD002_INTERFACE_STATUS,
        evidence_report_sha256=AORTIC_ARCH_FIELD002_INTERFACE_REPORT_SHA256,
        source_preflight_sha256=AORTIC_ARCH_FIELD002_INTERFACE_PREFLIGHT_SHA256,
        capture_input_bundle_sha256=AORTIC_ARCH_FIELD002_INTERFACE_CAPTURE_BUNDLE_SHA256,
        exact_pair_count=AORTIC_ARCH_FIELD002_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=AORTIC_ARCH_FIELD002_INTERFACE_PAIRSET_SHA256,
        source_rows=AORTIC_ARCH_SOURCE_ROWS,
        pair_dtype="<u4",
    )


def _validate_aortic_arch_descending_interface_certificate(
    certificate, surfaces, network_keys,
):
    """Validate the exact, separately proven 7↔8 source interface."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE,
        schema="numi.human.verified-aortic-source-interface-handoff.v1",
        status="VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE",
        evidence_report_sha256=AORTIC_ARCH_DESCENDING_INTERFACE_REPORT_SHA256,
        source_preflight_sha256=AORTIC_ARCH_DESCENDING_INTERFACE_PREFLIGHT_SHA256,
        capture_input_bundle_sha256=(
            AORTIC_ARCH_DESCENDING_INTERFACE_CAPTURE_BUNDLE_SHA256),
        exact_pair_count=AORTIC_ARCH_DESCENDING_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=AORTIC_ARCH_DESCENDING_INTERFACE_PAIRSET_SHA256,
        source_rows=AORTIC_ARCH_DESCENDING_SOURCE_ROWS,
        pair_dtype="<u4",
    )

def _validate_aortic_arch_descending_field002_interface_certificate(
    certificate, surfaces, network_keys,
):
    """Validate the fresh 7↔8 interface proof bound to the field-002 arch."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE,
        schema=AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_SCHEMA,
        status=AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_STATUS,
        evidence_report_sha256=(
            AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_REPORT_SHA256),
        source_preflight_sha256=(
            AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PREFLIGHT_SHA256),
        capture_input_bundle_sha256=(
            AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_CAPTURE_BUNDLE_SHA256),
        exact_pair_count=AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=(
            AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_PAIRSET_SHA256),
        source_rows=AORTIC_ARCH_DESCENDING_SOURCE_ROWS,
        pair_dtype="<u4",
    )


def _validate_aortic_arch_carotid_interface_certificate(certificate, surfaces, network_keys):
    """Validate the frozen, non-flow arch/left-carotid source interface."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_CAROTID_SOURCE_INTERFACE,
        schema="numi.human.verified-aortic-source-interface-handoff.v1",
        status="VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE",
        evidence_report_sha256=AORTIC_ARCH_CAROTID_INTERFACE_REPORT_SHA256,
        source_preflight_sha256=AORTIC_ARCH_CAROTID_INTERFACE_PREFLIGHT_SHA256,
        capture_input_bundle_sha256=(
            AORTIC_ARCH_CAROTID_INTERFACE_CAPTURE_BUNDLE_SHA256),
        exact_pair_count=AORTIC_ARCH_CAROTID_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=AORTIC_ARCH_CAROTID_INTERFACE_PAIRSET_SHA256,
        source_rows=AORTIC_ARCH_CAROTID_SOURCE_ROWS,
        pair_dtype="<u4",
    )


def audit_network_rows_against_all_targets(
    verifier, surfaces, network_edges, source_face_pairs, *,
    verified_network_continuity=None, verified_network_source_inventory=None,
    verified_nonnetwork_interface_certificates=None, progress_callback=None,
):
    """Run the shared all-target collision screen for a frozen vessel graph.

    Exact source collar patches are passed only for declared graph edges. The
    central verifier still rejects every other vessel, bone, muscle, skin
    boundary, airway, or organ contact. A passing screen is not a continuity,
    calibre, or anatomy-admission result.
    """
    _require(hasattr(verifier, "audit_changed_surfaces"),
             "the current shared resting-anatomy verifier is required")
    _require(isinstance(surfaces, Mapping) and bool(surfaces),
             "all-target screen requires the complete captured surface mapping")
    _require(isinstance(network_edges, (tuple, list)) and bool(network_edges),
             "network edge list is empty")
    normalized_edges = []
    network_keys = set()
    for edge in network_edges:
        _require(isinstance(edge, (tuple, list)) and len(edge) == 2,
                 "network edges must contain two typed surface identities")
        first, second = _identity(edge[0]), _identity(edge[1])
        _require(first != second and (first, second) not in normalized_edges
                 and (second, first) not in normalized_edges,
                 "network edge list contains a self-link or duplicate")
        normalized_edges.append((first, second))
        network_keys.update((first, second))
    _require(network_keys.issubset(surfaces),
             "captured surface mapping is missing a declared network vessel")
    _require(isinstance(source_face_pairs, Mapping)
             and set(source_face_pairs) == set(normalized_edges),
             "exact source collar ancestry must cover precisely the declared network edges")

    nonnetwork_interface_handoffs = {}
    if verified_nonnetwork_interface_certificates is not None:
        supported_interfaces = {
            AORTIC_ARCH_SOURCE_INTERFACE:
                _validate_aortic_arch_interface_certificate,
            AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE:
                _validate_aortic_arch_descending_interface_certificate,
            AORTIC_ARCH_CAROTID_SOURCE_INTERFACE:
                _validate_aortic_arch_carotid_interface_certificate,
        }
        _require(isinstance(verified_nonnetwork_interface_certificates, Mapping)
                 and bool(verified_nonnetwork_interface_certificates)
                 and set(verified_nonnetwork_interface_certificates)
                 .issubset(supported_interfaces),
                 "only the exact source-bound aortic interface certificates are supported")
        for edge, certificate in verified_nonnetwork_interface_certificates.items():
            validator = supported_interfaces[edge]
            if isinstance(certificate, Mapping):
                if (edge == AORTIC_ARCH_SOURCE_INTERFACE
                        and certificate.get("schema")
                        == AORTIC_ARCH_FIELD002_INTERFACE_SCHEMA):
                    validator = _validate_aortic_arch_field002_interface_certificate
                elif (edge == AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE
                      and certificate.get("schema")
                      == AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_SCHEMA):
                    validator = (
                        _validate_aortic_arch_descending_field002_interface_certificate)
            nonnetwork_interface_handoffs[edge] = validator(
                certificate, surfaces, network_keys)

    interfaces = {}
    interface_records = []
    normalized_source_pairs = {}
    for first, second in normalized_edges:
        first_value, second_value = surfaces[first], surfaces[second]
        prepared_type = getattr(verifier, "PreparedSurface", ())
        if isinstance(first_value, prepared_type):
            first_positions, first_raw_faces = first_value.positions, first_value.faces
        else:
            first_positions, first_raw_faces = first_value[0], first_value[1]
        if isinstance(second_value, prepared_type):
            second_positions, second_raw_faces = second_value.positions, second_value.faces
        else:
            second_positions, second_raw_faces = second_value[0], second_value[1]
        first_faces = _faces(first_raw_faces, name=f"network row {first[1]}",
                             vertex_count=len(first_positions))
        second_faces = _faces(second_raw_faces, name=f"network row {second[1]}",
                              vertex_count=len(second_positions))
        raw_pairs = source_face_pairs[(first, second)]
        pairs = _pair_rows(raw_pairs, name=f"source collar {first[1]}->{second[1]}",
                           bridge_face_count=len(first_faces),
                           target_face_count=len(second_faces))
        normalized_source_pairs[(first, second)] = pairs
        first_patch = sorted({pair[0] for pair in pairs})
        second_patch = sorted({pair[1] for pair in pairs})
        allowed = tuple((a, b) for a in first_patch for b in second_patch)
        interfaces[(first, second)] = allowed
        interfaces[(second, first)] = tuple((b, a) for a, b in allowed)
        interface_records.append({
            "parent": list(first), "child": list(second),
            "source_pair_count": len(pairs),
            "source_pairs_sha256": hashlib.sha256(
                np.ascontiguousarray(pairs, dtype="<i8").tobytes()).hexdigest(),
            "parent_patch_face_count": len(first_patch),
            "child_patch_face_count": len(second_patch),
            "allowed_patch_cross_product_pair_count": len(allowed),
        })


    if verified_network_continuity is not None:
        handoffs = _verified_network_link_handoff(
            verifier, surfaces, tuple(normalized_edges), normalized_source_pairs,
            verified_network_continuity,
            source_inventory=verified_network_source_inventory)
        verified_edges = {frozenset(edge) for edge in normalized_edges}
        changed = tuple(sorted(network_keys))
        nonnetwork_targets = tuple(sorted(set(surfaces) - network_keys))
        certified_interface_target_keys = {
            edge[1] for edge in nonnetwork_interface_handoffs}
        all_target_screen_targets = tuple(
            key for key in nonnetwork_targets
            if key not in certified_interface_target_keys)
        skin_targets = tuple(key for key in all_target_screen_targets if key[0] == 51007)
        enclosures = [(vessel, skin) for vessel in changed for skin in skin_targets]
        if progress_callback is not None:
            progress_callback("STARTING_NONNETWORK_TARGET_SCREEN")
        shared_result = verifier.audit_changed_surfaces(
            surfaces, changed=changed, targets=all_target_screen_targets,
            interfaces={}, required_closed=changed,
            intended_enclosures=enclosures)
        if progress_callback is not None:
            progress_callback(
                "FINISHED_NONNETWORK_TARGET_SCREEN_PASSED"
                if shared_result.get("passed") is True
                else "FINISHED_NONNETWORK_TARGET_SCREEN_REJECTED")
        if shared_result.get("passed") is not True:
            return {
                "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
                "status": "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_REJECTED",
                "passed": False,
                "network_edge_count": len(normalized_edges),
                "network_surface_count": len(network_keys),
                "all_capture_target_count": len(surfaces),
                "nonnetwork_target_count": len(nonnetwork_targets),
                "verified_junction_handoff": {
                    "status": "VERIFIED_PER_LINK_CONTINUITY_BOUND",
                    "links": handoffs,
                    "only_declared_neighbor_pairs_omitted_from_duplicate_pair_audit": True,
                },
                "verified_nonnetwork_interface_handoffs": list(
                    nonnetwork_interface_handoffs.values()),
                "certified_interface_target_checks": [],
                "shared_verifier_result": shared_result,
                "nonedge_network_pair_checks": [],
                "failed_stage": "network_surface_against_nonnetwork_target",
                "anatomical_admission": False,
            }

        prepared_cache = {}
        prepared_type = getattr(verifier, "PreparedSurface", ())

        def prepared_for_pair(key):
            value = surfaces[key]
            if isinstance(value, prepared_type):
                return value
            if key not in prepared_cache:
                prepared_cache[key] = verifier.prepare_surface(
                    key, value[0], value[1],
                    normals=value[2] if len(value) > 2 else None,
                    source_vertex_ids=value[3] if len(value) > 3 else
                    np.arange(len(value[0]), dtype=np.uint64))
            return prepared_cache[key]

        certified_target_checks = []
        for interface_edge in nonnetwork_interface_handoffs:
            interface_network_key, interface_target_key = interface_edge
            for network_key in changed:
                if network_key == interface_network_key:
                    continue
                if progress_callback is not None:
                    progress_callback(
                        f"CHECKING_CERTIFIED_INTERFACE_TARGET_PAIR_"
                        f"{network_key[0]}_{network_key[1]}__"
                        f"{interface_target_key[0]}_{interface_target_key[1]}")
                pair_result = verifier.audit_surface_pair(
                    prepared_for_pair(network_key),
                    prepared_for_pair(interface_target_key))
                pair_row = {
                    "identities": [list(network_key), list(interface_target_key)],
                    "passed": pair_result.get("passed") is True,
                    "audit": pair_result,
                    "strict_unrelated_network_to_interface_target_pair": True,
                }
                certified_target_checks.append(pair_row)
                if progress_callback is not None:
                    progress_callback(
                        f"FINISHED_CERTIFIED_INTERFACE_TARGET_PAIR_"
                        f"{network_key[0]}_{network_key[1]}__"
                        f"{interface_target_key[0]}_{interface_target_key[1]}_"
                        f"{'PASSED' if pair_row['passed'] else 'REJECTED'}")
                if not pair_row["passed"]:
                    return {
                        "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
                        "status": "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_REJECTED",
                        "passed": False,
                        "network_edge_count": len(normalized_edges),
                        "network_surface_count": len(network_keys),
                        "all_capture_target_count": len(surfaces),
                        "nonnetwork_target_count": len(nonnetwork_targets),
                        "verified_junction_handoff": {
                            "status": "VERIFIED_PER_LINK_CONTINUITY_BOUND",
                            "links": handoffs,
                            "only_declared_neighbor_pairs_omitted_from_duplicate_pair_audit": True,
                        },
                        "verified_nonnetwork_interface_handoffs": list(
                            nonnetwork_interface_handoffs.values()),
                        "certified_interface_target_checks": certified_target_checks,
                        "shared_verifier_result": shared_result,
                        "nonedge_network_pair_checks": [],
                        "failed_stage": "certified_interface_target_against_other_network_surface",
                        "failed_pair": pair_row["identities"],
                        "anatomical_admission": False,
                    }

        nonedge_rows = []
        for left_index, first in enumerate(changed):
            for second in changed[left_index + 1:]:
                if frozenset((first, second)) in verified_edges:
                    continue
                if progress_callback is not None:
                    progress_callback(
                        f"CHECKING_UNDECLARED_NETWORK_PAIR_{first[0]}_{first[1]}__"
                        f"{second[0]}_{second[1]}")
                pair_result = verifier.audit_surface_pair(
                    prepared_for_pair(first), prepared_for_pair(second))
                pair_row = {
                    "identities": [list(first), list(second)],
                    "passed": pair_result.get("passed") is True,
                    "audit": pair_result,
                    "declared_network_edge": False,
                }
                nonedge_rows.append(pair_row)
                if progress_callback is not None:
                    progress_callback(
                        f"FINISHED_UNDECLARED_NETWORK_PAIR_{first[0]}_{first[1]}__"
                        f"{second[0]}_{second[1]}_"
                        f"{'PASSED' if pair_row['passed'] else 'REJECTED'}")
                if not pair_row["passed"]:
                    return {
                        "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
                        "status": "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_REJECTED",
                        "passed": False,
                        "network_edge_count": len(normalized_edges),
                        "network_surface_count": len(network_keys),
                        "all_capture_target_count": len(surfaces),
                        "nonnetwork_target_count": len(nonnetwork_targets),
                        "verified_junction_handoff": {
                            "status": "VERIFIED_PER_LINK_CONTINUITY_BOUND",
                            "links": handoffs,
                            "only_declared_neighbor_pairs_omitted_from_duplicate_pair_audit": True,
                        },
                        "verified_nonnetwork_interface_handoffs": list(
                            nonnetwork_interface_handoffs.values()),
                        "certified_interface_target_checks": certified_target_checks,
                        "shared_verifier_result": shared_result,
                        "nonedge_network_pair_checks": nonedge_rows,
                        "failed_stage": "undeclared_network_pair",
                        "anatomical_admission": False,
                    }

        return {
            "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
            "status": "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_PASSED_UNADMITTED",
            "passed": True,
            "network_edge_count": len(normalized_edges),
            "network_surface_count": len(network_keys),
            "all_capture_target_count": len(surfaces),
            "all_capture_target_identity_sha256": hashlib.sha256(
                b"".join(np.asarray(key, dtype="<u4").tobytes()
                         for key in sorted(surfaces))
            ).hexdigest(),
            "nonnetwork_target_count": len(nonnetwork_targets),
            "verified_nonnetwork_interface_handoffs": list(
                nonnetwork_interface_handoffs.values()),
            "certified_interface_target_checks": certified_target_checks,
            "certified_interface_target_count_omitted_from_generic_screen": len(
                certified_interface_target_keys),
            "verified_junction_handoff": {
                "status": "VERIFIED_PER_LINK_CONTINUITY_BOUND",
                "links": handoffs,
                "only_declared_neighbor_pairs_omitted_from_duplicate_pair_audit": True,
                "omission_reason": (
                    "each direct neighbor pair is independently bound to current Float32 "
                    "positions/faces/GIDs, exact collar ancestry, positive solid overlap, "
                    "exact inside-both witness, and source-baseline calibre"),
            },
            "shared_verifier_result": shared_result,
            "nonedge_network_pair_checks": nonedge_rows,
            "declared_skin_enclosures": len(enclosures),
            "scope_limit": (
                "All network surfaces were checked against every nonnetwork captured surface "
                "except the one row-6 target paired with row 7, which was independently bound "
                "to its exact current capture arrays and source-baseline closed-solid, exact "
                "inside-both, connected-union, and own-station calibre proof. Row 6 was then "
                "checked strictly against every other network surface. Every non-edge network "
                "pair was checked with the shared exact pair oracle; only direct neighbors with "
                "complete per-link 3D continuity proofs were omitted from duplicate checks. "
                "This is collision and source-calibre evidence, not all-pose/native/anatomical "
                "admission."),
            "anatomical_admission": False,
        }

    targets = tuple(sorted(surfaces))
    changed = tuple(sorted(network_keys))
    skins = tuple(key for key in targets if key[0] == 51007)
    enclosures = [(vessel, skin) for vessel in changed for skin in skins]
    result = verifier.audit_changed_surfaces(
        surfaces,
        changed=changed,
        targets=targets,
        interfaces=interfaces,
        required_closed=changed,
        intended_enclosures=enclosures,
    )
    passed = result.get("passed") is True
    return {
        "schema": "numi.human.brachiocephalic.current-network-all-target-screen.v1",
        "status": ("CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_PASSED_UNADMITTED"
                   if passed else "CURRENT_NETWORK_ALL_TARGET_COLLISION_SCREEN_REJECTED"),
        "passed": passed,
        "network_edge_count": len(normalized_edges),
        "network_surface_count": len(network_keys),
        "all_capture_target_count": len(targets),
        "all_capture_target_identity_sha256": hashlib.sha256(
            b"".join(np.asarray(key, dtype="<u4").tobytes() for key in targets)
        ).hexdigest(),
        "frozen_source_interface_patches": interface_records,
        "declared_skin_enclosures": len(enclosures),
        "shared_verifier_result": result,
        "scope_limit": (
            "Collision/self/closure screen only. Frozen face patches do not prove lumen "
            "continuity; independent 3D inside-both, connected-solid, and source-station "
            "calibre gates remain required."),
        "anatomical_admission": False,
    }


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
    prepared_type = getattr(verifier, "PreparedSurface", ())
    if isinstance(value, prepared_type):
        _require(value.key == key,
                 f"prepared surface identity differs from requested surface {key}")
        prepared = value
        _require(prepared.closed and prepared.self_audit()["count"] == 0,
                 f"surface {key} must be closed and exact-self-clear")
        return prepared
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
    requested_offset = float(spec["station_offset_from_anchor_m"])
    station_scope = spec.get("station_scope")
    if station_scope == "TARGET_COLLAR_LOCAL_SWEEP_V1":
        _require(np.isfinite(requested_offset)
                 and requested_offset >= 0.0
                 and requested_offset <= patch_extent + margin,
                 "target-local station must remain inside its frozen collar extent")
        offset = requested_offset
    else:
        _require(station_scope is None,
                 "unknown material station scope")
        offset = max(requested_offset, patch_extent + margin)
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



TARGET_COLLAR_STATION_FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def _target_collar_material_station_candidates(
    surface, witness_points, collar_face_ids, *, fractions=TARGET_COLLAR_STATION_FRACTIONS,
):
    """Derive <=4 target-local planes from frozen collar source ancestry.

    The tangent points from matched collar vertices toward non-collar target
    source vertices. Offsets are fixed fractions of the positive target-collar
    projection extent. These are bounded source geometry probes, not fitted
    anatomical centerlines or clinical station locations.
    """
    _require(isinstance(fractions, (tuple, list)) and 0 < len(fractions) <= 4,
             "target-collar selector accepts one to four frozen fractions")
    normalized = tuple(float(value) for value in fractions)
    _require(all(np.isfinite(value) and 0.0 < value < 1.0 for value in normalized)
             and bool(np.all(np.diff(np.asarray(normalized, dtype=np.float64)) > 0.0)),
             "target-collar fractions must be finite, unique, increasing, and inside (0, 1)")
    positions = surface.positions.astype(np.float64)
    faces = surface.faces
    source_ids = np.asarray(surface.source_vertex_ids)
    collar_ids = np.asarray(collar_face_ids, dtype=np.int64)
    _require(collar_ids.ndim == 1 and len(collar_ids) > 0
             and int(collar_ids.min()) >= 0 and int(collar_ids.max()) < len(faces),
             "target-collar station needs valid frozen target face ancestry")
    witness = np.asarray(witness_points, dtype=np.float64)
    _require(witness.ndim == 2 and witness.shape[1] == 3 and len(witness) > 0
             and np.isfinite(witness).all(),
             "target-collar station requires exact interface witness points")
    collar_vertices = np.unique(faces[collar_ids].reshape(-1))
    noncollar_vertices = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), collar_vertices, assume_unique=True)
    _require(len(noncollar_vertices) >= 2,
             "target-collar station needs at least two non-collar source vertices")
    collar_center = positions[collar_vertices].mean(axis=0)
    interior_center = positions[noncollar_vertices].mean(axis=0)
    direction = interior_center - collar_center
    direction_norm = float(np.linalg.norm(direction))
    _require(np.isfinite(direction_norm) and direction_norm > 1e-15,
             "matched target collar has no stable source-interior direction")
    direction /= direction_norm
    ancestry = _material_vertex_ancestry(surface, direction, collar_ids)
    tangent_support = np.asarray(
        ancestry["tangent_support_local_vertex_ids"], dtype=np.int64)
    tangent = positions[int(tangent_support[1])] - positions[int(tangent_support[0])]
    tangent_norm = float(np.linalg.norm(tangent))
    _require(np.isfinite(tangent_norm) and tangent_norm > 1e-15,
             "target source tangent support collapsed")
    tangent /= tangent_norm
    if float(np.dot(tangent, direction)) < 0.0:
        tangent = -tangent
        ancestry["tangent_support_local_vertex_ids"] = [
            int(tangent_support[1]), int(tangent_support[0])]
        ancestry["tangent_support_source_vertex_ids"] = [
            int(source_ids[int(tangent_support[1])]),
            int(source_ids[int(tangent_support[0])])]
    anchor_ids = np.asarray(
        ancestry["material_anchor_local_vertex_ids"], dtype=np.int64)
    weights = np.asarray(ancestry["material_anchor_weights"], dtype=np.float64)
    material_anchor = np.sum(positions[anchor_ids] * weights[:, None], axis=0)
    collar_points = positions[faces[collar_ids]].reshape(-1, 3)
    collar_projection = (collar_points - material_anchor) @ tangent
    positive_extent = float(collar_projection.max())
    lower_extent = float(collar_projection.min())
    margin = max(float(np.max(np.abs(np.spacing(surface.positions))) * 8.0),
                 float(np.finfo(np.float32).tiny))
    _require(np.isfinite(positive_extent) and positive_extent > margin,
             "matched target collar has no positive source-tangent extent")
    junction = witness.mean(axis=0)
    result = []
    for index, fraction in enumerate(normalized):
        offset = positive_extent * fraction
        spec = {
            **ancestry,
            "tangent_world_unit": tangent.tolist(),
            "provenance": (
                "inferred target-source tangent from matched collar centroid "
                "toward non-collar target-source centroid"),
            "inferred": True,
            "station_offset_from_anchor_m": offset,
            "station_scope": "TARGET_COLLAR_LOCAL_SWEEP_V1",
            "station_candidate_index": index,
            "station_candidate_fraction_of_target_collar_extent": fraction,
            "target_collar_projection_bounds_from_material_anchor_m": [
                lower_extent, positive_extent],
            "target_source_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.positions).tobytes()).hexdigest(),
            "target_source_faces_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.faces).tobytes()).hexdigest(),
            "target_source_vertex_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(source_ids).tobytes()).hexdigest(),
            "exact_interface_witness_points_sha256": hashlib.sha256(
                np.ascontiguousarray(witness, dtype="<f8").tobytes()).hexdigest(),
            "frozen_target_collar_face_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(collar_ids, dtype="<i8").tobytes()).hexdigest(),
            "target_collar_source_face_ids": [int(value) for value in collar_ids],
            "target_collar_source_face_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(collar_ids, dtype="<i8").tobytes()).hexdigest(),
            "target_collar_vertex_ids": [int(value) for value in collar_vertices],
            "target_collar_vertex_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(collar_vertices, dtype="<i8").tobytes()).hexdigest(),
            "target_noncollar_vertex_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(noncollar_vertices, dtype="<i8").tobytes()).hexdigest(),
            "target_collar_centroid_world_m": collar_center.tolist(),
            "target_noncollar_centroid_world_m": interior_center.tolist(),
            "target_source_direction_unit": direction.tolist(),
            "target_collar_positive_extent_m": positive_extent,
            "target_collar_fraction_offset_m": offset,
            "junction_witness_centroid_world_m": junction.tolist(),
            "junction_witness_signed_distance_to_plane_m": float(
                np.dot(junction - material_anchor, tangent) - offset),
            "junction_witness_perpendicular_distance_to_plane_m": float(
                np.linalg.norm((junction - material_anchor)
                               - tangent * np.dot(junction - material_anchor, tangent))),
            "station_plane_normal_is_target_source_tangent": True,
            "station_plane_inferred_from_bridge_tangent": False,
            "mesh_coordinates_modified": False,
        }
        result.append(_transport_material_station(surface, spec))
    return result


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
        station_scope = value.get("station_scope")
        _require(np.isfinite(offset)
                 and (offset > 0.0 if station_scope is None else offset >= 0.0),
                 "explicit material station offset is invalid")
        _require(station_scope in (None, "TARGET_COLLAR_LOCAL_SWEEP_V1"),
                 "explicit material station scope is not supported")
        spec = {
            "tangent_world_unit": tangent.tolist(),
            "provenance": value["provenance"],
            "inferred": bool(value.get("inferred", False)),
            "station_offset_from_anchor_m": offset,
            **({"station_scope": station_scope} if station_scope is not None else {}),
            **{key: value[key] for key in (
                "station_candidate_index",
                "station_candidate_fraction_of_target_collar_extent",
                "target_collar_projection_bounds_from_material_anchor_m",
                "target_source_positions_sha256", "target_source_faces_sha256",
                "target_source_vertex_ids_sha256",
                "exact_interface_witness_points_sha256",
                "frozen_target_collar_face_ids_sha256",
                "target_collar_source_face_ids",
                "target_collar_source_face_ids_sha256",
                "target_collar_vertex_ids", "target_collar_vertex_ids_sha256",
                "target_noncollar_vertex_ids_sha256",
                "target_collar_centroid_world_m",
                "target_noncollar_centroid_world_m",
                "target_source_direction_unit",
                "target_collar_positive_extent_m",
                "target_collar_fraction_offset_m",
                "junction_witness_centroid_world_m",
                "junction_witness_signed_distance_to_plane_m",
                "junction_witness_perpendicular_distance_to_plane_m",
                "station_plane_normal_is_target_source_tangent",
                "station_plane_inferred_from_bridge_tangent",
                "mesh_coordinates_modified",
            ) if key in value},
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


def _declared_partial_vessel_port_overlap(
    bridge_key, target_key, port_audit, observed_pairs, unexpected_pairs,
    exact_points, allowed_pairs, *, declared_vessel_ports=None,
) -> dict:
    """Classify partial overlap at an explicitly named vascular interface.

    The default explicit set is the six frozen bridge ports plus the two named
    left ulnar source branch links. A higher-level source-network owner may pass
    its separately validated canonical edge set. This only continues interface
    boundary handling; independent source section/calibre and
    positive-intersection/exact-inside/connected-union gates remain mandatory.
    """
    typed_port = (_identity(bridge_key), _identity(target_key))
    declared_ports = DECLARED_VESSEL_PORTS
    if declared_vessel_ports is not None:
        _require(isinstance(declared_vessel_ports, (set, frozenset, tuple, list))
                 and bool(declared_vessel_ports),
                 "explicit vessel port registry must be a nonempty collection")
        declared_ports = frozenset(
            (_identity(edge[0]), _identity(edge[1]))
            for edge in declared_vessel_ports
            if isinstance(edge, (tuple, list)) and len(edge) == 2)
        _require(len(declared_ports) == len(declared_vessel_ports),
                 "explicit vessel port registry contains malformed or duplicate links")
    audit = port_audit if isinstance(port_audit, Mapping) else {}
    containment_value = audit.get("containment", [])
    containment = containment_value if isinstance(containment_value, list) else []
    component_rows_complete = (
        isinstance(containment_value, list) and len(containment) == 2
        and all(
            isinstance(row, Mapping)
            and row.get("status") == "checked_exact_rational_metres"
            and bool(row.get("inside_component_witnesses"))
            and bool(row.get("outside_component_witnesses"))
            and bool(row.get("component_checks"))
            and all(
                isinstance(item, Mapping)
                and item.get("sampled_face_count", 0) > 0
                and item.get("has_inside_face_interior") is True
                and item.get("has_outside_face_interior") is True
                for item in row["component_checks"])
            for row in containment)
    )
    observed = tuple(tuple(map(int, pair)) for pair in observed_pairs)
    unexpected = tuple(tuple(map(int, pair)) for pair in unexpected_pairs)
    allowed = {tuple(map(int, pair)) for pair in allowed_pairs}
    conditions = {
        "declared_typed_vessel_port": typed_port in declared_ports,
        "surface_pair_audit_identity_matches": audit.get("identities") == [
            list(typed_port[0]), list(typed_port[1])],
        "central_p6_rejected_only_partial_overlap": audit.get("passed") is False,
        "observed_exact_interface_pairs_nonempty": bool(observed),
        "all_observed_pairs_inside_frozen_collar_patch": (
            bool(observed) and all(pair in allowed for pair in observed)),
        "no_unexpected_pairs_outside_frozen_patch": not unexpected,
        "exact_interface_points_nonempty": bool(exact_points),
        "no_rejected_declared_feature_contacts": not audit.get(
            "rejected_interface_feature_contacts", []),
        "no_other_declared_feature_contacts": not audit.get(
            "declared_interface_feature_contacts", []),
        "symmetric_exact_inside_and_outside_face_witnesses": component_rows_complete,
        "no_intended_enclosure_exception": not any(
            row.get("declared_intended_enclosure") for row in containment
            if isinstance(row, Mapping)),
    }
    return {
        "passed": all(conditions.values()),
        "classification": (
            "DECLARED_VESSEL_PORT_PARTIAL_OVERLAP_BOUNDARY_ONLY"
            if all(conditions.values()) else "NOT_A_DECLARED_PARTIAL_VESSEL_PORT"),
        "typed_port": [list(typed_port[0]), list(typed_port[1])],
        "conditions": conditions,
        "observed_pair_count": len(observed),
        "unexpected_pair_count": len(unexpected),
        "exact_interface_point_count": len(exact_points),
        "containment_statuses": [
            row.get("status") for row in containment if isinstance(row, Mapping)],
        "requires_separate_solid_and_calibre_proofs": True,
        "clinical_or_anatomical_admission": False,
    }


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
    target_station_candidate_fractions=None,
):
    """Check one declared bridge port as closed solids and local section loops.

    This is a numerical geometry gate only. It does not establish measured
    physiology, native acceptance, or clinical calibre.
    """
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    if target_station_candidate_fractions is not None:
        _require((bridge_key, target_key) == ((51011, 526), (51011, 472)),
                 "target-collar station search is restricted to bridge 526 / target 472")
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
    partial_overlap_boundary = _declared_partial_vessel_port_overlap(
        bridge_key, target_key, port_audit, observed,
        port_audit.get("unexpected_triangle_pairs", ()),
        exact_points, allowed)
    port_pair_passed = bool(exact_points) and (
        bool(port_audit.get("passed")) or partial_overlap_boundary["passed"])

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

    if target_station_candidate_fractions is not None:
        fractions = tuple(float(value) for value in target_station_candidate_fractions)
        candidates = _target_collar_material_station_candidates(
            target, exact_points, target_patch, fractions=fractions)
        search = {
            "status": "RUNNING_TARGET_COLLAR_LOCAL_STATION_CANDIDATES",
            "bridge_owner_id": bridge_key[1],
            "target_owner_id": target_key[1],
            "candidate_count_requested": len(candidates),
            "candidate_count_evaluated": 0,
            "candidate_fraction_order": list(fractions),
            "selector_method": (
                "target source tangent from matched target-collar centroid toward "
                "non-collar target-source centroid; candidate planes lie at fixed "
                "fractions of the frozen target-collar projection extent"),
            "plane_normal_source_owner_id": target_key[1],
            "plane_normal_inferred_from_bridge_tangent": False,
            "mesh_coordinates_modified": False,
            "candidates": [],
            "selected_candidate_index": None,
            "passed": False,
        }
        selected = None
        for candidate_spec in candidates:
            candidate_index = int(candidate_spec["station_candidate_index"])
            baseline_spec = _transport_material_station(base_target, candidate_spec)
            candidate_branch_sections = {
                bridge_key[1]: _source_surface_section(bridge, candidate_spec),
                target_key[1]: _source_surface_section(target, candidate_spec),
            }
            baseline_branch_sections = {
                bridge_key[1]: _source_surface_section(base_bridge, baseline_spec),
                target_key[1]: _source_surface_section(base_target, baseline_spec),
            }
            candidate_union = _section_loops(
                union_positions, union_faces,
                candidate_spec["section_center_world_m"],
                candidate_spec["tangent_world_unit"], derived_csg=True,
                face_owner_ids=union_lineage["face_owner_ids"],
                face_source_ids=union_lineage["source_face_ids"],
                face_generated_mask=union_lineage["generated_face_mask"])
            baseline_union = _section_loops(
                base_union_positions, base_union_faces,
                baseline_spec["section_center_world_m"],
                baseline_spec["tangent_world_unit"], derived_csg=True,
                face_owner_ids=base_union_lineage["face_owner_ids"],
                face_source_ids=base_union_lineage["source_face_ids"],
                face_generated_mask=base_union_lineage["generated_face_mask"])
            declaration = {
                "declaration_id": (
                    "target-collar-source-station-v1:"
                    + str(candidate_spec["target_source_positions_sha256"])
                    + ":" + str(candidate_index)),
                "station_role": "target",
                "expected_owner_ids": sorted((bridge_key[1], target_key[1])),
                "bridge_owner_id": bridge_key[1],
                "target_owner_id": target_key[1],
                "anchor_owner_id": target_key[1],
                "source_ancestry": {
                    key: candidate_spec[key] for key in (
                        "material_anchor_source_vertex_ids",
                        "tangent_support_source_vertex_ids",
                        "frozen_collar_face_ids",
                    )
                },
                "candidate_station": candidate_spec,
                "baseline_station": baseline_spec,
            }
            proof = _junction_aware_two_loop_section_proof(
                expected_owner_ids=(bridge_key[1], target_key[1]),
                candidate_union_section=candidate_union,
                baseline_union_section=baseline_union,
                candidate_branch_sections=candidate_branch_sections,
                baseline_branch_sections=baseline_branch_sections,
                solid_evidence=junction_solid_evidence(),
                station_declaration=declaration,
            )
            try:
                bridge_section_comparison = _section_comparison(
                    candidate_branch_sections[bridge_key[1]],
                    baseline_branch_sections[bridge_key[1]])
                target_section_comparison = _section_comparison(
                    candidate_branch_sections[target_key[1]],
                    baseline_branch_sections[target_key[1]])
                branch_comparisons = {
                    str(bridge_key[1]): bridge_section_comparison,
                    str(target_key[1]): target_section_comparison,
                }
                branch_calibre_passed = (
                    bridge_section_comparison.get("passed") is True
                    and target_section_comparison.get("passed") is True)
            except ValueError as error:
                branch_comparisons = {
                    "error": {"type": type(error).__name__, "message": str(error)}}
                branch_calibre_passed = False
            tangent_dot = float(np.dot(
                np.asarray(candidate_spec["tangent_world_unit"], dtype=np.float64),
                np.asarray(bridge_station["tangent_world_unit"], dtype=np.float64)))
            candidate_result = {
                "candidate_index": candidate_index,
                "fraction_of_frozen_target_collar_extent": (
                    candidate_spec["station_candidate_fraction_of_target_collar_extent"]),
                "station": candidate_spec,
                "baseline_station_transported_by_target_vertex_ancestry": baseline_spec,
                "target_local_tangent_vs_previous_bridge_plane_normal": {
                    "signed_dot": tangent_dot,
                    "absolute_normal_dot": abs(tangent_dot),
                    "unsigned_normal_angle_degrees": float(np.degrees(
                        np.arccos(np.clip(abs(tangent_dot), -1.0, 1.0)))),
                },
                "candidate_source_sections": {
                    str(owner): {
                        "passed": section.get("passed"),
                        "loop_count": section.get("loop_count"),
                        "area_mm2": section.get("area_mm2"),
                        "minimum_feret_width_mm": section.get(
                            "minimum_feret_width_mm"),
                        "owner_ids": section.get("owner_ids"),
                        "lineage_complete": section.get("lineage_complete"),
                        "source_face_ancestor_ids_sha256": section.get(
                            "source_face_ancestor_ids_sha256"),
                        "reason": section.get("reason"),
                    } for owner, section in candidate_branch_sections.items()
                },
                "baseline_source_sections": {
                    str(owner): {
                        "passed": section.get("passed"),
                        "loop_count": section.get("loop_count"),
                        "area_mm2": section.get("area_mm2"),
                        "minimum_feret_width_mm": section.get(
                            "minimum_feret_width_mm"),
                        "owner_ids": section.get("owner_ids"),
                        "lineage_complete": section.get("lineage_complete"),
                        "source_face_ancestor_ids_sha256": section.get(
                            "source_face_ancestor_ids_sha256"),
                        "reason": section.get("reason"),
                    } for owner, section in baseline_branch_sections.items()
                },
                "candidate_union_section": {
                    "passed": candidate_union.get("passed"),
                    "loop_count": candidate_union.get("loop_count"),
                    "loop_metrics": candidate_union.get("loop_metrics"),
                    "reason": candidate_union.get("reason"),
                },
                "baseline_union_section": {
                    "passed": baseline_union.get("passed"),
                    "loop_count": baseline_union.get("loop_count"),
                    "loop_metrics": baseline_union.get("loop_metrics"),
                    "reason": baseline_union.get("reason"),
                },
                "branch_source_calibre_comparisons": branch_comparisons,
                "branch_source_calibre_passed": branch_calibre_passed,
                "lineaged_union_proof": proof,
                "passed": bool(branch_calibre_passed and proof.get("passed") is True),
            }
            search["candidates"].append(candidate_result)
            search["candidate_count_evaluated"] += 1
            if candidate_result["passed"]:
                selected = candidate_result
                break
        if selected is not None:
            search["status"] = "PASS_TARGET_COLLAR_LOCAL_TWO_VESSEL_STATION"
            search["selected_candidate_index"] = selected["candidate_index"]
            search["passed"] = True
        else:
            search["status"] = "REJECT_NO_TARGET_COLLAR_STATION_PASSED_STRICT_JUNCTION_PROOF"
        station_results["target_collar_station_candidate_search"] = search
        station_results["union_at_bridge_station"] = {
            "status": "NOT_USED_BY_TARGET_LOCAL_STATION_SELECTOR",
            "reason": (
                "This pair is evaluated on source-bound target 472 collar-local "
                "planes; the frozen 003 bridge-tangent plane remains preserved as "
                "prior evidence and is not reused as the selected target station."),
        }
        station_results["union_at_target_station"] = (
            {} if selected is None else {
                "status": "SELECTED_TARGET_COLLAR_JUNCTION_STATION",
                "station": selected["station"],
                "baseline_station_transported_by_target_vertex_ancestry":
                    selected["baseline_station_transported_by_target_vertex_ancestry"],
                "candidate": selected["candidate_union_section"],
                "baseline": selected["baseline_union_section"],
                "candidate_vs_baseline": selected["lineaged_union_proof"],
            })
        if selected is None:
            error = ValueError(
                "no frozen target-collar station candidate passed the strict "
                "one-section-per-vessel, lineaged-union, positive-solid, and "
                "per-branch calibre gates")
            error.target_station_candidate_search = search
            raise error
    if target_station_candidate_fractions is None:
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
            "central_p6_audit_passed": bool(port_audit.get("passed")),
            "declared_partial_overlap_boundary_only": partial_overlap_boundary,
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
    junction_station_declarations=None, station_candidate_fractions=None,
):
    """Run source-ordered bridge port checks and stop at the first failure."""
    station_specs = {} if station_specs is None else station_specs
    route_anchors = {} if route_anchors is None else route_anchors
    junction_station_declarations = (
        {} if junction_station_declarations is None else junction_station_declarations)
    station_candidate_fractions = (
        {} if station_candidate_fractions is None else station_candidate_fractions)
    _require(isinstance(station_candidate_fractions, Mapping),
             "target station candidate fractions must be source-bound by port")
    _require(set(station_candidate_fractions) <= {((51011, 526), (51011, 472))},
             "target-collar station search is restricted to the 526-to-472 port")
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
                target_station_candidate_fractions=station_candidate_fractions.get(pair),
            )
        except ValueError as error:
            report = {
                "schema": "numi.human.brachiocephalic-port-solid-continuity.v1",
                "bridge": list(pair[0]), "target": list(pair[1]),
                "passed": False, "failure": str(error),
                "source_geometry_mutated": False,
            }
            candidate_search = getattr(error, "target_station_candidate_search", None)
            if candidate_search is not None:
                report["target_station_candidate_search"] = candidate_search
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
