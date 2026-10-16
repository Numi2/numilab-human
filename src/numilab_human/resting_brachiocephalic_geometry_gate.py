"""Captured-pose surface-intersection stage for appended brachiocephalic veins.

This wrapper uses the existing resting-anatomy exact packed-Float32 verifier.
It checks each predicted bridge against captured anatomy through the frozen
P6 exact predicates, then exposes a separate source-bound solid-continuity and
local-calibre screen. It is not clinical or physiological qualification.
"""
from __future__ import annotations

import hashlib
import json
import sys
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
# Current 6<->7 source-bound certificate for the rigid-collar/section-support
# candidate on accepted capture 024. This profile is tied to one immutable
# proof/preflight and exact live Float32 fingerprints; it does not broaden the
# generic pair policy.
AORTIC_ARCH_REBASE026_INTERFACE_SCHEMA = (
    "numi.human.verified-aortic-arch-rebase-026-interface-handoff.v1")
AORTIC_ARCH_REBASE026_INTERFACE_STATUS = (
    "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_REBASE_026_INTERFACE")
AORTIC_ARCH_REBASE026_INTERFACE_REPORT_SHA256 = (
    "cd4639bfb0da634acfd06178ac6b58a966212e0ff9f8f25d4ffe29741f81ddfa")
AORTIC_ARCH_REBASE026_INTERFACE_PREFLIGHT_SHA256 = (
    "f01059adaaa6662ba13bd2f0ae135893bac3c14a6a9d5a720975c7559abed2ae")
AORTIC_ARCH_REBASE026_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "81102e440325e89cfd4fbf74ee3fe9a45bdbeaadee1375757a14ebf6b8ee5501")
AORTIC_ARCH_REBASE026_INTERFACE_PACK_SHA256 = (
    "4fd1b291d4ae54096d298cae2a1b56cdb1e0c158ad5ce215d2865fefb9110d61")
AORTIC_ARCH_REBASE026_INTERFACE_RECEIPT_SHA256 = (
    "f53f5f8dcafa7a809fb18570e051da330651e72bd6dc66e68342e429acb77219")
AORTIC_ARCH_REBASE026_INTERFACE_PAIRSET_SHA256 = (
    "478c7ad156fddfe8166c1e49a63d74f5d2c625c4cde39a8d3ffe1fd9dda2d212")
AORTIC_ARCH_REBASE026_INTERFACE_PAIR_COUNT = 36
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
# Fresh 7↔465 source-bound certificate for capture 024 and the current
# field-002 arch. The preceding certificate remains available for its original
# capture only.
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_REPORT_SHA256 = (
    "025c7d12e652232bec8a4e4587f6d747ba7cdd332cd681974dc74b12c1c889ad")
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PREFLIGHT_SHA256 = (
    "64bc6e1208924354b0782fecd83995e59c32885186efee64c5ec371522bb7b93")
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_HANDOFF_SHA256 = (
    "b4dbc76123412e4114a8475c36ed3a559c520c5aed4cc0defe3be837bfaf8c94")
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_CAPTURE_BUNDLE_SHA256 = (
    "81102e440325e89cfd4fbf74ee3fe9a45bdbeaadee1375757a14ebf6b8ee5501")
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PAIRSET_SHA256 = (
    "f408e4ad80d0be34e95c96dcf50834b45bb7894f748335e78f8128c5d1aac90a")
AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PAIR_COUNT = 30
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
            section_records = {}
            for section_name in ("candidate_section", "baseline_section"):
                section = calibre.get(section_name)
                _require(isinstance(section, Mapping)
                         and section.get("passed") is True
                         and type(section.get("loop_count")) is int
                         and section.get("loop_count") >= 1
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
                selection = section.get("local_loop_selection")
                if selection is None:
                    _require(section.get("loop_count") == 1,
                             f"{role} {section_name} has unselected extra loops for {key}")
                else:
                    loops = section.get("loop_metrics")
                    distances = selection.get("all_loop_source_distance_diagnostics")
                    _require(
                        selection.get("schema")
                            == "numi.human.exact-source-face-collar-distance.v1"
                        and selection.get("passed") is True
                        and selection.get("candidate_geometry_used_for_selection") is False
                        and selection.get("source_typed_identity") == list(key)
                        and selection.get("source_face_graph_component_count") == 1
                        and valid_sha(selection.get("source_face_u32_sha256"))
                        and valid_sha(selection.get("source_face_graph_sha256"))
                        and valid_sha(selection.get("frozen_collar_face_ids_sha256"))
                        and section.get("selected_local_loop_count") == 1
                        and type(section.get("selected_local_loop_index")) is int
                        and isinstance(loops, list)
                        and len(loops) == section.get("loop_count")
                        and isinstance(distances, list)
                        and len(distances) == len(loops)
                        and all(metric.get("passed") is True
                                and metric.get("lineage_complete") is True
                                and metric.get("owner_ids") == [key[1]]
                                and metric.get("generated_face_count") == 0
                                and metric.get("unmapped_face_count") == 0
                                for metric in loops)
                        and [item.get("loop_index") for item in distances]
                            == list(range(len(loops)))
                        and all(type(item.get("minimum_source_face_graph_distance")) is int
                                and item["minimum_source_face_graph_distance"] >= 0
                                for item in distances),
                        f"{role} {section_name} has invalid source-local loop selection for {key}")
                    minima = [item["minimum_source_face_graph_distance"]
                              for item in distances]
                    winner = int(np.argmin(minima))
                    _require(minima.count(min(minima)) == 1
                             and winner == section.get("selected_local_loop_index")
                             and selection.get("selected_local_loop_index") == winner
                             and selection.get("selected_local_loop_count") == 1
                             and selection.get("selected_local_loop_source_face_ancestor_ids_sha256")
                                 == loops[winner].get("source_face_ancestor_ids_sha256")
                             and selection.get("remote_loops_retained_as_diagnostics")
                                 == [index for index in range(len(loops)) if index != winner]
                             and section.get("area_m2") == loops[winner].get("area_m2")
                             and section.get("minimum_feret_width_m")
                                 == loops[winner].get("minimum_feret_width_m")
                             and section.get("source_face_ancestor_ids")
                                 == loops[winner].get("source_face_ancestor_ids"),
                             f"{role} {section_name} did not select the unique nearest source loop for {key}")
                section_records[section_name] = section
            candidate_selection = section_records["candidate_section"].get(
                "local_loop_selection")
            baseline_selection = section_records["baseline_section"].get(
                "local_loop_selection")
            _require((candidate_selection is None) == (baseline_selection is None)
                     and (candidate_selection is None or (
                         candidate_selection.get("source_face_graph_sha256")
                             == baseline_selection.get("source_face_graph_sha256")
                         and candidate_selection.get("source_face_u32_sha256")
                             == baseline_selection.get("source_face_u32_sha256")
                         and candidate_selection.get("frozen_collar_face_ids")
                             == baseline_selection.get("frozen_collar_face_ids")
                         and candidate_selection.get("frozen_collar_face_ids_sha256")
                             == baseline_selection.get("frozen_collar_face_ids_sha256"))),
                     f"{role} candidate and baseline local selectors differ for {key}")
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
        # Preserve the proof fields so this validated handoff can be replayed
        # by the same generic validator at the final all-target boundary.
        "source_topology_passed": True,
        "interface_gate_passed": True,
        "solid_gate_passed": True,
        "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
        "exact_face_pairs": [list(pair) for pair in pairs],
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


def _validate_aortic_arch_rebase026_interface_certificate(
        certificate, surfaces, network_keys):
    """Validate the exact current 6<->7 collar proof for rebase candidate 026."""
    _require(isinstance(certificate, Mapping)
             and certificate.get("capture_pack_sha256")
             == AORTIC_ARCH_REBASE026_INTERFACE_PACK_SHA256
             and certificate.get("capture_receipt_sha256")
             == AORTIC_ARCH_REBASE026_INTERFACE_RECEIPT_SHA256,
             "current rebase-026 certificate is bound to another accepted capture")
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_SOURCE_INTERFACE,
        schema=AORTIC_ARCH_REBASE026_INTERFACE_SCHEMA,
        status=AORTIC_ARCH_REBASE026_INTERFACE_STATUS,
        evidence_report_sha256=AORTIC_ARCH_REBASE026_INTERFACE_REPORT_SHA256,
        source_preflight_sha256=AORTIC_ARCH_REBASE026_INTERFACE_PREFLIGHT_SHA256,
        capture_input_bundle_sha256=(
            AORTIC_ARCH_REBASE026_INTERFACE_CAPTURE_BUNDLE_SHA256),
        exact_pair_count=AORTIC_ARCH_REBASE026_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=AORTIC_ARCH_REBASE026_INTERFACE_PAIRSET_SHA256,
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


def _validate_aortic_arch_carotid_current_interface_certificate(
    certificate, surfaces, network_keys,
):
    """Validate the fresh source-bound 7↔465 proof for capture 024."""
    return _validate_source_interface_certificate(
        certificate, surfaces, network_keys,
        edge=AORTIC_ARCH_CAROTID_SOURCE_INTERFACE,
        schema="numi.human.verified-aortic-source-interface-handoff.v1",
        status="VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE",
        evidence_report_sha256=(
            AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_REPORT_SHA256),
        source_preflight_sha256=(
            AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PREFLIGHT_SHA256),
        capture_input_bundle_sha256=(
            AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_CAPTURE_BUNDLE_SHA256),
        exact_pair_count=AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PAIR_COUNT,
        exact_pairset_sha256=(
            AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_PAIRSET_SHA256),
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
                        == AORTIC_ARCH_REBASE026_INTERFACE_SCHEMA):
                    validator = _validate_aortic_arch_rebase026_interface_certificate
                elif (edge == AORTIC_ARCH_SOURCE_INTERFACE
                        and certificate.get("schema")
                        == AORTIC_ARCH_FIELD002_INTERFACE_SCHEMA):
                    validator = _validate_aortic_arch_field002_interface_certificate
                elif (edge == AORTIC_ARCH_DESCENDING_SOURCE_INTERFACE
                      and certificate.get("schema")
                      == AORTIC_ARCH_DESCENDING_FIELD002_INTERFACE_SCHEMA):
                    validator = (
                        _validate_aortic_arch_descending_field002_interface_certificate)
                elif (edge == AORTIC_ARCH_CAROTID_SOURCE_INTERFACE
                      and certificate.get("capture_input_bundle_sha256")
                      == AORTIC_ARCH_CAROTID_CURRENT_INTERFACE_CAPTURE_BUNDLE_SHA256):
                    validator = (
                        _validate_aortic_arch_carotid_current_interface_certificate)
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


def _material_vertex_ancestry(
    surface, tangent, collar_face_ids, *, tangent_support_vertex_ids=None,
):
    faces = surface.faces
    collar_face_ids = np.asarray(collar_face_ids, dtype=np.int64)
    collar_vertex_ids = np.unique(faces[collar_face_ids].reshape(-1))
    _require(len(collar_vertex_ids) > 0, "collar has no source vertex ancestry")
    anchor_ids = collar_vertex_ids.astype(np.int64)
    anchor_weights = np.full(len(anchor_ids), 1.0 / len(anchor_ids), dtype=np.float64)
    positions = surface.positions.astype(np.float64)
    if tangent_support_vertex_ids is None:
        tangent_support = np.arange(len(positions), dtype=np.int64)
    else:
        tangent_support = np.asarray(tangent_support_vertex_ids, dtype=np.int64)
        _require(tangent_support.ndim == 1 and len(tangent_support) >= 2
                 and len(np.unique(tangent_support)) == len(tangent_support)
                 and int(tangent_support.min()) >= 0
                 and int(tangent_support.max()) < len(positions),
                 "local tangent support must name distinct source vertices")
    projections = positions[tangent_support] @ tangent
    low_id = int(tangent_support[int(np.argmin(projections))])
    high_id = int(tangent_support[int(np.argmax(projections))])
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


def _transport_material_station_by_local_proper_rigid_fit(
    baseline_surface, candidate_surface, baseline_station, *,
    maximum_fit_residual_m=5.0e-7,
):
    """Transport a source material plane by fitting its authenticated section.

    Fit observations are the in-plane material-section nodes on the independent
    baseline: vertices lying on the station plane and edge crossings reconstructed
    from their exact source endpoints and recorded fractions. Off-plane endpoints
    only reconstruct those crossings; they are not fit observations. The indexed
    one-ring is excluded because it can cross a nonrigid taper. Candidate geometry
    estimates a local proper rigid transform, and the strict Float32 residual bound
    still rejects nonrigid change on the measured section.
    """
    _require(_identity(baseline_surface.key) == _identity(candidate_surface.key),
             "station rigid transport requires the same typed source surface")
    _require(maximum_fit_residual_m is None
             or (np.isfinite(maximum_fit_residual_m)
                 and maximum_fit_residual_m > 0.0),
             "station rigid transport residual bound must be positive and finite")
    baseline_positions = np.asarray(baseline_surface.positions)
    candidate_positions = np.asarray(candidate_surface.positions)
    _require(baseline_positions.dtype == np.dtype("<f4")
             and candidate_positions.dtype == np.dtype("<f4")
             and baseline_positions.ndim == 2 and baseline_positions.shape[1] == 3
             and candidate_positions.shape == baseline_positions.shape
             and np.isfinite(baseline_positions).all()
             and np.isfinite(candidate_positions).all(),
             "station rigid transport requires matching finite Float32 positions")
    baseline_faces = _faces(
        baseline_surface.faces, name="baseline station rigid-fit faces",
        vertex_count=len(baseline_positions))
    candidate_faces = _faces(
        candidate_surface.faces, name="candidate station rigid-fit faces",
        vertex_count=len(candidate_positions))
    _require(baseline_faces.dtype == np.dtype("<u4")
             and candidate_faces.dtype == np.dtype("<u4")
             and baseline_faces.shape == candidate_faces.shape
             and baseline_faces.tobytes() == candidate_faces.tobytes(),
             "station rigid transport requires exact ordered source-face ancestry")
    baseline_gids = np.asarray(baseline_surface.source_vertex_ids)
    candidate_gids = np.asarray(candidate_surface.source_vertex_ids)
    _require(baseline_gids.shape == (len(baseline_positions),)
             and candidate_gids.shape == baseline_gids.shape
             and baseline_gids.dtype.kind in "iu"
             and candidate_gids.dtype.kind in "iu"
             and len(np.unique(baseline_gids)) == len(baseline_gids)
             and len(np.unique(candidate_gids)) == len(candidate_gids)
             and np.array_equal(baseline_gids, candidate_gids),
             "station rigid transport requires unchanged unique source vertex IDs")
    _require(isinstance(baseline_station, Mapping)
             and baseline_station.get("transported_from_source_vertex_ancestry") is True,
             "station rigid transport requires a baseline material station")
    collar = baseline_station.get("frozen_collar_face_ids")
    _require(isinstance(collar, (list, tuple)) and bool(collar)
             and all(type(value) is int and 0 <= value < len(baseline_faces)
                     for value in collar)
             and len(set(collar)) == len(collar),
             "station rigid transport requires unique frozen source collar faces")
    collar_ids = np.asarray(collar, dtype=np.int64)
    baseline_section = _collar_local_source_section(
        baseline_surface, baseline_station,
        source_topology_surface=baseline_surface,
        collar_face_ids=collar,
        label="baseline material-section rigid-fit support",
        include_material_point_records=True)
    section_metric, section_issue = _section_metric_for_owner(
        baseline_section, baseline_surface.key[1],
        label="baseline material-section rigid-fit support")
    _require(section_issue is None and isinstance(section_metric, Mapping),
             "station rigid transport requires one authenticated baseline material section")
    section_face_ids_raw = section_metric.get("section_face_ids")
    section_ancestors = section_metric.get("source_face_ancestor_ids")
    _require(isinstance(section_face_ids_raw, list) and bool(section_face_ids_raw)
             and all(type(value) is int and 0 <= value < len(baseline_faces)
                     for value in section_face_ids_raw)
             and len(set(section_face_ids_raw)) == len(section_face_ids_raw)
             and section_ancestors == section_face_ids_raw,
             "baseline material-section faces lack exact local source-face ancestry")
    section_face_ids = np.asarray(section_face_ids_raw, dtype=np.int64)
    material_records = section_metric.get("material_point_records")
    _require(isinstance(material_records, list) and len(material_records) >= 3
             and section_metric.get("material_point_records_sha256")
             == _canonical_record_sha256(material_records),
             "baseline material section lacks authenticated intersection records")
    candidate_by_gid = {
        int(gid): local_id for local_id, gid in enumerate(candidate_gids)}
    alias_local_ids = set()
    alias_coordinate_classes = set()
    plane_vertex_local_ids = set()
    plane_vertex_coordinate_classes = set()
    interpolation_endpoint_local_ids = set()
    source_points_list = []
    candidate_points_list = []
    normalized_records = []

    def _class_points(local_ids, source_ids, label, *, plane_vertex=False,
                      interpolation_endpoint=False):
        _require(isinstance(local_ids, list) and bool(local_ids)
                 and all(type(value) is int and 0 <= value < len(baseline_positions)
                         for value in local_ids)
                 and isinstance(source_ids, list) and len(source_ids) == len(local_ids)
                 and all(type(value) is int for value in source_ids),
                 f"{label} has invalid source aliases")
        _require(all(local == source_id
                     for local, source_id in zip(local_ids, source_ids)),
                 f"{label} section-local vertex IDs are inconsistent")
        source = baseline_positions[local_ids[0]]
        _require(all(np.array_equal(source, baseline_positions[local])
                     for local in local_ids[1:]),
                 f"{label} source aliases are not coordinate-equal")
        candidate_locals = [
            candidate_by_gid[int(baseline_gids[local])] for local in local_ids]
        candidate = candidate_positions[candidate_locals[0]]
        _require(all(np.array_equal(candidate, candidate_positions[local])
                     for local in candidate_locals[1:]),
                 "candidate material-point endpoint aliases diverged")
        alias_local_ids.update(local_ids)
        coordinate_class = tuple(
            0.0 if float(value) == 0.0 else float(value) for value in source)
        alias_coordinate_classes.add(coordinate_class)
        if plane_vertex:
            plane_vertex_local_ids.update(local_ids)
            plane_vertex_coordinate_classes.add(coordinate_class)
        if interpolation_endpoint:
            interpolation_endpoint_local_ids.update(local_ids)
        return source.astype(np.float64), candidate.astype(np.float64)

    for record_index, record in enumerate(material_records):
        _require(isinstance(record, Mapping),
                 "material-section point record is not an object")
        kind = record.get("kind")
        if kind == "source_vertex":
            local_ids = record.get("coordinate_alias_local_vertex_ids")
            source_ids = record.get("coordinate_alias_source_vertex_ids")
            source, candidate = _class_points(
                local_ids, source_ids, f"material section vertex {record_index}",
                plane_vertex=True)
            source_points_list.append(source)
            candidate_points_list.append(candidate)
            normalized_records.append({
                "kind": kind,
                "source_vertex_ids": [
                    int(baseline_gids[value]) for value in local_ids],
            })
        elif kind == "source_edge_intersection":
            local_endpoints = record.get("endpoint_local_vertex_ids")
            local_aliases = record.get("endpoint_alias_local_vertex_ids")
            fraction = record.get("interpolation_fraction_from_first_endpoint")
            _require(isinstance(local_endpoints, list) and len(local_endpoints) == 2
                     and isinstance(local_aliases, list) and len(local_aliases) == 2
                     and isinstance(record.get("endpoint_alias_source_vertex_ids"), list)
                     and len(record["endpoint_alias_source_vertex_ids"]) == 2
                     and type(local_endpoints[0]) is int
                     and type(local_endpoints[1]) is int
                     and 0 <= local_endpoints[0] < len(baseline_positions)
                     and 0 <= local_endpoints[1] < len(baseline_positions)
                     and isinstance(fraction, (int, float))
                     and not isinstance(fraction, bool)
                     and np.isfinite(float(fraction))
                     and 0.0 < float(fraction) < 1.0,
                     f"material edge point {record_index} has invalid endpoints or fraction")
            endpoint_pairs = [
                _class_points(
                    local_aliases[i], local_aliases[i],
                    f"material edge point {record_index} endpoint {i}",
                    interpolation_endpoint=True)
                for i in range(2)]
            class_ids = record.get("canonical_endpoint_coordinate_class_ids")
            _require(isinstance(class_ids, list) and len(class_ids) == 2
                     and all(type(value) is int for value in class_ids)
                     and class_ids[0] < class_ids[1]
                     and all(type(value) is int for value in local_endpoints)
                     and all(local_endpoints[i] == min(local_aliases[i])
                             for i in range(2))
                     and all(isinstance(values, list)
                             and values == local_aliases[i]
                             for i, values in enumerate(
                                 record["endpoint_alias_source_vertex_ids"])),
                     f"material edge point {record_index} endpoint order/binding changed")
            endpoint_gids = [int(baseline_gids[local])
                             for local in local_endpoints]
            fraction = float(fraction)
            source_points_list.append(
                (1.0 - fraction) * endpoint_pairs[0][0]
                + fraction * endpoint_pairs[1][0])
            candidate_points_list.append(
                (1.0 - fraction) * endpoint_pairs[0][1]
                + fraction * endpoint_pairs[1][1])
            normalized_records.append({
                "kind": kind,
                "endpoint_source_vertex_ids": endpoint_gids,
                "interpolation_fraction_from_first_endpoint": fraction,
            })
        else:
            raise ValueError(
                f"material-section point {record_index} has unknown kind {kind!r}")

    alias_local_ids = sorted(alias_local_ids)
    plane_vertex_local_ids = sorted(plane_vertex_local_ids)
    interpolation_endpoint_local_ids = sorted(interpolation_endpoint_local_ids)
    _require(bool(alias_local_ids),
             "material-section point records have no bound source aliases")
    source_points = np.asarray(source_points_list, dtype=np.float64)
    candidate_points = np.asarray(candidate_points_list, dtype=np.float64)
    _require(source_points.shape == candidate_points.shape
             and source_points.ndim == 2 and source_points.shape[1] == 3
             and np.isfinite(source_points).all()
             and np.isfinite(candidate_points).all(),
             "material-section points are malformed")
    baseline_center = np.asarray(
        baseline_station.get("section_center_world_m"), dtype=np.float64)
    baseline_tangent = np.asarray(
        baseline_station.get("tangent_world_unit"), dtype=np.float64)
    _require(baseline_center.shape == (3,) and baseline_tangent.shape == (3,)
             and np.isfinite(baseline_center).all()
             and np.isfinite(baseline_tangent).all()
             and float(np.linalg.norm(baseline_tangent)) > 0.0,
             "baseline material station has invalid plane coordinates")
    baseline_tangent /= float(np.linalg.norm(baseline_tangent))
    section_plane_tolerance = section_metric.get("coordinate_ulp_tolerance_m")
    _require(isinstance(section_plane_tolerance, (int, float))
             and not isinstance(section_plane_tolerance, bool)
             and np.isfinite(float(section_plane_tolerance))
             and float(section_plane_tolerance) > 0.0,
             "baseline section has no finite coordinate-ULP plane tolerance")
    baseline_plane_residuals = np.abs(
        (source_points - baseline_center) @ baseline_tangent)
    maximum_baseline_plane_residual = float(
        baseline_plane_residuals.max(initial=0.0))
    _require(maximum_baseline_plane_residual
             <= float(section_plane_tolerance),
             "material-section support contains a point outside its baseline station plane")

    source_vertex_point_count = sum(
        record.get("kind") == "source_vertex" for record in material_records)
    source_edge_intersection_point_count = sum(
        record.get("kind") == "source_edge_intersection"
        for record in material_records)

    source_center = source_points.mean(axis=0)
    candidate_center = candidate_points.mean(axis=0)
    source_centered = source_points - source_center
    singular_values = np.linalg.svd(source_centered, compute_uv=False)
    _require(len(singular_values) >= 2
             and singular_values[1] > max(singular_values[0], 1e-30) * 1e-10,
             "station rigid-fit material section is geometrically underdetermined")
    covariance = source_centered.T @ (candidate_points - candidate_center)
    left, _values, right_t = np.linalg.svd(covariance)
    orientation = 1.0 if np.linalg.det(left @ right_t) >= 0.0 else -1.0
    rotation = left @ np.diag([1.0, 1.0, orientation]) @ right_t
    _require(float(np.linalg.det(rotation)) > 0.999999,
             "station rigid-fit transform is not a proper rotation")
    translation = candidate_center - source_center @ rotation
    residuals = np.linalg.norm(
        source_points @ rotation + translation - candidate_points, axis=1)
    maximum_residual = float(residuals.max(initial=0.0))
    rms_residual = float(np.sqrt(np.mean(residuals * residuals)))
    if maximum_fit_residual_m is not None:
        _require(maximum_residual <= maximum_fit_residual_m,
                 "candidate material-section support is not one local proper rigid motion "
                 f"(maximum residual {maximum_residual:.9g} m exceeds "
                 f"{maximum_fit_residual_m:.9g} m)")
    candidate_center_world = baseline_center @ rotation + translation
    candidate_tangent = baseline_tangent @ rotation
    candidate_tangent /= float(np.linalg.norm(candidate_tangent))
    source_binding = _source_geometry_binding(baseline_surface)
    candidate_binding = _source_geometry_binding(candidate_surface)
    alias_gids = np.asarray(baseline_gids[alias_local_ids], dtype="<u8")
    plane_vertex_gids = np.asarray(
        baseline_gids[plane_vertex_local_ids], dtype="<u8")
    interpolation_endpoint_gids = np.asarray(
        baseline_gids[interpolation_endpoint_local_ids], dtype="<u8")
    proof = {
        "schema": "numi.human.source-local-proper-rigid-station-transport.v1",
        "passed": True,
        "typed_identity": list(_identity(baseline_surface.key)),
        "method": (
            "proper Kabsch fit on exact source section vertices and edge-intersection "
            "points transported with their recorded interpolation fractions"),
        "candidate_geometry_used_only_for_local_motion_fit": True,
        "source_station_ancestry_sha256": _canonical_record_sha256({
            key: baseline_station.get(key) for key in (
                "material_anchor_source_vertex_ids",
                "tangent_support_source_vertex_ids",
                "frozen_collar_face_ids")}),
        "frozen_collar_face_ids": [int(value) for value in collar_ids],
        "frozen_collar_face_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(collar_ids, dtype="<u4").tobytes()).hexdigest(),
        "fit_support_scope": "BASELINE_IN_PLANE_MATERIAL_SECTION_NODES_V1",
        "baseline_selected_loop_face_ids": [int(value) for value in section_face_ids],
        "baseline_selected_loop_face_count": len(section_face_ids),
        "baseline_selected_loop_face_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(section_face_ids, dtype="<u4").tobytes()).hexdigest(),
        "baseline_selected_loop_ancestry_sha256": section_metric[
            "source_face_ancestor_ids_sha256"],
        "source_section_fit_point_count": len(material_records),
        "edge_intersection_points_are_fit_observations": True,
        "off_plane_edge_endpoints_are_fit_observations": False,
        "source_section_plane_vertex_local_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(plane_vertex_local_ids, dtype="<u4").tobytes()).hexdigest(),
        "source_section_plane_vertex_alias_count": len(plane_vertex_local_ids),
        "source_section_plane_vertex_coordinate_class_count": len(
            plane_vertex_coordinate_classes),
        "source_section_interpolation_endpoint_local_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(interpolation_endpoint_local_ids,
                                 dtype="<u4").tobytes()).hexdigest(),
        "source_section_interpolation_endpoint_vertex_count": len(
            interpolation_endpoint_local_ids),
        "source_section_alias_local_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(alias_local_ids, dtype="<u4").tobytes()).hexdigest(),
        "source_section_alias_vertex_count": len(alias_local_ids),
        "source_section_alias_unique_coordinate_class_count": len(
            alias_coordinate_classes),
        "source_section_source_vertex_point_count": source_vertex_point_count,
        "source_section_edge_intersection_point_count": source_edge_intersection_point_count,
        "all_source_material_points_on_baseline_station_plane": True,
        "maximum_baseline_station_plane_residual_m": maximum_baseline_plane_residual,
        "baseline_station_plane_residual_bound_m": float(section_plane_tolerance),
        "source_section_material_point_records_sha256": section_metric[
            "material_point_records_sha256"],
        "source_section_material_point_binding_sha256": _canonical_record_sha256(
            normalized_records),
        "baseline_material_points_sha256": hashlib.sha256(
            np.ascontiguousarray(source_points, dtype="<f8").tobytes()).hexdigest(),
        "candidate_material_points_sha256": hashlib.sha256(
            np.ascontiguousarray(candidate_points, dtype="<f8").tobytes()).hexdigest(),
        "source_section_plane_vertex_source_vertex_ids": [
            int(value) for value in plane_vertex_gids],
        "source_section_plane_vertex_source_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(plane_vertex_gids, dtype="<u8").tobytes()).hexdigest(),
        "source_section_interpolation_endpoint_source_vertex_ids": [
            int(value) for value in interpolation_endpoint_gids],
        "source_section_interpolation_endpoint_source_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(interpolation_endpoint_gids, dtype="<u8").tobytes()).hexdigest(),
        "source_section_alias_source_vertex_ids": [
            int(value) for value in alias_gids],
        "source_section_alias_source_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(alias_gids, dtype="<u8").tobytes()).hexdigest(),
        "baseline_surface_binding": source_binding,
        "candidate_surface_binding": candidate_binding,
        "rotation_row_vector": rotation.tolist(),
        "translation_m": translation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "maximum_fit_residual_m": maximum_residual,
        "rms_fit_residual_m": rms_residual,
        "maximum_fit_residual_bound_m": (
            None if maximum_fit_residual_m is None
            else float(maximum_fit_residual_m)),
        "all_exact_coordinate_aliases_preserved": True,
        "anatomical_admission": False,
        "native_admission": False,
    }
    if maximum_fit_residual_m is None:
        material_calibre = _material_section_ring_calibre(
            source_points, candidate_points, baseline_station, rotation,
            section_metric)
        proof.update({
            "schema": "numi.human.source-material-section-transport.v1",
            "method": (
                "same authenticated edge-fraction material ring in the "
                "best-fit transported frame; no rigid-fit admission"),
            "passed": material_calibre["passed"],
            "proper_rigid_applicability_bound_m": 5.0e-7,
            "proper_rigid_applicability_passed": maximum_residual <= 5.0e-7,
            "material_section_calibre": material_calibre,
        })
    candidate_station = {
        **dict(baseline_station),
        "section_center_world_m": candidate_center_world.tolist(),
        "tangent_world_unit": candidate_tangent.tolist(),
        "material_anchor_world_m": (
            np.asarray(baseline_station["material_anchor_world_m"], dtype=np.float64)
            @ rotation + translation).tolist(),
        "station_transport_method": (
            "SOURCE_LOCAL_PROPER_RIGID_FIT_V1"
            if maximum_fit_residual_m is not None
            else "SOURCE_MATERIAL_SECTION_FRAME_V1"),
        "transported_from_source_vertex_ancestry": True,
    }
    return candidate_station, proof


def _transport_material_station_by_material_section(
    baseline_surface, candidate_surface, baseline_station,
):
    """Transport an authenticated section by its material ring, not rigid fit."""
    candidate_station, proof = (
        _transport_material_station_by_local_proper_rigid_fit(
            baseline_surface, candidate_surface, baseline_station,
            maximum_fit_residual_m=None))
    calibre = proof.get("material_section_calibre")
    _require(isinstance(calibre, Mapping)
             and proof.get("schema")
                 == "numi.human.source-material-section-transport.v1"
             and calibre.get("declared_maximum_warp_to_feret_ratio") == 0.01,
             "material-section transport did not produce its pinned calibre proof")
    proof["passed"] = calibre.get("passed") is True
    candidate_station["station_transport_method"] = (
        "SOURCE_MATERIAL_SECTION_FRAME_V1")
    return candidate_station, proof


def _source_route_anchor_from_evidence(surface, evidence):
    _require(isinstance(evidence, Mapping),
             "local source station requires frozen neighbor-route evidence")
    typed_link_raw = evidence.get("typed_link")
    _require(isinstance(typed_link_raw, (tuple, list)) and len(typed_link_raw) == 2,
             "neighbor-route evidence needs one typed source link")
    typed_link = tuple(_identity(value) for value in typed_link_raw)
    _require(typed_link[0] != typed_link[1],
             "neighbor-route evidence must join distinct typed surfaces")
    anchor_identity = _identity(surface.key)
    _require(anchor_identity in typed_link,
             "neighbor-route link must share the station's source surface")
    anchor_position = typed_link.index(anchor_identity)
    peer = evidence.get("peer_surface")
    _require(hasattr(peer, "key") and hasattr(peer, "faces")
             and hasattr(peer, "positions") and hasattr(peer, "source_vertex_ids")
             and _identity(peer.key) == typed_link[1 - anchor_position],
             "neighbor-route peer must be the other typed source surface")
    raw_pairs = evidence.get("frozen_face_pairs")
    first_surface, second_surface = (
        (surface, peer) if anchor_position == 0 else (peer, surface))
    pairs = _pair_rows(
        raw_pairs, name="frozen neighbor-route collar pairset",
        bridge_face_count=len(first_surface.faces),
        target_face_count=len(second_surface.faces))
    pair_bytes = np.ascontiguousarray(raw_pairs, dtype="<u4").tobytes()
    pairset_sha256 = hashlib.sha256(pair_bytes).hexdigest()
    report_sha256 = evidence.get("source_report_sha256")
    for label, digest in (("neighbor-route source report", report_sha256),
                          ("neighbor-route source pairset",
                           evidence.get("source_pairset_sha256"))):
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 label + " must be a lowercase SHA-256")
    _require(pairset_sha256 == evidence["source_pairset_sha256"],
             "neighbor-route frozen pairset differs from its source pin")
    report_path = evidence.get("source_report_path")
    _require(isinstance(report_path, str) and report_path.startswith("/"),
             "neighbor-route source report path must be absolute")
    pair_field = evidence.get("source_pair_field")
    typed_field = evidence.get("source_typed_link_field")
    _require(isinstance(pair_field, str) and pair_field
             and isinstance(typed_field, str) and typed_field,
             "neighbor-route evidence must name its report pair and typed-link fields")
    try:
        with open(report_path, "rb") as report_file:
            report_bytes = report_file.read()
        report = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "brachiocephalic captured-pose geometry: cannot read frozen neighbor-route report"
        ) from error
    _require(hashlib.sha256(report_bytes).hexdigest() == report_sha256,
             "neighbor-route report bytes differ from their frozen SHA-256")
    _require(isinstance(report, Mapping)
             and report.get("passed") is True
             and report.get("source_topology_passed") is True
             and report.get("interface_gate_passed") is True
             and report.get("solid_gate_passed") is True
             and report.get("independent_source_collar_calibre_passed") is True,
             "neighbor-route report lacks its complete source-interface proof")
    report_link_raw = report.get(typed_field)
    _require(isinstance(report_link_raw, (tuple, list))
             and len(report_link_raw) == 2
             and tuple(_identity(value) for value in report_link_raw) == typed_link,
             "neighbor-route report has a different typed source link")
    report_pairs_raw = report.get(pair_field)
    _require(isinstance(report_pairs_raw, (tuple, list))
             and [list(pair) if isinstance(pair, (tuple, list)) else pair
                  for pair in report_pairs_raw]
             == [list(pair) for pair in raw_pairs],
             "neighbor-route report face pairs differ from the frozen source pairset")
    report_pairs = _pair_rows(
        report_pairs_raw, name="neighbor-route report face pairset",
        bridge_face_count=len(first_surface.faces),
        target_face_count=len(second_surface.faces))
    _require(report_pairs == pairs,
             "neighbor-route report pairset differs from its typed source surfaces")
    if "source_pair_count" in evidence:
        _require(type(evidence["source_pair_count"]) is int
                 and evidence["source_pair_count"] == len(pairs),
                 "neighbor-route source pair count differs from its frozen pairset")
    anchor_face_ids = sorted({pair[anchor_position] for pair in pairs})
    anchor_vertices = np.unique(
        np.asarray(surface.faces, dtype=np.int64)[anchor_face_ids].reshape(-1))
    _require(len(anchor_vertices) > 0,
             "neighbor-route collar has no source vertices")
    positions = np.asarray(surface.positions, dtype=np.float64)
    source_vertex_ids = np.asarray(surface.source_vertex_ids)
    anchor_world = positions[anchor_vertices].mean(axis=0)
    return anchor_world, {
        "schema": "numi.human.frozen-source-neighbor-route-anchor.v1",
        "typed_link": [list(key) for key in typed_link],
        "station_surface_identity": list(anchor_identity),
        "peer_surface_identity": list(typed_link[1 - anchor_position]),
        "source_report_path": report_path,
        "source_report_sha256": report_sha256,
        "source_pair_field": pair_field,
        "source_typed_link_field": typed_field,
        "source_pairset_sha256": pairset_sha256,
        "source_pair_count": len(pairs),
        "anchor_face_ids": anchor_face_ids,
        "anchor_face_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(anchor_face_ids, dtype="<u4").tobytes()).hexdigest(),
        "anchor_source_vertex_ids": [
            int(value) for value in source_vertex_ids[anchor_vertices]],
        "anchor_source_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(
                source_vertex_ids[anchor_vertices], dtype="<u8").tobytes()).hexdigest(),
        "anchor_world_m": [float(value) for value in anchor_world],
        "station_surface_binding": _source_geometry_binding(surface),
        "peer_surface_binding": _source_geometry_binding(peer),
    }


def _port_station_spec(
    surface, witness_points, collar_face_ids, *, route_anchor=None,
    station_policy="GLOBAL_SOURCE_PCA_V1",
):
    positions = surface.positions.astype(np.float64)
    faces = surface.faces
    witness = np.asarray(witness_points, dtype=np.float64)
    _require(witness.ndim == 2 and witness.shape[1] == 3 and len(witness) > 0,
             "station inference requires exact interface witness points")
    _require(station_policy in (
        "GLOBAL_SOURCE_PCA_V1", "COLLAR_LOCAL_PCA_V1"),
        "unknown source station selection policy")
    junction = witness.mean(axis=0)
    local_tangent_support = None
    local_pca_eigenvalues = None
    if station_policy == "COLLAR_LOCAL_PCA_V1":
        _require(route_anchor is not None,
                 "collar-local source station needs a frozen neighbor-route anchor")
        collar = np.asarray(collar_face_ids, dtype=np.int64)
        _require(collar.ndim == 1 and len(collar) > 0
                 and int(collar.min()) >= 0 and int(collar.max()) < len(faces),
                 "collar-local station needs valid frozen source collar faces")
        local_tangent_support = np.unique(faces[collar].reshape(-1))
        cloud = positions[local_tangent_support]
        local_center = cloud.mean(axis=0)
        centered = cloud - local_center
        covariance = centered.T @ centered / max(1, len(cloud))
        local_pca_eigenvalues, vectors = np.linalg.eigh(covariance)
        _require(np.isfinite(local_pca_eigenvalues).all()
                 and local_pca_eigenvalues[-1] - local_pca_eigenvalues[-2]
                 > max(1e-24, abs(float(local_pca_eigenvalues[-1])) * 1e-6),
                 "frozen collar does not define a unique local source tangent")
        direction = vectors[:, -1]
        route_direction = np.asarray(route_anchor, dtype=np.float64) - junction
        route_norm = float(np.linalg.norm(route_direction))
        _require(route_direction.shape == (3,) and np.isfinite(route_direction).all()
                 and route_norm > 1e-15,
                 "frozen neighbor-route anchor does not orient the local source tangent")
        alignment = float(np.dot(direction, route_direction / route_norm))
        _require(abs(alignment) > 1e-8,
                 "local source tangent is orthogonal to its frozen route anchor")
        if alignment < 0.0:
            direction = -direction
        method = (
            "PCA of frozen source collar-face vertices oriented toward the "
            "independently pinned neighboring route collar")
        tangent = direction / float(np.linalg.norm(direction))
        ancestry = _material_vertex_ancestry(
            surface, tangent, collar_face_ids,
            tangent_support_vertex_ids=local_tangent_support)
    else:
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
        "station_selection_policy": station_policy,
        **({"local_pca_eigenvalues_m2": local_pca_eigenvalues.tolist(),
            "local_tangent_support_local_vertex_ids": local_tangent_support.tolist(),
            "local_tangent_support_source_vertex_ids": [
                int(surface.source_vertex_ids[index])
                for index in local_tangent_support]}
           if local_pca_eigenvalues is not None else {}),
        **ancestry,
    }
    return _transport_material_station(surface, spec)


def _source_geometry_binding(surface):
    return {
        "identity": list(surface.key),
        "positions_sha256": hashlib.sha256(
            np.ascontiguousarray(surface.positions, dtype="<f4").tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(
            np.ascontiguousarray(surface.faces, dtype="<u4").tobytes()).hexdigest(),
        "source_vertex_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(surface.source_vertex_ids, dtype="<u8").tobytes()).hexdigest(),
    }


def _canonical_record_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _source_interface_points_sha256(points):
    rows = np.asarray(points, dtype="<f8")
    if rows.size == 0:
        rows = rows.reshape((0, 3))
    _require(rows.ndim == 2 and rows.shape[1] == 3 and np.isfinite(rows).all(),
             "source interface points must be finite XYZ rows")
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def _transverse_overlap_proof_passed(proof, observed_pairs):
    if not isinstance(proof, Mapping) or proof.get("passed") is not True:
        return False
    if (proof.get("requires_strict_positive_exact_barycentric_coordinates") is not True
            or proof.get("requires_all_four_witnesses_from_same_face_pair") is not True
            or proof.get("uses_exact_rational_point_location_in_opposite_closed_surface")
                is not True):
        return False
    pair = proof.get("passing_face_pair")
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(type(index) is not int or index < 0 for index in pair)):
        return False
    if not isinstance(observed_pairs, (tuple, list)):
        return False
    observed = set()
    for raw_pair in observed_pairs:
        if (not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2
                or any(type(index) is not int or index < 0 for index in raw_pair)):
            return False
        observed.add(tuple(raw_pair))
    if tuple(map(int, pair)) not in observed:
        return False
    directions = proof.get("directional_surface_witnesses")
    if not isinstance(directions, Mapping):
        return False
    for role in ("first_surface_against_second", "second_surface_against_first"):
        samples = directions.get(role)
        if not isinstance(samples, Mapping):
            return False
        for side in ("inside", "outside"):
            witness = samples.get(side)
            if (not isinstance(witness, Mapping)
                    or witness.get("location_in_opposite_surface") != side
                    or witness.get("face_pair") != list(pair)
                    or not isinstance(witness.get("face_barycentric_rational"), list)):
                return False
            barycentric = witness["face_barycentric_rational"]
            if len(barycentric) != 3:
                return False
            exact_barycentric = []
            for rational in barycentric:
                if (not isinstance(rational, list) or len(rational) != 2
                        or not all(isinstance(value, str) and value.isdecimal()
                                   for value in rational)
                        or int(rational[0]) <= 0 or int(rational[1]) <= 0):
                    return False
                exact_barycentric.append(Fraction(
                    int(rational[0]), int(rational[1])))
            if sum(exact_barycentric, Fraction(0)) != 1:
                return False
    return True


def _make_transverse_overlap_witnesses(provider, verifier, bridge, target, observed_pairs):
    if provider is None:
        return None
    owner_module = sys.modules.get(getattr(provider, "__module__", ""))
    exact_owner = getattr(
        owner_module, "_exact_transverse_overlap_side_witnesses", None)
    _require(callable(provider) and provider is exact_owner,
             "transverse overlap witness provider must be the pinned exact owner")
    proof = exact_owner(verifier, bridge, target, observed_pairs)
    _require(isinstance(proof, Mapping),
             "exact transverse overlap owner returned a malformed proof")
    return dict(proof)


def build_source_junction_station_declaration(
    verifier, *, bridge_key, target_key, baseline_bridge_surface,
    baseline_target_surface, frozen_face_pairs, declared_vessel_ports,
    anchor_role="bridge", transverse_witness_provider=None,
    station_policy="GLOBAL_SOURCE_PCA_V1", route_anchor_evidence=None,
):
    """Freeze a junction station from baseline source evidence only.

    This function has no candidate-surface argument. It binds the selected
    source station to the independent baseline surfaces, exact baseline face
    contacts, and the frozen collar ancestry. Candidate stations are created
    later by transporting this source material station through unchanged
    source-vertex ancestry.
    """
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    _require(anchor_role in ("bridge", "target"),
             "source station anchor role must be bridge or target")
    _require(
        isinstance(declared_vessel_ports, (set, frozenset, tuple, list))
        and len(declared_vessel_ports) == 1,
        "source station needs one independently declared typed vessel edge")
    edges = tuple(declared_vessel_ports)
    _require(all(isinstance(edge, (tuple, list)) and len(edge) == 2 for edge in edges),
             "source station typed edge is malformed")
    normalized_edges = tuple((_identity(edge[0]), _identity(edge[1])) for edge in edges)
    _require(normalized_edges == ((bridge_key, target_key),),
             "source station typed edge must exactly match the audited link")
    if station_policy == "COLLAR_LOCAL_PCA_V1":
        _require(isinstance(route_anchor_evidence, Mapping),
                 "collar-local station needs an independent typed source route")
        route_link_raw = route_anchor_evidence.get("typed_link")
        _require(isinstance(route_link_raw, (tuple, list))
                 and len(route_link_raw) == 2,
                 "collar-local route anchor needs one typed source link")
        route_link = tuple(_identity(value) for value in route_link_raw)
        _require(set(route_link) != {bridge_key, target_key},
                 "collar-local tangent route must be an independent port, not the measured link")
    bridge = _prepare_vessel(verifier, bridge_key, baseline_bridge_surface)
    target = _prepare_vessel(verifier, target_key, baseline_target_surface)
    pairs = _pair_rows(
        frozen_face_pairs, name="frozen source station face-pair ancestry",
        bridge_face_count=len(bridge.faces), target_face_count=len(target.faces))
    bridge_patch = sorted({first for first, _ in pairs})
    target_patch = sorted({second for _, second in pairs})
    allowed = tuple((first, second)
                    for first in bridge_patch for second in target_patch)
    audit = verifier.audit_surface_pair(bridge, target, intended_face_pairs=allowed)
    observed = tuple(tuple(map(int, pair))
                     for pair in audit.get("declared_interface_triangle_pairs", []))
    exact_points = _exact_interface_points(verifier, bridge, target, observed)
    boundary = _declared_partial_vessel_port_overlap(
        bridge_key, target_key, audit, observed,
        audit.get("unexpected_triangle_pairs", ()), exact_points, allowed,
        declared_vessel_ports=normalized_edges)
    transverse = _make_transverse_overlap_witnesses(
        transverse_witness_provider, verifier, bridge, target, observed)
    transverse_passed = _transverse_overlap_proof_passed(transverse, observed)
    source_pair_passed = bool(exact_points) and (
        bool(audit.get("passed")) or boundary.get("passed") is True
        or transverse_passed)
    _require(source_pair_passed,
             "independent source baseline does not prove the frozen typed interface")
    anchor_key = bridge_key if anchor_role == "bridge" else target_key
    anchor = bridge if anchor_role == "bridge" else target
    anchor_patch = bridge_patch if anchor_role == "bridge" else target_patch
    route_anchor = None
    route_anchor_record = None
    if station_policy == "COLLAR_LOCAL_PCA_V1":
        route_anchor, route_anchor_record = _source_route_anchor_from_evidence(
            anchor, route_anchor_evidence)
    else:
        _require(route_anchor_evidence is None,
                 "neighbor-route evidence is only valid for collar-local stations")
    source_station = _port_station_spec(
        anchor, exact_points, anchor_patch, route_anchor=route_anchor,
        station_policy=station_policy)
    baseline_source_station_section = None
    if station_policy == "COLLAR_LOCAL_PCA_V1":
        baseline_source_station_section = _collar_local_source_section(
            anchor, source_station, source_topology_surface=anchor,
            collar_face_ids=source_station["frozen_collar_face_ids"],
            label="baseline source collar-local station")
        _require(baseline_source_station_section.get("passed") is True
                 and baseline_source_station_section.get(
                     "selected_local_loop_count") == 1,
                 "source collar-local station must select one closed baseline loop")
    source_ancestry = {
        key: source_station[key] for key in (
            "material_anchor_source_vertex_ids",
            "tangent_support_source_vertex_ids",
            "frozen_collar_face_ids",
        )
    }
    pair_record = [list(pair) for pair in pairs]
    observed_record = [list(pair) for pair in observed]
    source_evidence = {
        "schema": "numi.human.source-baseline-vessel-junction-station.v1",
        "independent_source_baseline_only": True,
        "typed_link": [list(bridge_key), list(target_key)],
        "anchor_role": anchor_role,
        "anchor_identity": list(anchor_key),
        "baseline_source_surface_bindings": {
            "bridge": _source_geometry_binding(bridge),
            "target": _source_geometry_binding(target),
        },
        "frozen_face_pairs_sha256": _canonical_record_sha256(pair_record),
        "frozen_face_pair_count": len(pair_record),
        "baseline_observed_face_pairs_sha256": _canonical_record_sha256(observed_record),
        "baseline_observed_face_pair_count": len(observed_record),
        "baseline_exact_interface_points_sha256": _source_interface_points_sha256(exact_points),
        "baseline_exact_interface_point_count": len(exact_points),
        "baseline_port_audit_passed": bool(audit.get("passed")),
        "baseline_port_pair_passed": source_pair_passed,
        "baseline_port_pair_witness_kind": (
            "central_p6" if audit.get("passed") is True else
            "coarse_exact_boundary" if boundary.get("passed") is True else
            "exact_transverse_inside_outside"),
        "baseline_declared_partial_overlap_boundary": boundary,
        "baseline_exact_transverse_overlap_side_witnesses": transverse,
        "station_selection_policy": station_policy,
        "neighbor_route_anchor_evidence": route_anchor_record,
        "baseline_source_station_section": (
            None if baseline_source_station_section is None else {
                "passed": baseline_source_station_section.get("passed"),
                "loop_count": baseline_source_station_section.get("loop_count"),
                "selected_local_loop_count": baseline_source_station_section.get(
                    "selected_local_loop_count"),
                "selected_local_loop_index": baseline_source_station_section.get(
                    "selected_local_loop_index"),
                "local_loop_selection": baseline_source_station_section.get(
                    "local_loop_selection"),
                "area_m2": baseline_source_station_section.get("area_m2"),
                "minimum_feret_width_m": baseline_source_station_section.get(
                    "minimum_feret_width_m"),
                "source_face_ancestor_ids": baseline_source_station_section.get(
                    "source_face_ancestor_ids"),
            }),
        "source_station_provenance": source_station.get("provenance"),
        "source_station_spec_sha256": _canonical_record_sha256(source_station),
        "source_station_is_candidate_independent": True,
    }
    declaration_id = "source-baseline-junction-station-v1:" + _canonical_record_sha256({
        "evidence": source_evidence,
        "station": source_station,
    })
    return {
        "schema": "numi.human.source-baseline-junction-station-declaration.v1",
        "declaration_id": declaration_id,
        "typed_link": [list(bridge_key), list(target_key)],
        "station_role": anchor_role,
        "expected_owner_ids": sorted((bridge_key[1], target_key[1])),
        "bridge_owner_id": bridge_key[1],
        "target_owner_id": target_key[1],
        "anchor_owner_id": anchor_key[1],
        "source_ancestry": source_ancestry,
        "source_station_spec": source_station,
        "source_evidence": source_evidence,
    }


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


def _orient2d_exact(first, second, third):
    ax, ay = (Fraction.from_float(float(value)) for value in first)
    bx, by = (Fraction.from_float(float(value)) for value in second)
    cx, cy = (Fraction.from_float(float(value)) for value in third)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_on_segment2d_exact(point, first, second):
    return all(
        min(Fraction.from_float(float(first[axis])),
            Fraction.from_float(float(second[axis])))
        <= Fraction.from_float(float(point[axis]))
        <= max(Fraction.from_float(float(first[axis])),
               Fraction.from_float(float(second[axis])))
        for axis in range(2)
    )


def _segments_intersect2d_exact(first_a, first_b, second_a, second_b):
    orientations = (
        _orient2d_exact(first_a, first_b, second_a),
        _orient2d_exact(first_a, first_b, second_b),
        _orient2d_exact(second_a, second_b, first_a),
        _orient2d_exact(second_a, second_b, first_b),
    )
    if orientations[0] == 0 and _point_on_segment2d_exact(
            second_a, first_a, first_b):
        return True
    if orientations[1] == 0 and _point_on_segment2d_exact(
            second_b, first_a, first_b):
        return True
    if orientations[2] == 0 and _point_on_segment2d_exact(
            first_a, second_a, second_b):
        return True
    if orientations[3] == 0 and _point_on_segment2d_exact(
            first_b, second_a, second_b):
        return True
    return (
        ((orientations[0] < 0 < orientations[1])
         or (orientations[1] < 0 < orientations[0]))
        and
        ((orientations[2] < 0 < orientations[3])
         or (orientations[3] < 0 < orientations[2]))
    )


def _simple_closed_polygon2d_exact(points):
    polygon = np.asarray(points, dtype=np.float64)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        return False
    if any(np.array_equal(polygon[index], polygon[(index + 1) % len(polygon)])
           for index in range(len(polygon))):
        return False
    for first_edge in range(len(polygon)):
        first_next = (first_edge + 1) % len(polygon)
        for second_edge in range(first_edge + 1, len(polygon)):
            second_next = (second_edge + 1) % len(polygon)
            if first_next == second_edge or second_next == first_edge:
                continue
            if _segments_intersect2d_exact(
                    polygon[first_edge], polygon[first_next],
                    polygon[second_edge], polygon[second_next]):
                return False
    return True


def _signed_polygon_area2d_exact(points):
    polygon = np.asarray(points, dtype=np.float64)
    total = Fraction(0)
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        ax, ay = (Fraction.from_float(float(value)) for value in first)
        bx, by = (Fraction.from_float(float(value)) for value in second)
        total += ax * by - ay * bx
    return float(total / 2)


def _material_section_ring_calibre(
    source_points, candidate_points, baseline_station, rotation,
    baseline_section_metric, *, maximum_warp_to_feret_ratio=0.01,
):
    """Measure one transported source ring without requiring a rigid fit."""
    source = np.asarray(source_points, dtype=np.float64)
    candidate = np.asarray(candidate_points, dtype=np.float64)
    _require(source.shape == candidate.shape and source.ndim == 2
             and source.shape[1] == 3 and len(source) >= 3
             and np.isfinite(source).all() and np.isfinite(candidate).all(),
             "material-section ring arrays are invalid")
    _require(np.isfinite(maximum_warp_to_feret_ratio)
             and maximum_warp_to_feret_ratio > 0.0,
             "material-section planarity ratio must be positive and finite")
    tangent = np.asarray(
        baseline_station.get("tangent_world_unit"), dtype=np.float64)
    _require(tangent.shape == (3,) and np.isfinite(tangent).all()
             and float(np.linalg.norm(tangent)) > 0.0,
             "material-section baseline tangent is invalid")
    tangent /= float(np.linalg.norm(tangent))
    helper = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    first_axis = np.cross(tangent, helper)
    first_axis /= float(np.linalg.norm(first_axis))
    second_axis = np.cross(tangent, first_axis)
    candidate_first_axis = first_axis @ rotation
    candidate_second_axis = second_axis @ rotation
    candidate_normal = tangent @ rotation
    candidate_normal /= float(np.linalg.norm(candidate_normal))

    source_center = source.mean(axis=0)
    candidate_center = candidate.mean(axis=0)
    source_projected = np.column_stack((
        (source - source_center) @ first_axis,
        (source - source_center) @ second_axis,
    ))
    candidate_projected = np.column_stack((
        (candidate - candidate_center) @ candidate_first_axis,
        (candidate - candidate_center) @ candidate_second_axis,
    ))
    source_signed_area = _signed_polygon_area2d_exact(source_projected)
    candidate_signed_area = _signed_polygon_area2d_exact(candidate_projected)
    source_area = abs(source_signed_area)
    candidate_projected_area = abs(candidate_signed_area)
    source_width = _minimum_feret_width(_convex_hull_2d(source_projected))
    candidate_width = _minimum_feret_width(_convex_hull_2d(candidate_projected))
    source_simple = _simple_closed_polygon2d_exact(source_projected)
    candidate_simple = _simple_closed_polygon2d_exact(candidate_projected)

    source_center2 = source_projected.mean(axis=0)
    candidate_center2 = candidate_projected.mean(axis=0)
    source_orientation = 1 if source_signed_area > 0.0 else -1
    candidate_orientation = 1 if candidate_signed_area > 0.0 else -1
    source_fan_oriented = all(
        source_orientation * _orient2d_exact(
            source_center2, source_projected[index],
            source_projected[(index + 1) % len(source_projected)]) >= 0
        for index in range(len(source_projected)))
    candidate_fan_oriented = all(
        candidate_orientation * _orient2d_exact(
            candidate_center2, candidate_projected[index],
            candidate_projected[(index + 1) % len(candidate_projected)]) >= 0
        for index in range(len(candidate_projected)))

    source_fan_center = source.mean(axis=0)
    candidate_fan_center = candidate.mean(axis=0)
    source_fan_area = sum(
        0.5 * float(np.linalg.norm(np.cross(
            source[index] - source_fan_center,
            source[(index + 1) % len(source)] - source_fan_center)))
        for index in range(len(source)))
    candidate_fan_area = sum(
        0.5 * float(np.linalg.norm(np.cross(
            candidate[index] - candidate_fan_center,
            candidate[(index + 1) % len(candidate)] - candidate_fan_center)))
        for index in range(len(candidate)))
    source_warp = float(np.max(np.abs((source - source_center) @ tangent)))
    candidate_warp = float(np.max(np.abs(
        (candidate - candidate_center) @ candidate_normal)))
    source_perimeter = float(np.linalg.norm(
        np.roll(source, -1, axis=0) - source, axis=1).sum())
    candidate_perimeter = float(np.linalg.norm(
        np.roll(candidate, -1, axis=0) - candidate, axis=1).sum())
    _require(source_fan_area > 0.0 and candidate_fan_area > 0.0
             and source_perimeter > 0.0,
             "material-section ring has zero 3-D area or perimeter")

    baseline_area = float(baseline_section_metric.get("area_m2"))
    baseline_width = float(baseline_section_metric.get("minimum_feret_width_m"))
    _require(np.isfinite(baseline_area) and baseline_area > 0.0
             and np.isfinite(baseline_width) and baseline_width > 0.0,
             "material-section baseline calibre is invalid")
    _require(abs(source_area - baseline_area)
             <= max(baseline_area * 1.0e-7, 1.0e-18)
             and abs(source_width - baseline_width)
             <= max(baseline_width * 1.0e-7, 1.0e-12),
             "material ring does not reproduce the independent source section")
    area_bounds = (0.8, 1.25)
    width_bounds = (0.9, 1.1)
    projected_area_ratio = candidate_projected_area / baseline_area
    fan_area_ratio = candidate_fan_area / source_fan_area
    width_ratio = candidate_width / baseline_width
    width_ratio_bounds = (
        (candidate_width - 2.0 * candidate_warp) / baseline_width,
        (candidate_width + 2.0 * candidate_warp) / baseline_width,
    )
    warp_ratio = candidate_warp / baseline_width
    projection_error = abs(candidate_fan_area - candidate_projected_area)
    perimeter_ratio = candidate_perimeter / source_perimeter
    passed = bool(
        source_simple and candidate_simple
        and source_fan_oriented and candidate_fan_oriented
        and source_orientation == candidate_orientation
        and warp_ratio <= maximum_warp_to_feret_ratio
        and area_bounds[0] <= projected_area_ratio <= area_bounds[1]
        and area_bounds[0] <= fan_area_ratio <= area_bounds[1]
        and width_bounds[0] <= width_ratio_bounds[0]
        and width_ratio_bounds[1] <= width_bounds[1]
    )
    return {
        "passed": passed,
        "method": (
            "same-source-edge-fraction material ring, best-fit transported "
            "frame, projected polygon plus 3D centroid-fan area"),
        "source_material_ring_is_simple_closed": source_simple,
        "candidate_material_ring_is_simple_closed": candidate_simple,
        "source_centroid_fan_stays_inside_or_on_ring": source_fan_oriented,
        "candidate_centroid_fan_stays_inside_or_on_ring": candidate_fan_oriented,
        "source_and_candidate_ring_orientation_match": (
            source_orientation == candidate_orientation),
        "source_projected_area_m2": source_area,
        "candidate_projected_area_m2": candidate_projected_area,
        "source_centroid_fan_area_m2": source_fan_area,
        "candidate_centroid_fan_area_m2": candidate_fan_area,
        "projected_area_ratio": projected_area_ratio,
        "centroid_fan_area_ratio": fan_area_ratio,
        "source_minimum_feret_width_m": source_width,
        "candidate_projected_minimum_feret_width_m": candidate_width,
        "projected_minimum_feret_width_ratio": width_ratio,
        "minimum_feret_width_ratio_interval": list(width_ratio_bounds),
        "source_maximum_out_of_plane_warp_m": source_warp,
        "candidate_maximum_out_of_plane_warp_m": candidate_warp,
        "maximum_warp_to_source_minimum_feret_ratio": warp_ratio,
        "declared_maximum_warp_to_feret_ratio": maximum_warp_to_feret_ratio,
        "projection_area_error_m2": projection_error,
        "projection_area_error_fraction_of_3d_fan": (
            projection_error / candidate_fan_area
            if candidate_fan_area > 0.0 else None),
        "source_perimeter_m": source_perimeter,
        "candidate_3d_perimeter_m": candidate_perimeter,
        "perimeter_ratio_diagnostic_only": perimeter_ratio,
        "declared_preservation_intervals": {
            "area": list(area_bounds),
            "minimum_feret_width": list(width_bounds),
        },
        "clinical_calibre_claim": False,
    }



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
                   face_generated_mask=None, include_material_point_records=False):
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
    canonical_local_ids = {}
    for local_id, canonical_id in enumerate(canonical_source_ids):
        canonical_local_ids.setdefault(int(canonical_id), []).append(local_id)
    source_vertex_ids = np.arange(len(points), dtype="<u8")

    def representative_local_id(canonical_id):
        local_ids = canonical_local_ids[int(canonical_id)]
        return min(local_ids, key=lambda local_id: (
            int(source_vertex_ids[local_id]), local_id))

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
        edge_intersections[edge] = (
            key, np.asarray(point, dtype=np.float64), float(fraction))
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
            loop_records.append((loops[-1], tuple(loop_edges), tuple(loop)))
        else:
            return {
                "passed": False, "loop_count": len(loops),
                "reason": "section cycle did not close",
                "segment_count": len(segments), **details,
            }
    positive_floor = tolerance * 64.0
    loop_metrics = []
    for loop, loop_edges, loop_node_keys in loop_records:
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
        metric = {
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
        }
        if include_material_point_records:
            material_records = []
            for node_key in loop_node_keys:
                if node_key[0] == "v":
                    canonical_id = int(node_key[1])
                    local_id = representative_local_id(canonical_id)
                    aliases = canonical_local_ids[canonical_id]
                    material_records.append({
                        "kind": "source_vertex",
                        "canonical_coordinate_class_id": canonical_id,
                        "local_vertex_id": int(local_id),
                        "coordinate_alias_local_vertex_ids": [
                            int(value) for value in aliases],
                        "coordinate_alias_source_vertex_ids": [
                            int(source_vertex_ids[value]) for value in aliases],
                    })
                else:
                    first_canonical, second_canonical = (
                        int(node_key[1]), int(node_key[2]))
                    edge = tuple(sorted((first_canonical, second_canonical)))
                    fraction = edge_intersections[edge][2]
                    first_local = representative_local_id(edge[0])
                    second_local = representative_local_id(edge[1])
                    first_aliases = canonical_local_ids[edge[0]]
                    second_aliases = canonical_local_ids[edge[1]]
                    material_records.append({
                        "kind": "source_edge_intersection",
                        "canonical_endpoint_coordinate_class_ids": [
                            int(edge[0]), int(edge[1])],
                        "endpoint_local_vertex_ids": [
                            int(first_local), int(second_local)],
                        "endpoint_alias_local_vertex_ids": [
                            [int(value) for value in first_aliases],
                            [int(value) for value in second_aliases]],
                        "endpoint_alias_source_vertex_ids": [
                            [int(source_vertex_ids[value]) for value in first_aliases],
                            [int(source_vertex_ids[value]) for value in second_aliases]],
                        "interpolation_fraction_from_first_endpoint": fraction,
                    })
            metric["material_point_records"] = material_records
            metric["material_point_records_sha256"] = _canonical_record_sha256(
                material_records)
        loop_metrics.append(metric)
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


def _source_face_graph_distances(surface, collar_face_ids):
    """Build exact-coordinate-quotient face distances from a frozen source collar."""
    positions = _float32_xyz(surface.positions, name="source station topology")
    faces = _faces(surface.faces, name="source station topology",
                   vertex_count=len(positions))
    _require(isinstance(collar_face_ids, (list, tuple, np.ndarray))
             and len(collar_face_ids) > 0
             and all(type(value) is int and value >= 0 for value in collar_face_ids),
             "frozen source collar face IDs must be nonempty nonnegative integers")
    seeds = tuple(sorted(set(collar_face_ids)))
    _require(len(seeds) == len(collar_face_ids)
             and all(value < len(faces) for value in seeds),
             "frozen source collar faces are duplicated or out of range")

    coordinate_to_vertex = {}
    canonical = np.empty(len(positions), dtype="<i8")
    for vertex_id, point in enumerate(positions):
        coordinate = tuple(0.0 if float(value) == 0.0 else float(value)
                           for value in point)
        canonical_id = coordinate_to_vertex.get(coordinate)
        if canonical_id is None:
            canonical_id = len(coordinate_to_vertex)
            coordinate_to_vertex[coordinate] = canonical_id
        canonical[vertex_id] = canonical_id
    quotient_faces = canonical[faces]
    _require(not np.any(
        (quotient_faces[:, 0] == quotient_faces[:, 1])
        | (quotient_faces[:, 1] == quotient_faces[:, 2])
        | (quotient_faces[:, 2] == quotient_faces[:, 0])),
        "source station topology has a degenerate face after exact-coordinate quotient")

    edge_faces = {}
    for face_id, face in enumerate(quotient_faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_id)
    _require(all(len(incident) == 2 for incident in edge_faces.values()),
             "source station topology is not a closed two-face edge graph")
    adjacency = [set() for _ in range(len(faces))]
    for incident in edge_faces.values():
        first, second = incident
        adjacency[first].add(second)
        adjacency[second].add(first)

    component_ids = np.full(len(faces), -1, dtype="<i8")
    component_count = 0
    for start in range(len(faces)):
        if component_ids[start] >= 0:
            continue
        component_ids[start] = component_count
        pending = [start]
        while pending:
            current = pending.pop()
            for following in adjacency[current]:
                if component_ids[following] < 0:
                    component_ids[following] = component_count
                    pending.append(following)
        component_count += 1
    _require(component_count == 1,
             "source station face graph must have one connected source component")

    distances = np.full(len(faces), -1, dtype="<i8")
    pending = []
    for seed in seeds:
        distances[seed] = 0
        pending.append(seed)
    cursor = 0
    while cursor < len(pending):
        current = pending[cursor]
        cursor += 1
        next_distance = int(distances[current]) + 1
        for following in sorted(adjacency[current]):
            if distances[following] < 0:
                distances[following] = next_distance
                pending.append(following)

    graph_edges = sorted(
        (min(first, second), max(first, second))
        for first, neighbors in enumerate(adjacency) for second in neighbors
        if first < second)
    return distances, {
        "schema": "numi.human.exact-source-face-collar-distance.v1",
        "selection_basis": "minimum exact-source-face-graph distance from frozen collar faces",
        "source_typed_identity": list(surface.key),
        "source_face_count": len(faces),
        "source_face_u32_sha256": hashlib.sha256(
            np.ascontiguousarray(faces, dtype="<u4").tobytes()).hexdigest(),
        "exact_coordinate_quotient_vertex_count": len(coordinate_to_vertex),
        "exact_coordinate_duplicate_vertex_count": int(len(positions) - len(coordinate_to_vertex)),
        "signed_zero_canonicalization": "numeric zero maps to positive zero for topology only",
        "source_face_graph_edge_count": len(graph_edges),
        "source_face_graph_sha256": _canonical_record_sha256(
            [list(edge) for edge in graph_edges]),
        "source_face_graph_component_count": component_count,
        "frozen_collar_face_ids": list(seeds),
        "frozen_collar_face_ids_sha256": hashlib.sha256(
            np.asarray(seeds, dtype="<u8").tobytes()).hexdigest(),
    }


def _collar_local_source_section(
    section_surface, station, *, source_topology_surface, collar_face_ids, label,
    include_material_point_records=False,
):
    """Select one collar-local section loop; preserve all other loop diagnostics.

    Selection is based only on the independent source face graph and frozen
    collar IDs. Candidate coordinates can change section geometry, but cannot
    change the ancestry graph or the local-versus-remote selection rule.
    """
    source_faces = _faces(
        source_topology_surface.faces, name=f"{label} source topology",
        vertex_count=len(source_topology_surface.positions))
    measured_faces = _faces(
        section_surface.faces, name=f"{label} measured surface",
        vertex_count=len(section_surface.positions))
    _require(measured_faces.shape == source_faces.shape
             and measured_faces.tobytes() == source_faces.tobytes(),
             f"{label} ordered faces differ from independent source topology")
    raw = _source_surface_section(
        section_surface, station,
        include_material_point_records=include_material_point_records)
    _require(isinstance(raw.get("loop_metrics"), list)
             and bool(raw["loop_metrics"])
             and len(raw["loop_metrics"]) == raw.get("loop_count"),
             f"{label} section does not provide complete closed-loop diagnostics")
    distances, graph_record = _source_face_graph_distances(
        source_topology_surface, collar_face_ids)
    loop_distances = []
    for loop_index, metric in enumerate(raw["loop_metrics"]):
        ancestors = metric.get("source_face_ancestor_ids")
        _require(metric.get("passed") is True
                 and metric.get("lineage_available") is True
                 and metric.get("lineage_complete") is True
                 and metric.get("owner_ids") == [section_surface.key[1]]
                 and metric.get("generated_face_count") == 0
                 and metric.get("unmapped_face_count") == 0
                 and isinstance(ancestors, list) and bool(ancestors)
                 and all(type(value) is int and 0 <= value < len(distances)
                         for value in ancestors),
                 f"{label} loop {loop_index} has incomplete source-face ancestry")
        values = [int(distances[value]) for value in ancestors]
        _require(all(value >= 0 for value in values),
                 f"{label} loop {loop_index} is disconnected from source face graph")
        loop_distances.append({
            "loop_index": loop_index,
            "source_face_ancestor_count": len(ancestors),
            "source_face_ancestor_ids_sha256": metric.get(
                "source_face_ancestor_ids_sha256"),
            "minimum_source_face_graph_distance": min(values),
            "median_source_face_graph_distance": float(np.median(values)),
            "maximum_source_face_graph_distance": max(values),
        })
    minimum = min(item["minimum_source_face_graph_distance"]
                  for item in loop_distances)
    winners = [item["loop_index"] for item in loop_distances
               if item["minimum_source_face_graph_distance"] == minimum]
    _require(len(winners) == 1,
             f"{label} source collar does not select a unique nearest section loop")
    selected_index = winners[0]
    selected = raw["loop_metrics"][selected_index]
    selection = {
        **graph_record,
        "passed": True,
        "selected_local_loop_count": 1,
        "selected_local_loop_index": selected_index,
        "selected_local_loop_source_face_ancestor_ids_sha256": selected.get(
            "source_face_ancestor_ids_sha256"),
        "all_loop_source_distance_diagnostics": loop_distances,
        "remote_loops_retained_as_diagnostics": [
            item["loop_index"] for item in loop_distances
            if item["loop_index"] != selected_index],
        "candidate_geometry_used_for_selection": False,
    }
    return {
        **selected,
        "passed": True,
        "loop_count": raw["loop_count"],
        "selected_local_loop_count": 1,
        "selected_local_loop_index": selected_index,
        "loop_metrics": raw["loop_metrics"],
        "local_loop_selection": selection,
        "observed_section_reason": raw.get("reason"),
        "section_center_world_m": raw.get("section_center_world_m"),
        "section_tangent_world_unit": raw.get("section_tangent_world_unit"),
        "segment_count": raw.get("segment_count"),
        "section_node_count": raw.get("section_node_count"),
        "coordinate_dtype": raw.get("coordinate_dtype"),
        "coordinate_ulp_tolerance_m": raw.get("coordinate_ulp_tolerance_m"),
    }


def _validated_source_collar_faces_by_owner(source_declarations, owner_by_role):
    """Return collars from already-validated independent source declarations."""
    if set(source_declarations) < set(owner_by_role):
        return None
    result = {}
    for role, owner_id in owner_by_role.items():
        declaration = source_declarations.get(role)
        spec = (declaration.get("source_station_spec")
                if isinstance(declaration, Mapping) else None)
        collar_faces = spec.get("frozen_collar_face_ids") if isinstance(spec, Mapping) else None
        if (not isinstance(collar_faces, (list, tuple))
                or not collar_faces
                or not all(type(value) is int and value >= 0 for value in collar_faces)):
            return None
        result[owner_id] = list(collar_faces)
    return result


def _collar_local_union_section(
    raw_section, *, source_surfaces, collar_face_ids_by_owner,
    expected_owner_ids, label,
):
    """Select one source-collar-local loop per vessel; keep remote loops visible."""
    expected = tuple(sorted(set(map(int, expected_owner_ids))))
    _require(len(expected) == 2 and set(source_surfaces) == set(expected)
             and set(collar_face_ids_by_owner) == set(expected),
             f"{label} requires one source topology and frozen collar per vessel")
    loops = raw_section.get("loop_metrics")
    _require(isinstance(loops, list) and len(loops) == raw_section.get("loop_count")
             and len(loops) > 1,
             f"{label} must provide every multi-loop section diagnostic")
    source_graphs = {}
    for owner_id in expected:
        surface = source_surfaces[owner_id]
        _require(tuple(surface.key) == (51011, owner_id),
                 f"{label} source topology identity does not match its vessel owner")
        distances, graph_record = _source_face_graph_distances(
            surface, collar_face_ids_by_owner[owner_id])
        source_graphs[owner_id] = (distances, graph_record)
    local_by_owner = {owner_id: [] for owner_id in expected}
    diagnostics = []
    for loop_index, metric in enumerate(loops):
        owners = metric.get("owner_ids")
        ancestors = metric.get("source_face_ancestor_ids")
        _require(metric.get("passed") is True
                 and metric.get("lineage_available") is True
                 and metric.get("lineage_complete") is True
                 and isinstance(owners, list) and len(owners) == 1
                 and type(owners[0]) is int and owners[0] in local_by_owner
                 and isinstance(ancestors, list) and bool(ancestors)
                 and all(type(face_id) is int and face_id >= 0
                         for face_id in ancestors)
                 and metric.get("generated_face_count") == 0
                 and metric.get("unmapped_face_count") == 0,
                 f"{label} loop {loop_index} has mixed, generated, or incomplete source lineage")
        owner_id = owners[0]
        distances, graph_record = source_graphs[owner_id]
        _require(all(face_id < len(distances) for face_id in ancestors),
                 f"{label} loop {loop_index} ancestry is outside owner {owner_id}")
        values = [int(distances[face_id]) for face_id in ancestors]
        _require(all(value >= 0 for value in values),
                 f"{label} loop {loop_index} is disconnected from owner source graph")
        record = {
            "loop_index": loop_index,
            "owner_id": owner_id,
            "source_face_ancestor_count": len(ancestors),
            "source_face_ancestor_ids_sha256": metric.get(
                "source_face_ancestor_ids_sha256"),
            "minimum_source_face_graph_distance": min(values),
            "median_source_face_graph_distance": float(np.median(values)),
            "maximum_source_face_graph_distance": max(values),
            "source_face_graph_sha256": graph_record["source_face_graph_sha256"],
        }
        diagnostics.append(record)
        local_by_owner[owner_id].append(record)
    selected = []
    for owner_id in expected:
        candidates = local_by_owner[owner_id]
        _require(bool(candidates),
                 f"{label} has no source-lineaged loop for owner {owner_id}")
        nearest = min(item["minimum_source_face_graph_distance"]
                      for item in candidates)
        winners = [item for item in candidates
                   if item["minimum_source_face_graph_distance"] == nearest]
        _require(len(winners) == 1,
                 f"{label} source collar does not select a unique nearest loop for owner {owner_id}")
        selected.append(winners[0])
    selected_indices = [item["loop_index"] for item in selected]
    selected_metrics = [loops[index] for index in selected_indices]
    selection = {
        "schema": "numi.human.exact-source-collar-local-union-loops.v1",
        "passed": True,
        "selection_basis": "minimum exact source face graph distance to each owner's frozen collar",
        "expected_owner_ids": list(expected),
        "selected_loop_indices_by_owner": {
            str(item["owner_id"]): item["loop_index"] for item in selected},
        "selected_source_face_ancestor_ids_sha256_by_owner": {
            str(item["owner_id"]): item["source_face_ancestor_ids_sha256"]
            for item in selected},
        "all_loop_source_distance_diagnostics": diagnostics,
        "remote_loops_retained_as_diagnostics": [
            item for item in diagnostics if item["loop_index"] not in selected_indices],
        "source_face_graphs_by_owner": {
            str(owner_id): record for owner_id, (_distances, record)
            in source_graphs.items()},
        "candidate_geometry_used_for_selection": False,
    }
    return {
        **dict(raw_section),
        "passed": all(metric.get("passed") is True for metric in selected_metrics),
        "loop_count": len(selected_metrics),
        "loop_metrics": selected_metrics,
        "reason": "section must have exactly one closed loop",
        "raw_observed_loop_count": len(loops),
        "raw_loop_metrics": loops,
        "source_local_union_selection": selection,
    }


def _source_surface_section(surface, station, *, include_material_point_records=False):
    return _section_loops(
        surface.positions, surface.faces,
        station["section_center_world_m"], station["tangent_world_unit"],
        include_material_point_records=include_material_point_records,
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
    loop_metrics = section.get("loop_metrics", [])
    if section.get("selected_local_loop_count") == 1:
        selection = section.get("local_loop_selection")
        selected_index = section.get("selected_local_loop_index")
        if (not isinstance(selection, Mapping)
                or selection.get("passed") is not True
                or selection.get("candidate_geometry_used_for_selection") is not False
                or type(selected_index) is not int
                or not isinstance(loop_metrics, list)
                or selected_index < 0 or selected_index >= len(loop_metrics)
                or section.get("loop_count") != len(loop_metrics)):
            return None, f"{label} has an invalid frozen source-local loop selection"
        metric = loop_metrics[selected_index]
    elif (section.get("loop_count") == 1 and section.get("passed") is True
          and isinstance(loop_metrics, list) and len(loop_metrics) == 1):
        metric = loop_metrics[0]
    else:
        return None, f"{label} must have one positive closed section loop"
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
                               candidate_branch_sections, baseline_branch_sections,
                               solid_evidence):
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
    source_evidence = declaration.get("source_evidence")
    boundary_fallback_required = any(
        solid_evidence.get(key) is not True
        for key in ("candidate_port_audit_passed", "baseline_port_audit_passed"))
    if boundary_fallback_required:
        if not isinstance(source_evidence, Mapping):
            issues.append(
                "partial-overlap two-loop station needs independent source-baseline evidence")
        else:
            expected_link = solid_evidence.get("typed_vessel_link")
            if (source_evidence.get("schema")
                    != "numi.human.source-baseline-vessel-junction-station.v1"
                    or source_evidence.get("independent_source_baseline_only") is not True
                    or source_evidence.get("source_station_is_candidate_independent") is not True
                    or source_evidence.get("typed_link") != expected_link):
                issues.append(
                    "source station evidence is not independently bound to the audited typed link")
            if source_evidence.get("baseline_source_surface_bindings") != (
                    solid_evidence.get("baseline_source_surface_bindings")):
                issues.append(
                    "source station evidence is not bound to the current independent baseline surfaces")
            for source_key, evidence_key in (
                ("frozen_face_pairs_sha256", "baseline_frozen_face_pairs_sha256"),
                ("baseline_observed_face_pairs_sha256",
                 "baseline_observed_face_pairs_sha256"),
                ("baseline_exact_interface_points_sha256",
                 "baseline_exact_interface_points_sha256"),
            ):
                if source_evidence.get(source_key) != solid_evidence.get(evidence_key):
                    issues.append(
                        f"source station evidence {source_key} differs from current baseline")
            if source_evidence.get("baseline_exact_interface_point_count") != (
                    solid_evidence.get("baseline_port_exact_interface_point_count")):
                issues.append(
                    "source station evidence exact source-interface point count differs")
            if source_evidence.get("source_station_spec_sha256") != (
                    _canonical_record_sha256(declaration.get("source_station_spec", {}))):
                issues.append("source station specification hash does not match its declaration")
            if source_evidence.get("baseline_declared_partial_overlap_boundary") != (
                    solid_evidence.get("baseline_port_boundary")):
                issues.append(
                    "source station declaration differs from the recomputed baseline port proof")
            if source_evidence.get("baseline_exact_transverse_overlap_side_witnesses") != (
                    solid_evidence.get("baseline_exact_transverse_overlap_side_witnesses")):
                issues.append(
                    "source station exact transverse witnesses differ from current baseline")
            if source_evidence.get("baseline_port_audit_passed") is not (
                    solid_evidence.get("baseline_port_audit_passed")):
                issues.append("source station baseline P6 status differs from current source")
            if source_evidence.get("baseline_port_pair_passed") is not (
                    solid_evidence.get("baseline_port_pair_passed")):
                issues.append("source station baseline exact port status differs from current source")
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
        "source_evidence": dict(source_evidence) if isinstance(source_evidence, Mapping) else None,
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
        candidate_branch_sections, baseline_branch_sections, solid_evidence)
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
    for side in ("candidate", "baseline"):
        audit_key = f"{side}_port_audit_passed"
        boundary = solid_evidence.get(f"{side}_port_boundary")
        typed_link = solid_evidence.get("typed_vessel_link")
        conditions = boundary.get("conditions") if isinstance(boundary, Mapping) else None
        boundary_passed = bool(
            isinstance(boundary, Mapping)
            and boundary.get("passed") is True
            and boundary.get("classification")
                == "DECLARED_VESSEL_PORT_PARTIAL_OVERLAP_BOUNDARY_ONLY"
            and boundary.get("typed_port") == typed_link
            and boundary.get("requires_separate_solid_and_calibre_proofs") is True
            and type(boundary.get("observed_pair_count")) is int
            and boundary["observed_pair_count"] > 0
            and type(boundary.get("exact_interface_point_count")) is int
            and boundary["exact_interface_point_count"]
                == solid_evidence.get(f"{side}_port_exact_interface_point_count")
            and isinstance(conditions, Mapping)
            and bool(conditions)
            and all(value is True for value in conditions.values()))
        transverse_passed = _transverse_overlap_proof_passed(
            solid_evidence.get(
                f"{side}_exact_transverse_overlap_side_witnesses"),
            solid_evidence.get(
                f"{side}_observed_port_face_pairs", ()))
        if solid_evidence.get(audit_key) is not True and not (
                boundary_passed or transverse_passed):
            issues.append(
                f"{audit_key} must pass or have a complete source-scoped exact boundary or transverse proof")
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
    source_station_declarations=None,
    candidate_station_transport_policy=None,
    transverse_witness_provider=None,
    target_station_candidate_fractions=None,
    declared_vessel_ports=None,
):
    """Check one declared bridge port as closed solids and local section loops.

    declared_vessel_ports may carry one separately source-validated typed edge
    for a non-flow anatomical interface. It only scopes the partial-overlap
    boundary classification; the exact collar, solid, and calibre proofs below
    remain mandatory. The default registry is unchanged.

    This is a numerical geometry gate only. It does not establish measured
    physiology, native acceptance, or clinical calibre.
    """
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    if declared_vessel_ports is not None:
        _require(
            isinstance(declared_vessel_ports, (set, frozenset, tuple, list))
            and len(declared_vessel_ports) == 1,
            "a per-port declaration must contain exactly one typed edge")
        edge = next(iter(declared_vessel_ports))
        _require(isinstance(edge, (tuple, list)) and len(edge) == 2,
                 "a per-port declaration must be an edge of two surface keys")
        declared_edge = (_identity(edge[0]), _identity(edge[1]))
        _require(declared_edge == (bridge_key, target_key),
                 "a per-port declaration must exactly match the audited typed edge")
        declared_vessel_ports = (declared_edge,)
    if target_station_candidate_fractions is not None:
        _require((bridge_key, target_key) == ((51011, 526), (51011, 472)),
                 "target-collar station search is restricted to bridge 526 / target 472")
    if junction_station_declarations is None:
        junction_station_declarations = {}
    _require(isinstance(junction_station_declarations, Mapping),
             "junction station declarations must be a source-bound mapping")
    if source_station_declarations is None:
        source_station_declarations = {}
    _require(isinstance(source_station_declarations, Mapping)
             and set(source_station_declarations) <= {"bridge", "target"},
             "source station declarations must be keyed by branch role")
    _require(candidate_station_transport_policy in (
                 None, "SOURCE_LOCAL_PROPER_RIGID_FIT_V1",
                 "SOURCE_MATERIAL_SECTION_FRAME_V1"),
             "unknown candidate source-station transport policy")
    if candidate_station_transport_policy is not None:
        _require(set(source_station_declarations) == {"bridge", "target"},
                 "source-station transport needs both independent source station declarations")
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
        exact_points, allowed,
        declared_vessel_ports=declared_vessel_ports)
    candidate_transverse_witnesses = _make_transverse_overlap_witnesses(
        transverse_witness_provider, verifier, bridge, target, observed)
    candidate_transverse_passed = _transverse_overlap_proof_passed(
        candidate_transverse_witnesses, observed)
    port_pair_passed = bool(exact_points) and (
        bool(port_audit.get("passed")) or partial_overlap_boundary["passed"]
        or candidate_transverse_passed)
    baseline_port_audit = verifier.audit_surface_pair(
        base_bridge, base_target, intended_face_pairs=allowed)
    baseline_observed = tuple(tuple(map(int, pair))
                              for pair in baseline_port_audit.get(
                                  "declared_interface_triangle_pairs", []))
    baseline_exact_points = _exact_interface_points(
        verifier, base_bridge, base_target, baseline_observed)
    baseline_partial_overlap_boundary = _declared_partial_vessel_port_overlap(
        bridge_key, target_key, baseline_port_audit, baseline_observed,
        baseline_port_audit.get("unexpected_triangle_pairs", ()),
        baseline_exact_points, allowed,
        declared_vessel_ports=declared_vessel_ports)
    baseline_transverse_witnesses = _make_transverse_overlap_witnesses(
        transverse_witness_provider, verifier, base_bridge, base_target, baseline_observed)
    baseline_transverse_passed = _transverse_overlap_proof_passed(
        baseline_transverse_witnesses, baseline_observed)
    baseline_port_pair_passed = bool(baseline_exact_points) and (
        bool(baseline_port_audit.get("passed"))
        or baseline_partial_overlap_boundary["passed"]
        or baseline_transverse_passed)

    def surface_binding(surface):
        return _source_geometry_binding(surface)

    source_pair_record = [list(pair) for pair in pairs]
    baseline_observed_record = [list(pair) for pair in baseline_observed]
    source_baseline_bindings = {
        "bridge": surface_binding(base_bridge),
        "target": surface_binding(base_target),
    }
    source_baseline_evidence_fields = {
        "typed_vessel_link": [list(bridge_key), list(target_key)],
        "baseline_source_surface_bindings": source_baseline_bindings,
        "baseline_frozen_face_pairs_sha256": _canonical_record_sha256(source_pair_record),
        "baseline_observed_face_pairs_sha256": _canonical_record_sha256(
            baseline_observed_record),
        "baseline_exact_interface_points_sha256": _source_interface_points_sha256(
            baseline_exact_points),
        "baseline_port_exact_interface_point_count": len(baseline_exact_points),
        "baseline_port_audit_passed": bool(baseline_port_audit.get("passed")),
        "baseline_port_pair_passed": baseline_port_pair_passed,
        "baseline_port_pair_witness_kind": (
            "central_p6" if baseline_port_audit.get("passed") is True else
            "coarse_exact_boundary" if baseline_partial_overlap_boundary.get("passed") is True else
            "exact_transverse_inside_outside"),
        "baseline_exact_transverse_overlap_side_witnesses": baseline_transverse_witnesses,
        "baseline_port_boundary": baseline_partial_overlap_boundary,
    }
    runtime_source_station_declarations = {}
    for role, surface, base_surface, patch_faces in (
        ("bridge", bridge, base_bridge, bridge_patch),
        ("target", target, base_target, target_patch),
    ):
        declaration = source_station_declarations.get(role)
        if declaration is None:
            continue
        _require(isinstance(declaration, Mapping)
                 and declaration.get("schema")
                     == "numi.human.source-baseline-junction-station-declaration.v1",
                 f"{role} source station declaration has an unknown schema")
        source_evidence = declaration.get("source_evidence")
        source_spec = declaration.get("source_station_spec")
        _require(isinstance(source_evidence, Mapping)
                 and source_evidence.get("independent_source_baseline_only") is True
                 and source_evidence.get("source_station_is_candidate_independent") is True,
                 f"{role} source station is not independent baseline evidence")
        _require(declaration.get("typed_link")
                 == [list(bridge_key), list(target_key)]
                 and declaration.get("station_role") == role
                 and declaration.get("anchor_owner_id")
                     == (bridge_key[1] if role == "bridge" else target_key[1]),
                 f"{role} source station declaration has the wrong typed owner")
        _require(source_evidence.get("typed_link")
                 == source_baseline_evidence_fields["typed_vessel_link"],
                 f"{role} source station typed link differs from this audit")
        for key in ("baseline_source_surface_bindings",
                    "frozen_face_pairs_sha256",
                    "baseline_observed_face_pairs_sha256",
                    "baseline_exact_interface_points_sha256",
                    "baseline_exact_interface_point_count",
                    "baseline_port_audit_passed",
                    "baseline_port_pair_passed",
                    "baseline_port_pair_witness_kind",
                    "baseline_exact_transverse_overlap_side_witnesses",
                    "baseline_declared_partial_overlap_boundary"):
            evidence_key = {
                "frozen_face_pairs_sha256": "baseline_frozen_face_pairs_sha256",
                "baseline_exact_interface_point_count":
                    "baseline_port_exact_interface_point_count",
                "baseline_declared_partial_overlap_boundary": "baseline_port_boundary",
            }.get(key, key)
            _require(
                source_evidence.get(key) == source_baseline_evidence_fields.get(evidence_key),
                f"{role} source station {key} differs from current independent baseline")
        _require(isinstance(source_spec, Mapping)
                 and source_evidence.get("source_station_spec_sha256")
                     == _canonical_record_sha256(source_spec)
                 and source_evidence.get("station_selection_policy")
                     == source_spec.get("station_selection_policy")
                 and source_spec.get("station_selection_policy") in (
                     "GLOBAL_SOURCE_PCA_V1", "COLLAR_LOCAL_PCA_V1"),
                 f"{role} source station specification hash or policy does not match")
        expected_declaration_id = (
            "source-baseline-junction-station-v1:"
            + _canonical_record_sha256({
                "evidence": source_evidence, "station": source_spec}))
        _require(declaration.get("declaration_id") == expected_declaration_id,
                 f"{role} source station declaration ID does not match its source evidence")
        expected_station_ancestry = {
            key: source_spec.get(key) for key in (
                "material_anchor_source_vertex_ids",
                "tangent_support_source_vertex_ids",
                "frozen_collar_face_ids",
            )
        }
        _require(declaration.get("source_ancestry") == expected_station_ancestry
                 and source_evidence.get("anchor_role") == role,
                 f"{role} source station ancestry or role differs")
        _require(
            source_evidence.get("baseline_source_surface_bindings")
            == source_baseline_bindings,
            f"{role} source station baseline surfaces differ")
        _explicit_material_vertex_ancestry(base_surface, source_spec, patch_faces)
        if source_evidence.get("station_selection_policy") == "COLLAR_LOCAL_PCA_V1":
            declared_section = source_evidence.get("baseline_source_station_section")
            actual_section = _collar_local_source_section(
                base_surface, source_spec, source_topology_surface=base_surface,
                collar_face_ids=source_spec.get("frozen_collar_face_ids", []),
                label=f"{role} independent baseline local station")
            _require(isinstance(declared_section, Mapping)
                     and declared_section.get("passed") is True
                     and declared_section.get("loop_count") == actual_section.get("loop_count")
                     and declared_section.get("selected_local_loop_count") == 1
                     and declared_section.get("selected_local_loop_index")
                         == actual_section.get("selected_local_loop_index")
                     and declared_section.get("local_loop_selection")
                         == actual_section.get("local_loop_selection")
                     and declared_section.get("area_m2") == actual_section.get("area_m2")
                     and declared_section.get("minimum_feret_width_m")
                         == actual_section.get("minimum_feret_width_m")
                     and declared_section.get("source_face_ancestor_ids")
                         == actual_section.get("source_face_ancestor_ids"),
                     f"{role} source-local section is not reproduced from the current independent baseline")
        baseline_spec = _transport_material_station(base_surface, source_spec)
        candidate_station_transport_proof = None
        if candidate_station_transport_policy == "SOURCE_LOCAL_PROPER_RIGID_FIT_V1":
            candidate_ancestry_station = _transport_material_station(
                surface, source_spec)
            candidate_spec, candidate_station_transport_proof = (
                _transport_material_station_by_local_proper_rigid_fit(
                    base_surface, surface, baseline_spec))
            candidate_station_transport_proof[
                "candidate_ancestry_station_center_world_m"] = (
                    candidate_ancestry_station["section_center_world_m"])
            candidate_station_transport_proof[
                "candidate_ancestry_station_tangent_world_unit"] = (
                    candidate_ancestry_station["tangent_world_unit"])
        elif candidate_station_transport_policy == "SOURCE_MATERIAL_SECTION_FRAME_V1":
            candidate_spec, candidate_station_transport_proof = (
                _transport_material_station_by_material_section(
                    base_surface, surface, baseline_spec))
            _require(candidate_station_transport_proof.get("passed") is True,
                     f"{role} same-material section calibre/planarity proof failed")
        else:
            candidate_spec = _transport_material_station(surface, source_spec)
        runtime_source_station_declarations[role] = {
            "declaration_id": declaration.get("declaration_id"),
            "station_role": role,
            "expected_owner_ids": sorted((bridge_key[1], target_key[1])),
            "bridge_owner_id": bridge_key[1],
            "target_owner_id": target_key[1],
            "anchor_owner_id": bridge_key[1] if role == "bridge" else target_key[1],
            "source_ancestry": expected_station_ancestry,
            "source_station_spec": dict(source_spec),
            "candidate_station": candidate_spec,
            "baseline_station": baseline_spec,
            "candidate_station_transport_proof": candidate_station_transport_proof,
            "source_evidence": dict(source_evidence),
        }

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

    station_specs_by_role = {}
    for role, surface, base_surface, supplied_station, collar_faces, route_anchor in (
        ("bridge", bridge, base_bridge, bridge_station, bridge_patch,
         bridge_route_anchor_world_m),
        ("target", target, base_target, target_station, target_patch,
         target_route_anchor_world_m),
    ):
        source_declaration = runtime_source_station_declarations.get(role)
        if source_declaration is not None:
            candidate_spec = source_declaration["candidate_station"]
            baseline_spec = source_declaration["baseline_station"]
        else:
            candidate_spec = _validated_station_spec(
                surface, supplied_station, witness_points=exact_points,
                collar_face_ids=collar_faces, route_anchor=route_anchor)
            baseline_spec = _transport_material_station(base_surface, candidate_spec)
        station_specs_by_role[role] = {
            "candidate": candidate_spec,
            "baseline": baseline_spec,
            "source_declaration": source_declaration,
        }
    bridge_station = station_specs_by_role["bridge"]["candidate"]
    base_bridge_station = station_specs_by_role["bridge"]["baseline"]
    target_station = station_specs_by_role["target"]["candidate"]
    base_target_station = station_specs_by_role["target"]["baseline"]
    station_results = {}
    for role, surface, base_surface in (
        ("bridge", bridge, base_bridge),
        ("target", target, base_target),
    ):
        spec = station_specs_by_role[role]["candidate"]
        baseline_spec = station_specs_by_role[role]["baseline"]
        source_declaration = station_specs_by_role[role]["source_declaration"]
        if (isinstance(source_declaration, Mapping)
                and source_declaration.get("source_evidence", {}).get(
                    "station_selection_policy") == "COLLAR_LOCAL_PCA_V1"):
            collar_faces = spec.get("frozen_collar_face_ids", [])
            candidate_section = _collar_local_source_section(
                surface, spec, source_topology_surface=base_surface,
                collar_face_ids=collar_faces,
                label=f"{role} candidate local station")
            baseline_section = _collar_local_source_section(
                base_surface, baseline_spec, source_topology_surface=base_surface,
                collar_face_ids=collar_faces,
                label=f"{role} baseline local station")
        else:
            candidate_section = _source_surface_section(surface, spec)
            baseline_section = _source_surface_section(base_surface, baseline_spec)
        station_results[role] = {
            "station": spec,
            "baseline_station_transported_by_ancestry": baseline_spec,
            "source_station_declaration": (
                station_specs_by_role[role]["source_declaration"]),
            "candidate_station_transport_proof": (
                station_specs_by_role[role]["source_declaration"].get(
                    "candidate_station_transport_proof")
                if station_specs_by_role[role]["source_declaration"] is not None
                else None),
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
        base_observed = baseline_observed
        base_exact_points = baseline_exact_points
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
            "typed_vessel_link": [list(bridge_key), list(target_key)],
            "candidate_port_audit_passed": bool(port_audit.get("passed")),
            "baseline_port_audit_passed": bool(baseline_port_audit.get("passed")),
            "baseline_port_pair_passed": baseline_port_pair_passed,
            "candidate_port_boundary": partial_overlap_boundary,
            "baseline_port_boundary": baseline_partial_overlap_boundary,
            "candidate_observed_port_face_pairs": [list(pair) for pair in observed],
            "candidate_exact_transverse_overlap_side_witnesses":
                candidate_transverse_witnesses,
            "baseline_exact_transverse_overlap_side_witnesses":
                baseline_transverse_witnesses,
            "candidate_port_exact_interface_point_count": len(exact_points),
            "baseline_port_exact_interface_point_count": len(base_exact_points),
            "baseline_source_surface_bindings": source_baseline_bindings,
            "baseline_frozen_face_pairs_sha256": _canonical_record_sha256(
                source_pair_record),
            "baseline_observed_face_pairs_sha256": _canonical_record_sha256(
                baseline_observed_record),
            "baseline_exact_interface_points_sha256": _source_interface_points_sha256(
                base_exact_points),
            "baseline_observed_port_face_pairs": [list(pair) for pair in base_observed],
            "baseline_exact_interface_points_world_m": base_exact_points,
        }
        return junction_evidence_cache

    def compare_union_station(label, station_role, candidate_section, baseline_section,
                              candidate_station, baseline_station):
        source_declaration = runtime_source_station_declarations.get(station_role)
        collar_faces = _validated_source_collar_faces_by_owner(
            runtime_source_station_declarations,
            {"bridge": bridge_key[1], "target": target_key[1]})
        use_local_union = collar_faces is not None
        candidate_union_view = candidate_section
        baseline_union_view = baseline_section
        if use_local_union:
            source_surfaces = {
                bridge_key[1]: base_bridge, target_key[1]: base_target}
            if candidate_section.get("loop_count", 0) > 1:
                candidate_union_view = _collar_local_union_section(
                    candidate_section, source_surfaces=source_surfaces,
                    collar_face_ids_by_owner=collar_faces,
                    expected_owner_ids=(bridge_key[1], target_key[1]),
                    label=f"{label} candidate union")
            if baseline_section.get("loop_count", 0) > 1:
                baseline_union_view = _collar_local_union_section(
                    baseline_section, source_surfaces=source_surfaces,
                    collar_face_ids_by_owner=collar_faces,
                    expected_owner_ids=(bridge_key[1], target_key[1]),
                    label=f"{label} baseline union")

        if (candidate_union_view.get("loop_count") == 1
                and baseline_union_view.get("loop_count") == 1
                and candidate_union_view.get("passed") is True
                and baseline_union_view.get("passed") is True):
            comparison = _section_comparison(
                candidate_union_view, baseline_union_view)
            _require(comparison["passed"],
                     f"{label} calibre ratio is outside the declared local screen")
            return {
                **comparison,
                "candidate_source_local_union_selection":
                    candidate_union_view.get("source_local_union_selection"),
                "baseline_source_local_union_selection":
                    baseline_union_view.get("source_local_union_selection"),
            }
        if (candidate_union_view.get("loop_count") != 2
                or baseline_union_view.get("loop_count") != 2):
            _require(False,
                     f"{label} requires matching single loops or the explicit two-vessel junction proof")

        candidate_branch_sections = {}
        baseline_branch_sections = {}
        for role, candidate_surface, baseline_surface, owner_id in (
            ("bridge", bridge, base_bridge, bridge_key[1]),
            ("target", target, base_target, target_key[1]),
        ):
            candidate_raw = _source_surface_section(candidate_surface, candidate_station)
            baseline_raw = _source_surface_section(baseline_surface, baseline_station)
            declaration = runtime_source_station_declarations.get(role)
            spec = declaration.get("source_station_spec") if isinstance(
                declaration, Mapping) else None
            if (use_local_union and isinstance(spec, Mapping)
                    and candidate_raw.get("loop_count", 0) > 1):
                candidate_raw = _collar_local_source_section(
                    candidate_surface, candidate_station,
                    source_topology_surface=baseline_surface,
                    collar_face_ids=spec["frozen_collar_face_ids"],
                    label=f"{label} candidate branch {role}")
            if (use_local_union and isinstance(spec, Mapping)
                    and baseline_raw.get("loop_count", 0) > 1):
                baseline_raw = _collar_local_source_section(
                    baseline_surface, baseline_station,
                    source_topology_surface=baseline_surface,
                    collar_face_ids=spec["frozen_collar_face_ids"],
                    label=f"{label} baseline branch {role}")
            candidate_branch_sections[owner_id] = candidate_raw
            baseline_branch_sections[owner_id] = baseline_raw

        if source_declaration is not None:
            station_declaration = source_declaration
        elif isinstance(junction_station_declarations.get(station_role), Mapping):
            station_declaration = {
                **dict(junction_station_declarations[station_role]),
                "candidate_station": candidate_station,
                "baseline_station": baseline_station,
            }
        else:
            station_declaration = None
        proof = _junction_aware_two_loop_section_proof(
            expected_owner_ids=(bridge_key[1], target_key[1]),
            candidate_union_section=candidate_union_view,
            baseline_union_section=baseline_union_view,
            candidate_branch_sections=candidate_branch_sections,
            baseline_branch_sections=baseline_branch_sections,
            solid_evidence=junction_solid_evidence(),
            station_declaration=station_declaration,
        )
        _require(proof["passed"],
                 f"{label} junction-aware section proof rejected: "
                 + "; ".join(proof["rejections"][:5]))
        return {
            "passed": True,
            "method": "explicit one-loop-per-vessel source-collar-local junction proof",
            "junction_aware_proof": proof,
            "candidate_source_local_union_selection":
                candidate_union_view.get("source_local_union_selection"),
            "baseline_source_local_union_selection":
                baseline_union_view.get("source_local_union_selection"),
            "candidate_branch_sections": candidate_branch_sections,
            "baseline_branch_sections": baseline_branch_sections,
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
