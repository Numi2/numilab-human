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
from pathlib import Path

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




CURRENT_SOURCE_INTERFACE_REPORT_REFERENCE_SCHEMA = (
    "numi.human.current-source-interface-report-reference.v1")
_CURRENT_SOURCE_INTERFACE_REPORT_SCHEMA = (
    "numi.human.current-aortic-6-7-collar-proof.v1")


def _certificate_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_capture_bundle_sha256(bundle):
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_current_source_interface_report_reference(
    reference, surfaces, network_keys, *, current_capture_input_bundle,
    source_inventory,
):
    """Replay a caller-pinned current-capture proof report as a narrow handoff.

    The reference names immutable report, preflight, source, and code inputs.
    Its evidence is checked against the current explicit capture bundle, the
    live candidate arrays, and the ABI5 row identity before it is normalized
    into the existing source-interface handoff contract.
    """
    _require(isinstance(reference, Mapping)
             and reference.get("schema")
             == CURRENT_SOURCE_INTERFACE_REPORT_REFERENCE_SCHEMA,
             "current source-interface report reference schema is invalid")
    required = {
        "schema", "typed_link", "report_path", "report_sha256",
        "preflight_path", "preflight_sha256", "source_input_pins",
        "proof_code_pins", "runtime_pin",
    }
    _require(set(reference) == required,
             "current source-interface report reference has missing/extra fields")
    raw_edge = reference.get("typed_link")
    _require(isinstance(raw_edge, (tuple, list)) and len(raw_edge) == 2,
             "current source-interface report reference lacks a typed link")
    edge = (_identity(raw_edge[0]), _identity(raw_edge[1]))
    _require(edge == AORTIC_ARCH_SOURCE_INTERFACE
             and edge[0] in network_keys and edge[1] not in network_keys,
             "current source-interface report must bind the exact aortic 7->6 pair")
    _require(isinstance(current_capture_input_bundle, Mapping)
             and isinstance(source_inventory, Mapping),
             "current source-interface report requires live capture and source inventories")

    def resolved_pin(path_value, digest, label):
        _require(isinstance(path_value, str) and Path(path_value).is_absolute()
                 and isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 f"{label} pin is malformed")
        path = Path(path_value).resolve(strict=True)
        _require(path.is_file() and _certificate_file_sha256(path) == digest,
                 f"{label} pin digest differs from its file")
        return path, digest

    report_path, report_sha = resolved_pin(
        reference["report_path"], reference["report_sha256"], "proof report")
    preflight_path, preflight_sha = resolved_pin(
        reference["preflight_path"], reference["preflight_sha256"],
        "proof preflight")
    report = json.loads(report_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    _require(report.get("schema") == _CURRENT_SOURCE_INTERFACE_REPORT_SCHEMA
             and report.get("status", "").startswith("PASS_SOURCE_BOUND_CURRENT_")
             and report.get("passed") is True
             and report.get("stage") == "complete"
             and report.get("typed_interface") == [list(edge[0]), list(edge[1])]
             and report.get("preflight_sha256") == preflight_sha,
             "current source-interface report identity/status/preflight binding is invalid")
    _require(preflight.get("passed") is True
             and preflight.get("anatomical_admission") is False
             and preflight.get("native_or_gpu_work_performed") is False
             and preflight.get("geometry_predicates_run") is False,
             "current source-interface preflight is not a source-only bound input")

    bundle = dict(current_capture_input_bundle)
    bundle_required = {
        "capture_pack_path", "capture_pack_sha256", "capture_receipt_path",
        "capture_receipt_sha256", "anatomy_receipt_path",
        "anatomy_receipt_sha256", "anatomy_payload_sha256", "payload_path",
    }
    _require(set(bundle) == bundle_required,
             "live explicit capture bundle has missing/extra fields")
    capture_pins = {}
    for path_key, sha_key in (
        ("capture_pack_path", "capture_pack_sha256"),
        ("capture_receipt_path", "capture_receipt_sha256"),
        ("anatomy_receipt_path", "anatomy_receipt_sha256"),
        ("payload_path", "anatomy_payload_sha256"),
    ):
        path, digest = resolved_pin(bundle[path_key], bundle[sha_key], path_key)
        capture_pins[str(path)] = digest
    source_pins = reference["source_input_pins"]
    _require(isinstance(source_pins, Mapping) and bool(source_pins),
             "current source-interface report needs explicit source provenance pins")
    normalized_source_pins = {}
    for raw_path, digest in source_pins.items():
        path, actual = resolved_pin(raw_path, digest, "source provenance")
        _require(str(path) not in normalized_source_pins,
                 "source provenance pins repeat a resolved path")
        normalized_source_pins[str(path)] = actual
    expected_data_pins = {**capture_pins, **normalized_source_pins}
    def normalized_report_pins(raw, label):
        _require(isinstance(raw, Mapping) and bool(raw),
                 f"{label} must be a nonempty path-to-digest mapping")
        result = {}
        for raw_path, digest in raw.items():
            path, actual = resolved_pin(raw_path, digest, label)
            _require(str(path) not in result,
                     f"{label} repeat a resolved path")
            result[str(path)] = actual
        return result

    report_data_pins = normalized_report_pins(
        report.get("input_pins"), "proof report data input")
    preflight_data_pins = normalized_report_pins(
        preflight.get("capture_pins"), "proof preflight data input")
    _require(report_data_pins == expected_data_pins
             and preflight_data_pins == expected_data_pins,
             "proof report data inputs differ from explicit current/source pins")

    code_pins = reference["proof_code_pins"]
    _require(isinstance(code_pins, Mapping) and bool(code_pins),
             "current source-interface report needs explicit executable code pins")
    normalized_code_pins = {}
    for raw_path, digest in code_pins.items():
        path, actual = resolved_pin(raw_path, digest, "proof code")
        _require(str(path) not in normalized_code_pins,
                 "proof code pins repeat a resolved path")
        normalized_code_pins[str(path)] = actual
    report_code_pins = normalized_report_pins(
        report.get("code_pins"), "proof report executable dependency")
    preflight_code_pins = normalized_report_pins(
        preflight.get("code_pins"), "proof preflight executable dependency")
    _require(report_code_pins == normalized_code_pins
             and preflight_code_pins == normalized_code_pins,
             "proof report executable dependencies differ from explicit code pins")
    runtime_pin = reference["runtime_pin"]
    runtime_report = report.get("manifold3d")
    _require(isinstance(runtime_pin, Mapping)
             and set(runtime_pin) == {"module_path", "sha256", "version"}
             and isinstance(runtime_report, Mapping)
             and runtime_report == dict(runtime_pin),
             "proof runtime dependency identity differs from its explicit pin")
    resolved_pin(runtime_pin["module_path"], runtime_pin["sha256"],
                 "proof runtime module")

    source_ancestry = report.get("source_collar_ancestry")
    _require(isinstance(source_ancestry, Mapping)
             and source_ancestry.get("typed_link")
             == [list(edge[0]), list(edge[1])]
             and source_ancestry.get("current_face_hashes_match") is True
             and source_ancestry.get("old_candidate_geometry_reused") is False,
             "current source-collar proof lacks live ordered-face ancestry")
    rowbinding_path = source_ancestry.get("rowbinding_path")
    _require(isinstance(rowbinding_path, str) and Path(rowbinding_path).is_absolute(),
             "current source-collar proof lacks its bound rowbinding path")
    rowbinding_path = str(Path(rowbinding_path).resolve(strict=True))
    _require(normalized_source_pins.get(rowbinding_path)
             == source_ancestry.get("rowbinding_sha256"),
             "current rowbinding path/digest differs from its explicit source pin")
    handoff_digest = source_ancestry.get("handoff_sha256")
    handoff_paths = [
        path for path, digest in normalized_source_pins.items()
        if digest == handoff_digest and path != rowbinding_path
    ]
    _require(isinstance(handoff_digest, str) and len(handoff_digest) == 64
             and len(handoff_paths) == 1
             and set(normalized_source_pins) == {rowbinding_path, handoff_paths[0]},
             "current handoff/rowbinding source pins are incomplete or ambiguous")
    raw_pairs = source_ancestry.get("face_pairs")
    _require(isinstance(raw_pairs, list) and bool(raw_pairs)
             and all(isinstance(pair, list) and len(pair) == 2
                     and all(type(face) is int and face >= 0 for face in pair)
                     for pair in raw_pairs),
             "current source-collar pair ancestry is malformed")
    pairs = tuple((pair[0], pair[1]) for pair in raw_pairs)
    pair_sha = hashlib.sha256(
        np.asarray(pairs, dtype="<u4").tobytes()).hexdigest()
    _require(pairs == tuple(sorted(set(pairs)))
             and len(pairs) == source_ancestry.get("exact_pair_count")
             and pair_sha == source_ancestry.get("pairset_sha256")
             and report.get("candidate_pairset") == [list(pair) for pair in pairs]
             and report.get("baseline_pairset") == [list(pair) for pair in pairs]
             and report.get("candidate_baseline_pairsets_equal") is True
             and report.get("candidate_unexpected_pairs") == []
             and report.get("baseline_unexpected_pairs") == [],
             "candidate, baseline, and source pairsets are not the same exact set")

    row_inventory = report.get("row_inventory")
    topology = report.get("source_topology")
    _require(isinstance(row_inventory, Mapping) and isinstance(topology, Mapping)
             and set(row_inventory) == {"6", "7"}
             and report.get("row_inventory") == preflight.get("row_inventory"),
             "current source-interface report row inventory is incomplete")
    fingerprints = {}
    source_rows = {}
    gid_details = source_inventory.get("native_source_slot_identity")
    _require(isinstance(gid_details, Mapping)
             and type(gid_details.get("native_nha_gid_base_offset")) is int
             and gid_details["native_nha_gid_base_offset"] >= 0,
             "current capture lacks the native NHA GID base identity")
    base_offset = gid_details["native_nha_gid_base_offset"]
    for key in edge:
        static = AORTIC_ARCH_SOURCE_ROWS[key]
        row = row_inventory[str(key[1])]
        expected_identity = {
            "typed_identity": list(key), "source_member": static["source_member"],
            "fma": static["fma"], "name": static["name"],
            "body": static["body_index"], "layer": static["layer"],
            "first_vertex": static["first_vertex"],
            "source_faces_sha256": static["faces_sha256"],
        }
        _require(all(row.get(field) == expected for field, expected
                     in expected_identity.items())
                 and row.get("gid_offset") == base_offset,
                 f"proof row identity or ABI5 source ancestry differs for {key}")
        observed = _aortic_arch_surface_fingerprint(surfaces.get(key), key=key)
        expected_candidate = {
            "candidate_positions_sha256": observed["positions_f32_sha256"],
            "candidate_faces_sha256": observed["faces_u32_sha256"],
            "candidate_normals_sha256": observed["normals_f32_sha256"],
            "candidate_gids_sha256": observed["source_vertex_ids_u64_sha256"],
        }
        _require(all(row.get(field) == expected for field, expected
                     in expected_candidate.items()),
                 f"proof report is stale for current candidate surface {key}")
        candidate_value = surfaces[key]
        gids_value = (candidate_value.source_vertex_ids
                      if hasattr(candidate_value, "source_vertex_ids")
                      else candidate_value[3])
        gids = np.ascontiguousarray(gids_value, dtype="<u8")
        expected_gids = (np.uint64(base_offset + static["first_vertex"])
                         + np.arange(static["vertex_count"], dtype="<u8"))
        _require(gids.shape == expected_gids.shape
                 and np.array_equal(gids, expected_gids),
                 f"current capture GIDs do not bind native source slots for {key}")
        inventory_key = f"{key[0]}:{key[1]}"
        candidate_inventory = source_inventory.get("candidate_surfaces", {}).get(
            inventory_key)
        if candidate_inventory is not None:
            _require(candidate_inventory.get("typed_identity") == list(key)
                     and candidate_inventory.get("positions_f32_sha256")
                     == observed["positions_f32_sha256"]
                     and candidate_inventory.get("faces_u32_sha256")
                     == observed["faces_u32_sha256"]
                     and candidate_inventory.get(
                         "captured_global_vertex_ids_u64_sha256")
                     == observed["source_vertex_ids_u64_sha256"],
                     f"live source inventory differs from proof candidate {key}")
        source_face_inventory = source_inventory.get(
            "source_face_ancestry_by_typed_row", {}).get(inventory_key)
        if source_face_inventory is not None:
            _require(source_face_inventory.get("typed_identity") == list(key)
                     and source_face_inventory.get("status")
                     == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
                     and source_face_inventory.get("capture_faces_u32_sha256")
                     == observed["faces_u32_sha256"]
                     and source_face_inventory.get("source_faces_u32_sha256")
                     == static["faces_sha256"],
                     f"live face ancestry differs from proof source row {key}")
        face_map = row.get("face_corner_ancestry")
        _require(isinstance(face_map, Mapping)
                 and face_map.get("status")
                 == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
                 and face_map.get("source_face_count") == static["face_count"]
                 and face_map.get("capture_face_count") == static["face_count"]
                 and face_map.get("source_faces_u32_sha256") == static["faces_sha256"]
                 and face_map.get("capture_faces_u32_sha256") == observed["faces_u32_sha256"],
                 f"proof face-corner map is not exact for {key}")
        for role in ("candidate", "independent_source_rigid_baseline"):
            item = topology.get(f"{role}:{key[1]}")
            _require(isinstance(item, Mapping)
                     and item.get("identity") == list(key)
                     and item.get("closed_self_free_single_component") is True
                     and item.get("exact_self_audit_count") == 0
                     and item.get("referenced_face_component_count") == 1,
                     f"{role} source topology failed for {key}")
            prepared = item.get("prepared_surface")
            quotient = (prepared.get("exact_coordinate_quotient_topology")
                        if isinstance(prepared, Mapping) else None)
            _require(isinstance(prepared, Mapping)
                     and isinstance(quotient, Mapping)
                     and quotient.get("closed_oriented_manifold_candidate") is True
                     and quotient.get("boundary_edge_count") == 0
                     and quotient.get("nonmanifold_edges_count") == 0
                     and quotient.get("orientation_defect_edges_count") == 0
                     and quotient.get("face_component_count") == 1,
                     f"{role} exact-coordinate topology is not closed for {key}")
            if role == "candidate":
                _require(prepared.get("positions_sha256")
                         == observed["positions_f32_sha256"]
                         and prepared.get("faces_sha256") == observed["faces_u32_sha256"]
                         and item.get("source_vertex_ids_sha256")
                         == observed["source_vertex_ids_u64_sha256"],
                         f"candidate topology bytes differ for {key}")
            else:
                _require(prepared.get("positions_sha256")
                         == row.get("baseline_positions_sha256")
                         and prepared.get("faces_sha256") == observed["faces_u32_sha256"]
                         and item.get("source_vertex_ids_sha256")
                         == observed["source_vertex_ids_u64_sha256"],
                         f"independent source baseline bytes differ for {key}")
        row_fingerprint = {
            **static,
            "vertex_count": observed["vertex_count"],
            "face_count": observed["face_count"],
        }
        source_rows[key] = row_fingerprint
        fingerprints[f"{key[0]}:{key[1]}"] = observed

    interface_gates = report.get("interface_gates")
    _require(report.get("source_topology_passed") is True
             and report.get("interface_gates_passed") is True
             and isinstance(interface_gates, Mapping)
             and set(interface_gates) == {
                 "candidate", "independent_source_rigid_baseline"},
             "source topology/interface gates did not pass in both roles")
    for role, gate in interface_gates.items():
        audit = gate.get("audit_result") if isinstance(gate, Mapping) else None
        scope = gate.get("documented_vessel_junction_scope")
        _require(gate.get("passed") is True
                 and gate.get("typed_port") == [list(edge[0]), list(edge[1])]
                 and gate.get("observed_face_pairs") == [list(pair) for pair in pairs]
                 and gate.get("unexpected_face_pairs") == []
                 and gate.get("observed_pairs_within_frozen_face_patch_cross_product") is True
                 and gate.get("exact_interface_point_count", 0) > 0
                 and isinstance(scope, Mapping) and scope.get("passed") is True
                 and scope.get("scope_kind") == "declared_source_network_link"
                 and scope.get("typed_port") == [list(edge[0]), list(edge[1])]
                 and scope.get("allowed_typed_port") == [list(edge[0]), list(edge[1])]
                 and scope.get("frozen_matched_face_patch_required") is True
                 and scope.get("downstream_closed_solid_confirmation_required") is True
                 and isinstance(audit, Mapping)
                 and audit.get("unexpected_triangle_pairs") == []
                 and audit.get("rejected_interface_feature_contacts") == []
                 and audit.get("rejected_source_patch_complex_contacts") == [],
                 f"{role} exact source-interface gate failed")

    solid_gates = report.get("solid_gates")
    _require(report.get("solid_gates_passed") is True
             and isinstance(solid_gates, Mapping)
             and set(solid_gates) == {
                 "candidate", "independent_source_rigid_baseline"},
             "closed-solid gates did not pass in both roles")
    for role, solid in solid_gates.items():
        witness = solid.get("exact_inside_both_witness") if isinstance(solid, Mapping) else None
        lineage = (solid.get("union_float64_mesh64_lineage")
                   if isinstance(solid, Mapping) else None)
        volume_accounting = (solid.get("volume_accounting")
                             if isinstance(solid, Mapping) else None)
        _require(solid.get("passed") is True
                 and solid.get("manifold_inputs_passed") is True
                 and solid.get("bridge_status") == "Error.NoError"
                 and solid.get("target_status") == "Error.NoError"
                 and solid.get("positive_intersection_volume") is True
                 and isinstance(solid.get("intersection_volume_m3"), (int, float))
                 and np.isfinite(solid["intersection_volume_m3"])
                 and solid["intersection_volume_m3"] > 0.0
                 and solid.get("intersection_component_count") == 1
                 and solid.get("union_status") == "Error.NoError"
                 and solid.get("union_component_count") == 1
                 and solid.get("union_connected_numerical_solid") is True
                 and solid.get("union_lineage_has_both_sources_without_generated_or_unmapped_faces") is True
                 and isinstance(lineage, Mapping)
                 and lineage.get("generated_face_count") == 0
                 and lineage.get("unmapped_face_count") == 0
                 and set(lineage.get("run_original_ids", []))
                 == {edge[0][1], edge[1][1]}
                 and solid.get("floating_boolean_exact_topology_proven") is False
                 and isinstance(volume_accounting, Mapping)
                 and volume_accounting.get("passed") is True
                 and isinstance(witness, Mapping)
                 and witness.get("first_location") == "inside"
                 and witness.get("second_location") == "inside"
                 and "exact rational point location" in witness.get("predicate", ""),
                 f"{role} solid/inside/connected-union proof failed")

    calibres = report.get("independent_source_station_calibre")
    _require(report.get("calibre_passed") is True
             and isinstance(calibres, Mapping)
             and set(calibres) == {"arch", "ascending"},
             "independent source station calibre did not pass for both rows")
    for role, key in (("arch", edge[0]), ("ascending", edge[1])):
        calibre = calibres[role]
        comparison = calibre.get("candidate_to_baseline_comparison")
        ratios = comparison.get("candidate_to_baseline_ratios", {})
        bounds = comparison.get("declared_preservation_intervals", {})
        station = calibre.get("station")
        baseline_station = calibre.get("baseline_station_transported_by_source_ancestry")
        _require(calibre.get("passed") is True and calibre.get("owner_id") == key[1]
                 and calibre.get("role") == role
                 and isinstance(comparison, Mapping)
                 and comparison.get("passed") is True
                 and bounds == {"area": [0.8, 1.25],
                                "minimum_feret_width": [0.9, 1.1]}
                 and all(isinstance(ratios.get(metric), (int, float))
                         and bounds[metric][0] <= ratios[metric] <= bounds[metric][1]
                         for metric in ("area", "minimum_feret_width"))
                 and isinstance(station, Mapping)
                 and isinstance(baseline_station, Mapping)
                 and station.get("transported_from_source_vertex_ancestry") is True
                 and baseline_station.get("transported_from_source_vertex_ancestry") is True
                 and station.get("frozen_collar_face_ids")
                 == baseline_station.get("frozen_collar_face_ids"),
                 f"{role} source-calibre/station proof failed for {key}")
        for section_name in ("candidate_section", "baseline_section"):
            section = calibre.get(section_name)
            _require(isinstance(section, Mapping)
                     and section.get("passed") is True
                     and section.get("lineage_complete") is True
                     and section.get("generated_face_count") == 0
                     and section.get("unmapped_face_count") == 0
                     and type(section.get("loop_count")) is int
                     and section["loop_count"] >= 1,
                     f"{role} {section_name} lacks source lineage/calibre")

    descriptor_limits = report.get("limitations")
    _require(report.get("anatomical_admission") is False
             and report.get("native_or_gpu_work_performed") is False
             and isinstance(descriptor_limits, Mapping)
             and descriptor_limits.get("all_nearby_admission_not_established") is True
             and descriptor_limits.get("clinical_calibre_not_established") is True,
             "source interface evidence was mislabeled as anatomy/native admission")

    pair_sha = source_ancestry["pairset_sha256"]
    bundle_sha = _canonical_capture_bundle_sha256(bundle)
    canonical = {
        "schema": "numi.human.verified-current-source-interface-report.v1",
        "status": "VERIFIED_SOURCE_BOUND_CURRENT_INTERFACE_REPORT",
        "passed": True,
        "typed_link": [list(edge[0]), list(edge[1])],
        "evidence_report_sha256": report_sha,
        "source_preflight_sha256": preflight_sha,
        "capture_input_bundle_sha256": bundle_sha,
        "source_topology_passed": True,
        "interface_gate_passed": True,
        "solid_gate_passed": True,
        "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
        "source_rows": {f"{key[0]}:{key[1]}": AORTIC_ARCH_SOURCE_ROWS[key]
                        for key in edge},
        "exact_face_pairs": [list(pair) for pair in pairs],
        "exact_face_pairset_sha256": pair_sha,
        "candidate_surface_bindings": fingerprints,
    }
    handoff = _validate_source_interface_certificate(
        canonical, surfaces, network_keys, edge=edge,
        schema=canonical["schema"], status=canonical["status"],
        evidence_report_sha256=report_sha,
        source_preflight_sha256=preflight_sha,
        capture_input_bundle_sha256=bundle_sha,
        exact_pair_count=len(pairs), exact_pairset_sha256=pair_sha,
        source_rows=source_rows, pair_dtype="<u4")
    handoff.update({
        "report_reference_schema": CURRENT_SOURCE_INTERFACE_REPORT_REFERENCE_SCHEMA,
        "report_path": str(report_path),
        "preflight_path": str(preflight_path),
        "source_input_pins": normalized_source_pins,
        "proof_code_pins": normalized_code_pins,
        "runtime_pin": dict(runtime_pin),
        "current_capture_bundle": bundle,
        "anatomical_admission": False,
    })
    return handoff

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
    verified_nonnetwork_interface_certificates=None,
    current_capture_input_bundle=None, progress_callback=None,
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
                 and bool(verified_nonnetwork_interface_certificates),
                 "source-bound nonnetwork interface certificates must be a nonempty map")
        for edge, certificate in verified_nonnetwork_interface_certificates.items():
            _require(edge in supported_interfaces,
                     "only the exact source-bound aortic interfaces or current report references are supported")
            if (isinstance(certificate, Mapping)
                    and certificate.get("schema")
                    == CURRENT_SOURCE_INTERFACE_REPORT_REFERENCE_SCHEMA):
                nonnetwork_interface_handoffs[edge] = (
                    _validate_current_source_interface_report_reference(
                        certificate, surfaces, network_keys,
                        current_capture_input_bundle=current_capture_input_bundle,
                        source_inventory=verified_network_source_inventory))
                continue
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
    station_scope = spec.get("station_scope")
    if station_scope == "SOURCE_INTERFACE_PLANE_V1":
        tangent = np.asarray(spec.get("tangent_world_unit"), dtype=np.float64)
        norm = float(np.linalg.norm(tangent))
        _require(tangent.shape == (3,) and np.isfinite(tangent).all()
                 and abs(norm - 1.0) <= 1e-12,
                 "source-interface station normal must be finite and unit length")
        tangent = tangent / norm
    else:
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
    if station_scope in (
            "TARGET_COLLAR_LOCAL_SWEEP_V1", "SOURCE_INTERFACE_PLANE_V1"):
        _require(np.isfinite(requested_offset)
                 and requested_offset >= 0.0
                 and requested_offset <= patch_extent + margin,
                 "local source station must remain inside its frozen collar extent")
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
        "GLOBAL_SOURCE_PCA_V1", "COLLAR_LOCAL_PCA_V1",
        "SOURCE_INTERFACE_PCA_PLANE_V1"),
        "unknown source station selection policy")
    junction = witness.mean(axis=0)
    local_tangent_support = None
    local_pca_eigenvalues = None
    source_interface_pca = None
    if station_policy == "SOURCE_INTERFACE_PCA_PLANE_V1":
        _require(route_anchor is None,
                 "source-interface station orientation is fixed by its witness plane")
        centered = witness - junction
        covariance = centered.T @ centered / max(1, len(witness))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        separation = float(eigenvalues[1] - eigenvalues[0])
        _require(np.isfinite(eigenvalues).all()
                 and separation > max(1e-24, abs(float(eigenvalues[-1])) * 1e-6),
                 "exact source interface points do not define a unique plane normal")
        direction = eigenvectors[:, 0]
        dominant = int(np.argmax(np.abs(direction)))
        if direction[dominant] < 0.0:
            direction = -direction
        tangent = direction / float(np.linalg.norm(direction))
        collar = np.asarray(collar_face_ids, dtype=np.int64)
        _require(collar.ndim == 1 and len(collar) > 0
                 and int(collar.min()) >= 0 and int(collar.max()) < len(faces),
                 "source-interface station needs valid frozen source collar faces")
        local_tangent_support = np.unique(faces[collar].reshape(-1))
        ancestry = _material_vertex_ancestry(
            surface, tangent, collar_face_ids,
            tangent_support_vertex_ids=local_tangent_support)
        anchor_ids = np.asarray(
            ancestry["material_anchor_local_vertex_ids"], dtype=np.int64)
        anchor_weights = np.asarray(
            ancestry["material_anchor_weights"], dtype=np.float64)
        material_anchor = np.sum(positions[anchor_ids] * anchor_weights[:, None], axis=0)
        offset = float(np.dot(junction - material_anchor, tangent))
        if offset < 0.0:
            tangent = -tangent
            offset = -offset
            ancestry = _material_vertex_ancestry(
                surface, tangent, collar_face_ids,
                tangent_support_vertex_ids=local_tangent_support)
            anchor_ids = np.asarray(
                ancestry["material_anchor_local_vertex_ids"], dtype=np.int64)
            anchor_weights = np.asarray(
                ancestry["material_anchor_weights"], dtype=np.float64)
            material_anchor = np.sum(
                positions[anchor_ids] * anchor_weights[:, None], axis=0)
            offset = float(np.dot(junction - material_anchor, tangent))
        collar_points = positions[faces[collar]].reshape(-1, 3)
        patch_extent = float(np.max((collar_points - material_anchor) @ tangent))
        margin = max(float(np.max(np.abs(np.spacing(surface.positions)))) * 8.0,
                     float(np.finfo(np.float32).tiny))
        _require(offset >= 0.0 and offset <= patch_extent + margin,
                 "source interface plane lies outside its anchored collar extent")
        source_interface_pca = {
            "method": "least-variance eigenvector of the frozen exact source-interface point cloud",
            "covariance_eigenvalues_m2": eigenvalues.tolist(),
            "least_variance_eigenvalue_gap_m2": separation,
            "normal_uniqueness_relative_gap": separation / max(
                abs(float(eigenvalues[-1])), 1e-300),
            "witness_points_sha256": _source_interface_points_sha256(witness),
            "witness_point_count": len(witness),
            "witness_centroid_world_m": junction.tolist(),
            "plane_normal_world_unit": tangent.tolist(),
            "candidate_geometry_used_for_selection": False,
        }
        local_pca_eigenvalues = eigenvalues
        method = "least-variance PCA normal of exact source interface points"
    elif station_policy == "COLLAR_LOCAL_PCA_V1":
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
        "station_offset_from_anchor_m": (
            offset if station_policy == "SOURCE_INTERFACE_PCA_PLANE_V1"
            else patch_extent + margin),
        "station_selection_policy": station_policy,
        **({"station_scope": "SOURCE_INTERFACE_PLANE_V1",
            "source_interface_plane": source_interface_pca}
           if source_interface_pca is not None else {}),
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
    anchor_role="bridge", station_role=None, transverse_witness_provider=None,
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
    station_role = anchor_role if station_role is None else station_role
    _require(station_role in ("bridge", "target", "junction"),
             "source station role must be bridge, target, or junction")
    _require(station_role != "junction" or anchor_role in ("bridge", "target"),
             "junction station needs one explicit branch material anchor")
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
    baseline_shared_source_station_sections = None
    if station_policy == "SOURCE_INTERFACE_PCA_PLANE_V1":
        baseline_shared_source_station_sections = {}
        for branch_role, branch_surface, owner_id in (
            ("bridge", bridge, bridge_key[1]),
            ("target", target, target_key[1]),
        ):
            section = _source_surface_section(branch_surface, source_station)
            _require(section.get("passed") is True
                     and section.get("loop_count") == 1
                     and len(section.get("loop_metrics", [])) == 1
                     and section["loop_metrics"][0].get("lineage_complete") is True
                     and section["loop_metrics"][0].get("owner_ids") == [owner_id],
                     f"source interface plane lacks one complete lineage-bearing section "
                     f"for {branch_role} owner {owner_id}")
            baseline_shared_source_station_sections[str(owner_id)] = section
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
        "station_role": station_role,
        "neighbor_route_anchor_evidence": route_anchor_record,
        "baseline_shared_source_station_sections": (
            baseline_shared_source_station_sections),
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
        "station_role": station_role,
        "anchor_role": anchor_role,
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


def _point_in_or_on_triangle2d_exact(point, first, second, third, orientation):
    return all(
        orientation * _orient2d_exact(a, b, point) >= 0
        for a, b in ((first, second), (second, third), (third, first))
    )


def _triangulate_simple_polygon2d_exact(
    points, *, start_offset=0, prefer_last_ear=False,
    quality_priority=False,
):
    """Ear-clip a simple polygon using exact orientation predicates."""
    polygon = np.asarray(points, dtype=np.float64)
    if not _simple_closed_polygon2d_exact(polygon):
        return None
    signed_area = _signed_polygon_area2d_exact(polygon)
    if signed_area == 0.0:
        return None
    orientation = 1 if signed_area > 0.0 else -1
    remaining = list(range(len(polygon)))
    offset = int(start_offset) % len(remaining)
    remaining = remaining[offset:] + remaining[:offset]
    triangles = []
    while len(remaining) > 3:
        ears = []
        scan = range(len(remaining) - 1, -1, -1) if prefer_last_ear else range(len(remaining))
        for current_index in scan:
            previous_index = (current_index - 1) % len(remaining)
            next_index = (current_index + 1) % len(remaining)
            previous = remaining[previous_index]
            current = remaining[current_index]
            following = remaining[next_index]
            signed_double_area = orientation * _orient2d_exact(
                polygon[previous], polygon[current], polygon[following])
            if signed_double_area <= 0:
                continue
            if any(
                _point_in_or_on_triangle2d_exact(
                    polygon[other], polygon[previous], polygon[current],
                    polygon[following], orientation)
                for other in remaining
                if other not in (previous, current, following)
            ):
                continue
            if quality_priority:
                edge_lengths_squared = (
                    float(np.dot(polygon[previous] - polygon[current],
                                 polygon[previous] - polygon[current])),
                    float(np.dot(polygon[current] - polygon[following],
                                 polygon[current] - polygon[following])),
                    float(np.dot(polygon[following] - polygon[previous],
                                 polygon[following] - polygon[previous])),
                )
                score = float(signed_double_area) / max(edge_lengths_squared)
            else:
                score = 0.0
            stable_tie_break = (
                current_index if prefer_last_ear else -current_index)
            ears.append((score, stable_tie_break, current_index,
                         (previous, current, following)))
        if not ears:
            return None
        _score, _stable_tie_break, current_index, ear = max(ears)
        triangles.append(ear)
        del remaining[current_index]
    final_triangle = tuple(remaining)
    if orientation * _orient2d_exact(
            polygon[final_triangle[0]], polygon[final_triangle[1]],
            polygon[final_triangle[2]]) <= 0:
        return None
    triangles.append(final_triangle)
    return tuple(triangles)


def _triangulated_polygon_area3d(points, triangles):
    polygon = np.asarray(points, dtype=np.float64)
    return sum(
        0.5 * float(np.linalg.norm(np.cross(
            polygon[first] - polygon[third],
            polygon[second] - polygon[third])))
        for first, second, third in triangles
    )


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

    source_orientation = 1 if source_signed_area > 0.0 else -1
    candidate_orientation = 1 if candidate_signed_area > 0.0 else -1
    source_triangles = _triangulate_simple_polygon2d_exact(
        source_projected, quality_priority=True)
    source_alternate_triangles = _triangulate_simple_polygon2d_exact(
        source_projected, start_offset=1, prefer_last_ear=True)
    candidate_primary_triangles = _triangulate_simple_polygon2d_exact(
        candidate_projected, quality_priority=True)
    candidate_alternate_triangles = _triangulate_simple_polygon2d_exact(
        candidate_projected, start_offset=1, prefer_last_ear=True)
    source_triangulation_passed = source_triangles is not None
    source_alternate_triangulation_passed = (
        source_alternate_triangles is not None)
    candidate_primary_triangulation_passed = (
        candidate_primary_triangles is not None)
    candidate_alternate_triangulation_passed = (
        candidate_alternate_triangles is not None)
    source_index_transfer_failures = []
    if source_triangles is not None:
        for triangle in source_triangles:
            source_double_area = source_orientation * _orient2d_exact(
                source_projected[triangle[0]], source_projected[triangle[1]],
                source_projected[triangle[2]])
            candidate_double_area = source_orientation * _orient2d_exact(
                candidate_projected[triangle[0]],
                candidate_projected[triangle[1]],
                candidate_projected[triangle[2]])
            if candidate_double_area <= 0:
                source_index_transfer_failures.append({
                    "material_ring_vertex_indices": list(triangle),
                    "source_projected_double_area_m2": float(source_double_area),
                    "candidate_projected_double_area_m2": float(candidate_double_area),
                })
    def candidate_orientation_failures(triangles):
        return [
            {"material_ring_vertex_indices": list(triangle),
             "candidate_projected_double_area_m2": float(
                 _orient2d_exact(candidate_projected[triangle[0]],
                                 candidate_projected[triangle[1]],
                                 candidate_projected[triangle[2]]))}
            for triangle in triangles or ()
            if candidate_orientation * _orient2d_exact(
                candidate_projected[triangle[0]],
                candidate_projected[triangle[1]],
                candidate_projected[triangle[2]]) <= 0
        ]
    candidate_primary_orientation_failures = candidate_orientation_failures(
        candidate_primary_triangles)
    candidate_alternate_orientation_failures = candidate_orientation_failures(
        candidate_alternate_triangles)
    candidate_triangulations_oriented = bool(
        candidate_primary_triangulation_passed
        and candidate_alternate_triangulation_passed
        and not candidate_primary_orientation_failures
        and not candidate_alternate_orientation_failures)
    source_primary_virtual_disk_area = (
        _triangulated_polygon_area3d(source, source_triangles)
        if source_triangles is not None else 0.0)
    source_alternate_virtual_disk_area = (
        _triangulated_polygon_area3d(source, source_alternate_triangles)
        if source_alternate_triangles is not None else 0.0)
    candidate_primary_virtual_disk_area = (
        _triangulated_polygon_area3d(candidate, candidate_primary_triangles)
        if candidate_primary_triangles is not None else 0.0)
    candidate_alternate_area = (
        _triangulated_polygon_area3d(candidate, candidate_alternate_triangles)
        if candidate_alternate_triangles is not None else 0.0)
    source_warp = float(np.max(np.abs((source - source_center) @ tangent)))
    candidate_warp = float(np.max(np.abs(
        (candidate - candidate_center) @ candidate_normal)))
    source_perimeter = float(np.linalg.norm(
        np.roll(source, -1, axis=0) - source, axis=1).sum())
    candidate_perimeter = float(np.linalg.norm(
        np.roll(candidate, -1, axis=0) - candidate, axis=1).sum())
    _require(source_triangulation_passed
             and source_alternate_triangulation_passed
             and candidate_primary_triangulation_passed
             and candidate_alternate_triangulation_passed
             and source_primary_virtual_disk_area > 0.0
             and source_alternate_virtual_disk_area > 0.0
             and candidate_primary_virtual_disk_area > 0.0
             and candidate_alternate_area > 0.0 and source_perimeter > 0.0,
             "material-section ring has no valid simple-polygon triangulation or 3-D area")

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
    source_primary_virtual_disk_area_ratio = (
        source_primary_virtual_disk_area / baseline_area)
    source_alternate_virtual_disk_area_ratio = (
        source_alternate_virtual_disk_area / baseline_area)
    candidate_primary_virtual_disk_area_ratio = (
        candidate_primary_virtual_disk_area / baseline_area)
    candidate_alternate_virtual_disk_area_ratio = (
        candidate_alternate_area / baseline_area)
    projected_area_ratio = candidate_projected_area / baseline_area
    width_ratio = candidate_width / baseline_width
    width_ratio_bounds = (
        (candidate_width - 2.0 * candidate_warp) / baseline_width,
        (candidate_width + 2.0 * candidate_warp) / baseline_width,
    )
    warp_ratio = candidate_warp / baseline_width
    projection_error = abs(
        candidate_primary_virtual_disk_area - candidate_projected_area)
    perimeter_ratio = candidate_perimeter / source_perimeter
    passed = bool(
        source_simple and candidate_simple
        and source_triangulation_passed
        and candidate_primary_triangulation_passed
        and candidate_alternate_triangulation_passed
        and candidate_triangulations_oriented
        and source_orientation == candidate_orientation
        and warp_ratio <= maximum_warp_to_feret_ratio
        and area_bounds[0] <= projected_area_ratio <= area_bounds[1]
        and area_bounds[0] <= source_primary_virtual_disk_area_ratio
        <= area_bounds[1]
        and area_bounds[0] <= source_alternate_virtual_disk_area_ratio
        <= area_bounds[1]
        and area_bounds[0] <= candidate_primary_virtual_disk_area_ratio
        <= area_bounds[1]
        and area_bounds[0] <= candidate_alternate_virtual_disk_area_ratio
        <= area_bounds[1]
        and width_bounds[0] <= width_ratio_bounds[0]
        and width_ratio_bounds[1] <= width_bounds[1]
    )
    return {
        "passed": passed,
        "method": (
            "same-source-edge-fraction material ring, best-fit transported "
            "independently ear-clipped source and candidate simple polygons; "
            "two candidate triangulations report virtual-disk area sensitivity"),
        "source_material_ring_is_simple_closed": source_simple,
        "candidate_material_ring_is_simple_closed": candidate_simple,
        "source_primary_simple_polygon_triangulation_passed": (
            source_triangulation_passed),
        "source_alternate_simple_polygon_triangulation_passed": (
            source_alternate_triangulation_passed),
        "candidate_primary_simple_polygon_triangulation_passed": (
            candidate_primary_triangulation_passed),
        "candidate_alternate_simple_polygon_triangulation_passed": (
            candidate_alternate_triangulation_passed),
        "candidate_triangulations_preserve_candidate_ring_orientation": (
            candidate_triangulations_oriented),
        "source_index_triangulation_transfer_failure_count_diagnostic_only": (
            len(source_index_transfer_failures)),
        "source_index_triangulation_transfer_failure_witnesses_diagnostic_only": (
            source_index_transfer_failures[:12]),
        "source_primary_triangulation_triangle_count": (
            len(source_triangles) if source_triangles is not None else 0),
        "source_alternate_triangulation_triangle_count": (
            len(source_alternate_triangles)
            if source_alternate_triangles is not None else 0),
        "candidate_primary_triangulation_triangle_count": (
            len(candidate_primary_triangles)
            if candidate_primary_triangles is not None else 0),
        "candidate_alternate_triangulation_triangle_count": (
            len(candidate_alternate_triangles)
            if candidate_alternate_triangles is not None else 0),
        "source_primary_triangulation_selection_policy": (
            "exact ear clipping from the independently declared source ring; "
            "eligible ears maximize baseline projected double-area divided by "
            "longest-edge squared, with source-index tie break"),
        "source_alternate_triangulation_selection_policy": (
            "same simple source ring, with traversal offset one and reverse "
            "ear scan to expose virtual-disk triangulation sensitivity"),
        "candidate_triangulation_selection_policy": (
            "the same deterministic ear rule applied independently to each "
            "candidate projected ring; these are virtual-disk measurement "
            "triangulations, not tissue elements"),
        "source_primary_triangulation_vertex_index_triples": (
            [list(triangle) for triangle in source_triangles]
            if source_triangles is not None else []),
        "source_alternate_triangulation_vertex_index_triples": (
            [list(triangle) for triangle in source_alternate_triangles]
            if source_alternate_triangles is not None else []),
        "candidate_primary_triangulation_vertex_index_triples": (
            [list(triangle) for triangle in candidate_primary_triangles]
            if candidate_primary_triangles is not None else []),
        "candidate_alternate_triangulation_vertex_index_triples": (
            [list(triangle) for triangle in candidate_alternate_triangles]
            if candidate_alternate_triangles is not None else []),
        "candidate_orientation_failure_count": (
            len(candidate_primary_orientation_failures)
            + len(candidate_alternate_orientation_failures)),
        "candidate_orientation_failure_witnesses": (
            (candidate_primary_orientation_failures
             + candidate_alternate_orientation_failures)[:12]),
        "source_primary_triangulation_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(source_triangles or (), dtype="<u4").tobytes()
        ).hexdigest(),
        "source_alternate_triangulation_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(
                source_alternate_triangles or (), dtype="<u4").tobytes()
        ).hexdigest(),
        "candidate_primary_triangulation_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(
                candidate_primary_triangles or (), dtype="<u4").tobytes()
        ).hexdigest(),
        "candidate_alternate_triangulation_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(
                candidate_alternate_triangles or (), dtype="<u4").tobytes()
        ).hexdigest(),
        "candidate_area_triangulations_are_measurement_only": True,
        "source_and_candidate_ring_orientation_match": (
            source_orientation == candidate_orientation),
        "source_projected_area_m2": source_area,
        "candidate_projected_area_m2": candidate_projected_area,
        "source_primary_virtual_disk_area_m2": (
            source_primary_virtual_disk_area),
        "source_alternate_virtual_disk_area_m2": (
            source_alternate_virtual_disk_area),
        "candidate_primary_virtual_disk_area_m2": (
            candidate_primary_virtual_disk_area),
        "candidate_alternate_virtual_disk_area_m2": (
            candidate_alternate_area),
        "source_primary_virtual_disk_area_ratio": (
            source_primary_virtual_disk_area_ratio),
        "source_alternate_virtual_disk_area_ratio": (
            source_alternate_virtual_disk_area_ratio),
        "candidate_projected_area_ratio": projected_area_ratio,
        "candidate_primary_virtual_disk_area_ratio": (
            candidate_primary_virtual_disk_area_ratio),
        "candidate_alternate_virtual_disk_area_ratio": (
            candidate_alternate_virtual_disk_area_ratio),
        "source_virtual_disk_triangulation_sensitivity_m2": abs(
            source_alternate_virtual_disk_area
            - source_primary_virtual_disk_area),
        "candidate_virtual_disk_triangulation_sensitivity_m2": abs(
            candidate_alternate_area - candidate_primary_virtual_disk_area),
        "source_minimum_feret_width_m": source_width,
        "candidate_projected_minimum_feret_width_m": candidate_width,
        "projected_minimum_feret_width_ratio": width_ratio,
        "minimum_feret_width_ratio_interval": list(width_ratio_bounds),
        "source_maximum_out_of_plane_warp_m": source_warp,
        "candidate_maximum_out_of_plane_warp_m": candidate_warp,
        "maximum_warp_to_source_minimum_feret_ratio": warp_ratio,
        "declared_maximum_warp_to_feret_ratio": maximum_warp_to_feret_ratio,
        "projection_area_error_m2": projection_error,
        "projection_area_error_fraction_of_candidate_primary_virtual_disk": (
            projection_error / candidate_primary_virtual_disk_area
            if candidate_primary_virtual_disk_area > 0.0 else None),
        "source_perimeter_m": source_perimeter,
        "candidate_3d_perimeter_m": candidate_perimeter,
        "perimeter_ratio_diagnostic_only": perimeter_ratio,
        "declared_preservation_intervals": {
            "area": list(area_bounds),
            "minimum_feret_width": list(width_bounds),
        },
        "virtual_disk_area_not_physical_tissue_area": True,
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


def _section_vertex_classes(points_for_distance, precision_points, center, tangent):
    """Apply the exact section zero-class rule shared by source and CSG paths."""
    points_for_distance = np.asarray(points_for_distance, dtype=np.float64)
    precision_points = np.asarray(precision_points)
    _require(points_for_distance.ndim == 2 and points_for_distance.shape[1] == 3
             and precision_points.ndim == 2 and precision_points.shape[1] == 3
             and precision_points.dtype.kind == "f"
             and np.isfinite(points_for_distance).all()
             and np.isfinite(precision_points).all(),
             "section vertex classification arrays must be finite XYZ coordinates")
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    _require(center.shape == (3,) and np.isfinite(center).all()
             and tangent.shape == (3,) and np.isfinite(tangent).all()
             and float(np.linalg.norm(tangent)) > 0.0,
             "section vertex classification needs a finite plane")
    tangent = tangent / np.linalg.norm(tangent)
    world_scale = max(1.0, float(np.max(np.abs(precision_points))))
    coordinate_precision = np.finfo(precision_points.dtype)
    ulp = max(float(np.max(np.abs(np.spacing(precision_points)))),
              float(coordinate_precision.eps) * world_scale)
    tolerance = max(ulp * 2.0, float(coordinate_precision.tiny))
    signed_distances = (points_for_distance - center) @ tangent
    signs = np.zeros(len(points_for_distance), dtype=np.int8)
    signs[signed_distances > tolerance] = 1
    signs[signed_distances < -tolerance] = -1
    return signed_distances, signs, tolerance


def _section_loops(positions, faces, center, tangent, *, derived_csg=False,
                   face_owner_ids=None, face_source_ids=None,
                   face_generated_mask=None, include_material_point_records=False,
                   include_projected_loop_vertices=False,
                   include_face_segment_records=False):
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

    signed_distances, signs, tolerance = _section_vertex_classes(
        canonical_points, points, center, tangent)

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
        if include_face_segment_records:
            # Retain the ordered per-segment face ancestry and exact Mesh64
            # section endpoints for source-tangent canonicalization. Normalize
            # loop direction to CCW and rotate by lineage tokens, so edge start
            # node allocation cannot change the ordered chain proof.
            ordered_nodes = list(loop_node_keys)
            signed_area_twice = float(
                np.dot(loop[:, 0], np.roll(loop[:, 1], -1))
                - np.dot(loop[:, 1], np.roll(loop[:, 0], -1)))
            if signed_area_twice < 0.0:
                ordered_nodes = [ordered_nodes[0], *reversed(ordered_nodes[1:])]
            edge_records = []
            for edge_index, first_node in enumerate(ordered_nodes):
                second_node = ordered_nodes[(edge_index + 1) % len(ordered_nodes)]
                edge_key = tuple(sorted((first_node, second_node)))
                incident_faces = sorted(segment_faces[edge_key])
                face_records = []
                for source_face_index in incident_faces:
                    face_record = {
                        "owner_id": (
                            int(face_owner_ids[source_face_index])
                            if lineage_available else None),
                        "source_face_id": (
                            int(face_source_ids[source_face_index])
                            if lineage_available else None),
                        "generated": bool(face_generated_mask[source_face_index]),
                        "triangle_corner_positions_world_m": points[
                            triangles[source_face_index], :3].astype(
                                np.float64).tolist(),
                    }
                    face_records.append(face_record)
                edge_records.append({
                    "section_edge_index": edge_index,
                    "segment_endpoints_world_m": [
                        np.asarray(node_positions[first_node], dtype=np.float64).tolist(),
                        np.asarray(node_positions[second_node], dtype=np.float64).tolist()],
                    "face_lineage": face_records,
                })
            edge_tokens = [tuple(
                (item["owner_id"], item["source_face_id"], item["generated"])
                for item in edge_record["face_lineage"])
                for edge_record in edge_records]
            if edge_tokens:
                rotation = min(
                    range(len(edge_tokens)),
                    key=lambda offset: tuple(
                        edge_tokens[offset:] + edge_tokens[:offset]))
                edge_records = edge_records[rotation:] + edge_records[:rotation]
            metric["ordered_face_segment_records"] = edge_records
            if all(len(record["face_lineage"]) == 1
                   and record["face_lineage"][0]["owner_id"] is not None
                   and record["face_lineage"][0]["source_face_id"] is not None
                   and not record["face_lineage"][0]["generated"]
                   for record in edge_records):
                metric["ordered_source_face_ancestor_cycle"] = [
                    record["face_lineage"][0]["source_face_id"]
                    for record in edge_records]
            else:
                metric["ordered_source_face_ancestor_cycle"] = None
        if include_projected_loop_vertices:
            metric["projected_loop_vertices"] = loop.tolist()
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


def _source_surface_section(
    surface, station, *, include_material_point_records=False,
    include_projected_loop_vertices=False,
    include_face_segment_records=False,
):
    return _section_loops(
        surface.positions, surface.faces,
        station["section_center_world_m"], station["tangent_world_unit"],
        include_material_point_records=include_material_point_records,
        include_projected_loop_vertices=include_projected_loop_vertices,
        include_face_segment_records=include_face_segment_records,
        **_source_section_lineage(surface))


def _source_station_plane_side(surface, station):
    """Prove a closed source mesh misses a plane by strict one-sided vertices."""
    positions = _float32_xyz(surface.positions, name="source station applicability")
    center = np.asarray(station.get("section_center_world_m"), dtype=np.float64)
    tangent = np.asarray(station.get("tangent_world_unit"), dtype=np.float64)
    _require(center.shape == (3,) and np.isfinite(center).all()
             and tangent.shape == (3,) and np.isfinite(tangent).all()
             and float(np.linalg.norm(tangent)) > 0.0,
             "source station applicability has an invalid plane")
    tangent /= np.linalg.norm(tangent)
    distances = (positions.astype(np.float64) - center) @ tangent
    world_scale = max(1.0, float(np.max(np.abs(positions))))
    ulp = max(float(np.max(np.abs(np.spacing(positions)))),
              float(np.finfo(positions.dtype).eps) * world_scale)
    tolerance = max(2.0 * ulp, float(np.finfo(positions.dtype).tiny))
    if float(np.min(distances)) > tolerance:
        side = "STRICTLY_POSITIVE"
    elif float(np.max(distances)) < -tolerance:
        side = "STRICTLY_NEGATIVE"
    else:
        side = "NOT_PROVEN_ABSENT"
    return {
        "status": side,
        "minimum_signed_vertex_distance_m": float(np.min(distances)),
        "maximum_signed_vertex_distance_m": float(np.max(distances)),
        "strict_side_tolerance_m": tolerance,
        "source_vertex_count": len(positions),
        "source_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "plane_station_sha256": _canonical_record_sha256(dict(station)),
    }


def _source_station_owner_section(surface, station, owner_id, *, label):
    """Classify one source owner as a complete cut or a strictly absent side."""
    section = _source_surface_section(surface, station)
    loops = section.get("loop_metrics")
    if (type(section.get("loop_count")) is int
            and section["loop_count"] > 0
            and isinstance(loops, list)
            and len(loops) == section["loop_count"]):
        records = []
        for loop_index, metric in enumerate(loops):
            ancestors = metric.get("source_face_ancestor_ids")
            _require(metric.get("passed") is True
                     and metric.get("lineage_available") is True
                     and metric.get("lineage_complete") is True
                     and metric.get("owner_ids") == [owner_id]
                     and isinstance(ancestors, list) and bool(ancestors)
                     and all(type(face_id) is int and face_id >= 0 for face_id in ancestors)
                     and len(set(ancestors)) == len(ancestors)
                     and metric.get("generated_face_count") == 0
                     and metric.get("unmapped_face_count") == 0
                     and isinstance(metric.get("area_m2"), (int, float))
                     and np.isfinite(metric["area_m2"]) and metric["area_m2"] > 0.0
                     and isinstance(metric.get("minimum_feret_width_m"), (int, float))
                     and np.isfinite(metric["minimum_feret_width_m"])
                     and metric["minimum_feret_width_m"] > 0.0,
                     f"{label} loop {loop_index} lacks complete positive source lineage")
            records.append({
                "loop_index": loop_index,
                "source_face_ancestor_ids": list(ancestors),
                "source_face_ancestor_ids_sha256": metric.get(
                    "source_face_ancestor_ids_sha256"),
                "area_m2": float(metric["area_m2"]),
                "minimum_feret_width_m": float(metric["minimum_feret_width_m"]),
            })
        return {
            "status": "PRESENT_COMPLETE_SOURCE_LOOPS",
            "typed_identity": list(surface.key),
            "section_plane_sha256": _canonical_record_sha256({
                "center": section.get("section_center_world_m"),
                "tangent": section.get("section_tangent_world_unit")}),
            "loop_count": len(records),
            "loops": records,
            "section_passed_as_single_loop": section.get("passed") is True,
        }
    side = _source_station_plane_side(surface, station)
    _require(side["status"] in ("STRICTLY_POSITIVE", "STRICTLY_NEGATIVE"),
             f"{label} is neither a complete section nor proven absent on one plane side")
    _require(section.get("loop_count") == 0,
             f"{label} has malformed section loops despite one-sided absence")
    return {
        "status": "ABSENT_STRICTLY_ON_ONE_SIDE",
        "typed_identity": list(surface.key),
        "section_plane_sha256": _canonical_record_sha256({
            "center": station.get("section_center_world_m"),
            "tangent": station.get("tangent_world_unit")}),
        "loop_count": 0,
        "one_sided_absence": side,
        "section_reason": section.get("reason"),
    }


def _source_station_applicability_body(
    bridge_surface, target_surface, *, bridge_key, target_key,
    station_role, anchor_owner_id, station,
):
    owner_records = {}
    for surface, key in ((bridge_surface, bridge_key), (target_surface, target_key)):
        _require(surface.key == key,
                 "source station applicability surface identity changed")
        owner_records[str(key[1])] = _source_station_owner_section(
            surface, station, key[1], label=f"source baseline owner {key[1]}")
    present = sorted(int(owner) for owner, item in owner_records.items()
                     if item["status"] == "PRESENT_COMPLETE_SOURCE_LOOPS")
    absent = sorted(int(owner) for owner, item in owner_records.items()
                    if item["status"] == "ABSENT_STRICTLY_ON_ONE_SIDE")
    expected_owners = sorted((bridge_key[1], target_key[1]))
    _require(sorted(present + absent) == expected_owners,
             "source station applicability does not partition both typed owners")
    if present == expected_owners and not absent:
        mode = "SHARED_TWO_OWNER_CUT"
    elif present == [anchor_owner_id] and len(absent) == 1:
        mode = "ANCHOR_ONLY_CUT"
    else:
        raise ValueError(
            "source station is neither a shared two-owner cut nor an anchor-only cut")
    return {
        "schema": "numi.human.source-baseline-station-applicability.v1",
        "typed_link": [list(bridge_key), list(target_key)],
        "station_role": station_role,
        "anchor_owner_id": anchor_owner_id,
        "source_station_sha256": _canonical_record_sha256(dict(station)),
        "baseline_source_surface_bindings": {
            "bridge": _source_geometry_binding(bridge_surface),
            "target": _source_geometry_binding(target_surface),
        },
        "owner_status_by_id": owner_records,
        "present_owner_ids": present,
        "absent_owner_ids": absent,
        "applicability_mode": mode,
        "independent_source_baseline_only": True,
        "candidate_geometry_used_for_declaration": False,
    }


def declare_source_baseline_station_applicability(
    verifier, *, bridge_key, baseline_bridge_surface, target_key,
    baseline_target_surface, station_role, anchor_owner_id, source_station,
):
    """Create an explicit, reproducible source-only cut-applicability declaration."""
    bridge_key, target_key = _identity(bridge_key), _identity(target_key)
    _require(station_role in ("bridge", "target", "junction")
             and anchor_owner_id in (bridge_key[1], target_key[1]),
             "source station applicability has the wrong anchor role")
    bridge = _prepare_vessel(verifier, bridge_key, baseline_bridge_surface)
    target = _prepare_vessel(verifier, target_key, baseline_target_surface)
    body = _source_station_applicability_body(
        bridge, target, bridge_key=bridge_key, target_key=target_key,
        station_role=station_role, anchor_owner_id=anchor_owner_id,
        station=source_station)
    return {**body, "declaration_sha256": _canonical_record_sha256(body)}


def _validated_source_station_applicability_modes(records):
    _require(isinstance(records, Mapping)
             and {"bridge", "target"} <= set(records)
             and set(records) <= {"bridge", "target", "junction"},
             "source applicability must declare bridge/target and optional junction stations")
    modes = {}
    for role, record in records.items():
        _require(isinstance(record, Mapping)
                 and record.get("station_role") == role
                 and record.get("applicability_mode") in (
                     "ANCHOR_ONLY_CUT", "SHARED_TWO_OWNER_CUT"),
                 f"{role} source station applicability is malformed")
        modes[role] = record["applicability_mode"]
    _require("SHARED_TWO_OWNER_CUT" in modes.values(),
             "a source-applicability opt-in needs at least one declared shared two-owner cut")
    return modes


def _complete_station_loop_metrics(section, owner_id, *, label):
    loops = section.get("loop_metrics")
    _require(type(section.get("loop_count")) is int and section["loop_count"] > 0
             and isinstance(loops, list) and len(loops) == section["loop_count"],
             f"{label} must retain every closed section loop")
    records = []
    for index, metric in enumerate(loops):
        ancestors = metric.get("source_face_ancestor_ids")
        _require(metric.get("passed") is True
                 and metric.get("lineage_available") is True
                 and metric.get("lineage_complete") is True
                 and metric.get("owner_ids") == [owner_id]
                 and isinstance(ancestors, list) and bool(ancestors)
                 and all(type(face_id) is int and face_id >= 0 for face_id in ancestors)
                 and len(set(ancestors)) == len(ancestors)
                 and metric.get("generated_face_count") == 0
                 and metric.get("unmapped_face_count") == 0,
                 f"{label} loop {index} has mixed, incomplete, or generated lineage")
        records.append(metric)
    return sorted(records, key=lambda item: (
        -float(item["area_m2"]), -float(item["minimum_feret_width_m"]),
        str(item["source_face_ancestor_ids_sha256"])))


def _compare_complete_anchor_only_loop_sets(
    candidate, baseline, *, owner_id, label, exact_ancestry=False,
    tangent_canonicalization=None,
):
    candidate_loops = _complete_station_loop_metrics(
        candidate, owner_id, label=f"{label} candidate")
    baseline_loops = _complete_station_loop_metrics(
        baseline, owner_id, label=f"{label} baseline")
    _require(len(candidate_loops) == len(baseline_loops),
             f"{label} loop count differs across the source-baseline comparison")
    tangent_proofs_by_loop = {}
    if tangent_canonicalization is not None:
        _require(tangent_canonicalization.get("passed") is True
                 and isinstance(tangent_canonicalization.get("loop_proofs"), list),
                 f"{label} source-tangent canonicalization proof is malformed")
        tangent_proofs_by_loop = {
            int(item["loop_index"]): item
            for item in tangent_canonicalization["loop_proofs"]}
    comparisons = []
    for index, (current, prior) in enumerate(zip(candidate_loops, baseline_loops)):
        current_ancestors = set(current["source_face_ancestor_ids"])
        prior_ancestors = set(prior["source_face_ancestor_ids"])
        tangent_proof = tangent_proofs_by_loop.get(index)
        if exact_ancestry:
            if current_ancestors == prior_ancestors:
                _require(tangent_proof is None or tangent_proof.get("passed") is True,
                         f"{label} exact source ancestry proof is invalid")
            else:
                _require(tangent_proof is not None
                         and tangent_proof.get("passed") is True
                         and set(tangent_proof.get("source_section_ancestor_cycle", []))
                             == prior_ancestors
                         and set(tangent_proof.get("union_section_ancestor_cycle", []))
                             == current_ancestors
                         and set(tangent_proof.get("tangent_face_ids_canonicalized", []))
                             == current_ancestors - prior_ancestors
                         and prior_ancestors <= current_ancestors,
                         f"{label} same-state union loop ancestry differs outside the source-F32 tangent proof")
        comparison = _section_comparison(current, prior)
        _require(comparison["passed"],
                 f"{label} loop {index} area/width is outside the declared bounds")
        comparisons.append({
            "loop_rank_by_area_then_width": index,
            "candidate_source_face_ancestor_ids_sha256": current.get(
                "source_face_ancestor_ids_sha256"),
            "baseline_source_face_ancestor_ids_sha256": prior.get(
                "source_face_ancestor_ids_sha256"),
            "candidate_source_face_ancestor_count": len(current_ancestors),
            "baseline_source_face_ancestor_count": len(prior_ancestors),
            "source_f32_tangent_face_canonicalization": tangent_proof,
            **comparison,
        })
    return comparisons


def _canonical_oriented_ancestor_cycle(face_ids, *, label):
    _require(isinstance(face_ids, list) and len(face_ids) >= 3
             and all(type(face_id) is int and face_id >= 0 for face_id in face_ids)
             and len(set(face_ids)) == len(face_ids),
             f"{label} ordered face cycle is missing, repeated, or malformed")
    rotations = [tuple(face_ids[offset:] + face_ids[:offset])
                 for offset in range(len(face_ids))]
    return list(min(rotations))


def _source_tangent_face_classification(surface, station):
    positions = _float32_xyz(surface.positions, name="tangent-face source")
    faces = _faces(surface.faces, name="tangent-face source", vertex_count=len(positions))
    _require(surface.key[0] == 51011 and surface.key[1] > 0,
             "tangent-face proof requires a typed native vessel source")
    source_ids = np.asarray(surface.source_vertex_ids)
    _require(source_ids.shape == (len(positions),) and source_ids.dtype.kind in "iu",
             "tangent-face source vertex IDs are malformed")
    if hasattr(surface, "source_vertex_to_quotient"):
        source_to_quotient = np.asarray(surface.source_vertex_to_quotient)
        quotient_faces = np.asarray(surface.quotient_faces)
        _require(source_to_quotient.shape == (len(positions),)
                 and quotient_faces.shape == faces.shape
                 and np.array_equal(source_to_quotient[faces], quotient_faces),
                 "tangent-face source row mapping is not exact and ordered")
    center = np.asarray(station.get("section_center_world_m"), dtype=np.float64)
    tangent = np.asarray(station.get("tangent_world_unit"), dtype=np.float64)
    _require(center.shape == (3,) and np.isfinite(center).all()
             and tangent.shape == (3,) and np.isfinite(tangent).all()
             and float(np.linalg.norm(tangent)) > 0.0,
             "tangent-face station is malformed")
    tangent /= float(np.linalg.norm(tangent))
    signed_distances, signs, tolerance = _section_vertex_classes(
        positions.astype(np.float64), positions, center, tangent)
    face_signs = signs[faces]
    records = {}
    for face_id, (face, signs_for_face) in enumerate(zip(faces, face_signs)):
        zero = np.flatnonzero(signs_for_face == 0)
        if len(zero) != 1:
            continue
        nonzero = [int(signs_for_face[index]) for index in range(3) if index not in zero]
        if len(nonzero) != 2 or nonzero[0] == 0 or nonzero[0] != nonzero[1]:
            continue
        zero_index = int(zero[0])
        records[face_id] = {
            "face_id": face_id,
            "source_vertex_ids": [int(source_ids[index]) for index in face],
            "local_vertex_ids": [int(index) for index in face],
            "f32_vertex_signs": [int(value) for value in signs_for_face],
            "signed_plane_distances_m": [float(signed_distances[index]) for index in face],
            "zero_class_local_vertex_id": int(face[zero_index]),
            "zero_class_source_vertex_id": int(source_ids[int(face[zero_index])]),
            "zero_class_signed_distance_m": float(signed_distances[int(face[zero_index])]),
            "nonzero_side_sign": nonzero[0],
            "source_f32_zero_tolerance_m": tolerance,
        }
    return {
        "typed_identity": list(surface.key),
        "surface_binding": _source_geometry_binding(surface),
        "station_sha256": _canonical_record_sha256(dict(station)),
        "section_precision_contract": "shared _section_vertex_classes with source Float32 precision",
        "source_f32_zero_tolerance_m": tolerance,
        "source_tangent_face_records": records,
    }


def _exact_triangle_point_record(point, triangle, station):
    point = np.asarray(point, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    center = np.asarray(station["section_center_world_m"], dtype=np.float64)
    tangent = np.asarray(station["tangent_world_unit"], dtype=np.float64)
    tangent = tangent / float(np.linalg.norm(tangent))
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    drop_axis = int(np.argmax(np.abs(normal)))
    triangle_2d = np.delete(triangle, drop_axis, axis=1)
    point_2d = np.delete(point, drop_axis)
    denominator = _orient2d_exact(triangle_2d[0], triangle_2d[1], triangle_2d[2])
    _require(denominator != 0, "source-face barycentric proof found a degenerate triangle")
    weights = (
        _orient2d_exact(triangle_2d[1], triangle_2d[2], point_2d) / denominator,
        _orient2d_exact(triangle_2d[2], triangle_2d[0], point_2d) / denominator,
        _orient2d_exact(triangle_2d[0], triangle_2d[1], point_2d) / denominator,
    )
    inside_projection = all(weight >= 0 for weight in weights) and sum(weights) == 1
    f = Fraction.from_float
    tri_q = [[f(float(component)) for component in row] for row in triangle]
    point_q = [f(float(component)) for component in point]
    center_q = [f(float(component)) for component in center]
    tangent_q = [f(float(component)) for component in tangent]
    first_edge = [tri_q[1][axis] - tri_q[0][axis] for axis in range(3)]
    second_edge = [tri_q[2][axis] - tri_q[0][axis] for axis in range(3)]
    face_normal = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    residual_numerator = sum(
        (point_q[axis] - tri_q[0][axis]) * face_normal[axis]
        for axis in range(3))
    normal_squared = sum(value * value for value in face_normal)
    source_face_plane_residual_m = (
        float(abs(residual_numerator)) / float(normal_squared) ** 0.5)
    station_distance = sum(
        (point_q[axis] - center_q[axis]) * tangent_q[axis]
        for axis in range(3))
    station_plane_residual_m = abs(float(station_distance))
    # Reconstruct the exact same section-edge interpolation on this source
    # triangle. Its nonnegative barycentrics, not a spatial epsilon, establish
    # that the point belongs to the triangle face.
    signed_distances, _, _ = _section_vertex_classes(
        triangle, triangle.astype("<f4"), center, tangent)
    crossing_edges = [
        (index, (index + 1) % 3)
        for index in range(3)
        if signed_distances[index] * signed_distances[(index + 1) % 3] < 0.0
    ]
    interpolation_matches = []
    for first, second in crossing_edges:
        fraction = float(signed_distances[first] / (
            signed_distances[first] - signed_distances[second]))
        expected = triangle[first] + fraction * (triangle[second] - triangle[first])
        weights_from_edge = [0.0, 0.0, 0.0]
        weights_from_edge[first] = 1.0 - fraction
        weights_from_edge[second] = fraction
        interpolation_matches.append({
            "edge_corner_indices": [first, second],
            "edge_fraction": fraction,
            "source_face_barycentric_weights": weights_from_edge,
            "barycentric_is_inside_closed_face": bool(
                0.0 <= fraction <= 1.0 and sum(weights_from_edge) == 1.0),
            "reconstructed_section_point_world_m": expected.tolist(),
            "point_reconstruction_residual_inf_m": float(
                np.max(np.abs(point - expected))),
        })
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma_16 = (16.0 * unit_roundoff) / (1.0 - 16.0 * unit_roundoff)
    operation_scale_m = max(
        1.0, float(np.max(np.abs(triangle))), float(np.max(np.abs(point))),
        float(np.max(np.abs(center))))
    arithmetic_bound_m = float(np.sqrt(3.0) * gamma_16 * operation_scale_m)
    best_interpolation = min(
        interpolation_matches,
        key=lambda record: record["point_reconstruction_residual_inf_m"],
        default=None)
    construction_barycentric_inside = bool(
        len(crossing_edges) == 2 and best_interpolation is not None
        and best_interpolation["barycentric_is_inside_closed_face"]
        and best_interpolation["point_reconstruction_residual_inf_m"]
            <= arithmetic_bound_m)
    return {
        "barycentric_weights_exact_binary_rational": [str(value) for value in weights],
        "inside_source_triangle_projection_exact": bool(inside_projection),
        "source_face_plane_residual_m": source_face_plane_residual_m,
        "station_plane_residual_m": station_plane_residual_m,
        "source_edge_interpolation_candidates": interpolation_matches,
        "source_edge_barycentric_inclusion_within_binary64_construction_bound":
            construction_barycentric_inside,
        "binary64_edge_intersection_operation_bound_m": arithmetic_bound_m,
        "both_residuals_within_binary64_operation_bound": bool(
            source_face_plane_residual_m <= arithmetic_bound_m
            and station_plane_residual_m <= arithmetic_bound_m
            and construction_barycentric_inside),
    }


def _source_tangent_face_loop_proof(
    *, surface, station, source_loop_metric, union_loop_metric, owner_id, loop_index,
    expected_source_binding=None,
):
    classification = _source_tangent_face_classification(surface, station)
    if expected_source_binding is not None:
        _require(classification["surface_binding"] == expected_source_binding,
                 "tangent-face source surface differs from its independently pinned binding")
    source_cycle_raw = source_loop_metric.get("ordered_source_face_ancestor_cycle")
    union_cycle_raw = union_loop_metric.get("ordered_source_face_ancestor_cycle")
    source_cycle = _canonical_oriented_ancestor_cycle(
        source_cycle_raw, label="source section")
    union_cycle = _canonical_oriented_ancestor_cycle(
        union_cycle_raw, label="union section")
    source_ancestors = set(source_loop_metric.get("source_face_ancestor_ids", []))
    union_ancestors = set(union_loop_metric.get("source_face_ancestor_ids", []))
    _require(set(source_cycle) == source_ancestors
             and set(union_cycle) == union_ancestors,
             "ordered source-face cycles do not cover their complete loop ancestry")
    missing = sorted(source_ancestors - union_ancestors)
    extra = sorted(union_ancestors - source_ancestors)
    _require(not missing,
             "union section dropped a source face from its ordered branch loop")
    tangent_records = classification["source_tangent_face_records"]
    _require(set(extra) <= set(tangent_records),
             "union-only source ancestry contains a non-tangent source face")
    source_faces = np.asarray(surface.faces, dtype=np.int64)
    source_loop_vertices = set(map(int, np.unique(
        source_faces[np.asarray(sorted(source_ancestors), dtype=np.int64)].reshape(-1))))
    union_segments = union_loop_metric.get("ordered_face_segment_records")
    _require(isinstance(union_segments, list)
             and len(union_segments) == len(union_cycle),
             "union ordered section segment records are incomplete")
    segment_records_by_face = {}
    for segment in union_segments:
        lineages = segment.get("face_lineage")
        _require(isinstance(lineages, list) and bool(lineages),
                 "union segment has no source-face lineage")
        for lineage in lineages:
            _require(lineage.get("owner_id") == owner_id
                     and lineage.get("source_face_id") is not None
                     and lineage.get("generated") is False,
                     "union segment contains foreign, generated, or unmapped ownership")
            segment_records_by_face.setdefault(
                int(lineage["source_face_id"]), []).append((segment, lineage))
    loop_proofs = []
    for face_id in extra:
        source_face_record = tangent_records[face_id]
        local_face_ids = source_face_record["local_vertex_ids"]
        _require(bool(set(local_face_ids) & source_loop_vertices),
                 "source-tangent ancestor is disconnected from this source section loop")
        edge_records = segment_records_by_face.get(face_id, [])
        _require(len(edge_records) == 1,
                 "source-tangent ancestor does not map to exactly one union section edge")
        segment, lineage = edge_records[0]
        source_triangle = np.asarray(surface.positions, dtype=np.float64)[
            np.asarray(surface.faces[face_id], dtype=np.int64)]
        csg_triangle = np.asarray(
            lineage.get("triangle_corner_positions_world_m"), dtype=np.float64)
        _require(csg_triangle.shape == (3, 3) and np.isfinite(csg_triangle).all(),
                 "union tangent-face triangle coordinates are malformed")
        source_corner_tokens = sorted(tuple(float(value) for value in row)
                                      for row in source_triangle)
        csg_corner_tokens = sorted(tuple(float(value) for value in row)
                                   for row in csg_triangle)
        _require(csg_corner_tokens == source_corner_tokens,
                 "Mesh64 union face moved or quantized away from its exact source triangle")
        endpoints = segment.get("segment_endpoints_world_m")
        _require(isinstance(endpoints, list) and len(endpoints) == 2,
                 "union tangent-face section edge needs two endpoints")
        barycentric_records = [
            _exact_triangle_point_record(point, source_triangle, station)
            for point in endpoints]
        _require(all(record[
                         "source_edge_barycentric_inclusion_within_binary64_construction_bound"]
                     and record["both_residuals_within_binary64_operation_bound"]
                     for record in barycentric_records),
                 "Mesh64 tangent-face section edge is outside its exact source-face envelope")
        loop_proofs.append({
            "source_face_id": face_id,
            "source_vertex_ids": source_face_record["source_vertex_ids"],
            "source_f32_signs": source_face_record["f32_vertex_signs"],
            "source_zero_class_vertex_id": source_face_record["zero_class_source_vertex_id"],
            "source_zero_class_signed_distance_m": source_face_record[
                "zero_class_signed_distance_m"],
            "source_f32_zero_tolerance_m": source_face_record[
                "source_f32_zero_tolerance_m"],
            "source_triangle_positions_f32": source_triangle.tolist(),
            "mesh64_triangle_positions_exactly_match_source_f32_corners": True,
            "mesh64_section_edge_endpoints_world_m": endpoints,
            "endpoint_source_triangle_barycentric_proofs": barycentric_records,
        })
    filtered_union_cycle = [face_id for face_id in union_cycle if face_id not in set(extra)]
    filtered_union_canonical = _canonical_oriented_ancestor_cycle(
        filtered_union_cycle, label="union cycle after source-tangent canonicalization")
    _require(filtered_union_canonical == source_cycle,
             "source-tangent canonicalization changed ordered branch chain or direction")
    return {
        "passed": True,
        "schema": "numi.human.source-f32-tangent-face-canonicalization.v1",
        "typed_identity": list(surface.key),
        "surface_binding": classification["surface_binding"],
        "station_sha256": classification["station_sha256"],
        "loop_index": int(loop_index),
        "source_f32_zero_tolerance_m": classification["source_f32_zero_tolerance_m"],
        "source_section_ancestor_cycle": source_cycle,
        "union_section_ancestor_cycle": union_cycle,
        "canonicalized_union_section_ancestor_cycle": filtered_union_canonical,
        "tangent_face_ids_canonicalized": extra,
        "missing_source_face_ids": missing,
        "foreign_owner_or_generated_segment_count": 0,
        "source_tangent_face_proofs": loop_proofs,
        "all_face_geometry_is_exactly_source_bound": True,
        "anatomical_admission": False,
        "native_admission": False,
    }


def _anchor_only_tangent_canonicalization(
    *, surface, station, source_section, union_section, owner_id,
    expected_source_binding=None, label,
):
    source_loops = _complete_station_loop_metrics(
        source_section, owner_id, label=f"{label} source branch")
    union_loops = _complete_station_loop_metrics(
        union_section, owner_id, label=f"{label} union boundary")
    _require(len(source_loops) == len(union_loops),
             f"{label} source-tangent proof changed the number of section lobes")
    loop_proofs = [
        _source_tangent_face_loop_proof(
            surface=surface, station=station,
            source_loop_metric=source_loop, union_loop_metric=union_loop,
            owner_id=owner_id, loop_index=index,
            expected_source_binding=expected_source_binding)
        for index, (source_loop, union_loop)
        in enumerate(zip(source_loops, union_loops))
    ]
    return {
        "passed": True,
        "typed_identity": list(surface.key),
        "surface_binding": _source_geometry_binding(surface),
        "station_sha256": _canonical_record_sha256(dict(station)),
        "source_loop_count": len(source_loops),
        "union_loop_count": len(union_loops),
        "source_loop_face_counts": [
            len(item["source_face_ancestor_ids"]) for item in source_loops],
        "union_loop_face_counts": [
            len(item["source_face_ancestor_ids"]) for item in union_loops],
        "loop_proofs": loop_proofs,
    }


def _anchor_only_union_station_proof(
    *, applicability, candidate_applicability, candidate_union, baseline_union,
    candidate_owner_sections, baseline_owner_sections, expected_owner_ids,
    candidate_anchor_surface=None, baseline_anchor_surface=None,
    candidate_station=None, baseline_station=None, anchor_role=None,
    solid_evidence=None,
):
    """Compare every loop at a source-declared anchor-only cut, with no owner omission."""
    _require(applicability.get("applicability_mode") == "ANCHOR_ONLY_CUT"
             and applicability.get("independent_source_baseline_only") is True
             and applicability.get("candidate_geometry_used_for_declaration") is False,
             "anchor-only section needs an explicit source-baseline applicability proof")
    anchor = applicability.get("anchor_owner_id")
    expected = tuple(sorted(expected_owner_ids))
    _require(anchor in expected and set(candidate_owner_sections) == set(expected)
             and set(baseline_owner_sections) == set(expected),
             "anchor-only comparison does not bind both branch sources")
    _require(candidate_applicability.get("applicability_mode") == "ANCHOR_ONLY_CUT"
             and candidate_applicability.get("present_owner_ids") == [anchor]
             and candidate_applicability.get("absent_owner_ids")
                 == [owner for owner in expected if owner != anchor],
             "candidate station no longer has the declared anchor-only applicability")
    _require(applicability.get("present_owner_ids") == [anchor]
             and applicability.get("absent_owner_ids")
                 == [owner for owner in expected if owner != anchor],
             "baseline station does not prove exactly the declared anchor-only cut")
    absent_owner = next(owner for owner in expected if owner != anchor)
    _require(candidate_applicability["owner_status_by_id"][str(absent_owner)]["status"]
             == "ABSENT_STRICTLY_ON_ONE_SIDE"
             and applicability["owner_status_by_id"][str(absent_owner)]["status"]
                 == "ABSENT_STRICTLY_ON_ONE_SIDE",
             "the non-anchor vessel is not proven absent on one strict plane side")

    candidate_anchor = candidate_owner_sections[anchor]
    baseline_anchor = baseline_owner_sections[anchor]
    _require(_same_section_plane(candidate_union, candidate_anchor)
             and _same_section_plane(baseline_union, baseline_anchor),
             "anchor-only union and source branch sections are on different planes")
    _require(candidate_union.get("loop_count")
             == candidate_anchor.get("loop_count")
             and baseline_union.get("loop_count")
             == baseline_anchor.get("loop_count"),
             "anchor-only union omitted or added a source section loop")
    candidate_union_anchor = _complete_station_loop_metrics(
        candidate_union, anchor, label="candidate anchor-only union")
    baseline_union_anchor = _complete_station_loop_metrics(
        baseline_union, anchor, label="baseline anchor-only union")
    candidate_source_anchor = _complete_station_loop_metrics(
        candidate_anchor, anchor, label="candidate anchor-only source branch")
    baseline_source_anchor = _complete_station_loop_metrics(
        baseline_anchor, anchor, label="baseline anchor-only source branch")
    candidate_tangent_canonicalization = None
    baseline_tangent_canonicalization = None
    if any(value is not None for value in (
            candidate_anchor_surface, baseline_anchor_surface,
            candidate_station, baseline_station, anchor_role)):
        _require(candidate_anchor_surface is not None
                 and baseline_anchor_surface is not None
                 and candidate_station is not None and baseline_station is not None
                 and anchor_role in ("bridge", "target")
                 and candidate_anchor_surface.key == (51011, anchor)
                 and baseline_anchor_surface.key == (51011, anchor),
                 "source-tangent canonicalization requires both typed anchor surfaces and stations")
        baseline_binding = applicability.get(
            "baseline_source_surface_bindings", {}).get(anchor_role)
        _require(isinstance(baseline_binding, Mapping),
                 "source-tangent proof lacks the independent baseline surface binding")
        candidate_tangent_canonicalization = _anchor_only_tangent_canonicalization(
            surface=candidate_anchor_surface, station=candidate_station,
            source_section=candidate_anchor, union_section=candidate_union,
            owner_id=anchor, label="candidate anchor-only source-face ancestry")
        baseline_tangent_canonicalization = _anchor_only_tangent_canonicalization(
            surface=baseline_anchor_surface, station=baseline_station,
            source_section=baseline_anchor, union_section=baseline_union,
            owner_id=anchor, expected_source_binding=baseline_binding,
            label="baseline anchor-only source-face ancestry")
    if solid_evidence is not None:
        for side in ("candidate", "baseline"):
            witness = solid_evidence.get(f"{side}_exact_inside_both_witness")
            _require(solid_evidence.get(f"{side}_union_component_count") == 1
                     and str(solid_evidence.get(f"{side}_union_status", "")).endswith("NoError")
                     and isinstance(solid_evidence.get(f"{side}_union_volume_m3"), (int, float))
                     and solid_evidence[f"{side}_union_volume_m3"] > 0.0
                     and isinstance(solid_evidence.get(f"{side}_intersection_volume_m3"), (int, float))
                     and solid_evidence[f"{side}_intersection_volume_m3"] > 0.0
                     and isinstance(witness, Mapping)
                     and witness.get("first_location") == "inside"
                     and witness.get("second_location") == "inside",
                     f"{side} anchor-only source section lacks a connected inside-both solid union")
    candidate_union_vs_source = _compare_complete_anchor_only_loop_sets(
        candidate_union, candidate_anchor, owner_id=anchor,
        label="candidate anchor-only union/source", exact_ancestry=True,
        tangent_canonicalization=candidate_tangent_canonicalization)
    baseline_union_vs_source = _compare_complete_anchor_only_loop_sets(
        baseline_union, baseline_anchor, owner_id=anchor,
        label="baseline anchor-only union/source", exact_ancestry=True,
        tangent_canonicalization=baseline_tangent_canonicalization)
    union_cross_time = _compare_complete_anchor_only_loop_sets(
        candidate_union, baseline_union, owner_id=anchor,
        label="anchor-only union candidate/baseline")
    source_cross_time = _compare_complete_anchor_only_loop_sets(
        candidate_anchor, baseline_anchor, owner_id=anchor,
        label="anchor-only source candidate/baseline")
    return {
        "passed": True,
        "station_applicability_mode": "ANCHOR_ONLY_CUT",
        "method": "all-loop source-lineaged union and branch comparison with strict one-sided absence of the other owner",
        "source_baseline_applicability": applicability,
        "candidate_station_applicability": candidate_applicability,
        "anchor_owner_id": anchor,
        "absent_owner_id": absent_owner,
        "candidate_union_loop_count": len(candidate_union_anchor),
        "baseline_union_loop_count": len(baseline_union_anchor),
        "candidate_source_loop_count": len(candidate_source_anchor),
        "baseline_source_loop_count": len(baseline_source_anchor),
        "candidate_union_vs_source_loop_comparisons": candidate_union_vs_source,
        "baseline_union_vs_source_loop_comparisons": baseline_union_vs_source,
        "union_candidate_vs_baseline_loop_comparisons": union_cross_time,
        "source_candidate_vs_baseline_loop_comparisons": source_cross_time,
        "candidate_source_f32_tangent_canonicalization":
            candidate_tangent_canonicalization,
        "baseline_source_f32_tangent_canonicalization":
            baseline_tangent_canonicalization,
        "connected_inside_both_solid_evidence": (
            dict(solid_evidence) if solid_evidence is not None else None),
        "other_owner_remains_absent": True,
        "all_loops_have_complete_source_lineage": True,
    }


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
    anchor_role = declaration.get("anchor_role", role)
    ancestry = declaration.get("source_ancestry")
    if not isinstance(declaration_id, str) or not declaration_id.strip():
        issues.append("junction station declaration_id must be a nonempty string")
    if role not in ("bridge", "target", "junction"):
        issues.append("junction station role must be bridge, target, or junction")
    if anchor_role not in ("bridge", "target"):
        issues.append("junction station material anchor role must be bridge or target")
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
    expected_anchor_owner = (bridge_owner if anchor_role == "bridge"
                             else target_owner if anchor_role == "target" else None)
    if anchor_owner not in expected_owner_ids or anchor_owner != expected_anchor_owner:
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
                    or source_evidence.get("typed_link") != expected_link
                    or source_evidence.get("anchor_role", role) != anchor_role
                    or source_evidence.get("station_role", anchor_role) != role):
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


def _shared_cut_mixed_union_loop_proof(
    *,
    expected_owner_ids,
    candidate_union_section,
    baseline_union_section,
    candidate_branch_sections,
    baseline_branch_sections,
    solid_evidence,
    station_declaration,
):
    """Prove a shared source cut whose connected union has one mixed-owner loop.

    A single mixed-owner loop is not split into fabricated per-vessel boundary
    loops. Both complete source branch sections, the exact positive-volume and
    inside-both witnesses, the connected closed union, and same-station calibre
    comparisons remain mandatory. A one-owner union boundary is a distinct
    nested-section state and is rejected until a separate containment proof exists.
    """
    expected = tuple(sorted(set(map(int, expected_owner_ids))))
    issues = []
    station_evidence = _junction_station_evidence(
        station_declaration, expected,
        candidate_union_section, baseline_union_section,
        candidate_branch_sections, baseline_branch_sections, solid_evidence)
    if not station_evidence.get("declared"):
        issues.extend(station_evidence.get("rejections", []))
    if len(expected) != 2:
        issues.append("shared-cut mixed-loop proof needs exactly two stable owners")

    def one_loop(section, *, label, owner_ids):
        loops = section.get("loop_metrics")
        if (section.get("passed") is not True
                or section.get("loop_count") != 1
                or not isinstance(loops, list) or len(loops) != 1):
            issues.append(f"{label} must retain exactly one closed loop")
            return None
        metric = loops[0]
        if (metric.get("passed") is not True
                or metric.get("lineage_available") is not True
                or metric.get("lineage_complete") is not True
                or metric.get("owner_ids") != list(owner_ids)
                or not metric.get("source_face_ancestor_ids")
                or metric.get("generated_face_count") != 0
                or metric.get("unmapped_face_count") != 0
                or not isinstance(metric.get("area_m2"), (int, float))
                or not np.isfinite(metric["area_m2"]) or metric["area_m2"] <= 0.0
                or not isinstance(metric.get("minimum_feret_width_m"), (int, float))
                or not np.isfinite(metric["minimum_feret_width_m"])
                or metric["minimum_feret_width_m"] <= 0.0):
            issues.append(
                f"{label} loop must have complete positive lineage from both source owners")
            return None
        return metric

    candidate_union = one_loop(
        candidate_union_section, label="candidate shared union",
        owner_ids=expected)
    baseline_union = one_loop(
        baseline_union_section, label="baseline shared union",
        owner_ids=expected)
    candidate_branches = {}
    baseline_branches = {}
    for owner_id in expected:
        candidate_branch = candidate_branch_sections.get(owner_id)
        baseline_branch = baseline_branch_sections.get(owner_id)
        if not isinstance(candidate_branch, Mapping):
            issues.append(f"candidate source section is missing for owner {owner_id}")
            continue
        if not isinstance(baseline_branch, Mapping):
            issues.append(f"baseline source section is missing for owner {owner_id}")
            continue
        candidate_metric = _section_metric_for_owner(
            candidate_branch, owner_id, label=f"candidate source section {owner_id}")[0]
        baseline_metric = _section_metric_for_owner(
            baseline_branch, owner_id, label=f"baseline source section {owner_id}")[0]
        if candidate_metric is None or baseline_metric is None:
            issues.append(f"owner {owner_id} lacks a complete single source section")
            continue
        candidate_branches[owner_id] = candidate_metric
        baseline_branches[owner_id] = baseline_metric

    for side in ("candidate", "baseline"):
        intersection_volume = solid_evidence.get(f"{side}_intersection_volume_m3")
        union_volume = solid_evidence.get(f"{side}_union_volume_m3")
        union_count = solid_evidence.get(f"{side}_union_component_count")
        witness = solid_evidence.get(f"{side}_exact_inside_both_witness")
        if (not isinstance(intersection_volume, (int, float))
                or not np.isfinite(intersection_volume) or intersection_volume <= 0.0
                or not isinstance(union_volume, (int, float))
                or not np.isfinite(union_volume) or union_volume <= 0.0
                or union_count != 1
                or not isinstance(witness, Mapping)
                or witness.get("first_location") != "inside"
                or witness.get("second_location") != "inside"):
            issues.append(
                f"{side} shared junction needs positive overlap, exact inside-both witness, and one connected union")
    if candidate_union is not None and baseline_union is not None:
        union_comparison = _section_comparison(candidate_union, baseline_union)
        if not union_comparison["passed"]:
            issues.append("shared mixed-owner union calibre changed across the source state")
    else:
        union_comparison = None
    branch_comparisons = {}
    for owner_id in expected:
        if owner_id not in candidate_branches or owner_id not in baseline_branches:
            continue
        comparison = _section_comparison(
            candidate_branches[owner_id], baseline_branches[owner_id])
        branch_comparisons[str(owner_id)] = comparison
        if not comparison["passed"]:
            issues.append(
                f"owner {owner_id} shared-station section calibre changed across the source state")

    return {
        "passed": not issues,
        "status": ("PASS_SHARED_TWO_OWNER_MIXED_UNION_LOOP"
                   if not issues else "REJECT_SHARED_TWO_OWNER_MIXED_UNION_LOOP"),
        "method": (
            "one closed mixed-owner union loop, complete source section for each branch, "
            "positive intersection, exact inside-both witnesses, connected union, "
            "and unchanged same-station area/minimum-Feret bounds"),
        "scope": (
            "only the independently source-declared shared station; no owner loop is "
            "fabricated and no nested one-owner union boundary is accepted"),
        "station_applicability_mode": "SHARED_TWO_OWNER_CUT",
        "junction_station": station_evidence,
        "candidate_union_loop": candidate_union,
        "baseline_union_loop": baseline_union,
        "candidate_source_sections_by_owner": {
            str(owner): metric for owner, metric in sorted(candidate_branches.items())},
        "baseline_source_sections_by_owner": {
            str(owner): metric for owner, metric in sorted(baseline_branches.items())},
        "shared_union_calibre_comparison": union_comparison,
        "source_branch_calibre_comparisons": branch_comparisons,
        "source_pair_and_solid_evidence": dict(solid_evidence),
        "rejections": issues,
        "anatomical_admission": False,
        "native_admission": False,
    }


def _validated_same_material_station_calibre_by_owner(
    value, *, expected_owner_ids, station_declaration, solid_evidence,
):
    """Validate owner-bound intrinsic calibre proofs for cross-time comparison."""
    issues = []
    expected = tuple(sorted(expected_owner_ids))
    if not isinstance(value, Mapping) or set(value) != set(expected):
        return {}, [
            "same-material-station calibre evidence must cover exactly both vessel owners"]
    bridge_owner = station_declaration.get("bridge_owner_id")
    target_owner = station_declaration.get("target_owner_id")
    role_owner = {"bridge": bridge_owner, "target": target_owner}
    if set(role_owner.values()) != set(expected):
        issues.append("station declaration does not map both material-section roles")
    ancestry_keys = (
        "material_anchor_source_vertex_ids",
        "tangent_support_source_vertex_ids",
        "frozen_collar_face_ids",
    )
    expected_bounds = {"area": [0.8, 1.25], "minimum_feret_width": [0.9, 1.1]}
    baseline_bindings = solid_evidence.get("baseline_source_surface_bindings")
    normalized = {}
    for owner_id in expected:
        entry = value.get(owner_id)
        if not isinstance(entry, Mapping):
            issues.append(f"same-material-station proof for owner {owner_id} is missing")
            continue
        role = entry.get("station_role")
        if role not in ("bridge", "target") or role_owner.get(role) != owner_id:
            issues.append(f"same-material-station proof has a wrong role for owner {owner_id}")
        declaration_id = entry.get("source_station_declaration_id")
        if not isinstance(declaration_id, str) or not declaration_id.strip():
            issues.append(f"same-material-station proof has no declaration id for owner {owner_id}")
        ancestry = entry.get("source_station_ancestry")
        if not isinstance(ancestry, Mapping):
            issues.append(f"same-material-station proof has no source ancestry for owner {owner_id}")
            ancestry = {}
        normalized_ancestry = {}
        for key in ancestry_keys:
            ids = ancestry.get(key)
            if (not isinstance(ids, (list, tuple)) or not ids
                    or any(type(item) is not int or item < 0 for item in ids)):
                issues.append(
                    f"same-material-station source ancestry {key} is invalid for owner {owner_id}")
                normalized_ancestry[key] = []
            else:
                normalized_ancestry[key] = list(ids)

        proof = entry.get("transport_proof")
        if not isinstance(proof, Mapping):
            issues.append(f"same-material-station transport proof is missing for owner {owner_id}")
            continue
        if (proof.get("schema") != "numi.human.source-material-section-transport.v1"
                or proof.get("passed") is not True
                or proof.get("typed_identity") != [51011, owner_id]
                or proof.get("source_station_ancestry_sha256")
                    != _canonical_record_sha256(normalized_ancestry)
                or proof.get("candidate_geometry_used_only_for_local_motion_fit") is not True
                or proof.get("all_exact_coordinate_aliases_preserved") is not True
                or proof.get("anatomical_admission") is not False
                or proof.get("native_admission") is not False):
            issues.append(f"same-material-station transport binding failed for owner {owner_id}")

        baseline_binding = proof.get("baseline_surface_binding")
        candidate_binding = proof.get("candidate_surface_binding")
        declared_baseline = entry.get("baseline_surface_binding")
        declared_candidate = entry.get("candidate_surface_binding")
        for label, binding in (("baseline", baseline_binding), ("candidate", candidate_binding)):
            if (not isinstance(binding, Mapping)
                    or binding.get("identity") != [51011, owner_id]
                    or any(not isinstance(binding.get(name), str)
                           or len(binding[name]) != 64
                           or any(ch not in "0123456789abcdef" for ch in binding[name])
                           for name in ("positions_sha256", "faces_sha256",
                                        "source_vertex_ids_sha256"))):
                issues.append(
                    f"same-material-station {label} surface binding is invalid for owner {owner_id}")
        if baseline_binding != declared_baseline or candidate_binding != declared_candidate:
            issues.append(f"same-material-station surfaces differ from current owner {owner_id}")
        if (baseline_binding is not None and isinstance(baseline_bindings, Mapping)
                and baseline_binding != baseline_bindings.get(role)):
            issues.append(f"same-material-station baseline differs from junction source {owner_id}")
        if (isinstance(baseline_binding, Mapping)
                and isinstance(candidate_binding, Mapping)
                and (baseline_binding.get("faces_sha256")
                     != candidate_binding.get("faces_sha256")
                     or baseline_binding.get("source_vertex_ids_sha256")
                     != candidate_binding.get("source_vertex_ids_sha256"))):
            issues.append(f"same-material-station ancestry changed for owner {owner_id}")

        calibre = proof.get("material_section_calibre")
        if not isinstance(calibre, Mapping):
            issues.append(f"same-material-station calibre record is missing for owner {owner_id}")
            continue
        if (calibre.get("passed") is not True
                or calibre.get("declared_preservation_intervals") != expected_bounds
                or calibre.get("declared_maximum_warp_to_feret_ratio") != 0.01
                or calibre.get("source_material_ring_is_simple_closed") is not True
                or calibre.get("candidate_material_ring_is_simple_closed") is not True
                or calibre.get("source_primary_simple_polygon_triangulation_passed") is not True
                or calibre.get("source_alternate_simple_polygon_triangulation_passed") is not True
                or calibre.get("candidate_primary_simple_polygon_triangulation_passed") is not True
                or calibre.get("candidate_alternate_simple_polygon_triangulation_passed") is not True
                or calibre.get("candidate_triangulations_preserve_candidate_ring_orientation") is not True
                or calibre.get("source_and_candidate_ring_orientation_match") is not True
                or calibre.get("candidate_area_triangulations_are_measurement_only") is not True
                or calibre.get("virtual_disk_area_not_physical_tissue_area") is not True
                or calibre.get("clinical_calibre_claim") is not False):
            issues.append(f"same-material-station calibre invariants failed for owner {owner_id}")
        ratio_fields = (
            "source_primary_virtual_disk_area_ratio",
            "source_alternate_virtual_disk_area_ratio",
            "candidate_projected_area_ratio",
            "candidate_primary_virtual_disk_area_ratio",
            "candidate_alternate_virtual_disk_area_ratio",
        )
        for key in ratio_fields:
            ratio = calibre.get(key)
            if (not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
                    or not np.isfinite(ratio) or not 0.8 <= ratio <= 1.25):
                issues.append(f"same-material-station {key} is outside bounds for owner {owner_id}")
        width_interval = calibre.get("minimum_feret_width_ratio_interval")
        if (not isinstance(width_interval, (list, tuple)) or len(width_interval) != 2
                or any(not isinstance(item, (int, float)) or isinstance(item, bool)
                       or not np.isfinite(item) for item in width_interval)
                or width_interval[0] < 0.9 or width_interval[1] > 1.1):
            issues.append(f"same-material-station width interval is outside bounds for owner {owner_id}")
        warp = calibre.get("maximum_warp_to_source_minimum_feret_ratio")
        if (not isinstance(warp, (int, float)) or isinstance(warp, bool)
                or not np.isfinite(warp) or warp > 0.01):
            issues.append(f"same-material-station warp exceeds its method bound for owner {owner_id}")
        normalized[owner_id] = {
            "station_role": role,
            "source_station_declaration_id": declaration_id,
            "source_station_ancestry": normalized_ancestry,
            "transport_proof": dict(proof),
        }
    return normalized, issues


def _junction_aware_two_loop_section_proof(
    *,
    expected_owner_ids,
    candidate_union_section,
    baseline_union_section,
    candidate_branch_sections,
    baseline_branch_sections,
    solid_evidence,
    station_declaration=None,
    same_material_station_calibre_by_owner=None,
    cross_time_calibre_policy=None,
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
    material_policy_active = (
        cross_time_calibre_policy == "SOURCE_MATERIAL_SECTION_FRAME_V1")
    same_material_calibre = {}
    if material_policy_active:
        if not isinstance(station_declaration, Mapping):
            issues.append("same-material-station calibre needs an explicit station declaration")
        else:
            same_material_calibre, material_issues = (
                _validated_same_material_station_calibre_by_owner(
                    same_material_station_calibre_by_owner,
                    expected_owner_ids=expected_owner_ids,
                    station_declaration=station_declaration,
                    solid_evidence=solid_evidence))
            issues.extend(material_issues)
    elif cross_time_calibre_policy is not None:
        issues.append("unsupported cross-time calibre measurement policy")
    elif same_material_station_calibre_by_owner is not None:
        issues.append("same-material-station proofs require their explicit measurement policy")
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
            if material_policy_active:
                comparison = {
                    **comparison,
                    "measurement_scope": "CROSS_OWNER_PLANE_RESLICE_DIAGNOSTIC_ONLY",
                    "accepted_as_intrinsic_material_calibre": False,
                    "reason": (
                        "the shared junction plane belongs to one source vessel; "
                        "cross-time lobe change is evaluated at each owner's "
                        "authenticated material station instead"),
                }
            comparisons.setdefault(str(owner_id), {})[
                "candidate_union_lobe_vs_baseline_union_lobe"] = comparison
            if not comparison["passed"] and not material_policy_active:
                issues.append(
                    f"candidate union lobe calibre changed for vessel {owner_id}")
    if set(candidate_branches) == set(expected_owner_ids) and \
            set(baseline_branches) == set(expected_owner_ids):
        for owner_id in expected_owner_ids:
            comparison = _section_comparison(
                candidate_branches[owner_id], baseline_branches[owner_id])
            if material_policy_active:
                comparison = {
                    **comparison,
                    "measurement_scope": "CROSS_OWNER_PLANE_RESLICE_DIAGNOSTIC_ONLY",
                    "accepted_as_intrinsic_material_calibre": False,
                    "reason": (
                        "this branch section is measured on the other vessel's "
                        "junction plane; intrinsic change is checked at its own "
                        "authenticated material station"),
                }
            comparisons.setdefault(str(owner_id), {})[
                "candidate_source_section_vs_baseline_source_section"] = comparison
            if not comparison["passed"] and not material_policy_active:
                issues.append(
                    f"candidate source-section calibre changed for vessel {owner_id}")
    if same_material_calibre:
        for owner_id, record in same_material_calibre.items():
            proof = record["transport_proof"]
            calibre = proof["material_section_calibre"]
            comparisons.setdefault(str(owner_id), {})[
                "candidate_vs_baseline_at_owner_material_station"] = {
                    "passed": proof.get("passed") is True
                    and calibre.get("passed") is True,
                    "measurement_scope": "AUTHENTICATED_SAME_MATERIAL_STATION",
                    "station_role": record["station_role"],
                    "source_station_declaration_id": record[
                        "source_station_declaration_id"],
                    "typed_identity": proof.get("typed_identity"),
                    "source_station_ancestry_sha256": proof.get(
                        "source_station_ancestry_sha256"),
                    "candidate_to_baseline_ratios": {
                        "projected_area": calibre.get(
                            "candidate_projected_area_ratio"),
                        "minimum_feret_width_interval": calibre.get(
                            "minimum_feret_width_ratio_interval"),
                    },
                    "declared_preservation_intervals": calibre.get(
                        "declared_preservation_intervals"),
                }

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
        "cross_time_calibre_policy": cross_time_calibre_policy,
        "same_material_station_calibre_by_owner": {
            str(owner): {
                "station_role": record["station_role"],
                "source_station_declaration_id": record[
                    "source_station_declaration_id"],
                "transport_proof": record["transport_proof"],
            } for owner, record in sorted(same_material_calibre.items())},
        "rejections": issues,
        "floating_csg_topology_limit": "Manifold NoError and decompose() are numerical CSG diagnostics, not exact Boolean topology proof",
        "calibre_scope": "source-geometry consistency against the same compiled branch lineage, not measured or clinical calibre",
    }


def _strict_polygon_containment2d(inner_points, outer_points):
    """Classify one simple polygon strictly inside another with exact predicates."""
    inner = np.asarray(inner_points, dtype=np.float64)
    outer = np.asarray(outer_points, dtype=np.float64)
    if not (_simple_closed_polygon2d_exact(inner)
            and _simple_closed_polygon2d_exact(outer)):
        return {"passed": False, "reason": "both section loops must be simple closed polygons"}
    crossings = []
    for first in range(len(inner)):
        first_next = (first + 1) % len(inner)
        for second in range(len(outer)):
            second_next = (second + 1) % len(outer)
            if _segments_intersect2d_exact(
                    inner[first], inner[first_next], outer[second], outer[second_next]):
                crossings.append([first, second])
    if crossings:
        return {
            "passed": False,
            "reason": "nested section boundaries touch or cross",
            "boundary_contact_or_crossing_edge_pairs": crossings,
        }

    def inside(point, polygon):
        px, py = (Fraction.from_float(float(value)) for value in point)
        is_inside = False
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            ax, ay = (Fraction.from_float(float(value)) for value in first)
            bx, by = (Fraction.from_float(float(value)) for value in second)
            orientation = _orient2d_exact(first, second, point)
            if orientation == 0 and _point_on_segment2d_exact(point, first, second):
                return False
            if (ay > py) != (by > py):
                x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
                if x_cross == px:
                    return False
                if x_cross > px:
                    is_inside = not is_inside
        return is_inside

    point_results = [inside(point, outer) for point in inner]
    return {
        "passed": bool(point_results) and all(point_results),
        "reason": None if point_results and all(point_results)
            else "not every nested owner section vertex is strictly inside the union boundary",
        "inner_vertex_count": len(inner),
        "strictly_inside_vertex_count": sum(point_results),
        "boundary_contact_or_crossing_edge_pairs": [],
        "uses_exact_float64_rational_orientation_predicates": True,
    }


def _shared_cut_nested_union_loop_proof(
    *, expected_owner_ids, candidate_union_section, baseline_union_section,
    candidate_branch_sections, baseline_branch_sections, solid_evidence,
    station_declaration,
):
    """Prove nested sections without inventing an owner loop on the union boundary."""
    expected = tuple(sorted(set(map(int, expected_owner_ids))))
    issues = []
    station_evidence = _junction_station_evidence(
        station_declaration, expected, candidate_union_section,
        baseline_union_section, candidate_branch_sections,
        baseline_branch_sections, solid_evidence)
    if not station_evidence.get("declared"):
        issues.extend(station_evidence.get("rejections", []))
    if len(expected) != 2:
        issues.append("nested shared-cut proof needs exactly two stable owners")

    def one_loop(section, label):
        loops = section.get("loop_metrics")
        if (section.get("passed") is not True or section.get("loop_count") != 1
                or not isinstance(loops, list) or len(loops) != 1):
            issues.append(f"{label} must retain one closed union/owner loop")
            return None
        metric = loops[0]
        owners = metric.get("owner_ids")
        points = metric.get("projected_loop_vertices")
        if (metric.get("passed") is not True
                or metric.get("lineage_available") is not True
                or metric.get("lineage_complete") is not True
                or not isinstance(owners, list) or len(owners) != 1
                or owners[0] not in expected
                or not metric.get("source_face_ancestor_ids")
                or metric.get("generated_face_count") != 0
                or metric.get("unmapped_face_count") != 0
                or not isinstance(points, list) or len(points) < 3):
            issues.append(f"{label} lacks one complete single-owner source loop")
            return None
        return metric

    candidate_union = one_loop(candidate_union_section, "candidate union boundary")
    baseline_union = one_loop(baseline_union_section, "baseline union boundary")
    if candidate_union is None or baseline_union is None:
        union_owner = None
    elif candidate_union["owner_ids"] != baseline_union["owner_ids"]:
        issues.append("candidate and baseline nested union boundary owners differ")
        union_owner = None
    else:
        union_owner = int(candidate_union["owner_ids"][0])

    branch_metrics = {"candidate": {}, "baseline": {}}
    for state, sections in (("candidate", candidate_branch_sections),
                            ("baseline", baseline_branch_sections)):
        for owner_id in expected:
            section = sections.get(owner_id)
            if not isinstance(section, Mapping):
                issues.append(f"{state} complete owner section is missing for {owner_id}")
                continue
            metric, reason = _section_metric_for_owner(
                section, owner_id, label=f"{state} nested owner section {owner_id}")
            if metric is None:
                issues.append(reason or f"{state} owner section {owner_id} is invalid")
                continue
            points = metric.get("projected_loop_vertices")
            if not isinstance(points, list) or len(points) < 3:
                issues.append(f"{state} owner section {owner_id} lacks ordered boundary points")
                continue
            branch_metrics[state][owner_id] = metric

    containment = {}
    branch_comparisons = {}
    union_comparison = None
    outer_correspondence = {}
    if (union_owner is not None
            and set(branch_metrics["candidate"]) == set(expected)
            and set(branch_metrics["baseline"]) == set(expected)):
        nested_owner = next(owner for owner in expected if owner != union_owner)
        for state, union_metric, metrics in (
            ("candidate", candidate_union, branch_metrics["candidate"]),
            ("baseline", baseline_union, branch_metrics["baseline"]),
        ):
            containment[state] = _strict_polygon_containment2d(
                metrics[nested_owner]["projected_loop_vertices"],
                union_metric["projected_loop_vertices"])
            if not containment[state]["passed"]:
                issues.append(f"{state} nested owner section is not strictly inside the union boundary")
            outer_correspondence[state] = _section_comparison(
                metrics[union_owner], union_metric)
            if not outer_correspondence[state]["passed"]:
                issues.append(f"{state} union boundary does not match outer owner section calibre")
            outer_ancestors = set(metrics[union_owner].get("source_face_ancestor_ids", []))
            union_ancestors = set(union_metric.get("source_face_ancestor_ids", []))
            if not union_ancestors or not union_ancestors <= outer_ancestors:
                issues.append(f"{state} union boundary ancestry is not contained in outer owner section ancestry")
        for owner_id in expected:
            branch_comparisons[str(owner_id)] = _section_comparison(
                branch_metrics["candidate"][owner_id],
                branch_metrics["baseline"][owner_id])
            if not branch_comparisons[str(owner_id)]["passed"]:
                issues.append(f"owner {owner_id} nested section calibre changed")
        union_comparison = _section_comparison(candidate_union, baseline_union)
        if not union_comparison["passed"]:
            issues.append("nested union boundary calibre changed")
    else:
        nested_owner = None
        issues.append("candidate and baseline need complete single sections for both owners")

    for side in ("candidate", "baseline"):
        intersection_volume = solid_evidence.get(f"{side}_intersection_volume_m3")
        union_volume = solid_evidence.get(f"{side}_union_volume_m3")
        witness = solid_evidence.get(f"{side}_exact_inside_both_witness")
        if (not isinstance(intersection_volume, (int, float))
                or not np.isfinite(intersection_volume) or intersection_volume <= 0.0
                or not isinstance(union_volume, (int, float))
                or not np.isfinite(union_volume) or union_volume <= 0.0
                or solid_evidence.get(f"{side}_union_component_count") != 1
                or not isinstance(witness, Mapping)
                or witness.get("first_location") != "inside"
                or witness.get("second_location") != "inside"):
            issues.append(
                f"{side} nested port needs positive overlap, exact inside-both witness, and one connected union")

    return {
        "passed": not issues,
        "status": (
            "PASS_NESTED_SINGLE_OWNER_UNION_WITH_COMPLETE_INNER_SECTION"
            if not issues else "REJECT_NESTED_SINGLE_OWNER_UNION_SECTION"),
        "method": (
            "one source-lineaged union boundary, complete source sections for both owners, "
            "strict exact 2D containment, positive overlap, inside-both witness, "
            "connected union, and unchanged section calibre bounds"),
        "station_applicability_mode": "SHARED_TWO_OWNER_CUT",
        "union_boundary_owner_id": union_owner,
        "nested_owner_id": nested_owner,
        "junction_station": station_evidence,
        "candidate_union_loop": candidate_union,
        "baseline_union_loop": baseline_union,
        "nested_section_containment": containment,
        "outer_owner_union_calibre_correspondence": outer_correspondence,
        "source_branch_calibre_comparisons": branch_comparisons,
        "shared_union_calibre_comparison": union_comparison,
        "source_pair_and_solid_evidence": dict(solid_evidence),
        "rejections": issues,
        "fabricated_owner_loops": False,
        "anatomical_admission": False,
        "native_admission": False,
    }


def audit_brachiocephalic_port_pair_continuity(
    verifier, manifold3d, *, bridge_key, bridge_surface, target_key, target_surface,
    frozen_face_pairs, baseline_bridge_surface, baseline_target_surface,
    bridge_station=None, target_station=None, bridge_route_anchor_world_m=None,
    target_route_anchor_world_m=None,
    junction_station_declarations=None,
    source_station_declarations=None,
    source_station_applicability_by_role=None,
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
             and set(source_station_declarations) <= {"bridge", "target", "junction"},
             "source station declarations must be keyed by branch role or junction")
    if source_station_applicability_by_role is None:
        source_station_applicability_by_role = {}
    _require(isinstance(source_station_applicability_by_role, Mapping)
             and set(source_station_applicability_by_role)
                 <= {"bridge", "target", "junction"},
             "source station applicability must be keyed by branch role or junction")
    _require(candidate_station_transport_policy in (
                 None, "SOURCE_LOCAL_PROPER_RIGID_FIT_V1",
                 "SOURCE_MATERIAL_SECTION_FRAME_V1"),
             "unknown candidate source-station transport policy")
    required_station_roles = {"bridge", "target"}
    if "junction" in source_station_declarations:
        required_station_roles.add("junction")
    if candidate_station_transport_policy is not None:
        _require(set(source_station_declarations) == required_station_roles,
                 "source-station transport needs every declared branch and junction station")
    if candidate_station_transport_policy == "SOURCE_MATERIAL_SECTION_FRAME_V1":
        _require(set(source_station_applicability_by_role) == required_station_roles,
                 "material-section transport needs explicit source-baseline applicability for every station")
    elif source_station_applicability_by_role:
        _require(set(source_station_declarations) >= required_station_roles
                 and set(source_station_applicability_by_role)
                     <= set(source_station_declarations),
                 "source-baseline station applicability needs matching declared stations")
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
    for role in ("bridge", "target", "junction"):
        declaration = source_station_declarations.get(role)
        if declaration is None:
            continue
        _require(isinstance(declaration, Mapping),
                 f"{role} source station declaration must be an object")
        anchor_role = declaration.get("anchor_role", role)
        _require(role in ("bridge", "target", "junction")
                 and anchor_role in ("bridge", "target")
                 and (role != "junction" or declaration.get("station_role") == "junction"),
                 f"{role} source station has an invalid material anchor role")
        surface, base_surface, patch_faces = (
            (bridge, base_bridge, bridge_patch)
            if anchor_role == "bridge"
            else (target, base_target, target_patch))
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
                 and declaration.get("anchor_role", role) == anchor_role
                 and declaration.get("anchor_owner_id")
                     == (bridge_key[1] if anchor_role == "bridge" else target_key[1]),
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
                     "GLOBAL_SOURCE_PCA_V1", "COLLAR_LOCAL_PCA_V1",
                     "SOURCE_INTERFACE_PCA_PLANE_V1"),
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
                 and source_evidence.get("anchor_role", role) == anchor_role
                 and source_evidence.get("station_role", anchor_role) == role,
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
        if source_evidence.get("station_selection_policy") == "SOURCE_INTERFACE_PCA_PLANE_V1":
            declared_sections = source_evidence.get(
                "baseline_shared_source_station_sections")
            _require(isinstance(declared_sections, Mapping)
                     and set(declared_sections) == {
                         str(bridge_key[1]), str(target_key[1])},
                     f"{role} source-interface declaration must retain both owner sections")
            for branch_surface, owner_id in (
                (base_bridge, bridge_key[1]), (base_target, target_key[1])):
                actual_section = _source_surface_section(branch_surface, source_spec)
                _require(actual_section.get("passed") is True
                         and actual_section.get("loop_count") == 1
                         and len(actual_section.get("loop_metrics", [])) == 1
                         and actual_section["loop_metrics"][0].get("lineage_complete") is True
                         and actual_section["loop_metrics"][0].get("owner_ids") == [owner_id]
                         and declared_sections.get(str(owner_id)) == actual_section,
                         f"{role} baseline source-interface section differs for owner {owner_id}")
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
            "anchor_role": anchor_role,
            "expected_owner_ids": sorted((bridge_key[1], target_key[1])),
            "bridge_owner_id": bridge_key[1],
            "target_owner_id": target_key[1],
            "anchor_owner_id": bridge_key[1] if anchor_role == "bridge" else target_key[1],
            "candidate_station_transport_policy": candidate_station_transport_policy,
            "source_ancestry": expected_station_ancestry,
            "source_station_spec": dict(source_spec),
            "candidate_station": candidate_spec,
            "baseline_station": baseline_spec,
            "candidate_station_transport_proof": candidate_station_transport_proof,
            "source_evidence": dict(source_evidence),
        }

    runtime_source_station_applicability = {}
    for role, declaration in source_station_applicability_by_role.items():
        source_declaration = runtime_source_station_declarations.get(role)
        _require(isinstance(source_declaration, Mapping),
                 f"{role} station applicability lacks its source station declaration")
        anchor_owner = source_declaration["anchor_owner_id"]
        recomputed_applicability = declare_source_baseline_station_applicability(
            verifier,
            bridge_key=bridge_key,
            baseline_bridge_surface=base_bridge,
            target_key=target_key,
            baseline_target_surface=base_target,
            station_role=role,
            anchor_owner_id=anchor_owner,
            source_station=source_declaration["source_station_spec"])
        _require(declaration == recomputed_applicability,
                 f"{role} station applicability differs from the current source baseline")
        runtime_source_station_applicability[role] = recomputed_applicability

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
    station_inputs = [
        ("bridge", bridge, base_bridge, bridge_station, bridge_patch,
         bridge_route_anchor_world_m),
        ("target", target, base_target, target_station, target_patch,
         target_route_anchor_world_m),
    ]
    if "junction" in runtime_source_station_declarations:
        junction_anchor_role = runtime_source_station_declarations[
            "junction"]["anchor_role"]
        station_inputs.append((
            "junction",
            bridge if junction_anchor_role == "bridge" else target,
            base_bridge if junction_anchor_role == "bridge" else base_target,
            None,
            bridge_patch if junction_anchor_role == "bridge" else target_patch,
            None))
    for role, surface, base_surface, supplied_station, collar_faces, route_anchor in station_inputs:
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
    for role, surface, base_surface, *_rest in station_inputs:
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
            candidate_section = _source_surface_section(
                surface, spec,
                include_projected_loop_vertices=(role == "junction"),
                include_face_segment_records=True)
            baseline_section = _source_surface_section(
                base_surface, baseline_spec,
                include_projected_loop_vertices=(role == "junction"),
                include_face_segment_records=True)
        station_results[role] = {
            "station": spec,
            "baseline_station_transported_by_ancestry": baseline_spec,
            "source_station_declaration": source_declaration,
            "candidate_station_transport_proof": (
                source_declaration.get("candidate_station_transport_proof")
                if source_declaration is not None else None),
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
        face_generated_mask=union_lineage["generated_face_mask"],
        include_face_segment_records=True)
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
        face_generated_mask=base_union_lineage["generated_face_mask"],
        include_face_segment_records=True)
    union_target_section = _section_loops(
        union_positions, union_faces,
        target_station["section_center_world_m"], target_station["tangent_world_unit"],
        derived_csg=True,
        face_owner_ids=union_lineage["face_owner_ids"],
        face_source_ids=union_lineage["source_face_ids"],
        face_generated_mask=union_lineage["generated_face_mask"],
        include_face_segment_records=True)
    baseline_union_target_section = _section_loops(
        base_union_positions, base_union_faces,
        base_target_station["section_center_world_m"],
        base_target_station["tangent_world_unit"], derived_csg=True,
        face_owner_ids=base_union_lineage["face_owner_ids"],
        face_source_ids=base_union_lineage["source_face_ids"],
        face_generated_mask=base_union_lineage["generated_face_mask"],
        include_face_segment_records=True)
    if "junction" in station_specs_by_role:
        junction_station = station_specs_by_role["junction"]["candidate"]
        base_junction_station = station_specs_by_role["junction"]["baseline"]
        union_junction_section = _section_loops(
            union_positions, union_faces,
            junction_station["section_center_world_m"],
            junction_station["tangent_world_unit"], derived_csg=True,
            face_owner_ids=union_lineage["face_owner_ids"],
            face_source_ids=union_lineage["source_face_ids"],
            face_generated_mask=union_lineage["generated_face_mask"],
            include_projected_loop_vertices=True)
        baseline_union_junction_section = _section_loops(
            base_union_positions, base_union_faces,
            base_junction_station["section_center_world_m"],
            base_junction_station["tangent_world_unit"], derived_csg=True,
            face_owner_ids=base_union_lineage["face_owner_ids"],
            face_source_ids=base_union_lineage["source_face_ids"],
            face_generated_mask=base_union_lineage["generated_face_mask"],
            include_projected_loop_vertices=True)
    else:
        junction_station = base_junction_station = None
        union_junction_section = baseline_union_junction_section = None

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

    def same_material_station_calibre_by_owner():
        if candidate_station_transport_policy != "SOURCE_MATERIAL_SECTION_FRAME_V1":
            return None
        result = {}
        for role, owner_id, current_surface, baseline_surface in (
            ("bridge", bridge_key[1], bridge, base_bridge),
            ("target", target_key[1], target, base_target),
        ):
            declaration = runtime_source_station_declarations.get(role)
            _require(isinstance(declaration, Mapping),
                     f"{role} same-material station declaration is missing")
            proof = declaration.get("candidate_station_transport_proof")
            _require(isinstance(proof, Mapping),
                     f"{role} same-material station transport proof is missing")
            result[owner_id] = {
                "station_role": role,
                "source_station_declaration_id": declaration.get("declaration_id"),
                "source_station_ancestry": declaration.get("source_ancestry"),
                "transport_proof": proof,
                "baseline_surface_binding": _source_geometry_binding(baseline_surface),
                "candidate_surface_binding": _source_geometry_binding(current_surface),
            }
        return result

    def observe_candidate_station_applicability(station_role, station):
        source_declaration = runtime_source_station_declarations.get(station_role)
        anchor_owner = (
            source_declaration["anchor_owner_id"]
            if isinstance(source_declaration, Mapping)
            else bridge_key[1] if station_role == "bridge" else target_key[1])
        records = {
            str(bridge_key[1]): _source_station_owner_section(
                bridge, station, bridge_key[1],
                label=f"candidate bridge owner {bridge_key[1]}"),
            str(target_key[1]): _source_station_owner_section(
                target, station, target_key[1],
                label=f"candidate target owner {target_key[1]}"),
        }
        present = sorted(int(owner) for owner, item in records.items()
                         if item["status"] == "PRESENT_COMPLETE_SOURCE_LOOPS")
        absent = sorted(int(owner) for owner, item in records.items()
                        if item["status"] == "ABSENT_STRICTLY_ON_ONE_SIDE")
        if present == sorted((bridge_key[1], target_key[1])) and not absent:
            mode = "SHARED_TWO_OWNER_CUT"
        elif present == [anchor_owner] and len(absent) == 1:
            mode = "ANCHOR_ONLY_CUT"
        else:
            raise ValueError(
                f"{station_role} candidate station is neither a complete shared cut "
                "nor a strictly absent anchor-only cut")
        return {
            "applicability_mode": mode,
            "station_role": station_role,
            "anchor_owner_id": anchor_owner,
            "candidate_station_sha256": _canonical_record_sha256(dict(station)),
            "owner_status_by_id": records,
            "present_owner_ids": present,
            "absent_owner_ids": absent,
            "candidate_geometry_used_for_observation": True,
            "candidate_geometry_used_for_source_declaration": False,
        }

    def compare_union_station(label, station_role, candidate_section, baseline_section,
                              candidate_station, baseline_station):
        source_declaration = runtime_source_station_declarations.get(station_role)
        station_applicability = runtime_source_station_applicability.get(station_role)
        if station_applicability is not None:
            candidate_applicability = observe_candidate_station_applicability(
                station_role, candidate_station)
            _require(candidate_applicability["applicability_mode"]
                     == station_applicability["applicability_mode"],
                     f"{label} candidate cut applicability differs from its source declaration")
            if station_applicability["applicability_mode"] == "ANCHOR_ONLY_CUT":
                candidate_owner_sections = {
                    bridge_key[1]: _source_surface_section(
                        bridge, candidate_station, include_face_segment_records=True),
                    target_key[1]: _source_surface_section(
                        target, candidate_station, include_face_segment_records=True),
                }
                baseline_owner_sections = {
                    bridge_key[1]: _source_surface_section(
                        base_bridge, baseline_station, include_face_segment_records=True),
                    target_key[1]: _source_surface_section(
                        base_target, baseline_station, include_face_segment_records=True),
                }
                anchor_role = (
                    "bridge" if station_applicability["anchor_owner_id"] == bridge_key[1]
                    else "target")
                candidate_anchor_surface = (
                    bridge if anchor_role == "bridge" else target)
                baseline_anchor_surface = (
                    base_bridge if anchor_role == "bridge" else base_target)
                owner_only = _anchor_only_union_station_proof(
                    applicability=station_applicability,
                    candidate_applicability=candidate_applicability,
                    candidate_union=candidate_section,
                    baseline_union=baseline_section,
                    candidate_owner_sections=candidate_owner_sections,
                    baseline_owner_sections=baseline_owner_sections,
                    expected_owner_ids=(bridge_key[1], target_key[1]),
                    candidate_anchor_surface=candidate_anchor_surface,
                    baseline_anchor_surface=baseline_anchor_surface,
                    candidate_station=candidate_station,
                    baseline_station=baseline_station,
                    anchor_role=anchor_role,
                    solid_evidence=junction_solid_evidence())
                return {
                    **owner_only,
                    "station_role": station_role,
                    "source_station_declaration": source_declaration,
                    "candidate_station": candidate_station,
                    "baseline_station": baseline_station,
                }
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
                and baseline_union_view.get("passed") is True
                and (station_applicability is None
                     or station_applicability["applicability_mode"]
                        != "SHARED_TWO_OWNER_CUT")):
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
        if (candidate_union_view.get("loop_count") == 1
                and baseline_union_view.get("loop_count") == 1
                and station_applicability is not None
                and station_applicability["applicability_mode"]
                    == "SHARED_TWO_OWNER_CUT"):
            candidate_branch_sections = {
                bridge_key[1]: _source_surface_section(
                    bridge, candidate_station, include_projected_loop_vertices=True),
                target_key[1]: _source_surface_section(
                    target, candidate_station, include_projected_loop_vertices=True),
            }
            baseline_branch_sections = {
                bridge_key[1]: _source_surface_section(
                    base_bridge, baseline_station, include_projected_loop_vertices=True),
                target_key[1]: _source_surface_section(
                    base_target, baseline_station, include_projected_loop_vertices=True),
            }
            union_metrics = candidate_union_view.get("loop_metrics", [])
            union_owners = (union_metrics[0].get("owner_ids", [])
                            if len(union_metrics) == 1 else [])
            if union_owners == sorted((bridge_key[1], target_key[1])):
                proof = _shared_cut_mixed_union_loop_proof(
                expected_owner_ids=(bridge_key[1], target_key[1]),
                candidate_union_section=candidate_union_view,
                baseline_union_section=baseline_union_view,
                candidate_branch_sections=candidate_branch_sections,
                baseline_branch_sections=baseline_branch_sections,
                solid_evidence=junction_solid_evidence(),
                station_declaration=source_declaration)
            elif len(union_owners) == 1 and union_owners[0] in (bridge_key[1], target_key[1]):
                proof = _shared_cut_nested_union_loop_proof(
                    expected_owner_ids=(bridge_key[1], target_key[1]),
                    candidate_union_section=candidate_union_view,
                    baseline_union_section=baseline_union_view,
                    candidate_branch_sections=candidate_branch_sections,
                    baseline_branch_sections=baseline_branch_sections,
                    solid_evidence=junction_solid_evidence(),
                    station_declaration=source_declaration)
            else:
                proof = {"passed": False,
                         "rejections": ["shared union section has neither mixed-owner nor nested single-owner lineage"]}
            _require(proof["passed"],
                     f"{label} shared mixed-owner loop proof rejected: "
                     + "; ".join(proof["rejections"][:5]))
            return {
                "passed": True,
                "station_applicability_mode": "SHARED_TWO_OWNER_CUT",
                "method": "explicit complete shared-cut source sections and one mixed-owner connected-union loop",
                "source_baseline_applicability": station_applicability,
                "candidate_station_applicability": candidate_applicability,
                "source_station_declaration": source_declaration,
                "candidate_station": candidate_station,
                "baseline_station": baseline_station,
                "junction_aware_proof": proof,
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
            same_material_station_calibre_by_owner=(
                same_material_station_calibre_by_owner()),
            cross_time_calibre_policy=candidate_station_transport_policy,
        )
        _require(proof["passed"],
                 f"{label} junction-aware section proof rejected: "
                 + "; ".join(proof["rejections"][:5]))
        return {
            "passed": True,
            "method": "explicit one-loop-per-vessel source-collar-local junction proof",
            "station_applicability_mode": (
                None if station_applicability is None else
                station_applicability["applicability_mode"]),
            "source_baseline_applicability": station_applicability,
            "candidate_station_applicability": (
                None if station_applicability is None else candidate_applicability),
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
                same_material_station_calibre_by_owner=(
                    same_material_station_calibre_by_owner()),
                cross_time_calibre_policy=candidate_station_transport_policy,
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
        if junction_station is not None:
            station_results["union_at_junction_station"] = {
                "station": junction_station,
                "baseline_station_transported_by_ancestry": base_junction_station,
                "candidate": union_junction_section,
                "baseline": baseline_union_junction_section,
                "candidate_vs_baseline": compare_union_station(
                    "union section at source-derived shared junction station",
                    "junction", union_junction_section,
                    baseline_union_junction_section,
                    junction_station, base_junction_station),
            }
    if runtime_source_station_applicability:
        declared_modes = _validated_source_station_applicability_modes(
            runtime_source_station_applicability)
        station_output_by_role = {
            "bridge": "union_at_bridge_station",
            "target": "union_at_target_station",
            "junction": "union_at_junction_station",
        }
        observed_modes = {
            role: station_results[station_output_by_role[role]][
                "candidate_vs_baseline"].get("station_applicability_mode")
            for role in declared_modes
        }
        _require(observed_modes == declared_modes,
                 "candidate station results do not preserve every source-declared applicability mode")

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
        "source_baseline_station_applicability": {
            role: record for role, record
            in sorted(runtime_source_station_applicability.items())},
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
