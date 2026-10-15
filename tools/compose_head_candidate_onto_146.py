"""Bind the reviewed head candidate to the current 146 delivery receipt.

This is a source-composition step only. The old candidate receipt contains stale
bone and muscle bindings, so only the pinned changed-row identities, central-canal
owner bindings, and head-specific proof records are transferred.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/Users/n/numi-human-final-integration-publish-1397")
D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
HEAD = D / "pons-fourth-ventricle-reconstruction-001"
BASE_RECEIPT = D / "teres-medial-triceps-native-source-146/prepared/resting-anatomy-receipt.json"
CANDIDATE_RECEIPT = HEAD / "central-canal-medulla-priority-source-export-002/resting-anatomy-receipt.json"
OWNER_RECEIPT = HEAD / "central-canal-owner-rebind-001/resting-anatomy-receipt.json"
OWNER_PAYLOAD = HEAD / "central-canal-owner-rebind-001/resting-thorax.nhanatomy"
BASE_PAYLOAD_SHA = "5c955be571416fed6a126d793a9737cf856c51b7355300761890e908f9b0921f"
CANDIDATE_PAYLOAD_SHA = "e197a6a73ea1344a4bc540c7d9118ae290ff643ae96a54839b1de705fc3437b6"
BASE_RECEIPT_SHA = "8139d5f034aaba2adb4158451e20913531e6cc862c9fd436589fc445bd76a8c1"
CANDIDATE_RECEIPT_SHA = "053e94f1180f957f05337d8bcfac84c4a0a4477606b52fe1d981736b1f564e5d"
OWNER_RECEIPT_SHA = "6a762240e5e168db7231da8f89746c6b7a456aee090ce5ed7866af7006af3d40"
OWNER_PAYLOAD_SHA = "48781120419c325be12c2ee72d0b0af502baf86f01f8db9dac14433fdab86317"
HEAD_PROOF_KEYS = (
    "central_canal_common_field_binding_recovery_002",
    "central_canal_fourth_ventricle_priority_partition_source_preparation",
    "central_canal_medulla_priority_partition_source_preparation",
    "central_canal_owner_rebind",
)
CHANGED_IDS = (12, 323, 343)
EXPECTED_IDENTITY_DIFFS = CHANGED_IDS
OUT = HEAD / "head-on-muscle146-compose-001"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    require(not OUT.exists(), "preserve prior composition attempts")
    for path, expected in (
        (BASE_RECEIPT, BASE_RECEIPT_SHA),
        (CANDIDATE_RECEIPT, CANDIDATE_RECEIPT_SHA),
        (OWNER_RECEIPT, OWNER_RECEIPT_SHA),
        (OWNER_PAYLOAD, OWNER_PAYLOAD_SHA),
    ):
        require(sha_file(path) == expected, f"pinned input changed: {path}")

    sys.path.insert(0, str(ROOT / "src"))
    from numilab_human.resting_brachiocephalic_common_field import _resolve_sidecar
    from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
    from numilab_human.resting_pleura_proxy import _parse_payload, _record_content_bytes

    base = json.loads(BASE_RECEIPT.read_text())
    candidate_evidence = json.loads(CANDIDATE_RECEIPT.read_text())
    owner = json.loads(OWNER_RECEIPT.read_text())
    base_raw = Path(base["payload"]["path"]).read_bytes()
    candidate_raw = Path(candidate_evidence["payload"]["path"]).read_bytes()
    require(sha_bytes(base_raw) == BASE_PAYLOAD_SHA
            and sha_bytes(candidate_raw) == CANDIDATE_PAYLOAD_SHA,
            "base or candidate NHA bytes differ from the retained receipts")
    require(base["payload"]["sha256"] == BASE_PAYLOAD_SHA
            and base["functional_bindings"]["anatomy_payload_sha256"] == BASE_PAYLOAD_SHA,
            "146 receipt does not bind the exact NHA base")
    require(candidate_evidence["payload"]["sha256"] == CANDIDATE_PAYLOAD_SHA
            and candidate_evidence["payload"]["input_payload_sha256"] == BASE_PAYLOAD_SHA,
            "head candidate receipt does not bind its payload to the exact 146 NHA base")

    base_header, _base_records, base_rows = _parse_payload(base_raw)
    cand_header, _cand_records, cand_rows = _parse_payload(candidate_raw)
    require(base_header[:3] == cand_header[:3] and base_header[5:] == cand_header[5:]
            and list(base_rows) == list(cand_rows) and len(base_rows) == 525,
            "ordered NHA ABI, registration, or stable-ID set changed")
    record_diffs = [sid for sid in base_rows
                    if _record_content_bytes(base_rows[sid]) != _record_content_bytes(cand_rows[sid])]
    typed_diffs = [sid for sid in base_rows if any(
        base_rows[sid][key] != cand_rows[sid][key]
        for key in ("body_index", "layer", "flags"))]
    require(tuple(record_diffs) == EXPECTED_IDENTITY_DIFFS,
            f"unexpected changed NHA records: {record_diffs}")
    require(typed_diffs == [12], f"unexpected typed-owner differences: {typed_diffs}")
    require(base_rows[12]["body_index"] == 20 and cand_rows[12]["body_index"] == 23
            and base_rows[12]["layer"] == cand_rows[12]["layer"] == 3
            and base_rows[12]["flags"] == cand_rows[12]["flags"],
            "central-canal row does not match the one proved owner migration")
    require(all(base_rows[sid][key] == cand_rows[sid][key]
                for sid in (323, 343) for key in ("body_index", "layer", "flags")),
            "fourth-ventricle or brain row changed typed ownership")

    base_map = base["provenance"]["source_id_map"]
    candidate_map = candidate_evidence["provenance"]["source_id_map"]
    require(set(base_map) == set(candidate_map) == {str(sid) for sid in base_rows},
            "source identity map does not cover the exact ordered row set")
    identity_diffs = [int(sid) for sid in base_map if base_map[sid] != candidate_map[sid]]
    require(tuple(sorted(identity_diffs)) == EXPECTED_IDENTITY_DIFFS,
            f"unexpected source identity map changes: {identity_diffs}")
    require(candidate_evidence["functional_bindings"].get("central_canal_stable_id") == 12
            and type(candidate_evidence["functional_bindings"].get(
                "central_canal_stable_id")) is int
            and candidate_evidence["functional_bindings"].get(
                "central_canal_passive_owner_body_index") == 23
            and type(candidate_evidence["functional_bindings"].get(
                "central_canal_passive_owner_body_index")) is int,
            "head candidate is missing the authenticated central-canal functional bindings")
    for sid in CHANGED_IDS:
        identity = candidate_map[str(sid)]
        require(type(identity.get("body_index")) is int
                and type(identity.get("layer")) is int
                and identity["body_index"] == cand_rows[sid]["body_index"]
                and identity["layer"] == cand_rows[sid]["layer"],
                f"candidate identity/typed row mismatch for stable ID {sid}")
        if sid != 12:
            require(type(identity.get("source_member")) is str
                    and len(identity.get("source_sha256", "")) == 64,
                    f"candidate source member/hash is incomplete for stable ID {sid}")
    require(owner["provenance"]["source_id_map"]["12"] == candidate_map["12"],
            "head candidate row12 identity differs from retained owner-migration proof")
    require(base.get("mass_geometry_accounting") == candidate_evidence.get(
                "mass_geometry_accounting"),
            "head candidate has a different physical mass-geometry accounting block")
    require(all(key in candidate_evidence["provenance"] for key in HEAD_PROOF_KEYS)
            and all(key not in base["provenance"] for key in HEAD_PROOF_KEYS),
            "head proof keys are missing or collide with the 146 receipt")

    source_members = {
        323: HEAD / "source-inventory-001/FJ1731.obj",
        343: HEAD / "source-inventory-001/FJ1769.obj",
    }
    for sid, path in source_members.items():
        identity = candidate_map[str(sid)]
        require(sha_file(path) == identity["source_sha256"],
                f"source OBJ member bytes differ for stable ID {sid}")
    owner_migration = {
        "receipt_raw": OWNER_RECEIPT.read_bytes(),
        "payload_raw": OWNER_PAYLOAD.read_bytes(),
        "source_member_raw": (HEAD / "source-inventory-001/FJ1737.obj").read_bytes(),
    }

    output_payload = OUT / "resting-thorax.nhanatomy"
    # Keep owner identity and geometry edits as two separately proved owner steps:
    # first rebind only the authenticated 20→23 migration with unchanged geometry,
    # then apply the registered head geometry/source-lineage candidate under owner 23.
    owner_result, owner_audit = rebind_passive_rows(
        base, base_raw, owner_migration["payload_raw"], changed_ids=(12,),
        output_path=str(OWNER_PAYLOAD), owner_migrations={12: owner_migration})
    result, audit = rebind_passive_rows(
        owner_result, owner_migration["payload_raw"], candidate_raw,
        changed_ids=CHANGED_IDS, output_path=str(output_payload))
    migration_record = result["provenance"].get(
        "verified_passive_owner_migrations", {}).get("12")
    require(isinstance(migration_record, dict),
            "first rebind stage did not retain the verified central-canal owner migration")
    migration_record["final_payload_sha256"] = sha_bytes(candidate_raw)
    rebind_chain = [
        {"stage": "authenticated_central_canal_owner_migration", "audit": owner_audit},
        {"stage": "registered_head_geometry_and_lineage", "audit": audit},
    ]
    result["provenance"]["verified_passive_common_field_rebind_chain"] = rebind_chain
    require(result["mass_geometry_accounting"] == base["mass_geometry_accounting"],
            "rebind changed mass geometry accounting")
    for key in ("bones_payload", "bones_payload_sha256", "native_muscle_surfaces"):
        require(result["provenance"].get(key) == base["provenance"].get(key),
                f"source candidate overwrote the current 146 {key}")
    for key in HEAD_PROOF_KEYS:
        result["provenance"][key] = copy.deepcopy(candidate_evidence["provenance"][key])
    for sid in (323, 343):
        result["provenance"]["source_id_map"][str(sid)] = copy.deepcopy(candidate_map[str(sid)])
    result["functional_bindings"]["central_canal_stable_id"] = 12
    result["functional_bindings"]["central_canal_passive_owner_body_index"] = 23

    common = result["provenance"]["cardiac_geometry_binding"]["common_field"]
    sidecars = {key: _resolve_sidecar(BASE_RECEIPT, BASE_RECEIPT, common[key])
                for key in ("map", "polynomials", "domain_boxes")}
    require(len({path.name for path in sidecars.values()}) == 3,
            "cardiac sidecar names collide")

    OUT.mkdir(parents=True)
    shutil.copyfile(Path(candidate_evidence["payload"]["path"]), output_payload)
    for key, source in sidecars.items():
        destination = OUT / source.name
        shutil.copyfile(source, destination)
        require(sha_file(destination) == common[key]["sha256"],
                f"146 cardiac sidecar bytes changed: {key}")
        common[key]["path"] = destination.name

    report = {
        "status": "SOURCE_COMPOSITION_ONLY_NOT_ANATOMY_OR_NATIVE_ADMISSION",
        "inputs": {
            "base_receipt_sha256": BASE_RECEIPT_SHA,
            "base_payload_sha256": BASE_PAYLOAD_SHA,
            "head_candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA,
            "head_candidate_payload_sha256": CANDIDATE_PAYLOAD_SHA,
            "owner_migration_receipt_sha256": OWNER_RECEIPT_SHA,
            "owner_migration_payload_sha256": OWNER_PAYLOAD_SHA,
        },
        "ordered_nha_rows": len(base_rows),
        "record_changed_stable_ids": record_diffs,
        "typed_owner_changed_stable_ids": typed_diffs,
        "source_identity_changed_stable_ids": identity_diffs,
        "central_canal_owner_migration": {
            "stable_id": 12, "from_core_body_index": 20,
            "to_core_body_index": 23, "layer": 3,
            "source_member": "FJ1737",
            "source_sha256": hashlib.sha256(owner_migration["source_member_raw"]).hexdigest(),
            "source_face_order_preserved": True,
            "native_motion_admission": False,
        },
        "head_identity_overrides": {
            str(sid): {"source_member": candidate_map[str(sid)]["source_member"],
                       "source_sha256": candidate_map[str(sid)]["source_sha256"]}
            for sid in CHANGED_IDS
        },
        "head_proof_keys_imported": list(HEAD_PROOF_KEYS),
        "passive_common_field_rebind": audit,
        "passive_common_field_rebind_chain": rebind_chain,
        "candidate_stale_provenance_excluded": [
            "bones_payload", "bones_payload_sha256", "native_muscle_surfaces",
            "candidate mass/functional receipt wholesale",
        ],
        "current_146_bones_and_nhtiss_provenance_preserved": True,
        "current_146_mass_geometry_accounting_preserved": True,
        "current_146_parent_bindings_preserved_except_nha_payload_sha256": True,
        "current_146_common_field_sidecars_sha256": {
            key: common[key]["sha256"] for key in sidecars
        },
        "code_sha256": {
            str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
            str(Path(rebind_passive_rows.__code__.co_filename).resolve()): sha_file(
                Path(rebind_passive_rows.__code__.co_filename)),
        },
        "remaining_gates": ["head all-nearby source geometry", "retained eight poses plus late pose",
                            "fresh native captures", "integrated regional acceptance"],
        "native_admission": False,
        "anatomical_admission": False,
    }
    report_path = OUT / "head-source-composition-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    result["provenance"]["head_candidate_source_composition_onto_146"] = {
        "report_path": str(report_path),
        "report_sha256": sha_file(report_path),
        "source_preparation_only": True,
        "native_admission": False,
        "anatomical_admission": False,
    }
    receipt_path = OUT / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path),
        "report_path": str(report_path), "report_sha256": sha_file(report_path),
        "payload_path": str(output_payload), "payload_sha256": sha_file(output_payload),
        "row_changes": record_diffs, "typed_changes": typed_diffs,
        "source_identity_changes": identity_diffs,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
