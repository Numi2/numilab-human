import hashlib
import json
import os
import shutil
import struct
import traceback
from pathlib import Path

from numilab_human.passive_attachment_composition import compose, _read_nhtiss4

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
R = A / "current-manifest-recovery-004"
accepted = A / "forearm492-native-source-preparation-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
original_owner = R / "bilateral-b42-full-owner-base-001/owner-base/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
source_registration = R / "bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
accepted_registration = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json")
old_map = R / "source-evolution-proof-001/accepted-bodyparts3d-myosim-surface-map.v1.json"
new_map = Path("/Users/n/numi-human-anatomy-finish-1378/config/bodyparts3d-myosim-surface-map.v1.json")
compat = R / "bilateral-regional-b42-current-source-compatibility-001/compatibility-report.json"
attempt = R / "accepted-registration-binding-retarget-006"
output = attempt / "output"
attempt.mkdir(exist_ok=False)
owner_dir = attempt / "owner-base-input"
owner_dir.mkdir()
owner = owner_dir / original_owner.name
owner_manifest_path = original_owner.with_suffix(".manifest.json")
clone_manifest_path = owner.with_suffix(".manifest.json")
owner_doc = json.loads(owner_manifest_path.read_text())
registration_name = owner_doc["source"]["registration"]["file"]
staged_registration = owner_dir / registration_name
for source_path, target_path in (
    (original_owner, owner), (owner_manifest_path, clone_manifest_path),
    (source_registration, staged_registration),
):
    if target_path.exists():
        raise RuntimeError(f"staging target already exists: {target_path}")
    if source_path == original_owner:
        os.link(source_path, target_path)
    else:
        shutil.copyfile(source_path, target_path)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(p):
    return json.loads(Path(p).with_suffix(".manifest.json").read_text())
def resolve_registration(payload):
    m = manifest(payload)
    p = Path(m["source"]["registration"]["file"])
    if not p.is_absolute():
        p = Path(payload).with_suffix(".manifest.json").parent / p
    return p.resolve()
pins = {
    "accepted_payload": {"path": str(accepted), "sha256": sha(accepted)},
    "accepted_manifest": {"path": str(accepted.with_suffix(".manifest.json")),
                          "sha256": sha(accepted.with_suffix(".manifest.json"))},
    "original_owner_payload": {"path": str(original_owner), "sha256": sha(original_owner)},
    "original_owner_manifest": {"path": str(owner_manifest_path), "sha256": sha(owner_manifest_path)},
    "staged_owner_payload": {"path": str(owner), "sha256": sha(owner)},
    "staged_owner_manifest": {"path": str(clone_manifest_path), "sha256": sha(clone_manifest_path)},
    "staged_registration": {"path": str(staged_registration), "sha256": sha(staged_registration)},
    "source_registration": {"path": str(source_registration), "sha256": sha(source_registration)},
    "accepted_registration": {"path": str(accepted_registration), "sha256": sha(accepted_registration)},
    "old_surface_map": {"path": str(old_map), "sha256": sha(old_map)},
    "current_surface_map": {"path": str(new_map), "sha256": sha(new_map)},
    "compatibility_report": {"path": str(compat), "sha256": sha(compat)},
    "composer_source": {"path": "/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/passive_attachment_composition.py",
                        "sha256": sha("/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/passive_attachment_composition.py")},
}
expected = {
    "accepted_payload": "c9d5c8de77d398b047075cc222cdb2bfd7db08d37dbd1a52d0bc802664805ca2",
    "original_owner_payload": "13fb46a268ee57201c5558d583c19facde384c1aa3f57694f513d5fc98ee1645",
    "source_registration": "31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c",
    "accepted_registration": "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e",
    "old_surface_map": "bd08a3d604a754065fd693028c1a307487006f7d13acd40153bda87ed7e423f9",
    "current_surface_map": "1d4bbbf216e1a67ac7e27f87d61cbda513f5661b703c5c450c05a51c3c31caa2",
    "compatibility_report": "f70ac7ced0b1fbcc6186c24f67aab47461425d5f76c7a191903050552293190e",
}
for name, expected_sha in expected.items():
    if pins[name]["sha256"] != expected_sha:
        raise RuntimeError(f"pinned input changed: {name}")
if (pins["staged_owner_payload"]["sha256"] != pins["original_owner_payload"]["sha256"]
        or pins["staged_owner_manifest"]["sha256"] != pins["original_owner_manifest"]["sha256"]
        or pins["staged_registration"]["sha256"] != pins["source_registration"]["sha256"]):
    raise RuntimeError("isolated owner-base input clone is not byte-identical to declared source assets")
if resolve_registration(owner) != staged_registration.resolve():
    raise RuntimeError("staged owner-base registration locator does not resolve exactly")
accepted_manifest = manifest(accepted)
owner_manifest = manifest(owner)
accepted_source = accepted_manifest["source"]
owner_source = owner_manifest["source"]
if accepted_source["registration"]["sha256"] != pins["accepted_registration"]["sha256"]:
    raise RuntimeError("retained accepted registration does not match accepted manifest identity")
if accepted_source["bodyparts"] != owner_source["bodyparts"]:
    raise RuntimeError("BodyParts descriptor drifted")
if accepted_source["myosim_source_archive_sha256"] != owner_source["myosim_source_archive_sha256"]:
    raise RuntimeError("MyoSim archive drifted")
old_map_doc, new_map_doc = json.loads(old_map.read_text()), json.loads(new_map.read_text())
if (old_map_doc["schema"] != new_map_doc["schema"]
        or old_map_doc["entries"] != new_map_doc["entries"][:len(old_map_doc["entries"])]):
    raise RuntimeError("old map is not an exact prefix of current map")
accepted_data, _ = _read_nhtiss4(accepted), _read_nhtiss4(owner)
row134 = next(row for row in accepted_data["records"] if int(row[6]) == 134)
row134meta = next(row for row in accepted_source["surfaces"] if int(row["stable_id"]) == 134)
if int(row134[1]) != 21 or len(row134meta["body_bindings"]) != 20:
    raise RuntimeError("row 134 is not the expected single-terminal-name omission")
terminal_core = struct.unpack_from("<I", accepted_data["raw"],
                                   accepted_data["binding_start"] + (int(row134[0])+20)*36)[0]
if terminal_core != 91:
    raise RuntimeError(f"row 134 terminal binding changed: {terminal_core}")
if row134meta["body_bindings"][-1]["core_body_index"] != struct.unpack_from(
        "<I", accepted_data["raw"],
        accepted_data["binding_start"] + (int(row134[0])+19)*36)[0]:
    raise RuntimeError("row 134 metadata prefix does not match serialized binding identities")
recovery = {
    "stable_id": 134, "binding_ordinal": 20,
    "core_body_index": 91, "myosim_body": "humerus_l",
    "method": "one_missing_terminal_binding_name_from_accepted_registration_and_current_owner_core_map",
    "accepted_payload_sha256": pins["accepted_payload"]["sha256"],
    "accepted_manifest_sha256": pins["accepted_manifest"]["sha256"],
    "owner_base_payload_sha256": pins["staged_owner_payload"]["sha256"],
    "owner_base_manifest_sha256": pins["staged_owner_manifest"]["sha256"],
    "accepted_registration_path": str(accepted_registration),
    "accepted_registration_sha256": pins["accepted_registration"]["sha256"],
}
proof = {
    "schema": "numi.human.passive-registration-source-evolution-proof.v1",
    "compatibility_report_path": str(compat),
    "compatibility_report_sha256": pins["compatibility_report"]["sha256"],
    "accepted_surface_map_path": str(old_map),
    "owner_surface_map_path": str(new_map),
    "accepted_binding_metadata_recoveries": [recovery],
}
preflight = {
    "schema": "numi.human.passive-registration-binding-retarget-preflight.v1",
    "status": "inputs_frozen_before_source_only_retarget",
    "pins": pins,
    "accepted_registration_declared_locator": accepted_source["registration"]["file"],
    "accepted_registration_manifest_locator_present": False,
    "accepted_registration_external_recovery_is_sha_pinned": True,
    "selected_stable_ids": list(range(69, 151)),
    "recovery": recovery,
    "output_path": str(output),
    "geometry_predicates_run": False,
}
(attempt / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
json.loads((attempt / "preflight.json").read_text())
runner = Path(__file__).resolve()
shutil.copyfile(runner, attempt / "runner.py")
try:
    result = compose(
        accepted, output, [],
        registration_metadata_base=owner,
        registration_path=staged_registration,
        registration_binding_rows=tuple(range(69, 151)),
        source_evolution_proof=proof,
    )
except BaseException as exc:
    failure = {
        "schema": "numi.human.passive-registration-binding-retarget-result.v1",
        "status": "failed_before_admission",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "preflight_sha256": sha(attempt / "preflight.json"),
        "pins": pins,
        "geometry_predicates_run": False,
    }
    (attempt / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    raise
(attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": result["status"],
    "output_payload": result["output_payload_path"],
    "output_payload_sha256": result["output_payload_sha256"],
    "output_manifest": result["output_manifest_path"],
    "output_manifest_sha256": result["output_manifest_sha256"],
    "selected_row_count": len(result["operation"]["selected_stable_ids"]),
    "changed_binding_records": result["operation"]["changed_binding_record_count"],
    "recovered_binding_metadata": result["operation"]["accepted_binding_metadata_recoveries"],
    "source_evolution_proof": result["operation"]["source_evolution_proof"],
    "attempt": str(attempt),
}, indent=2, sort_keys=True))
