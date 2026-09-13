"""Join Human organ, vessel, cavity, and blood-owner receipts.

This is an ownership and provenance admission, not a physical qualification.
It makes the currently available source/body/hydraulic identities auditable in
one graph and rejects duplicate or hidden mechanical ownership.  The synthetic
blood-owner receipt is retained as its own authority class; it is never
promoted to an anatomical blood-mass or tissue-transfer claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-blood-ownership-ledger.v1"

TORSO = ROOT / "Docs/media/organ-torso-body-link-20260913/body-links.json"
CARDIAC_BODY = ROOT / "Docs/media/organ-cardiac-cavity-body-link-20260913/body-links.json"
CARDIAC_OWNERSHIP = ROOT / "Docs/media/organ-cardiac-cavity-ownership-20260913/ownership.json"
CAVITY_BRIDGE = ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json"
VESSEL_REGISTRATION = ROOT / "Docs/media/organ-vessel-registration-corrected-20260913/registration.json"
VESSEL_BODY = ROOT / "Docs/media/organ-vessel-body-link-20260913/body-links.json"
VESSEL_MASS_MOMENTS = ROOT / (
    "Docs/media/vessel-mass-moment-owner-corrected-20260913/receipt.json"
)
BLOOD_OWNER = ROOT / "Docs/media/blood-mass-owner-20260913/receipt.json"
FULLBODY_VASCULAR = ROOT / (
    "Docs/media/organ-blood-cavity-bridge-20260913/"
    "native-fullbody-vascular-admission/receipt.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("organ/blood ownership ledger: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, *, canonical_json: bool = True) -> tuple[dict[str, Any], str]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"input is not a regular file: {path}")
    raw = path.read_bytes()
    value = read_json(path)
    require(isinstance(value, dict), f"input is not an object: {path}")
    if canonical_json:
        require(raw == canonical(value) + b"\n", f"noncanonical receipt encoding: {path}")
    return value, _sha256(raw)


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value),
            f"{label} is not a lowercase SHA-256")
    return value


def _rows(document: dict[str, Any], label: str, count: int | None = None) -> list[dict[str, Any]]:
    rows = document.get("bindings")
    require(isinstance(rows, list), f"{label} has no bindings")
    if count is not None:
        require(len(rows) == count, f"{label} must contain {count} bindings")
    require(all(isinstance(row, dict) for row in rows), f"{label} contains a malformed binding")
    return rows


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        require(isinstance(value, str) and value.strip(), f"{label} has an invalid {key}")
        require(value not in result, f"{label} repeats {key}: {value}")
        result[value] = row
    return result


def _semantic(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} has no semantic ID")
    return value.replace("FMA:", "FMA", 1)


def _input_record(path: Path, value: dict[str, Any], raw_sha: str) -> dict[str, Any]:
    schema = value.get("schema")
    require(isinstance(schema, str) and schema, f"receipt has no schema: {path}")
    result: dict[str, Any] = {"path": _rel(path), "schema": schema, "file_sha256": raw_sha}
    identity = value.get("identity_sha256")
    if identity is not None:
        result["identity_sha256"] = _sha(identity, f"identity {path}")
    return result


def _check_unassigned(rows: list[dict[str, Any]], label: str) -> None:
    for row in rows:
        require(row.get("mechanical_mass_owner") is None,
                f"{label} admits a mechanical mass owner for {row.get('member_id')}")
        if "physical_volume_owner" in row:
            require(row.get("physical_volume_owner") is None,
                    f"{label} admits a physical volume owner for {row.get('member_id')}")


def compile_ledger(*, torso: Path = TORSO, cardiac_body: Path = CARDIAC_BODY,
                   cardiac_ownership: Path = CARDIAC_OWNERSHIP,
                   cavity_bridge: Path = CAVITY_BRIDGE,
                   vessel_registration: Path = VESSEL_REGISTRATION,
                   vessel_body: Path = VESSEL_BODY,
                   vessel_mass_moments: Path = VESSEL_MASS_MOMENTS,
                   blood_owner: Path = BLOOD_OWNER,
                   fullbody_vascular: Path = FULLBODY_VASCULAR) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    docs: dict[str, dict[str, Any]] = {}
    for name, path, canonical_json in (
        ("torso_body_links", torso, True),
        ("cardiac_body_links", cardiac_body, True),
        ("cardiac_ownership_candidates", cardiac_ownership, True),
        ("cavity_bridge", cavity_bridge, True),
        ("vessel_registration", vessel_registration, True),
        ("vessel_body_links", vessel_body, True),
        ("vessel_mass_moment_candidate", vessel_mass_moments, True),
        # These two native receipts predate this ledger and intentionally retain
        # their original serialization; their raw hashes remain bound.
        ("synthetic_blood_owner", blood_owner, False),
        ("synthetic_fullbody_vascular", fullbody_vascular, False),
    ):
        value, raw_sha = _read(Path(path), canonical_json=canonical_json)
        docs[name] = value
        inputs[name] = _input_record(Path(path), value, raw_sha)

    require(docs["torso_body_links"].get("schema") ==
            "HumanPack.organ-torso-body-link-registration.v1",
            "unsupported torso body-link schema")
    torso_rows = _rows(docs["torso_body_links"], "torso body links", 12)
    torso_by_member = _unique(torso_rows, "member_id", "torso body links")
    require(docs["torso_body_links"].get("qualification", {}).get("body_link_registration") is True,
            "torso body links are not frame-qualified")
    _check_unassigned(torso_rows, "torso body links")

    require(docs["cardiac_body_links"].get("schema") ==
            "HumanPack.organ-cardiac-cavity-body-link-registration.v1",
            "unsupported cardiac body-link schema")
    cardiac_rows = _rows(docs["cardiac_body_links"], "cardiac cavity body links", 4)
    cardiac_by_member = _unique(cardiac_rows, "member_id", "cardiac cavity body links")
    require(docs["cardiac_body_links"].get("qualification", {}).get("body_link_registration") is True,
            "cardiac cavity body links are not frame-qualified")
    _check_unassigned(cardiac_rows, "cardiac cavity body links")

    require(docs["cavity_bridge"].get("schema") == "HumanPack.organ-blood-cavity-bridge.v1",
            "unsupported cavity bridge schema")
    bridge_rows = _rows(docs["cavity_bridge"], "cavity bridge", 4)
    bridge_by_member = _unique(bridge_rows, "member_id", "cavity bridge")
    require(set(bridge_by_member) == set(cardiac_by_member),
            "cavity bridge and cardiac body links have different members")
    bridge_qualification = docs["cavity_bridge"].get("qualification", {})
    require(bridge_qualification.get("exact_source_member_hash_binding") is True and
            bridge_qualification.get("blood_mass_assigned") is False and
            bridge_qualification.get("physical_volume_authority_assigned") is False,
            "cavity bridge boundary changed")
    for member_id, body_row in cardiac_by_member.items():
        bridge_row = bridge_by_member[member_id]
        require(bridge_row.get("source_member_sha256") == body_row.get("source_member_sha256"),
                f"cardiac source hash disagrees between body link and bridge: {member_id}")
        require(_semantic(bridge_row.get("semantic_id"), f"bridge {member_id}") ==
                _semantic(body_row.get("semantic_id"), f"cardiac body {member_id}"),
                f"cardiac semantic ID disagrees between body link and bridge: {member_id}")
        require(bridge_row.get("organ_region_ids") == body_row.get("organ_region_ids"),
                f"cardiac region binding disagrees between body link and bridge: {member_id}")
        require(isinstance(bridge_row.get("hydraulic_volume_authority"), str) and
                bridge_row["hydraulic_volume_authority"].strip(),
                f"cavity bridge has no hydraulic authority: {member_id}")
    hydraulic_cavity_ids = [row["compartment_stable_identifier"] for row in bridge_rows]
    require(len(set(hydraulic_cavity_ids)) == len(hydraulic_cavity_ids),
            "cavity bridge repeats a hydraulic compartment")
    hydraulic_authorities = {row["hydraulic_volume_authority"] for row in bridge_rows}
    require(len(hydraulic_authorities) == 4 and
            all(value.startswith("source_blood:CVSim21:v") for value in hydraulic_authorities),
            "cavity bridge hydraulic authority identity changed")

    require(docs["cardiac_ownership_candidates"].get("schema") ==
            "HumanPack.organ-cardiac-cavity-ownership-candidates.v1",
            "unsupported cardiac ownership schema")
    ownership_rows = docs["cardiac_ownership_candidates"].get("candidates")
    require(isinstance(ownership_rows, list) and len(ownership_rows) == 2,
            "cardiac ownership must retain two candidates")
    require(docs["cardiac_ownership_candidates"].get("selection") is None,
            "cardiac ownership has selected a candidate")
    candidate_ids = {row.get("candidate_id") for row in ownership_rows}
    require(candidate_ids == {"right_atrium_priority", "right_ventricle_priority"},
            "cardiac ownership candidate identity changed")
    for row in ownership_rows:
        require(row.get("four_cavity_interiors_disjoint") is True and
                row.get("biological_selection") is False and
                row.get("mechanical_mass_assigned") is False,
                f"cardiac ownership candidate boundary changed: {row.get('candidate_id')}")
        _sha(row.get("geometry_sha256"), f"candidate {row.get('candidate_id')} geometry")
    ownership_qualification = docs["cardiac_ownership_candidates"].get("qualification", {})
    require(ownership_qualification.get("cavity_domains_disjoint_in_original_source") is False and
            ownership_qualification.get("blood_mass_assigned") is False,
            "cardiac ownership promotes an unresolved physical claim")

    require(docs["vessel_registration"].get("schema") ==
            "HumanPack.organ-vessel-registration.v1",
            "unsupported vessel registration schema")
    vessel_rows = _rows(docs["vessel_registration"], "vessel registration", 6)
    vessel_by_member = _unique(vessel_rows, "member_id", "vessel registration")
    require(docs["vessel_registration"].get("qualification", {}).get("source_to_world_frame_registered") is True,
            "vessel registration is not frame-qualified")
    _check_unassigned(vessel_rows, "vessel registration")

    require(docs["vessel_body_links"].get("schema") ==
            "HumanPack.organ-vessel-body-link-registration.v1",
            "unsupported vessel body-link schema")
    vessel_body_rows = _rows(docs["vessel_body_links"], "vessel body links", 6)
    vessel_body_by_member = _unique(vessel_body_rows, "member_id", "vessel body links")
    require(set(vessel_body_by_member) == set(vessel_by_member),
            "vessel registration and body links have different members")
    require(docs["vessel_body_links"].get("qualification", {}).get("body_link_registration") is True,
            "vessel body links are not frame-qualified")
    _check_unassigned(vessel_body_rows, "vessel body links")
    require(set(vessel_by_member) <= set(torso_by_member),
            "vessel body links are not covered by the torso source register")
    for member_id, registration_row in vessel_by_member.items():
        body_row = vessel_body_by_member[member_id]
        torso_row = torso_by_member[member_id]
        for key in ("source_name", "region_id", "myosim_body"):
            require(registration_row.get(key) == body_row.get(key),
                    f"vessel {key} disagrees between registration and body link: {member_id}")
        require(_semantic(registration_row.get("semantic_id"), f"vessel registration {member_id}") ==
                _semantic(body_row.get("semantic_id"), f"vessel body link {member_id}"),
                f"vessel semantic ID disagrees: {member_id}")
        require(registration_row.get("source_member_sha256") == torso_row.get("source_member_sha256"),
                f"vessel source hash disagrees with torso register: {member_id}")

    mass_moment_candidate = docs["vessel_mass_moment_candidate"]
    require(mass_moment_candidate.get("schema") ==
            "HumanPack.vessel-mass-moment-owner-candidate.v1",
            "unsupported vessel mass-moment candidate schema")
    require(mass_moment_candidate.get("status") == "partial",
            "vessel mass-moment candidate is not explicitly partial")
    candidate_source = mass_moment_candidate.get("source")
    require(isinstance(candidate_source, dict),
            "vessel mass-moment candidate has no source record")
    require(candidate_source.get("registration_sha256") == inputs["vessel_registration"]["file_sha256"],
            "vessel mass-moment candidate registration hash disagrees")
    require(candidate_source.get("body_links_sha256") == inputs["vessel_body_links"]["file_sha256"],
            "vessel mass-moment candidate body-link hash disagrees")
    candidate_rows = mass_moment_candidate.get("owners")
    require(isinstance(candidate_rows, list) and len(candidate_rows) == 6,
            "vessel mass-moment candidate must contain six owners")
    candidate_by_member = _unique(candidate_rows, "member_id", "vessel mass-moment candidate")
    require(set(candidate_by_member) == set(vessel_by_member),
            "vessel mass-moment candidate and registration have different members")
    candidate_density = mass_moment_candidate.get("density", {})
    require(candidate_density.get("status") == "candidate_not_subject_calibrated",
            "vessel mass-moment candidate density was promoted")
    candidate_qualification = mass_moment_candidate.get("qualification", {})
    require(candidate_qualification.get("zeroth_first_second_mass_moments") is True and
            candidate_qualification.get("atomic_checkpoint_restore") is True and
            candidate_qualification.get("anatomical_blood_mass_owner") is False and
            candidate_qualification.get("pressure_gradient_momentum_transfer") is False and
            candidate_qualification.get("two_way_blood_tissue_transfer") is False,
            "vessel mass-moment candidate boundary changed")
    for member_id, candidate_row in candidate_by_member.items():
        registration_row = vessel_by_member[member_id]
        require(candidate_row.get("source_member_sha256") == registration_row.get("source_member_sha256"),
                f"vessel mass-moment source hash disagrees: {member_id}")
        require(candidate_row.get("lumen_or_tube_admitted") is False and
                candidate_row.get("subject_calibration") is False,
                f"vessel mass-moment candidate promoted a physical gate: {member_id}")
    candidate_totals = mass_moment_candidate.get("totals", {})
    require(candidate_totals.get("owner_count") == 6 and
            candidate_totals.get("unique_member_count") == 6,
            "vessel mass-moment candidate owner count changed")

    synthetic_owner = docs["synthetic_blood_owner"]
    require(synthetic_owner.get("schema") == "numi.human.blood-mass-owner-evidence.v1",
            "unsupported synthetic blood-owner schema")
    synthetic_qualification = synthetic_owner.get("qualification", {})
    require(synthetic_qualification.get("zeroth_order_spatial_mass_owner") is True and
            synthetic_qualification.get("anatomical_registration") is False and
            synthetic_qualification.get("subject_specific_density") is False and
            synthetic_qualification.get("momentum_transfer") is False,
            "synthetic blood-owner receipt boundary changed")
    require(synthetic_owner.get("implementation", {}).get("real_fem_region_required") is True and
            synthetic_owner.get("implementation", {}).get("duplicate_owner_rejected") is True,
            "synthetic blood-owner admission contract changed")

    fullbody_owner = docs["synthetic_fullbody_vascular"]
    require(fullbody_owner.get("schema") == "numi.human.fullbody-vascular-admission.v1",
            "unsupported fullbody vascular schema")
    fullbody_qualification = fullbody_owner.get("qualification", {})
    require(fullbody_owner.get("result") == "pass" and
            fullbody_qualification.get("package_admission") == "pass" and
            fullbody_qualification.get("blood_momentum_transfer") == "explicit",
            "fullbody vascular package admission is not a passing synthetic receipt")
    not_qualified = fullbody_owner.get("not_qualified")
    require(isinstance(not_qualified, list) and
            {"anatomical_registration", "activation", "standing", "walking", "unresolved materials"} <=
            set(not_qualified),
            "fullbody vascular receipt no longer exposes its unresolved gates")

    source_reference_rows = {
        "torso_body_frames": sorted(torso_by_member),
        "cardiac_cavity_body_frames": sorted(cardiac_by_member),
        "vessel_body_frames": sorted(vessel_body_by_member),
        "vessel_surface_registration": sorted(vessel_by_member),
        "vessel_surface_mass_moment_candidate": sorted(candidate_by_member),
    }
    hydraulic_rows = [{
        "member_id": row["member_id"],
        "organ_region_ids": row["organ_region_ids"],
        "compartment_stable_identifier": row["compartment_stable_identifier"],
        "hydraulic_volume_authority": row["hydraulic_volume_authority"],
        "physical_volume_owner": row.get("physical_volume_owner"),
        "mechanical_mass_owner": row.get("mechanical_mass_owner"),
        "blood_mass_assigned": False,
    } for row in sorted(bridge_rows, key=lambda item: item["member_id"])]
    vessel_rows_out = [{
        "member_id": row["member_id"],
        "region_id": row["region_id"],
        "hydraulic_volume_owner_id": row["hydraulic_volume_owner_id"],
        "body_link_registration": vessel_body_by_member[row["member_id"]]["body_link_registration"],
        "tubular_field_registered": vessel_body_by_member[row["member_id"]]["tubular_field_registered"],
        "mechanical_mass_owner": row.get("mechanical_mass_owner"),
    } for row in sorted(vessel_rows, key=lambda item: item["member_id"])]
    identity = {
        "inputs": inputs,
        "source_reference_rows": source_reference_rows,
        "hydraulic_rows": hydraulic_rows,
        "vessel_rows": vessel_rows_out,
        "candidate_ids": sorted(candidate_ids),
        "candidate_selection": None,
        "vessel_mass_moment_candidate": inputs["vessel_mass_moment_candidate"]["file_sha256"],
        "synthetic_blood_owner_native_commit": synthetic_owner.get("native_commit"),
        "synthetic_fullbody_native_commit": fullbody_owner.get("native_commit"),
    }
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-blood-ownership-ledger.1",
        "inputs": inputs,
        "source_reference": {
            "body_frame_layers": source_reference_rows,
            "source_member_hashes_unique_within_layer": True,
            "cross_layer_overlap_policy": (
                "The same source member may be referenced by a visual/body-frame receipt "
                "and a hydraulic comparison receipt. It is not a second physical owner."
            ),
        },
        "hydraulic_ownership": {
            "cavity_bindings": hydraulic_rows,
            "unique_compartment_identifiers": True,
            "unique_hydraulic_authorities": True,
            "anatomical_physical_volume_owner": False,
            "anatomical_blood_mass_assigned": False,
        },
        "vessel_ownership": {
            "surface_bindings": vessel_rows_out,
            "unique_regions": True,
            "physical_lumen_owner": False,
            "mechanical_mass_owner": False,
            "surface_mass_moment_candidate": {
                "file": inputs["vessel_mass_moment_candidate"]["path"],
                "owner_count": candidate_totals["owner_count"],
                "source_surface_proxy": True,
                "anatomical_blood_mass_owner": False,
            },
        },
        "synthetic_evidence": {
            "blood_owner": {
                "file": inputs["synthetic_blood_owner"]["path"],
                "native_commit": synthetic_owner.get("native_commit"),
                "zeroth_order_spatial_mass_owner": True,
                "anatomical_registration": False,
                "momentum_transfer": False,
                "subject_specific_density": False,
            },
            "fullbody_vascular_package": {
                "file": inputs["synthetic_fullbody_vascular"]["path"],
                "native_commit": fullbody_owner.get("native_commit"),
                "package_admission": "pass",
                "blood_momentum_transfer": "explicit",
                "anatomical_registration": False,
                "dynamic_vascular_root": False,
            },
        },
        "integrity": {
            "cross_domain_identity_bound": True,
            "duplicate_physical_owner_rejected": True,
            "hydraulic_owner_identity_bound": True,
            "candidate_selection_absent": True,
            "source_hash_agreement": True,
            "physical_mass_owner_count": 0,
        },
        "qualification": {
            "ownership_ledger_bound": True,
            "anatomical_body_frame_registration": True,
            "anatomical_physical_volume_owner": False,
            "anatomical_blood_mass_transfer": False,
            "zeroth_first_second_mass_moments_candidate": True,
            "organ_fem_or_mpm": False,
            "vessel_tube_or_lumen_mechanics": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "activation": False,
            "material_density_calibrated": False,
            "subject_calibration": False,
            "force_convergence": False,
            "exact_clock": False,
            "standing": False,
            "walking": False,
            "integrated_qualification": False,
        },
        "identity_sha256": _sha256(canonical(identity) + b"\n"),
        "boundary": (
            "This ledger binds exact source member hashes, body-frame links, four CVSim "
            "cavity authorities, six vessel hydraulic identities, two unresolved cardiac "
            "geometry candidates, the corrected six-vessel source-surface mass-moment "
            "candidate, and the separate synthetic blood-owner/fullbody package receipts. "
            "It proves ownership bookkeeping, candidate zeroth/first/second moments, and "
            "duplicate-owner rejection only. "
            "No anatomical physical volume, blood mass transfer, organ mechanics, activation, "
            "calibrated materials, force convergence, exact clock, standing, walking, or "
            "integrated Human qualification is promoted."
        ),
    }
    canonical(result)
    return result


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso", type=Path, default=TORSO)
    parser.add_argument("--cardiac-body", type=Path, default=CARDIAC_BODY)
    parser.add_argument("--cardiac-ownership", type=Path, default=CARDIAC_OWNERSHIP)
    parser.add_argument("--cavity-bridge", type=Path, default=CAVITY_BRIDGE)
    parser.add_argument("--vessel-registration", type=Path, default=VESSEL_REGISTRATION)
    parser.add_argument("--vessel-body", type=Path, default=VESSEL_BODY)
    parser.add_argument("--vessel-mass-moments", type=Path, default=VESSEL_MASS_MOMENTS)
    parser.add_argument("--blood-owner", type=Path, default=BLOOD_OWNER)
    parser.add_argument("--fullbody-vascular", type=Path, default=FULLBODY_VASCULAR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_ledger(
            torso=args.torso, cardiac_body=args.cardiac_body,
            cardiac_ownership=args.cardiac_ownership, cavity_bridge=args.cavity_bridge,
            vessel_registration=args.vessel_registration, vessel_body=args.vessel_body,
            vessel_mass_moments=args.vessel_mass_moments,
            blood_owner=args.blood_owner, fullbody_vascular=args.fullbody_vascular,
        )
        output = args.output.resolve()
        digest = immutable_write(output, result)
        print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                          "cavity_bindings": len(result["hydraulic_ownership"]["cavity_bindings"]),
                          "vessel_bindings": len(result["vessel_ownership"]["surface_bindings"]),
                          "physical_mass_owner_count": 0,
                          "anatomical_blood_mass_transfer": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"organ/blood ownership ledger: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
