#!/usr/bin/env python3
"""Emit a pinned source-only intercostal index rebind receipt from source004."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/n/numi-human-cap-parent-proof-001")
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from numilab_human import resting_intercostal_mesh_arrangement as inter
from numilab_human.resting_pleura_proxy import _parse_payload

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
D = A / "full-anatomy-delivery-audit-1378"
N27_RECEIPT = A / "liver-island-native-027-002/resting-anatomy-receipt.json"
SOURCE004 = D / "forearm492-head-cord-vessel-seam-integrated-source-004"
SOURCE004_RECEIPT = SOURCE004 / "resting-anatomy-receipt.json"
SOURCE004_PAYLOAD = SOURCE004 / "resting-thorax.nhanatomy"
CANDIDATE_SOURCE = D / "intercostal-parent-face-retirement-compose-002/resting-thorax-candidate.nhanatomy"
COMPOSITION_REPORT = D / "intercostal-parent-face-retirement-compose-002/source-only-composition-report.json"
DESCRIPTORS = D / "intercostal-scoped-composer-descriptors-046-001/arrangement-inputs.json"
OUTPUT = D / "intercostal-parent-face-retirement-compose-002/index-only-respiratory-rebind-001"
NATIVE_SOURCE_ROOT = Path("/Users/n/numi-lab-resting-qualification-publish-1395")
NATIVE_OWNERS = {
    "anatomy": NATIVE_SOURCE_ROOT / "apps/NumiHumanRestingAnatomy.hpp",
    "visual": NATIVE_SOURCE_ROOT / "apps/NumiHumanRestingVisual.hpp",
    "respiration_kernel": NATIVE_SOURCE_ROOT / "matter/src/human_respiration.metal",
}
EXPECTED = {
    "source004_receipt": "98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d",
    "source004_payload": "9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09",
    "candidate_payload": "5613eb15de08a7dfe38f583f24c67c8cd4848f7f809c7e4783d27b3ccf326104",
    "composition_report": "c10a12145733fa73f20211bea75fb2c442ea7883ab275b9ef30c294c24999a1d",
    "descriptors_046": "f52417e691a28c2a3419916845599fbf2fcb13db615d377a92794e9912fbe498",
    "anatomy_owner": "10bf76ddfb596ca4ce16bef387f2e509aefc562f3e84bf37520c161c3324b547",
    "visual_owner": "626454ee9b10001ce5754bb539743156761926afc32de1a81950babb2a615839",
    "respiration_kernel": "581626802b9d5b1a9dc28d333669c395deaa506199304488d0f21feb9ecf72f0",
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    require(not OUTPUT.exists(), f"refusing to replace existing output: {OUTPUT}")
    code_commit = git_commit(REPO)
    code_status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not code_status, "source owner checkout is dirty; commit code before preparation")

    input_paths = {
        "source004_receipt": SOURCE004_RECEIPT,
        "source004_payload": SOURCE004_PAYLOAD,
        "candidate_payload_source": CANDIDATE_SOURCE,
        "composition_report": COMPOSITION_REPORT,
        "arrangement_descriptors_046": DESCRIPTORS,
    }
    inventory = {}
    for name, path in input_paths.items():
        require(path.is_file(), f"missing source-only input: {name}")
        digest = sha_file(path)
        expected_key = {
            "candidate_payload_source": "candidate_payload",
            "arrangement_descriptors_046": "descriptors_046",
        }.get(name, name)
        expected = EXPECTED[expected_key]
        require(digest == expected, f"source-only input hash mismatch: {name}")
        inventory[name] = {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}

    source004_receipt = json.loads(SOURCE004_RECEIPT.read_bytes())
    report = json.loads(COMPOSITION_REPORT.read_bytes())
    descriptor_object = json.loads(DESCRIPTORS.read_bytes())
    source004_common = source004_receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    common_inputs = {}
    for key in ("map", "polynomials", "domain_boxes"):
        item = source004_common[key]
        path = (SOURCE004 / item["path"]).resolve()
        require(path.is_file() and sha_file(path) == item["sha256"], f"source004 common sidecar drift: {key}")
        common_inputs[key] = {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}

    owner_files = {
        "intercostal_composer": SRC / "numilab_human/resting_intercostal_mesh_arrangement.py",
        "capture_verifier": SRC / "numilab_human/resting_anatomy_verify.py",
        "abi5_codec": SRC / "numilab_human/resting_anatomy_interface_patch.py",
        "payload_parser": SRC / "numilab_human/resting_pleura_proxy.py",
    }
    owner_pins = {
        name: {"path": str(path.resolve()), "sha256": sha_file(path)}
        for name, path in owner_files.items()
    }
    native_pins = {}
    for name, path in NATIVE_OWNERS.items():
        require(path.is_file() and sha_file(path) == EXPECTED[name if name != "anatomy" else "anatomy_owner"],
                f"functional respiratory source owner changed: {name}")
        native_pins[name] = {"path": str(path.resolve()), "sha256": sha_file(path)}
    native_pins["source_git_commit"] = git_commit(NATIVE_SOURCE_ROOT)

    original_receipt = copy.deepcopy(source004_receipt)
    receipt = copy.deepcopy(source004_receipt)
    candidate_raw = CANDIDATE_SOURCE.read_bytes()
    header, _, candidate_rows = _parse_payload(candidate_raw)
    candidate_sha = sha_bytes(candidate_raw)
    require(candidate_sha == EXPECTED["candidate_payload"], "candidate ABI5 payload changed")
    aggregate = receipt["provenance"]["local_intercostal_mesh_repair"]
    aggregate["abi5_payload_reemit_proof"] = report["abi5_reemit_proof"]

    OUTPUT.mkdir(parents=True)
    try:
        output_payload = OUTPUT / "resting-thorax.nhanatomy"
        shutil.copyfile(CANDIDATE_SOURCE, output_payload)
        require(sha_file(output_payload) == candidate_sha, "copied ABI5 payload differs from source candidate")
        sidecar_outputs = {}
        for key in ("map", "polynomials", "domain_boxes"):
            source = Path(common_inputs[key]["path"])
            destination = OUTPUT / source.name
            shutil.copyfile(source, destination)
            require(sha_file(destination) == common_inputs[key]["sha256"], f"common sidecar copy differs: {key}")
            sidecar_outputs[key] = destination

        declaration = {
            "schema": "numi.human.intercostal.source004-index-only-reemit.v1",
            "source004_receipt_path": str(SOURCE004_RECEIPT.resolve()),
            "source004_receipt_sha256": EXPECTED["source004_receipt"],
            "source004_payload_path": str(SOURCE004_PAYLOAD.resolve()),
            "source004_payload_sha256": EXPECTED["source004_payload"],
            "candidate_payload_path": str(output_payload.resolve()),
            "candidate_payload_sha256": candidate_sha,
            "candidate_payload_source_path": str(CANDIDATE_SOURCE.resolve()),
            "candidate_payload_source_sha256": candidate_sha,
            "composition_report_path": str(COMPOSITION_REPORT.resolve()),
            "composition_report_sha256": EXPECTED["composition_report"],
            "arrangement_descriptors_path": str(DESCRIPTORS.resolve()),
            "arrangement_descriptors_sha256": EXPECTED["descriptors_046"],
            "changed_stable_ids": sorted(inter.REPAIRED_IDS),
            "abi5_reemit_proof": report["abi5_reemit_proof"],
        }
        aggregate["source004_index_only_reemit"] = declaration

        row_summaries = report["row_summaries"]
        descriptor_rows = descriptor_object["rows"]
        for row_record in aggregate["rows"]:
            stable_id = int(row_record["stable_id"])
            key = str(stable_id)
            summary = row_summaries[key]
            row_record["full_candidate"] = descriptor_rows[key]
            for field in (
                "captured_reference_row",
                "compiled_row_sha256",
                "full_candidate_patch_face_count",
                "new_appended_vertex_count",
                "retained_n27_face_map_sha256",
                "retained_n27_source_face_count",
            ):
                row_record[field] = summary[field]
            row_record["null_face_retirement_proof"] = {
                "all_retained_origin_kind0_faces_byte_exact_in_original_order":
                    summary["all_retained_origin_kind0_faces_byte_exact_in_original_order"],
                "candidate_null_face_slot_count": summary["full_candidate_null_face_slot_count"],
                "classification_counts": summary["null_face_retirement_classification_counts"],
                "explicit_null_retirements_complete": summary["explicit_null_retirements_complete"],
                "ledger_sha256": summary["null_face_retirement_ledger_sha256"],
            }
            repair = receipt["provenance"]["source_id_map"][key].setdefault("repair", {})
            repair["local_intersection_remesh"] = row_record

        receipt["payload"].update({
            "path": str(output_payload.resolve()),
            "sha256": candidate_sha,
            "surface_count": int(header[2]),
            "vertex_count": int(header[3]),
            "index_count": int(header[4]),
        })
        for bindings_key in ("functional_bindings", "passive_tissue_parent_bindings"):
            receipt[bindings_key]["anatomy_payload_sha256"] = candidate_sha
        common_field = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
        common_field["anatomy_payload_sha256"] = candidate_sha
        for key, path in sidecar_outputs.items():
            record = common_field[{"map": "map", "polynomials": "polynomials", "domain_boxes": "domain_boxes"}[key]]
            record["path"] = path.name

        cap_resolution = inter.validate_receipt_composition(
            base_receipt_path=N27_RECEIPT,
            base_payload_sha256=inter.N27_PAYLOAD_SHA256,
            receipt=receipt,
            candidate_rows=candidate_rows,
        )
        require(cap_resolution["all_unmodified_original_rows_byte_exact"], "source-only ABI5 row validation failed")
        require(cap_resolution["unchanged_original_row_count"] == 522, "source004 unchanged row count differs")
        require(cap_resolution["historical_retirement_proof_resolution"]["parent_receipt_sha256"]
                == inter.FJ2217_CAP_PARENT_RECEIPT_SHA256, "authenticated a779 cap proof resolution missing")

        old_intercostals = original_receipt["functional_bindings"]["intercostal_stable_ids"]
        new_intercostals = receipt["functional_bindings"]["intercostal_stable_ids"]
        require(old_intercostals == new_intercostals == [312, 313, 314, 315, 316, 317],
                "functional intercostal ID ownership changed")
        for bindings_key in ("functional_bindings", "passive_tissue_parent_bindings"):
            before = copy.deepcopy(original_receipt[bindings_key])
            after = copy.deepcopy(receipt[bindings_key])
            before.pop("anatomy_payload_sha256", None)
            after.pop("anatomy_payload_sha256", None)
            require(before == after, f"non-payload {bindings_key} fields changed")
        require(receipt["mass_geometry_accounting"] == original_receipt["mass_geometry_accounting"],
                "mass geometry accounting changed")
        require(receipt["functional_bindings"]["cardiac_chamber_compartment_bindings"]
                == original_receipt["functional_bindings"]["cardiac_chamber_compartment_bindings"],
                "cardiac circulation compartment binding changed")

        row_audits = {}
        _, _, source004_rows = _parse_payload(SOURCE004_PAYLOAD.read_bytes())
        for stable_id in sorted(inter.REPAIRED_IDS):
            before, after = source004_rows[stable_id], candidate_rows[stable_id]
            require(inter._row_identity(before, stable_id) == inter._row_identity(after, stable_id),
                    f"typed respiratory owner identity changed for {stable_id}")
            require(np.ascontiguousarray(before["vertices6"], dtype="<f4").tobytes()
                    == np.ascontiguousarray(after["vertices6"], dtype="<f4").tobytes(),
                    f"respiratory row vertices/normals changed for {stable_id}")
            row_audits[str(stable_id)] = {
                "typed_identity": [51010, stable_id],
                "body_index": int(after["body_index"]),
                "layer": int(after["layer"]),
                "flags": int(after["flags"]),
                "vertex_count": len(after["vertices6"]),
                "face_count_source004": len(before["faces"]),
                "face_count_candidate": len(after["faces"]),
                "source004_face_sha256": sha_bytes(np.ascontiguousarray(before["faces"], dtype="<u4").tobytes()),
                "candidate_face_sha256": sha_bytes(np.ascontiguousarray(after["faces"], dtype="<u4").tobytes()),
                "vertices6_byte_exact": True,
            }

        receipt["provenance"]["intercostal_index_only_respiratory_rebind"] = {
            "schema": "numi.human.intercostal.index-only-respiratory-rebind.v1",
            "source004_receipt_sha256": EXPECTED["source004_receipt"],
            "source004_payload_sha256": EXPECTED["source004_payload"],
            "candidate_payload_sha256": candidate_sha,
            "functional_ids": new_intercostals,
            "source_only_abi5_validation_status": "PASS_SOURCE_COMPOSITION_ONLY",
            "native_visual_map_rebuild_required": True,
            "native_visual_map_rebuild_performed": False,
            "respiratory_state_or_mass_added": False,
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
            "verification_path": str((OUTPUT / "verification.json").resolve()),
            "owner_files": owner_pins,
            "native_functional_owner_files": native_pins,
            "authenticated_fj2217_parent_proof": cap_resolution["historical_retirement_proof_resolution"],
        }
        receipt_path = OUTPUT / "resting-anatomy-receipt.json"
        write_json(receipt_path, receipt)

        report_out = {
            "status": "SOURCE_ONLY_INDEX_ONLY_RESPIRATORY_REBIND_PASS_NOT_ADMISSION",
            "anatomical_admission": False,
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
            "source_owner_git_commit": code_commit,
            "source_owner_files": owner_pins,
            "native_functional_owner_sources": native_pins,
            "inputs": inventory,
            "source004_common_field_sidecars": common_inputs,
            "output_common_field_sidecars": {
                key: {"path": str(path.resolve()), "sha256": sha_file(path)}
                for key, path in sidecar_outputs.items()
            },
            "candidate_payload": {
                "path": str(output_payload.resolve()),
                "sha256": sha_file(output_payload),
                "bytes": output_payload.stat().st_size,
            },
            "receipt": {
                "path": str(receipt_path.resolve()),
                "sha256": sha_file(receipt_path),
            },
            "source_composition_validation": {
                "status": "PASS_SOURCE_COMPOSITION_ONLY",
                "base_row_count": cap_resolution["base_row_count"],
                "unchanged_source004_rows": cap_resolution["unchanged_original_row_count"],
                "route_transferred_ids": cap_resolution["route_transferred_stable_ids"],
                "cap_operation": cap_resolution["fj2217_cap_operation"],
                "authenticated_cap_parent_resolution":
                    cap_resolution["historical_retirement_proof_resolution"],
            },
            "respiratory_owner": {
                "stable_ids": new_intercostals,
                "changed_rows": row_audits,
                "map_rule": {
                    "functional_owner": "NumiHumanRestingVisual",
                    "functional_deformation_kind": 7,
                    "vertex_source": "primitive face index stream",
                    "runtime_map_rebuild_required": True,
                    "runtime_map_rebuild_performed": False,
                },
                "kernel_rule": {
                    "functional_owner": "human_respiration.metal",
                    "motion": "intercostal deformation uses respiratory state motion.y and paired rib rotations",
                    "physiological_state_unchanged": True,
                },
                "cardiac_and_respiration_bindings_unchanged_except_payload_identity": True,
            },
            "mass_or_physical_state_added": False,
            "source004_original_and_candidate_receipts_untouched": True,
        }
        write_json(OUTPUT / "verification.json", report_out)
        print(json.dumps({
            "status": report_out["status"],
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": report_out["receipt"]["sha256"],
            "payload_path": str(output_payload.resolve()),
            "payload_sha256": report_out["candidate_payload"]["sha256"],
            "verification_path": str((OUTPUT / "verification.json").resolve()),
            "verification_sha256": sha_file(OUTPUT / "verification.json"),
            "source_owner_git_commit": code_commit,
            "unchanged_source004_rows": cap_resolution["unchanged_original_row_count"],
            "native_or_geometry": False,
        }, sort_keys=True))
    except Exception:
        # This output path was proven absent before creation and belongs to this run.
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
