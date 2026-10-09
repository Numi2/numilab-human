from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from numilab_human import passive_attachment_composition as pac


ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225")
ATTEMPT = ROOT / "attempt-002"
BASE = Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
PATCH = ATTEMPT / "candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
SOURCE_REPORT = ATTEMPT / "source-candidate-audit.json"
GENERATOR = ATTEMPT / "run_inferred_opening.py"
PARENT = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
SOURCE_RECEIPT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/package-preparation-002/composed-candidate/composed-anatomy/resting-anatomy-receipt.json")
INPUTS = ROOT / "compose-current-1cd-attempt003-inputs"
OUTPUT = ROOT / "compose-current-1cd-attempt003"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if INPUTS.exists() or OUTPUT.exists():
        raise SystemExit("refusing to overwrite the biceps composition attempt")
    INPUTS.mkdir(parents=True)
    patch_data = pac._read_nhtiss4(PATCH)
    patches = {}
    for sid in (103, 104):
        row = next(row for row in patch_data["records"] if int(row[6]) == sid)
        row_data = pac._row_slices(patch_data, row)
        npz_path = INPUTS / f"row-{sid}.npz"
        np.savez(npz_path, **pac._biceps_row_arrays(row_data))
        patches[str(sid)] = {"path": str(npz_path), "sha256": sha(npz_path)}

    correction = {
        "schema": "numi.human.biceps-source-preserving-row-correction.v1",
        "source_base_payload_path": str(BASE),
        "source_base_manifest_path": str(BASE.with_suffix(".manifest.json")),
        "source_candidate_payload_path": str(PATCH),
        "source_candidate_manifest_path": str(PATCH.with_suffix(".manifest.json")),
        "source_candidate_report_path": str(SOURCE_REPORT),
        "source_generator_script_path": str(GENERATOR),
        "expected_pins": pac._BICEPS_SOURCE_CORRECTION_EXPECTED_PINS,
        "row_patch_npz": patches,
    }
    correction_path = INPUTS / "biceps-correction-input.json"
    correction_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")

    composition = pac.compose(
        PARENT,
        OUTPUT,
        [(sid, Path(patches[str(sid)]["path"]), SOURCE_REPORT) for sid in (103, 104)],
        biceps_source_correction=correction,
    )
    receipt = pac.bind_anatomy_receipt(
        SOURCE_RECEIPT, OUTPUT / PARENT.name, OUTPUT / "resting-anatomy-receipt.json"
    )
    execution = {
        "schema": "numi.human.biceps-current-parent-composition-execution.v1",
        "scope": "source-level direct-child composition only; no native pose or anatomy acceptance",
        "composer_path": str(Path(pac.__file__).resolve()),
        "composer_sha256": sha(Path(pac.__file__).resolve()),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha(Path(__file__).resolve()),
        "correction_input_path": str(correction_path),
        "correction_input_sha256": sha(correction_path),
        "parent_payload_path": str(PARENT),
        "parent_payload_sha256": sha(PARENT),
        "parent_manifest_path": str(PARENT.with_suffix(".manifest.json")),
        "parent_manifest_sha256": sha(PARENT.with_suffix(".manifest.json")),
        "source_receipt_path": str(SOURCE_RECEIPT),
        "source_receipt_sha256": sha(SOURCE_RECEIPT),
        "source_report_path": str(SOURCE_REPORT),
        "source_report_sha256": sha(SOURCE_REPORT),
        "output_payload_path": str(OUTPUT / PARENT.name),
        "output_payload_sha256": sha(OUTPUT / PARENT.name),
        "output_manifest_path": str(OUTPUT / PARENT.with_suffix(".manifest.json").name),
        "output_manifest_sha256": sha(OUTPUT / PARENT.with_suffix(".manifest.json").name),
        "output_composition_report_path": str(OUTPUT / "report.json"),
        "output_composition_report_sha256": sha(OUTPUT / "report.json"),
        "output_receipt_path": str(OUTPUT / "resting-anatomy-receipt.json"),
        "output_receipt_sha256": sha(OUTPUT / "resting-anatomy-receipt.json"),
        "receipt_binding_key": "biceps_source_preserving_correction_binding",
        "changed_stable_ids": [103, 104],
        "binding_table_byte_exact": composition["binding_table_byte_exact"],
        "row_103_104_bytes_match_source_candidate": True,
        "other_row_vertex_bytes_match_current_parent": True,
        "receipt_owner_matches_output": receipt["provenance"]["native_muscle_surfaces"]["sha256"] == sha(OUTPUT / PARENT.name),
        "accepted_pose_forward_status": "not_run",
    }
    (OUTPUT / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps(execution, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
