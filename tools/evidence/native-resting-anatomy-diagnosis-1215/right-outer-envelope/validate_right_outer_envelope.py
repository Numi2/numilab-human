#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1206")
OUT = BASE / "right-outer-envelope-validation-001"
SOURCE = Path("/Users/n/numi-human-clearance-envelope-source-1215")
EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
RUN_1191 = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run")
RUN_1201 = Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201/native-run")
NHA = EVIDENCE / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"
DRIVER = BASE / "fit_three_poses_outer_envelope.py"
PLAN = BASE / "fit-attempt-004-plan.json"
REVISION = SOURCE / "source-revision.json"
HELPER = SOURCE / "src/numilab_human/common_atlas_skin_clearance.py"
TARGET = (51005, 63)
STEPS = [(RUN_1191, 0), (RUN_1191, 10000), (RUN_1201, 20000)]

sys.path.insert(0, str(SOURCE / "src"))
from numilab_human import common_atlas_skin_clearance as clearance

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def pin(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"missing required input: {path}")
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")

allowed_existing = {"validate_right_outer_envelope.py"}
if OUT.exists():
    allowed_existing.update(path.name for path in OUT.iterdir()
                            if path.name.startswith("validation-attempt-") and path.name.endswith(".log"))
if OUT.exists() and ((OUT / "validation.json").exists()
                     or any(path.name not in allowed_existing for path in OUT.iterdir())):
    raise SystemExit(f"refuse to overwrite completed or unexpected evidence in {OUT}")
if sha(HELPER) != "ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f":
    raise RuntimeError("frozen 178f569 clearance helper SHA changed")
revision = json.loads(REVISION.read_text())
if revision.get("revision") != "178f56910332326cabb7ba4462a01fbafe45b801":
    raise RuntimeError("clearance source snapshot is not the pinned 178f569 revision")
manifest_entry = next(row for row in revision["files"] if row["path"] == "src/numilab_human/common_atlas_skin_clearance.py")
if manifest_entry.get("sha256") != sha(HELPER):
    raise RuntimeError("source-revision manifest does not bind the exact clearance helper")

nha_sha = sha(NHA)
SCRIPT = Path(__file__).resolve()
source_pins = [pin(path) for path in (REVISION, HELPER, NHA, DRIVER, PLAN, SCRIPT)]
pose_rows = []
for run, step in STEPS:
    pack = run / "accepted-geometry" / f"step-{step}.mrvpack"
    receipt = run / "accepted-geometry" / f"step-{step}.receipt.json"
    pack_pin, receipt_pin = pin(pack), pin(receipt)
    receipt_data = json.loads(receipt.read_text())
    if (receipt_data.get("accepted_step") != step
            or receipt_data.get("physical_endpoint") != "accepted"
            or receipt_data.get("surface_audit_endpoint") != "passed"
            or receipt_data.get("common_field_source_anatomy_payload_sha256") != nha_sha):
        raise RuntimeError(f"accepted pack receipt failed source/endpoint binding at step {step}")

    positions, surfaces, pack_shape = clearance._pack_surfaces(pack, {TARGET})
    surface = surfaces[TARGET]
    faces = surface["faces"]
    if len(faces) != 6688:
        raise RuntimeError(f"right vastus face count changed at step {step}: {len(faces)}")
    target_vertex_ids = np.unique(faces)
    target_local_faces = np.searchsorted(target_vertex_ids, faces)
    target_vertices = positions[target_vertex_ids].astype("<f4", copy=True)
    prepared = clearance._prepare_closed_clearance_target(
        target_vertices, target_local_faces, allow_nested_enclosure=True)
    report = prepared["report"]
    if (report.get("inside_semantics") != "external_skin_enclosure_only"
            or report.get("embedded_closed_target") is not True
            or report.get("face_count") != 6688
            or report.get("outer_component_face_count") != sum(
                row["face_count"] for row in report["component_face_rows"] if
                row["component_id"] == report["outer_component_id"])):
        raise RuntimeError(f"outer-envelope proof did not preserve full face inventory at step {step}")

    pose_rows.append({
        "step": step,
        "simulated_seconds": receipt_data.get("simulated_seconds"),
        "pack": pack_pin,
        "receipt": receipt_pin,
        "pack_shape": pack_shape,
        "target_stable_id": [51005, 63],
        "target_body_id": int(surface["body"]),
        "target_face_count_full": int(len(faces)),
        "target_vertex_count_referenced": int(len(target_vertex_ids)),
        "target_face_row_identity": "every captured row retained in prepared['faces']; no filtering",
        "outer_envelope_validation": report,
    })
    del positions, surfaces, prepared

expected_pins = source_pins + [
    pinrow for row in pose_rows for pinrow in (row["pack"], row["receipt"])
]
expected_by_path = {pinrow["path"]: pinrow["sha256"] for pinrow in expected_pins}
pre = {path: sha(Path(path)) for path in expected_by_path}
if pre != expected_by_path:
    raise RuntimeError("internal input pin mismatch")

OUT.mkdir(exist_ok=True)
result = {
    "schema": "numi.human.right-vastus-outer-envelope-validation.v1",
    "status": "passed_bounded_helper_preparation_only",
    "scope": "prepare and validate the exact right-vastus target outer envelope at source-fit poses; no solver, candidate scan, or native run",
    "source_revision": revision["revision"],
    "source_revision_manifest": pin(REVISION),
    "helper_source": pin(HELPER),
    "helper_manifest_sha256_matches": True,
    "fit004_plan": pin(PLAN),
    "fit004_runner": pin(DRIVER),
    "audit_script": pin(SCRIPT),
    "command": f"PYTHONPATH={SOURCE / 'src'} /Users/n/numi-human-prep-venv-20261005/bin/python3.13 {SCRIPT}",
    "target": "51005:63 right vastus lateralis",
    "poses": pose_rows,
    "qualification_limits": [
        "This is target preparation and component-containment proof only.",
        "It does not evaluate skin-target crossings, skin self-intersections, a fitted candidate, or native qualification.",
        "All target face rows are retained; the external-skin enclosure interpretation does not label inner components as cavities or markers.",
    ],
    "input_hashes_before": pre,
}
post = {path: sha(Path(path)) for path in pre}
result["input_hashes_after"] = post
result["inputs_unchanged"] = pre == post
if pre != post:
    result["status"] = "failed_input_changed_during_validation"
write_json(OUT / "validation.json", result)
if pre != post:
    raise RuntimeError("one or more pinned inputs changed during outer-envelope validation")
print(json.dumps({
    "status": result["status"],
    "output": str(OUT / "validation.json"),
    "steps": [{
        "step": row["step"],
        "face_count": row["target_face_count_full"],
        "component_face_counts": [c["face_count"] for c in row["outer_envelope_validation"]["component_face_rows"]],
        "outer_component_id": row["outer_envelope_validation"]["outer_component_id"],
        "pair_proofs": len(row["outer_envelope_validation"]["component_pair_proofs"]),
    } for row in pose_rows],
}, sort_keys=True))
