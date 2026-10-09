#!/usr/bin/env python3
"""Prepare, but never launch, a 40 s native replay with the verified one-edge NHTISS candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import shlex
import struct
from pathlib import Path

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009")
TASK = ROOT / "muscle-conforming-refinement-1216"
PARENT = ROOT / "native-flat-reference-40s-1201"
BASE_ASSET = Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001")
BASE_PAYLOAD = BASE_ASSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
BASE_MANIFEST = BASE_ASSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
BASE_RECEIPT = Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/resting-anatomy-receipt.json")
SUBSET = TASK / "producer-attempt-002/producer-subset-001"
SUBSET_PAYLOAD = SUBSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
SUBSET_MANIFEST = SUBSET / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
SUBSET_PROOF = TASK / "subset64-accepted039-verification.json"
SUBSET_PROOF_SCRIPT = TASK / "verify_subset64_against_039.py"
HEADER = struct.Struct("<8s6I32s")
RECORD = struct.Struct("<8I")
BINDING = struct.Struct("<I8f")
VERTEX = struct.Struct("<6f4I4f")
CAPTURES = "0,10000,20000"
DEFAULT_NAME = "native-refinement-1217"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"preparation rejected: {message}")


def parse_payload(path: Path) -> dict:
    raw = path.read_bytes()
    magic, abi, nrecords, nbindings, nvertices, nindices, regfp, source = HEADER.unpack_from(raw)
    require(magic == b"NHTISS4\0", f"unexpected NHTISS magic in {path}")
    cursor = HEADER.size
    records = [RECORD.unpack_from(raw, cursor + i * RECORD.size) for i in range(nrecords)]
    cursor += nrecords * RECORD.size
    binding_offset = cursor
    bindings = [BINDING.unpack_from(raw, cursor + i * BINDING.size) for i in range(nbindings)]
    cursor += nbindings * BINDING.size
    vertex_offset = cursor
    vertices = [VERTEX.unpack_from(raw, cursor + i * VERTEX.size) for i in range(nvertices)]
    cursor += nvertices * VERTEX.size
    indices = list(struct.unpack_from(f"<{nindices}I", raw, cursor)) if nindices else []
    cursor += nindices * 4
    require(cursor == len(raw), f"NHTISS byte length mismatch for {path}")
    return {
        "raw": raw, "abi": abi, "nrecords": nrecords, "nbindings": nbindings,
        "nvertices": nvertices, "nindices": nindices, "registration_fingerprint32": regfp,
        "source_sha256": source.hex(), "records": records, "bindings": bindings,
        "vertices": vertices, "indices": indices, "binding_offset": binding_offset,
        "vertex_offset": vertex_offset,
    }


def surface_row(payload: dict, stable_id: int) -> dict:
    found = [(i, r) for i, r in enumerate(payload["records"]) if r[6] == stable_id]
    require(len(found) == 1, f"expected exactly one NHTISS stable ID {stable_id}")
    row_index, record = found[0]
    first_binding, binding_count, first_vertex, vertex_count, first_index, index_count, sid, layer = record
    require(index_count % 3 == 0, f"stable ID {sid} has an incomplete triangle")
    vertex_bytes = payload["raw"][
        payload["vertex_offset"] + first_vertex * VERTEX.size:
        payload["vertex_offset"] + (first_vertex + vertex_count) * VERTEX.size
    ]
    binding_bytes = payload["raw"][
        payload["binding_offset"] + first_binding * BINDING.size:
        payload["binding_offset"] + (first_binding + binding_count) * BINDING.size
    ]
    global_indices = payload["indices"][first_index:first_index + index_count]
    require(all(first_vertex <= i < first_vertex + vertex_count for i in global_indices),
            f"stable ID {sid} has an index outside its vertex span")
    local_indices = [i - first_vertex for i in global_indices]
    faces = [local_indices[i:i + 3] for i in range(0, len(local_indices), 3)]
    return {
        "row_index": row_index, "record": record, "first_binding": first_binding,
        "binding_count": binding_count, "first_vertex": first_vertex,
        "vertex_count": vertex_count, "index_count": index_count,
        "vertex_bytes": vertex_bytes, "binding_bytes": binding_bytes, "faces": faces,
    }


def resolve_asset(receipt: dict, receipt_path: Path, key: str, default_name: str) -> Path:
    record = receipt["provenance"]["native_muscle_surfaces"]
    raw = record.get(key, default_name)
    path = Path(raw)
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def require_source_composition_binding(manifest: dict, expected: dict[str, str]) -> dict:
    source = manifest.get("source", {})
    proof = source.get("conforming_edge_refinement_composition")
    require(isinstance(proof, dict), "composed manifest lacks conforming_edge_refinement_composition provenance")
    for key, digest in expected.items():
        require(proof.get(key) == digest, f"composition provenance {key} does not bind the pinned input")
    require(proof.get("stable_id") == 64 and proof.get("edge_local_vertex_ids") == [2597, 3054],
            "composition provenance identifies a different surface edge")
    require(proof.get("binding_table_byte_exact") is True
            and proof.get("physical_route_mass_and_force_state_unchanged") is True,
            "composition does not assert unchanged physical ownership")
    return proof


def resolve_receipt_asset(receipt: dict, receipt_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def validate_receipt_delta(base_receipt: dict, final_receipt: dict,
                           base_receipt_path: Path, final_receipt_path: Path,
                           payload_path: Path, manifest_path: Path,
                           final_payload: dict) -> tuple[dict, dict, list[Path]]:
    base_copy = copy.deepcopy(base_receipt)
    final_copy = copy.deepcopy(final_receipt)
    base_prov = base_copy.get("provenance", {})
    final_prov = final_copy.get("provenance", {})
    base_muscle = base_prov.pop("native_muscle_surfaces", None)
    final_muscle = final_prov.pop("native_muscle_surfaces", None)
    require(isinstance(base_muscle, dict) and isinstance(final_muscle, dict),
            "native muscle surface receipt binding is missing")
    require("conforming_surface_refinement_binding" not in base_prov,
            "base receipt unexpectedly contains a refinement binding")
    binding = final_prov.pop("conforming_surface_refinement_binding", None)
    require(binding == {
        "prior_receipt_path": str(base_receipt_path.resolve()),
        "prior_receipt_sha256": sha(base_receipt_path),
        "composition_manifest_sha256": sha(manifest_path),
        "changed_stable_ids": [64],
        "scope": "Explicit source-derived conforming surface refinement only; existing physical owners, mass, forces, and tendon state are unchanged.",
    }, "receipt refinement binding is missing or does not match the pinned composition")

    allowed_owner_changes = {
        "payload_path", "sha256", "manifest_path", "manifest_sha256",
        "surface_count", "body_binding_count", "vertex_count", "index_count",
    }
    require(base_muscle.keys() == final_muscle.keys(),
            "candidate native muscle receipt changed its field set")
    for key in base_muscle:
        if key not in allowed_owner_changes:
            require(base_muscle[key] == final_muscle[key],
                    f"candidate native muscle receipt changed non-payload identity {key}")
    require(final_muscle["payload_path"] == str(payload_path.resolve())
            and final_muscle["sha256"] == sha(payload_path),
            "candidate native muscle receipt does not bind composed NHTISS bytes")
    require(final_muscle["manifest_path"] == str(manifest_path.resolve())
            and final_muscle["manifest_sha256"] == sha(manifest_path),
            "candidate native muscle receipt does not bind composed manifest bytes")
    for key, actual in (("surface_count", final_payload["nrecords"]),
                        ("body_binding_count", final_payload["nbindings"]),
                        ("vertex_count", final_payload["nvertices"]),
                        ("index_count", final_payload["nindices"])):
        require(final_muscle.get(key) == actual, f"candidate receipt {key} differs from NHTISS")

    # The binder canonicalizes the anatomy payload path and relocates three immutable
    # cardiac descriptor files beside the derived receipt. Resolve both sides and
    # compare file identities before normalizing only those path strings.
    base_anatomy = base_copy.get("payload", {})
    final_anatomy = final_copy.get("payload", {})
    base_anatomy_path = resolve_receipt_asset(base_receipt, base_receipt_path, base_anatomy.get("path", ""))
    final_anatomy_path = resolve_receipt_asset(final_receipt, final_receipt_path, final_anatomy.get("path", ""))
    require(base_anatomy.get("sha256") == final_anatomy.get("sha256")
            and sha(base_anatomy_path) == base_anatomy.get("sha256")
            and sha(final_anatomy_path) == final_anatomy.get("sha256"),
            "anatomical payload bytes changed while binding the refinement")
    final_anatomy["path"] = base_anatomy.get("path")

    base_common = base_prov.get("cardiac_geometry_binding", {}).get("common_field", {})
    final_common = final_prov.get("cardiac_geometry_binding", {}).get("common_field", {})
    relocated_common = []
    for key in ("map", "polynomials", "domain_boxes"):
        old_record, new_record = base_common.get(key), final_common.get(key)
        require(isinstance(old_record, dict) and isinstance(new_record, dict),
                f"cardiac common-field {key} identity is missing")
        old_path = resolve_receipt_asset(base_receipt, base_receipt_path, old_record.get("path", ""))
        new_path = resolve_receipt_asset(final_receipt, final_receipt_path, new_record.get("path", ""))
        require(old_record.get("sha256") == new_record.get("sha256")
                and sha(old_path) == old_record.get("sha256")
                and sha(new_path) == new_record.get("sha256"),
                f"relocated cardiac common-field {key} bytes differ")
        require(new_path.parent == final_receipt_path.resolve().parent,
                f"relocated cardiac common-field {key} is not beside its derived receipt")
        relocated_common.append(new_path)
        new_record["path"] = old_record.get("path")
    require(base_copy == final_copy,
            "candidate receipt changes non-refinement anatomy or physical bindings")
    return base_muscle, final_muscle, relocated_common


def validate_candidate(composition_dir: Path, candidate_receipt_path: Path) -> dict:
    payload_path = composition_dir / "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
    manifest_path = composition_dir / "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
    report_path = composition_dir / "report.json"
    for path in (payload_path, manifest_path, report_path, candidate_receipt_path):
        require(path.is_file(), f"composer output is missing: {path}")
    require(BASE_PAYLOAD.is_file() and BASE_MANIFEST.is_file() and BASE_RECEIPT.is_file(),
            "accepted039 baseline asset set is incomplete")
    require(SUBSET_PROOF.is_file() and SUBSET_PROOF_SCRIPT.is_file(),
            "verified stable64 subset proof is missing")
    proof = json.loads(SUBSET_PROOF.read_text())
    require(proof.get("status") == "pass", "stable64 subset proof is not passing")
    require(proof["accepted039"]["payload_sha256"] == sha(BASE_PAYLOAD), "accepted039 payload pin drifted")
    require(proof["producer"]["payload_sha256"] == sha(SUBSET_PAYLOAD), "producer subset payload pin drifted")
    require(proof["producer"]["manifest_sha256"] == sha(SUBSET_MANIFEST), "producer subset manifest pin drifted")
    report = json.loads(report_path.read_text())
    require(report.get("inputs_unchanged") is True,
            "composer report does not certify unchanged inputs")
    require(report.get("stable_id") == 64 and report.get("edge_local_vertex_ids") == [2597, 3054],
            "composer report identifies a different source edge")
    expected_unchanged = [sid for sid in range(1, 151) if sid != 64]
    require(report.get("unchanged_row_vertex_bytes_and_local_faces") == expected_unchanged,
            "composer report does not enumerate all149 unchanged rows")
    require(report.get("binding_table_byte_exact") is True,
            "composer report does not certify exact binding-table preservation")
    require(report.get("payload_sha256") == sha(payload_path),
            "composer report does not bind its candidate NHTISS payload")
    require(report.get("manifest_sha256") == sha(manifest_path),
            "composer report does not bind its candidate manifest")
    reported_inputs = report.get("input_sha256")
    require(isinstance(reported_inputs, dict), "composer report lacks exact input hash ledger")
    expected_report_inputs = {
        str(BASE_PAYLOAD.resolve()): sha(BASE_PAYLOAD),
        str(BASE_MANIFEST.resolve()): sha(BASE_MANIFEST),
        str(SUBSET_PAYLOAD.resolve()): sha(SUBSET_PAYLOAD),
        str(SUBSET_MANIFEST.resolve()): sha(SUBSET_MANIFEST),
    }
    for input_path, digest in expected_report_inputs.items():
        require(reported_inputs.get(input_path) == digest,
                f"composer report omits or mismatches required input {input_path}")
    for input_path, digest in reported_inputs.items():
        source_path = Path(input_path)
        require(source_path.is_file() and sha(source_path) == digest,
                f"composer source input changed after replay: {input_path}")
    composer_source = Path(json.loads(manifest_path.read_text())["source"]["conforming_edge_refinement_composition"]["helper_source_path"]).with_name("passive_attachment_composition.py")
    recorded_composer_sha = report.get("composer_source_sha256")
    require(isinstance(recorded_composer_sha, str) and len(recorded_composer_sha) == 64,
            "composer report lacks an exact implementation hash")

    base_payload = parse_payload(BASE_PAYLOAD)
    final_payload = parse_payload(payload_path)
    sub_payload = parse_payload(SUBSET_PAYLOAD)
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    final_manifest = json.loads(manifest_path.read_text())
    sub_manifest = json.loads(SUBSET_MANIFEST.read_text())
    base_receipt = json.loads(BASE_RECEIPT.read_text())
    final_receipt = json.loads(candidate_receipt_path.read_text())
    base_muscle, final_muscle, relocated_common = validate_receipt_delta(
        base_receipt, final_receipt, BASE_RECEIPT, candidate_receipt_path,
        payload_path, manifest_path, final_payload)

    require(base_payload["abi"] == final_payload["abi"] == sub_payload["abi"] == 5,
            "NHTISS ABI changed")
    require(base_payload["nrecords"] == final_payload["nrecords"] == 150,
            "final payload must retain all 150 stable surface records")
    require(base_payload["nbindings"] == final_payload["nbindings"] == 512,
            "full binding record count changed")
    require(base_payload["nvertices"] + 1 == final_payload["nvertices"],
            "final payload must add exactly one vertex")
    require(base_payload["nindices"] + 6 == final_payload["nindices"],
            "final payload must add exactly two triangles")
    require(base_payload["registration_fingerprint32"] == final_payload["registration_fingerprint32"],
            "full registration fingerprint changed")
    require(base_payload["source_sha256"] == final_payload["source_sha256"],
            "MyoSim archive source fingerprint changed")
    require(base_payload["raw"][:20] == final_payload["raw"][:20]
            and base_payload["raw"][28:64] == final_payload["raw"][28:64],
            "NHTISS fixed header identity changed outside vertex/index counts")
    require(base_payload["raw"][base_payload["binding_offset"]:
                                base_payload["binding_offset"] + base_payload["nbindings"] * BINDING.size]
            == final_payload["raw"][final_payload["binding_offset"]:
                                    final_payload["binding_offset"] + final_payload["nbindings"] * BINDING.size],
            "global binding table bytes changed")

    base_rows = [r[6] for r in base_payload["records"]]
    final_rows = [r[6] for r in final_payload["records"]]
    require(final_rows == base_rows == list(range(1, 151)), "stable row order/identity changed")
    base_surfaces = {s["stable_id"]: s for s in base_manifest["source"]["surfaces"]}
    final_surfaces = {s["stable_id"]: s for s in final_manifest["source"]["surfaces"]}
    subset_surface = next(s for s in sub_manifest["source"]["surfaces"] if s["stable_id"] == 64)
    base_surface = base_surfaces[64]
    final_surface = final_surfaces[64]
    operation = subset_surface["conforming_edge_refinement"]
    require(final_surface.get("conforming_edge_refinement") == operation,
            "composed stable64 operation differs from verified producer operation")
    expected_target = copy.deepcopy(base_surface)
    expected_target["vertex_count"] = base_surface["vertex_count"] + 1
    expected_target["triangle_count"] = base_surface["triangle_count"] + 2
    expected_target["conforming_edge_refinement"] = operation
    if isinstance(expected_target.get("route_binding"), dict):
        rb = expected_target["route_binding"]
        rb["maximum_vertex_influences"] = max(
            int(rb.get("maximum_vertex_influences", 0)),
            int(operation["new_vertex_active_route_influence_count"]))
        rb["maximum_nearest_route_distance_m"] = max(
            float(rb.get("maximum_nearest_route_distance_m", 0)),
            float(operation["new_vertex_max_nearest_route_distance_m"]))
        rb["conforming_refinement_midpoint_count"] = 1
    require(final_surface == expected_target,
            "stable64 source manifest changed outside its explicit one-edge refinement fields")
    composition = require_source_composition_binding(final_manifest, {
        "source_payload_sha256": sha(BASE_PAYLOAD),
        "source_manifest_sha256": sha(BASE_MANIFEST),
        "context_payload_sha256": sha(SUBSET_PAYLOAD),
        "context_manifest_sha256": sha(SUBSET_MANIFEST),
    })
    expected_composition_paths = {
        "source_payload_path": str(BASE_PAYLOAD.resolve()),
        "source_manifest_path": str(BASE_MANIFEST.resolve()),
        "context_payload_path": str(SUBSET_PAYLOAD.resolve()),
        "context_manifest_path": str(SUBSET_MANIFEST.resolve()),
    }
    for key, value in expected_composition_paths.items():
        require(composition.get(key) == value,
                f"composition provenance {key} points outside the pinned inputs")
    helper_path = Path(composition.get("helper_source_path", "")).resolve()
    helper_digest = composition.get("helper_source_sha256")
    require(helper_path.is_file() and sha(helper_path) == helper_digest
            and reported_inputs.get(str(helper_path)) == helper_digest,
            "composition helper source is not bound consistently to its exact replay")
    expected_unchanged = [sid for sid in range(1, 151) if sid != 64]
    require(composition.get("unchanged_row_vertex_bytes_and_local_faces") == expected_unchanged,
            "manifest provenance does not enumerate all149 unchanged rows")

    expected_manifest = copy.deepcopy(base_manifest)
    expected_manifest["payload"].update(
        file=payload_path.name,
        sha256=sha(payload_path),
        bytes=payload_path.stat().st_size,
        vertex_count=final_payload["nvertices"],
        index_count=final_payload["nindices"],
    )
    expected_manifest["source"]["surfaces"] = [
        expected_target if int(record["stable_id"]) == 64 else record
        for record in expected_manifest["source"]["surfaces"]
    ]
    expected_manifest["source"]["conforming_edge_refinement_composition"] = composition
    require(final_manifest == expected_manifest,
            "full NHTISS manifest changed outside the declared one-edge refinement")

    unchanged_rows = 0
    for stable_id in base_rows:
        old = surface_row(base_payload, stable_id)
        new = surface_row(final_payload, stable_id)
        require(old["binding_bytes"] == new["binding_bytes"], f"stable ID {stable_id} bindings changed")
        if stable_id != 64:
            require(old["vertex_bytes"] == new["vertex_bytes"], f"unrelated stable ID {stable_id} vertex bytes changed")
            require(old["faces"] == new["faces"], f"unrelated stable ID {stable_id} face indices changed")
            require(base_surfaces[stable_id] == final_surfaces[stable_id],
                    f"unrelated stable ID {stable_id} source manifest record changed")
            unchanged_rows += 1
            continue
        old_vertex_count = operation["old_vertex_count"]
        require(old["vertex_count"] == old_vertex_count, "stable64 base vertex count differs from operation")
        require(new["vertex_count"] == old["vertex_count"] + 1, "stable64 must append one vertex")
        require(old["vertex_bytes"] == new["vertex_bytes"][:len(old["vertex_bytes"])],
                "stable64 preexisting vertex records changed")
        require(new["vertex_bytes"][-VERTEX.size:] == surface_row(sub_payload, 64)["vertex_bytes"][-VERTEX.size:],
                "composed stable64 midpoint record differs from producer subset")
        require(old["faces"] == [new["faces"][i] if i not in (5197, 5198) else old["faces"][i]
                                 for i in range(len(old["faces"]))],
                "stable64 face count or unrelated face order changed")
        require(len(old["faces"]) == 6688 and len(new["faces"]) == 6690, "stable64 face counts differ")
        expected_faces = {
            5197: [2597, 2599, 3974],
            5198: [2597, 3974, 2788],
            6688: [3974, 2599, 3054],
            6689: [3974, 3054, 2788],
        }
        for face_row, expected in expected_faces.items():
            require(new["faces"][face_row] == expected, f"stable64 face row {face_row} differs")
        for key in ("member_id", "member", "member_sha256", "layer", "body_bindings", "matched_muscles"):
            require(base_surface.get(key) == final_surface.get(key), f"stable64 identity changed: {key}")
        require(final_surface["vertex_count"] == base_surface["vertex_count"] + 1,
                "stable64 manifest vertex count mismatch")
        require(final_surface["triangle_count"] == base_surface["triangle_count"] + 2,
                "stable64 manifest triangle count mismatch")
    require(unchanged_rows == 149, "expected 149 byte-identical non-target surface rows")

    comp = final_manifest["source"]["conforming_edge_refinement_composition"]
    return {
        "composer_dir": str(composition_dir.resolve()),
        "composition_report": str(report_path.resolve()),
        "composition_report_sha256": sha(report_path),
        "composer_source_path": str(composer_source.resolve()),
        "composer_source_sha256_recorded": recorded_composer_sha,
        "composer_source_sha256_current": sha(composer_source),
        "composer_source_matches_current": recorded_composer_sha == sha(composer_source),
        "candidate_payload": str(payload_path.resolve()),
        "candidate_payload_sha256": sha(payload_path),
        "candidate_manifest": str(manifest_path.resolve()),
        "candidate_manifest_sha256": sha(manifest_path),
        "candidate_receipt": str(candidate_receipt_path.resolve()),
        "candidate_receipt_sha256": sha(candidate_receipt_path),
        "candidate_native_muscle_surfaces_record": final_muscle,
        "base_native_muscle_surfaces_record": base_muscle,
        "relocated_cardiac_common_field_assets": [
            {"path": str(path), "sha256": sha(path)} for path in relocated_common
        ],
        "unchanged_non_target_rows": unchanged_rows,
        "source_composition_provenance": comp,
        "stable64_operation": operation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-dir", type=Path, required=True,
                        help="composer output directory containing the full150-row NHTISS, manifest and report")
    parser.add_argument("--anatomy-receipt", type=Path, required=True,
                        help="composer-bound resting-anatomy-receipt.json")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--minimum-free-gib", type=float, default=1.5)
    parser.add_argument("--validate-only", action="store_true",
                        help="validate a composition interface without creating a runnable directory")
    args = parser.parse_args()
    require(args.name and all(c.isalnum() or c == "-" for c in args.name),
            "--name must contain only letters, digits and hyphens")
    require(math.isfinite(args.minimum_free_gib) and args.minimum_free_gib > 0,
            "minimum free-space guard must be finite and positive")
    dest = TASK / args.name
    composition_dir = args.composition_dir.resolve()
    receipt_path = args.anatomy_receipt.resolve()
    candidate = validate_candidate(composition_dir, receipt_path)
    if args.validate_only:
        print(json.dumps({"status": "composition_interface_validated_only",
                          "native_preparation_created": False,
                          "candidate": candidate}, indent=2))
        return 0
    require(candidate["composer_source_matches_current"],
            "composer source differs from the source hash recorded by this composition; recompose from frozen source")
    require(not dest.exists(), f"refusing to overwrite {dest}")
    free_bytes = shutil.disk_usage(ROOT).free
    require(free_bytes >= args.minimum_free_gib * (1024 ** 3),
            f"insufficient disk headroom: {free_bytes} bytes available")

    declaration_path = PARENT / "run-declaration.json"
    execution_path = PARENT / "execution.json"
    parent_run_metadata = PARENT / "native-run/run-metadata.json"
    require(declaration_path.is_file() and execution_path.is_file() and parent_run_metadata.is_file(),
            "1201 completed declaration, execution and native run metadata are required")
    parent = json.loads(declaration_path.read_text())
    execution = json.loads(execution_path.read_text())
    metadata = json.loads(parent_run_metadata.read_text())
    require(execution.get("returncode") == 0 and not execution.get("changed_inputs"),
            "1201 must have completed successfully without changing declared inputs")
    require(metadata.get("exit_code") == 0 and not metadata.get("source_files_changed_during_run"),
            "1201 native metadata must be successful and unchanged")
    require(parent.get("seconds") == 40 and parent.get("dt") == 0.002
            and parent.get("capture_steps") == [0, 20000]
            and parent.get("contact_iterations") == 64,
            "1201 reference protocol identity differs")
    require(parent.get("hip_capsule_enabled") is False, "1201 unexpectedly enabled hip reference")
    original_argv = list(parent["argv"])
    require("--rigid-hands" in original_argv and "--release-initialization" in original_argv,
            "1201 rigid-hands/release-initialization settings missing")
    require("--contoured-bed" not in original_argv and "--hip-capsule-reference" not in original_argv,
            "1201 must use the original flat support without hip reference")
    body_scene = Path(original_argv[original_argv.index("--body-scene") + 1]).resolve()
    scene = json.loads(body_scene.read_text())
    require("heightfield" not in scene.get("bed", {}), "1201 body scene is not the original flat-support scene")
    original_receipt = Path(original_argv[original_argv.index("--anatomy-receipt") + 1]).resolve()
    require(original_receipt == BASE_RECEIPT.resolve(), "1201 anatomy receipt pin differs from accepted039")
    for path, digest in parent["immutable_assets"].items():
        require(Path(path).is_file() and sha(Path(path)) == digest,
                f"1201 immutable input changed: {path}")

    argv = list(original_argv)
    outdir = dest / "native-run"
    argv[argv.index("--output") + 1] = str(outdir)
    argv[argv.index("--anatomy-receipt") + 1] = str(receipt_path)
    changed_env = []
    for i, item in enumerate(argv):
        if item.startswith("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS="):
            argv[i] = "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=" + CAPTURES
            changed_env.append("capture_steps")
        elif item.startswith("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT="):
            argv[i] = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=" + str(outdir / "common-field-failure.json")
            changed_env.append("failure_receipt_path")
    require(set(changed_env) == {"capture_steps", "failure_receipt_path"},
            "could not safely rewrite only the 1201 output/capture paths")
    require(argv[argv.index("--seconds") + 1] == "40", "duration must remain exactly 40 seconds")
    require(argv[argv.index("--dt") + 1] == "0.002", "timestep must remain exactly 2 ms")
    require(argv[argv.index("--contact-iterations") + 1] == "64", "contact iteration count changed")
    require(argv[argv.index("--body-scene") + 1] == str(body_scene), "body scene changed")
    require("--rigid-hands" in argv and "--release-initialization" in argv,
            "rigid hands/release initialization changed")

    dest.mkdir(parents=True)
    asset_hashes = dict(parent["immutable_assets"])
    for path in (Path(candidate["candidate_payload"]), Path(candidate["candidate_manifest"]),
                 Path(candidate["candidate_receipt"]), Path(candidate["composition_report"]),
                 *(Path(item["path"]) for item in candidate["relocated_cardiac_common_field_assets"]),
                 SUBSET_PAYLOAD, SUBSET_MANIFEST, SUBSET_PROOF, SUBSET_PROOF_SCRIPT,
                 PARENT / "run-declaration.json", PARENT / "execution.json",
                 PARENT / "native-run/run-metadata.json", declaration_path, execution_path,
                 body_scene, original_receipt, Path(candidate["composer_source_path"]),
                 Path("/Users/n/numi-human-conforming-composition-source-1216/source-revision.json"),
                 TASK / "compose-final-001/producer-declaration.json",
                 TASK / "compose-final-001/producer-execution.json"):
        asset_hashes[str(path.resolve())] = sha(path)
    composition_report = json.loads(Path(candidate["composition_report"]).read_text())
    for path_text, digest in composition_report.get("input_sha256", {}).items():
        path = Path(path_text).resolve()
        require(path.is_file() and sha(path) == digest,
                f"composer evidence input changed before preparation: {path}")
        asset_hashes[str(path)] = digest
    run_script = dest / "run.py"
    shutil.copyfile(PARENT / "run.py", run_script)
    launch = dest / "launch-command.sh"
    launch.write_text("#!/bin/sh\nexec " + shlex.join(argv) + "\n")
    launch.chmod(0o755)
    asset_hashes[str(run_script)] = sha(run_script)
    asset_hashes[str(launch)] = sha(launch)
    asset_hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())

    # The old 1201 inputs remain pinned and are never overwritten; the new
    # anatomy receipt selects the only changed physical geometry payload.
    declaration = {
        "status": "prepared_only_native_not_launched",
        "candidate_source_revision": "2eace39df60d72ba1197a775aea6ca4205d6e484",
        "parent_run": str(PARENT),
        "parent_declaration_sha256": sha(declaration_path),
        "parent_execution_sha256": sha(execution_path),
        "parent_native_metadata_sha256": sha(parent_run_metadata),
        "argv": argv,
        "immutable_assets": asset_hashes,
        "seconds": 40,
        "dt": 0.002,
        "capture_steps": [0, 10000, 20000],
        "contact_iterations": 64,
        "postural_activation_cap": 0.01,
        "body_scene": str(body_scene),
        "flat_support_unchanged": True,
        "hip_reference_enabled": False,
        "rigid_hands_enabled": True,
        "release_initialization": True,
        "candidate": candidate,
        "allowed_changes_vs_1201": [
            "native_muscle_surfaces NHTISS payload and matching manifest",
            "native_muscle_surfaces record in the anatomy receipt",
            "output directory and failure-receipt output path",
            "captured accepted steps 0, 10000, and 20000",
        ],
        "unchanged_1201_physical_and_runtime_inputs": True,
        "disk_free_bytes_at_preparation": free_bytes,
        "launch_status": "not launched by preparation",
        "qualification_boundary": (
            "A future40s replay with three captures tests this refined NHTISS package only. "
            "It is not full-breath clearance or a 300s anatomy qualification."
        ),
    }
    (dest / "run-declaration.json").write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    (dest / "preparation-verification.json").write_text(
        json.dumps({"status": "pass", "candidate": candidate,
                    "parent_1201_declaration_sha256": sha(declaration_path),
                    "prepared_argv": argv, "native_launched": False}, indent=2, sort_keys=True) + "\n"
    )
    (dest / "launch-command.sh").write_text("#!/bin/sh\nexec " + shlex.join(argv) + "\n")
    (dest / "launch-command.sh").chmod(0o755)
    print(json.dumps({"status": "prepared", "directory": str(dest),
                      "declaration_sha256": sha(dest / "run-declaration.json"),
                      "verification_sha256": sha(dest / "preparation-verification.json"),
                      "native_launched": False, "capture_steps": [0, 10000, 20000]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
