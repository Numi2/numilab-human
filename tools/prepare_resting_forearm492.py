#!/usr/bin/env python3
"""Prepare the reviewed FDS492 source through existing NHTISS/parent owners.

This replaces row132 and parent source0 only. It is source preparation, not
anatomical, accepted-motion, or native admission. FDP134 and back rows stay exact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from numilab_human import passive_attachment_composition as composition
from numilab_human.passive_tissue_parent_carriers import CarrierRow, encode_carriers
from numilab_human.resting_anatomy_verify import keyed_row_indices

CANDIDATE_SHA = "5b49cc37518a3a43704240afeece534b957483541aa2b9e60123570c1215d00f"
BASE_RECEIPT_SHA = "86a227bf20861cd68c9597e91d15fa3fd4d03c696d85c29229fa3f45c373ed31"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_owner(extension, replacement, carrier_path, carrier_sha, child_sha):
    """Replace the exclusive existing FDS source slot, never append FDP state."""
    require(len(extension["source_payloads"]) <= 8, "native parent source capacity")
    require(replacement["stable_id"] == 132, "replacement must be FDS132")
    selected = [row for row in extension["surfaces"] if row["stable_id"] == 132]
    require(len(selected) == 1, "exactly one existing FDS owner required")
    for group in ("surfaces", "anatomy_surfaces"):
        for row in extension.get(group, []):
            if group == "surfaces" and row["stable_id"] == 132:
                continue
            require(all(term[0] != 0 for terms in row["vertices"] for term in terms),
                    "source0 is used by an unrelated surface")
    result = copy.deepcopy(extension)
    result["source_payloads"][0] = {"path": str(carrier_path), "sha256": carrier_sha}
    result["child_payload_sha256"] = child_sha
    for index, row in enumerate(result["surfaces"]):
        if row["stable_id"] == 132:
            result["surfaces"][index] = replacement
    require(result["source_payloads"][1:] == extension["source_payloads"][1:],
            "unrelated source slots changed")
    return result


def prepare(base_path, candidate_path, handoff_path, ancestry_path, output):
    base_path = Path(base_path).resolve(strict=True)
    candidate_path = Path(candidate_path).resolve(strict=True)
    handoff_path = Path(handoff_path).resolve(strict=True)
    ancestry_path = Path(ancestry_path).resolve(strict=True)
    output = Path(output).resolve()
    require(not output.exists(), "preserve previous preparation")
    require(sha(base_path) == BASE_RECEIPT_SHA and sha(candidate_path) == CANDIDATE_SHA,
            "reviewed base or FDS492 candidate changed")
    receipt = json.loads(base_path.read_text())
    description = receipt["provenance"]["native_muscle_surfaces"]
    child_path = Path(description["payload_path"]).resolve(strict=True)
    require(sha(child_path) == description["sha256"], "base muscle payload changed")
    require(sha(description["manifest_path"]) == description["manifest_sha256"],
            "base muscle manifest changed")
    ancestry = json.loads(ancestry_path.read_text())
    require(ancestry.get("candidate_sha256") == CANDIDATE_SHA
            and ancestry.get("positions_and_material_maps_byteexact_to_491") is True
            and ancestry.get("changed_face_ids") == [24306, 24783],
            "local two-face remesh ancestry changed")
    for raw, digest in ancestry["inputs_sha256"].items():
        require(sha(raw) == digest, "source lineage input changed: " + raw)
    extension = receipt["passive_tissue_parent_bindings"]
    require(extension["child_payload_sha256"] == description["sha256"],
            "child/parent owner binding mismatch")
    for source in extension["source_payloads"]:
        require(sha(source["path"]) == source["sha256"], "retained carrier changed")
    tissue = composition._read_nhtiss4(child_path)
    old_rows = {int(row[6]): composition._row_slices(tissue, row)
                for row in tissue["records"]}
    old = old_rows[132]
    count = old["vertex_count"]
    old_vertices = np.ndarray((count, 6), dtype="<f4",
                              buffer=old["vertex_bytes"], strides=(56, 4))
    old_lanes = np.ndarray((count, 4), dtype="<u4",
                          buffer=old["vertex_bytes"], offset=24, strides=(56, 4))
    old_weights = np.ndarray((count, 4), dtype="<f4",
                            buffer=old["vertex_bytes"], offset=40, strides=(56, 4))
    with np.load(candidate_path, allow_pickle=False) as candidate, np.load(
            handoff_path, allow_pickle=False) as handoff:
        gids = candidate["full_handoff_gids"]
        require(np.array_equal(gids, handoff["gids"]), "source GID allocation order changed")
        keyed_row_indices(gids, gids)
        positions = candidate["full_handoff_source_positions_after_f32"]
        normals = candidate["full_handoff_source_normals_after_f32"]
        faces = candidate["full_handoff_faces_local"]
        source_weights = candidate["full_handoff_source_weights_float64"]
        cores = candidate["core_body_indices"]
        require(len(gids) == count and positions.dtype == normals.dtype == np.dtype("<f4")
                and positions.shape == normals.shape == (count, 3)
                and source_weights.dtype == np.dtype("<f8")
                and faces.dtype == np.dtype("<u4"), "exact source array shape/dtype")
        vertices6 = np.column_stack((positions, normals)).astype("<f4")
        require(np.isfinite(vertices6).all() and np.isfinite(source_weights).all(),
                "nonfinite source fields")
        require(np.max(np.abs(np.linalg.norm(normals.astype(float), axis=1)-1)) <= 2e-3,
                "undefined source normals")
        binding = old["binding_bytes"]
        actual_cores = [struct.unpack_from("<I", binding, offset)[0]
                        for offset in range(0, len(binding), 36)]
        require(actual_cores == cores.tolist(), "existing ordered body basis changed")
        require(np.array_equal(handoff["source_positions_f32"], old_vertices[:, :3]),
                "base child no longer matches source471 allocation")
        unused = np.setdiff1d(np.arange(count), np.unique(faces))
        require(len(unused) == 4
                and np.array_equal(vertices6[unused], old_vertices[unused]),
                "unused ancestry slots changed")
        carrier_bytes, surfaces, carrier_report = encode_carriers(
            [CarrierRow(132, vertices6, faces, binding, source_weights, layer=1)],
            registration_fingerprint=tissue["fingerprint"],
            source_digest=tissue["source_digest"])
        patch_arrays = {"vertices6": vertices6, "binding_indices": old_lanes,
                        "weights": old_weights, "faces": faces}
        changed_positions = int(np.any(positions != old_vertices[:, :3], axis=1).sum())
        changed_normals = int(np.any(normals != old_vertices[:, 3:], axis=1).sum())
    pins = {str(path): sha(path) for path in (
        base_path, candidate_path, handoff_path, ancestry_path, child_path,
        Path(description["manifest_path"]), Path(__file__).resolve(),
        Path(composition.__file__).resolve())}
    output.mkdir()
    patch_path = output / "row-132.npz"
    np.savez_compressed(patch_path, **patch_arrays)
    patch_report = output / "row-132-report.json"
    proof = {
        "status": "prepared_experimental_not_admitted",
        "stable_id": 132, "source_ancestry_report": str(ancestry_path),
        "source_ancestry_report_sha256": sha(ancestry_path),
        "candidate_sha256": CANDIDATE_SHA,
        "changed_source_positions": changed_positions,
        "changed_source_normals": changed_normals,
        "unchanged_unused_ancestry_slots": 4,
        "attachment_body_basis_and_child_lanes_unchanged": True,
        "gpu_run_performed": False, "physical_mass_and_state_added": False,
    }
    patch_report.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    composed = composition.compose(child_path, output / "assets",
                                   [(132, patch_path, patch_report)],
                                   reference_surface_rows=(132,))
    new_child = output / "assets" / child_path.name
    new_tissue = composition._read_nhtiss4(new_child)
    require(new_tissue["raw"][:64] == tissue["raw"][:64], "header/state inventory changed")
    for row in new_tissue["records"]:
        sid = int(row[6])
        actual = composition._row_slices(new_tissue, row)
        if sid != 132:
            require(composition._row_local_equal(actual, old_rows[sid]),
                    "unrelated muscle or back row changed")
    carrier_path = output / "assets" / "shared-field-carriers-132.nhtissue"
    carrier_path.write_bytes(carrier_bytes)
    result = copy.deepcopy(receipt)
    result["passive_tissue_parent_bindings"] = replace_owner(
        extension, surfaces[0], carrier_path, sha(carrier_path), sha(new_child))
    new_manifest = new_child.with_suffix(".manifest.json")
    result["provenance"]["native_muscle_surfaces"].update(
        payload_path=str(new_child), sha256=sha(new_child),
        manifest_path=str(new_manifest), manifest_sha256=sha(new_manifest))
    result["provenance"]["forearm492_source_preparation"] = proof
    require(result["functional_bindings"] == receipt["functional_bindings"]
            and result["mass_geometry_accounting"] == receipt["mass_geometry_accounting"],
            "passive repair changed functional state or mass ownership")
    target = output / "resting-anatomy-receipt.json"
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    report = {**proof, "input_sha256": pins, "composition": composed,
              "carrier_summary": carrier_report,
              "receipt_path": str(target), "receipt_sha256": sha(target),
              "child_path": str(new_child), "child_sha256": sha(new_child),
              "carrier_path": str(carrier_path), "carrier_sha256": sha(carrier_path),
              "parent_source_count": len(extension["source_payloads"]),
              "unchanged_rows": sorted(set(old_rows)-{132})}
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "candidate", "source-handoff", "ancestry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.base, args.candidate, args.source_handoff, args.ancestry, args.output)
    print(json.dumps({key: result[key] for key in (
        "status", "receipt_path", "receipt_sha256", "child_sha256",
        "carrier_sha256", "parent_source_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
