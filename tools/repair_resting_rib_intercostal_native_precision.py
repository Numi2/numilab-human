#!/usr/bin/env python3
"""Reproduce the witnessed resting rib/intercostal precision repair.

Source preparation only. Retain every source attribute, slot, rigid owner and
attachment map; replay existing bounded topology constructors and ABI encoders.
This command does not grant anatomy, retained-motion or native admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numilab_human import composed_bone_body_frame as bones, model
from numilab_human import surface_precision_retriangulation as precision
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
from numilab_human.resting_pleura_proxy import _parse_payload

BONES_SHA = "77186bce8808c21841785a395ac6a0d46402874d2be5fda43411f06593c26bf6"
NHA_SHA = "6edd6c7bf48eeeba445606347ff585afb199e9b5497efec6cb01ea40d11b8768"
CANDIDATE_SHA = {
    "51004-49.npz": "233f42de3b53404dff819948ee4702cf7bb87120e6d766f91b3417d583ce9e0f",
    "51010-313.npz": "66419827f64233f5e55e122e437e4000c2a5f10b3803e9b54d16c7da42c93b79",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bones-artifact", type=Path, required=True)
    parser.add_argument("--nha", type=Path, required=True)
    parser.add_argument("--witness-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = args.bones_artifact.resolve()
    source_bones = current/"bodyparts3d-myosim-major-bones.nhbones"
    if digest(source_bones) != BONES_SHA or digest(args.nha) != NHA_SHA:
        raise ValueError("precision recipe requires its exact retained source owners")
    for name, expected in CANDIDATE_SHA.items():
        if digest(args.witness_candidate/name) != expected:
            raise ValueError("precision witness candidate drifted: "+name)
    args.output.mkdir(exist_ok=False)
    source_manifest = json.loads((current/"bodyparts3d-myosim-major-bones.manifest.json").read_text())
    for stage, (keep, remove, flips, bound) in enumerate((
            (7527, 7528, [[7522, 7525]], 2e-8),
            (7467, 7469, [], 2e-6),
            (10565, 10628, [], 2e-6))):
        output = args.output/("compiled" if stage == 2 else f"rib49-stage{stage}")
        output.mkdir()
        payload = current/source_bones.name
        mp = current/"bodyparts3d-myosim-major-bones.manifest.json"
        parameters = dict(stable_id=49, keep_vertex=keep, remove_vertex=remove,
                          flip_edges=flips, max_endpoint_displacement_m=bound,
                          minimum_altitude_m=1e-12, max_abs_volume_delta_m3=2e-12)
        raw, proof = bones.repair_bone_source_precision_edges(payload.read_bytes(), **parameters)
        output_payload = output/payload.name
        output_payload.write_bytes(raw)
        manifest = json.loads(mp.read_text())
        manifest["payload"].update(sha256=digest(output_payload), bytes=len(raw),
                                   index_count=bones._BONE_HEADER.unpack_from(raw)[4])
        manifest["source_precision_retriangulation"] = {
            "input_artifact": str(current), "input_manifest_sha256": digest(mp),
            "input_payload_sha256": digest(payload), "parameters": parameters, "proof": proof}
        write_json(output/mp.name, manifest)
        bones.validate_composed_rib_reframe(output, manifest, raw)
        current = output
    header, records, _, io = bones._decode_bones(raw)
    record = next(row for row in records if row[5] == 49)
    faces = np.frombuffer(raw, "<u4", count=record[4], offset=io+record[3]*4).reshape(-1, 3)
    with np.load(args.witness_candidate/"51004-49.npz", allow_pickle=False) as witness:
        if not np.array_equal(faces.astype(np.int64)-record[1], witness["faces"]):
            raise ValueError("bone source recipe differs from the witnessed candidate")

    original = args.nha.read_bytes()
    _, _, rows = _parse_payload(original)
    row = dict(rows[313])
    attributes = np.asarray(row["vertices6"], dtype="<f4")
    faces = np.asarray(row["faces"])
    lineage = [[i] for i in range(len(faces))]
    operations = []
    for index, (keep, remove, bound) in enumerate((
            (101843, 101838, 2e-8), (99273, 99073, 2e-6), (97484, 97485, 2e-6))):
        faces, proof = precision.collapse_closed_component_edge_preserving_allocations(
            attributes[:, :3], faces, identity=(51010, 313),
            keep_vertex=keep, remove_vertex=remove,
            max_endpoint_displacement_m=bound, max_abs_volume_delta_m3=2e-12)
        lineage = [lineage[i] for i in proof["kept_parent_face_origins"]]
        operations.append(proof)
        if index == 0:
            for edge in ((101839, 101842), (16933, 16935)):
                faces, proof = precision.flip_closed_surface_edge_preserving_allocations(
                    attributes[:, :3], faces, identity=(51010, 313), edge=edge,
                    minimum_altitude_m=1e-12, repair_folded_source=True)
                pair = proof["changed_parent_face_indices"]
                parents = sorted({parent for i in pair for parent in lineage[i]})
                for i in pair:
                    lineage[i] = parents.copy()
                operations.append(proof)
    with np.load(args.witness_candidate/"51010-313.npz", allow_pickle=False) as witness:
        if (not np.array_equal(faces, witness["faces"])
                or not np.array_equal(attributes, witness["source_vertices6"])):
            raise ValueError("intercostal source recipe differs from the witnessed candidate")
    row["faces"] = faces
    payload, emission = encode_payload_with_row_replacements(original, {313: row})
    nha = current/"resting-thorax.nhanatomy"
    nha.write_bytes(payload)
    write_json(args.output/"intercostal313-face-ancestry.json", {
        "identity": [51010, 313], "input_payload": str(args.nha.resolve()),
        "input_payload_sha256": NHA_SHA, "operations": operations,
        "output_face_parent_origins": lineage})
    decoded, metadata, path = model._numi_human_bone_envelope_surfaces(
        current, source_manifest["source"]["myosim_source_archive_sha256"])
    if sum(map(len, decoded.values())) != 185:
        raise ValueError("precision source recipe changed the complete bone inventory")
    result = {
        "status": "PASS_EXISTING_SOURCE_OWNER_EXPORT_ONLY",
        "source_bones_sha256": BONES_SHA, "source_nha_sha256": NHA_SHA,
        "output_nha_path": str(nha.resolve()), "output_nha_sha256": digest(nha),
        "output_bones_path": str(path), "output_bones_sha256": metadata["sha256"],
        "output_bones_manifest_sha256": digest(current/"bodyparts3d-myosim-major-bones.manifest.json"),
        "candidate_npz_pins": CANDIDATE_SHA, "nha_emission": emission,
        "full_existing_185_bone_envelope_loader_passed": True,
        "source_xyz_normals_allocations_and_attachment_maps_exact": True,
        "physical_mass_or_state_changed": False,
        "anatomical_admission": False, "native_admission": False}
    write_json(args.output/"source-preparation-report.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
