#!/usr/bin/env python3
"""Run the bounded all-captured-target exact surface stage for both veins.

This consumes the offline prediction receipt and existing accepted capture
packs. It does not launch simulation and cannot establish lumen continuity,
calibre, clearance, native common-field parity, or anatomical admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
P6 = Path("/Users/n/numi-human-anatomy-input-guard-publish-1386/src")
DEFAULT_CAPTURE_MANIFEST = A / "accepted-capture-inputs-1398-001.json"
EXPECTED_STEPS = (0, 3807, 4767, 5023, 5599, 6207, 7423, 8000)
P6_VERIFIER_SHA256 = "c9f47cb68fdef1acb4d544dcceb06775c38d555db85d1618131efed3b68fbb9c"
P6_PREDICATE_SHA256 = "4e6fb1f4ee75ea0ba9cdd06a072702a238f066fc8e63a61aa9eae32f2356a384"
BRIDGES = {525: "FJ3482", 526: "FJ3583"}
PORTS = {525: (10, 469, 473), 526: (10, 468, 472)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("captured brachiocephalic surface audit: " + message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arr_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_local_modules():
    if str(P6) not in sys.path:
        sys.path.insert(0, str(P6))
    from numilab_human import resting_anatomy_verify as av
    from numilab_human import cardiac_cavity_intersections as ci
    import importlib.util
    helper_path = Path(__file__).resolve().parents[1] / "src/numilab_human/resting_brachiocephalic_geometry_gate.py"
    helper_spec = importlib.util.spec_from_file_location("_local_brachiocephalic_geometry_gate", helper_path)
    require(helper_spec is not None and helper_spec.loader is not None,
            "local bounded surface-stage helper cannot be imported")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    BRIDGE_KEYS = helper.BRIDGE_KEYS
    audit_captured_pose_surface_intersections = helper.audit_captured_pose_surface_intersections
    require(sha_file(P6 / "numilab_human/resting_anatomy_verify.py") == P6_VERIFIER_SHA256,
            "frozen P6 shared verifier source hash changed")
    require(sha_file(P6 / "numilab_human/cardiac_cavity_intersections.py") == P6_PREDICATE_SHA256,
            "frozen P6 exact predicate source hash changed")
    return av, ci, BRIDGE_KEYS, audit_captured_pose_surface_intersections


def load_prediction_owner():
    path = Path(__file__).resolve().with_name("predict_brachiocephalic_captured_svc_poses.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_captured_svc_prediction_owner", path)
    require(spec is not None and spec.loader is not None,
            "captured SVC prediction owner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_witness_pairs(path: Path, rows: dict, base_payload_sha256: str) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    witness = json.loads(raw)
    require(witness.get("schema") == "numi.anatomy.bridge-collar-ancestry.v1",
            "port witness schema is not the frozen source ancestry schema")
    require(witness.get("current_nha_sha256") == base_payload_sha256,
            "port witness is not bound to this candidate's pinned source-rest NHA")
    members = {row.get("member"): row for row in witness.get("members", [])}
    require(set(members) == {"FJ3482", "FJ3583"},
            "port witness must contain exactly the left and right source bridges")
    pairs = {}
    for stable_id, member_name in BRIDGES.items():
        member = members[member_name]
        require(set(member.get("ports", {})) == {str(i) for i in PORTS[stable_id]},
                f"{member_name} does not have exactly its three declared ports")
        bridge = rows[stable_id]
        bridge_xyz = np.ascontiguousarray(bridge["vertices6"][:, :3], dtype="<f4")
        bridge_faces = np.ascontiguousarray(bridge["faces"], dtype="<u4")
        bridge_xyz_sha, bridge_faces_sha = arr_sha(bridge_xyz), arr_sha(bridge_faces)
        for target_id in PORTS[stable_id]:
            port = member["ports"][str(target_id)]
            bridge_witness = port.get("bridge_witness", {})
            require(bridge_witness.get("source_row_xyz_bytes_sha256") == bridge_xyz_sha
                    and bridge_witness.get("source_row_face_bytes_sha256") == bridge_faces_sha,
                    f"{member_name} port {target_id} bridge source row differs from NHA candidate")
            target = rows.get(target_id)
            require(target is not None, f"candidate payload lacks port target {target_id}")
            target_faces = np.ascontiguousarray(target["faces"], dtype="<u4")
            target_record = target.get("record")
            require(target_record is not None and len(target_record) >= 2,
                    f"candidate target {target_id} lacks its NHA vertex-range record")
            target_vertex_start = int(target_record[1])
            local_face_ids = [int(i) for i in port.get("nha_target_local_face_ids", [])]
            triplets = np.asarray(port.get("nha_target_triangle_global_vertex_triplets", []),
                                  dtype="<u4")
            require(len(local_face_ids) > 0 and triplets.shape == (len(local_face_ids), 3)
                    and max(local_face_ids) < len(target_faces),
                    f"{member_name} target {target_id} witness has invalid source-face ancestry")
            candidate_global_triplets = target_faces[local_face_ids] + np.uint32(target_vertex_start)
            require(candidate_global_triplets.tobytes() == triplets.tobytes(),
                    f"{member_name} target {target_id} source face ancestry differs from candidate NHA")
            exact_pairs = port.get("exact_face_pairs", [])
            pair_array = np.asarray(exact_pairs, dtype=np.int64)
            require(pair_array.ndim == 2 and pair_array.shape[1] == 2 and len(pair_array) > 0
                    and np.array_equal(pair_array[:, 0], np.asarray(
                        bridge_witness.get("bridge_pair_face_ids_for_each_pair", []), dtype=np.int64))
                    and np.array_equal(pair_array[:, 1], np.asarray(
                        port.get("target_face_ids_for_each_pair", []), dtype=np.int64)),
                    f"{member_name} target {target_id} pair ancestry is inconsistent")
            pairs[((51011, stable_id), (51011, target_id))] = pair_array.tolist()
    require(witness.get("status") == "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
            "port witness status is not the frozen source-only diagnostic")
    return pairs, hashlib.sha256(raw).hexdigest()


def _check_prediction_report(report_path: Path, npz_path: Path,
                             candidate_payload_path: Path,
                             candidate_payload_sha256: str) -> tuple[dict, dict]:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    require(report.get("status") == "OFFLINE_PREDICTIONS_READY_NOT_GEOMETRICALLY_ADMITTED",
            "prediction report is not the bounded offline predictor receipt")
    require(report.get("anatomical_admission") is False
            and report.get("all_pose_admission") is False,
            "prediction report carries an unexpected admission claim")
    require(sha_file(npz_path) == report.get("output_npz_sha256"),
            "predicted positions NPZ differs from its receipt")
    pins = report.get("inputs", {})
    payload_pin = pins.get(str(candidate_payload_path.resolve()))
    require(payload_pin == candidate_payload_sha256,
            "prediction report was created for a different candidate payload")
    poses = report.get("poses", [])
    require(tuple(int(row.get("step", -1)) for row in poses) == EXPECTED_STEPS,
            "prediction report does not contain the exact eight accepted steps")
    archive = np.load(npz_path, allow_pickle=False)
    require(set(archive.files) == {f"step_{step}_stable_{sid}_xyz_f32"
                                   for step in EXPECTED_STEPS for sid in BRIDGES},
            "prediction NPZ has missing or extra bridge arrays")
    arrays = {name: archive[name] for name in archive.files}
    archive.close()
    for pose in poses:
        for sid, member in BRIDGES.items():
            array = arrays[f"step_{pose['step']}_stable_{sid}_xyz_f32"]
            row = pose["bridges"][member]
            require(array.dtype == np.dtype("<f4") and array.ndim == 2 and array.shape[1] == 3
                    and arr_sha(array) == row.get("predicted_positions_sha256"),
                    f"predicted Float32 position array differs at step {pose['step']} id {sid}")
    return report, arrays


def run(candidate_receipt_path: Path, preparation_path: Path,
        prediction_report_path: Path, prediction_npz_path: Path,
        port_witness_path: Path, capture_manifest_path: Path,
        output_dir: Path) -> dict:
    owner, prediction_owner_path = load_prediction_owner()
    inputs = owner.validated_inputs(candidate_receipt_path, preparation_path,
                                   capture_manifest_path)
    av, ci, bridge_keys, geometry_gate = load_local_modules()
    candidate_payload_sha256 = inputs["payload_sha256"]
    witness_pairs, witness_sha256 = load_witness_pairs(
        port_witness_path, inputs["rows"], inputs["base_payload_sha256"])
    prediction_report, prediction_arrays = _check_prediction_report(
        prediction_report_path.resolve(), prediction_npz_path.resolve(),
        inputs["payload_path"], candidate_payload_sha256)
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "output directory already exists")
    output_dir.mkdir(parents=True)
    progress_path = output_dir / "progress.json"
    pins = {
        str(inputs["receipt_path"]): inputs["receipt_sha256"],
        str(inputs["payload_path"]): candidate_payload_sha256,
        str(inputs["preparation_path"]): inputs["preparation_sha256"],
        str(inputs["common_map_path"]): inputs["common_map_sha256"],
        str(inputs["capture_manifest_path"]): inputs["capture_manifest_sha256"],
        str(Path(prediction_report_path).resolve()): sha_file(prediction_report_path),
        str(Path(prediction_npz_path).resolve()): sha_file(prediction_npz_path),
        str(Path(port_witness_path).resolve()): witness_sha256,
        str(prediction_owner_path): sha_file(prediction_owner_path),
        str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
        str(P6 / "numilab_human/resting_anatomy_verify.py"): sha_file(
            P6 / "numilab_human/resting_anatomy_verify.py"),
        str(P6 / "numilab_human/cardiac_cavity_intersections.py"): sha_file(
            P6 / "numilab_human/cardiac_cavity_intersections.py"),
        str(Path(__file__).resolve().parents[1] /
            "src/numilab_human/resting_brachiocephalic_geometry_gate.py"): sha_file(
            Path(__file__).resolve().parents[1] /
            "src/numilab_human/resting_brachiocephalic_geometry_gate.py"),
    }
    progress = {"schema": "numi.human.brachiocephalic-captured-pose-surface-audit.v1",
                "status": "RUNNING_SURFACE_INTERSECTION_STAGE_ONLY",
                "input_sha256": pins, "poses": [],
                "lumen_continuity": "not_evaluated", "calibre_continuity": "not_evaluated",
                "native_confirmation": "required", "anatomical_admission": False,
                "all_pose_admission": False}
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    for capture, predicted_pose in zip(inputs["captures"], prediction_report["poses"], strict=True):
        step = int(capture["step"])
        require(step == int(predicted_pose["step"]), "capture/prediction pose order differs")
        pack, receipt = Path(capture["pack_path"]), Path(capture["receipt_path"])
        require(sha_file(pack) == capture["recorded_pack_sha256"]
                and sha_file(receipt) == capture["receipt_sha256"],
                f"capture inputs differ from the accepted manifest at step {step}")
        surfaces, identity = av.load_capture(pack, receipt)
        require(identity["pack_sha256"] == predicted_pose["pack_sha256"]
                and identity["capture_receipt_sha256"] == predicted_pose["receipt_sha256"]
                and identity["accepted_step"] == step
                and abs(float(identity["accepted_time_s"]) - float(capture["accepted_time_s"])) <= 1e-12
                and abs(float(predicted_pose["accepted_time_s"]) - float(capture["accepted_time_s"])) <= 1e-12,
                f"prediction and decoded capture identity differs at step {step}")
        for target_id in sorted({target for ids in PORTS.values() for target in ids}):
            target_key = (51011, target_id)
            require(target_key in surfaces and target_id in inputs["rows"],
                    f"accepted pack or source candidate lacks port target {target_id} at step {step}")
            target_source_faces = np.asarray(inputs["rows"][target_id]["faces"], dtype="<u4")
            target_capture_faces = np.asarray(surfaces[target_key][1], dtype="<u4")
            require(target_capture_faces.tobytes() == target_source_faces.tobytes(),
                    f"captured port target {target_id} face order differs from candidate NHA at step {step}")
        for sid, member in BRIDGES.items():
            target_faces = np.asarray(inputs["rows"][sid]["faces"], dtype="<u4")
            expected_count = int(inputs["preparation"]["bridges"][member]["triangle_count"])
            require(len(target_faces) == expected_count,
                    f"candidate bridge face count differs for {member}")
            captured_key = f"step_{step}_stable_{sid}_xyz_f32"
            positions = prediction_arrays[captured_key]
            require(len(positions) == len(inputs["rows"][sid]["vertices6"]),
                    f"predicted bridge vertex count differs for {member} at step {step}")
        bridge_positions = {
            (51011, sid): prediction_arrays[f"step_{step}_stable_{sid}_xyz_f32"]
            for sid in BRIDGES
        }
        bridge_faces = {
            (51011, sid): np.asarray(inputs["rows"][sid]["faces"], dtype="<u4")
            for sid in BRIDGES
        }
        audit = geometry_gate(av, surfaces, bridge_positions, bridge_faces, witness_pairs)
        audit.update({"step": step, "accepted_time_s": float(capture["accepted_time_s"]),
                      "pack_sha256": identity["pack_sha256"],
                      "capture_receipt_sha256": identity["capture_receipt_sha256"]})
        progress["poses"].append(audit)
        progress["last_completed_step"] = step
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    complete = all(row["surface_intersection_stage_passed"] for row in progress["poses"])
    report = {
        **progress,
        "status": ("EIGHT_CAPTURE_SURFACE_STAGE_PASSED_CONTINUITY_UNRESOLVED" if complete
                   else "EIGHT_CAPTURE_SURFACE_STAGE_REJECTED"),
        "all_eight_surface_intersection_stages_passed": complete,
        "limits": [
            "Only decoded accepted-pack anatomy surfaces are targets; no comparison to undrawn NHA rows was made.",
            "The frozen P6 verifier allows only the supplied exact triangle face-pair identities at the three declared ports.",
            "Face-pair identity does not establish the geometric intersection locus, solid containment, or lumen continuity.",
            "No cross-section area, minimum diameter, or clearance margin was evaluated.",
            "Offline SVC displacement predictions are not a native common-field replay.",
            "Native combined 16-pose confirmation and separate lumen/calibre evidence remain required.",
        ],
        "anatomical_admission": False,
        "all_pose_admission": False,
    }
    report_path = output_dir / "surface-stage-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress_path.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--common-field-preparation", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--prediction-npz", type=Path, required=True)
    parser.add_argument("--port-witness-report", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.candidate_receipt, args.common_field_preparation,
                 args.prediction_report, args.prediction_npz,
                 args.port_witness_report, args.capture_manifest, args.output_dir)
    print(json.dumps({"status": result["status"],
                      "pose_steps": [row["step"] for row in result["poses"]],
                      "all_eight_surface_intersection_stages_passed":
                          result["all_eight_surface_intersection_stages_passed"],
                      "anatomical_admission": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
