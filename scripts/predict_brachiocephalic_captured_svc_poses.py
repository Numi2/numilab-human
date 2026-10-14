#!/usr/bin/env python3
"""Build offline bridge-pose predictions from the pinned eight accepted captures.

This reads immutable captured targets and never launches simulation. Output is
only a prerequisite for separate all-target, collar, calibre, and native gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
P6 = Path("/Users/n/numi-human-anatomy-input-guard-publish-1386/src")
DEFAULT_CAPTURE_MANIFEST = A / "accepted-capture-inputs-1398-001.json"
CAPTURE_MANIFEST_SHA256 = "3df61c8bf202e343a9124588fa94be17e9876db17d1a6ef5dd5fa615aaa7b815"
EXPECTED_STEPS = (0, 3807, 4767, 5023, 5599, 6207, 7423, 8000)
N27_RECEIPT_SHA256 = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
N27_PAYLOAD_SHA256 = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
REGISTRATION_SHA256 = "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"
RIGID_SHA256 = "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
ARCHIVE_SHA256 = "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
BRIDGES = {525: "FJ3482", 526: "FJ3583"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("captured SVC bridge prediction: " + message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_payload(receipt_path: Path, receipt: dict) -> Path:
    path = Path(receipt["payload"]["path"])
    if not path.is_absolute():
        path = receipt_path.parent / path
    path = path.resolve()
    require(path.is_file() and sha_file(path) == receipt["payload"].get("sha256"),
            "receipt payload path/hash mismatch")
    return path


def resolve_sidecar(receipt_path: Path, base_receipt_path: Path, record: dict) -> Path:
    raw = Path(record["path"])
    options = [raw] if raw.is_absolute() else [
        receipt_path.parent / raw, base_receipt_path.parent / raw]
    for path in options:
        if path.is_file() and sha_file(path) == record.get("sha256"):
            return path.resolve()
    raise ValueError("common-field sidecar missing/hash mismatch")


def transform_f32(q_xyzw, position_m, vertices_m):
    """Use the retained explicit Float32 quaternion rotation/addition order."""
    f = np.float32
    q = np.asarray(q_xyzw, dtype="<f4")
    t = np.asarray(position_m, dtype="<f4")
    p = np.asarray(vertices_m, dtype="<f4")
    if q.shape != (4,) or t.shape != (3,) or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("invalid accepted Float32 body transform")
    a = q[:3]
    cross = np.stack((
        f(f(a[1] * p[:, 2]) - f(a[2] * p[:, 1])),
        f(f(a[2] * p[:, 0]) - f(a[0] * p[:, 2])),
        f(f(a[0] * p[:, 1]) - f(a[1] * p[:, 0])),
    ), axis=1).astype(f)
    twice = np.stack([f(f(2.0) * cross[:, k]) for k in range(3)], axis=1).astype(f)
    correction = np.stack((
        f(f(f(q[3] * twice[:, 0]) + f(a[1] * twice[:, 2])) - f(a[2] * twice[:, 1])),
        f(f(f(q[3] * twice[:, 1]) + f(a[2] * twice[:, 0])) - f(a[0] * twice[:, 2])),
        f(f(f(q[3] * twice[:, 2]) + f(a[0] * twice[:, 1])) - f(a[1] * twice[:, 0])),
    ), axis=1).astype(f)
    return np.asarray(t + (p + correction), dtype="<f4")


def load_frozen_verifier():
    require(P6.is_dir(), "pinned shared verifier source directory is missing")
    if str(P6) not in sys.path:
        sys.path.insert(0, str(P6))
    from numilab_human import resting_anatomy_verify as av
    from numilab_human.resting_pleura_proxy import _parse_payload
    return av, _parse_payload


def load_predictor():
    path = Path(__file__).resolve().parents[1] / "src/numilab_human/resting_brachiocephalic_pose_predictor.py"
    spec = importlib.util.spec_from_file_location("_vein_pose_predictor", path)
    require(spec is not None and spec.loader is not None, "predictor source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validated_inputs(candidate_receipt_path: Path, preparation_path: Path,
                     capture_manifest_path: Path) -> dict:
    candidate_receipt_path = candidate_receipt_path.resolve()
    preparation_path = preparation_path.resolve()
    capture_manifest_path = capture_manifest_path.resolve()
    for path in (candidate_receipt_path, preparation_path, capture_manifest_path):
        require(path.is_file(), "missing declared input " + str(path))
    manifest_raw = capture_manifest_path.read_bytes()
    require(hashlib.sha256(manifest_raw).hexdigest() == CAPTURE_MANIFEST_SHA256,
            "accepted capture manifest differs from pinned 1398 input")
    manifest = json.loads(manifest_raw)
    captures = manifest.get("capture_rows", [])
    require(tuple(int(row.get("step", -1)) for row in captures) == EXPECTED_STEPS
            and "N27" in manifest.get("source", ""),
            "capture manifest is not the exact eight-pose N27 set")

    receipt_raw = candidate_receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    payload_path = resolve_payload(candidate_receipt_path, receipt)
    preparation_raw = preparation_path.read_bytes()
    preparation = json.loads(preparation_raw)
    require(preparation.get("input_candidate_payload_sha256") == sha_file(payload_path),
            "map preparation belongs to a different candidate payload")
    require(preparation.get("appended_record_count") == 651
            and preparation.get("existing_cvsim_compartment_id") == 4
            and preparation.get("anatomical_admission") is False
            and preparation.get("all_pose_admission") is False
            and preparation.get("native_admission") is False,
            "map preparation scope/status differs from the unadmitted two-bridge source map")

    provenance = receipt.get("provenance", {})
    extension = provenance.get("passive_brachiocephalic_vein_extension", {})
    base_receipt_path = Path(extension.get("base_receipt", "")).resolve()
    require(base_receipt_path.is_file()
            and sha_file(base_receipt_path) == N27_RECEIPT_SHA256
            and extension.get("base_receipt_sha256") == N27_RECEIPT_SHA256
            and extension.get("base_payload_sha256") == N27_PAYLOAD_SHA256,
            "candidate is not a direct append of pinned N27")
    base_receipt = json.loads(base_receipt_path.read_text())
    base_payload_path = resolve_payload(base_receipt_path, base_receipt)
    require(sha_file(base_payload_path) == N27_PAYLOAD_SHA256,
            "pinned N27 anatomy payload changed")
    require(provenance.get("rigid_payload_sha256") == RIGID_SHA256
            and provenance.get("bodyparts_registration_sha256") == REGISTRATION_SHA256
            and any(row.get("hierarchy") == "part_of" and row.get("sha256") == ARCHIVE_SHA256
                    for row in provenance.get("bodyparts3d_archives", [])),
            "NHRIGID, registration, or source archive provenance changed")
    source_rest_paths = {}
    for path_key, expected in (("verified_registration_path", REGISTRATION_SHA256),
                               ("verified_rigid_payload_path", RIGID_SHA256)):
        path = Path(extension.get(path_key, ""))
        require(path.is_file() and sha_file(path) == expected,
                "verified source-rest input changed: " + path_key)
        source_rest_paths[path_key] = path
    context = extension.get("verified_rest_body_context", {})
    require(context.get("validated_against_receipt_torso_rest") is True
            and context.get("source_body_id") == 8 and context.get("core_body_index") == 20,
            "validated source-body rest context differs")

    common = provenance.get("cardiac_geometry_binding", {}).get("common_field", {})
    require(common.get("anatomy_payload_sha256") == sha_file(payload_path),
            "common map is not bound to candidate anatomy")
    require(int(common.get("map", {}).get("record_count", -1)) == 621369
            and int(common.get("map", {}).get("record_stride_bytes", -1)) == 112
            and preparation.get("output_map_sha256") == common["map"].get("sha256"),
            "compiled map range count/hash differs from preparation report")
    common_map_path = resolve_sidecar(candidate_receipt_path, base_receipt_path, common["map"])
    base_common = base_receipt.get("provenance", {}).get("cardiac_geometry_binding", {}).get("common_field", {})
    base_map_path = resolve_sidecar(base_receipt_path, base_receipt_path, base_common["map"])
    base_map = base_map_path.read_bytes()
    child_map = common_map_path.read_bytes()
    require(len(base_map) == 620718 * 112
            and child_map[:len(base_map)] == base_map
            and len(child_map) == 621369 * 112,
            "candidate changed the pinned N27 common-map prefix")
    for owner in ("polynomials", "domain_boxes"):
        child = resolve_sidecar(candidate_receipt_path, base_receipt_path, common[owner])
        base = resolve_sidecar(base_receipt_path, base_receipt_path, base_common[owner])
        require(child.read_bytes() == base.read_bytes(),
                "pinned common-field volume/domain bytes changed: " + owner)

    av, parse_payload = load_frozen_verifier()
    candidate_raw = payload_path.read_bytes()
    _, _, rows = parse_payload(candidate_raw)
    _, _, base_rows = parse_payload(base_payload_path.read_bytes())
    require(set(BRIDGES).issubset(rows) and 10 in rows and 10 in base_rows,
            "candidate lacks SVC10 or both bridge rows")
    require(rows[10]["record"] == base_rows[10]["record"]
            and rows[10]["vertices6"].tobytes() == base_rows[10]["vertices6"].tobytes()
            and rows[10]["faces"].tobytes() == base_rows[10]["faces"].tobytes(),
            "SVC10 source row changed from pinned N27")
    for sid, member in BRIDGES.items():
        identity = provenance.get("source_id_map", {}).get(str(sid), {})
        row = rows[sid]
        report = preparation.get("bridges", {}).get(member, {})
        xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
        faces = np.asarray(row["faces"], dtype="<i4")
        require(row["body_index"] == 20 and identity.get("source_member") == member
                and int(identity.get("body_index", -1)) == 20
                and int(identity.get("vascular_compartment_id", -1)) == 4,
                "bridge ID/body/compartment lineage changed: " + member)
        require(hashlib.sha256(xyz.tobytes()).hexdigest() == report.get("source_local_xyz_sha256")
                and hashlib.sha256(faces.tobytes()).hexdigest() == report.get("source_faces_sha256")
                and len(report.get("vertex_ancestry", [])) == len(xyz),
                "map ancestry does not match current bridge coordinates: " + member)
        map_range = next((item for item in common["passive_attachment_ranges"]
                          if int(item.get("stable_id", -1)) == sid), None)
        require(map_range is not None and int(map_range.get("body_index", -1)) == 20
                and int(map_range.get("semantic", -1)) == 51011
                and int(map_range.get("vertex_count", -1)) == len(xyz),
                "common-map range differs from bridge source row: " + member)

    return {
        "receipt_path": candidate_receipt_path,
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "payload_path": payload_path,
        "payload_sha256": sha_file(payload_path),
        "receipt": receipt,
        "rows": rows,
        "preparation_path": preparation_path,
        "preparation_sha256": hashlib.sha256(preparation_raw).hexdigest(),
        "preparation": preparation,
        "common_map_path": common_map_path,
        "common_map_sha256": sha_file(common_map_path),
        "capture_manifest_path": capture_manifest_path,
        "capture_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "base_receipt_path": base_receipt_path,
        "base_receipt_sha256": N27_RECEIPT_SHA256,
        "base_payload_path": base_payload_path,
        "base_payload_sha256": N27_PAYLOAD_SHA256,
        "source_rest_paths": source_rest_paths,
        "captures": captures,
        "av": av,
    }


def run(candidate_receipt_path: Path, preparation_path: Path,
        capture_manifest_path: Path, output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), "output directory already exists")
    inputs = validated_inputs(candidate_receipt_path, preparation_path, capture_manifest_path)
    predictor, predictor_path = load_predictor()
    rows = inputs["rows"]
    preparation = inputs["preparation"]
    bridge_rows = {sid: rows[sid] for sid in BRIDGES}
    arrays = {}
    pose_reports = []
    verifier_files = (P6 / "numilab_human/resting_anatomy_verify.py",
                      P6 / "numilab_human/cardiac_cavity_intersections.py")
    pins = {
        str(inputs["receipt_path"]): inputs["receipt_sha256"],
        str(inputs["payload_path"]): inputs["payload_sha256"],
        str(inputs["preparation_path"]): inputs["preparation_sha256"],
        str(inputs["common_map_path"]): inputs["common_map_sha256"],
        str(inputs["capture_manifest_path"]): inputs["capture_manifest_sha256"],
        str(inputs["base_receipt_path"]): inputs["base_receipt_sha256"],
        str(inputs["base_payload_path"]): inputs["base_payload_sha256"],
        str(inputs["source_rest_paths"]["verified_registration_path"]): REGISTRATION_SHA256,
        str(inputs["source_rest_paths"]["verified_rigid_payload_path"]): RIGID_SHA256,
        str(predictor_path): sha_file(predictor_path),
        str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
    }
    pins.update({str(path): sha_file(path) for path in verifier_files})
    output_dir.mkdir(parents=True)
    progress_path = output_dir / "progress.json"
    progress = {
        "schema": "numi.human.brachiocephalic-captured-svc-pose-prediction.v1",
        "status": "RUNNING_OFFLINE_PREDICTION_ONLY",
        "inputs": pins,
        "poses": [],
        "anatomical_admission": False,
        "all_pose_admission": False,
        "native_confirmation": "required",
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    for capture in inputs["captures"]:
        step = int(capture["step"])
        pack = Path(capture["pack_path"])
        receipt_path = Path(capture["receipt_path"])
        require(pack.is_file() and sha_file(pack) == capture["recorded_pack_sha256"],
                f"step {step} pack hash differs from 1398 manifest")
        require(receipt_path.is_file() and sha_file(receipt_path) == capture["receipt_sha256"],
                f"step {step} receipt hash differs from 1398 manifest")
        capture_receipt = json.loads(receipt_path.read_text())
        surfaces, identity = inputs["av"].load_capture(pack, receipt_path)
        require(identity["accepted_step"] == step
                and abs(float(identity["accepted_time_s"]) - float(capture["accepted_time_s"])) <= 1e-12,
                f"step/time differs at capture {step}")
        svc_capture = surfaces.get((51011, 10))
        require(svc_capture is not None, f"capture {step} lacks SVC10")
        svc = rows[10]
        require(np.array_equal(np.asarray(svc_capture[1], dtype="<u4"),
                               np.asarray(svc["faces"], dtype="<u4")),
                f"captured SVC10 face order changed at step {step}")
        bodyposes = {int(row["body_index"]): row for row in capture_receipt["accepted_body_poses"]}
        require(20 in bodyposes, f"capture {step} lacks accepted body20 pose")
        q, t = bodyposes[20]["quaternion_xyzw"], bodyposes[20]["position_m"]
        svc_rigid = transform_f32(q, t, svc["vertices6"][:, :3])
        svc_native = np.asarray(svc_capture[0], dtype="<f4")
        pose = {
            "step": step,
            "accepted_time_s": float(capture["accepted_time_s"]),
            "pack_sha256": identity["pack_sha256"],
            "receipt_sha256": identity["capture_receipt_sha256"],
            "captured_surface_count": len(surfaces),
            "body20_position_m": [float(x) for x in t],
            "body20_quaternion_xyzw": [float(x) for x in q],
            "bridges": {},
        }
        for sid, member in BRIDGES.items():
            source = bridge_rows[sid]
            bridge_rigid = transform_f32(q, t, source["vertices6"][:, :3])
            ancestry = preparation["bridges"][member]["vertex_ancestry"]
            prediction = predictor.predict_brachiocephalic_world_positions(
                bridge_rigid, svc_native, svc_rigid, ancestry)
            positions = prediction["positions_world_m"]
            arrays[f"step_{step}_stable_{sid}_xyz_f32"] = positions
            pose["bridges"][member] = {
                "stable_id": sid,
                "vertex_count": int(len(positions)),
                "predicted_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                "projected_displacement_sha256": hashlib.sha256(
                    prediction["projected_bridge_displacement_world_m"].tobytes()).hexdigest(),
                "interpolation_roundoff_bound_max_m":
                    prediction["float32_interpolation_roundoff_bound_max_m"],
                "interpolation_roundoff_observed_max_m":
                    prediction["float32_interpolation_roundoff_observed_max_m"],
                "native_field_error_bound": prediction["native_field_error_bound"],
                "status": prediction["status"],
            }
        pose_reports.append(pose)
        progress["poses"] = pose_reports
        progress["last_completed_step"] = step
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")

    npz_path = output_dir / "predicted-bridge-positions.npz"
    with npz_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        **progress,
        "status": "OFFLINE_PREDICTIONS_READY_NOT_GEOMETRICALLY_ADMITTED",
        "output_npz_path": str(npz_path),
        "output_npz_sha256": sha_file(npz_path),
        "predicted_arrays": sorted(arrays),
        "limits": [
            "Bound covers Float32 captured-displacement barycentric interpolation only.",
            "It does not bound native common-map coefficient/polynomial/kernel evaluation error.",
            "Packed targets were read-only; no all-target, exact-collar, lumen-continuity, calibre, or native gate was run.",
            "Native 16-pose confirmation is still mandatory; prediction does not admit anatomy.",
        ],
    }
    (output_dir / "prediction-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    progress_path.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--common-field-preparation", required=True, type=Path)
    parser.add_argument("--capture-manifest", type=Path, default=DEFAULT_CAPTURE_MANIFEST)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.candidate_receipt, args.common_field_preparation,
                 args.capture_manifest, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "output_npz_path": report["output_npz_path"],
        "output_npz_sha256": report["output_npz_sha256"],
        "pose_steps": [pose["step"] for pose in report["poses"]],
        "native_confirmation": report["native_confirmation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
