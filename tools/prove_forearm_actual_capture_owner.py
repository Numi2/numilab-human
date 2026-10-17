#!/usr/bin/env python3
"""Authenticate accepted run002 actual 132/134 capture points and shared-GID XYZ."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np

REPO = Path("/Users/numi/numi-human-forearm-parent-bridge-20261017")
RETAINED = Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
D1284 = RETAINED / "forearm-native007-actual972-full21-witness-batch-1284"
ACCEPTED = RETAINED / "pons323-348-current972-existing-owner-rebind-001/native-run-002"
SOURCE_MAP = RETAINED / "forearm-native007-current972-joint-source-containing-006"
INDEX = RETAINED / "forearm-native007-actual972-witness-star-retriangulation-1274"
PINNED_CODE_COMMIT = "c02ccdf01475e06852d3944c3d640362b218d472"
MAX_SECONDS = 900


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def fail(message: str) -> None:
    raise RuntimeError(message)


def alarm_handler(signum, frame):
    raise TimeoutError(f"actual capture proof exceeded {MAX_SECONDS}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        fail(f"refusing to overwrite existing proof directory: {out}")
    out.mkdir(parents=True)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAX_SECONDS)
    progress = out / "progress.json"
    launch = {
        "schema": "numi.human.forearm-actual-capture-proof-launch.v1",
        "output": str(out),
        "producer_arrays": str(D1284 / "prepared-owner-arrays.npz"),
        "producer_metadata": str(D1284 / "prepared-owner-arrays.json"),
        "accepted_pack": str(ACCEPTED / "accepted-geometry/step-0.mrvpack"),
        "accepted_receipt": str(ACCEPTED / "accepted-geometry/step-0.receipt.json"),
        "pinned_code_commit": PINNED_CODE_COMMIT,
        "max_seconds": MAX_SECONDS,
        "admission": False,
    }
    atomic_json(out / "launch.json", launch)
    atomic_json(progress, {"stage": "source_snapshot", "elapsed_s": 0.0})

    # Run the existing capture/source owner code at the exact 1284 producer revision.
    archive = out / "pinned-source-c02.tar"
    proc = subprocess.run(
        ["git", "archive", "--format=tar", PINNED_CODE_COMMIT, "src/numilab_human"],
        cwd=REPO, check=True, stdout=archive.open("wb"))
    del proc
    snapshot_root = out / "pinned-code"
    snapshot_root.mkdir()
    with tarfile.open(archive, "r:") as tf:
        tf.extractall(snapshot_root, filter="data")
    snapshot_src = snapshot_root / "src"
    archived_modules = {}
    original_metadata_path = D1284 / "prepared-owner-arrays.json"
    metadata = json.loads(original_metadata_path.read_text())
    original_metadata_sha = sha(original_metadata_path)
    original_pins = metadata.get("input_sha256")
    if not isinstance(original_pins, dict) or not original_pins:
        fail("producer metadata has no exact input pins")
    source_aliases = {}
    remapped_pins = {}
    repo_root = REPO.resolve(strict=True)
    for pinned_path, expected in original_pins.items():
        path = Path(pinned_path)
        try:
            relative = path.resolve(strict=False).relative_to(repo_root)
        except ValueError:
            remapped_pins[pinned_path] = expected
            continue
        if relative.parts[:2] == ("src", "numilab_human"):
            archived = snapshot_root / relative
            actual = sha(archived)
            if actual != expected:
                fail(f"pinned source blob does not match producer hash: {relative}")
            source_aliases[pinned_path] = str(archived)
            remapped_pins[str(archived)] = expected
            archived_modules[str(relative)] = actual
        else:
            remapped_pins[pinned_path] = expected
    if len(source_aliases) < 5:
        fail(f"expected all five pinned owner modules, found {len(source_aliases)}")
    metadata["input_sha256"] = remapped_pins
    metadata["source_code_pin_aliases"] = source_aliases
    metadata["source_code_pin_snapshot_commit"] = PINNED_CODE_COMMIT
    metadata["source_code_snapshot_tar_sha256"] = sha(archive)
    metadata["original_producer_metadata_sha256"] = original_metadata_sha
    metadata_copy = out / "prepared-owner-arrays-pinned-paths.json"
    metadata_copy.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    atomic_json(progress, {
        "stage": "authenticated_source_snapshot_ready",
        "elapsed_s": time.monotonic() - start,
        "source_module_count": len(archived_modules),
        "source_tar_sha256": sha(archive),
        "source_modules": archived_modules,
    })

    sys.path.insert(0, str(snapshot_src))
    import numilab_human
    numilab_human.__path__.append(str(REPO / "src/numilab_human"))
    from numilab_human.forearm_actual_capture_owner import build_actual_capture_owner_proof

    proof_builder = Path(numilab_human.forearm_actual_capture_owner.__file__).resolve()
    verify_module = __import__(
        "numilab_human.resting_anatomy_verify", fromlist=["load_capture"])
    verify_path = Path(verify_module.__file__).resolve()
    atomic_json(progress, {
        "stage": "proof_owner_loaded",
        "elapsed_s": time.monotonic() - start,
        "proof_builder_path": str(proof_builder),
        "proof_builder_sha256": sha(proof_builder),
        "capture_loader_path": str(verify_path),
        "capture_loader_sha256": sha(verify_path),
        "capture_loader_revision": PINNED_CODE_COMMIT,
    })

    proof = build_actual_capture_owner_proof(
        prepared_owner_arrays_path=D1284 / "prepared-owner-arrays.npz",
        prepared_owner_metadata_path=metadata_copy,
        accepted_pack_path=ACCEPTED / "accepted-geometry/step-0.mrvpack",
        accepted_step_receipt_path=ACCEPTED / "accepted-geometry/step-0.receipt.json",
        source_child_path=RETAINED / "deltoid75-78-existing-shared-parent-source-824/conformed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue",
        source_map_candidate_path=SOURCE_MAP / "candidate.npz",
        source_map_report_path=SOURCE_MAP / "report.json",
        index_candidate_path=INDEX / "candidate.npz",
        index_report_path=INDEX / "report.json",
        index_ancestry_path=INDEX / "candidate-face-ancestry.json",
        witness_batch_report_path=D1284 / "report.json",
        witness_batch_checkpoint_path=D1284 / "stop-checkpoint.json",
    )
    atomic_json(progress, {"stage": "capture_and_shared_gid_proof_complete",
                          "elapsed_s": time.monotonic() - start})

    rows = proof["rows"]
    np.savez(
        out / "actual-132-134-owner-arrays.npz",
        row132_positions_f32=rows["132"]["actual_capture_positions_f32"],
        row132_faces_u32=rows["132"]["actual_capture_faces_u32"],
        row132_gids_u64=rows["132"]["canonical_gids_u64"],
        row132_source_slots_i64=rows["132"]["capture_to_source_slot_i64"],
        row132_face_corner_source_slots_i64=rows["132"]["ordered_face_corner_source_slots_i64"],
        row132_face_corner_gids_u64=rows["132"]["ordered_face_corner_canonical_gids_u64"],
        row132_full21_weights_f64=rows["132"]["full21_weights_f64"],
        row134_positions_f32=rows["134"]["actual_capture_positions_f32"],
        row134_faces_u32=rows["134"]["actual_capture_faces_u32"],
        row134_gids_u64=rows["134"]["canonical_gids_u64"],
        row134_source_slots_i64=rows["134"]["capture_to_source_slot_i64"],
        row134_face_corner_source_slots_i64=rows["134"]["ordered_face_corner_source_slots_i64"],
        row134_face_corner_gids_u64=rows["134"]["ordered_face_corner_canonical_gids_u64"],
        row134_full21_weights_f64=rows["134"]["full21_weights_f64"],
    )
    def row_summary(value):
        return {
            "typed_identity": value["typed_identity"],
            "position_count": int(len(value["actual_capture_positions_f32"])),
            "face_count": int(len(value["actual_capture_faces_u32"])),
            "actual_positions_sha256": hashlib.sha256(
                value["actual_capture_positions_f32"].tobytes()).hexdigest(),
            "ordered_faces_sha256": hashlib.sha256(
                value["actual_capture_faces_u32"].tobytes()).hexdigest(),
            "canonical_gids_sha256": hashlib.sha256(
                value["canonical_gids_u64"].tobytes()).hexdigest(),
            "capture_to_source_slots_sha256": hashlib.sha256(
                value["capture_to_source_slot_i64"].tobytes()).hexdigest(),
            "full21_weights_sha256": hashlib.sha256(
                value["full21_weights_f64"].tobytes()).hexdigest(),
            "ordered_face_corner_source_slots_sha256":
                value["report"]["ordered_face_corner_source_slots_sha256"],
            "ordered_face_corner_canonical_gids_sha256":
                value["report"]["ordered_face_corner_canonical_gids_sha256"],
            "row_owner_report": value["report"],
        }

    summary = {
        "schema": proof["schema"],
        "proof_status": (
            "PASS_ACTUAL_SHARED_GID_POINT_IDENTITY"
            if proof["shared_actual_capture_gid_points"]["all_shared_gid_actual_positions_equal"]
            else "FAIL_SHARED_GID_ACTUAL_POSITION_MISMATCH"),
        "accepted_step": proof["accepted_step"],
        "accepted_pack_sha256": proof["accepted_pack_sha256"],
        "accepted_step_receipt_sha256": proof["accepted_step_receipt_sha256"],
        "source_child_sha256": proof["source_child_sha256"],
        "source_map_candidate_sha256": proof["source_map_candidate_sha256"],
        "index_candidate_sha256": proof["index_candidate_sha256"],
        "prepared_owner_arrays_sha256": proof["prepared_owner_arrays_sha256"],
        "prepared_owner_metadata_sha256": proof["prepared_owner_metadata_sha256"],
        "original_producer_metadata_sha256": original_metadata_sha,
        "derived_metadata_sha256": sha(metadata_copy),
        "index_ancestry_sha256": proof["index_ancestry_sha256"],
        "witness_batch_report_sha256": proof["witness_batch_report_sha256"],
        "witness_batch_checkpoint_sha256": proof["witness_batch_checkpoint_sha256"],
        "producer_input_pin_count_verified": proof["producer_input_pin_count_verified"],
        "pinned_code_commit": PINNED_CODE_COMMIT,
        "pinned_code_snapshot_tar_sha256": sha(archive),
        "pinned_code_module_sha256": archived_modules,
        "proof_builder_sha256": sha(proof_builder),
        "capture_loader_sha256": sha(verify_path),
        "actual_owner_arrays_sha256": sha(out / "actual-132-134-owner-arrays.npz"),
        "rows": {row_id: row_summary(value) for row_id, value in rows.items()},
        "shared_actual_capture_gid_points": proof["shared_actual_capture_gid_points"],
        "face_topology_provenance": proof["face_topology_provenance"],
        "not_admission": True,
        "geometry_intersection_waiver": False,
        "evidence_boundary": (
            "actual accepted002 XYZ bytes are loaded from the pack/receipt and compared "
            "across every shared canonical GID; ordered source-slot/capture/GID face-corner "
            "maps are authenticated separately; full21 and source ancestry do not waive "
            "off-feature exact intersections or constitute anatomy/native admission"
        ),
        "elapsed_s": time.monotonic() - start,
    }
    atomic_json(out / "report.json", summary)
    atomic_json(progress, {"stage": "complete", "elapsed_s": time.monotonic() - start,
                           "report_sha256": sha(out / "report.json"),
                           "proof_status": summary["proof_status"]})
    print(json.dumps({"output": str(out), "proof_status": summary["proof_status"],
                      "shared_gids": summary["shared_actual_capture_gid_points"][
                          "shared_canonical_gid_count"],
                      "exact_shared_points": summary["shared_actual_capture_gid_points"][
                          "exact_shared_actual_point_count"],
                      "position_mismatches": summary["shared_actual_capture_gid_points"][
                          "mismatched_shared_gid_position_count"],
                      "elapsed_s": summary["elapsed_s"]}, sort_keys=True))


if __name__ == "__main__":
    main()
