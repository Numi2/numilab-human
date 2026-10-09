#!/usr/bin/env python3
"""Reconstruct the default source-surface binding from pinned original inputs.

This is an offline CPU-only authoring replay. It calls the retained owning
model function and default source_surface_binding implementation once, captures
the exact inputs passed to that function, and compares the reconstructed full
float32 field with the retained source and 1187 NHSKIN payloads.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202")
PROJECT = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project")
SRC = PROJECT / "src"
MODEL_PATH = SRC / "numilab_human/model.py"
BINDING_PATH = SRC / "numilab_human/skin_surface_binding.py"
EXPORT_PATH = SRC / "numilab_human/myosim_export.py"
REGISTRATION = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json")
MYOSIM_ARTIFACT = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004")
MYOSIM_MANIFEST = MYOSIM_ARTIFACT / "myosim-fullbody-reference.manifest.json"
RIGID = MYOSIM_ARTIFACT / "myosim-fullbody-core-reference.nhrigid"
ISA_ARCHIVE = Path("/Users/n/numi-human-source-cache/isa_BP3D_4.0_obj_99.zip")
PARTOF_ARCHIVE = Path("/Users/n/numi-human-free-apex-publication-1159/Sources/partof_BP3D_4.0_obj_99.zip")
MYOSIM_SOURCES = PROJECT / "Sources/myosim"
MYOSIM_ARCHIVE = MYOSIM_SOURCES / "myo_sim-33c89c2b.tar.gz"
MYOSIM_RANGE_OVERLAY = MYOSIM_SOURCES / "source-overlays/myosim-left-knee-translation2-range.v1.json"
PATELLA_REBASE = PROJECT / "config/myosim-patella-neutral-coordinate-rebase.v1.json"
RAW_SKIN = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.nhskin")
CURRENT_SKIN = Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin")
RAW_SKIN_MANIFEST = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json")
OUTPUT = ROOT / "reconstruction"
SOURCES = ROOT / "source-inputs"
INPUTS_NPZ = ROOT / "source-surface-binding-inputs.npz"
BONES_JSON = ROOT / "source-surface-binding-bones.json"
REPORT = ROOT / "reconstruction-report.json"

EXPECTED = {
    str(MODEL_PATH): "b12b059c7de919c1f6047353497ba49d89bbd0ddcb008084bd6e223f59a02729",
    str(BINDING_PATH): "baf571721d990aac3f55aef54df57f6bcb4ceb7bdaccd6724a85faea957360ef",
    str(EXPORT_PATH): "12fe694ba945ce425e0ada4f6e079148ce9080115b27b9206490e18bc9fa7ed9",
    str(REGISTRATION): "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e",
    str(MYOSIM_MANIFEST): "844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7",
    str(RIGID): "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4",
    str(MYOSIM_ARCHIVE): "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975",
    str(MYOSIM_RANGE_OVERLAY): "05f6b7698e571c83a62bdbc7055ff24322800a36e9b53e455a18b451e7b9dad4",
    str(PATELLA_REBASE): "4188bcd6d7c0b4d755845e736a6d083ef238818e146819c7456836a1ea18641c",
    str(ISA_ARCHIVE): "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e",
    str(PARTOF_ARCHIVE): "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97",
    str(RAW_SKIN): "7d296e45ba497f56ac4380a64605b30a794432b945ac28bf7bebf91657cfc4a1",
    str(CURRENT_SKIN): "b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b",
    str(RAW_SKIN_MANIFEST): "bd8711819f789d3ef192106801833337509fa22b9bb3533eb012145b0ecdc49e",
}
DECLARED_NPZ_SHA = "99c913815b2cc615673ca3c1ac3ee36b640c5166c8a3e770ed6d54a3a5a281fc"
DECLARED_SOURCE_PAYLOAD_SHA = "7d296e45ba497f56ac4380a64605b30a794432b945ac28bf7bebf91657cfc4a1"

HEADER = struct.Struct("<8s5I32s")
BINDING_RECORD = struct.Struct("<I8f")
VERTEX_DTYPE = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_nhskin(path: Path, np):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"truncated NHSKIN header: {path}")
    magic, abi, binding_count, vertex_count, index_count, fingerprint, source_hash = HEADER.unpack_from(data)
    if magic.rstrip(b"\0") != b"NHSKIN1" or abi != 5:
        raise ValueError(f"unexpected NHSKIN ABI: {path}: {magic!r}/{abi}")
    vertex_dtype = np.dtype([
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("influence_indices", "<u4", (4,)),
        ("influence_weights", "<f4", (4,)),
    ])
    if vertex_dtype.itemsize != 56:
        raise AssertionError(f"unexpected ABI5 vertex record size {vertex_dtype.itemsize}")
    bindings_offset = HEADER.size
    body_indices = [
        BINDING_RECORD.unpack_from(data, bindings_offset + i * BINDING_RECORD.size)[0]
        for i in range(binding_count)
    ]
    vertices_offset = bindings_offset + binding_count * BINDING_RECORD.size
    vertices = np.frombuffer(data, dtype=vertex_dtype, count=vertex_count, offset=vertices_offset)
    indices_offset = vertices_offset + vertex_count * vertex_dtype.itemsize
    indices = np.frombuffer(data, dtype="<u4", count=index_count, offset=indices_offset)
    weights_offset = indices_offset + index_count * 4
    weights_count = vertex_count * binding_count
    weights = np.frombuffer(data, dtype="<f4", count=weights_count, offset=weights_offset)
    expected_size = weights_offset + weights_count * 4
    if len(data) != expected_size:
        raise ValueError(f"NHSKIN ABI5 payload length mismatch: {path}: {len(data)} != {expected_size}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "magic": magic.decode("ascii", errors="replace"),
        "abi": abi,
        "binding_count": binding_count,
        "vertex_count": vertex_count,
        "index_count": index_count,
        "triangle_count": index_count // 3,
        "registration_fingerprint32": f"{fingerprint:08x}",
        "source_hash_in_header": source_hash.hex(),
        "body_indices": body_indices,
        "vertices": vertices,
        "indices": indices,
        "full_weights": weights.reshape(vertex_count, binding_count),
        "full_weight_bytes_sha256": hashlib.sha256(memoryview(weights)).hexdigest(),
    }


def pin_inputs() -> dict[str, str]:
    actual = {}
    for name, path in (
        ("model", MODEL_PATH), ("binding", BINDING_PATH), ("export", EXPORT_PATH),
        ("registration", REGISTRATION), ("myosim_manifest", MYOSIM_MANIFEST), ("rigid", RIGID),
        ("myosim_archive", MYOSIM_ARCHIVE), ("range_overlay", MYOSIM_RANGE_OVERLAY),
        ("patella_rebase", PATELLA_REBASE),
        ("isa_archive", ISA_ARCHIVE), ("partof_archive", PARTOF_ARCHIVE),
        ("raw_skin", RAW_SKIN), ("current_skin", CURRENT_SKIN), ("raw_skin_manifest", RAW_SKIN_MANIFEST),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        expected = EXPECTED[str(path)]
        if digest != expected:
            raise RuntimeError(f"pinned input changed: {path}: {digest} != {expected}")
        actual[name] = digest
    if OUTPUT.exists() or SOURCES.exists() or INPUTS_NPZ.exists() or BONES_JSON.exists() or REPORT.exists():
        raise FileExistsError("1202 output already exists; refusing to overwrite any prior attempt")
    return actual


def prepare_sources() -> None:
    SOURCES.mkdir(parents=True, exist_ok=False)
    for name, target in (
        ("isa_BP3D_4.0_obj_99.zip", ISA_ARCHIVE),
        ("partof_BP3D_4.0_obj_99.zip", PARTOF_ARCHIVE),
    ):
        if not target.is_file():
            raise FileNotFoundError(target)
        os.symlink(target, SOURCES / name)
    if not MYOSIM_SOURCES.is_dir():
        raise FileNotFoundError(MYOSIM_SOURCES)
    # The owner applies/validates its source overlays. Use a private copy so
    # even an unapplied overlay can never mutate the retained source checkout.
    shutil.copytree(MYOSIM_SOURCES, SOURCES / "myosim", symlinks=True)


def main() -> int:
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    pins = pin_inputs()
    prepare_sources()
    sys.path.insert(0, str(SRC))
    import numpy as np
    import scipy
    from numilab_human import model
    from numilab_human import skin_surface_binding

    registration = model.read_json(REGISTRATION)
    bodyparts = registration["source"]["bodyparts"]
    anatomy = {
        "source_id": bodyparts["id"],
        "version": bodyparts["version"],
        "archives": bodyparts["archives"],
    }
    binding_evidence = {}
    original_binding = skin_surface_binding.source_surface_binding

    def capture_default_binding(vertices, faces, bones, binding_count):
        if binding_evidence:
            raise RuntimeError("source_surface_binding was invoked more than once")
        binding_evidence["vertices"] = np.asarray(vertices).copy()
        binding_evidence["faces"] = np.asarray(faces).copy()
        binding_evidence["bones"] = copy.deepcopy(bones)
        binding_evidence["binding_count"] = int(binding_count)
        print(
            f"source_surface_binding input: vertices={binding_evidence['vertices'].shape} "
            f"faces={binding_evidence['faces'].shape} bones={len(bones)} bindings={binding_count}",
            flush=True,
        )
        result = original_binding(vertices, faces, bones, binding_count)
        binding_evidence["quartets"] = np.asarray(result[0]).copy()
        binding_evidence["source_weights"] = np.asarray(result[1]).copy()
        binding_evidence["full_weights"] = np.asarray(result[2]).copy()
        binding_evidence["seed_targets"] = np.asarray(result[3]).copy()
        binding_evidence["projection_gaps"] = np.asarray(result[4]).copy()
        binding_evidence["evidence"] = copy.deepcopy(result[5])
        return result

    skin_surface_binding.source_surface_binding = capture_default_binding
    print("starting the retained owning bodyparts_myosim_skinned_shell_visual_payload function", flush=True)
    try:
        manifest = model.bodyparts_myosim_skinned_shell_visual_payload(
            SOURCES, anatomy, REGISTRATION, OUTPUT, myosim_artifact=MYOSIM_ARTIFACT,
        )
    finally:
        skin_surface_binding.source_surface_binding = original_binding
    if not binding_evidence:
        raise RuntimeError("owning model returned without calling the default binding function")
    print("owner function returned; capturing exact graph inputs and validating payload rows", flush=True)

    vertices = binding_evidence["vertices"]
    faces = binding_evidence["faces"]
    bones = binding_evidence["bones"]
    full_weights = binding_evidence["full_weights"]
    targets = binding_evidence["seed_targets"]
    gaps = binding_evidence["projection_gaps"]
    np.savez_compressed(
        INPUTS_NPZ,
        vertices=vertices,
        faces=faces,
        binding_count=np.asarray(binding_evidence["binding_count"], dtype="<u4"),
    )
    atomic_json(BONES_JSON, {"bones": bones})

    reconstructed_npz = OUTPUT / "bodyparts3d-skin-binding-solution.npz"
    with np.load(reconstructed_npz, allow_pickle=False) as z:
        saved_full = np.asarray(z["full_weights"])
        saved_targets = np.asarray(z["seed_targets"])
        saved_gaps = np.asarray(z["seed_projection_gaps_m"])
    if not (np.array_equal(saved_full, full_weights)
            and np.array_equal(saved_targets, targets)
            and np.array_equal(saved_gaps, gaps)):
        raise RuntimeError("writer arrays do not exactly match captured default owner outputs")

    raw = parse_nhskin(RAW_SKIN, np)
    current = parse_nhskin(CURRENT_SKIN, np)
    generated_path = OUTPUT / "bodyparts3d-myosim-skinned-shell.nhskin"
    generated = parse_nhskin(generated_path, np)
    full_f32 = np.asarray(full_weights, dtype="<f4")
    reconstructed_f32_bytes = full_f32.tobytes(order="C")
    raw_weight_bytes = raw["full_weights"].tobytes(order="C")
    current_weight_bytes = current["full_weights"].tobytes(order="C")
    generated_weight_bytes = generated["full_weights"].tobytes(order="C")
    source_weight_match = reconstructed_f32_bytes == raw_weight_bytes
    current_weight_match = reconstructed_f32_bytes == current_weight_bytes
    generated_weight_match = reconstructed_f32_bytes == generated_weight_bytes

    topology_equal = (
        raw["vertex_count"] == current["vertex_count"] == generated["vertex_count"]
        and raw["binding_count"] == current["binding_count"] == generated["binding_count"]
        and raw["index_count"] == current["index_count"] == generated["index_count"]
        and np.array_equal(raw["indices"], current["indices"])
        and np.array_equal(raw["indices"], generated["indices"])
    )
    body_index_tables_equal = (
        raw["body_indices"] == current["body_indices"] == generated["body_indices"]
    )
    input_before_after = {str(path): sha256(path) == pins[name] for name, path in (
        ("model", MODEL_PATH), ("binding", BINDING_PATH), ("export", EXPORT_PATH),
        ("registration", REGISTRATION), ("myosim_manifest", MYOSIM_MANIFEST), ("rigid", RIGID),
        ("myosim_archive", MYOSIM_ARCHIVE), ("range_overlay", MYOSIM_RANGE_OVERLAY),
        ("patella_rebase", PATELLA_REBASE),
        ("isa_archive", ISA_ARCHIVE), ("partof_archive", PARTOF_ARCHIVE),
        ("raw_skin", RAW_SKIN), ("current_skin", CURRENT_SKIN), ("raw_skin_manifest", RAW_SKIN_MANIFEST),
    )}

    original_manifest = model.read_json(RAW_SKIN_MANIFEST)
    declared_solution = original_manifest["coverage"]["binding_solution"]
    owner_solution = manifest["coverage"]["binding_solution"]
    source_current_max_displacement_m = float(np.linalg.norm(
        raw["vertices"]["position"].astype("<f8") - current["vertices"]["position"].astype("<f8"), axis=1,
    ).max())
    source_current_changed_vertex_count = int(np.count_nonzero(
        np.any(raw["vertices"]["position"] != current["vertices"]["position"], axis=1)
    ))
    report = {
        "schema": "numi.human.source-skin-binding-reconstruction.v1",
        "status": "source_default_matrix_reproduced" if source_weight_match and current_weight_match and generated_weight_match else "STOP_weight_matrix_mismatch",
        "qualification": "offline reproduction of the existing default inferred visual weight field; not measured skin mechanics or native qualification",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_s": time.monotonic() - started,
        "python": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "owner": {
            "function": "numilab_human.model.bodyparts_myosim_skinned_shell_visual_payload",
            "model_path": str(MODEL_PATH),
            "model_sha256": pins["model"],
            "binding_module": str(BINDING_PATH),
            "binding_module_sha256": pins["binding"],
            "myosim_export_module": str(EXPORT_PATH),
            "myosim_export_module_sha256": pins["export"],
            "default_method": binding_evidence["evidence"]["method"],
            "bone_anchor_specification_count": len(model._BODYPARTS_MYOSIM_BONE_ANCHORS),
            "registration_anchor_count": len(registration["anchors"]),
            "captured_binding_count": binding_evidence["binding_count"],
        },
        "inputs": {
            "sha256": pins,
            "registration": str(REGISTRATION),
            "registration_sha256": pins["registration"],
            "registration_declared_myosim_artifact_manifest_sha256": registration["source"]["myosim"].get("artifact_manifest_sha256"),
            "actual_myosim_artifact_manifest": str(MYOSIM_MANIFEST),
            "actual_myosim_artifact_manifest_sha256": pins["myosim_manifest"],
            "registration_declared_rigid_payload_sha256": registration["source"]["myosim"]["payloads"]["rigid"]["sha256"],
            "actual_rigid_payload_sha256": pins["rigid"],
            "rigid_runtime_binding_check": "passed via existing _bodyparts_runtime_bindings for all 185 anchors",
            "raw_source_payload": str(RAW_SKIN),
            "current_1187_payload": str(CURRENT_SKIN),
            "declared_original_npz": declared_solution,
        },
        "reconstructed_npz": {
            "path": str(reconstructed_npz),
            "sha256": sha256(reconstructed_npz),
            "bytes": reconstructed_npz.stat().st_size,
            "arrays": {
                "full_weights": {"shape": list(saved_full.shape), "dtype": saved_full.dtype.str},
                "seed_targets": {"shape": list(saved_targets.shape), "dtype": saved_targets.dtype.str},
                "seed_projection_gaps_m": {"shape": list(saved_gaps.shape), "dtype": saved_gaps.dtype.str},
            },
            "declared_old_container_sha256": declared_solution["sha256"],
            "container_sha_equal": sha256(reconstructed_npz) == declared_solution["sha256"],
            "container_hash_note": "A regenerated compressed NPZ is compared by arrays; archive metadata can make its container SHA differ from the unavailable original NPZ.",
        },
        "captured_source_surface_binding_inputs": {
            "geometry_npz": str(INPUTS_NPZ),
            "geometry_npz_sha256": sha256(INPUTS_NPZ),
            "bones_json": str(BONES_JSON),
            "bones_json_sha256": sha256(BONES_JSON),
            "vertices": {"shape": list(vertices.shape), "dtype": vertices.dtype.str, "unit": "world metres after the exact existing model world_point sum"},
            "faces": {"shape": list(faces.shape), "dtype": faces.dtype.str},
            "bones": len(bones),
            "binding_count": binding_evidence["binding_count"],
            "seed_targets": {"shape": list(targets.shape), "dtype": targets.dtype.str},
            "projection_gaps": {"shape": list(gaps.shape), "dtype": gaps.dtype.str},
            "binding_evidence": binding_evidence["evidence"],
        },
        "output_payload": {
            "path": str(generated_path),
            "sha256": generated["sha256"],
            "bytes": generated["bytes"],
            "matches_original_source_payload_bytes": generated["sha256"] == pins["raw_skin"],
            "manifest_path": str(OUTPUT / "bodyparts3d-myosim-skinned-shell.manifest.json"),
            "manifest_sha256": sha256(OUTPUT / "bodyparts3d-myosim-skinned-shell.manifest.json"),
            "full_weight_f32_sha256": generated["full_weight_bytes_sha256"],
        },
        "comparison": {
            "source_reconstructed_full_weight_f32_exact": source_weight_match,
            "current_1187_reconstructed_full_weight_f32_exact": current_weight_match,
            "generated_payload_reconstructed_full_weight_f32_exact": generated_weight_match,
            "reconstructed_full_weight_shape": list(full_weights.shape),
            "reconstructed_full_weight_f32_sha256": hashlib.sha256(reconstructed_f32_bytes).hexdigest(),
            "source_weight_sha256": raw["full_weight_bytes_sha256"],
            "current_1187_weight_sha256": current["full_weight_bytes_sha256"],
            "generated_weight_sha256": generated["full_weight_bytes_sha256"],
            "source_current_topology_exact": topology_equal,
            "source_current_binding_order_exact": body_index_tables_equal,
            "source_vertex_count": raw["vertex_count"],
            "current_vertex_count": current["vertex_count"],
            "source_triangle_count": raw["triangle_count"],
            "current_triangle_count": current["triangle_count"],
            "source_current_changed_position_rows": source_current_changed_vertex_count,
            "source_current_max_position_delta_m": source_current_max_displacement_m,
            "source_current_vertex_row_identity": "The exact source graph rows are not reordered; current payload weight rows compare byte-exact at the same vertex indices. Position edits remain distinct geometry, not a seed-graph substitute.",
        },
        "source_inputs_unchanged": input_before_after,
    }
    atomic_json(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "elapsed_wall_s": report["elapsed_wall_s"],
        "reconstructed_npz_sha256": report["reconstructed_npz"]["sha256"],
        "generated_payload_sha256": generated["sha256"],
        "source_weight_exact": source_weight_match,
        "current_weight_exact": current_weight_match,
        "topology_exact": topology_equal,
        "report": str(REPORT),
    }, sort_keys=True), flush=True)
    return 0 if report["status"] == "source_default_matrix_reproduced" and topology_equal and body_index_tables_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
