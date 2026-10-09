#!/usr/bin/env python3
"""Read-only MRVPACK audit for fixed-world contoured-bed mesh and capture identity."""
from __future__ import annotations
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/n/numi-human-performance-source-014/matter/tools")
import accepted_mrvpack_surface_audit as pack_audit

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199")
RUN = ROOT / "native-run"
GEOM = RUN / "accepted-geometry"
BASE = Path("/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/native-run")
BASE_GEOM = BASE / "accepted-geometry"
SCENE = Path("/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196")
MANIFEST = Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/resting-supine-scene-distributed-support.manifest.json")
GRID = SCENE / "heightfields-runtime-f32-5mm.npz"
AUDITOR = Path("/Users/n/numi-human-performance-source-014/matter/tools/accepted_mrvpack_surface_audit.py")
VISUAL_SOURCE = Path("/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/runtime-source/apps/NumiHumanRestingVisual.hpp")
BED_SOURCE = Path("/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/runtime-source/apps/NumiHumanRestingBedSurface.hpp")
TYPE_SOURCE = Path("/Users/n/numi-human-performance-source-014/include/metalrobo/visual_platform_types.h")
SUPPORT = Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/myosim-fullbody-distributed-rigid-digit-support.nhcnt")
DECLARATION = ROOT / "run-declaration.json"
INVOCATION = RUN / "invocation.json"
RUN_META = RUN / "run-metadata.json"
NATIVE_LOG = RUN / "native.log"
OUTPUT = Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/bed-mesh-capture-audit-1199/bed-mesh-capture-audit.json")
STEPS = [0, 10000]
BED_KEY = (51999, 1)
U32_MAX = 0xFFFFFFFF
HEADER = pack_audit.HEADER
DIRECTORY = pack_audit.DIRECTORY


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sections(mm):
    header = HEADER.unpack_from(mm, 0)
    if header[0] != b"MRVPACK2" or header[1] != 2:
        raise ValueError(f"unsupported MRVPACK2 header: {header[:3]}")
    return {row[0]: row for row in [DIRECTORY.unpack_from(mm, HEADER.size + i * DIRECTORY.size)
                                     for i in range(header[2])]}


def key_of_primitive(mm, row, index):
    off = row[2] + index * row[5]
    geom = struct.unpack_from("<4I", mm, off)
    ident = struct.unpack_from("<4I", mm, off + 16)
    return (ident[0], ident[3]), geom, ident, off


def table(mm, sec):
    prim_row = sec[4]
    result = {}
    for i in range(prim_row[4]):
        key, geom, ident, off = key_of_primitive(mm, prim_row, i)
        if key in result:
            raise ValueError(f"duplicate primitive key {key}")
        result[key] = {"index": i, "geom": geom, "ident": ident, "offset": off,
                       "raw": mm[off:off + prim_row[5]]}
    return result


def row_record(mm, sec, row_type, index):
    row = sec[row_type]
    off = row[2] + index * row[5]
    return mm[off:off + row[5]]


def instance_table(mm, sec):
    row = sec[5]
    result = {}
    for i in range(row[4]):
        raw = row_record(mm, sec, 5, i)
        ident = struct.unpack_from("<4I", raw, 48)
        key = (ident[0], ident[3])
        if key in result:
            raise ValueError(f"duplicate instance key {key}")
        result[key] = {"index": i, "identity": ident, "raw": raw}
    return result


def index_bytes(mm, sec, geom):
    row = sec[3]
    start, count = geom[0], geom[1]
    return mm[row[2] + start * 4:row[2] + (start + count) * 4]


def record_receipt(pack_path: Path, step: int):
    receipt_path = pack_path.with_name(f"step-{step}.receipt.json")
    with pack_path.open("rb") as stream:
        mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        sec = sections(mm)
        receipt = pack_audit.validate_accepted_receipt(pack_path, receipt_path, step, mm,
                                                        sec[2][2], {})
        mm.close()
    return receipt


def open_pack(path: Path):
    stream = path.open("rb")
    mm = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    sec = sections(mm)
    if sec[2][5] != 80 or sec[3][5] != 4 or sec[4][5] != 64 or sec[5][5] != 80:
        raise ValueError(f"unexpected section stride in {path}")
    return stream, mm, sec


def bytes_at(mm, row, start, size):
    return mm[row[2] + start:row[2] + start + size]


def main():
    input_paths = [DECLARATION, INVOCATION, RUN_META, NATIVE_LOG, MANIFEST, GRID, SUPPORT,
                   AUDITOR, VISUAL_SOURCE, BED_SOURCE, TYPE_SOURCE,
                   BASE_GEOM / "step-0.mrvpack", BASE_GEOM / "step-0.receipt.json"]
    input_paths += [p for step in STEPS for p in (GEOM / f"step-{step}.mrvpack",
                                                    GEOM / f"step-{step}.receipt.json")]
    input_hashes = {str(p): sha256(p) for p in input_paths}

    declaration = json.loads(DECLARATION.read_text())
    invocation = json.loads(INVOCATION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    support = manifest.get("outputs", {}).get("support_contact", {})
    if support.get("path") != str(SUPPORT) or support.get("sha256") != sha256(SUPPORT):
        raise ValueError("1199 manifest does not bind its distributed support payload")
    if invocation.get("asset_sha256", {}).get(str(SUPPORT)) != sha256(SUPPORT):
        raise ValueError("1199 invocation does not bind its distributed support payload")
    baseline_manifest = json.loads((SCENE / "resting-supine-scene-contoured-5mm.manifest.json").read_text())
    if manifest.get("bed", {}).get("heightfield") != baseline_manifest.get("bed", {}).get("heightfield"):
        raise ValueError("1199 heightfield declaration differs from the fixed 1196 bed")
    run_meta = json.loads(RUN_META.read_text())
    native_text = NATIVE_LOG.read_text(errors="replace")
    if not all(str(GEOM / f"step-{s}.mrvpack") in input_hashes for s in STEPS):
        raise AssertionError("internal capture-hash bookkeeping failed")
    if not any("resting_integrated_body=completed" in line and "simulated_s=20.000000949949026" in line
               for line in native_text.splitlines()):
        raise ValueError("native log does not contain the expected completed 20-second marker")

    baseline_pack = BASE_GEOM / "step-0.mrvpack"
    current_pack = GEOM / "step-0.mrvpack"
    old_receipt = record_receipt(baseline_pack, 0)
    new_receipt = record_receipt(current_pack, 0)
    old_raw_receipt = json.loads((BASE_GEOM / "step-0.receipt.json").read_text())
    new_raw_receipt = json.loads((GEOM / "step-0.receipt.json").read_text())
    for name in ("accepted_body_state_sha256", "accepted_respiration_state_sha256",
                 "accepted_registered_body_poses"):
        if old_raw_receipt[name] != new_raw_receipt[name]:
            raise ValueError(f"initial accepted-state mismatch for {name}")

    old_stream, old_mm, old_sec = open_pack(baseline_pack)
    new_stream, new_mm, new_sec = open_pack(current_pack)
    old_prims, new_prims = table(old_mm, old_sec), table(new_mm, new_sec)
    old_insts, new_insts = instance_table(old_mm, old_sec), instance_table(new_mm, new_sec)
    if set(old_prims) != set(new_prims) or set(old_insts) != set(new_insts):
        raise ValueError("primitive or instance identities changed at step 0")
    if BED_KEY not in new_prims or BED_KEY not in new_insts:
        raise ValueError("expected contoured bed primitive/instance identity is absent")
    nonbed = sorted(k for k in old_prims if k != BED_KEY)
    changed_prims, changed_insts, changed_index = [], [], []
    max_nonbed_vertex = -1
    for key in nonbed:
        a, b = old_prims[key], new_prims[key]
        if a["raw"] != b["raw"]:
            changed_prims.append(key)
        if index_bytes(old_mm, old_sec, a["geom"]) != index_bytes(new_mm, new_sec, b["geom"]):
            changed_index.append(key)
        if old_insts[key]["raw"] != new_insts[key]["raw"]:
            changed_insts.append(key)
        # Index streams are the same; retain the maximum referenced non-bed vertex index.
        start, count = b["geom"][0], b["geom"][1]
        vals = np.frombuffer(new_mm, dtype="<u4", count=count,
                             offset=new_sec[3][2] + start * 4)
        if vals.size:
            max_nonbed_vertex = max(max_nonbed_vertex, int(vals.max()))
    if changed_prims or changed_index or changed_insts:
        raise ValueError(f"non-bed geometry changed: prim={len(changed_prims)} index={len(changed_index)} instance={len(changed_insts)}")

    bed_prim = new_prims[BED_KEY]
    old_bed = old_prims[BED_KEY]
    new_bed_indices = np.frombuffer(index_bytes(new_mm, new_sec, bed_prim["geom"]), dtype="<u4")
    if new_bed_indices.size != 717600:
        raise ValueError(f"unexpected bed index count {new_bed_indices.size}")
    bed_first, bed_last = int(new_bed_indices.min()), int(new_bed_indices.max())
    if max_nonbed_vertex >= bed_first:
        raise ValueError("non-bed primitive references a vertex in the appended bed range")
    if old_sec[2][4] != bed_first + 120321 or new_sec[2][4] != bed_first + 120321:
        raise ValueError("baseline and candidate bed vertex ranges do not match the 261x461 grid")
    old_vertex_prefix = old_mm[old_sec[2][2]:old_sec[2][2] + bed_first * 80]
    new_vertex_prefix = new_mm[new_sec[2][2]:new_sec[2][2] + bed_first * 80]
    if old_vertex_prefix != new_vertex_prefix:
        raise ValueError("step-0 non-bed vertex prefix changed")

    # Rebuild the expected node positions from the retained runtime-F32 lattice.
    with np.load(GRID) as lattice:
        xs = lattice["x_nodes_f32"]
        ys = lattice["y_nodes_f32"]
        heights = lattice["heights_yx_f32"]
        if (xs.shape, ys.shape, heights.shape) != ((261,), (461,), (461, 261)):
            raise ValueError("unexpected retained Float32 bed lattice dimensions")
        xyz_expected = np.column_stack((np.tile(xs, ys.size), np.repeat(ys, xs.size),
                                        heights.reshape(-1))).astype("<f4", copy=False)
    if xyz_expected.shape != (120321, 3):
        raise ValueError(f"unexpected expected bed position shape {xyz_expected.shape}")
    expected_index = []
    nx, ny = xs.size, ys.size
    for y in range(ny - 1):
        a = bed_first + y * nx + np.arange(nx - 1, dtype=np.uint32)
        b, c, d = a + 1, a + nx, a + nx + 1
        expected_index.append(np.stack((a, b, c, b, d, c), axis=1).reshape(-1))
    expected_index = np.concatenate(expected_index).astype("<u4", copy=False)
    if new_bed_indices.tobytes() != expected_index.tobytes():
        raise ValueError("bed indices do not match the declared 00_10_01__10_11_01 cell diagonal/order")
    bed_xyz = np.ndarray((120321, 20), dtype="<f4", buffer=new_mm,
                         offset=new_sec[2][2] + bed_first * 80)[:, :3]
    if bed_xyz.tobytes() != xyz_expected.tobytes():
        raise ValueError("step-0 bed positions differ from the retained runtime-F32 lattice")

    instance = new_insts[BED_KEY]
    instance_raw = instance["raw"]
    translation_scale = struct.unpack_from("<4f", instance_raw, 0)
    quaternion = struct.unpack_from("<4f", instance_raw, 16)
    binding = struct.unpack_from("<4I", instance_raw, 32)
    identity = struct.unpack_from("<4I", instance_raw, 48)
    if translation_scale != (0., 0., 0., 1.) or quaternion != (0., 0., 0., 1.):
        raise ValueError("bed world transform is not the declared identity transform")
    if binding[2] != 0 or identity != (51999, 1, U32_MAX, 1):
        raise ValueError("bed is not the expected world-bound stable primitive")

    per_capture = []
    reference_bed_records = None
    reference_bed_indices = None
    reference_instance = None
    for step in STEPS:
        path = GEOM / f"step-{step}.mrvpack"
        receipt_info = record_receipt(path, step)
        stream, mm, sec = open_pack(path)
        prims, insts = table(mm, sec), instance_table(mm, sec)
        if BED_KEY not in prims or BED_KEY not in insts:
            raise ValueError(f"bed identity missing at accepted step {step}")
        bed = prims[BED_KEY]
        if bed["geom"][1] != 717600 or bed["ident"] != (51999, 1, U32_MAX, 1):
            raise ValueError(f"bed primitive geometry/identity changed at step {step}")
        ib = index_bytes(mm, sec, bed["geom"])
        first = int(np.frombuffer(ib, dtype="<u4").min())
        if first != bed_first or int(np.frombuffer(ib, dtype="<u4").max()) != bed_last:
            raise ValueError(f"bed vertex range changed at step {step}")
        xyz_rows = mm[sec[2][2] + first * 80:sec[2][2] + (first + 120321) * 80]
        # Save complete packed bed records, position bytes, indices, and instance bytes for exact cross-pose comparisons.
        packed_pos = np.ndarray((120321, 20), dtype="<f4", buffer=mm,
                                offset=sec[2][2] + first * 80)[:, :4].tobytes()
        inst_raw = insts[BED_KEY]["raw"]
        if reference_bed_records is None:
            reference_bed_records = xyz_rows
            reference_bed_indices = ib
            reference_instance = inst_raw
        elif xyz_rows != reference_bed_records or ib != reference_bed_indices or inst_raw != reference_instance:
            raise ValueError(f"bed vertex records, indices, or instance changed at step {step}")
        if packed_pos[:120321 * 16] != np.column_stack((xyz_expected, np.ones(120321, dtype="<f4"))).astype("<f4").tobytes():
            raise ValueError(f"bed position+homogeneous coordinate mismatch at step {step}")
        per_capture.append({"step": step, "receipt": receipt_info,
                            "pack_sha256": input_hashes[str(path)],
                            "bed_position_xyzw_sha256": hashlib.sha256(packed_pos).hexdigest(),
                            "bed_full_vertex_record_sha256": hashlib.sha256(xyz_rows).hexdigest(),
                            "bed_index_sha256": hashlib.sha256(ib).hexdigest(),
                            "bed_instance_sha256": hashlib.sha256(inst_raw).hexdigest()})
        mm.close(); stream.close()

    # The 1198 reference already had the same fixed bed; compare its full mesh directly.
    if old_bed["ident"] != (51999, 1, U32_MAX, 1) or old_bed["geom"][1] != 717600:
        raise ValueError("1198 step-0 bed mesh identity/count differs from expected")
    old_bed_indices = np.frombuffer(index_bytes(old_mm, old_sec, old_bed["geom"]), dtype="<u4")
    old_bed_xyz = np.ndarray((120321, 20), dtype="<f4", buffer=old_mm,
                             offset=old_sec[2][2] + bed_first * 80)[:, :3]
    old_bed_records = old_mm[old_sec[2][2] + bed_first * 80:old_sec[2][2] + (bed_first + 120321) * 80]
    new_bed_records = new_mm[new_sec[2][2] + bed_first * 80:new_sec[2][2] + (bed_first + 120321) * 80]
    if old_bed_indices.tobytes() != expected_index.tobytes() or old_bed_xyz.tobytes() != xyz_expected.tobytes():
        raise ValueError("1198 step-0 bed differs from the exact retained runtime Float32 lattice")
    if old_bed_records != new_bed_records:
        raise ValueError("1198 and 1199 complete bed vertex records differ")
    if old_insts[BED_KEY]["raw"] != new_insts[BED_KEY]["raw"]:
        raise ValueError("1198 and 1199 fixed-bed instance transforms differ")

    result = {
        "schema": "numi.human.contoured-bed-capture-audit.v1",
        "status": "pass_for_source_geometry_and_capture_binding_only",
        "scope_limit": "This is a byte-level scene-pack audit. It does not qualify presentation, contact physics, equilibrium, or anatomical fit.",
        "sources": {
            "declaration": {"path": str(DECLARATION), "sha256": input_hashes[str(DECLARATION)]},
            "invocation": {"path": str(INVOCATION), "sha256": input_hashes[str(INVOCATION)]},
            "run_metadata": {"path": str(RUN_META), "sha256": input_hashes[str(RUN_META)]},
            "native_log": {"path": str(NATIVE_LOG), "sha256": input_hashes[str(NATIVE_LOG)]},
            "manifest": {"path": str(MANIFEST), "sha256": input_hashes[str(MANIFEST)]},
            "runtime_lattice_npz": {"path": str(GRID), "sha256": input_hashes[str(GRID)]},
            "pack_receipt_validator": {"path": str(AUDITOR), "sha256": input_hashes[str(AUDITOR)]},
            "runtime_visual_source": {"path": str(VISUAL_SOURCE), "sha256": input_hashes[str(VISUAL_SOURCE)]},
            "runtime_bed_loader_source": {"path": str(BED_SOURCE), "sha256": input_hashes[str(BED_SOURCE)]},
            "visual_abi_header": {"path": str(TYPE_SOURCE), "sha256": input_hashes[str(TYPE_SOURCE)]},
            "baseline_1198_step0": {"pack_path": str(BASE_GEOM / "step-0.mrvpack"), "pack_sha256": input_hashes[str(BASE_GEOM / "step-0.mrvpack")],
                                     "receipt_path": str(BASE_GEOM / "step-0.receipt.json"), "receipt_sha256": input_hashes[str(BASE_GEOM / "step-0.receipt.json")]}
        },
        "initial_capture_comparison": {
            "steps": [0, 0],
            "accepted_body_state_sha256_equal": True,
            "accepted_respiration_state_sha256_equal": True,
            "accepted_root_fingerprint_equal": old_receipt["accepted_root_fingerprint"] == new_receipt["accepted_root_fingerprint"],
            "baseline_accepted_root_fingerprint": old_receipt["accepted_root_fingerprint"],
            "candidate_accepted_root_fingerprint": new_receipt["accepted_root_fingerprint"],
            "accepted_body_state_sha256": old_raw_receipt["accepted_body_state_sha256"],
            "accepted_respiration_state_sha256": old_raw_receipt["accepted_respiration_state_sha256"],
            "registered_body_pose_record_count": len(old_raw_receipt["accepted_registered_body_poses"]),
            "root_fingerprint_note": "Recorded separately; direct body and respiration hashes plus registered body poses are the state identity checks.",
            "registered_body_poses_equal": True,
            "nonbed_primitive_records_equal": len(nonbed),
            "nonbed_primitive_index_streams_equal": len(nonbed),
            "nonbed_instance_records_equal": len(nonbed),
            "nonbed_referenced_vertices_byte_equal": True,
            "nonbed_vertex_max_index": max_nonbed_vertex,
            "first_bed_vertex_index": bed_first,
            "baseline_bed_index_count": old_bed["geom"][1],
            "candidate_bed_triangle_count": bed_prim["geom"][1] // 3,
            "candidate_bed_vertex_count": int(bed_xyz.shape[0])
        },
        "bed_lattice_check": {
            "primitive_key": {"semantic": BED_KEY[0], "stable_id": BED_KEY[1]},
            "world_binding": list(binding),
            "translation_scale_f32": list(translation_scale),
            "orientation_xyzw_f32": list(quaternion),
            "instance_raw_sha256": hashlib.sha256(instance_raw).hexdigest(),
            "grid_shape_nodes_yx": list((ny, nx)),
            "grid_cell_count": int((ny - 1) * (nx - 1)),
            "triangle_count": int((ny - 1) * (nx - 1) * 2),
            "index_count": int(expected_index.size),
            "vertex_positions_bitwise_equal_to_runtime_f32_lattice": True,
            "indices_bitwise_equal_to_declared_cell_diagonal": True,
            "bed_xyz_sha256": hashlib.sha256(bed_xyz.tobytes()).hexdigest(),
            "bed_indices_sha256": hashlib.sha256(expected_index.tobytes()).hexdigest(),
            "normal_usage_note": "Static vertex normals/tangents are render shading records; this audit does not use them as the contact surface. Contact geometry is assessed by separate support-query evidence."
        },
        "native_run": {
            "accepted_steps": STEPS,
            "completed_marker_present": True,
            "presentation_qualification": "pending in native log; not claimed here",
            "capture_rows": per_capture
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT),
                      "status": result["status"],
                      "steps": STEPS,
                      "initial": result["initial_capture_comparison"],
                      "bed": result["bed_lattice_check"]}, indent=2))
    # Mapped views are intentionally left for interpreter teardown; NumPy retains
    # exported read-only views until process exit.

if __name__ == "__main__":
    main()
