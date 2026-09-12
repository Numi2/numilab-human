"""Source-bound behavior metric-program authoring; no physical stepping or metrics.

NHBHV1 stores source-frame bindings; it does not replace a generic TaskPack. The native compiler must apply its actual
COM rebase and report its coverage. This artifact alone is never trial evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

from .behavior_qualification import EvidenceError, _decode

SCHEMA = "numi.human.behavior-task.v1"
MANIFEST_SCHEMA = "numi.human.behavior-task-source.v1"
MAGIC = b"NHBHV1\0\0"
HEADER = struct.Struct("<8s8I2Q192s")
BINDING = struct.Struct("<2I7d")
NUMERICAL = struct.Struct("<16d")
TASKS = {"standing": 0, "recovery": 1, "walking": 2}
CRITERIA = frozenset({"minimum_root_height_m", "maximum_trunk_tilt_rad",
    "maximum_planar_speed_mps", "root_body_semantic_id", "trunk_body_semantic_id",
    "world_reference_origin_m", "world_up_axis", "world_forward_axis",
    "trunk_up_axis_body", "velocity_observable", "velocity_body_semantic_id",
    "forbidden_contact_semantic_ids"})


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read(path: Path) -> bytes:
    _need(not path.is_symlink() and path.is_file(), f"not a regular immutable input: {path}")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
          (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"input changed while reading: {path}")
    return raw


def _number(value: Any, label: str) -> float:
    try:
        result = float(value) if type(value) in (float, int) else math.nan
    except (OverflowError, ValueError):
        result = math.nan
    _need(math.isfinite(result), f"{label}: finite number required")
    return result


def _vector(value: Any, label: str, size: int = 3, unit: bool = False) -> list[float]:
    _need(type(value) is list and len(value) == size, f"{label}: expected {size} coordinates")
    result = [_number(v, label) for v in value]
    if unit:
        _need(abs(sum(x*x for x in result) - 1.0) <= 1e-10, label + ": unit vector required")
    return result


def _inverse_rotate(q: list[float], p: list[float]) -> list[float]:
    x, y, z, w = q
    v = [-x, -y, -z]
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    t = [2*x for x in cross(v, p)]
    c = cross(v, t)
    return [p[i] + w*t[i] + c[i] for i in range(3)]


def _validate_task(task: dict) -> dict:
    _need(set(task) == {"schema", "task", "step_ns", "criteria", "target_speed_mps"}, "unknown or missing task fields")
    _need(task["schema"] == SCHEMA and type(task["task"]) is str and task["task"] in TASKS, "invalid task schema or kind")
    _need(type(task["step_ns"]) is int and 0 < task["step_ns"] <= (1 << 63)-1, "step_ns must be a positive exact int64")
    c = task["criteria"]
    _need(type(c) is dict and set(c) == CRITERIA, "unknown or missing criterion")
    _need(c["velocity_observable"] == "body_com_linear_velocity", "whole-Human COM is unsupported until all mechanical mass owners are included")
    for field in ("root_body_semantic_id", "trunk_body_semantic_id", "velocity_body_semantic_id"):
        _need(type(c[field]) is str and bool(c[field]) and c[field] == c[field].strip(), "invalid semantic body id")
    forbidden = c["forbidden_contact_semantic_ids"]
    _need(type(forbidden) is list and bool(forbidden) and all(type(x) is str and bool(x) for x in forbidden)
          and len(set(forbidden)) == len(forbidden), "forbidden contacts require unique source semantic ids")
    _vector(c["world_reference_origin_m"], "world origin")
    up = _vector(c["world_up_axis"], "world up", unit=True)
    forward = _vector(c["world_forward_axis"], "world forward", unit=True)
    _need(abs(sum(a*b for a, b in zip(up, forward))) <= 1e-10, "world axes must be orthogonal")
    _vector(c["trunk_up_axis_body"], "trunk up", unit=True)
    _need(_number(c["minimum_root_height_m"], "minimum height") >= 0, "negative minimum height")
    _need(0 <= _number(c["maximum_trunk_tilt_rad"], "maximum tilt") <= math.pi, "tilt outside [0, pi]")
    _need(_number(c["maximum_planar_speed_mps"], "maximum speed") >= 0, "negative maximum speed")
    if task["task"] == "walking":
        _need(_number(task["target_speed_mps"], "target speed") >= 0, "negative target speed")
    else:
        _need(task["target_speed_mps"] is None, "nonwalking target speed must be null")
    return c


def author_task(*, task_path: Path, human_manifest: Path, source_export: Path) -> tuple[bytes, dict]:
    """Validate exact consumed inputs and return deterministic bytes + manifest.

    Selected body names come from the pinned source export and every source
    body mapping/pose is compared with the actual NHRIGID2 payload. There is no
    fallback from a semantic name to an assumed body number.
    """
    task_raw, human_raw, export_raw = (_read(p) for p in (task_path, human_manifest, source_export))
    task, human, exported = (_decode(raw, label) for raw, label in
        ((task_raw, "task"), (human_raw, "Human manifest"), (export_raw, "source export")))
    c = _validate_task(task)
    _need(human.get("schema") == "numi.human.myosim-fullbody-reference.v1", "wrong Human reference schema")
    _need(exported.get("schema") == "numi.human.myosim-mujoco-export.v1", "wrong source export schema")
    _need(type(human.get("source")) is dict and human["source"] == exported.get("source"), "source export/manifest identity mismatch")
    record = human.get("payloads", {}).get("rigid", {})
    name = record.get("file")
    _need(type(name) is str and name not in ("", ".", "..") and Path(name).name == name, "rigid payload must be a sibling file")
    rigid = _read(human_manifest.parent / name)
    _need(len(rigid) == record.get("bytes") and _sha(rigid) == record.get("sha256"), "rigid payload hash/size mismatch")
    _need(len(rigid) >= 224, "truncated rigid payload")
    h = struct.unpack_from("<8s10I32s", rigid)
    _need(h[0] == b"NHRIGID2" and h[1] == 1 and h[8] == 0 and h[10] == 0, "unsupported rigid payload header")
    source_count, body_count, joints, nq, nv = h[3:8]
    _need(0 < source_count <= body_count <= 4096 and 0 < nq <= 8192 and 0 < nv <= 4096, "invalid rigid shape")
    _need(len(rigid) == 224 + 160*body_count + 144*joints + 64*nv + 4*(nq+nv) + 32*source_count,
          "rigid record count mismatch")
    _need(h[-1].hex() == human["source"].get("archive_sha256"), "rigid source archive mismatch")
    records = human.get("core_tree", {}).get("source_body_records")
    bodies = exported.get("bodies")
    _need(type(records) is list and len(records) == source_count and type(bodies) is list, "missing source body catalog")
    by_id = {}
    for b in bodies:
        _need(type(b) is dict and type(b.get("id")) is int and b["id"] not in by_id, "duplicate or invalid source body id")
        by_id[b["id"]] = b
    _need(len(by_id) == source_count, "export/source body count mismatch")
    catalog, seen_core, seen_ids = {}, set(), set()
    map_offset = len(rigid) - 32*source_count
    for i, r in enumerate(records):
        _need(type(r) is dict, "invalid source body record")
        sid, core, semantic = r.get("source_body_id"), r.get("core_body_index"), r.get("name")
        _need(type(sid) is int and 0 <= sid < 2**32 and sid not in seen_ids and sid in by_id, "source body identity mismatch")
        _need(type(core) is int and 0 <= core < body_count and core not in seen_core and
              core == struct.unpack_from("<I", rigid, map_offset+4*i)[0], "source/core mapping mismatch")
        _need(type(semantic) is str and bool(semantic) and semantic not in catalog, "duplicate or empty semantic body name")
        b = by_id[sid]
        _need(semantic == b.get("name"), "source semantic name mismatch")
        pose = _vector(r.get("default_com_position_world_m"), "COM pose") + _vector(
            r.get("default_inertial_quaternion_world_xyzw"), "inertial pose", size=4, unit=True)
        _need(pose == _vector(b.get("default_com_position_world_m"), "source COM pose") + _vector(
            b.get("default_inertial_quaternion_world_xyzw"), "source inertial pose", size=4, unit=True),
            "export/reference source pose mismatch")
        _need(struct.pack("<7f", *pose) == rigid[map_offset+4*source_count+28*i:map_offset+4*source_count+28*(i+1)],
              "source pose disagrees with actual rigid bytes")
        inertia_origin = _vector(b.get("inertial_position_body_m"), "source inertial origin")
        source_to_inertia = _vector(b.get("inertial_quaternion_body_xyzw"), "source inertial rotation", size=4, unit=True)
        origin = _inverse_rotate(source_to_inertia, [-x for x in inertia_origin])
        # The independent default source-body pose must agree with the same
        # transform. This denies a forged local origin while retaining COM.
        world_origin = _vector(b.get("default_body_position_world_m"), "source body origin")
        world_q = _vector(b.get("default_body_quaternion_world_xyzw"), "source body orientation", size=4, unit=True)
        expected_local = _inverse_rotate(pose[3:], [world_origin[k]-pose[k] for k in range(3)])
        _need(max(abs(a-b) for a,b in zip(expected_local, origin)) < 1e-9, "source origin transform mismatch")
        # Check orientation composition through all three basis vectors; q and
        # -q represent the same orientation and are deliberately equivalent.
        for axis in ([1.,0.,0.], [0.,1.,0.], [0.,0.,1.]):
            qinv = [-x for x in world_q[:3]] + [world_q[3]]
            in_world = _inverse_rotate(qinv, axis)
            expected_axis = _inverse_rotate(pose[3:], in_world)
            actual_axis = _inverse_rotate(source_to_inertia, axis)
            _need(max(abs(a-b) for a,b in zip(expected_axis, actual_axis)) < 1e-9, "source orientation transform mismatch")
        catalog[semantic] = {"source_body_id": sid, "source_record_index": i, "core_body_index": core,
            "source_origin_in_original_com_frame_m": origin,
            "source_inertial_quaternion_body_xyzw": source_to_inertia}
        seen_ids.add(sid); seen_core.add(core)
    selected_names = [c[key] for key in ("root_body_semantic_id", "trunk_body_semantic_id", "velocity_body_semantic_id")]
    forbidden_names = c["forbidden_contact_semantic_ids"]
    _need(all(name in catalog for name in selected_names+forbidden_names), "unresolved semantic body or forbidden contact id")
    catalog_bytes = _canonical(catalog)
    hashes = [h[-1].hex(), _sha(rigid), _sha(human_raw), _sha(export_raw), _sha(task_raw), _sha(catalog_bytes)]
    numerical = [*c["world_reference_origin_m"], *c["world_up_axis"], *c["world_forward_axis"], *c["trunk_up_axis_body"],
        c["minimum_root_height_m"], c["maximum_trunk_tilt_rad"], c["maximum_planar_speed_mps"], task["target_speed_mps"] or 0.0]
    bindings = b"".join(BINDING.pack(catalog[name]["source_record_index"], catalog[name]["core_body_index"],
        *catalog[name]["source_origin_in_original_com_frame_m"], *catalog[name]["source_inertial_quaternion_body_xyzw"])
        for name in selected_names)
    contacts = b"".join(struct.pack("<2I", catalog[name]["source_record_index"], catalog[name]["core_body_index"]) for name in forbidden_names)
    total = HEADER.size + len(bindings) + NUMERICAL.size + len(contacts)
    raw = HEADER.pack(MAGIC, 1, HEADER.size, total, body_count, nq, nv, len(forbidden_names), TASKS[task["task"]],
                      task["step_ns"], 0, bytes.fromhex("".join(hashes))) + bindings + NUMERICAL.pack(*numerical) + contacts
    manifest = {"schema": MANIFEST_SCHEMA, "task": task, "catalog": catalog,
        "inputs": {label: {"sha256": _sha(data), "bytes": len(data)} for label, data in
                   (("human_manifest", human_raw), ("source_export", export_raw), ("rigid", rigid), ("task", task_raw))},
        "catalog_sha256": _sha(catalog_bytes), "source_archive_sha256": h[-1].hex(),
        "payload": {"file": "task.nhbhv", "sha256": _sha(raw), "bytes": len(raw)},
        "qualification": {"kind": "source_bound_task_authoring", "physical_steps": 0,
            "accepted_root_telemetry": False, "behavior_qualified": False,
            "native_cooked_origin_rebase_required": True,
            "forbidden_contact_coverage": "requires_native_owner", "audit_coverage": "requires_native_owner",
            "whole_human_com_supported": False}}
    return raw, manifest


def compile_task(*, task_path: Path, human_manifest: Path, source_export: Path, output: Path) -> dict:
    raw, manifest = author_task(task_path=task_path, human_manifest=human_manifest, source_export=source_export)
    expected = {"task.nhbhv": raw, "manifest.json": _canonical(manifest)}
    if output.exists():
        _need(not output.is_symlink() and output.is_dir() and set(p.name for p in output.iterdir()) == set(expected), "existing output inventory mismatch")
        for name, data in expected.items():
            _need(_read(output/name) == data, "existing immutable output changed: " + name)
        return manifest
    output.mkdir(parents=True)
    # Exclusive creation: never overwrite a previous task or partially promote
    # changed inputs. A failed write remains visibly incomplete.
    for name, data in expected.items():
        with (output/name).open("xb") as stream:
            stream.write(data)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compile_task(task_path=args.task, human_manifest=args.human_manifest, source_export=args.source_export, output=args.output)
    except (EvidenceError, OSError, struct.error, OverflowError) as error:
        parser.exit(2, f"behavior task admission failed: {error}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
