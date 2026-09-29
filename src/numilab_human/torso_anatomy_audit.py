"""Verify native torso surfaces against retained source meshes and poses.

This is a geometry/ownership oracle. It does not admit organ mechanics,
whole-organ completeness, or clinical anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pack_sections(path: Path) -> dict[int, tuple[bytes, int, int]]:
    raw = path.read_bytes()
    _require(len(raw) >= 32, "truncated native visual pack")
    magic, version, count, directory, file_bytes = struct.unpack_from("<8sIIQQ", raw)
    _require(magic == b"MRVPACK2" and version == 2 and file_bytes == len(raw), "native visual pack header")
    _require(directory + 72 * count <= len(raw), "native visual pack directory")
    sections = {}
    for i in range(count):
        kind, index, offset, size, elements, stride, _, digest = struct.unpack_from(
            "<IIQQQII32s", raw, directory + 72 * i)
        _require(index == 0 and kind not in sections and offset + size <= len(raw), "native visual section range")
        section = raw[offset:offset + size]
        _require(hashlib.sha256(section).digest() == digest, "native visual section hash")
        sections[kind] = section, elements, stride
    return sections


def _native_source_family_coverage(requirements, measured, relations, owners):
    """Independently compare rendered members/owners with source family sets."""
    _require(isinstance(requirements, list) and bool(requirements), "native source coverage requirements")
    checks = []
    seen = set()
    for requirement in requirements:
        identity = requirement["id"]
        _require(identity not in seen and requirement["selection"] == "complete_source_membership",
                 "native source coverage requirement identity/selection")
        seen.add(identity)
        expected = {member for concept, name, member in relations[requirement["hierarchy"]]
                    if concept == requirement["concept_id"] and name == requirement["source_name"]}
        _require(bool(expected), "native source family has no exact source members")
        layers = {member: requirement["layer"] for member in expected}
        if requirement["layer"] == "source_typed":
            for member in expected:
                matches = [rule["layer"] for rule in requirement["source_type_layers"] if
                           (rule["concept_id"], rule["source_name"], member) in relations["is_a"]]
                _require(len(matches) == 1 and matches[0] != "source_typed",
                         "native family source type missing or ambiguous")
                layers[member] = matches[0]
        actual_rows = [row for row in measured if row["member_id"] in expected]
        actual = {row["member_id"] for row in actual_rows}
        owner = owners[requirement["myosim_body"]][0]
        wrong = sorted(row["member_id"] for row in actual_rows
                       if row["core_body_index"] != owner or row["layer"] != layers[row["member_id"]])
        checks.append({"id": identity, "expected_members": sorted(expected), "rendered_members": sorted(actual),
                       "missing_members": sorted(expected - actual), "wrong_layer_or_body_members": wrong,
                       "required_member_layers": layers,
                       "passed": actual == expected and len(actual_rows) == len(actual) and not wrong})
    return {"requirements": checks, "passed": all(row["passed"] for row in checks),
            "boundary": "Declared atlas family coverage in native geometry, not clinical or whole-organ completeness."}


def audit_torso_anatomy(
    sources: Path, artifact: Path, registration_path: Path, payload: Path,
    native_pack: Path, native_poses: Path, pose: tuple[tuple[int, float], ...] | None,
    *, native_surface_count: int | None = None, visible_layer_mask: int = 63,
) -> dict[str, Any]:
    import mujoco
    import numpy as np
    from myo_sim.build.compose import build_model
    from . import model as human
    from .joint_constraint_consistency import source_equality_projection_oracle
    from .torso_anatomy_coverage import source_family_topology, source_organ_coverage
    from .upper_limb_pose_audit import _pose_qpos

    registration = json.loads(registration_path.read_text())
    source_checks = human._require_myosim_rigid_program(sources, artifact)
    _, owners = human._bodyparts_runtime_bindings(registration, artifact)
    mapping = json.loads((human.REPOSITORY_ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json").read_text())
    specs = mapping["entries"]
    manifest_path = payload.with_name("bodyparts3d-myosim-torso-anatomy.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["payload"]["sha256"] == human.sha256(payload), "torso payload manifest hash")
    _require(manifest["source"]["registration"]["sha256"] == human.sha256(registration_path), "torso registration hash")
    _require(manifest["source"]["surface_map"]["sha256"] == human.sha256(
        human.REPOSITORY_ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json"), "torso map hash")
    _require(manifest["source"]["bodyparts"] == registration["source"]["bodyparts"], "torso atlas provenance")
    for archive in registration["source"]["bodyparts"]["archives"]:
        _require(human.sha256(sources / archive["file"]) == archive["sha256"], "torso source archive hash")
    tables = manifest["source"]["source_relation_tables"]
    _require({row["hierarchy"] for row in tables} == {"part_of", "is_a"} and len(tables) == 2, "torso source relation coverage")
    for table in tables:
        _require(human.sha256(sources / table["file"]) == table["sha256"], "torso source relation hash")
    surface_sources = manifest["source"]["surfaces"]
    _require(len(surface_sources) == len(specs), "torso surface provenance coverage")
    raw = payload.read_bytes()
    magic, abi, surface_count, vertex_count, index_count, fingerprint, source_sha = struct.unpack_from("<8s5I32s", raw)
    _require(magic == b"NHANAT1\0" and abi in {1, 2} and surface_count == len(specs), "torso payload header")
    _require(surface_count <= (64 if abi == 1 else 1024), "torso payload ABI surface capacity")
    _require(fingerprint == int(human.sha256(registration_path)[:8], 16), "torso payload registration fingerprint")
    _require(source_sha.hex() == registration["source"]["myosim"]["source"]["archive_sha256"], "torso payload source hash")
    vertices_offset = 60 + 32 * surface_count
    indices_offset = vertices_offset + 24 * vertex_count
    _require(len(raw) == indices_offset + 4 * index_count, "torso payload byte count")
    records = np.frombuffer(raw, dtype="<u4", count=8 * surface_count, offset=60).reshape(-1, 8)
    vertices = np.frombuffer(raw, dtype="<f4", count=6 * vertex_count, offset=vertices_offset).reshape(-1, 6)
    indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=indices_offset)
    _require(bool(np.isfinite(vertices).all()), "nonfinite torso vertices")
    sections = _pack_sections(native_pack)
    for kind, stride in [(2, 80), (3, 4), (4, 64), (5, 80)]:
        _require(kind in sections and sections[kind][2] == stride
                 and len(sections[kind][0]) == sections[kind][1] * stride, "native visual section layout")
    packed_vertices = np.frombuffer(sections[2][0], dtype="<f4").reshape(-1, 20)
    packed_indices = np.frombuffer(sections[3][0], dtype="<u4")
    primitives = np.frombuffer(sections[4][0], dtype="<u4").reshape(-1, 16)
    instances_u = np.frombuffer(sections[5][0], dtype="<u4").reshape(-1, 20)
    instances_f = np.frombuffer(sections[5][0], dtype="<f4").reshape(-1, 20)
    semantic_codes = {"organ": 51010, "vessel": 51011, "nerve": 51012,
                      "airway": 51020, "pulmonary_artery": 51021, "pulmonary_vein": 51022}
    selected = primitives[np.isin(primitives[:, 4], list(semantic_codes.values()))]
    _require(len(selected) == surface_count and len(set(selected[:, 5])) == surface_count, "native torso surface coverage")
    by_id = {int(p[5]): p for p in selected}
    snapshot = json.loads(native_poses.read_text())
    expected_native_count = surface_count if native_surface_count is None else native_surface_count
    _require(surface_count <= expected_native_count <= 1024 and 0 < visible_layer_mask <= 255,
             "native extended anatomy count/visibility contract")
    _require(snapshot["schema"] == "numi.human.native-torso-anatomy-pose-snapshot.v1"
             and snapshot["surface_count"] == expected_native_count
             and snapshot["registration_fingerprint32"] == fingerprint, "native torso pose header")
    poses = {b["body_index"]: b for b in snapshot["bodies"]}
    _require(len(poses) == len(snapshot["bodies"]), "duplicate native torso pose owners")
    model = build_model("myofullbody")
    rest = mujoco.MjData(model)
    rest.qpos[:] = model.qpos0
    mujoco.mj_forward(model, rest)
    data = mujoco.MjData(model)
    if pose is None:
        data.qpos[:] = model.qpos0
    else:
        data.qpos[:] = _pose_qpos(model, pose, mujoco, np)[0]
    mujoco.mj_forward(model, data)
    equality_oracle = source_equality_projection_oracle(model, data, mujoco, 1e-9) if pose is not None else None
    _require(equality_oracle is None or equality_oracle["passed"], "source joint equality oracle")
    matrix = np.asarray(registration["coordinate_system"]["global_source_mm_to_myosim_world_m"])
    _require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), "source global transform")
    source_relations = {h: human._bodyparts_source_element_relation_names(sources, h)
                        for h in {s["hierarchy"] for s in specs} | {"is_a"}}
    family_members = {member for requirement in mapping["coverage_requirements"]
                      for concept, label, member in source_relations[requirement["hierarchy"]]
                      if concept == requirement["concept_id"] and label == requirement["source_name"]}
    rows = []
    layer_codes = {"organ": 1, "vessel": 2, "nerve": 3,
                   "airway": 4, "pulmonary_artery": 5, "pulmonary_vein": 6}
    for stable, (spec, record) in enumerate(zip(specs, records, strict=True), 1):
        body, first_v, count_v, first_i, count_i, identity, layer, reserved = map(int, record)
        _require((spec["concept_id"], spec["source_name"], spec["member_id"]) in source_relations[spec["hierarchy"]], "source surface relation")
        expected_body, source_body = owners[spec["myosim_body"]]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["myosim_body"])
        _require(sid == source_body["source_body_id"] and body == expected_body, "torso source body owner")
        _require(identity == stable and layer == layer_codes[spec["layer"]] and reserved == 0, "torso surface identity")
        _require(abi == 2 or layer <= 3, "torso payload ABI layer identity")
        _require(first_v + count_v <= vertex_count and first_i + count_i <= index_count, "torso surface range")
        _, member, obj = human._bodyparts_obj_member(sources, spec["hierarchy"], spec["member_id"])
        declared_source = surface_sources[stable - 1]
        _require(declared_source["stable_id"] == stable and declared_source["member_id"] == spec["member_id"]
                 and declared_source["member"] == member
                 and declared_source["member_sha256"] == hashlib.sha256(obj).hexdigest(), "torso source member hash/identity")
        source_types = [{"concept_id": concept, "label": name} for concept, name, typed_member
                        in sorted(source_relations["is_a"]) if typed_member == spec["member_id"]]
        _require(declared_source.get("source_is_a_types") == source_types, "torso source type provenance")
        if spec["layer"] == "organ":
            typed = source_organ_coverage(spec, source_types)
            _require(all(declared_source.get(key) == value for key, value in typed.items()), "torso source organ typing")
        topology = source_family_topology(obj, member) if spec["member_id"] in family_members else None
        _require(declared_source.get("source_family_topology") == topology, "torso source topology diagnostic")
        source_v, triangles = human._bodyparts_obj_triangles(obj, member)
        source_v = np.asarray(source_v)
        _require(count_v == len(source_v) and count_i == 3 * len(triangles), "source surface topology size")
        expected_indices = np.asarray(triangles).ravel()
        _require(np.array_equal(indices[first_i:first_i + count_i] - first_v, expected_indices), "source surface topology")
        world_rest = source_v @ matrix[:3, :3].T + matrix[:3, 3]
        # Independent MuJoCo inertial frames, rather than the compiler's
        # declared/default COM or world-to-body helper.
        local = (world_rest - rest.xipos[sid]) @ rest.ximat[sid].reshape(3, 3)
        expected_world = local @ data.ximat[sid].reshape(3, 3).T + data.xipos[sid]
        faces = np.asarray(triangles)
        face_normals = np.cross(source_v[faces[:, 1]] - source_v[faces[:, 0]],
                                source_v[faces[:, 2]] - source_v[faces[:, 0]])
        source_normals = np.zeros_like(source_v)
        for corner in range(3):
            np.add.at(source_normals, faces[:, corner], face_normals)
        lengths = np.linalg.norm(source_normals, axis=1)
        degenerate = lengths <= 1e-12
        source_normals[degenerate] = source_v[degenerate] - source_v.mean(axis=0)
        lengths = np.linalg.norm(source_normals, axis=1)
        _require(bool((lengths > 1e-12).all()), "source surface degenerate normal")
        source_normals /= lengths[:, None]
        normal_world_rest = source_normals @ matrix[:3, :3].T
        normal_world_rest /= np.linalg.norm(normal_world_rest, axis=1)[:, None]
        local_normals = normal_world_rest @ rest.ximat[sid].reshape(3, 3)
        primitive = by_id[stable]
        _require(int(primitive[4]) == semantic_codes[spec["layer"]] and int(primitive[6]) == body,
                 "native surface semantic/owner")
        p_first, p_count, _, instance_index = map(int, primitive[:4])
        _require(p_count == count_i and p_first + p_count <= len(packed_indices)
                 and instance_index < len(instances_u), "native surface index range")
        native_indices = packed_indices[p_first:p_first + p_count]
        p_vertex = int(native_indices.min())
        _require(np.array_equal(native_indices - p_vertex, expected_indices)
                 and p_vertex + count_v <= len(packed_vertices), "native source topology")
        iu, transform = instances_u[instance_index], instances_f[instance_index, :8]
        _require(int(iu[9]) == body and int(iu[10]) == 3
                 and np.array_equal(iu[12:16], primitive[4:8]), "native articulated instance owner")
        _require(int(iu[11]) == (11 if visible_layer_mask & (1 << (layer - 1)) else 0),
                 "native source anatomy visibility")
        _require(np.array_equal(transform, [0, 0, 0, 1, 0, 0, 0, 1]), "native unexpected surface local transform")
        native_local = packed_vertices[p_vertex:p_vertex + count_v, :3]
        _require(bool(np.isfinite(native_local).all()), "nonfinite native surface")
        native_pose = poses[body]
        quaternion = np.asarray(native_pose["orientation_world_xyzw"])
        position = np.asarray(native_pose["position_world_m"])
        _require(quaternion.shape == (4,) and position.shape == (3,)
                 and bool(np.isfinite(quaternion).all()) and bool(np.isfinite(position).all())
                 and abs(float(np.linalg.norm(quaternion)) - 1) <= 0.002, "native pose values")
        native_rotation = np.empty(9)
        mujoco.mju_quat2Mat(native_rotation, quaternion[[3, 0, 1, 2]] / np.linalg.norm(quaternion))
        native_world = native_local @ native_rotation.reshape(3, 3).T + position
        pose_position_error = float(np.linalg.norm(position - data.xipos[sid]))
        orientation_error = float(np.max(np.linalg.norm(
            native_rotation.reshape(3, 3) - data.ximat[sid].reshape(3, 3), axis=0)))
        local_error = float(np.max(np.linalg.norm(vertices[first_v:first_v + count_v, :3] - local, axis=1)))
        pack_error = float(np.max(np.linalg.norm(native_local - local, axis=1)))
        world_error = float(np.max(np.linalg.norm(native_world - expected_world, axis=1)))
        normals_error = float(np.max(np.abs(np.linalg.norm(vertices[first_v:first_v + count_v, 3:], axis=1) - 1)))
        native_normals = packed_vertices[p_vertex:p_vertex + count_v, 4:7]
        _require(bool(np.isfinite(native_normals).all()), "nonfinite native surface normals")
        normal_source_error = float(max(
            np.max(np.linalg.norm(vertices[first_v:first_v + count_v, 3:] - local_normals, axis=1)),
            np.max(np.linalg.norm(native_normals - local_normals, axis=1))))
        rows.append({
            "stable_id": stable, "member_id": spec["member_id"], "label": spec["source_name"], "layer": spec["layer"],
            "source_member_sha256": hashlib.sha256(obj).hexdigest(), "source_body_id": int(sid), "core_body_index": body,
            "source_family_topology": topology,
            "vertex_count": count_v, "triangle_count": len(triangles), "topology_exact": True,
            "payload_local_error_m": local_error, "native_pack_local_error_m": pack_error,
            "native_pose_world_error_m": world_error, "normal_unit_error": normals_error,
            "native_COM_position_error_m": pose_position_error,
            "native_orientation_unit_witness_error_m": orientation_error,
            "source_normal_direction_error": normal_source_error,
            "allowed_geometry_error_m": 2e-5, "allowed_COM_and_orientation_witness_error_m": 1e-6,
            "allowed_normal_unit_error": .002, "allowed_normal_direction_error": 2e-5,
            "passed": (max(local_error, pack_error, world_error) <= 2e-5 and normals_error <= .002
                       and normal_source_error <= 2e-5 and max(pose_position_error, orientation_error) <= 1e-6),
        })
    _require(set(poses) == set(records[:, 0]), "native torso pose coverage")
    family_coverage = _native_source_family_coverage(mapping.get("coverage_requirements"), rows, source_relations, owners)
    return {
        "schema": "numi.human.native-torso-anatomy-source-audit.v1",
        "passed": all(row["passed"] for row in rows) and family_coverage["passed"], "surface_count": surface_count,
        "native_source_family_coverage": family_coverage,
        "vertex_count": sum(row["vertex_count"] for row in rows), "rows": rows,
        "maximum_native_pose_world_error_m": max(row["native_pose_world_error_m"] for row in rows),
        "source_equality_projection_oracle": equality_oracle, "rigid_source_program_checks": source_checks,
        "inputs": {name: {"path": str(path.resolve()), "sha256": human.sha256(path)} for name, path in {
            "registration": registration_path, "payload": payload, "payload_manifest": manifest_path,
            "native_pack": native_pack, "native_poses": native_poses,
        }.items()},
        "pose_coordinates": list(pose) if pose is not None else None,
        "tolerance_basis": "existing source skin rest reconstruction tolerance, applied to organ float32 geometry",
        "boundary": "Exact source topology, native pose and single-link geometry binding only; no organ completeness, clinical registration, deformation, contact, physiology or load qualification.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["sources", "artifact", "registration", "payload", "native-pack", "native-poses", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--raw-source-rest", action="store_true")
    parser.add_argument("--pose-q", nargs=2, action="append", default=[], metavar=("INDEX", "VALUE"))
    args = parser.parse_args()
    if args.raw_source_rest and args.pose_q:
        parser.error("raw source rest cannot have pose coordinates")
    pose = None if args.raw_source_rest else tuple((int(i), float(v)) for i, v in args.pose_q)
    result = audit_torso_anatomy(args.sources, args.artifact, args.registration, args.payload,
                                args.native_pack, args.native_poses, pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["passed", "surface_count", "vertex_count", "maximum_native_pose_world_error_m"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
