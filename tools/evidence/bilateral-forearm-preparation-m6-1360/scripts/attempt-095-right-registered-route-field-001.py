#!/usr/bin/env python3
"""Validate retained right FDS/FDP partition geometry and build its registered route field.

This is a CPU-only authoring diagnostic. It does not edit the NHTISS source,
launch native work, or claim anatomical/native admission.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import resource
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
import numpy as np

ROOT = Path("/Users/n/numi-human-fds132-m6-1350")
A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
R = Path("/Users/n/numi-human-retained-delivery-20261009")
OUT = ROOT / "attempt-095-right-registered-route-field-001"
RES = OUT / "result-001"
PART = A / "fds-right-paired-partition-115"
CERT = A / "fds-right-paired-certificate-116"
PROJ = A / "fds-right-paired-shortedge-projection-121"
ROUTE = ROOT / "attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json"
ROUTE_EXTRACTOR = ROOT / "attempt-050/arrangement-001/diagnostic-061/trace_fds_fdp_source_routes_061.py"
REG = ROOT / "registration-source-b1b410ad.json"
REG_SHA = "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"
TISS = A / "temporalis164-native-m6-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS_MANIFEST = TISS.with_suffix(".manifest.json")
TISS_HELPER = A / "neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
RIGID = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
ARCHIVE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz")
CI_ROOT = Path("/Users/n/numi-human-exact-normalization-m6-1348/src")
SOURCE_ROWS = {131: PART / "candidate-row-131.npz", 133: PART / "candidate-row-133.npz"}
MAPS = {sid: PROJ / f"stable-{sid}-exact-binding-materialization.pkl" for sid in (131, 133)}
MAX_RSS_KIB = int(2.5 * 1024 * 1024)
START = time.monotonic()
F = Fraction


def watchdog() -> None:
    while True:
        try:
            rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip() or "0")
            if rss > MAX_RSS_KIB:
                print(json.dumps({"fatal": "RSS watchdog exceeded 2.5 GiB", "rss_kib": rss}), flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(2)


threading.Thread(target=watchdog, daemon=True).start()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")
    temp.replace(path)


def req(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exact_point(point) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(F.from_float(float(value)) for value in point)


def rotate_double(q, v):
    x, y, z, w = [float(value) for value in q]
    vx, vy, vz = [float(value) for value in v]
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    return np.asarray(
        [vx + 2.0 * (w * cx + y * cz - z * cy),
         vy + 2.0 * (w * cy + z * cx - x * cz),
         vz + 2.0 * (w * cz + x * cy - y * cx)],
        dtype=np.float64,
    )


def parse_nhrigid(path: Path, archive: Path) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    raw = path.read_bytes()
    fields = struct.unpack_from("<8s10I32s", raw, 0)
    magic, pabi, eabi, source_count, body_count, joint_count, nq, nv, root, virtual, reserved, source_sha = fields
    req(magic == b"NHRIGID2" and pabi == 1 and nq == nv + 1, "NHRIGID header invalid")
    req(source_sha.hex() == sha(archive), "NHRIGID embedded source archive SHA mismatch")
    body_offset = 224
    joint_offset = body_offset + 160 * body_count
    dof_offset = joint_offset + 144 * joint_count
    q_offset = dof_offset + 64 * nv
    q = np.frombuffer(raw, dtype="<f4", count=nq, offset=q_offset).copy()
    v_offset = q_offset + 4 * nq
    map_offset = v_offset + 4 * nv
    source_to_core = np.frombuffer(raw, dtype="<u4", count=source_count, offset=map_offset).copy()
    pose_offset = map_offset + 4 * source_count
    poses = {}
    for source_index, core in enumerate(source_to_core):
        x, y, z, qx, qy, qz, qw = struct.unpack_from("<7f", raw, pose_offset + 28 * source_index)
        poses[int(core)] = (np.asarray([x, y, z], dtype=np.float32), np.asarray([qx, qy, qz, qw], dtype=np.float32))
    req(len(poses) == source_count and pose_offset + 28 * source_count == len(raw), "NHRIGID pose layout mismatch")
    return poses, {
        "path": str(path), "sha256": sha(path), "archive_path": str(archive),
        "archive_sha256": sha(archive), "source_count": source_count,
        "engine_body_count": body_count, "joint_count": joint_count,
        "nq": nq, "nv": nv, "header_source_sha256": source_sha.hex(),
    }


def route_field(points_world: np.ndarray, control_xyz: np.ndarray, control_core: np.ndarray, core_ids: list[int]) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    controls = np.asarray(control_xyz, dtype=np.float64)
    core_index = {int(core): index for index, core in enumerate(core_ids)}
    control_index = np.asarray([core_index[int(core)] for core in control_core], dtype=np.int64)
    output = np.zeros((len(points_world), len(core_ids)), dtype=np.float64)
    for point_index, point in enumerate(points_world):
        distance2 = np.sum((controls - point[None, :]) ** 2, axis=1)
        exact = np.flatnonzero(distance2 == 0.0)
        if len(exact):
            node_weights = np.zeros(len(controls), dtype=np.float64)
            node_weights[exact] = 1.0 / len(exact)
        else:
            dmin = float(np.min(distance2))
            raw = np.power(dmin / distance2, 1.0)
            denominator = math.fsum(float(value) for value in raw)
            req(math.isfinite(denominator) and denominator > 0.0, "invalid route Shepard normalization")
            node_weights = raw / denominator
        core_weights = np.zeros(len(core_ids), dtype=np.float64)
        for index, value in enumerate(node_weights):
            core_weights[control_index[index]] += float(value)
        total = math.fsum(float(value) for value in core_weights)
        core_weights /= total
        if len(core_weights) > 1:
            core_weights[-1] = 1.0 - sum(float(value) for value in core_weights[:-1])
        req(np.all(core_weights > 0.0) or bool(len(exact)), "nonanchor route field lost positive support")
        req(abs(sum(float(value) for value in core_weights) - 1.0) <= 1e-12, "route field failed partition of unity")
        output[point_index] = core_weights
    return output


def load_pose_helper(path: Path):
    spec = importlib.util.spec_from_file_location("right_route_fw", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    RES.mkdir()
    progress = {
        "schema": "numi.human.right-fds131-fdp133-route-field-source-audit-095.v1",
        "status": "running_source_validation",
        "scope": "right 131/133 retained partition121, exact source topology/interface audit, and registered source-defined route-field artifact only",
        "gpu_launched": False,
        "native_candidate_emitted": False,
        "input_paths": {},
        "checks": {},
    }
    progress_path = RES / "source-geometry-field-report.json"
    save_json(progress_path, progress)

    for path in (
        *SOURCE_ROWS.values(), *MAPS.values(), CERT / "report.json",
        PART / "report.json", PROJ / "report.json", PROJ / "projection-characterization.json",
        ROUTE, ROUTE_EXTRACTOR, REG, TISS, TISS_MANIFEST, TISS_HELPER, RIGID, ARCHIVE,
    ):
        req(path.is_file(), f"missing required pinned input: {path}")
        progress["input_paths"][str(path)] = sha(path)

    certificate = json.loads((CERT / "report.json").read_text())
    partition_report = json.loads((PART / "report.json").read_text())
    projection_report = json.loads((PROJ / "report.json").read_text())
    projection_characterization = json.loads((PROJ / "projection-characterization.json").read_text())
    req(certificate.get("status") == "exact_partition_certified" and certificate.get("complete"), "retained exact partition certificate is not passing")
    req(certificate["certificate"]["source_face_coverage_exact"] and certificate["certificate"]["source_union_preserved"], "retained partition certificate lacks exact coverage/union")
    req(partition_report.get("status") == "arrangement_constructed" and partition_report.get("admitted") is False, "115 partition report changed or was mislabeled")
    req(projection_report.get("complete") and projection_report.get("admitted") is False, "121 projection report is not complete/provisionally unadmitted")
    req(projection_characterization.get("shared_interface_exact_opposite_chain"), "121 report does not retain exact opposite interface chain")
    for sid in (131, 133):
        row_record = next(row for row in projection_report["rows"] if int(row["sid"]) == sid)
        req(row_record["closed_edge_incidence"] and not row_record["degenerate"] and row_record["self_pairs"] == [], f"121 row-{sid} retained status failed")
        req(row_record["sha256"] == sha(MAPS[sid]), f"121 row-{sid} map hash differs from its report")
        req(certificate["input_sha256"][str(SOURCE_ROWS[sid])] == sha(SOURCE_ROWS[sid]), f"115 row-{sid} hash differs from exact certificate")
    progress["checks"]["retained_115_certificate"] = {
        "status": certificate["status"], "certificate_report_sha256": sha(CERT / "report.json"),
        "certificate_artifact_sha256": certificate["artifact_sha256"],
        "certificate_hash": certificate["canonical_certificate_sha256"],
        "source_intersecting_triangle_pair_count": certificate["construction"]["source_intersecting_triangle_pair_count"],
        "source_face_coverage_exact": certificate["certificate"]["source_face_coverage_exact"],
        "source_union_preserved": certificate["certificate"]["source_union_preserved"],
        "partition_topologies": certificate["certificate"]["partition_topologies"],
        "union_topology": certificate["certificate"]["union_topology"],
    }
    save_json(progress_path, progress)

    sys.path.insert(0, str(CI_ROOT))
    from numilab_human import cardiac_cavity_intersections as ci

    maps = {sid: pickle.loads(MAPS[sid].read_bytes()) for sid in (131, 133)}
    points = {sid: [exact_point(point) for point in maps[sid]["points"]] for sid in (131, 133)}
    faces = {sid: np.asarray(maps[sid]["faces"], dtype=np.int64) for sid in (131, 133)}
    roles = {sid: np.asarray(maps[sid]["roles"], dtype=np.int8) for sid in (131, 133)}
    req(np.array_equal(roles[131], np.asarray(maps[131]["roles"], dtype=np.int8)), "role array conversion failed")
    for sid in (131, 133):
        source_ids = set(map(int, maps[sid]["map_source"]))
        req(source_ids <= {131, 133}, f"row-{sid} map_source contains foreign stable ids")
        req(set(map(int, maps[sid]["parent_ids"])) <= {131, 133}, f"row-{sid} parent ancestry contains foreign ids")
        req(len(maps[sid]["faces"]) == len(maps[sid]["parent_ids"]) == len(maps[sid]["parent_faces"]) == len(roles[sid]), f"row-{sid} face ancestry lengths disagree")
        req(len(maps[sid]["points"]) == len(maps[sid]["wmaps"]) == len(maps[sid]["map_source"]) == len(maps[sid]["map_parent"]), f"row-{sid} vertex ancestry lengths disagree")
        req(np.all(np.asarray(maps[sid]["map_parent"], dtype=np.int64) >= 0), f"row-{sid} has negative map parent")
        parent_ids = np.asarray(maps[sid]["parent_ids"], dtype=np.int64)
        parent_faces = np.asarray(maps[sid]["parent_faces"], dtype=np.int64)
        for source_id in np.unique(parent_ids):
            row_npz = np.load(SOURCE_ROWS[int(source_id)], allow_pickle=False)
            req(np.all(parent_faces[parent_ids == source_id] < len(row_npz["faces"])), f"row-{sid} parent face index escapes stable-{source_id} source row")
        records = ci._records(points[sid], faces[sid].tolist())
        components = ci._validate_closed_oriented_record_topology(records)
        print(json.dumps({"event": "row_topology_checked", "sid": sid, "faces": len(records), "components": components}), flush=True)
        progress["checks"][f"map121_row_{sid}_exact_topology"] = {
            "status": "closed_oriented", "face_count": len(records), "vertex_count": len(points[sid]),
            "face_components": components, "map_source_ids": sorted(source_ids),
            "parent_ids": sorted(set(map(int, parent_ids))),
            "parent_face_bounds_valid": True,
        }
        save_json(progress_path, progress)
        pair_audit = ci._audit_pair(records, records, same_surface=True)
        progress["checks"][f"map121_row_{sid}_exact_self_intersections"] = {
            "triangle_pairs": pair_audit["triangle_pairs"], "count": pair_audit["count"],
            "aabb_candidate_pairs": pair_audit["aabb_candidate_pairs"],
            "allowed_shared_vertex_or_edge_pairs": pair_audit["allowed_shared_vertex_or_edge_pairs"],
        }
        req(pair_audit["count"] == 0, f"exact row-{sid} source self-intersection audit failed")
        print(json.dumps({"event": "row_self_checked", "sid": sid, "count": pair_audit["count"], "aabb_candidates": pair_audit["aabb_candidate_pairs"]}), flush=True)
        save_json(progress_path, progress)

    facet_maps = {}
    for sid in (131, 133):
        face_map = {}
        for index, face in enumerate(faces[sid]):
            key = tuple(sorted(points[sid][int(vertex)] for vertex in face))
            face_map[key] = (index, tuple(points[sid][int(vertex)] for vertex in face))
        facet_maps[sid] = face_map
    shared_facets = set(facet_maps[131]) & set(facet_maps[133])
    shared_vertices = set(points[131]) & set(points[133])
    role_facets = {}
    for sid in (131, 133):
        role_facets[sid] = {
            tuple(sorted(points[sid][int(vertex)] for vertex in faces[sid][face_index]))
            for face_index in np.flatnonzero(roles[sid] == 1)
        }
    req(len(role_facets[131]) == int(np.count_nonzero(roles[131] == 1)), "stable-131 interface contains duplicate coordinate facets")
    req(len(role_facets[133]) == int(np.count_nonzero(roles[133] == 1)), "stable-133 interface contains duplicate coordinate facets")
    req(role_facets[131] == role_facets[133] == shared_facets, "exact shared interface facets differ from role-1 ancestry")
    same_orientation = opposite_orientation = 0
    for key in shared_facets:
        tri131 = facet_maps[131][key][1]
        tri133 = facet_maps[133][key][1]
        rotations = {tuple(tri133[k:] + tri133[:k]) for k in range(3)}
        reversed_rotations = {tuple(tuple(reversed(tri133))[k:] + tuple(reversed(tri133))[:k]) for k in range(3)}
        if tri131 in rotations:
            same_orientation += 1
        elif tri131 in reversed_rotations:
            opposite_orientation += 1
    req(same_orientation == 0 and opposite_orientation == len(shared_facets), "shared interface winding is not exactly reversed")
    progress["checks"]["exact_shared_interface"] = {
        "role1_faces_131": len(role_facets[131]), "role1_faces_133": len(role_facets[133]),
        "exact_opposite_facets": opposite_orientation, "same_winding_facets": same_orientation,
        "common_exact_coordinate_vertices": len(shared_vertices),
        "shared_facet_set_equals_role1_on_both_rows": True,
    }
    save_json(progress_path, progress)

    coordinate_ids = {}
    union_points = []
    def global_id(point):
        if point not in coordinate_ids:
            coordinate_ids[point] = len(union_points)
            union_points.append(point)
        return coordinate_ids[point]

    union_faces = []
    union_face_provenance = []
    for sid in (131, 133):
        for face_index in np.flatnonzero(roles[sid] == 0):
            face = faces[sid][int(face_index)]
            union_faces.append([global_id(points[sid][int(vertex)]) for vertex in face])
            union_face_provenance.append((sid, int(face_index)))
    union_records = ci._records(union_points, union_faces)
    union_components = ci._validate_closed_oriented_record_topology(union_records)
    progress["checks"]["joined_external_boundary_exact_topology"] = {
        "status": "closed_oriented", "face_count": len(union_records),
        "vertex_count": len(union_points), "face_components": union_components,
        "role0_boundary_faces_only": True,
    }
    save_json(progress_path, progress)
    union_audit = ci._audit_pair(union_records, union_records, same_surface=True)
    progress["checks"]["joined_external_boundary_exact_self_intersections"] = {
        "triangle_pairs": union_audit["triangle_pairs"], "count": union_audit["count"],
        "aabb_candidate_pairs": union_audit["aabb_candidate_pairs"],
        "allowed_shared_vertex_or_edge_pairs": union_audit["allowed_shared_vertex_or_edge_pairs"],
    }
    req(union_audit["count"] == 0, "exact joined external boundary self-intersection audit failed")
    print(json.dumps({"event": "union_self_checked", "count": union_audit["count"], "components": union_components}), flush=True)
    save_json(progress_path, progress)

    route_doc = json.loads(ROUTE.read_text())
    req(route_doc.get("schema") == "numi.human.fds-fdp-source-route-provenance-061.v1", "right route source schema mismatch")
    expected_muscles = {"FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5"}
    muscles = [muscle for muscle in route_doc["matching_muscles"] if muscle.get("actuator_name") in expected_muscles]
    req({muscle["actuator_name"] for muscle in muscles} == expected_muscles, "right FDS/FDP authored route set differs")
    occurrences = [(muscle["actuator_name"], node) for muscle in muscles for node in muscle["route_nodes"]]
    deduplicated = {}
    for muscle_name, node in occurrences:
        key = (int(node["source_id"]), str(node["kind"]), int(node["core_body_index"]))
        identity = {name: node[name] for name in (
            "source_id", "name", "kind", "source_body_id", "source_body_name",
            "core_body_index", "position_body_m", "position_world_m",
        )}
        if key in deduplicated:
            req(deduplicated[key]["identity"] == identity, "duplicate route node identity/position disagreement")
            deduplicated[key]["muscles"].append(muscle_name)
        else:
            deduplicated[key] = {"identity": identity, "muscles": [muscle_name]}
    controls = [deduplicated[key]["identity"] for key in sorted(deduplicated)]
    core_ids = sorted({int(node["core_body_index"]) for node in controls})
    req(len(occurrences) == 81 and len(controls) == 74 and len(core_ids) == 21, "right route inventory count changed")
    route_xyz = np.asarray([node["position_world_m"] for node in controls], dtype=np.float64)
    route_core = np.asarray([node["core_body_index"] for node in controls], dtype=np.int64)
    req(len({tuple(row) for row in route_xyz}) == len(route_xyz), "unexpected coincident right control coordinates require explicit collision handling")
    body_names = {}
    for node in controls:
        core = int(node["core_body_index"])
        name = str(node["source_body_name"])
        req(core not in body_names or body_names[core] == name, f"right core {core} maps to multiple route body names")
        body_names[core] = name

    rigid_poses, rigid_meta = parse_nhrigid(RIGID, ARCHIVE)
    req(set(core_ids) <= set(rigid_poses), "NHRIGID lacks a right FDS/FDP route core")
    reg_doc = json.loads(REG.read_text())
    req(sha(REG) == REG_SHA, "recovered registration SHA mismatch")
    G = np.asarray(reg_doc["coordinate_system"]["global_source_mm_to_myosim_world_m"], dtype=np.float64)[:3].copy()
    G[:, :3] *= 1000.0
    source_binding_module = load_pose_helper(TISS_HELPER)
    tissue = source_binding_module.load_tissue(TISS)
    canonical_bindings = {}
    canonical_binding_sources = defaultdict(list)
    for record in tissue["records"]:
        stable_id = int(record[6])
        start, count = int(record[0]), int(record[1])
        for binding in tissue["bindings"][start : start + count]:
            core = int(binding["core"])
            value = np.asarray(binding["value"], dtype=np.float32)
            if core in canonical_bindings:
                req(np.array_equal(canonical_bindings[core], value), f"source TISS core {core} has inconsistent binding records")
            else:
                canonical_bindings[core] = value.copy()
            canonical_binding_sources[core].append(stable_id)
    req(set(core_ids) <= set(canonical_bindings), "current NHTISS lacks right route core binding transforms")

    registration_errors = {}
    for core in core_ids:
        binding = canonical_bindings[core]
        default_position, default_quaternion = rigid_poses[core]
        rotation = np.stack(
            [rotate_double(default_quaternion, rotate_double(binding[3:7], axis) * float(binding[7]))
             for axis in np.eye(3)],
            axis=1,
        )
        translation = default_position.astype(np.float64) + rotate_double(default_quaternion, binding[:3])
        actual = np.column_stack((rotation, translation))
        registration_errors[str(core)] = float(np.max(np.abs(actual - G)))
    max_registration_error = max(registration_errors.values())
    req(max_registration_error < 1e-6, f"current NHTISS registered-body transforms disagree with recovered G: {max_registration_error}")

    anchor_error_um = []
    for node in controls:
        core = int(node["core_body_index"])
        default_position, default_quaternion = rigid_poses[core]
        reconstructed = default_position.astype(np.float64) + rotate_double(default_quaternion, node["position_body_m"])
        authored = np.asarray(node["position_world_m"], dtype=np.float64)
        anchor_error_um.append(float(np.linalg.norm(reconstructed - authored)) * 1e6)

    field_weights = {}
    stored_positions = {}
    row_indices = {}
    common_float32 = set()
    for sid in (131, 133):
        stored_positions[sid] = np.asarray(maps[sid]["points"], dtype=np.float32)
        source_world = stored_positions[sid].astype(np.float64) @ G[:, :3].T + G[:, 3]
        field_weights[sid] = route_field(source_world, route_xyz, route_core, core_ids)
        row_indices[sid] = {tuple(float(value) for value in point): index for index, point in enumerate(stored_positions[sid])}
        req(len(row_indices[sid]) == len(stored_positions[sid]), f"stable-{sid} contains duplicate exact Float32 source coordinates")
    common_float32 = set(row_indices[131]) & set(row_indices[133])
    for point in common_float32:
        i, j = row_indices[131][point], row_indices[133][point]
        req(np.array_equal(field_weights[131][i], field_weights[133][j]), "same registered source coordinate received different route coefficients")
    for sid in (131, 133):
        req(np.isfinite(field_weights[sid]).all(), f"stable-{sid} field has nonfinite coefficient")
        req(float(np.min(field_weights[sid])) > 0, f"stable-{sid} field has a nonpositive route core")
        req(float(np.min(field_weights[sid].astype(np.float32))) > 0, f"stable-{sid} field underflows Float32")

    progress["checks"]["right_authored_route_field"] = {
        "method": "inverse-square interpolatory Shepard weights over 74 deduplicated right FDS2-5/FDP2-5 authored route controls; aggregate all controls by 21 sorted NHRIGID core IDs",
        "control_interpretation": route_doc["interpretation"],
        "muscles": sorted(expected_muscles), "route_occurrences": len(occurrences),
        "unique_controls": len(controls), "core_body_indices": core_ids,
        "core_body_names": {str(core): body_names[core] for core in core_ids},
        "anchor_world_reconstruction_um": {
            "count": len(anchor_error_um), "max": max(anchor_error_um),
            "rms": float(np.sqrt(np.mean(np.square(anchor_error_um)))),
            "p95": float(np.quantile(anchor_error_um, 0.95)),
        },
        "all_source_binding_records_per_core_identical": True,
        "source_binding_record_stable_ids": {str(core): sorted(set(canonical_binding_sources[core])) for core in core_ids},
        "registered_core_transform_max_abs_error_m": max_registration_error,
        "registered_core_transform_errors_m": registration_errors,
        "source_m_to_registered_world": G.tolist(),
        "same_coordinate_weight_rows_bit_identical": True,
        "common_coordinate_count": len(common_float32),
        "minimum_binary64_weight": min(float(np.min(field_weights[sid])) for sid in (131, 133)),
        "minimum_float32_weight": min(float(np.min(field_weights[sid].astype(np.float32))) for sid in (131, 133)),
        "parameter_power": 2.0,
        "sensitivity_tests": "not run; one declared inverse-square method only",
    }

    field_path = RES / "right-route-anchor-reference-field-001.npz"
    np.savez_compressed(
        field_path,
        route_core_body_indices=np.asarray(core_ids, dtype="<u4"),
        route_control_source_ids=np.asarray([node["source_id"] for node in controls], dtype="<u4"),
        route_control_kinds=np.asarray([node["kind"] for node in controls], dtype="U8"),
        route_control_core_body_indices=route_core.astype("<u4"),
        route_control_world_positions_m=route_xyz.astype("<f8"),
        stable131_geometry_vertex_ids=np.asarray([coordinate_ids.setdefault(point, len(coordinate_ids)) for point in []], dtype="<u8"),
        stable131_source_positions_m=stored_positions[131].astype("<f4"),
        stable131_faces=faces[131].astype("<u4"),
        stable131_roles=roles[131].astype("u1"),
        stable131_parent_ids=np.asarray(maps[131]["parent_ids"], dtype="<u4"),
        stable131_parent_faces=np.asarray(maps[131]["parent_faces"], dtype="<u4"),
        stable131_route_core_weights=field_weights[131].astype("<f8"),
        stable133_source_positions_m=stored_positions[133].astype("<f4"),
        stable133_faces=faces[133].astype("<u4"),
        stable133_roles=roles[133].astype("u1"),
        stable133_parent_ids=np.asarray(maps[133]["parent_ids"], dtype="<u4"),
        stable133_parent_faces=np.asarray(maps[133]["parent_faces"], dtype="<u4"),
        stable133_route_core_weights=field_weights[133].astype("<f8"),
    )
    # Add deterministic cross-row coordinate identity separately; this is derived from map121,
    # not an upstream stable asset ID.
    union_keys = sorted({tuple(float(value) for value in row) for sid in (131, 133) for row in stored_positions[sid]})
    geometry_id = {key: index for index, key in enumerate(union_keys)}
    field_tmp = np.load(field_path)
    arrays = {key: field_tmp[key] for key in field_tmp.files if key != "stable131_geometry_vertex_ids"}
    arrays["stable131_geometry_vertex_ids"] = np.asarray([geometry_id[tuple(float(value) for value in row)] for row in stored_positions[131]], dtype="<u8")
    arrays["stable133_geometry_vertex_ids"] = np.asarray([geometry_id[tuple(float(value) for value in row)] for row in stored_positions[133]], dtype="<u8")
    field_tmp.close()
    np.savez_compressed(field_path, **arrays)
    field_sha = sha(field_path)

    progress["status"] = "completed_source_union_and_registered_route_field"
    progress["field_artifact"] = {
        "path": str(field_path), "sha256": field_sha, "size_bytes": field_path.stat().st_size,
        "vertex_counts": {"131": len(stored_positions[131]), "133": len(stored_positions[133])},
        "face_counts": {"131": len(faces[131]), "133": len(faces[133])},
        "geometry_vertex_id_semantics": "deterministic IDs from exact stored Float32 coordinate tuples; not source stable-asset IDs",
    }
    progress["checks"]["joined_external_boundary_exact_topology"]["source_parent_face_count"] = len(union_records)
    progress["checks"]["joined_external_boundary_exact_self_intersections"]["source_parent_face_count"] = len(union_records)
    progress["registered_source_bindings"] = {
        "tissue_path": str(TISS), "tissue_sha256": sha(TISS),
        "manifest_path": str(TISS_MANIFEST), "manifest_sha256": sha(TISS_MANIFEST),
        "tissue_loader_path": str(TISS_HELPER), "tissue_loader_sha256": sha(TISS_HELPER),
        "rigid_reference": rigid_meta,
        "recovered_registration_path": str(REG), "recovered_registration_sha256": sha(REG),
    }
    progress["elapsed_seconds"] = time.monotonic() - START
    progress["limits"] = [
        "right source partition/map validation and field artifact only; no accepted-pose audit yet",
        "route nodes are authored force-path controls, not validated passive tissue attachment landmarks",
        "CPU authoring reference field is not native runtime or full-anatomy admission",
    ]
    save_json(progress_path, progress)
    print(json.dumps({
        "status": progress["status"], "field_path": str(field_path), "field_sha256": field_sha,
        "source_exact_topology": progress["checks"]["joined_external_boundary_exact_topology"],
        "source_exact_self_pairs": progress["checks"]["joined_external_boundary_exact_self_intersections"]["count"],
        "shared_interface": progress["checks"]["exact_shared_interface"],
        "right_route_controls": len(controls), "right_route_cores": core_ids,
        "max_registered_G_error_m": max_registration_error, "elapsed_seconds": progress["elapsed_seconds"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"attempt-095-right-route-field: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

