"""Prepare a source-bound supine Human seed and sparse bed witnesses.

The shell remains a visual NHSKIN1 atlas.  NHCNT1 records bind selected shell
vertices to their dominant articulated body as point contacts; they do not turn
the complete shell into collision geometry.  NHINIT2 is emitted only when the
caller supplies the exact runtime source/world fingerprints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .model import ImportError as HumanImportError, sha256, write_json

NHRIGID_HEADER = struct.Struct("<8s10I32s")
NHSKIN_HEADER = struct.Struct("<8s5I32s")
NHCNT_HEADER = struct.Struct("<8s4I32s7f")
NHCNT_RECORD = struct.Struct("<2I10f")
NHCNT_MAGIC = b"NHCNT1\0\0"
NHINIT2_MAGIC = b"NHINIT2\0"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("resting scene: " + message)


def _rotation_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _row_times_rotation_transpose(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a small 3x3 pose without dispatching platform BLAS kernels."""
    return np.sum(points[:, None, :] * rotation[None, :, :], axis=2)


def _rotation_transpose_times_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.sum(rotation * vector[:, None], axis=0)


def _rotation_times_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.sum(rotation * vector[None, :], axis=1)


def _multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    q = np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def load_rigid(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(len(raw) >= 224, "NHRIGID2 header is truncated")
    (magic, abi, engine_abi, source_count, body_count, joint_count, nq, nv,
     root_mode, carrier_count, reserved, source_sha) = NHRIGID_HEADER.unpack_from(raw)
    _require(magic == b"NHRIGID2" and abi == 1 and engine_abi == 5 and
             source_count > 0 and body_count > 0 and joint_count == body_count - 1 and
             nq >= 7 and nv >= 6 and root_mode == 0 and reserved == 0,
             "unsupported or malformed NHRIGID2 dimensions")
    q_offset = 224 + 160 * body_count + 144 * joint_count + 64 * nv
    source_map_offset = q_offset + 4 * (nq + nv)
    pose_offset = source_map_offset + 4 * source_count
    expected_bytes = pose_offset + 28 * source_count
    _require(len(raw) == expected_bytes, "NHRIGID2 source pose extent mismatch")
    q = np.frombuffer(raw, "<f4", count=nq, offset=q_offset).astype(np.float64)
    v = np.frombuffer(raw, "<f4", count=nv, offset=q_offset + 4 * nq).astype(np.float64)
    source_map = np.frombuffer(raw, "<u4", count=source_count, offset=source_map_offset)
    source_poses = np.frombuffer(raw, "<f4", count=source_count * 7,
                                 offset=pose_offset).reshape(source_count, 7)
    core_poses = {int(core): row.astype(np.float64)
                  for core, row in zip(source_map, source_poses, strict=True)}
    # NHRIGID2 body records are 160 bytes; the first mass lane is at +16.
    masses = np.array([struct.unpack_from("<f", raw, 224 + i * 160 + 16)[0]
                       for i in range(body_count)], dtype=np.float64)
    _require(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(masses).all() and
             np.all(masses >= 0) and 0 in core_poses,
             "NHRIGID2 contains invalid default state/mass")
    return {"raw": raw, "sha256": hashlib.sha256(raw).hexdigest(),
            "source_sha256": source_sha.hex(), "source_count": source_count,
            "body_count": body_count, "joint_count": joint_count, "nq": nq,
            "nv": nv, "q": q, "v": v, "core_poses": core_poses,
            "masses": masses, "source_map": source_map.astype(int).tolist(),
            "carrier_count": carrier_count}


def load_skin(path: Path, rigid: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    _require(len(raw) >= 60, "NHSKIN1 header is truncated")
    magic, abi, binding_count, vertex_count, index_count, registration_fp, source_sha = NHSKIN_HEADER.unpack_from(raw)
    _require(magic == b"NHSKIN1\0" and abi == 5 and binding_count > 0 and
             vertex_count > 0 and index_count > 0 and source_sha.hex() == rigid["source_sha256"],
             "NHSKIN1 is not ABI 5 or does not bind to this MyoSim source")
    binding_offset = 60
    vertex_offset = binding_offset + 36 * binding_count
    index_offset = vertex_offset + 56 * vertex_count
    weight_offset = index_offset + 4 * index_count
    expected = weight_offset + 4 * vertex_count * binding_count
    _require(len(raw) == expected, "NHSKIN1 extent does not match its header")
    bindings = []
    for i in range(binding_count):
        body = struct.unpack_from("<I", raw, binding_offset + 36 * i)[0]
        values = struct.unpack_from("<8f", raw, binding_offset + 36 * i + 4)
        translation = np.asarray(values[:3], dtype=np.float64)
        quaternion = np.asarray(values[3:7], dtype=np.float64)
        scale = float(values[7])
        _require(body < rigid["body_count"] and body in rigid["core_poses"] and
                 np.isfinite(translation).all() and np.isfinite(quaternion).all() and
                 abs(np.linalg.norm(quaternion) - 1.0) < 2e-5 and scale > 0,
                 f"NHSKIN1 binding {i} is not source-bound")
        bindings.append((body, translation, quaternion, scale))
    vertex_rows = np.frombuffer(raw, "<f4", count=vertex_count * 14,
                                offset=vertex_offset).reshape(vertex_count, 14)
    vertices = vertex_rows[:, :3].astype(np.float64)
    weights = np.frombuffer(raw, "<f4", count=vertex_count * binding_count,
                            offset=weight_offset).reshape(vertex_count, binding_count).astype(np.float64)
    _require(np.isfinite(vertices).all() and np.isfinite(weights).all() and
             np.all(weights >= 0.0) and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 3e-5,
             "NHSKIN1 vertices/complete weight rows are invalid")
    return {"raw": raw, "sha256": hashlib.sha256(raw).hexdigest(),
            "registration_fingerprint": f"{registration_fp:08x}",
            "binding_count": binding_count, "vertex_count": vertex_count,
            "index_count": index_count, "bindings": bindings,
            "vertices": vertices, "weights": weights}


def source_shell_world(rigid: dict[str, Any], skin: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ABI-5's full-weight source shell at the bound rest pose."""
    vertices = skin["vertices"]
    weights = skin["weights"]
    count = len(vertices)
    world = np.zeros((count, 3), dtype=np.float64)
    dominant_world = np.zeros_like(world)
    binding_weights = np.empty((count, len(skin["bindings"])), dtype=np.float64)
    for index, (body, translation, quaternion, scale) in enumerate(skin["bindings"]):
        body_pose = rigid["core_poses"][body]
        body_position = body_pose[:3]
        body_rotation = _rotation_xyzw(body_pose[3:7])
        bind_rotation = _rotation_xyzw(quaternion)
        local = translation + scale * _row_times_rotation_transpose(vertices, bind_rotation)
        body_world = _row_times_rotation_transpose(local, body_rotation) + body_position
        lane_weight = weights[:, index]
        world += lane_weight[:, None] * body_world
        binding_weights[:, index] = lane_weight
    dominant = np.argmax(binding_weights, axis=1)
    for index, (body, translation, quaternion, scale) in enumerate(skin["bindings"]):
        mask = dominant == index
        if not np.any(mask):
            continue
        pose = rigid["core_poses"][body]
        local = translation + scale * _row_times_rotation_transpose(
            vertices[mask], _rotation_xyzw(quaternion))
        dominant_world[mask] = _row_times_rotation_transpose(
            local, _rotation_xyzw(pose[3:7])) + pose[:3]
    dominant_weights = binding_weights[np.arange(count), dominant]
    dominant_bodies = np.asarray([row[0] for row in skin["bindings"]], dtype=np.uint32)[dominant]
    dominant_error = np.linalg.norm(dominant_world - world, axis=1)
    return world, dominant_bodies, dominant_error


def _region_masks(body_indices: np.ndarray, body_order: list[str]) -> dict[str, np.ndarray]:
    names = [body_order[int(body)].lower() for body in body_indices]
    contains = lambda tokens: np.asarray([any(token in name for token in tokens) for name in names])
    return {
        "head": contains(["head"]) & ~contains(["head_attach"]),
        "thorax": contains(["torso", "chest_r", "cervical_spine", "neck"]),
        "pelvis": contains(["pelvis", "sacrum", "lumbar"]),
        "right_arm": contains(["humerus_r", "ulna_r", "radius_r", "firstmc_r", "secondmc_r",
                                "thirdmc_r", "fourthmc_r", "fifthmc_r", "myoarm_r"]),
        "left_arm": contains(["humerus_l", "ulna_l", "radius_l", "firstmc_l", "secondmc_l",
                               "thirdmc_l", "fourthmc_l", "fifthmc_l", "myoarm_l"]),
        "right_leg": contains(["femur_r", "tibia_r", "calcn_r", "talus_r", "toes_r", "patella_r"]),
        "left_leg": contains(["femur_l", "tibia_l", "calcn_l", "talus_l", "toes_l", "patella_l"]),
    }


def _select_support_vertices(world: np.ndarray, body_indices: np.ndarray,
                             dominant_error: np.ndarray, skin: dict[str, Any],
                             body_order: list[str], rotation: np.ndarray,
                             pivot: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    posed = _row_times_rotation_transpose(world - pivot, rotation) + pivot
    minima = float(posed[:, 2].min())
    # Cover the torso and limbs, then explicitly cover independently moving
    # forearms, palms, fingertips and heels. A ten-point standing-style set
    # allowed unsupported fingers to pass through the bed in the resting run.
    # These are physical NHCNT witnesses; they are not ten HumanIO receptors.
    quotas = {"head": 1, "thorax": 3, "pelvis": 4,
              "right_arm": 1, "left_arm": 1, "right_leg": 1, "left_leg": 1}
    masks = _region_masks(body_indices, body_order)
    chosen: list[dict[str, Any]] = []
    for region, quota in quotas.items():
        candidates = np.flatnonzero(masks[region] & (dominant_error <= 0.01))
        _require(len(candidates) >= quota, f"not enough exact source shell witnesses for {region}")
        candidates = candidates[np.argsort(posed[candidates, 2], kind="stable")]
        region_min = float(posed[candidates[0], 2])
        selected: list[int] = []
        for vertex in candidates:
            if any(np.linalg.norm(posed[vertex] - posed[prior]) < 0.035 for prior in selected):
                continue
            selected.append(int(vertex))
            if len(selected) == quota:
                break
        _require(len(selected) == quota, f"could not separate {region} witnesses on the source surface")
        for vertex in selected:
            chosen.append({"region": region, "vertex_index": vertex,
                           "core_body_index": int(body_indices[vertex]),
                           "dominant_binding_error_m": float(dominant_error[vertex]),
                           "source_shell_gap_from_global_min_m": float(posed[vertex, 2] - minima),
                           "region_minimum_world_z_m": region_min})
    for side in ("r", "l"):
        for owner in ("ulna", "thirdmc", "fifthmc", "distal_thumb", "distph2",
                      "distph3", "distph4", "distph5", "calcn", "tibia"):
            name = f"{owner}_{side}"
            _require(name in body_order, f"missing resting support owner {name}")
            body = body_order.index(name)
            candidates = np.flatnonzero((body_indices == body) & (dominant_error <= 0.01))
            _require(len(candidates) > 0, f"no source skin support witness for {name}")
            used = {row["vertex_index"] for row in chosen}
            candidates = [int(v) for v in candidates[np.argsort(posed[candidates, 2], kind="stable")]
                          if int(v) not in used]
            _require(bool(candidates), f"no distinct support witness for {name}")
            vertex = candidates[0]
            chosen.append({"region": name, "vertex_index": vertex,
                           "core_body_index": body,
                           "dominant_binding_error_m": float(dominant_error[vertex]),
                           "source_shell_gap_from_global_min_m": float(posed[vertex, 2] - minima),
                           "region_minimum_world_z_m": float(posed[vertex, 2])})
    _require(len(chosen) <= 32 and len({row["vertex_index"] for row in chosen}) == len(chosen),
             "support witness count/identity is invalid")
    return chosen, posed


def _local_witnesses(rigid: dict[str, Any], skin: dict[str, Any],
                     world_vertices: np.ndarray, selected: list[dict[str, Any]],
                     delta_rotation: np.ndarray, pivot: np.ndarray,
                     root_shift_z: float) -> list[dict[str, Any]]:
    rows = []
    for row in selected:
        vertex = int(row["vertex_index"])
        body = int(row["core_body_index"])
        body_pose = rigid["core_poses"][body]
        body_rotation = _rotation_xyzw(body_pose[3:7])
        local = _rotation_transpose_times_vector(
            body_rotation, world_vertices[vertex] - body_pose[:3])
        desired_world = _row_times_rotation_transpose(
            (world_vertices[vertex] - pivot)[None, :], delta_rotation)[0] + pivot
        desired_world = desired_world + np.array([0.0, 0.0, root_shift_z])
        # Root motion left this material point in the same body-local frame.
        local_f32 = np.asarray(local, dtype=np.float32).astype(np.float64)
        # Reconstruct the point from the rest-frame local witness and root
        # rigid transform. Its residual is the exact point-to-source-shell fit.
        rest_bound = body_pose[:3] + _rotation_times_vector(body_rotation, local_f32)
        predicted_world = _row_times_rotation_transpose(
            (rest_bound - pivot)[None, :], delta_rotation)[0] + pivot
        predicted_world[2] += root_shift_z
        rows.append({**row, "local_point_m": local_f32.tolist(),
                     "rest_frame_witness_fit_error_m": float(np.linalg.norm(predicted_world - desired_world)),
                     "plane_gap_m": float(desired_world[2]),
                     "world_witness_m": [float(desired_world[0]), float(desired_world[1]), 0.0]})
    return rows


def encode_nhcnt1(rigid: dict[str, Any], rows: list[dict[str, Any]], *, friction: float = 0.55) -> bytes:
    _require(len(rows) <= 32 and len(rows) > 0 and math.isfinite(friction) and friction >= 0,
             "invalid NHCNT1 support rows/friction")
    header = NHCNT_HEADER.pack(NHCNT_MAGIC, 1, rigid["body_count"], len(rows), 0,
                               bytes.fromhex(rigid["source_sha256"]),
                               0.0, 0.0, 0.0, 0.0, 0.0, 1.0, friction)
    records = []
    for row in rows:
        x, y, z = row["local_point_m"]
        wx, wy, wz = row["world_witness_m"]
        geometry_id = int(row["vertex_index"]) + 1
        records.append(NHCNT_RECORD.pack(int(row["core_body_index"]), geometry_id,
                                         x, y, z, wx, wy, wz, friction,
                                         0.0, 0.0, 0.0))
    return header + b"".join(records)


def _fnv_bytes(value: int, data: bytes) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xffffffffffffffff
    return value


def _fnv_u64(value: int, number: int) -> int:
    return _fnv_bytes(value, struct.pack("<Q", number))


def base_runtime_source_fingerprint(rigid_path: Path, muscle_path: Path,
                                    support_path: Path,
                                    equality_path: Path | None = None,
                                    limits_path: Path | None = None) -> int:
    """Mirror NumanX's existing base and optional NHEQ2/NHLIM1 owner hashes."""
    raw = [rigid_path.read_bytes(), muscle_path.read_bytes(), support_path.read_bytes()]
    h = _fnv_bytes(FNV_OFFSET, b"mrnx.fullbody.source.v1")
    for payload in raw:
        h = _fnv_u64(h, len(payload))
        h = _fnv_bytes(h, payload)
    for domain, path in (("NHEQ2", equality_path), ("NHLIM1", limits_path)):
        if path is None:
            continue
        payload = path.read_bytes()
        ph = _fnv_bytes(FNV_OFFSET, payload)
        if ph == 0:
            ph = FNV_OFFSET
        h ^= _fnv_bytes(FNV_OFFSET, domain.encode("ascii"))
        h = (h * FNV_PRIME) & 0xffffffffffffffff
        h ^= ph
        h = (h * FNV_PRIME) & 0xffffffffffffffff
    return h or FNV_OFFSET


def encode_nhinit2(rigid: dict[str, Any], muscle_path: Path, q: np.ndarray,
                   *, human_source_fingerprint: int, world_fingerprint: int,
                   timestep_ns: int) -> bytes:
    _require(human_source_fingerprint > 0 and world_fingerprint > 0 and
             0 < timestep_ns <= 1_000_000_000,
             "NHINIT2 needs exact nonzero source/world fingerprints and a valid nanosecond step")
    muscle_raw = muscle_path.read_bytes()
    _require(len(muscle_raw) >= 80, "NHMYO2 header is truncated")
    (magic, abi, body_count, muscle_count, site_count, wrap_count, route_count,
     tendon_count, reserved0, reserved1, source_sha) = struct.unpack_from("<8s10I32s", muscle_raw)
    _require(magic == b"NHMYO2\0\0" and abi == 2 and body_count == rigid["body_count"] and
             muscle_count > 0 and source_sha.hex() == rigid["source_sha256"] and
             reserved0 == muscle_count and reserved1 == 32,
             "NHMYO2 does not exactly bind to this NHRIGID2")
    architecture_offset = 80 + 16 * site_count + 64 * wrap_count + 16 * route_count + 164 * muscle_count
    _require(len(muscle_raw) == architecture_offset + 32 * muscle_count,
             "NHMYO2 architecture extent mismatch")
    architectures = np.frombuffer(muscle_raw, "<f4", count=muscle_count * 8,
                                  offset=architecture_offset).reshape(muscle_count, 8)
    optimal = architectures[:, 0]
    _require(np.isfinite(optimal).all() and np.all(optimal > 0),
             "NHMYO2 has no positive source-bound optimal fibre lengths")
    _require(len(q) == rigid["nq"] and np.isfinite(q).all(), "NHINIT2 q dimension/state mismatch")
    q32 = np.asarray(q, dtype="<f4")
    v32 = np.zeros(rigid["nv"], dtype="<f4")
    muscles = np.column_stack([np.zeros(muscle_count), np.zeros(muscle_count), optimal, np.zeros(muscle_count)])
    muscles32 = np.asarray(muscles, dtype="<f4")
    header = b"".join([
        NHINIT2_MAGIC,
        struct.pack("<8I", 2, 160, rigid["nq"], rigid["nv"], muscle_count, 4, 0, 0),
        struct.pack("<3Q", human_source_fingerprint, world_fingerprint, 0),
        bytes.fromhex(rigid["source_sha256"]),
        struct.pack("<Q", timestep_ns),
        bytes(64),
    ])
    _require(len(header) == 160, "internal NHINIT2 header extent mismatch")
    return header + q32.tobytes() + v32.tobytes() + muscles32.tobytes()


def prepare(rigid_path: Path, muscle_path: Path, skin_path: Path,
            body_order_path: Path, output: Path, *, pitch_degrees: float = -89.0,
            friction: float = 1.0,
            human_source_fingerprint: int | None = None,
            world_fingerprint: int | None = None,
            timestep_ns: int = 12_500) -> dict[str, Any]:
    rigid = load_rigid(rigid_path)
    skin = load_skin(skin_path, rigid)
    body_order = json.loads(body_order_path.read_text(encoding="utf-8"))["core_tree"]["body_order"]
    _require(len(body_order) == rigid["body_count"], "manifest body order does not match rigid payload")
    world, dominant_bodies, dominant_error = source_shell_world(rigid, skin)
    angle = math.radians(pitch_degrees)
    delta_q = np.asarray([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)], dtype=np.float64)
    delta_rotation = _rotation_xyzw(delta_q)
    # The source torso's +X axis points toward the registered sternum. Check
    # the actual transformed direction: +91 degrees is prone in this source,
    # while -89 degrees is supine. Do not infer posture from a filename.
    torso = body_order.index("torso")
    anterior_world = _rotation_times_vector(delta_rotation, _rotation_times_vector(
        _rotation_xyzw(rigid["core_poses"][torso][3:7]), np.array([1., 0., 0.])))
    posture = "supine" if anterior_world[2] > .8 else "prone" if anterior_world[2] < -.8 else "side_lying"
    pivot = rigid["q"][:3].astype(np.float64)
    selected, posed = _select_support_vertices(world, dominant_bodies, dominant_error,
                                               skin, body_order, delta_rotation, pivot)
    # Put the lowest source-shell vertex just above the bed in the exact FP32
    # root state. No vertex is pushed through the bed to create extra support.
    clearance = 1.0e-5
    root_shift_z = -float(posed[:, 2].min()) + clearance
    q = rigid["q"].copy()
    q[0:3] += np.asarray([0.0, 0.0, root_shift_z])
    q[3:7] = _multiply_xyzw(delta_q, q[3:7])
    q = np.asarray([_f32(value) for value in q], dtype=np.float64)
    root_shift_z = float(q[2] - rigid["q"][2])
    actual = _row_times_rotation_transpose(world - pivot, delta_rotation) + pivot
    actual[:, 2] += root_shift_z
    minimum_gap = float(actual[:, 2].min())
    _require(minimum_gap >= -2e-6, f"supine source skin intersects bed plane by {-minimum_gap:.6g} m")
    support_rows = _local_witnesses(rigid, skin, world, selected, delta_rotation,
                                    pivot, root_shift_z)
    contact_raw = encode_nhcnt1(rigid, support_rows, friction=friction)
    output.mkdir(parents=True, exist_ok=True)
    contact_path = output / "myosim-fullbody-resting-bed-support.nhcnt"
    contact_path.write_bytes(contact_raw)
    init_path = None
    if human_source_fingerprint is not None or world_fingerprint is not None:
        _require(human_source_fingerprint is not None and world_fingerprint is not None,
                 "NHINIT2 requires both runtime fingerprints together")
        initial = encode_nhinit2(rigid, muscle_path, q,
                                 human_source_fingerprint=human_source_fingerprint,
                                 world_fingerprint=world_fingerprint,
                                 timestep_ns=timestep_ns)
        init_path = output / "myosim-fullbody-resting-supine.nhinit"
        init_path.write_bytes(initial)
    total_mass = float(rigid["masses"].sum())
    manifest = {
        "schema": "numi.human.resting-supine-source-scene.v1",
        "status": "prepared_source_skin_witnesses_not_mechanical_equilibrium",
        "pose": {"kind": posture, "source_torso_anterior_world": anterior_world.tolist(),
                 "root_delta_quaternion_xyzw": delta_q.tolist(),
                 "root_pitch_degrees": pitch_degrees, "root_translation_xyz_m": q[:3].tolist(),
                 "initial_generalized_velocity": "zero"},
        "bed": {"plane_point_m": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                "friction": friction, "minimum_source_skin_gap_m": minimum_gap,
                "maximum_bed_penetration_m": max(0.0, -minimum_gap),
                "support_witness_count": len(support_rows),
                "support_witnesses": support_rows,
                "contact_scope": "NHCNT1 point witnesses bound to dominant articulated bodies; the NHSKIN1 shell itself is not collision geometry"},
        "source": {"rigid": {"path": str(rigid_path.resolve()), "sha256": rigid["sha256"],
                              "source_archive_sha256": rigid["source_sha256"],
                              "body_count": rigid["body_count"], "source_body_count": rigid["source_count"],
                              "joint_count": rigid["joint_count"], "nq": rigid["nq"], "nv": rigid["nv"],
                              "zero_inertia_serial_transform_carriers": rigid["carrier_count"],
                              "source_rigid_body_mass_kg": total_mass},
                   "skin": {"path": str(skin_path.resolve()), "sha256": skin["sha256"],
                            "registration_fingerprint32": skin["registration_fingerprint"],
                            "binding_count": skin["binding_count"], "vertex_count": skin["vertex_count"],
                            "triangle_count": skin["index_count"] // 3,
                            "max_dominant_binding_witness_error_m": max(row["dominant_binding_error_m"] for row in support_rows),
                            "meaning": "distance between exact full-weight ABI-5 rest vertex and its single dominant body-bound source witness"}},
        "outputs": {"support_contact": {"path": str(contact_path.resolve()),
                                        "sha256": hashlib.sha256(contact_raw).hexdigest(),
                                        "bytes": len(contact_raw), "abi": 1,
                                        "record_count": len(support_rows)},
                    "initial_state": None if init_path is None else {
                        "path": str(init_path.resolve()), "sha256": sha256(init_path),
                        "bytes": init_path.stat().st_size, "abi": 2,
                        "human_source_fingerprint": f"0x{human_source_fingerprint:016x}",
                        "world_fingerprint": f"0x{world_fingerprint:016x}",
                        "timestep_nanoseconds": timestep_ns}},
        "qualification": {"source_skin_reconstruction": True,
                          "bed_nonpenetration_of_visual_shell_at_seed": minimum_gap >= -2e-6,
                          "dynamic_contact": False, "force_balance": False,
                          "sustained_resting": False, "clinical_registration": False},
        "license_and_provenance": {"MyoSim": "Apache-2.0; archive SHA bound in NHRIGID2",
                                    "BodyParts3D": "CC-BY-4.0; FJ2810 exterior shell, current NHSKIN1 manifest contains source asset identities",
                                    "mixed_source_reference": True},
    }
    receipt = output / "resting-supine-scene.manifest.json"
    write_json(receipt, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--muscle", type=Path, required=True)
    parser.add_argument("--skin", type=Path, required=True)
    parser.add_argument("--body-order-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pitch-degrees", type=float, default=-89.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--human-source-fingerprint", type=lambda x: int(x, 0))
    parser.add_argument("--world-fingerprint", type=lambda x: int(x, 0))
    parser.add_argument("--timestep-ns", type=int, default=12_500)
    args = parser.parse_args()
    result = prepare(args.rigid, args.muscle, args.skin, args.body_order_manifest,
                     args.output, pitch_degrees=args.pitch_degrees,
                     friction=args.friction,
                     human_source_fingerprint=args.human_source_fingerprint,
                     world_fingerprint=args.world_fingerprint,
                     timestep_ns=args.timestep_ns)
    print(json.dumps({"status": result["status"], "support_contact": result["outputs"]["support_contact"],
                      "initial_state": result["outputs"]["initial_state"],
                      "minimum_gap_m": result["bed"]["minimum_source_skin_gap_m"],
                      "contact_regions": [row["region"] for row in result["bed"]["support_witnesses"]]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
