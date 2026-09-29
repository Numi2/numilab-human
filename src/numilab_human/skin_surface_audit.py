"""Verify the complete native visual skin against source geometry and poses.

The offline oracle independently reconstructs the edge-connected exterior,
source surface seeds and MuJoCo COM frames, then verifies every screened graph
equation and native sparse influence. This admits source visual fidelity only,
never clinical weights, contact or continuum skin mechanics.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import heapq
import io
import json
from pathlib import Path
import struct


def _require(condition, message):
    if not condition:
        raise ValueError("skin source audit: " + message)


def _outer_sheet(vertices, faces):
    """Independent face-edge graph, preserving source vertex/face order."""
    import numpy as np
    parent = list(range(len(faces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = {}
    for i, (a, b, c) in enumerate(faces):
        for x, y in [(a, b), (b, c), (c, a)]:
            edge = tuple(sorted((int(x), int(y))))
            if edge in edges:
                first, second = root(i), root(edges[edge])
                parent[second] = first
            else:
                edges[edge] = i
    components = {}
    for i in range(len(faces)):
        components.setdefault(root(i), []).append(i)
    candidates = []
    for ids in components.values():
        vertex_ids = np.unique(faces[ids])
        bounds = np.array([vertices[vertex_ids].min(axis=0), vertices[vertex_ids].max(axis=0)])
        candidates.append((float(np.prod(bounds[1] - bounds[0])), len(vertex_ids), ids, vertex_ids, bounds))
    candidates.sort(key=lambda row: row[:2], reverse=True)
    _, _, face_ids, vertex_ids, bounds = candidates[0]
    _require(any(len(row[3]) >= .5 * len(vertex_ids) and (row[4][0] >= bounds[0]).all()
                 and (row[4][1] <= bounds[1]).all() for row in candidates[1:]), "nested source sheet missing")
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[vertex_ids] = np.arange(len(vertex_ids))
    return vertices[vertex_ids], remap[faces[face_ids]], {
        "edge_component_count": len(components),
        "source_vertex_ids_sha256": hashlib.sha256(vertex_ids.astype('<u4').tobytes()).hexdigest(),
        "source_face_ids_sha256": hashlib.sha256(np.asarray(face_ids, dtype='<u4').tobytes()).hexdigest(),
    }


def _visual_normals(vertices, faces):
    import numpy as np
    normals = np.zeros_like(vertices)
    crossed = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], crossed)
    missing = np.linalg.norm(normals, axis=1) <= 1e-12
    normals[missing] = vertices[missing] - vertices.mean(axis=0)
    for iteration in range(4):
        lengths = np.linalg.norm(normals, axis=1)
        _require(bool((lengths > 1e-12).all()), "unusable source normals")
        normals /= lengths[:, None]
        if iteration == 3:
            break
        combined = normals[faces].sum(axis=1)
        normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], combined)
    return normals


def _rotation(quaternion):
    import mujoco
    import numpy as np
    q = np.asarray(quaternion, dtype=float)
    _require(q.shape == (4,) and bool(np.isfinite(q).all())
             and abs(float(np.linalg.norm(q)) - 1) < .002, "invalid quaternion")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q[[3, 0, 1, 2]] / np.linalg.norm(q))
    return matrix.reshape(3, 3)


def _edge_deformation_diagnostics(world, faces, native):
    """Measured visual changes; no physiological strain limit is inferred."""
    import numpy as np
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                             faces[:, [2, 0]]]), axis=1), axis=0)
    source_length = np.linalg.norm(world[edges[:, 0]] - world[edges[:, 1]], axis=1)
    current_length = np.linalg.norm(native[edges[:, 0]] - native[edges[:, 1]], axis=1)
    _require(bool((source_length > 0).all()), 'source diagnostic edge metric')
    ratio = current_length / source_length
    increase = current_length - source_length
    worst = np.argsort(-increase, kind='stable')[:5]
    return {'edge_count': len(edges), 'maximum_stretch_ratio': float(ratio.max()),
            'edge_p99_stretch_ratio': float(np.quantile(ratio, .99)),
            'edges_over_5x': int((ratio > 5).sum()),
            'maximum_edge_length_increase_m': float(increase.max()),
            'worst_length_increases': [{'source_skin_vertex_ids': edges[i].tolist(),
                                       'source_length_m': float(source_length[i]),
                                       'native_length_m': float(current_length[i]),
                                       'stretch_ratio': float(ratio[i])} for i in worst],
            'boundary': 'Visual source-edge length diagnostics, not continuum strain, clinical limits or a skin-quality admission gate. Five-times counts are diagnostic only.'}


@lru_cache(maxsize=2)
def _source_oracle(sources, artifact, registration_bytes, rigid_sha, artifact_manifest_sha, skin_obj):
    """Cache immutable, hash-checked source inputs, never emitted skin bytes."""
    import mujoco
    import numpy as np
    from myo_sim.build.compose import build_model
    from . import model as human
    registration = json.loads(registration_bytes)
    reference, owners = human._bodyparts_runtime_bindings(registration, Path(artifact))
    _require(reference["rigid"]["sha256"] == rigid_sha, "source rigid identity")
    model = build_model("myofullbody")
    rest = mujoco.MjData(model)
    mujoco.mj_forward(model, rest)
    # Parsing source OBJ is shared; component selection, weights, normals and
    # world/rest reconstruction below do not invoke the skin compiler.
    raw_v, raw_f = human._bodyparts_obj_triangles(skin_obj, "FJ2810.obj")
    vertices, faces, subset = _outer_sheet(np.asarray(raw_v), np.asarray(raw_f))
    matrix = np.asarray(registration["coordinate_system"]["global_source_mm_to_myosim_world_m"])
    _require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), "source frame")
    common = human._bodyparts_source_common_frame(Path(sources), registration, owners)
    _require(np.max(np.abs(matrix - common["global_source_mm_to_myosim_world_m"])) <= 1e-9,
             "source atlas frame drift")
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    normals = _visual_normals(vertices, faces) @ matrix[:3, :3].T
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    bones = []
    source_ids = {}
    for anchor_index, anchor in enumerate(registration["anchors"]):
        target = anchor["target"]
        core, _ = owners[target["name"]]
        sid = target["source_body_id"]
        source_ids[core] = sid
        source = anchor["source"]
        archive, member, obj = human._bodyparts_obj_member(Path(sources), source["hierarchy"], source["member_id"])
        _require(human.sha256(archive) == source["archive_sha256"] and member == source["member"]
                 and hashlib.sha256(obj).hexdigest() == source["member_sha256"], "bone sample source hash")
        bone_v, _ = human._bodyparts_obj_triangles(obj, member)
        count = min(64, len(bone_v))
        ids = np.arange(count) * (len(bone_v) - 1) // (count - 1) if count > 1 else np.array([0])
        placed = np.asarray(bone_v) @ matrix[:3, :3].T + matrix[:3, 3]
        centroid = placed.mean(axis=0)
        bones.append((anchor_index, core, centroid, ids, placed[ids],
                      float(2 * np.linalg.norm(placed - centroid, axis=1).max())))
    core_ids = sorted(source_ids)
    binding_by_core = {core: i for i, core in enumerate(core_ids)}
    neighbours = [[] for _ in world]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    for a, b in edges:
        length = float(np.linalg.norm(world[a] - world[b]))
        _require(length > 0 and np.isfinite(length), 'source graph edge metric')
        neighbours[a].append((int(b), length))
        neighbours[b].append((int(a), length))
    targets = []
    projection_gaps = []
    for anchor, core, centroid, ids, points, diameter in bones:
        # Direct all-vertex distances are independent of the author's kd tree.
        center = int(np.argmin(np.sum((world - centroid) ** 2, axis=1)))
        radius = diameter + 2 * float(np.linalg.norm(world[center] - centroid))
        reachable = {center: 0.}
        queue = [(0., center)]
        while queue:
            distance, point = heapq.heappop(queue)
            if distance != reachable[point]:
                continue
            for neighbour, length in neighbours[point]:
                candidate = distance + length
                if candidate <= radius and candidate < reachable.get(neighbour, float('inf')):
                    reachable[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        for source_vertex, point in [(0xffffffff, centroid)] + list(zip(ids, points)):
            skin = int(np.argmin(np.sum((world - point) ** 2, axis=1)))
            targets.append((anchor, source_vertex, skin, binding_by_core[core], int(skin in reachable)))
            projection_gaps.append(float(np.linalg.norm(world[skin] - point)))
    return (vertices, faces, normals, world, core_ids, source_ids, subset,
            np.asarray(targets, dtype='<u4'), np.asarray(projection_gaps, dtype='<f8'),
            (len(raw_v), len(raw_f)))


@lru_cache(maxsize=2)
def _verified_weights(proof_bytes, world_bytes, face_bytes, target_bytes, gap_bytes, binding_count, method):
    """Verify every source graph equation directly, without the authoring solver."""
    import numpy as np
    world = np.frombuffer(world_bytes, '<f8').reshape(-1, 3)
    faces = np.frombuffer(face_bytes, '<u4').reshape(-1, 3)
    targets = np.frombuffer(target_bytes, '<u4').reshape(-1, 5)
    from .skin_surface_binding import LEGACY_METHOD, METHOD
    _require(method in {LEGACY_METHOD, METHOD}, 'source binding method')
    geometric = method == METHOD
    gaps = np.frombuffer(gap_bytes, '<f8')
    with np.load(io.BytesIO(proof_bytes), allow_pickle=False) as archive:
        fields = {'full_weights', 'seed_targets'} | ({'seed_projection_gaps_m'} if geometric else set())
        _require(set(archive.files) == fields, 'binding solution fields')
        full = archive['full_weights']
        _require(full.dtype == np.dtype('<f8') and full.shape == (len(world), binding_count)
                 and bool(np.isfinite(full).all()), 'binding solution layout')
        seed_targets = archive['seed_targets']
        _require(seed_targets.dtype == np.dtype('<u4') and seed_targets.shape == targets.shape
                 and np.array_equal(seed_targets, targets), 'source skin seed association')
        if geometric:
            declared = archive['seed_projection_gaps_m']
            _require(declared.dtype == np.dtype('<f8') and declared.shape == gaps.shape
                     and bool(np.isfinite(declared).all()) and bool((declared > 0).all())
                     and float(np.max(np.abs(declared - gaps))) <= 1e-12,
                     'source bone-to-skin projection gaps')
    _require(float(full.min()) >= (0 if geometric else -1e-12)
             and float(np.max(np.abs(full.sum(axis=1) - 1))) <= 1e-10,
             'binding solution positivity/partition')
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    conductance = 1 / np.linalg.norm(world[edges[:, 0]] - world[edges[:, 1]], axis=1)
    degree = np.zeros(len(world))
    np.add.at(degree, edges[:, 0], conductance)
    np.add.at(degree, edges[:, 1], conductance)
    owners = {}
    for (_, _, vertex, body, admitted), gap in zip(targets, gaps):
        if admitted:
            owners.setdefault(int(vertex), {}).setdefault(int(body), []).append(1 / float(gap) if geometric else 1.)
    _require({b for row in owners.values() for b in row} == set(range(binding_count)), 'source seed body coverage')
    flux = np.zeros_like(full)
    denominator = degree[:, None] * np.abs(full)
    for first in range(0, len(edges), 2048):
        edge = edges[first:first + 2048]
        weight = conductance[first:first + 2048, None]
        delta = weight * (full[edge[:, 0]] - full[edge[:, 1]])
        np.add.at(flux, edge[:, 0], delta)
        np.add.at(flux, edge[:, 1], -delta)
        np.add.at(denominator, edge[:, 0], weight * np.abs(full[edge[:, 1]]))
        np.add.at(denominator, edge[:, 1], weight * np.abs(full[edge[:, 0]]))
    for point, body_values in owners.items():
        target = np.zeros(binding_count)
        if geometric:
            values = {body: float(np.mean(weights)) for body, weights in body_values.items()}
            confidence = float(np.mean(list(values.values())))
            for body, value in values.items():
                target[body] = value / sum(values.values())
        else:
            confidence = degree[point]
            target[list(body_values)] = 1 / len(body_values)
        flux[point] += confidence * (full[point] - target)
        denominator[point] += confidence * (np.abs(full[point]) + target)
    residual = float(np.max(np.abs(flux) / np.maximum(denominator, np.finfo(float).tiny)))
    _require(residual <= 1e-10, 'source skin harmonic equations')
    positive = np.maximum(full, 0)
    quartet = np.argsort(-positive, axis=1, kind='stable')[:, :4]
    weights = np.take_along_axis(positive, quartet, axis=1)
    retained = weights.sum(axis=1)
    weights /= retained[:, None]
    quartet.setflags(write=False)
    weights.setflags(write=False)
    full.setflags(write=False)
    return quartet, weights, full, {'passed': True, 'method': method,
                              'maximum_relative_equation_residual': residual,
                              'source_bone_count': len(set(map(int, targets[:, 0]))),
                              'source_graph_edge_count': len(edges),
                              'seed_candidate_count': len(targets), 'seed_vertex_count': len(owners),
                              'rejected_seed_count': int((targets[:, 4] == 0).sum()),
                              'minimum_retained_four_weight_mass': float(retained.min()),
                              'maximum_discarded_weight_mass': float(1 - retained.min()),
                              'minimum_source_projection_gap_m': float(gaps.min()) if geometric else None,
                              'maximum_source_projection_gap_m': float(gaps.max()) if geometric else None,
                              'boundary': 'Independent source seed and full graph-equation certificate; not measured anatomical skin weights or deformation qualification.'}


def audit_skin_surface(sources: Path, artifact: Path, registration_path: Path,
                       payload: Path, native_pack: Path, native_poses: Path, pose=None):
    import mujoco
    import numpy as np
    from myo_sim.build.compose import build_model
    from . import model as human
    from .joint_constraint_consistency import source_equality_projection_oracle
    from .torso_anatomy_audit import _pack_sections
    from .upper_limb_pose_audit import _pose_qpos
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    checks = human._require_myosim_rigid_program(sources, artifact)
    for archive in registration["source"]["bodyparts"]["archives"]:
        _require(human.sha256(sources / archive["file"]) == archive["sha256"], "source archive hash")
    _, member, obj = human._bodyparts_obj_member(sources, "is_a", "FJ2810")
    manifest = json.loads(payload.with_name("bodyparts3d-myosim-skinned-shell.manifest.json").read_text())
    declared_abi = manifest['payload']['payload_abi']
    expected_status = {4: 'native_four_body_source_surface_local_linear_blend_skin_shell_visual_input_not_collision_or_physics',
                       5: 'native_full_body_source_surface_linear_blend_skin_shell_visual_input_not_collision_or_physics'}
    _require(declared_abi in expected_status
             and manifest.get('schema') == f'numi.human.bodyparts3d-myosim-skinned-shell-visual-payload.v{declared_abi}'
             and manifest.get('status') == expected_status[declared_abi],
             'payload type/ownership')
    _require(manifest['source']['bodyparts'] == registration['source']['bodyparts'], 'atlas provenance')
    _require(manifest["source"]["registration"]["sha256"] == hashlib.sha256(registration_bytes).hexdigest(), "registration hash")
    _require(manifest["payload"]["sha256"] == human.sha256(payload), "payload manifest hash")
    _require(manifest["source"]["skin"]["member"] == member
             and manifest["source"]["skin"]["member_sha256"] == hashlib.sha256(obj).hexdigest(), "skin member hash")
    raw = payload.read_bytes()
    magic, abi, nb, nv, ni, fingerprint, source_hash = struct.unpack_from('<8s5I32s', raw)
    _require(magic == b'NHSKIN1\0' and abi == declared_abi
             and len(raw) == 60 + 36 * nb + 56 * nv + 4 * ni + (4 * nv * nb if abi == 5 else 0),
             "payload layout")
    _require(fingerprint == int(hashlib.sha256(registration_bytes).hexdigest()[:8], 16)
             and source_hash.hex() == registration["source"]["myosim"]["source"]["archive_sha256"], "payload source identity")
    bindings_f = np.frombuffer(raw, '<f4', count=9 * nb, offset=60).reshape(nb, 9)
    bindings_u = np.frombuffer(raw, '<u4', count=9 * nb, offset=60).reshape(nb, 9)
    offset = 60 + 36 * nb
    vertices_f = np.frombuffer(raw, '<f4', count=14 * nv, offset=offset).reshape(nv, 14)
    vertices_u = np.frombuffer(raw, '<u4', count=14 * nv, offset=offset).reshape(nv, 14)
    indices = np.frombuffer(raw, '<u4', count=ni, offset=offset + 56 * nv)
    _require(bool(np.isfinite(bindings_f[:, 1:]).all()) and bool(np.isfinite(vertices_f[:, :6]).all())
             and bool(np.isfinite(vertices_f[:, 10:]).all()), "nonfinite payload")
    reference = _source_oracle(str(sources.resolve()), str(artifact.resolve()), registration_bytes,
                               human.sha256(artifact / 'myosim-fullbody-core-reference.nhrigid'),
                               human.sha256(artifact / 'myosim-fullbody-reference.manifest.json'), obj)
    source_v, faces, normals, world, core_ids, source_ids, subset, targets, gaps, raw_counts = reference
    binding = manifest['coverage']['source_surface_binding']
    from .skin_surface_binding import LEGACY_METHOD, METHOD
    _require(binding['method'] == (METHOD if abi == 5 else LEGACY_METHOD), 'source binding method')
    solution = manifest['coverage']['binding_solution']
    _require(Path(solution['file']).name == solution['file'] and solution['runtime_input'] is False,
             'binding solution path/ownership')
    proof = (payload.parent / solution['file']).read_bytes()
    _require(hashlib.sha256(proof).hexdigest() == solution['sha256'], 'binding solution hash')
    quartet, weights, full, certificate = _verified_weights(
        proof, np.asarray(world, dtype='<f8').tobytes(), faces.astype('<u4').tobytes(),
        targets.tobytes(), gaps.tobytes(), nb, binding['method'])
    _require(solution['bytes'] == len(proof) and solution['full_weight_shape'] == [nv, nb],
             'binding solution declared layout')
    for field in ['source_bone_count', 'source_graph_edge_count', 'seed_candidate_count',
                  'seed_vertex_count', 'rejected_seed_count']:
        _require(binding[field] == certificate[field], 'source binding declared counts')
    for field in ['minimum_retained_four_weight_mass', 'maximum_discarded_weight_mass']:
        _require(abs(binding[field] - certificate[field]) <= 1e-12, 'source binding declared truncation')
    if abi == 5:
        for field in ['minimum_source_projection_gap_m', 'maximum_source_projection_gap_m']:
            _require(abs(binding[field] - certificate[field]) <= 1e-12, 'source binding declared projection metric')
        _require(manifest['coverage']['influences_per_vertex'] == nb
                 and manifest['coverage']['diagnostic_influences_per_vertex'] == 4
                 and manifest['coverage']['maximum_runtime_discarded_weight_mass'] == 0,
                 'full runtime weight ownership/coverage')
        native_weights = np.frombuffer(raw, '<f4', count=nv*nb, offset=offset + 56*nv + 4*ni).reshape(nv, nb)
        _require(np.array_equal(native_weights, np.asarray(full, dtype='<f4')),
                 'source skin full influence weights')
    _require((manifest['source']['skin']['source_vertex_count'], manifest['source']['skin']['source_triangle_count']) == raw_counts,
             'source declared counts')
    _require(nv == len(source_v) and ni == faces.size and nb == len(core_ids), "source geometry counts")
    _require(np.array_equal(indices, faces.ravel()), "source skin topology")
    _require(np.array_equal(vertices_f[:, :3], np.asarray(source_v * .001, dtype='<f4')), "source skin coordinates")
    _require(np.array_equal(bindings_u[:, 0], core_ids), "source skin body ownership")
    _require(np.array_equal(vertices_u[:, 6:10], quartet), "source skin influence ownership")
    weight_error = float(np.max(np.abs(vertices_f[:, 10:] - weights)))
    _require(weight_error <= 1e-6, "source skin influence weights")
    source_normal_error = float(np.max(np.linalg.norm(vertices_f[:, 3:6] - normals, axis=1)))
    model = build_model('myofullbody')
    rest, data = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_forward(model, rest)
    data.qpos[:] = model.qpos0 if pose is None else _pose_qpos(model, pose, mujoco, np)[0]
    mujoco.mj_forward(model, data)
    equality = source_equality_projection_oracle(model, data, mujoco, 1e-9) if pose is not None else None
    _require(equality is None or equality['passed'], "source equality oracle")
    snapshot = json.loads(native_poses.read_text())
    _require(snapshot['schema'] == 'numi.human.native-skin-pose-snapshot.v1'
             and snapshot['registration_fingerprint32'] == fingerprint and snapshot['binding_count'] == nb
             and snapshot['vertex_count'] == nv and snapshot.get('payload_abi', 4) == abi, "native pose identity")
    poses = {b['body_index']: b for b in snapshot['bodies']}
    _require(len(poses) == len(snapshot['bodies']) and set(poses) == set(core_ids), "native pose owner coverage")
    expected_world = np.zeros_like(world)
    expected_normal = np.zeros_like(normals)
    body_checks = []
    matrix = np.asarray(registration['coordinate_system']['global_source_mm_to_myosim_world_m'])
    for binding, core in enumerate(core_ids):
        sid = source_ids[core]
        rest_r, current_r = rest.ximat[sid].reshape(3, 3), data.ximat[sid].reshape(3, 3)
        declared_r = _rotation(bindings_f[binding, 4:8]) * bindings_f[binding, 8]
        expected_r = rest_r.T @ (matrix[:3, :3] / .001)
        binding_position_error = float(np.linalg.norm(bindings_f[binding, 1:4] - rest_r.T @ (matrix[:3, 3] - rest.xipos[sid])))
        binding_rotation_error = float(np.max(np.abs(declared_r - expected_r)))
        errors = []
        for name, oracle in [('rest', rest), ('current', data)]:
            native = poses[core][name]
            position = np.asarray(native['position_world_m'])
            _require(position.shape == (3,) and bool(np.isfinite(position).all()), "native pose values")
            errors.append((float(np.linalg.norm(position - oracle.xipos[sid])),
                           float(np.max(np.abs(_rotation(native['orientation_world_xyzw']) - oracle.ximat[sid].reshape(3, 3))))))
        amount = full[:, binding] if abi == 5 else (weights * (quartet == binding)).sum(axis=1)
        rotation = current_r @ rest_r.T
        expected_world += amount[:, None] * ((world - rest.xipos[sid]) @ rotation.T + data.xipos[sid])
        expected_normal += amount[:, None] * (normals @ rotation.T)
        body_checks.append({'core_body_index': core, 'source_body_id': sid,
                            'binding_translation_error_m': binding_position_error,
                            'binding_rotation_scale_error': binding_rotation_error,
                            'rest_position_error_m': errors[0][0], 'rest_orientation_error': errors[0][1],
                            'current_position_error_m': errors[1][0], 'current_orientation_error': errors[1][1],
                            'passed': max(binding_position_error, errors[0][0], errors[1][0]) <= 1e-6
                                      and max(binding_rotation_error, errors[0][1], errors[1][1]) <= 1e-6})
    lengths = np.linalg.norm(expected_normal, axis=1)
    _require(bool((lengths > 1e-6).all()), "source blended normals cancel; explicit fallback oracle required")
    expected_normal /= lengths[:, None]
    sections = _pack_sections(native_pack)
    for kind, stride in [(2, 80), (3, 4), (4, 64), (5, 80)]:
        _require(kind in sections and sections[kind][2] == stride
                 and len(sections[kind][0]) == sections[kind][1] * stride, "native pack layout")
    packed_v = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
    packed_i = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
    selected = primitives[primitives[:, 4] == 51007]
    _require(len(selected) == 1, "native skin surface coverage")
    primitive = selected[0]
    first, count, _, instance = map(int, primitive[:4])
    _require(int(primitive[5]) == 1 and int(primitive[6]) == 0xffffffff
             and count == ni and first + count <= len(packed_i), "native skin semantic/owner")
    native_indices = packed_i[first:first + count]
    base = int(native_indices.min())
    _require(np.array_equal(native_indices - base, faces.ravel()) and base + nv <= len(packed_v), "native skin topology")
    instance_u = np.frombuffer(sections[5][0], '<u4').reshape(-1, 20)
    instance_f = np.frombuffer(sections[5][0], '<f4').reshape(-1, 20)
    _require(instance < len(instance_u) and instance_u[instance, 9] == 0xffffffff and instance_u[instance, 10] == 0
             and np.array_equal(instance_u[instance, 12:16], primitive[4:8])
             and np.array_equal(instance_f[instance, :8], [0, 0, 0, 1, 0, 0, 0, 1]), "native skin world instance")
    native_v = packed_v[base:base + nv]
    _require(bool(np.isfinite(native_v[:, :3]).all()) and bool(np.isfinite(native_v[:, 4:7]).all()), "nonfinite native skin")
    world_error = float(np.max(np.linalg.norm(native_v[:, :3] - expected_world, axis=1)))
    normal_error = float(np.max(np.linalg.norm(native_v[:, 4:7] - expected_normal, axis=1)))
    return {
        'schema': 'numi.human.native-skin-source-audit.v1',
        'passed': all(b['passed'] for b in body_checks) and world_error <= 2e-5
                  and max(source_normal_error, normal_error) <= 2e-5,
        'vertex_count': nv, 'triangle_count': ni // 3, 'binding_count': nb,
        'payload_abi': abi, 'runtime_influences_per_vertex': nb if abi == 5 else 4,
        'runtime_discarded_weight_mass': 0. if abi == 5 else certificate['maximum_discarded_weight_mass'],
        'source_member_sha256': hashlib.sha256(obj).hexdigest(), 'source_subset': subset,
        'source_weight_solution_certificate': certificate, 'source_weight_error': weight_error,
        'source_normal_error': source_normal_error, 'maximum_native_world_vertex_error_m': world_error,
        'maximum_native_normal_error': normal_error, 'body_checks': body_checks,
        'native_surface_edge_diagnostics': _edge_deformation_diagnostics(world, faces, native_v[:, :3]),
        'source_equality_projection_oracle': equality, 'rigid_source_program_checks': checks,
        'inputs': {label: {'path': str(path), 'sha256': human.sha256(path)} for label, path in
                   [('registration', registration_path), ('payload', payload),
                    ('manifest', payload.with_name('bodyparts3d-myosim-skinned-shell.manifest.json')),
                    ('binding_solution', payload.parent / solution['file']),
                    ('native_pack', native_pack), ('native_poses', native_poses)]},
        'boundary': 'Complete selected source exterior and sampled visual blend verification; not clinical skin weights, tissue deformation, thickness, mass, material, contact, self-intersection or whole-Human qualification.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['sources', 'artifact', 'registration', 'payload', 'native-pack', 'native-poses', 'output']:
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--raw-source-rest', action='store_true')
    parser.add_argument('--pose-q', nargs=2, action='append', default=[], metavar=('INDEX', 'VALUE'))
    args = parser.parse_args(argv)
    if args.raw_source_rest and args.pose_q:
        parser.error('raw source rest cannot include pose-q')
    pose = None if args.raw_source_rest else tuple((int(i), float(q)) for i, q in args.pose_q)
    report = audit_skin_surface(args.sources, args.artifact, args.registration, args.payload,
                                args.native_pack, args.native_poses, pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'vertex_count', 'binding_count', 'maximum_native_world_vertex_error_m']}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
