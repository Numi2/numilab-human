#!/usr/bin/env python3
"""Bounded EPL143-only NHSKIN position candidate and exact 8-pose differential audit."""
from __future__ import annotations
import collections, hashlib, importlib.util, json, math, pathlib, struct, sys, time
import sys as _sys
import numpy as np

E = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
EVIDENCE_ROOT = E / "native-skin-epl143-clearance-1187"
OUT = EVIDENCE_ROOT / "attempt-006"
BASE_AUDIT = E / "native-lung-late-skin-audit-runner-1172/closed-baseline-1170/full-attempt-002"
SCENE = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-baseline/output/scene")
SKIN = E / "native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_MANIFEST = SKIN.with_name("common-atlas-skin-geometry-registration.manifest.json")
DECL = BASE_AUDIT / "declaration.json"
NHA = E / "native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy"
SUPPORT = E / "final-native-scene-preflight-936/skin-927-lung-1159-viewer-018-v015-attempt1/resting-scene/myosim-fullbody-resting-bed-support.nhcnt"
TRANSFORM_SCRIPT = E / "native-vastus-lateralis-weight-row-erratum-1182/reproduce.py"
CLEARANCE_MODULE = pathlib.Path("/Users/n/numi-human-resting-final-integration-001/src/numilab_human/common_atlas_skin_clearance.py")
PREDICATE_MODULE = CLEARANCE_MODULE.with_name("cardiac_cavity_intersections.py")
GEOMETRY_OWNER = pathlib.Path("/Users/n/numi-human-lung-source-publication-902/src/numilab_human/common_atlas_skin_geometry_registration.py")
POSE_STEPS = (47519, 49151, 51903, 54047, 55647, 152191, 154143, 155000)
SEED_FACES = (1619, 1635)
SEED_VERTICES = (777, 778, 796, 803)
TARGET_KEY = (51005, 143)
WORLD_CAP_MM = 2.0
SCREEN_AMPLITUDES_MM = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
RADIUS_EDGE_MULTIPLE = 2.0
EXPECTED_SKIN_SHA = "bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1"
EXPECTED_NHA_SHA = "c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713"
EXPECTED_CLEARANCE_SHA = "1321c31c22e1b1dbf0062947c6767c2cfcae9aae35d29e9ea775862feed5ddd1"
EXPECTED_PREDICATE_SHA = "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb"
EXPECTED_GEOMETRY_OWNER_SHA = "61857a0edfc098a941f2923113ebbc2ecafbb3c3aafee3a7ff4a83ba28ffd1c0"
EXPECTED_927_GEOMETRY_BUILDER_SHA = "a0bd2271cdad2e13472b1770fab33469589bba1193a579e321de20f89752eb52"
EXPECTED_TRANSFORM_SHA = "9fad92a4acaf51b87e1e0a3cabd578f18e80646ce92c481e13a9fdca7dc8b03f"

sys.path.insert(0, "/Users/n/numi-human-resting-final-integration-001/src")
from numilab_human import common_atlas_skin_clearance as clearance
from numilab_human import cardiac_cavity_intersections as exact


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

transform_owner = import_file("epl143_transform_owner_1182", TRANSFORM_SCRIPT)
_geometry_name = "numilab_human.common_atlas_skin_geometry_registration"
_geometry_spec = importlib.util.spec_from_file_location(_geometry_name, GEOMETRY_OWNER)
geometry_owner = importlib.util.module_from_spec(_geometry_spec)
_sys.modules[_geometry_name] = geometry_owner
_geometry_spec.loader.exec_module(geometry_owner)


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_pins():
    d = json.loads(DECL.read_text())
    require(sha(SKIN) == EXPECTED_SKIN_SHA and d["NHSKIN"]["sha256"] == EXPECTED_SKIN_SHA, "NHSKIN927 changed")
    require(sha(NHA) == EXPECTED_NHA_SHA and d["NHA"]["sha256"] == EXPECTED_NHA_SHA, "NHA1159 changed")
    require(sha(CLEARANCE_MODULE) == EXPECTED_CLEARANCE_SHA, "common_atlas_skin_clearance source changed")
    require(d["loaded_exact_predicate"]["sha256"] == EXPECTED_PREDICATE_SHA and sha(PREDICATE_MODULE) == EXPECTED_PREDICATE_SHA, "exact predicate source changed")
    require(sha(GEOMETRY_OWNER) == EXPECTED_GEOMETRY_OWNER_SHA, "commongeometry owner source changed")
    require(sha(TRANSFORM_SCRIPT) == EXPECTED_TRANSFORM_SHA, "retained 1182 transform helper changed")
    require(sha(SKIN_MANIFEST) == d["NHSKIN"]["registration_manifest_sha256"], "skin registration manifest changed")
    skin_manifest = json.loads(SKIN_MANIFEST.read_text())
    require(skin_manifest["code"]["module_sha256"] == EXPECTED_927_GEOMETRY_BUILDER_SHA,
            "927 source registration builder provenance changed")
    require(d["NHA"]["path"] == str(NHA), "NHA path differs from baseline declaration")
    return d


def load_baseline(step, decl):
    pack = SCENE / "accepted-geometry" / f"step-{step}.mrvpack"
    receipt = SCENE / "accepted-geometry" / f"step-{step}.receipt.json"
    expected = decl["source_hashes_at_preflight"]
    require(sha(pack) == expected[str(pack)], f"pack hash mismatch at {step}")
    require(sha(receipt) == expected[str(receipt)], f"receipt hash mismatch at {step}")
    result_path = BASE_AUDIT / f"step-{step}.result.json"
    result = json.loads(result_path.read_text())
    require(result["accepted_step"] == step and result["pair_coverage_complete"], f"baseline audit incomplete at {step}")
    require(result["skin_self_crossing_pair_count"] == 0 and not result["skin_degenerate_face_rows"], f"baseline skin self/degenerate issue at {step}")
    require(result["invalid_target_surface_count"] == 0 and result["invalid_target_triangle_count"] == 0, f"baseline invalid targets at {step}")
    require(result["surface_target_count"] == 859, f"baseline target count mismatch at {step}")
    expected_witness_fields = {
        "crossing-witnesses": "crossing_witnesses_sha256",
        "self-witnesses": "self_witnesses_sha256",
        "invalid-triangles": "invalid_triangles_sha256",
        "targets": "targets_sha256",
    }
    for suffix, field in expected_witness_fields.items():
        f = BASE_AUDIT / f"step-{step}.{suffix}.jsonl"
        require(sha(f) == result[field], f"baseline {suffix} hash mismatch at {step}")
    return pack, receipt, result


def target_keys_at_terminal():
    rows = [json.loads(x) for x in (BASE_AUDIT / "step-155000.targets.jsonl").read_text().splitlines() if x]
    keys = {tuple(map(int, r["surface"])) for r in rows}
    require(len(rows) == 859 and len(keys) == 859 and all(r["pair_coverage_complete"] and not r["degenerate_face_rows"] for r in rows), "baseline 859-target inventory invalid")
    for step in POSE_STEPS:
        other = [json.loads(x) for x in (BASE_AUDIT / f"step-{step}.targets.jsonl").read_text().splitlines() if x]
        require({tuple(map(int, r["surface"])) for r in other} == keys and len(other) == 859, f"target set differs at {step}")
    return keys


def records_for_rows(positions, faces, rows):
    subset = np.asarray(faces, dtype=np.int64)[np.asarray(rows, dtype=np.int64)]
    records = clearance._exact_surface_records(positions, subset)
    return [(r[0], r[1], r[2], int(rows[i]), r[4]) for i, r in enumerate(records)]


def pair_set(audit):
    return {tuple(map(int, p)) for p in audit["triangle_pairs"]}


def read_witness_pairs(step):
    out = collections.defaultdict(set)
    f = BASE_AUDIT / f"step-{step}.crossing-witnesses.jsonl"
    for line in f.open():
        if line.strip():
            r = json.loads(line)
            key = tuple(map(int, r["target_surface"]))
            out[key].add((int(r["skin_source_face_row"]), int(r["target_surface_face_row"])))
    return out


def build_graph(faces, positions):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    xyz = positions.astype(np.float64)
    adjacency = [[] for _ in range(len(xyz))]
    for a, b in edges:
        dist = float(np.linalg.norm(xyz[int(a)] - xyz[int(b)]))
        adjacency[int(a)].append((int(b), dist))
        adjacency[int(b)].append((int(a), dist))
    return adjacency, edges


def geodesic_from_seeds(adjacency, seeds, radius):
    import heapq
    distances = {int(s): 0.0 for s in seeds}
    queue = [(0.0, int(s)) for s in seeds]
    heapq.heapify(queue)
    while queue:
        dist, vertex = heapq.heappop(queue)
        if dist != distances.get(vertex):
            continue
        if dist > radius:
            break
        for neighbor, length in adjacency[vertex]:
            new = dist + length
            if new <= radius and new < distances.get(neighbor, float("inf")):
                distances[neighbor] = new
                heapq.heappush(queue, (new, neighbor))
    return distances


def smoothstep_complement(t):
    t = min(1.0, max(0.0, float(t)))
    return 1.0 - (6 * t**5 - 15 * t**4 + 10 * t**3)


def source_direction_runtime_gain(skin, vertex_id, direction, body_pose):
    """Exact directional derivative of the retained full-weight LBS map."""
    weights = struct.unpack_from(
        f"<{skin['binding_count']}f", skin["raw"],
        skin["full_offset"] + int(vertex_id) * skin["binding_count"] * 4)
    result = (0.0, 0.0, 0.0)
    for index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        binding = skin["bindings"][index]
        body = int(binding[0])
        q_bind = tuple(float(x) for x in binding[4:8])
        q_body = body_pose[body][1]
        local = transform_owner.qrotate(q_bind, tuple(float(x) for x in direction))
        local = tuple(float(binding[8]) * x for x in local)
        world = transform_owner.qrotate(q_body, local)
        result = tuple(result[i] + float(weight) * float(world[i]) for i in range(3))
    return float(np.linalg.norm(np.asarray(result, dtype=np.float64)))


def source_candidate_skin(base_skin, field, source_amp):
    out = dict(base_skin)
    rows = list(base_skin["vertices"])
    normals = np.asarray([v[3:6] for v in rows], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    require(np.all(lengths > 1.0e-10), "invalid NHSKIN source normal")
    normals /= lengths[:, None]
    positions = {}
    for vertex, weight in field.items():
        base = np.asarray(rows[vertex][:3], dtype=np.float64)
        p32 = np.asarray(base + normals[vertex] * (source_amp * weight), dtype="<f4").astype(np.float64)
        if not np.array_equal(p32, np.asarray(rows[vertex][:3], dtype="<f4").astype(np.float64)):
            row = list(rows[vertex])
            row[:3] = p32.tolist()
            rows[vertex] = tuple(row)
            positions[int(vertex)] = p32
    out["vertices"] = rows
    return out, positions


def world_positions_for_candidate(pack_positions, base_skin, candidate_skin, changed_vertices, body_pose, pack_vertex_base):
    out = pack_positions.copy()
    max_base_residual = 0.0
    max_world_delta = 0.0
    for vertex in changed_vertices:
        base_pred, _ = transform_owner.skin_point(base_skin, int(vertex), body_pose)
        candidate_pred, _ = transform_owner.skin_point(candidate_skin, int(vertex), body_pose)
        captured = pack_positions[pack_vertex_base + int(vertex)].astype(np.float64)
        residual = float(np.linalg.norm(np.asarray(base_pred) - captured))
        max_base_residual = max(max_base_residual, residual)
        delta = np.asarray(candidate_pred, dtype=np.float64) - np.asarray(base_pred, dtype=np.float64)
        candidate = (captured + delta).astype("<f4").astype(np.float64)
        world_delta = float(np.linalg.norm(candidate - captured))
        max_world_delta = max(max_world_delta, world_delta)
        out[pack_vertex_base + int(vertex)] = candidate
    return out, max_base_residual, max_world_delta


def face_quality(base_pos, candidate_pos, faces, rows):
    tri_ids = np.asarray(faces, dtype=np.int64)[rows]
    old = base_pos[tri_ids]
    new = candidate_pos[tri_ids]
    old_n = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    new_n = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    old_a = np.linalg.norm(old_n, axis=1)
    new_a = np.linalg.norm(new_n, axis=1)
    alignment = np.einsum("ij,ij->i", old_n, new_n)
    return {"changed_face_count": int(len(rows)), "minimum_candidate_area_mm2": float(np.min(new_a) * 0.5e6),
            "minimum_area_ratio": float(np.min(new_a / old_a)), "minimum_orientation_dot": float(np.min(alignment)),
            "degenerate_count": int(np.sum(new_a == 0.0)), "reversed_count": int(np.sum(alignment <= 0.0))}


def local_self_audit(candidate_positions, skin_faces, changed_rows):
    all_records = clearance._exact_surface_records(candidate_positions, skin_faces)
    changed_records = [all_records[int(i)] for i in changed_rows]
    seen, rejected = set(), []
    candidates = allowed = 0
    for first, second in exact._aabb_candidate_pairs(changed_records, all_records, same_surface=False):
        i, j = int(first[3]), int(second[3])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        candidates += 1
        points = exact.triangle_intersection_points(first[0], second[0])
        if not points:
            continue
        shared = set(first[4]) & set(second[4])
        common = {first[0][first[4].index(vertex)] for vertex in shared}
        if len(shared) in (1, 2) and all(exact._allowed_shared_point(p, common) for p in points):
            allowed += 1
        else:
            rejected.append({"skin_face_rows": list(key), "shared_vertex_count": len(shared), "intersection_points": len(points)})
    return {"changed_face_count": len(changed_rows), "broadphase_face_pair_count": candidates,
            "allowed_shared_simplex_pair_count": allowed, "unallowed_pair_count": len(rejected), "unallowed_pairs": rejected}


def scan_changed_targets(candidate_positions, skin_faces, changed_rows, target_faces):
    records = records_for_rows(candidate_positions, skin_faces, changed_rows)
    return clearance._target_intersection_audit(records, target_faces, candidate_positions)


def pair_maps_from_scan(scan):
    return {tuple(map(int, key.split(":"))): pair_set(row) for key, row in scan.items()}


def pack_vertex_offset(pack_skin_faces, source_faces):
    require(pack_skin_faces.shape == source_faces.shape, "pack skin face shape changed")
    offsets = pack_skin_faces.astype(np.int64) - source_faces.astype(np.int64)
    require(np.all(offsets == offsets.flat[0]), "MRVPACK skin indices are not a constant source-vertex offset")
    return int(offsets.flat[0])


def load_pack_for_step(step, decl, target_keys):
    pack, receipt, result = load_baseline(step, decl)
    positions, surfaces, meta = clearance._pack_surfaces(pack, target_keys)
    skin_faces = np.asarray(surfaces[(51007, 1)]["faces"], dtype=np.int64)
    target_faces = {key: np.asarray(surfaces[key]["faces"], dtype=np.int64) for key in target_keys}
    pose, receipt_doc = transform_owner.pose_map(receipt)
    require(len(pose) == 86 and int(receipt_doc["accepted_step"]) == step, f"receipt has no exact 86-owner pose at {step}")
    return pack, receipt, positions, skin_faces, target_faces, pose, receipt_doc


def main():
    start = time.monotonic()
    require(not OUT.exists(), "refusing to overwrite existing E1187 evidence")
    OUT.mkdir(parents=True, exist_ok=False)
    declaration = load_pins()
    manifest = json.loads(SKIN_MANIFEST.read_text())
    global_matrix = np.asarray(manifest["common_atlas_binding_runtime_rest_validation"]["global_source_mm_to_world_m"], dtype=np.float64)
    require(global_matrix.shape == (4, 4), "invalid registered source transform")
    scale_m_per_unit = float(np.linalg.norm(global_matrix[:3, 0]))
    require(scale_m_per_unit > 0 and np.allclose(global_matrix[:3, :3].T @ global_matrix[:3, :3], np.eye(3) * scale_m_per_unit**2, atol=1e-12), "source transform is not uniform-scale/rotation")

    base_skin = transform_owner.parse_skin(SKIN)
    source_positions = np.asarray([v[:3] for v in base_skin["vertices"]], dtype=np.float64)
    source_normals = np.asarray([v[3:6] for v in base_skin["vertices"]], dtype=np.float64)
    source_normals /= np.linalg.norm(source_normals, axis=1)[:, None]
    source_faces = np.asarray(base_skin["indices"], dtype=np.int64).reshape(-1, 3)
    require(len(source_faces) == 109211 and base_skin["binding_count"] == 86, "unexpected NHSKIN identity")
    require(set(map(int, np.unique(source_faces[list(SEED_FACES)]))) == set(SEED_VERTICES), "EPL143 seed face vertex set changed")

    adjacency, edges = build_graph(source_faces, source_positions)
    seed_edges = np.any(np.isin(edges, SEED_VERTICES), axis=1)
    seed_edge_lengths = np.linalg.norm(source_positions[edges[seed_edges, 0]] - source_positions[edges[seed_edges, 1]], axis=1)
    median_edge = float(np.median(seed_edge_lengths))
    support_radius = RADIUS_EDGE_MULTIPLE * median_edge
    distances = geodesic_from_seeds(adjacency, SEED_VERTICES, support_radius)
    field = {v: smoothstep_complement(d / support_radius) for v, d in distances.items() if d < support_radius}
    require(all(field.get(v) == 1.0 for v in SEED_VERTICES), "seed field is not full amplitude")

    seed_geometry = []
    for row in SEED_FACES:
        tri = source_positions[source_faces[row]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal /= np.linalg.norm(normal)
        dots = [float(np.dot(normal, source_normals[int(v)])) for v in source_faces[row]]
        require(min(dots) > 0.8, "stored normal is inconsistent with seed-face outward winding")
        seed_geometry.append({"face_row": row, "vertex_rows": source_faces[row].astype(int).tolist(), "stored_normal_dots_face_normal": dots, "face_normal_source_frame": normal.tolist()})

    # The 927 registration matrix maps original atlas millimeter inputs, while this NHSKIN
    # payload is already in its runtime position frame. Convert trial millimeters using the
    # exact retained 86-weight skinning map, not that provenance matrix.
    runtime_gain_by_step = {}
    for step in POSE_STEPS:
        receipt_path = SCENE / "accepted-geometry" / f"step-{step}.receipt.json"
        require(sha(receipt_path) == declaration["source_hashes_at_preflight"][str(receipt_path)],
                f"receipt hash mismatch while measuring runtime gain at {step}")
        pose, receipt_doc = transform_owner.pose_map(receipt_path)
        require(len(pose) == 86 and int(receipt_doc["accepted_step"]) == step,
                f"incomplete exact 86-owner pose while measuring runtime gain at {step}")
        gains = [(source_direction_runtime_gain(base_skin, vertex, source_normals[vertex], pose), int(vertex))
                 for vertex in field]
        runtime_gain_by_step[str(step)] = {
            "maximum_directional_gain_m_per_nhskin_coordinate_unit": max(x[0] for x in gains),
            "maximum_gain_vertex": max(gains)[1],
            "measured_support_vertex_count": len(gains),
        }
    runtime_gain_max = max(row["maximum_directional_gain_m_per_nhskin_coordinate_unit"]
                           for row in runtime_gain_by_step.values())
    require(math.isfinite(runtime_gain_max) and runtime_gain_max > 0.0,
            "invalid exact runtime skinning directional gain")

    target_keys = target_keys_at_terminal()
    screen = {f"{a:.2f}": {} for a in SCREEN_AMPLITUDES_MM}
    skin_pack_faces_reference = None
    pack_vertex_base = None
    for step in POSE_STEPS:
        pack, receipt, pack_positions, skin_faces, target_faces, pose, receipt_doc = load_pack_for_step(step, declaration, target_keys)
        offset = pack_vertex_offset(skin_faces, source_faces)
        if skin_pack_faces_reference is None:
            skin_pack_faces_reference, pack_vertex_base = skin_faces.copy(), offset
        else:
            require(np.array_equal(skin_pack_faces_reference, skin_faces) and offset == pack_vertex_base, "captured skin face ordering/vertex offset changed")
        baseline_witnesses = read_witness_pairs(step)
        target143_records = clearance._exact_surface_records(pack_positions, target_faces[TARGET_KEY])
        for amplitude_mm in SCREEN_AMPLITUDES_MM:
            source_amplitude = (amplitude_mm / 1000.0) / runtime_gain_max
            candidate_skin, source_candidate_positions = source_candidate_skin(base_skin, field, source_amplitude)
            changed_vertices = sorted(source_candidate_positions)
            changed_rows = np.flatnonzero(np.any(np.isin(source_faces, changed_vertices), axis=1))
            candidate_positions, base_residual, world_delta = world_positions_for_candidate(pack_positions, base_skin, candidate_skin, changed_vertices, pose, offset)
            candidate_records = records_for_rows(candidate_positions, skin_faces, changed_rows)
            candidate_audit = exact._audit_pair(candidate_records, target143_records, same_surface=False)
            candidate_pairs = pair_set(candidate_audit)
            original_records = records_for_rows(pack_positions, skin_faces, changed_rows)
            original_audit = exact._audit_pair(original_records, target143_records, same_surface=False)
            original_pairs = pair_set(original_audit)
            expected_original = {pair for pair in baseline_witnesses.get(TARGET_KEY, set()) if pair[0] in set(map(int, changed_rows))}
            require(original_pairs == expected_original, f"local original EPL143 re-audit differs from 1172 at step {step}")
            require(base_residual <= 1.0e-6, f"baseline 86-weight transform residual exceeds 1µm at {step}")
            screen[f"{amplitude_mm:.2f}"][str(step)] = {
                "changed_vertex_count": len(changed_vertices), "changed_face_count": int(len(changed_rows)),
                "original_epl143_pairs_on_changed_faces": len(original_pairs), "candidate_epl143_pairs": len(candidate_pairs),
                "candidate_pairs": [list(p) for p in sorted(candidate_pairs)], "baseline_lbs_residual_max_um": base_residual * 1.0e6,
                "candidate_dynamic_displacement_max_mm": world_delta * 1000.0,
                "aabb_candidate_pairs": candidate_audit["aabb_candidate_pairs"],
            }
        del pack_positions, target_faces, skin_faces, candidate_records

    selected_options = [a for a in SCREEN_AMPLITUDES_MM if all(
        screen[f"{a:.2f}"][str(step)]["candidate_epl143_pairs"] == 0 and
        screen[f"{a:.2f}"][str(step)]["candidate_dynamic_displacement_max_mm"] <= WORLD_CAP_MM + 1.0e-6
        for step in POSE_STEPS)]
    if not selected_options:
        screen_report = {
            "schema": "numi.human.epl143-inferred-skin-clearance-candidate.v1",
            "status": "bounded_screen_no_amplitude_clears_within_cap",
            "qualification": "inferred NHSKIN reference-geometry screen only; no candidate asset emitted",
            "source_identity": {"skin_path": str(SKIN), "skin_sha256": sha(SKIN),
                                "skin_manifest_path": str(SKIN_MANIFEST), "skin_manifest_sha256": sha(SKIN_MANIFEST),
                                "927_registration_builder_sha256": EXPECTED_927_GEOMETRY_BUILDER_SHA,
                                "current_commongeometry_composer_sha256": sha(GEOMETRY_OWNER),
                                "NHA_path": str(NHA), "NHA_sha256": sha(NHA),
                                "baseline_declaration_path": str(DECL), "baseline_declaration_sha256": sha(DECL)},
            "target_identity": {"semantic_stable_id": list(TARGET_KEY), "label": "right extensor pollicis longus",
                                "source_skin_face_rows": list(SEED_FACES), "source_skin_vertex_rows": list(SEED_VERTICES)},
            "screen_amplitudes_mm": list(SCREEN_AMPLITUDES_MM), "maximum_world_displacement_cap_mm": WORLD_CAP_MM,
            "runtime_LBS_directional_gain_by_pose": runtime_gain_by_step,
            "maximum_runtime_LBS_directional_gain_m_per_nhskin_coordinate_unit": runtime_gain_max,
            "per_amplitude_per_pose": screen,
            "input_hashes": {str(p): sha(p) for p in (SKIN, SKIN_MANIFEST, NHA, DECL, TRANSFORM_SCRIPT, CLEARANCE_MODULE, PREDICATE_MODULE, GEOMETRY_OWNER)},
            "driver": {"path": str(pathlib.Path(__file__).resolve()), "sha256": sha(pathlib.Path(__file__).resolve())},
            "elapsed_wall_seconds": time.monotonic() - start,
        }
        write_json(OUT / "report.json", screen_report)
        (OUT / "README.md").write_text("# EPL143 bounded clearance screen\n\nNo trial in the declared 0.25–2.0 mm set both cleared EPL143 at all eight accepted poses and remained within the 2 mm maximum runtime displacement cap. No corrected NHSKIN asset was emitted. This is a source-pinned geometry screen only.\n")
        print(json.dumps({"status": screen_report["status"], "report_sha256": sha(OUT / "report.json"),
                          "elapsed_wall_seconds": screen_report["elapsed_wall_seconds"]}, indent=2), flush=True)
        return
    selected = selected_options[0]
    source_amplitude = (selected / 1000.0) / runtime_gain_max
    candidate_skin, source_candidate_positions = source_candidate_skin(base_skin, field, source_amplitude)
    changed_vertices = sorted(source_candidate_positions)
    changed_rows = np.flatnonzero(np.any(np.isin(source_faces, changed_vertices), axis=1))

    # Position-only patch is passed through the established geometry owner, which recomputes rest-world normals.
    correction = bytearray(base_skin["raw"])
    vertex_offset = 60 + 36 * base_skin["binding_count"]
    for vertex, p in source_candidate_positions.items():
        struct.pack_into("<3f", correction, vertex_offset + int(vertex) * 56, *map(float, p))
    correction_path = OUT / "epl143-position-correction.nhskin"
    correction_path.write_bytes(correction)
    composed_raw, owner_report = geometry_owner.compose_disjoint_skin_position_corrections(
        base_skin["raw"], [bytes(correction)], global_source_matrix=global_matrix.tolist())
    candidate_path = OUT / "bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
    candidate_path.write_bytes(composed_raw)
    final_skin = transform_owner.parse_skin(candidate_path)
    final_positions_source = np.asarray([v[:3] for v in final_skin["vertices"]], dtype="<f4")
    base_positions_source = np.asarray([v[:3] for v in base_skin["vertices"]], dtype="<f4")
    actual_changed = np.flatnonzero(np.any(final_positions_source != base_positions_source, axis=1))
    require(set(map(int, actual_changed)) == set(changed_vertices), "candidate position support differs from the fixed taper")
    base_normals = np.asarray([v[3:6] for v in base_skin["vertices"]], dtype="<f4")
    final_normals = np.asarray([v[3:6] for v in final_skin["vertices"]], dtype="<f4")
    changed_normals = np.flatnonzero(np.any(base_normals != final_normals, axis=1))
    weight_offset = base_skin["full_offset"]
    vertex_end = vertex_offset + 56 * base_skin["vertex_count"]
    require(bytes(correction)[:vertex_offset] == base_skin["raw"][:vertex_offset] and bytes(correction)[vertex_end:] == base_skin["raw"][vertex_end:], "position correction changed bindings or trailing weight matrix")
    require(final_skin["bindings"] == base_skin["bindings"] and final_skin["indices"] == base_skin["indices"], "owner composition changed bindings or topology")
    require(final_skin["raw"][weight_offset:] == base_skin["raw"][weight_offset:], "owner composition changed full 86-weight matrix")
    require(final_skin["source_sha256"] == base_skin["source_sha256"], "owner composition changed source archive identity")

    pose_results = {}
    for step in POSE_STEPS:
        pack, receipt, pack_positions, skin_faces, target_faces, pose, receipt_doc = load_pack_for_step(step, declaration, target_keys)
        offset = pack_vertex_offset(skin_faces, source_faces)
        require(offset == pack_vertex_base and np.array_equal(skin_faces, skin_pack_faces_reference), "captured skin topology changed")
        candidate_positions, base_residual, dynamic_delta = world_positions_for_candidate(
            pack_positions, base_skin, final_skin, changed_vertices, pose, offset)
        require(base_residual <= 1.0e-6, f"baseline source transform residual exceeds 1µm at {step}")
        require(dynamic_delta * 1000.0 <= WORLD_CAP_MM + 1e-6, f"candidate dynamic world displacement exceeds 2mm at {step}")
        quality = face_quality(pack_positions, candidate_positions, skin_faces, changed_rows)
        require(quality["degenerate_count"] == 0 and quality["reversed_count"] == 0, f"candidate face degeneracy/reversal at {step}")
        self_audit = local_self_audit(candidate_positions, skin_faces, changed_rows)
        baseline_scan = scan_changed_targets(pack_positions, skin_faces, changed_rows, target_faces)
        candidate_scan = scan_changed_targets(candidate_positions, skin_faces, changed_rows, target_faces)
        base_by_target = pair_maps_from_scan(baseline_scan)
        candidate_by_target = pair_maps_from_scan(candidate_scan)
        old_witnesses = read_witness_pairs(step)
        base_changed = {k: {p for p in pairs if p[0] in set(map(int, changed_rows))} for k, pairs in old_witnesses.items()}
        require({k: v for k, v in base_by_target.items() if v} == {k: v for k, v in base_changed.items() if v}, f"original changed-face/all-target re-audit differs from 1172 at {step}")
        unchanged = {k: {p for p in pairs if p[0] not in set(map(int, changed_rows))} for k, pairs in old_witnesses.items()}
        candidate_full = {k: set(unchanged.get(k, set())) | set(candidate_by_target.get(k, set())) for k in target_keys}
        baseline_full = {k: set(old_witnesses.get(k, set())) for k in target_keys}
        introduced = {k: candidate_full[k] - baseline_full[k] for k in target_keys if candidate_full[k] - baseline_full[k]}
        resolved = {k: baseline_full[k] - candidate_full[k] for k in target_keys if baseline_full[k] - candidate_full[k]}
        epl_pairs = candidate_full[TARGET_KEY]
        require(len(epl_pairs) == screen[f"{selected:.2f}"][str(step)]["candidate_epl143_pairs"], f"EPL143 screen/full scan mismatch at {step}")
        pose_results[str(step)] = {
            "pack_sha256": sha(pack), "receipt_sha256": sha(receipt), "changed_vertex_count": len(changed_vertices),
            "changed_face_count": int(len(changed_rows)), "baseline_lbs_residual_max_um": base_residual * 1.0e6,
            "candidate_dynamic_displacement_max_mm": dynamic_delta * 1000.0, "changed_face_geometry": quality,
            "skin_self": self_audit,
            "all_859_target_scan": {
                "scope": "complete differential; each changed skin face tested against all 859 target surfaces; untouched skin face-pairs retained from exact, pinned 1172 witness ledger after unchanged-position/topology proof",
                "target_surface_count": len(target_keys), "baseline_changed_face_pairs": sum(map(len, base_by_target.values())),
                "candidate_changed_face_pairs": sum(map(len, candidate_by_target.values())),
                "baseline_all_skin_pair_count": sum(map(len, baseline_full.values())),
                "candidate_all_skin_pair_count": sum(map(len, candidate_full.values())),
                "resolved_pair_count": sum(map(len, resolved.values())), "introduced_pair_count": sum(map(len, introduced.values())),
                "introduced_pairs_by_target": {f"{k[0]}:{k[1]}": [list(p) for p in sorted(v)] for k, v in introduced.items()},
                "resolved_count_by_target": {f"{k[0]}:{k[1]}": len(v) for k, v in resolved.items()},
                "candidate_epl143_pair_count": len(epl_pairs), "candidate_epl143_pairs": [list(p) for p in sorted(epl_pairs)],
            },
        }
        print(json.dumps({"step": step, "EPL143_pairs": len(epl_pairs), "skin_self_unallowed": self_audit["unallowed_pair_count"],
                          "introduced_target_pairs": sum(map(len, introduced.values())), "resolved_target_pairs": sum(map(len, resolved.values())),
                          "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
        del pack_positions, skin_faces, target_faces, candidate_positions, baseline_scan, candidate_scan

    all_epl_clear = all(pose_results[str(s)]["all_859_target_scan"]["candidate_epl143_pair_count"] == 0 for s in POSE_STEPS)
    self_clear = all(pose_results[str(s)]["skin_self"]["unallowed_pair_count"] == 0 for s in POSE_STEPS)
    no_new_pairs = all(pose_results[str(s)]["all_859_target_scan"]["introduced_pair_count"] == 0 for s in POSE_STEPS)
    status = "candidate_geometry_only_pass" if all_epl_clear and self_clear and no_new_pairs else "bounded_candidate_geometry_gate_failed"
    declaration = json.loads(DECL.read_text())
    step_inputs = {}
    for step in POSE_STEPS:
        pack, receipt, _ = load_baseline(step, declaration)
        step_inputs[str(step)] = {"pack": {"path": str(pack), "sha256": sha(pack)}, "receipt": {"path": str(receipt), "sha256": sha(receipt)},
                                  "result": {"path": str(BASE_AUDIT / f"step-{step}.result.json"), "sha256": sha(BASE_AUDIT / f"step-{step}.result.json")},
                                  "crossing_witnesses": {"path": str(BASE_AUDIT / f"step-{step}.crossing-witnesses.jsonl"), "sha256": sha(BASE_AUDIT / f"step-{step}.crossing-witnesses.jsonl")}}
    report = {
        "schema": "numi.human.epl143-inferred-skin-clearance-candidate.v1", "status": status,
        "qualification": "inferred NHSKIN reference geometry only; no native viewer or contact/physics qualification",
        "source_identity": {"skin_path": str(SKIN), "skin_sha256": sha(SKIN), "skin_manifest_path": str(SKIN_MANIFEST), "skin_manifest_sha256": sha(SKIN_MANIFEST),
                            "927_registration_builder_sha256": EXPECTED_927_GEOMETRY_BUILDER_SHA,
                            "current_commongeometry_composer_sha256": sha(GEOMETRY_OWNER),
                            "NHA_path": str(NHA), "NHA_sha256": sha(NHA), "baseline_declaration_path": str(DECL), "baseline_declaration_sha256": sha(DECL)},
        "target_identity": {"semantic_stable_id": list(TARGET_KEY), "label": "right extensor pollicis longus", "source_skin_face_rows": list(SEED_FACES), "source_skin_vertex_rows": list(SEED_VERTICES), "pack_skin_vertex_offset": pack_vertex_base},
        "source_frame_and_field": {"global_source_mm_to_world_m": global_matrix.tolist(), "source_scale_m_per_unit": scale_m_per_unit,
                                    "maximum_rest_world_displacement_cap_mm": WORLD_CAP_MM, "selected_engineering_trial_amplitude_mm": selected,
                                    "source_amplitude_nhskin_coordinate_units": source_amplitude, "radius_edge_multiple": RADIUS_EDGE_MULTIPLE,
                                    "median_seed_incident_edge_nhskin_coordinate_units": median_edge,
                                    "taper_radius_nhskin_coordinate_units": support_radius,
                                    "runtime_LBS_directional_gain_by_pose": runtime_gain_by_step,
                                    "maximum_runtime_LBS_directional_gain_m_per_nhskin_coordinate_unit": runtime_gain_max,
                                    "amplitude_conversion": "requested world mm converted through maximum exact 86-weight LBS directional gain over candidate support and all eight accepted source-pose bindings; final displacement checked on packed Float32 poses",
                                    "active_taper_vertices_before_float32_rounding": len(field), "changed_source_positions": len(changed_vertices),
                                    "changed_source_vertex_ids": list(map(int, changed_vertices)), "changed_incident_face_rows": list(map(int, changed_rows)),
                                    "seed_source_face_orientation": seed_geometry,
                                    "taper": "quintic smoothstep complement in source-mesh geodesic distance; zero outside radius; outward NHSKIN source normal per vertex"},
        "candidate_assets": {"position_only_correction_path": str(correction_path), "position_only_correction_sha256": sha(correction_path),
                             "composed_candidate_path": str(candidate_path), "composed_candidate_sha256": sha(candidate_path), "bytes": len(composed_raw),
                             "changed_normal_vertex_count": int(len(changed_normals)), "changed_normal_vertex_ids": list(map(int, changed_normals)),
                             "owner_composition_report": owner_report, "all_86_bindings_identical": True, "full_86_weight_matrix_identical": True,
                             "triangle_indices_and_order_identical": True, "source_archive_identity_identical": True,
                             "support_payload_path": str(SUPPORT), "support_payload_sha256": sha(SUPPORT), "support_payload_edited": False},
        "bounded_screen": {"amplitudes_mm": list(SCREEN_AMPLITUDES_MM), "per_amplitude_per_pose": screen,
                           "selected_first_amplitude_clear_across_all_8": selected, "selected_amplitude_clears_epl143_all_8": all(screen[f"{selected:.2f}"][str(s)]["candidate_epl143_pairs"] == 0 for s in POSE_STEPS)},
        "pose_results": pose_results,
        "gates": {"EPL143_clear_at_all_8_poses": all_epl_clear, "skin_self_clear_at_all_8_poses": self_clear,
                  "no_new_pairs_against_all_859_targets_at_all_8_poses": no_new_pairs, "all_859_target_surfaces_scanned_against_changed_faces": True,
                  "untouched_face_pair_results_carried_only_from_complete_pinned_1172_witnesses": True,
                  "native_run_or_physics_admission": "not performed"},
        "input_hashes": {str(p): sha(p) for p in (SKIN, SKIN_MANIFEST, NHA, DECL, SUPPORT, TRANSFORM_SCRIPT, CLEARANCE_MODULE, PREDICATE_MODULE, GEOMETRY_OWNER)},
        "driver": {"path": str(pathlib.Path(__file__).resolve()), "sha256": sha(pathlib.Path(__file__).resolve())},
        "step_inputs": step_inputs,
        "limits": ["The correction is inferred common-atlas reference geometry, not measured anatomy or measured skin thickness.",
                   "The exact screen covers eight discrete accepted baseline poses; it does not establish continuous-time or treatment-pose clearance.",
                   "All 859 target surfaces were tested against each changed skin face. Untouched skin face-pairs were retained from 1172 only after exact position/topology scope checks.",
                   "The candidate was not run in the native viewer and has no runtime support-selector/Jacobian qualification or collision/contact authority.",
                   "Displacement amplitudes are a bounded engineering choice, not anatomy measurements."],
        "elapsed_wall_seconds": time.monotonic() - start,
    }
    write_json(OUT / "report.json", report)
    (OUT / "README.md").write_text(
        "# EPL143 inferred skin-clearance candidate\n\n"
        "This separate candidate applies a bounded local reference-position field to the two NHSKIN source faces implicated in the late EPL143 crossing. It preserves all 86 skin bindings, the full 86-column weights, source face/index order and all other source records. The existing common-geometry owner recomputes rest-world shading normals after the position-only correction. The largest declared rest-world displacement is capped at 2 mm.\n\n"
        "The report describes exact differential scans over eight accepted baseline poses: the modified skin faces are tested against all 859 target surfaces, while untouched face-pair witnesses are inherited only from the hash-pinned complete 1172 audit. It also records the exact changed-star skin-self test. This is not native-tested and is not admitted for physics or contact use.\n\n"
        "Reproduce from the evidence root with `/Users/n/numi-human-prep-venv-20261005/bin/python prepare_epl143_candidate.py`. The script refuses an existing `attempt-006` output directory and verifies all source and pose hashes first.\n")
    print(json.dumps({"status": status, "selected_mm": selected, "EPL143_clear": all_epl_clear, "self_clear": self_clear,
                      "no_new_target_pairs": no_new_pairs, "candidate_sha256": sha(candidate_path), "report_sha256": sha(OUT / "report.json"),
                      "elapsed_wall_seconds": report["elapsed_wall_seconds"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
