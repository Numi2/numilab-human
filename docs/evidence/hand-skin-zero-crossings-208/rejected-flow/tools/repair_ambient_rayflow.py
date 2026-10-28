#!/usr/bin/env python3
"""Baseline-806 ambient flow with exact outward-cone bone ray exits.

The source mesh connectivity remains fixed. Bone targets are selected by exact
segment-triangle tests along directions that have positive outward skin-normal
component. No physics runtime is modified or invoked.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter

import numpy as np
from scipy import sparse

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001")
DROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207")
HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "repair_ambient_flow.py"
CONT_PATH = HERE / "continue_ambient_flow.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ENGINE = module("ambient_flow_engine", ENGINE_PATH)
QUERY_HELPER = module("ambient_query_helper", CONT_PATH)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_bank_cache(pair, triangles, ids, matrices, offsets, positions):
    body, bone = map(int, pair)
    idx = np.flatnonzero((ids[:, 0] == body) & (ids[:, 1] == bone))
    tri = np.asarray(triangles[idx], dtype=np.float64)
    stable = ids[idx, 3].astype(np.int64)
    if not len(tri):
        raise RuntimeError(f"empty typed bank {(body, bone)}")
    M = matrices[body]
    invM = np.linalg.inv(M)
    off = offsets[body]
    pos = positions[body]
    tri_world = (tri + off) @ invM.T + pos
    flat = tri.reshape(-1, 3)
    world_flat = tri_world.reshape(-1, 3)
    lo = tri.min(axis=1)
    hi = tri.max(axis=1)
    center = 0.5 * (world_flat.min(axis=0) + world_flat.max(axis=0))
    radius = float(np.max(np.linalg.norm(world_flat - center, axis=1)))
    return {
        "body": body,
        "bone": bone,
        "indices": idx,
        "triangles_local": tri,
        "triangle_stable_ids": stable,
        "triangle_aabb_low": lo,
        "triangle_aabb_high": hi,
        "world_to_body": M,
        "body_to_world": invM,
        "com_offset": off,
        "position": pos,
        "world_sphere_center": center,
        "world_sphere_radius": radius,
    }


def exact_segment_hits(a, b, bank, exact_classifier):
    """Fast vector AABB/Moller filter followed by pinned exact classification."""
    tri = bank["triangles_local"]
    lo = bank["triangle_aabb_low"]
    hi = bank["triangle_aabb_high"]
    seg_low = np.minimum(a, b)
    seg_high = np.maximum(a, b)
    aabb = np.flatnonzero(np.all(hi >= seg_low, axis=1) & np.all(lo <= seg_high, axis=1))
    if not len(aabb):
        return [], 0, 0, 0
    candidate = tri[aabb]
    d = b - a
    e1 = candidate[:, 1] - candidate[:, 0]
    e2 = candidate[:, 2] - candidate[:, 0]
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d)
    near = np.abs(det) <= 64 * np.finfo(float).eps * scale
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=det != 0)
    tvec = a - candidate[:, 0]
    u = inv * np.einsum("ij,ij->i", tvec, pvec)
    qvec = np.cross(tvec, e1)
    v = inv * (qvec @ d)
    t = inv * np.einsum("ij,ij->i", e2, qvec)
    possible = near | (
        (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
        & (t >= -1e-10) & (t <= 1 + 1e-10)
    )
    hits = []
    uncertain = 0
    exact_tests = int(np.count_nonzero(possible))
    for ci in np.flatnonzero(possible):
        result = exact_classifier(a, b, candidate[ci])
        kind = result["classification"]
        if kind in ("proper_interior_hit", "boundary_hit"):
            tt = float(result["t"])
            if tt > 1e-10:
                hits.append((tt, int(bank["triangle_stable_ids"][aabb[ci]]), kind))
        elif kind in ("coplanar_requires_2d_test", "invalid_degenerate_triangle"):
            uncertain += 1
    hits.sort(key=lambda x: x[0])
    return hits, int(len(aabb)), exact_tests, uncertain


def choose_ray_exit(point_world, point_local, nearest_local, normal_out, bank, betas, exact_classifier):
    """Choose shortest last positive intersection that clears the typed bone."""
    v_world = nearest_local @ bank["body_to_world"].T
    nearest_depth = float(np.linalg.norm(v_world))
    if nearest_depth > 1e-12:
        nearest_dir = v_world / nearest_depth
    else:
        nearest_dir = normal_out.copy()
    candidate_dirs = np.full((len(betas), 3), np.nan, dtype=np.float64)
    candidate_dots = np.full(len(betas), np.nan, dtype=np.float64)
    candidate_depths = np.full(len(betas), np.nan, dtype=np.float64)
    candidate_hits = np.zeros(len(betas), dtype=np.int32)
    candidate_exact = np.zeros(len(betas), dtype=np.int32)
    candidate_uncertain = np.zeros(len(betas), dtype=np.int32)
    candidate_exit_tri = np.full(len(betas), -1, dtype=np.int64)
    candidate_unique_t = np.zeros(len(betas), dtype=np.int32)
    valid = []

    center = bank["world_sphere_center"]
    radius = bank["world_sphere_radius"]
    ray_length = float(np.linalg.norm(point_world - center) + radius + 0.0005)
    M = bank["world_to_body"]
    off = bank["com_offset"]
    pos = bank["position"]
    start_local = point_local

    for i, beta in enumerate(betas):
        if beta is None:
            direction = normal_out.copy()
        else:
            raw = nearest_dir + float(beta) * normal_out
            norm = float(np.linalg.norm(raw))
            if norm <= 1e-15:
                continue
            direction = raw / norm
        outward_dot = float(np.dot(direction, normal_out))
        candidate_dirs[i] = direction
        candidate_dots[i] = outward_dot
        if outward_dot < 0.25:
            continue
        local_per_world_m = direction @ M.T
        end_local = start_local + ray_length * local_per_world_m
        hits, aabb_count, exact_count, uncertain = exact_segment_hits(
            start_local, end_local, bank, exact_classifier
        )
        candidate_hits[i] = len(hits)
        candidate_exact[i] = exact_count
        candidate_uncertain[i] = uncertain
        if uncertain or not hits:
            continue
        last_t = max(h[0] for h in hits)
        if last_t <= 1e-10 or last_t >= 1.0:
            continue
        last_ids = [h[1] for h in hits if abs(h[0] - last_t) <= 1e-9]
        unique_events = []
        for h in hits:
            if not unique_events or abs(h[0] - unique_events[-1]) > 1e-9:
                unique_events.append(h[0])
        exit_distance = last_t * ray_length
        candidate_depths[i] = exit_distance * 1000.0
        candidate_exit_tri[i] = int(last_ids[0]) if last_ids else -1
        candidate_unique_t[i] = len(unique_events)
        valid.append((exit_distance, i, direction, last_t, last_ids, len(hits), len(unique_events), aabb_count, exact_count))
    if not valid:
        return {
            "valid": False,
            "nearest_dir": nearest_dir,
            "nearest_depth_mm": nearest_depth * 1000.0,
            "candidate_dirs": candidate_dirs,
            "candidate_dots": candidate_dots,
            "candidate_depths_mm": candidate_depths,
            "candidate_hit_count": candidate_hits,
            "candidate_exact_count": candidate_exact,
            "candidate_uncertain_count": candidate_uncertain,
            "candidate_exit_triangle_id": candidate_exit_tri,
            "candidate_unique_event_count": candidate_unique_t,
        }
    valid.sort(key=lambda x: x[0])
    distance, chosen, direction, t_exit, exit_ids, hit_count, unique_count, aabb_count, exact_count = valid[0]
    target_distance = distance + 0.00015
    return {
        "valid": True,
        "nearest_dir": nearest_dir,
        "nearest_depth_mm": nearest_depth * 1000.0,
        "candidate_dirs": candidate_dirs,
        "candidate_dots": candidate_dots,
        "candidate_depths_mm": candidate_depths,
        "candidate_hit_count": candidate_hits,
        "candidate_exact_count": candidate_exact,
        "candidate_uncertain_count": candidate_uncertain,
        "candidate_exit_triangle_id": candidate_exit_tri,
        "candidate_unique_event_count": candidate_unique_t,
        "chosen_index": int(chosen),
        "chosen_beta": None if betas[chosen] is None else float(betas[chosen]),
        "chosen_mode": "pure_normal" if betas[chosen] is None else "nearest_plus_beta_normal",
        "chosen_direction": direction,
        "chosen_dot_outward": float(candidate_dots[chosen]),
        "exit_depth_mm": float(distance * 1000.0),
        "target_depth_mm": float(target_distance * 1000.0),
        "target_vector_mm": direction * target_distance * 1000.0,
        "exit_point_world_m": point_world + direction * distance,
        "exit_triangle_ids": [int(x) for x in exit_ids],
        "ray_hit_count": int(hit_count),
        "ray_unique_event_count": int(unique_count),
        "ray_aabb_triangle_count": int(aabb_count),
        "ray_exact_test_count": int(exact_count),
        "ray_length_m": ray_length,
        "last_hit_parameter": float(t_exit),
    }


def query_categories(sample_bary, sample_seed, support_count):
    kinds = []
    for bary in sample_bary:
        nz = int(np.count_nonzero(np.asarray(bary) > 1e-9))
        kinds.append({1: "face_vertex", 2: "edge_probe", 3: "face_centroid"}[nz])
    return (
        np.r_[np.asarray(kinds, dtype=object), np.full(support_count, "flow_support_vertex", dtype=object)],
        np.r_[sample_seed, np.zeros(support_count, dtype=bool)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rayexit-baseline-001")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--rings", type=int, default=3)
    ap.add_argument("--spacing-mm", type=float, default=6.0)
    ap.add_argument("--max-step-mm", type=float, default=0.8)
    ap.add_argument("--lipschitz-step", type=float, default=0.35)
    ap.add_argument("--maxiter", type=int, default=80)
    a = ap.parse_args()

    out = HERE / a.run
    out.mkdir(exist_ok=False)
    start_time = time.monotonic()
    base = ROOT / "inputs/baseline-806.npz"
    audit_path = ROOT / "baseline/audit-159/report.json"
    source = np.load(base, allow_pickle=False)
    payload = {k: source[k].copy() for k in source.files}
    X0 = payload["points_world_m"].copy()
    X = X0.copy()
    F = payload["source_O_faces"].copy()
    audit = json.load(audit_path.open())
    triangle_pairs = [
        p for bank in audit["banks"]
        for p in (bank.get("result") or {}).get("triangle_pairs", [])
    ]
    seed_faces = sorted({int(p[0]) for p in triangle_pairs})
    if not seed_faces:
        raise RuntimeError("baseline audit yielded no crossing faces")

    edges = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(X), len(X)),
    )
    query_nodes = set(map(int, F[seed_faces].ravel()))
    front = set(query_nodes)
    for _ in range(a.rings):
        new = set(map(int, adj[list(front)].indices)) - query_nodes
        query_nodes.update(new)
        front = new
    query_nodes = np.asarray(sorted(query_nodes), dtype=np.int64)
    query_mask = np.zeros(len(X), dtype=bool)
    query_mask[query_nodes] = True
    query_faces = np.flatnonzero(np.any(query_mask[F], axis=1))
    sample_face, sample_bary, sample_seed = ENGINE.sample_layout(F, query_faces, seed_faces)
    sample_vertices = F[sample_face]
    sample0 = np.einsum("ij,ijk->ik", sample_bary, X0[sample_vertices].astype(np.float64))
    h_mm = a.spacing_mm
    h = h_mm / 1000.0
    _, control_keys = ENGINE.spatial_basis(sample0, h=h)
    W0, _ = ENGINE.spatial_basis(X0.astype(np.float64), control_keys=control_keys, h=h)
    support_nodes = np.flatnonzero(np.asarray(W0.getnnz(axis=1)).ravel() > 0)
    support_mask = np.zeros(len(X), dtype=bool)
    support_mask[support_nodes] = True
    outside_nodes = np.flatnonzero(~support_mask)
    flow_faces = np.flatnonzero(np.any(support_mask[F], axis=1))
    Dmat = ENGINE.build_difference_operator(control_keys)

    helper = module("ray_bone_helper", ROOT / "tools/human_skin_fat_bone_audit.py")
    triangles, ids, gids, bodies, matrices, offsets, positions, bankkeys, bank_pins, exact = helper.read_bank_inputs(
        ROOT / "inputs/bank-export-039/source-preflight", ROOT / "tools"
    )
    exact_classifier = exact["classify_segment_triangle"]
    bank_cache = {}
    face_normals_in = np.cross(
        X0[F[:, 1]].astype(float) - X0[F[:, 0]].astype(float),
        X0[F[:, 2]].astype(float) - X0[F[:, 0]].astype(float),
    )
    vertex_normals_out = np.zeros_like(X0, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals_out, F[:, corner], face_normals_in)
    vertex_normals_out *= -1.0
    vertex_normals_out /= np.maximum(np.linalg.norm(vertex_normals_out, axis=1)[:, None], 1e-30)
    betas = [0.25, 0.5, 1.0, 2.0, 4.0, None]

    report = {
        "baseline_sha256": digest(base),
        "baseline_audit_sha256": digest(audit_path),
        "bank_pin_set": bank_pins,
        "script_sha256": digest(__file__),
        "candidate_scope": "baseline-806; outward-cone exact ray-exit target selection; offline geometry only",
        "native_admission": False,
        "seed_face_count": len(seed_faces),
        "seed_triangle_pair_count": len(triangle_pairs),
        "query_ring_count": a.rings,
        "query_face_count": int(len(query_faces)),
        "query_surface_sample_count": int(len(sample_face)),
        "flow_support_vertex_count": int(len(support_nodes)),
        "flow_support_face_count": int(len(flow_faces)),
        "control_count": int(len(control_keys)),
        "grid_spacing_mm": h_mm,
        "ray_beta_values": betas,
        "ray_direction_modes": ["normalize(nearest_exit_direction + beta * current_outward_skin_normal) for beta 0.25,0.5,1,2,4", "pure current outward skin normal"],
        "ray_direction_gate": "require dot(direction, outward_normal) >= 0.25 for all five beta candidates and pure-normal candidate",
        "ray_exit_rule": "exact segment-triangle hits in the matching typed bank; use the last positive hit before an endpoint beyond the bank's world-space bounding sphere, then add 0.15 mm",
        "step_bound": "alpha * sqrt(3) * maximum adjacent control difference / spacing <= lipschitz_step; alpha * max coefficient norm <= max_step_mm",
        "outside_rule": "source vertices with no active compact-support basis are copied bitwise",
        "iterations": [],
        "limitations": [
            "sampled inside counts are not the full exact skin-bone intersection gate",
            "same-mesh exact self-intersection audit remains separate",
            "the ray endpoint is proven outside the typed bank's vertex bounding sphere; exact triangles certify all intervening surface hits",
            "no native physics state is changed",
        ],
    }
    flows = []
    status = "MAX_STEPS"
    prior_inside = None
    prior_max_depth = None
    stalls = 0
    start_candidate = X0.copy()
    target_choice_files = []

    for step in range(a.max_steps):
        if time.monotonic() - start_time >= a.max_seconds:
            status = "TIME_LIMIT"
            break
        step_start = time.monotonic()
        face_points = np.einsum("ij,ijk->ik", sample_bary, X[sample_vertices].astype(np.float64))
        query_points = np.vstack([face_points, X[support_nodes].astype(np.float64)]).astype(np.float32)
        query_kind, query_seed = query_categories(sample_bary, sample_seed, len(support_nodes))
        rows, query_manifest, free_bytes = QUERY_HELPER.query_verified(
            query_points, out / f"step-{step:03d}" / "query", timeout_s=60
        )
        inside = [r for r in rows if r["classification"] == "inside"]
        row_ids = np.asarray([int(r["query_row"]) for r in inside], dtype=np.int64)
        entry = {
            "step": step,
            "queried_surface_witness_rows": int(len(face_points)),
            "queried_seed_face_witness_rows": int(sample_seed.sum()),
            "queried_flow_support_vertices": int(len(support_nodes)),
            "inside_rows": int(len(inside)),
            "inside_seed_face_rows": int(np.count_nonzero(query_seed[row_ids])) if len(row_ids) else 0,
            "inside_source_vertex_rows": int(sum(query_kind[int(r["query_row"])] in ("face_vertex", "flow_support_vertex") for r in inside)),
            "inside_edge_centroid_probe_rows": int(sum(query_kind[int(r["query_row"])] in ("edge_probe", "face_centroid") for r in inside)),
            "inside_by_witness_type": dict(sorted(Counter(str(query_kind[int(r["query_row"])]) for r in inside).items())),
            "inside_by_typed_bank": dict(sorted(Counter(f"{r['bank_body']}/{r['bank_stable_id']}" for r in inside).items())),
            "query_files": query_manifest,
            "free_disk_bytes_after_query": int(free_bytes),
            "elapsed_query_s": time.monotonic() - step_start,
        }
        print(json.dumps(entry), flush=True)
        if not inside:
            entry["status"] = "NO_INSIDE_ROWS_IN_SAMPLED_WITNESSES"
            report["iterations"].append(entry)
            status = entry["status"]
            break

        face_normals_in = np.cross(
            X[F[:, 1]].astype(float) - X[F[:, 0]].astype(float),
            X[F[:, 2]].astype(float) - X[F[:, 0]].astype(float),
        )
        vertex_normals_out = np.zeros_like(X, dtype=np.float64)
        for corner in range(3):
            np.add.at(vertex_normals_out, F[:, corner], face_normals_in)
        vertex_normals_out *= -1.0
        vertex_normals_out /= np.maximum(np.linalg.norm(vertex_normals_out, axis=1)[:, None], 1e-30)

        targets_mm = []
        target_points = []
        choices = []
        target_valid_inside = []
        nearest_depths = []
        ray_depths = []
        selected_betas = []
        dot_values = []
        ray_failures = Counter()
        ray_eval_counts = Counter()
        for r in inside:
            qrow = int(r["query_row"])
            body = int(r["bank_body"])
            bone = int(r["bank_stable_id"])
            key = (body, bone)
            if key not in bank_cache:
                bank_cache[key] = prepare_bank_cache(key, triangles, ids, matrices, offsets, positions)
            bank = bank_cache[key]
            point_world = query_points[qrow].astype(np.float64)
            p_local = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"], dtype=np.float64)
            closest_local = np.asarray([float(r[f"closest_{c}"]) for c in "xyz"], dtype=np.float64)
            vlocal = closest_local - p_local

            if qrow < len(sample_face):
                face_id = int(sample_face[qrow])
                bary = sample_bary[qrow]
                if np.count_nonzero(np.asarray(bary) > 1e-9) == 1:
                    vertex_id = int(F[face_id, int(np.argmax(bary))])
                    normal_out = vertex_normals_out[vertex_id]
                    normal_source = "averaged_vertex"
                else:
                    normal_in = face_normals_in[face_id]
                    normal_out = -normal_in / max(float(np.linalg.norm(normal_in)), 1e-30)
                    normal_source = "face"
            else:
                vertex_id = int(support_nodes[qrow - len(sample_face)])
                normal_out = vertex_normals_out[vertex_id]
                normal_source = "averaged_vertex"
            nlen = float(np.linalg.norm(normal_out))
            if nlen <= 1e-12:
                ray_failures["invalid_outward_normal"] += 1
                vworld = vlocal @ bank["body_to_world"].T
                nearest_depth = float(np.linalg.norm(vworld))
                nearest_dir = vworld / nearest_depth if nearest_depth > 1e-12 else np.full(3, np.nan)
                choices.append({
                    "query_row": qrow, "body": body, "bone": bone,
                    "inside_witness_type": str(query_kind[qrow]),
                    "seed_crossing_face_witness": bool(query_seed[qrow]),
                    "valid": False, "failure": "invalid_outward_normal",
                    "normal_source": normal_source,
                    "outward_normal": np.full(3, np.nan),
                    "nearest_exit_direction": nearest_dir,
                    "nearest_exit_depth_mm": nearest_depth * 1000.0,
                    "candidate_directions": np.full((len(betas), 3), np.nan),
                    "candidate_outward_dots": np.full(len(betas), np.nan),
                    "candidate_exit_depths_mm": np.full(len(betas), np.nan),
                    "candidate_hit_counts": np.zeros(len(betas), dtype=np.int32),
                    "candidate_exact_counts": np.zeros(len(betas), dtype=np.int32),
                    "candidate_uncertain_counts": np.zeros(len(betas), dtype=np.int32),
                    "candidate_exit_triangle_ids": np.full(len(betas), -1, dtype=np.int64),
                    "candidate_unique_event_counts": np.zeros(len(betas), dtype=np.int32),
                })
                continue
            normal_out = normal_out / nlen
            vworld = vlocal @ bank["body_to_world"].T
            nearest_depth = float(np.linalg.norm(vworld))
            nearest_dir = vworld / nearest_depth if nearest_depth > 1e-12 else normal_out.copy()
            ray = choose_ray_exit(
                point_world, p_local, vlocal, normal_out, bank, betas, exact_classifier
            )
            ray_eval_counts["rays_considered"] += len(betas)
            ray_eval_counts["rays_accepted_cone"] += int(np.count_nonzero(ray["candidate_dots"] >= 0.25))
            ray_eval_counts["rays_with_exact_exit"] += int(np.count_nonzero(np.isfinite(ray["candidate_depths_mm"])))
            ray_eval_counts["ray_exact_triangle_tests"] += int(np.nansum(ray["candidate_exact_count"]))
            nearest_depths.append(nearest_depth * 1000.0)
            choice = {
                "query_row": qrow,
                "body": body,
                "bone": bone,
                "inside_witness_type": str(query_kind[qrow]),
                "seed_crossing_face_witness": bool(query_seed[qrow]),
                "normal_source": normal_source,
                "outward_normal": normal_out,
                "nearest_exit_direction": ray["nearest_dir"],
                "nearest_exit_depth_mm": ray["nearest_depth_mm"],
                "candidate_directions": ray["candidate_dirs"],
                "candidate_outward_dots": ray["candidate_dots"],
                "candidate_exit_depths_mm": ray["candidate_depths_mm"],
                "candidate_hit_counts": ray["candidate_hit_count"],
                "candidate_exact_counts": ray["candidate_exact_count"],
                "candidate_uncertain_counts": ray["candidate_uncertain_count"],
                "candidate_exit_triangle_ids": ray["candidate_exit_triangle_id"],
                "candidate_unique_event_counts": ray["candidate_unique_event_count"],
                "valid": bool(ray["valid"]),
            }
            if not ray["valid"]:
                ray_failures["no_clearing_ray_exit"] += 1
                choices.append(choice)
                continue
            choice.update({
                "chosen_beta": ray["chosen_beta"],
                "chosen_direction": ray["chosen_direction"],
                "chosen_outward_dot": ray["chosen_dot_outward"],
                "exit_depth_mm": ray["exit_depth_mm"],
                "target_depth_mm": ray["target_depth_mm"],
                "exit_point_world_m": ray["exit_point_world_m"],
                "exit_triangle_ids": ray["exit_triangle_ids"],
                "ray_hit_count": ray["ray_hit_count"],
                "ray_unique_event_count": ray["ray_unique_event_count"],
                "ray_length_m": ray["ray_length_m"],
                "last_hit_parameter": ray["last_hit_parameter"],
            })
            choices.append(choice)
            targets_mm.append(ray["target_vector_mm"])
            target_points.append(point_world)
            target_valid_inside.append(r)
            ray_depths.append(ray["exit_depth_mm"])
            selected_betas.append("pure_normal" if ray["chosen_beta"] is None else float(ray["chosen_beta"]))
            dot_values.append(ray["chosen_dot_outward"])

        entry["ray_target_rows"] = int(len(targets_mm))
        entry["ray_failure_rows"] = int(sum(ray_failures.values()))
        entry["ray_failure_reasons"] = dict(ray_failures)
        entry["ray_evaluations"] = dict(ray_eval_counts)
        entry["nearest_exit_depth_mm_max"] = float(max(nearest_depths)) if nearest_depths else 0.0
        entry["actual_ray_exit_depth_mm_max"] = float(max(ray_depths)) if ray_depths else 0.0
        entry["actual_ray_exit_depth_mm_median"] = float(np.median(ray_depths)) if ray_depths else 0.0
        entry["selected_beta_counts"] = dict(sorted(Counter(selected_betas).items(), key=lambda item: str(item[0])))
        entry["selected_outward_dot_min"] = float(min(dot_values)) if dot_values else None
        entry["selected_outward_dot_mean"] = float(np.mean(dot_values)) if dot_values else None
        entry["nearest_exit_opposing_outward_normal_rows"] = int(sum(
            float(np.dot(c["nearest_exit_direction"], c["outward_normal"])) < 0
            for c in choices if c["valid"]
        ))
        # Keep per-witness chosen ray, alternatives, normal, exit depth and hit IDs.
        choice_file = out / f"ray-targets-step-{step:03d}.npz"
        np.savez_compressed(
            choice_file,
            query_rows=np.asarray([int(x["query_row"]) for x in choices], dtype=np.int32),
            body=np.asarray([int(x["body"]) for x in choices], dtype=np.int32),
            stable_bone_id=np.asarray([int(x["bone"]) for x in choices], dtype=np.int32),
            valid=np.asarray([bool(x["valid"]) for x in choices], dtype=bool),
            normal_source=np.asarray([0 if x["normal_source"] == "face" else 1 for x in choices], dtype=np.int8),
            outward_normals=np.asarray([x["outward_normal"] for x in choices], dtype=np.float32),
            nearest_exit_directions=np.asarray([x["nearest_exit_direction"] for x in choices], dtype=np.float32),
            nearest_exit_depth_mm=np.asarray([x["nearest_exit_depth_mm"] for x in choices], dtype=np.float32),
            candidate_directions=np.asarray([x["candidate_directions"] for x in choices], dtype=np.float32),
            candidate_outward_dots=np.asarray([x["candidate_outward_dots"] for x in choices], dtype=np.float32),
            candidate_exit_depths_mm=np.asarray([x["candidate_exit_depths_mm"] for x in choices], dtype=np.float32),
            candidate_hit_counts=np.asarray([x["candidate_hit_counts"] for x in choices], dtype=np.int32),
            candidate_exact_counts=np.asarray([x["candidate_exact_counts"] for x in choices], dtype=np.int32),
            candidate_uncertain_counts=np.asarray([x["candidate_uncertain_counts"] for x in choices], dtype=np.int32),
            candidate_exit_triangle_ids=np.asarray([x["candidate_exit_triangle_ids"] for x in choices], dtype=np.int64),
            chosen_beta=np.asarray([np.nan if x.get("chosen_beta") is None else x["chosen_beta"] for x in choices], dtype=np.float32),
            chosen_pure_normal=np.asarray([x.get("chosen_mode") == "pure_normal" for x in choices], dtype=bool),
            chosen_direction=np.asarray([x.get("chosen_direction", [np.nan]*3) for x in choices], dtype=np.float32),
            chosen_outward_dot=np.asarray([x.get("chosen_outward_dot", np.nan) for x in choices], dtype=np.float32),
            exit_depth_mm=np.asarray([x.get("exit_depth_mm", np.nan) for x in choices], dtype=np.float32),
            target_depth_mm=np.asarray([x.get("target_depth_mm", np.nan) for x in choices], dtype=np.float32),
            exit_point_world_m=np.asarray([x.get("exit_point_world_m", [np.nan]*3) for x in choices], dtype=np.float32),
            exit_triangle_ids_json=np.asarray([json.dumps(x.get("exit_triangle_ids", [])) for x in choices]),
            ray_hit_count=np.asarray([x.get("ray_hit_count", 0) for x in choices], dtype=np.int32),
            ray_unique_event_count=np.asarray([x.get("ray_unique_event_count", 0) for x in choices], dtype=np.int32),
            ray_length_m=np.asarray([x.get("ray_length_m", np.nan) for x in choices], dtype=np.float32),
            last_hit_parameter=np.asarray([x.get("last_hit_parameter", np.nan) for x in choices], dtype=np.float32),
        )
        entry["ray_target_file"] = str(choice_file)
        entry["ray_target_file_sha256"] = digest(choice_file)

        if not targets_mm:
            entry["status"] = "NO_VALID_OUTWARD_RAY_EXITS"
            report["iterations"].append(entry)
            status = entry["status"]
            break

        targets_mm = np.asarray(targets_mm, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        B, _ = ENGINE.spatial_basis(target_points, control_keys=control_keys, h=h)
        coverage = np.asarray(B.sum(axis=1)).ravel()
        entry["field_coverage_min"] = float(coverage.min())
        if np.any(coverage < 0.05):
            entry["status"] = "RAY_WITNESS_OUTSIDE_COMPACT_SUPPORT"
            report["iterations"].append(entry)
            status = entry["status"]
            break
        controls_mm, cg_info = ENGINE.solve_velocity(B, Dmat, targets_mm, maxiter=a.maxiter)
        entry["cg_info"] = cg_info
        fitted = B @ controls_mm
        fit_residual = np.linalg.norm(fitted - targets_mm, axis=1)
        entry["fit_residual_mm_mean"] = float(np.mean(fit_residual))
        entry["fit_residual_mm_max"] = float(np.max(fit_residual))
        entry["ray_target_depth_mm_max"] = float(np.max(np.linalg.norm(targets_mm, axis=1)))
        entry["ray_target_depth_mm_median"] = float(np.median(np.linalg.norm(targets_mm, axis=1)))

        L, max_adj = ENGINE.conservative_lipschitz(control_keys, controls_mm, h_mm)
        max_control = float(np.max(np.linalg.norm(controls_mm, axis=1)))
        Wnow, _ = ENGINE.spatial_basis(X.astype(np.float64), control_keys=control_keys, h=h)
        vertex_flow = np.asarray(Wnow @ controls_mm)
        max_vertex_flow = float(np.max(np.linalg.norm(vertex_flow[support_nodes], axis=1)))
        alpha = min(
            1.0,
            a.max_step_mm / max(max_control, 1e-12),
            a.lipschitz_step / max(L, 1e-12),
        )
        entry.update({
            "control_max_norm_mm": max_control,
            "control_max_adjacent_difference_mm": max_adj,
            "ambient_lipschitz_bound_per_step": float(alpha * L),
            "alpha": float(alpha),
            "max_vertex_flow_mm": max_vertex_flow,
            "max_ambient_step_bound_mm": float(alpha * max_control),
        })

        accepted = None
        area_trials = []
        for backtrack in range(10):
            aa = alpha * (0.5 ** backtrack)
            Y, flow, area_old, area_new, ratio, good = ENGINE.apply_flow_step(
                X, F, support_nodes, flow_faces, Wnow, controls_mm, aa
            )
            area_trials.append({
                "alpha": float(aa),
                "pass_finite_nondegenerate": good,
                "min_flow_face_area_ratio": float(np.min(ratio)),
                "min_flow_face_area_m2": float(np.min(area_new)),
            })
            if good:
                accepted = (Y, flow, aa, area_new, ratio)
                break
        entry["area_backtracks"] = area_trials
        if accepted is None:
            entry["status"] = "FINITE_NONDEGENERATE_AREA_GATE"
            report["iterations"].append(entry)
            status = entry["status"]
            break
        Y, flow, accepted_alpha, area_new, ratio = accepted
        outside_exact = bool(np.array_equal(Y[outside_nodes].view(np.uint32), X[outside_nodes].view(np.uint32)))
        if not outside_exact or not np.array_equal(F, payload["source_O_faces"]):
            raise RuntimeError("ambient ray flow changed outside-support source vertices or connectivity")
        displacement = np.linalg.norm(Y.astype(np.float64) - X.astype(np.float64), axis=1) * 1000.0
        entry.update({
            "status": "APPLIED_AMBIENT_STEP",
            "accepted_alpha": float(accepted_alpha),
            "max_applied_vertex_step_mm": float(displacement.max()),
            "moved_vertex_count": int(np.count_nonzero(displacement > 0)),
            "outside_bitwise_unchanged": outside_exact,
            "source_faces_bitwise_unchanged": True,
            "min_flow_face_area_ratio": float(np.min(ratio)),
            "min_flow_face_area_m2": float(np.min(area_new)),
            "elapsed_step_s": time.monotonic() - step_start,
        })
        X = Y
        flow_path = out / f"flow-step-{step:03d}.npz"
        np.savez_compressed(
            flow_path,
            control_keys=control_keys.astype(np.int32),
            coefficients_mm=controls_mm.astype(np.float32),
            alpha=np.float32(accepted_alpha),
            spacing_mm=np.float32(h_mm),
            max_step_mm=np.float32(a.max_step_mm),
            lipschitz_cap=np.float32(a.lipschitz_step),
            source_sha256=np.asarray([digest(base)]),
        )
        flows.append({"step": step, "path": str(flow_path), "sha256": digest(flow_path)})
        entry["flow_step_file"] = str(flow_path)
        report["iterations"].append(entry)

        progress = (
            prior_inside is None
            or len(inside) < prior_inside
            or (prior_max_depth is not None and entry["actual_ray_exit_depth_mm_max"] < prior_max_depth - 0.01)
        )
        stalls = 0 if progress else stalls + 1
        entry["consecutive_stalled_steps"] = stalls
        prior_inside = len(inside)
        prior_max_depth = entry["actual_ray_exit_depth_mm_max"]
        payload["points_world_m"] = X
        payload["ambient_rayflow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
        np.savez_compressed(out / f"checkpoint-{step:03d}.npz", **payload)
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "step": step,
            "inside_rows": len(inside),
            "inside_source_vertex_rows": entry["inside_source_vertex_rows"],
            "inside_edge_centroid_probe_rows": entry["inside_edge_centroid_probe_rows"],
            "ray_target_rows": entry["ray_target_rows"],
            "ray_failure_rows": entry["ray_failure_rows"],
            "actual_ray_exit_depth_mm_max": entry["actual_ray_exit_depth_mm_max"],
            "actual_ray_exit_depth_mm_median": entry["actual_ray_exit_depth_mm_median"],
            "nearest_exit_opposing_outward_normal_rows": entry["nearest_exit_opposing_outward_normal_rows"],
            "selected_outward_dot_min": entry["selected_outward_dot_min"],
            "fit_residual_mm_mean": entry["fit_residual_mm_mean"],
            "ambient_lipschitz_bound_per_step": entry["ambient_lipschitz_bound_per_step"],
            "max_applied_vertex_step_mm": entry["max_applied_vertex_step_mm"],
            "min_flow_face_area_ratio": entry["min_flow_face_area_ratio"],
            "elapsed_step_s": entry["elapsed_step_s"],
        }), flush=True)
        if stalls >= 4:
            status = "FOUR_STALLED_STEPS"
            break
        if time.monotonic() - start_time >= a.max_seconds:
            status = "TIME_LIMIT"
            break

    candidate = out / "candidate.npz"
    payload["points_world_m"] = X
    payload["ambient_rayflow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
    np.savez_compressed(candidate, **payload)
    (out / "flow-index.json").write_text(json.dumps(flows, indent=2) + "\n")
    changed = np.any(X.view(np.uint32) != X0.view(np.uint32), axis=1)
    changed_xyz = X[changed].astype(np.float64) * 1000.0
    seed_vertices = np.unique(F[seed_faces].ravel())
    distance_to_seed = np.linalg.norm(
        X0.astype(np.float64)[:, None, :] - X0[seed_vertices].astype(np.float64)[None, :, :],
        axis=2,
    ).min(axis=1)
    remote_mask = distance_to_seed > 0.100
    total_disp_mm = np.linalg.norm(X.astype(np.float64) - X0.astype(np.float64), axis=1) * 1000.0
    report.update({
        "status": status,
        "candidate_path": str(candidate),
        "candidate_sha256": digest(candidate),
        "candidate_vertex_count": int(len(X)),
        "candidate_face_count": int(len(F)),
        "source_faces_unchanged": bool(np.array_equal(F, payload["source_O_faces"])),
        "changed_vertex_count": int(changed.sum()),
        "changed_vertex_bbox_mm": {
            "min": changed_xyz.min(axis=0).tolist() if len(changed_xyz) else None,
            "max": changed_xyz.max(axis=0).tolist() if len(changed_xyz) else None,
        },
        "max_total_displacement_mm": float(total_disp_mm.max()),
        "remote_torso_back_proxy": {
            "definition": "baseline vertices more than 100 mm Euclidean distance from audited seed-face vertices",
            "vertex_count": int(remote_mask.sum()),
            "max_displacement_mm": float(total_disp_mm[remote_mask].max()) if np.any(remote_mask) else 0.0,
            "changed_vertex_count": int(np.count_nonzero(changed & remote_mask)),
        },
        "flow_steps": flows,
        "ray_target_choice_files": [e.get("ray_target_file") for e in report["iterations"] if e.get("ray_target_file")],
        "elapsed_s": time.monotonic() - start_time,
    })
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "iterations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
