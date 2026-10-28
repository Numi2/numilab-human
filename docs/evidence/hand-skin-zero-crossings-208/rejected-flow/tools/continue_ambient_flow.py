#!/usr/bin/env python3
"""Continue ambient-flow-030 from a measured candidate with verified query dumps."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001")
R = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207")
HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "repair_ambient_flow.py"
SPEC = importlib.util.spec_from_file_location("ambient_flow_engine", ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENGINE
SPEC.loader.exec_module(ENGINE)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def query_verified(points, folder, timeout_s=60):
    folder.mkdir(parents=True, exist_ok=False)
    pts = np.ascontiguousarray(points, dtype=np.float32)
    csv_path = folder / "points.csv"
    with csv_path.open("w") as f:
        f.write("query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32\n")
        bits = pts.view(np.uint32)
        for i, (v, b) in enumerate(zip(pts, bits)):
            f.write(",".join([str(i), *[format(float(x), ".9g") for x in v], *[str(int(x)) for x in b]]) + "\n")

    bank = BASE / "inputs/bank-export-039/source-preflight"
    binary = BASE / "tools/numi-matter-human-skin-fat-static-query"
    cp = subprocess.run([
        str(binary),
        str(bank / "exact-emitted.triangles.csv"),
        str(bank / "exact-emitted.bodies.csv"),
        str(csv_path),
        str(folder / "pairs.csv"),
        str(folder / "nodes.csv"),
        "185",
    ], capture_output=True, text=True, timeout=timeout_s)
    (folder / "run.log").write_text(cp.stdout + cp.stderr)
    if cp.returncode:
        raise RuntimeError(cp.stdout + cp.stderr)

    rows = list(csv.DictReader((folder / "pairs.csv").open()))
    nodes = list(csv.DictReader((folder / "nodes.csv").open()))
    if len(nodes) != len(pts):
        raise RuntimeError(f"query node rows {len(nodes)} != input points {len(pts)}")
    if not all(int(x["ambiguous_banks"]) == 0 and int(x["invalid_banks"]) == 0 for x in nodes):
        raise RuntimeError("native query reported ambiguous or invalid bank state")

    manifest = {}
    for name in ("points.csv", "pairs.csv", "nodes.csv"):
        p = folder / name
        raw_hash = digest(p)
        raw_size = p.stat().st_size
        subprocess.run(["gzip", "-1", str(p)], check=True)
        gz = Path(str(p) + ".gz")
        restored_hash = hashlib.sha256()
        restored_size = 0
        with gzip.open(gz, "rb") as src:
            for block in iter(lambda: src.read(1 << 20), b""):
                restored_hash.update(block)
                restored_size += len(block)
        if restored_hash.hexdigest() != raw_hash or restored_size != raw_size:
            raise RuntimeError(f"gzip round-trip verification failed for {name}")
        manifest[name] = {
            "raw_sha256": raw_hash,
            "raw_bytes": raw_size,
            "gzip_bytes": gz.stat().st_size,
            "gzip_roundtrip_verified": True,
        }
    return rows, manifest, shutil.disk_usage(folder).free


def classify_witnesses(query_rows, query_kinds, query_seed_mask):
    counts = Counter()
    seed_inside = 0
    source_vertex_inside = 0
    interior_probe_inside = 0
    for row in query_rows:
        if row["classification"] != "inside":
            continue
        q = int(row["query_row"])
        kind = str(query_kinds[q])
        is_seed = bool(query_seed_mask[q])
        counts[("seed_" if is_seed else "nearby_") + kind] += 1
        if kind in ("face_vertex", "flow_support_vertex"):
            source_vertex_inside += 1
        else:
            interior_probe_inside += 1
        if is_seed:
            seed_inside += 1
    return counts, seed_inside, source_vertex_inside, interior_probe_inside


def current_seed_faces(F, baseline_audit, current_bone_report):
    initial = {
        int(pair[0])
        for bank in baseline_audit["banks"]
        for pair in (bank.get("result") or {}).get("triangle_pairs", [])
    }
    current = {
        int(pair[0])
        for bank in current_bone_report["banks"]
        for pair in (bank.get("result") or {}).get("triangle_pairs", [])
    }
    return sorted(initial | current), sorted(initial), sorted(current)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="trial-001-continuation")
    ap.add_argument("--start-candidate", type=Path, required=True)
    ap.add_argument("--prior-flow-index", type=Path, required=True)
    ap.add_argument("--remaining-bone-report", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--rings", type=int, default=3)
    ap.add_argument("--spacing-mm", type=float, default=6.0)
    ap.add_argument("--max-step-mm", type=float, default=0.8)
    ap.add_argument("--lipschitz-step", type=float, default=0.35)
    ap.add_argument("--maxiter", type=int, default=80)
    a = ap.parse_args()

    out = HERE / a.run
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    start_candidate = a.start_candidate.resolve()
    start_hash = digest(start_candidate)
    start_bone = json.load(a.remaining_bone_report.open())
    if not start_bone.get("complete") or int(start_bone.get("bank_count", 0)) != 185:
        raise RuntimeError("continuation requires the complete 185-bank exact report")
    prior_flow_index = json.load(a.prior_flow_index.open())
    prior_flows = list(prior_flow_index)
    for item in prior_flows:
        p = Path(item["path"])
        if not p.exists() or digest(p) != item["sha256"]:
            raise RuntimeError(f"prior flow step missing or hash mismatch: {p}")

    source = np.load(BASE / "inputs/baseline-806.npz", allow_pickle=False)
    X0 = source["points_world_m"].copy()
    F = source["source_O_faces"].copy()
    cand = np.load(start_candidate, allow_pickle=False)
    X = cand["points_world_m"].copy()
    if not np.array_equal(cand["source_O_faces"], F):
        raise RuntimeError("start candidate connectivity differs from baseline")
    audit = json.load((BASE / "baseline/audit-159/report.json").open())
    seed_faces, initial_seed_faces, current_seed = current_seed_faces(F, audit, start_bone)

    # Build a small query ROI around both baseline and measured remaining exact
    # crossing faces. The compact spline support, not this ring boundary, controls
    # which source vertices may move.
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
    query_node_mask = np.zeros(len(X), dtype=bool)
    query_node_mask[query_nodes] = True
    query_faces = np.flatnonzero(np.any(query_node_mask[F], axis=1))
    sample_face, sample_bary, sample_is_seed = ENGINE.sample_layout(F, query_faces, seed_faces)
    sample_vertices = F[sample_face]
    sample0 = np.einsum("ij,ijk->ik", sample_bary, X[sample_vertices].astype(np.float64))
    h_mm = a.spacing_mm
    h = h_mm / 1000.0

    # Add controls exactly where the newly audited crossing faces require them;
    # the preceding composed maps remain in the prior flow index unchanged.
    _, keys = ENGINE.spatial_basis(sample0, h=h)
    Wstart, _ = ENGINE.spatial_basis(X.astype(np.float64), control_keys=keys, h=h)
    support_nodes = np.flatnonzero(np.asarray(Wstart.getnnz(axis=1)).ravel() > 0)
    support_mask = np.zeros(len(X), dtype=bool)
    support_mask[support_nodes] = True
    flow_faces = np.flatnonzero(np.any(support_mask[F], axis=1))
    D = ENGINE.build_difference_operator(keys)

    face_kind = []
    for bary in sample_bary:
        nz = int(np.count_nonzero(np.asarray(bary) > 1e-9))
        face_kind.append({1: "face_vertex", 2: "edge_probe", 3: "face_centroid"}[nz])
    face_kind = np.asarray(face_kind, dtype=object)

    helper = ENGINE.load_module("obstacle_bankhelper", BASE / "tools/human_skin_fat_bone_audit.py")
    *_, M, offset, position, _, _, _ = helper.read_bank_inputs(
        BASE / "inputs/bank-export-039/source-preflight", BASE / "tools"
    )
    invM = {int(k): np.linalg.inv(v) for k, v in M.items()}
    report = {
        "baseline_sha256": ENGINE.sha(BASE / "inputs/baseline-806.npz"),
        "start_candidate_sha256": start_hash,
        "remaining_exact_bone_report_sha256": digest(a.remaining_bone_report),
        "script_sha256": digest(__file__),
        "prior_flow_index_sha256": digest(a.prior_flow_index),
        "prior_flow_steps": prior_flows,
        "candidate_scope": "continued from trial-001; offline geometry reconstruction",
        "native_admission": False,
        "exact_zero_crossing_claim": False,
        "seed_face_sources": {
            "baseline_unique_faces": len(initial_seed_faces),
            "current_exact_unique_faces": len(current_seed),
            "union_unique_faces": len(seed_faces),
            "current_exact_pairs": sum((r.get("result") or {}).get("count", 0) for r in start_bone["banks"]),
        },
        "query_ring_count": a.rings,
        "query_vertex_count": int(len(query_nodes)),
        "query_face_count": int(len(query_faces)),
        "query_surface_sample_count": int(len(sample_face)),
        "flow_support_vertex_count": int(len(support_nodes)),
        "flow_support_face_count": int(len(flow_faces)),
        "control_count": int(len(keys)),
        "spacing_mm": h_mm,
        "steps": [],
        "limitations": [
            "sampled occupancy is not the exact fixed-bone crossing gate",
            "same-mesh self-intersection audit is separate",
            "named torso/back anatomy mask is unavailable in baseline-806; a distance-based remote-region proxy will be reported",
            "native physics admission remains false",
        ],
    }
    all_flows = list(prior_flows)
    status = "MAX_STEPS"
    previous_inside = None
    previous_max_depth = None
    stalls = 0
    previous_X = X.copy()

    for step in range(a.max_steps):
        if time.monotonic() - start >= a.max_seconds:
            status = "TIME_LIMIT"
            break
        step_started = time.monotonic()
        face_points = np.einsum(
            "ij,ijk->ik", sample_bary, X[sample_vertices].astype(np.float64)
        )
        query_points = np.vstack([face_points, X[support_nodes].astype(np.float64)]).astype(np.float32)
        query_kinds = np.r_[face_kind, np.full(len(support_nodes), "flow_support_vertex", dtype=object)]
        query_seed = np.r_[sample_is_seed, np.zeros(len(support_nodes), dtype=bool)]
        rows, query_files, free_bytes = query_verified(
            query_points, out / f"step-{step:03d}" / "query", timeout_s=60
        )
        inside = [r for r in rows if r["classification"] == "inside"]
        counts, seed_inside, source_inside, probe_inside = classify_witnesses(rows, query_kinds, query_seed)
        row_ids = np.asarray([int(r["query_row"]) for r in inside], dtype=np.int64)
        entry = {
            "step": step,
            "queried_face_witness_rows": int(len(face_points)),
            "queried_seed_crossing_face_rows": int(sample_is_seed.sum()),
            "queried_flow_support_vertices": int(len(support_nodes)),
            "inside_rows": int(len(inside)),
            "inside_seed_crossing_face_rows": int(seed_inside),
            "source_vertex_inside_rows": int(source_inside),
            "interior_probe_inside_rows": int(probe_inside),
            "inside_by_witness_type": dict(sorted(counts.items())),
            "inside_by_bank_body": dict(sorted(Counter(int(r["bank_body"]) for r in inside).items())),
            "max_residual_exit_depth_mm": None,
            "median_residual_exit_depth_mm": None,
            "query_files": query_files,
            "free_disk_bytes_after_query": int(free_bytes),
            "elapsed_query_s": time.monotonic() - step_started,
        }
        if not inside:
            entry["status"] = "NO_INSIDE_ROWS_IN_SAMPLED_WITNESSES"
            report["steps"].append(entry)
            status = entry["status"]
            break

        target_points, targets_mm, depths_mm = [], [], []
        for r in inside:
            body = int(r["bank_body"])
            p = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"])
            c = np.asarray([float(r[f"closest_{c}"]) for c in "xyz"])
            v = c - p
            dep = float(np.linalg.norm(v))
            if dep <= 1e-12:
                continue
            dw = (v * (1.0 + 0.00015 / dep)) @ invM[body].T
            target_points.append(query_points[int(r["query_row"])].astype(np.float64))
            targets_mm.append(dw * 1000.0)
            depths_mm.append(dep * 1000.0)
        if not targets_mm:
            entry["status"] = "NO_NONZERO_EXIT_VECTORS"
            report["steps"].append(entry)
            status = entry["status"]
            break
        target_points = np.asarray(target_points)
        targets_mm = np.asarray(targets_mm)
        depths_mm = np.asarray(depths_mm)
        max_depth = float(np.max(depths_mm))
        entry["max_residual_exit_depth_mm"] = max_depth
        entry["median_residual_exit_depth_mm"] = float(np.median(depths_mm))
        B, _ = ENGINE.spatial_basis(target_points, control_keys=keys, h=h)
        coverage = np.asarray(B.sum(axis=1)).ravel()
        entry["field_coverage_min"] = float(coverage.min())
        entry["unsupported_witness_rows"] = int(np.count_nonzero(coverage < 0.05))
        if np.any(coverage < 0.05):
            entry["status"] = "MEASURED_WITNESS_OUTSIDE_CURRENT_COMPACT_SUPPORT"
            report["steps"].append(entry)
            status = entry["status"]
            break

        controls_mm, cg_info = ENGINE.solve_velocity(B, D, targets_mm, maxiter=a.maxiter)
        entry["cg_info"] = cg_info
        residual = np.linalg.norm(B @ controls_mm - targets_mm, axis=1)
        entry["fit_residual_mm_mean"] = float(np.mean(residual))
        entry["fit_residual_mm_max"] = float(np.max(residual))
        entry["target_exit_depth_mm_max"] = max_depth
        entry["target_exit_depth_mm_median"] = float(np.median(depths_mm))
        L, max_adj = ENGINE.conservative_lipschitz(keys, controls_mm, h_mm)
        max_control = float(np.max(np.linalg.norm(controls_mm, axis=1)))
        Wnow, _ = ENGINE.spatial_basis(X.astype(np.float64), control_keys=keys, h=h)
        flow_mm = np.asarray(Wnow @ controls_mm)
        max_vertex_flow = float(np.max(np.linalg.norm(flow_mm[support_nodes], axis=1)))
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
        backtracks = []
        for backtrack in range(10):
            aa = alpha * (0.5 ** backtrack)
            Y, flow, area_old, area_new, ratio, good = ENGINE.apply_flow_step(
                X, F, support_nodes, flow_faces, Wnow, controls_mm, aa
            )
            backtracks.append({
                "alpha": float(aa),
                "pass_finite_nondegenerate": good,
                "min_flow_face_area_ratio": float(np.min(ratio)),
                "min_flow_face_area_m2": float(np.min(area_new)),
            })
            if good:
                accepted = (Y, flow, aa, area_new, ratio)
                break
        entry["area_backtracks"] = backtracks
        if accepted is None:
            entry["status"] = "FINITE_NONDEGENERATE_AREA_GATE"
            report["steps"].append(entry)
            status = entry["status"]
            break

        Y, flow, accepted_alpha, area_new, ratio = accepted
        outside_nodes = np.flatnonzero(~support_mask)
        outside_exact = bool(np.array_equal(Y[outside_nodes].view(np.uint32), X[outside_nodes].view(np.uint32)))
        if not outside_exact or not np.array_equal(F, source["source_O_faces"]):
            raise RuntimeError("flow modified outside support or changed source connectivity")
        per_step_mm = np.linalg.norm(Y.astype(np.float64) - X.astype(np.float64), axis=1) * 1000.0
        moved_bbox = Y[per_step_mm > 0].astype(np.float64) * 1000.0
        entry.update({
            "status": "APPLIED_AMBIENT_STEP",
            "accepted_alpha": float(accepted_alpha),
            "max_applied_vertex_step_mm": float(per_step_mm.max()),
            "moved_vertex_count": int(np.count_nonzero(per_step_mm > 0)),
            "outside_bitwise_unchanged": outside_exact,
            "source_faces_bitwise_unchanged": True,
            "min_flow_face_area_ratio": float(np.min(ratio)),
            "min_flow_face_area_m2": float(np.min(area_new)),
            "changed_vertex_bbox_mm": {
                "min": moved_bbox.min(axis=0).tolist(),
                "max": moved_bbox.max(axis=0).tolist(),
            },
            "elapsed_step_s": time.monotonic() - step_started,
        })
        X = Y
        step_file = out / f"flow-step-{step:03d}.npz"
        np.savez_compressed(
            step_file,
            control_keys=keys.astype(np.int32),
            coefficients_mm=controls_mm.astype(np.float32),
            alpha=np.float32(accepted_alpha),
            spacing_mm=np.float32(h_mm),
        )
        item = {"step": len(all_flows), "continuation_step": step, "path": str(step_file), "sha256": digest(step_file)}
        all_flows.append(item)
        entry["flow_step_file"] = str(step_file)
        report["steps"].append(entry)

        progress = (
            previous_inside is None
            or len(inside) < previous_inside
            or max_depth < previous_max_depth - 0.01
        )
        stalls = 0 if progress else stalls + 1
        entry["consecutive_stalled_steps"] = stalls
        previous_inside = len(inside)
        previous_max_depth = max_depth
        payload = {k: cand[k].copy() for k in cand.files}
        payload["points_world_m"] = X
        payload["ambient_flow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
        np.savez_compressed(out / f"checkpoint-{step:03d}.npz", **payload)
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: entry[k] for k in [
            "step", "status", "inside_rows", "source_vertex_inside_rows",
            "interior_probe_inside_rows", "inside_seed_crossing_face_rows",
            "max_residual_exit_depth_mm", "median_residual_exit_depth_mm",
            "fit_residual_mm_mean", "ambient_lipschitz_bound_per_step",
            "accepted_alpha", "max_applied_vertex_step_mm",
            "min_flow_face_area_ratio", "consecutive_stalled_steps",
            "elapsed_step_s",
        ]}), flush=True)
        if len(inside) == 0:
            status = "NO_INSIDE_ROWS_IN_SAMPLED_WITNESSES"
            break
        if stalls >= 4:
            status = "FOUR_STALLED_STEPS"
            break
        if time.monotonic() - start >= a.max_seconds:
            status = "TIME_LIMIT"
            break

    # Measure the final state once, unless the loop already ended on zero sampled hits.
    final_sampled = report["steps"][-1]["inside_rows"] if report["steps"] else None
    if final_sampled and time.monotonic() - start < a.max_seconds:
        face_points = np.einsum("ij,ijk->ik", sample_bary, X[sample_vertices].astype(np.float64))
        query_points = np.vstack([face_points, X[support_nodes].astype(np.float64)]).astype(np.float32)
        _, qmanifest, free_bytes = query_verified(query_points, out / "final-measurement" / "query", timeout_s=60)
        final_rows_path = out / "final-measurement/query/pairs.csv.gz"
        with gzip.open(final_rows_path, "rt") as fh:
            final_rows = list(csv.DictReader(fh))
        final_inside = [r for r in final_rows if r["classification"] == "inside"]
        counts, seed_inside, source_inside, probe_inside = classify_witnesses(final_rows, np.r_[face_kind, np.full(len(support_nodes), "flow_support_vertex", dtype=object)], np.r_[sample_is_seed, np.zeros(len(support_nodes), dtype=bool)])
        final_depths = []
        for r in final_inside:
            p = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"])
            c = np.asarray([float(r[f"closest_{c}"]) for c in "xyz"])
            final_depths.append(float(np.linalg.norm(c-p))*1000.0)
        report["final_sample_measurement"] = {
            "inside_rows": len(final_inside),
            "inside_seed_crossing_face_rows": seed_inside,
            "source_vertex_inside_rows": source_inside,
            "interior_probe_inside_rows": probe_inside,
            "inside_by_witness_type": dict(sorted(counts.items())),
            "max_residual_exit_depth_mm": max(final_depths) if final_depths else 0.0,
            "median_residual_exit_depth_mm": float(np.median(final_depths)) if final_depths else 0.0,
            "query_files": qmanifest,
            "free_disk_bytes_after_query": int(free_bytes),
        }
        if not final_inside:
            status = "NO_INSIDE_ROWS_IN_SAMPLED_WITNESSES"

    payload = {k: cand[k].copy() for k in cand.files}
    payload["points_world_m"] = X
    payload["ambient_flow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
    candidate = out / "candidate.npz"
    np.savez_compressed(candidate, **payload)
    (out / "full-flow-index.json").write_text(json.dumps(all_flows, indent=2) + "\n")

    changed = np.any(X.view(np.uint32) != X0.view(np.uint32), axis=1)
    changed_xyz = X[changed].astype(np.float64) * 1000.0
    baseline_seed_vertices = np.unique(F[seed_faces].ravel())
    distance_to_seed_m = cKDTree(X0[baseline_seed_vertices].astype(np.float64)).query(X0.astype(np.float64))[0]
    remote_mask = distance_to_seed_m > 0.100
    total_disp_mm = np.linalg.norm(X.astype(np.float64) - X0.astype(np.float64), axis=1) * 1000.0
    report.update({
        "status": status,
        "candidate_path": str(candidate),
        "candidate_sha256": digest(candidate),
        "source_faces_unchanged": bool(np.array_equal(F, payload["source_O_faces"])),
        "changed_vertex_count": int(changed.sum()),
        "changed_vertex_bbox_mm": {
            "min": changed_xyz.min(axis=0).tolist() if len(changed_xyz) else None,
            "max": changed_xyz.max(axis=0).tolist() if len(changed_xyz) else None,
        },
        "max_total_displacement_mm": float(total_disp_mm.max()),
        "torso_back_remote_proxy": {
            "definition": "baseline vertices more than 100 mm Euclidean distance from union audited seed-face vertices; coordinate proxy only, no named anatomical region mask in baseline-806",
            "vertex_count": int(remote_mask.sum()),
            "max_displacement_mm": float(total_disp_mm[remote_mask].max()) if np.any(remote_mask) else 0.0,
            "changed_vertex_count": int(np.count_nonzero(changed & remote_mask)),
        },
        "flow_steps_index": str(out / "full-flow-index.json"),
        "flow_step_count_total": len(all_flows),
        "elapsed_s": time.monotonic() - start,
    })
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
