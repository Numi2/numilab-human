#!/usr/bin/env python3
"""Bounded compact-support ambient flow for baseline-806.

Offline geometry experiment only. Mesh connectivity is never edited. Each
continuation step is one cubic B-spline ambient map, with a conservative
global derivative bound from adjacent control differences.
"""
from pathlib import Path
from itertools import product
from collections import Counter
import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001")
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def spatial_basis(points, control_keys=None, h=0.006):
    """Nonnegative C2 tensor-product cubic cardinal B-spline, compact support."""
    t = np.asarray(points, dtype=np.float64) / h
    cell = np.floor(t).astype(np.int64)
    u = t - cell
    coeff = np.stack([
        (1 - u) ** 3 / 6,
        (3 * u ** 3 - 6 * u * u + 4) / 6,
        (-3 * u ** 3 + 3 * u * u + 3 * u + 1) / 6,
        u ** 3 / 6,
    ], axis=2)
    offsets = np.asarray(list(product(range(4), repeat=3)), dtype=np.int64)
    keys_for_rows = cell[:, None, :] + offsets[None, :, :] - 1
    vals = np.prod(
        np.take_along_axis(coeff[:, None, :, :], offsets[None, :, :, None], axis=3)[:, :, :, 0],
        axis=2,
    )
    if control_keys is None:
        control_keys = np.unique(keys_for_rows.reshape(-1, 3), axis=0)
    lookup = {tuple(k): i for i, k in enumerate(control_keys)}
    columns = np.fromiter(
        (lookup.get(tuple(k), -1) for k in keys_for_rows.reshape(-1, 3)),
        dtype=np.int64,
        count=keys_for_rows.shape[0] * keys_for_rows.shape[1],
    ).reshape(keys_for_rows.shape[:2])
    rows = np.repeat(np.arange(len(t), dtype=np.int64), columns.shape[1])
    vals = vals.ravel()
    columns = columns.ravel()
    good = columns >= 0
    mat = sparse.csr_matrix(
        (vals[good], (rows[good], columns[good])),
        shape=(len(t), len(control_keys)),
    )
    return mat, control_keys


def query_native(points, folder):
    folder.mkdir(parents=True)
    pts = np.ascontiguousarray(points, dtype=np.float32)
    bits = pts.view(np.uint32)
    csv_path = folder / "points.csv"
    with csv_path.open("w") as f:
        f.write("query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32\n")
        for i, (v, b) in enumerate(zip(pts, bits)):
            f.write(",".join([str(i), *[format(float(x), ".9g") for x in v], *[str(int(x)) for x in b]]) + "\n")
    bank = ROOT / "inputs/bank-export-039/source-preflight"
    binary = ROOT / "tools/numi-matter-human-skin-fat-static-query"
    args = [
        str(binary),
        str(bank / "exact-emitted.triangles.csv"),
        str(bank / "exact-emitted.bodies.csv"),
        str(csv_path),
        str(folder / "pairs.csv"),
        str(folder / "nodes.csv"),
        "185",
    ]
    cp = subprocess.run(args, capture_output=True, text=True, timeout=300)
    (folder / "run.log").write_text(cp.stdout + cp.stderr)
    if cp.returncode:
        raise RuntimeError(cp.stdout + cp.stderr)
    rows = list(csv.DictReader((folder / "pairs.csv").open()))
    nodes = list(csv.DictReader((folder / "nodes.csv").open()))
    if len(nodes) != len(pts):
        raise RuntimeError(f"native query returned {len(nodes)} node rows for {len(pts)} points")
    if not all(int(r["ambiguous_banks"]) == 0 and int(r["invalid_banks"]) == 0 for r in nodes):
        raise RuntimeError("native query reported ambiguous or invalid bank state")
    subprocess.run(["gzip", "-1", str(csv_path), str(folder / "pairs.csv"), str(folder / "nodes.csv")], check=True)
    return rows


def sample_layout(F, active_faces, seed_faces):
    """Vertices, edge mids and centroids; audited crossing faces also get quarter-edge probes."""
    base = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
        [1/3, 1/3, 1/3],
    ]
    crossing = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [.75, .25, 0], [.5, .5, 0], [.25, .75, 0],
        [.75, 0, .25], [.5, 0, .5], [.25, 0, .75],
        [0, .75, .25], [0, .5, .5], [0, .25, .75],
        [1/3, 1/3, 1/3],
    ]
    seed_set = set(map(int, seed_faces))
    face_rows, bary_rows, seed_rows = [], [], []
    for face_id in active_faces:
        is_seed = int(face_id) in seed_set
        for bary in (crossing if is_seed else base):
            face_rows.append(int(face_id))
            bary_rows.append(bary)
            seed_rows.append(is_seed)
    return (
        np.asarray(face_rows, dtype=np.int64),
        np.asarray(bary_rows, dtype=np.float64),
        np.asarray(seed_rows, dtype=bool),
    )


def build_difference_operator(keys):
    """First differences; absent lattice neighbors are zero coefficients."""
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    rr, cc, vv = [], [], []
    row = 0
    for i, k0 in enumerate(keys):
        k = np.asarray(k0, dtype=np.int64)
        for axis in range(3):
            for sign in (-1, 1):
                kn = k.copy()
                kn[axis] += sign
                j = lookup.get(tuple(kn))
                if j is None:
                    rr.append(row); cc.append(i); vv.append(1.0)
                    row += 1
                elif sign > 0:
                    rr.extend((row, row)); cc.extend((i, j)); vv.extend((1.0, -1.0))
                    row += 1
    return sparse.csr_matrix((vv, (rr, cc)), shape=(row, len(keys)))


def conservative_lipschitz(keys, controls_mm, h_mm):
    """Bound ||grad u||_2 by sqrt(3) times the largest adjacent control difference / h."""
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    max_difference = 0.0
    for i, k0 in enumerate(keys):
        k = np.asarray(k0, dtype=np.int64)
        for axis in range(3):
            for sign in (-1, 1):
                kn = k.copy()
                kn[axis] += sign
                j = lookup.get(tuple(kn))
                diff = controls_mm[i] if j is None else controls_mm[i] - controls_mm[j]
                max_difference = max(max_difference, float(np.linalg.norm(diff)))
    return np.sqrt(3.0) * max_difference / h_mm, max_difference


def solve_velocity(B, D, targets_mm, maxiter=80):
    """Matrix-free regularized least squares; no dense RBF/normal matrix."""
    n = B.shape[1]
    observation_weight = 1.0
    ridge = 0.05
    smooth = 1.0
    diag = observation_weight * np.asarray(B.power(2).sum(axis=0)).ravel()
    if D.shape[0]:
        diag += smooth * np.asarray(D.power(2).sum(axis=0)).ravel()
    diag += ridge
    inv_diag = 1.0 / np.maximum(diag, 1e-12)
    pre = LinearOperator((n, n), matvec=lambda x: inv_diag * x, dtype=np.float64)

    def matvec(x):
        out = observation_weight * (B.T @ (B @ x)) + ridge * x
        if D.shape[0]:
            out = out + smooth * (D.T @ (D @ x))
        return np.asarray(out).ravel()

    A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    controls = np.zeros((n, 3), dtype=np.float64)
    infos = []
    for axis in range(3):
        rhs = observation_weight * np.asarray(B.T @ targets_mm[:, axis]).ravel()
        x, info = cg(A, rhs, M=pre, rtol=2e-4, atol=1e-8, maxiter=maxiter)
        controls[:, axis] = x
        infos.append(int(info))
    return controls, infos


def apply_flow_step(X, F, support_nodes, active_faces, Wnow, controls_mm, alpha, area_floor=1e-14):
    """Apply one and only one ambient map; test area without a source-normal gate."""
    flow = np.asarray(Wnow @ controls_mm)
    Y = X.copy()
    Y[support_nodes] = (
        X[support_nodes].astype(np.float64)
        + alpha * flow[support_nodes] / 1000.0
    ).astype(np.float32)
    tri_old = X[F[active_faces]].astype(np.float64)
    tri_new = Y[F[active_faces]].astype(np.float64)
    area_old = .5 * np.linalg.norm(
        np.cross(tri_old[:, 1] - tri_old[:, 0], tri_old[:, 2] - tri_old[:, 0]), axis=1
    )
    area_new = .5 * np.linalg.norm(
        np.cross(tri_new[:, 1] - tri_new[:, 0], tri_new[:, 2] - tri_new[:, 0]), axis=1
    )
    ratio = area_new / np.maximum(area_old, 1e-30)
    good = bool(
        np.isfinite(Y).all()
        and np.isfinite(area_new).all()
        and np.all(area_new > area_floor)
        and np.all(ratio > 1e-3)
    )
    return Y, flow, area_old, area_new, ratio, good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="trial-001")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--rings", type=int, default=3, help="query ROI around audited crossing faces")
    ap.add_argument("--spacing-mm", type=float, default=6.0)
    ap.add_argument("--max-step-mm", type=float, default=0.8)
    ap.add_argument("--lipschitz-step", type=float, default=0.35)
    ap.add_argument("--maxiter", type=int, default=80)
    a = ap.parse_args()

    out = OUT / a.run
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    base = ROOT / "inputs/baseline-806.npz"
    audit = ROOT / "baseline/audit-159/report.json"
    source = np.load(base, allow_pickle=False)
    payload = {k: source[k].copy() for k in source.files}
    X0 = payload["points_world_m"].copy()
    X = X0.copy()
    F = payload["source_O_faces"].copy()
    audit_doc = json.load(audit.open())
    triangle_pairs = [
        pair for bank in audit_doc["banks"]
        for pair in (bank.get("result") or {}).get("triangle_pairs", [])
    ]
    seed_faces = sorted({int(pair[0]) for pair in triangle_pairs})
    if not seed_faces:
        raise RuntimeError("baseline exact audit has no seed crossing faces")

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
    sample_face, sample_bary, sample_is_seed = sample_layout(F, query_faces, seed_faces)
    sample_vertices = F[sample_face]
    sample0 = np.einsum("ij,ijk->ik", sample_bary, X0[sample_vertices].astype(np.float64))
    h = a.spacing_mm / 1000.0

    # The queried source patch only seeds controls. The actual flow footprint is
    # the union of those controls' compact spline supports, which extends beyond
    # the query rings and moves every affected vertex coherently.
    _, control_keys = spatial_basis(sample0, h=h)
    W0, _ = spatial_basis(X0.astype(np.float64), control_keys=control_keys, h=h)
    support_nodes = np.flatnonzero(np.asarray(W0.getnnz(axis=1)).ravel() > 0)
    support_mask = np.zeros(len(X), dtype=bool)
    support_mask[support_nodes] = True
    flow_faces = np.flatnonzero(np.any(support_mask[F], axis=1))
    D = build_difference_operator(control_keys)

    helper = load_module("obstacle_bankhelper", ROOT / "tools/human_skin_fat_bone_audit.py")
    *_, M, offset, position, _, _, _ = helper.read_bank_inputs(
        ROOT / "inputs/bank-export-039/source-preflight", ROOT / "tools"
    )
    invM = {int(k): np.linalg.inv(v) for k, v in M.items()}
    report = {
        "baseline_sha256": sha(base),
        "audit_sha256": sha(audit),
        "script_sha256": sha(__file__),
        "host": "numi.local",
        "candidate_scope": "baseline-806 only; offline geometry reconstruction",
        "topology_method": "C2 compact-support ambient cubic B-spline flow; unchanged source_O_faces",
        "step_bound": "alpha * sqrt(3) * max adjacent control difference / grid spacing <= lipschitz_step; alpha * max coefficient norm <= max_step_mm",
        "outside_rule": "outside vertices have zero influence from every selected compact-support basis and are copied bitwise",
        "native_admission": False,
        "exact_zero_crossing_claim": False,
        "audit_triangle_pair_count": len(triangle_pairs),
        "seed_face_count": len(seed_faces),
        "seed_unique_vertex_count": int(len(np.unique(F[seed_faces]))),
        "query_ring_count": a.rings,
        "query_vertex_count": int(len(query_nodes)),
        "query_face_count": int(len(query_faces)),
        "query_surface_sample_count_per_iteration": int(len(sample_face)),
        "query_seed_edge_and_face_sample_count": int(sample_is_seed.sum()),
        "flow_support_vertex_count": int(len(support_nodes)),
        "flow_support_face_count": int(len(flow_faces)),
        "control_count": int(len(control_keys)),
        "spacing_mm": a.spacing_mm,
        "iterations": [],
        "limitations": [
            "inside counts cover vertices, edge points, centroids, plus quarter-edge samples on audited crossing faces",
            "the native exact fixed-bone intersection and same-mesh self-intersection checks remain separate gates",
            "the saved per-step spline controls are the ambient map for transferring the deformation",
            "offline geometry does not establish anatomical or native-physics acceptance",
        ],
    }
    flows = []
    status = "MAX_ITERATIONS"

    for iteration in range(a.iterations):
        iter_started = time.monotonic()
        # Query all triangle witnesses in the small ROI and every vertex touched
        # by the compact flow field; duplicate vertices are harmless witnesses.
        face_points = np.einsum(
            "ij,ijk->ik", sample_bary, X[sample_vertices].astype(np.float64)
        )
        query_points = np.vstack([face_points, X[support_nodes].astype(np.float64)]).astype(np.float32)
        is_seed_query = np.r_[sample_is_seed, np.zeros(len(support_nodes), dtype=bool)]
        native_rows = query_native(query_points, out / f"iteration-{iteration:03d}" / "query")
        inside = [r for r in native_rows if r["classification"] == "inside"]
        row_ids = np.asarray([int(r["query_row"]) for r in inside], dtype=np.int64)
        entry = {
            "iteration": iteration,
            "queried_face_witness_rows": int(len(face_points)),
            "queried_seed_crossing_face_rows": int(sample_is_seed.sum()),
            "queried_flow_support_vertices": int(len(support_nodes)),
            "inside_rows": int(len(inside)),
            "inside_unique_sample_points": int(len(np.unique(row_ids))),
            "inside_seed_crossing_face_rows": int(np.count_nonzero(is_seed_query[row_ids])) if len(row_ids) else 0,
            "inside_by_bank_body": dict(sorted(Counter(int(r["bank_body"]) for r in inside).items())),
            "elapsed_query_s": time.monotonic() - iter_started,
        }
        print(json.dumps(entry), flush=True)
        if not inside:
            entry["status"] = "NO_INSIDE_ROWS_IN_SAMPLED_WITNESSES"
            report["iterations"].append(entry)
            status = entry["status"]
            break

        target_points = []
        targets_mm = []
        depths_mm = []
        skipped_zero = 0
        for r in inside:
            body = int(r["bank_body"])
            p = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"])
            c = np.asarray([float(r[f"closest_{c}"]) for c in "xyz"])
            v = c - p
            dep = float(np.linalg.norm(v))
            if dep <= 1e-12:
                skipped_zero += 1
                continue
            dw = (v * (1.0 + 0.00015 / dep)) @ invM[body].T
            targets_mm.append(dw * 1000.0)
            target_points.append(query_points[int(r["query_row"])].astype(np.float64))
            depths_mm.append(dep * 1000.0)
        entry["zero_exit_vector_rows_skipped"] = skipped_zero
        if not targets_mm:
            entry["status"] = "NO_NONZERO_EXIT_VECTORS"
            report["iterations"].append(entry)
            status = entry["status"]
            break
        targets_mm = np.asarray(targets_mm, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        B, _ = spatial_basis(target_points, control_keys=control_keys, h=h)
        coverage = np.asarray(B.sum(axis=1)).ravel()
        entry["field_coverage_min"] = float(coverage.min())
        entry["field_coverage_median"] = float(np.median(coverage))
        entry["unsupported_exit_rows"] = int(np.count_nonzero(coverage < 0.05))
        if np.any(coverage < 0.05):
            entry["status"] = "MEASURED_WITNESS_OUTSIDE_CURRENT_COMPACT_SUPPORT"
            entry["unsupported_witness_points_mm"] = (target_points[coverage < 0.05] * 1000).tolist()[:32]
            report["iterations"].append(entry)
            status = entry["status"]
            break

        controls_mm, cg_info = solve_velocity(B, D, targets_mm, maxiter=a.maxiter)
        entry["cg_info"] = cg_info
        fit = B @ controls_mm
        residual = np.linalg.norm(fit - targets_mm, axis=1)
        entry["fit_residual_mm_mean"] = float(np.mean(residual))
        entry["fit_residual_mm_max"] = float(np.max(residual))
        entry["target_exit_depth_mm_max"] = float(np.max(depths_mm))
        entry["target_exit_depth_mm_median"] = float(np.median(depths_mm))

        L, max_adj = conservative_lipschitz(control_keys, controls_mm, a.spacing_mm)
        max_control = float(np.max(np.linalg.norm(controls_mm, axis=1)))
        Wnow, _ = spatial_basis(X.astype(np.float64), control_keys=control_keys, h=h)
        vertex_flow_mm = np.asarray(Wnow @ controls_mm)
        max_vertex_flow = float(np.max(np.linalg.norm(vertex_flow_mm[support_nodes], axis=1)))
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
        if not np.isfinite(alpha) or alpha <= 0:
            entry["status"] = "INVALID_AMBIENT_STEP"
            report["iterations"].append(entry)
            status = entry["status"]
            break

        current_outside = np.flatnonzero(~support_mask)
        old_support_bits = X[support_nodes].view(np.uint32).copy()
        accepted = None
        area_trials = []
        for backtrack in range(10):
            aa = alpha * (0.5 ** backtrack)
            Y, flow, old_area, new_area, ratio, good = apply_flow_step(
                X, F, support_nodes, flow_faces, Wnow, controls_mm, aa
            )
            area_trials.append({
                "alpha": float(aa),
                "pass_finite_nondegenerate": good,
                "min_flow_face_area_ratio": float(np.min(ratio)),
                "min_flow_face_area_m2": float(np.min(new_area)),
            })
            if good:
                accepted = (Y, flow, aa, old_area, new_area, ratio)
                break
        entry["area_backtracks"] = area_trials
        if accepted is None:
            entry["status"] = "FINITE_NONDEGENERATE_AREA_GATE"
            report["iterations"].append(entry)
            status = entry["status"]
            break

        Y, flow, accepted_alpha, old_area, new_area, ratio = accepted
        outside_exact = bool(np.array_equal(Y[current_outside].view(np.uint32), X[current_outside].view(np.uint32)))
        faces_exact = bool(np.array_equal(F, payload["source_O_faces"]))
        if not outside_exact or not faces_exact:
            raise RuntimeError("ambient flow changed vertices outside its compact support or altered connectivity")
        displacement = np.linalg.norm(Y.astype(np.float64) - X.astype(np.float64), axis=1) * 1000.0
        entry.update({
            "status": "APPLIED_AMBIENT_STEP",
            "accepted_alpha": float(accepted_alpha),
            "max_applied_vertex_step_mm": float(displacement.max()),
            "moved_vertex_count": int(np.count_nonzero(displacement > 0)),
            "outside_bitwise_unchanged": outside_exact,
            "source_faces_bitwise_unchanged": faces_exact,
            "min_flow_face_area_ratio": float(np.min(ratio)),
            "min_flow_face_area_m2": float(np.min(new_area)),
            "elapsed_iteration_s": time.monotonic() - iter_started,
        })
        X = Y
        flow_file = out / f"flow-step-{iteration:03d}.npz"
        np.savez_compressed(
            flow_file,
            control_keys=control_keys.astype(np.int32),
            coefficients_mm=controls_mm.astype(np.float32),
            alpha=np.float32(accepted_alpha),
            spacing_mm=np.float32(a.spacing_mm),
        )
        flows.append({"step": iteration, "path": str(flow_file), "sha256": sha(flow_file)})
        entry["flow_step_file"] = str(flow_file)
        report["iterations"].append(entry)
        payload["points_world_m"] = X
        payload["ambient_flow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
        np.savez_compressed(out / f"checkpoint-{iteration:03d}.npz", **payload)
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: entry[k] for k in [
            "iteration", "status", "inside_rows", "inside_seed_crossing_face_rows",
            "fit_residual_mm_mean", "target_exit_depth_mm_max",
            "ambient_lipschitz_bound_per_step", "accepted_alpha",
            "max_applied_vertex_step_mm", "moved_vertex_count",
            "outside_bitwise_unchanged", "min_flow_face_area_ratio",
            "elapsed_iteration_s",
        ]}), flush=True)

    payload["points_world_m"] = X
    payload["ambient_flow_030_delta_m"] = X.astype(np.float64) - X0.astype(np.float64)
    candidate = out / "candidate.npz"
    np.savez_compressed(candidate, **payload)
    (out / "flow-index.json").write_text(json.dumps(flows, indent=2) + "\n")
    report.update({
        "status": status,
        "candidate_path": str(candidate),
        "candidate_sha256": sha(candidate),
        "source_faces_unchanged": bool(np.array_equal(F, payload["source_O_faces"])),
        "outside_vertices_bitwise_unchanged": bool(np.array_equal(X[~support_mask].view(np.uint32), X0[~support_mask].view(np.uint32))),
        "changed_vertex_count": int(np.count_nonzero(np.any(X.view(np.uint32) != X0.view(np.uint32), axis=1))),
        "max_total_displacement_mm": float(np.max(np.linalg.norm(X.astype(np.float64) - X0.astype(np.float64), axis=1)) * 1000.0),
        "candidate_vertex_count": int(len(X)),
        "candidate_face_count": int(len(F)),
        "flow_steps": flows,
        "elapsed_s": time.monotonic() - started,
    })
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "iterations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
