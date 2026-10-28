#!/usr/bin/env python3
"""Read-only residual-direction report from a saved ambient-flow query."""
from pathlib import Path
import csv
import gzip
import importlib.util
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001")
D = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207")
HERE = Path(__file__).resolve().parent
RUN = HERE / "trial-001-local3mm"
QUERY = RUN / "final-measurement/query"
BONE_REPORT = HERE / "trial-001/ordered-verification/bone/report.json"
FINAL_CANDIDATE = RUN / "candidate.npz"
START_CANDIDATE = HERE / "trial-001-continuation/candidate.npz"
LAST_FLOW = RUN / "flow-step-015.npz"
OUT = RUN / "residual-direction-report.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    eng = module("ambient_flow_engine", HERE / "repair_ambient_flow.py")
    helper = module("ambient_bone_helper", ROOT / "tools/human_skin_fat_bone_audit.py")
    src = np.load(ROOT / "inputs/baseline-806.npz", allow_pickle=False)
    F = src["source_O_faces"]
    X0 = src["points_world_m"]
    audit = json.load((ROOT / "baseline/audit-159/report.json").open())
    current_bone = json.load(BONE_REPORT.open())
    seed_faces = sorted({
        int(p[0])
        for doc in (audit, current_bone)
        for bank in doc["banks"]
        for p in (bank.get("result") or {}).get("triangle_pairs", [])
    })

    edges = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(X0), len(X0)),
    )
    nodes = set(map(int, F[seed_faces].ravel()))
    front = set(nodes)
    for _ in range(3):
        new = set(map(int, adj[list(front)].indices)) - nodes
        nodes.update(new)
        front = new
    node_mask = np.zeros(len(X0), dtype=bool)
    node_mask[list(nodes)] = True
    faces = np.flatnonzero(np.any(node_mask[F], axis=1))
    sample_face, sample_bary, is_seed = eng.sample_layout(F, faces, seed_faces)
    face_sample_count = len(sample_face)

    final = np.load(FINAL_CANDIDATE, allow_pickle=False)
    X = final["points_world_m"]
    flow = np.load(LAST_FLOW, allow_pickle=False)
    keys = flow["control_keys"].astype(np.int64)
    h_mm = float(flow["spacing_mm"])
    h = h_mm / 1000.0
    start = np.load(START_CANDIDATE, allow_pickle=False)
    Wstart, _ = eng.spatial_basis(start["points_world_m"].astype(float), control_keys=keys, h=h)
    support = np.flatnonzero(np.asarray(Wstart.getnnz(axis=1)).ravel() > 0)

    _, _, _, _, M, _, _, _, _, _ = helper.read_bank_inputs(
        ROOT / "inputs/bank-export-039/source-preflight", ROOT / "tools"
    )
    invM = {int(k): np.linalg.inv(v) for k, v in M.items()}
    bank_meta = json.load(BONE_REPORT.open())
    stable_id = {int(r["body"]): int(r["stable_bone_id"]) for r in bank_meta["banks"]}

    face_normals_in = np.cross(
        X[F[:, 1]].astype(float) - X[F[:, 0]].astype(float),
        X[F[:, 2]].astype(float) - X[F[:, 0]].astype(float),
    )
    vertex_normals_out = np.zeros_like(X, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals_out, F[:, corner], face_normals_in)
    vertex_normals_out *= -1
    vertex_normals_out /= np.maximum(np.linalg.norm(vertex_normals_out, axis=1)[:, None], 1e-30)

    world_points = {}
    with gzip.open(QUERY / "points.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            world_points[int(r["query_row"])] = np.asarray([float(r[k]) for k in ("x_m", "y_m", "z_m")])

    witnesses = []
    with gzip.open(QUERY / "pairs.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["classification"] != "inside":
                continue
            row = int(r["query_row"])
            body = int(r["bank_body"])
            point = world_points[row]
            if row < face_sample_count:
                face = int(sample_face[row])
                bary = sample_bary[row]
                nz = int(np.count_nonzero(np.asarray(bary) > 1e-9))
                kind = {1: "face_vertex", 2: "edge_probe", 3: "centroid"}[nz]
                if nz == 1:
                    vertex = int(F[face, int(np.argmax(bary))])
                    normal = vertex_normals_out[vertex]
                    source_row = vertex
                else:
                    normal = -face_normals_in[face] / max(np.linalg.norm(face_normals_in[face]), 1e-30)
                    source_row = face
                seed = bool(is_seed[row])
            else:
                kind = "flow_support_vertex"
                vertex = int(support[row - face_sample_count])
                normal = vertex_normals_out[vertex]
                source_row = vertex
                seed = False

            p = np.asarray([float(r[f"query_local_{c}"]) for c in "xyz"])
            c = np.asarray([float(r[f"closest_{c}"]) for c in "xyz"])
            v = c - p
            local_depth = float(np.linalg.norm(v))
            if local_depth <= 1e-12:
                continue
            world_exit_m = v @ invM[body].T
            world_depth_mm = float(np.linalg.norm(world_exit_m) * 1000.0)
            target_mm = world_exit_m * (1.0 + 0.00015 / local_depth) * 1000.0
            target_hat = target_mm / max(np.linalg.norm(target_mm), 1e-30)
            witnesses.append({
                "query_row": row,
                "body": body,
                "stable_bone_id": stable_id.get(body),
                "kind": kind,
                "seed_crossing_face_witness": seed,
                "face_or_vertex_row": source_row,
                "point_world_mm": point * 1000.0,
                "target_mm": target_mm,
                "target_hat": target_hat,
                "world_depth_mm": world_depth_mm,
                "target_outward_normal_dot": float(np.dot(target_hat, normal)),
            })

    points = np.asarray([x["point_world_mm"] for x in witnesses]) / 1000.0
    targets = np.asarray([x["target_mm"] for x in witnesses])
    B, _ = eng.spatial_basis(points, control_keys=keys, h=h)
    Dmat = eng.build_difference_operator(keys)
    controls, cg_info = eng.solve_velocity(B, Dmat, targets, maxiter=80)
    fitted = B @ controls
    for i, w in enumerate(witnesses):
        resid = fitted[i] - targets[i]
        w["fit_signed_error_along_exit_mm"] = float(np.dot(resid, w["target_hat"]))
        w["fit_residual_norm_mm"] = float(np.linalg.norm(resid))
        w["fit_projection_along_exit_mm"] = float(np.dot(fitted[i], w["target_hat"]))
        w["point_world_mm"] = np.asarray(w["point_world_mm"]).round(4).tolist()
        w["target_mm"] = np.asarray(w["target_mm"]).round(6).tolist()
        w["target_hat"] = np.asarray(w["target_hat"]).round(8).tolist()

    grouped = defaultdict(list)
    for w in witnesses:
        grouped[w["body"]].append(w)
    by_body = []
    neighbor_opposition = {}
    for body, rows in grouped.items():
        depth = np.asarray([x["world_depth_mm"] for x in rows])
        cos = np.asarray([x["target_outward_normal_dot"] for x in rows])
        by_body.append({
            "body": body,
            "stable_bone_id": stable_id.get(body),
            "inside_rows": len(rows),
            "max_world_depth_mm": float(depth.max()),
            "median_world_depth_mm": float(np.median(depth)),
            "mean_target_outward_normal_dot": float(cos.mean()),
            "negative_outward_dot_rows": int(np.count_nonzero(cos < 0)),
        })
        P = np.asarray([x["point_world_mm"] for x in rows])
        U = np.asarray([x["target_hat"] for x in rows])
        tree = cKDTree(P)
        dots = []
        for i, neighbors in enumerate(tree.query_ball_point(P, 6.0)):
            for j in neighbors:
                if j > i:
                    dots.append(float(np.dot(U[i], U[j])))
        if dots:
            neighbor_opposition[str(body)] = {
                "pairs_within_6mm": len(dots),
                "negative_direction_pairs": int(sum(d < 0 for d in dots)),
                "minimum_direction_dot": float(min(dots)),
                "median_direction_dot": float(np.median(dots)),
            }
    by_body.sort(key=lambda x: -x["max_world_depth_mm"])
    deepest = sorted(witnesses, key=lambda x: -x["world_depth_mm"])[:20]
    for w in deepest:
        w["world_depth_mm"] = float(w["world_depth_mm"])
        w["target_outward_normal_dot"] = float(w["target_outward_normal_dot"])
        w["fit_signed_error_along_exit_mm"] = float(w["fit_signed_error_along_exit_mm"])
        w["fit_residual_norm_mm"] = float(w["fit_residual_norm_mm"])
        w["fit_projection_along_exit_mm"] = float(w["fit_projection_along_exit_mm"])

    out = {
        "candidate_path": str(FINAL_CANDIDATE),
        "candidate_sha256": __import__("hashlib").sha256(FINAL_CANDIDATE.read_bytes()).hexdigest(),
        "query_source": str(QUERY),
        "inside_rows": len(witnesses),
        "world_exit_depth_max_mm": max(w["world_depth_mm"] for w in witnesses),
        "world_exit_depth_median_mm": float(np.median([w["world_depth_mm"] for w in witnesses])),
        "rows_with_exit_against_outward_normal": int(sum(w["target_outward_normal_dot"] < 0 for w in witnesses)),
        "rows_with_exit_against_outward_normal_fraction": float(np.mean([w["target_outward_normal_dot"] < 0 for w in witnesses])),
        "current_outward_normal_rule": "baseline face winding is inward; face probes use negative current face cross-product; vertex probes use negative area-weighted incident-face normal",
        "row_fit": {
            "control_spacing_mm": h_mm,
            "fit_witness_count": len(witnesses),
            "cg_info_per_axis": cg_info,
            "signed_residual_definition": "dot(fitted_velocity - full_world_closest_exit_target, unit_exit_direction); negative means underfit along requested exit",
        },
        "per_body": by_body,
        "neighbor_exit_direction_opposition_within_6mm": neighbor_opposition,
        "deepest_witness_rows": deepest,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(str(OUT))
    print(json.dumps({
        "inside_rows": len(witnesses),
        "max_world_exit_depth_mm": out["world_exit_depth_max_mm"],
        "negative_normal_dot_rows": out["rows_with_exit_against_outward_normal"],
        "deepest_witness": {k: deepest[0][k] for k in [
            "body", "stable_bone_id", "query_row", "kind", "face_or_vertex_row",
            "point_world_mm", "world_depth_mm", "target_outward_normal_dot",
            "fit_signed_error_along_exit_mm", "fit_residual_norm_mm",
        ]},
    }, indent=2))


if __name__ == "__main__":
    main()
