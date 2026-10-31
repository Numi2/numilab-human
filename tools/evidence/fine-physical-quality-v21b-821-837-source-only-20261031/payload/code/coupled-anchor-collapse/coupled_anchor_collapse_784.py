#!/usr/bin/env python3
"""784: I-only bounded inner-layer topology-repair preflight/runner.

Imports frozen 783 (and therefore frozen 782/781). Current inner-back I rows
are eligible for the I edge roster, while actual I rim and 149 route anchors
remain fixed. Each fitted anchor is limited to 50 micrometres from its current
coordinate and from the authenticated 805 baseline. Preparation never starts
the solver; root owns any later candidate run.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "coupled_anchor_collapse_783.py"
BASE_SHA256 = "b78358fccf208a12a24b229608c10c231817a8f8c15e4e4c0da45f9352585841"
INNER_CAP_M = 5.0e-5
EDGE_CAP_M = 5.0e-3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_783():
    if sha(BASE) != BASE_SHA256:
        raise RuntimeError("frozen 783 operation source changed")
    spec = importlib.util.spec_from_file_location("coupled_anchor_collapse_783_for_784", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen 783")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V783 = import_783()
G = V783.G
M = V783.M
BASE_PREPARE = M.prepare_operation
BASELINE_I_POINTS = None


def _load(entry, label):
    return M.load_pinned_npz(entry, "784 " + label)


def _row_distance(a, b, rows=None):
    aa = np.asarray(a, np.float32)
    bb = np.asarray(b, np.float32)
    if aa.shape != bb.shape:
        raise ValueError("point coordinate domains differ")
    rr = np.arange(len(aa), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    return np.linalg.norm(aa[rr].astype(np.float64) - bb[rr].astype(np.float64), axis=1)


def _prepare_operation_784(surface, volumes, keep, drop, *, protected_surface_rows=(),
                           route_anchor_rows=(), move_cap_m=INNER_CAP_M):
    """Reuse 783's complete-star fit, then enforce cumulative 805 I displacement."""
    if BASELINE_I_POINTS is None:
        return {"status": "REJECT_784_POLICY_NOT_BOUND"}
    points = np.asarray(surface["points_world_m_f32"], np.float32)
    if points.shape != BASELINE_I_POINTS.shape:
        return {"status": "REJECT_784_BASELINE_POINT_DOMAIN"}
    result = BASE_PREPARE(surface, volumes, keep, drop,
        protected_surface_rows=protected_surface_rows,
        route_anchor_rows=route_anchor_rows,
        move_cap_m=min(float(move_cap_m), INNER_CAP_M))
    if result.get("status") != "PASS":
        return result
    keep = int(keep)
    anchor = np.asarray(result["anchor_f32"], np.float32).reshape(3)
    cumulative = float(np.linalg.norm(anchor.astype(np.float64) -
                                      BASELINE_I_POINTS[keep].astype(np.float64)))
    if cumulative > INNER_CAP_M:
        return {"status": "REJECT_805_CUMULATIVE_I_POINT_CAP",
                "keep_row": keep, "baseline_displacement_m": cumulative,
                "maximum_displacement_m": INNER_CAP_M,
                "step_displacement_m": float(result["fit"]["anchor_displacement_m"])}
    result["inner_layer_topology_repair"] = True
    result["inner_back_layer_repair"] = True
    result["anchor_displacement_from_805_baseline_m"] = cumulative
    result["maximum_anchor_displacement_from_805_baseline_m"] = INNER_CAP_M
    result["current_back_rows_allowed_for_topology"] = True
    return result


M.prepare_operation = _prepare_operation_784


def state_config_784(state_path, out_dir, *, max_operations=32, time_limit_s=90,
                     max_edge_length_m=EDGE_CAP_M, max_edges=100000):
    """Build a measured edge roster; does not invoke the operation solver."""
    global BASELINE_I_POINTS
    state_path = Path(state_path).resolve()
    out = Path(out_dir).resolve()
    state = json.loads(state_path.read_text())
    for key in ("source_surfaces", "Skin_current", "Fat_current", "exact_context"):
        if key not in state:
            raise ValueError("verified state is missing " + key)
    if state.get("physical_admission") is not False or int(state.get("native_steps", 0)) != 0:
        raise ValueError("784 accepts construction-only verified states")

    proof = G._source_evidence(state_path, state)
    mode = proof.get("audit_implementation", {}).get("mode")
    if mode != "AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_805":
        raise ValueError("784 requires a current state authenticated under 805 inner-layer policy")
    source_cfg_ref = proof.get("config")
    if not isinstance(source_cfg_ref, dict) or sha(source_cfg_ref.get("path", "")) != source_cfg_ref.get("sha256"):
        raise ValueError("807 proof lacks a valid 805 config pin")
    source_cfg_path = Path(source_cfg_ref["path"]).resolve()
    source_cfg = json.loads(source_cfg_path.read_text())
    baseline_refs = source_cfg.get("baseline_candidates")
    if not isinstance(baseline_refs, dict) or set(baseline_refs) != {"O", "I", "D"}:
        raise ValueError("805 config must pin O/I/D baselines")
    bounded = source_cfg.get("bounded_point_repair", {})
    for role in ("I", "D"):
        item = bounded.get(role, {})
        if item.get("inner_back_layer_repair") is not True or float(item.get("maximum_displacement_m", -1)) > INNER_CAP_M:
            raise ValueError("current 805 baseline does not grant 50 micrometre inner-layer movement")

    surfaces = state["source_surfaces"]
    if set(surfaces) != {"O", "I", "D"}:
        raise ValueError("current state must pin O/I/D")
    current = {r: _load(surfaces[r], "current " + r) for r in ("O", "I", "D")}
    baseline = {r: _load(baseline_refs[r], "805 baseline " + r) for r in ("O", "I", "D")}
    # 805 keeps the complete outer O geometry fixed.
    for lane in ("points_world_m_f32", "points_world_m", "faces"):
        if lane in current["O"] or lane in baseline["O"]:
            if lane not in current["O"] or lane not in baseline["O"] or not M._array_equal(current["O"][lane], baseline["O"][lane]):
                raise ValueError("current O differs from 805 fixed outer source")

    ip = np.asarray(current["I"]["points_world_m_f32"], np.float32)
    faces = np.asarray(current["I"]["faces"], np.int64)
    base_i = np.asarray(baseline["I"]["points_world_m_f32"], np.float32)
    if ip.shape != base_i.shape:
        raise ValueError("805 I point row identity is not preserved")
    prior_i_disp = _row_distance(ip, base_i)
    if np.any(prior_i_disp > INNER_CAP_M):
        bad = np.flatnonzero(prior_i_disp > INNER_CAP_M)
        raise ValueError(f"current I already exceeds 805 cap at rows {bad[:16].tolist()}")
    BASELINE_I_POINTS = base_i.copy()
    if not np.isfinite(max_edge_length_m) or max_edge_length_m <= 0 or max_edge_length_m > EDGE_CAP_M:
        raise ValueError("edge cap must be positive and no greater than 5 mm")
    if int(max_edges) < 1:
        raise ValueError("max_edges must be positive")

    scope = _load({"path": str(M.BACK_SCOPE_PATH), "sha256": M.BACK_SCOPE_SHA}, "back412 scope")
    back_rows = np.asarray(scope["I_vertex_rows"], np.int64)
    if np.any(back_rows < 0) or np.any(back_rows >= len(ip)):
        raise ValueError("back412 I rows are outside the current point domain")
    back_disp = _row_distance(ip, base_i, back_rows)
    if np.any(back_disp > INNER_CAP_M):
        raise ValueError("current inner-back I points exceed the 805 cap")

    # Current inner-back rows are intentionally NOT protected. Only routes and
    # actual source-rim vertices remain fixed in the collapse star.
    route_records = json.loads(M.A.PROT.read_text()).get("surface_routes")
    if not isinstance(route_records, list) or len(route_records) != 149:
        raise ValueError("pinned 149 route inventory is malformed")
    coordmap = M._f32_coord_row_map(ip)
    route_rows = []
    for rec in route_records:
        if rec.get("role") != "I":
            continue
        key = M.f32key(np.asarray(rec["target_world_m_f32"], np.float32))
        if key not in coordmap:
            raise ValueError("registered I route anchor is absent")
        route_rows.append(coordmap[key])
    route_rows = np.asarray(sorted(set(map(int, route_rows))), np.int64)
    es = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1), axis=1)
    surface_edges, edge_counts = np.unique(es, axis=0, return_counts=True)
    rim_rows = np.unique(surface_edges[edge_counts == 1].ravel()).astype(np.int64)
    protected = np.asarray(sorted(set(map(int, route_rows)) | set(map(int, rim_rows))), np.int64)
    protected_set = set(map(int, protected))

    interior_edges = surface_edges[edge_counts == 2]
    edge_lookup = {tuple(map(int, e)): i for i, e in enumerate(interior_edges)}
    score = np.zeros(len(interior_edges), np.int64)
    role_recs = {}
    crosswalk_recs = {}
    for role in ("Skin", "Fat"):
        vol_ref = state[role + "_current"]
        volume = _load(vol_ref, role + " volume")
        vp = np.asarray(volume["points_world_m"], np.float32)
        tt = np.asarray(volume["tetrahedra"], np.int64)
        det, q = M.measurements(vp, tt)
        bad = ~M.M.passes(det, q)
        cross, cwrec = G.exact_used_row_crosswalk(ip, faces, vp, role)
        crosswalk_recs[role] = cwrec
        inv = np.full(len(vp), -1, np.int64)
        used_i_rows = np.unique(faces.reshape(-1))
        inv[cross[used_i_rows]] = used_i_rows
        counts = {}
        bad_tets = tt[bad]
        for a, b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            rows = bad_tets[:, (a,b)]
            sa, sb = inv[rows[:,0]], inv[rows[:,1]]
            ok = (sa >= 0) & (sb >= 0) & (sa != sb)
            if not np.any(ok):
                continue
            pairs = np.sort(np.column_stack((sa[ok], sb[ok])), axis=1)
            for x, y in pairs:
                ei = edge_lookup.get((int(x), int(y)))
                if ei is not None:
                    counts[ei] = counts.get(ei, 0) + 1
        for ei, n in counts.items():
            score[ei] += int(n)
        role_recs[role] = {"path": vol_ref["path"], "sha256": vol_ref["sha256"],
            "cells": int(len(tt)), "native_bad": int(np.count_nonzero(bad)),
            "face_referenced_surface_rows": int(cwrec["face_referenced_surface_rows"])}

    lengths = np.linalg.norm(ip[interior_edges[:,0]].astype(np.float64) -
                             ip[interior_edges[:,1]].astype(np.float64), axis=1)
    valid = (score > 0) & (lengths <= float(max_edge_length_m))
    eligible = []
    for ei in np.flatnonzero(valid):
        k, d = map(int, interior_edges[ei])
        fr = np.flatnonzero(np.any((faces == k) | (faces == d), axis=1))
        star_vertices = set(map(int, faces[fr].ravel()))
        if star_vertices & protected_set:
            continue
        eligible.append((int(ei), k, d, int(len(fr))))
    eligible.sort(key=lambda z: (-int(score[z[0]]), float(lengths[z[0]]), z[1], z[2]))
    eligible = eligible[:int(max_edges)]
    undirected = np.asarray([[k, d] for _, k, d, _ in eligible], np.int64).reshape((-1, 2))
    orientations = np.asarray([(k, d) for _, k, d, _ in eligible] +
                              [(d, k) for _, k, d, _ in eligible], np.int64).reshape((-1, 2))
    keep_rows = orientations[:, 0]
    keep_baseline_disp = prior_i_disp[keep_rows]
    keep_headroom = np.maximum(0.0, INNER_CAP_M - keep_baseline_disp)
    keep_is_back = np.isin(keep_rows, back_rows)

    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    edge_path = inputs / "candidate-I-edges.npz"
    edge_len = np.asarray([lengths[ei] for ei, *_ in eligible], np.float64)
    edge_score = np.asarray([score[ei] for ei, *_ in eligible], np.int64)
    np.savez_compressed(edge_path,
        candidate_edges=orientations,
        edge_length_m=np.repeat(edge_len, 2),
        bad_tet_incidence=np.repeat(edge_score, 2),
        undirected_edges=undirected,
        keep_row_805_baseline_displacement_m=keep_baseline_disp,
        keep_row_remaining_805_headroom_m=keep_headroom,
        keep_row_is_authenticated_inner_back=keep_is_back)
    protection_path = inputs / "protected-I-rows.npz"
    np.savez_compressed(protection_path, protected_surface_rows=protected,
        route_anchor_rows=route_rows, source_rim_rows=rim_rows,
        authenticated_inner_back_rows=back_rows,
        authenticated_inner_back_rows_candidate_eligible=np.asarray([True]))
    cfg = {"schema":"h706.coupled-anchor-collapse-run-config.v1",
        "surface_I":surfaces["I"], "baseline_surfaces":surfaces,
        "volumes":{"Skin":state["Skin_current"],"Fat":state["Fat_current"]},
        "candidate_edges":{"path":str(edge_path),"sha256":sha(edge_path),"key":"candidate_edges"},
        "protected_surface_rows":{"path":str(protection_path),"sha256":sha(protection_path),"key":"protected_surface_rows"},
        "route_anchor_rows":{"path":str(protection_path),"sha256":sha(protection_path),"key":"route_anchor_rows"},
        "move_cap_m":INNER_CAP_M,"max_operations":int(max_operations),"time_limit_s":float(time_limit_s),
        "inner_layer_topology_repair":{"enabled":True,"role":"I","maximum_displacement_m":INNER_CAP_M,
            "audit":"805","back_rows_are_candidate_eligible":True,"rim_rows_fixed":True,
            "149_routes_fixed":True,"outer_O_fixed":True,"D_source_unchanged":True},
        "inner_layer_audit_baseline_candidates":baseline_refs,
        "inner_layer_source_proof":{"path":str(Path(proof["state"]["path"]).resolve()),
            "sha256":proof["state"]["sha256"],"mode":mode}}
    cfg_path = inputs / "run-config.json"
    M._write_json(cfg_path, cfg)

    base_d = np.asarray(baseline["D"]["points_world_m_f32"], np.float32)
    curr_d = np.asarray(current["D"]["points_world_m_f32"], np.float32)
    d_back = _row_distance(curr_d, base_d, np.asarray(scope["D_vertex_rows"], np.int64))
    core_edges = np.any(np.isin(undirected, back_rows), axis=1)
    prep = {"schema":"h706.coupled-anchor-collapse-inner-layer-preparation.v1",
        "status":"PASS_PREFLIGHT_ONLY_NO_SOLVER",
        "state":{"path":str(state_path),"sha256":sha(state_path)},
        "source_807":{"path":str(Path(proof["state"]["path"]).resolve()),"sha256":proof["state"]["sha256"],"mode":mode},
        "prior_805_context":{"path":str(Path(state["exact_context"]["path"]).resolve()),
            "sha256":state["exact_context"]["sha256"],
            "status":json.loads(Path(state["exact_context"]["path"]).read_text()).get("status")},
        "current_surface_pins":surfaces,"current_volume_inputs":role_recs,
        "current_inner_back_policy":{"authenticated_I_rows":int(len(back_rows)),
            "rows_candidate_eligible":True,
            "current_I_max_back_displacement_from_805_baseline_m":float(np.max(back_disp)) if len(back_disp) else 0.0,
            "current_D_max_back_displacement_from_805_baseline_m":float(np.max(d_back)) if len(d_back) else 0.0,
            "candidate_I_anchor_cap_from_805_baseline_m":INNER_CAP_M},
        "protected_surface_rows":{"route_anchors":int(len(route_rows)),"source_rim_vertices":int(len(rim_rows)),
            "union_count":int(len(protected)),
            "back_rows_in_protected_union":int(len(set(map(int,back_rows)) & protected_set))},
        "edge_roster":{"interior_edges":int(len(interior_edges)),
            "bad_incident_before_length_and_star_protection":int(np.count_nonzero(valid)),
            "eligible_after_route_rim_fullstar_protection":int(len(eligible)),
            "eligible_edges_touching_inner_back_rows":int(np.count_nonzero(core_edges)),
            "eligible_orientations":int(len(orientations)),
            "edge_length_cap_m":float(max_edge_length_m),
            "inner_back_orientations":int(np.count_nonzero(keep_is_back)),
            "inner_back_keep_orientation_mean_remaining_headroom_m":float(np.mean(keep_headroom[keep_is_back])) if np.any(keep_is_back) else None,
            "edge_path":{"path":str(edge_path),"sha256":sha(edge_path)}},
        "protection_input":{"path":str(protection_path),"sha256":sha(protection_path)},
        "run_config":{"path":str(cfg_path),"sha256":sha(cfg_path)},
        "acceptance_contract":{"full_keep_and_drop_stars":True,"all_native_and_exact_positive":True,
            "exact_paired_volume_delta_integer":"0","local_volume_budget_fraction":0.01,
            "regional_volume_budget_fraction":0.0001,"anchor_step_cap_m":INNER_CAP_M,
            "cumulative_I_baseline_cap_m":INNER_CAP_M,"O_source_unchanged":True,
            "D_source_unchanged_in_this_increment":True,"149_routes_fixed":True,"I_rim_fixed":True,
            "805_audit_required":True,"775_not_used_as_final_source_audit":True},
        "candidate_solver":"NOT_STARTED","physical_admission":False,"native_steps":0,
        "mass_reconciliation":"PENDING"}
    M._write_json(inputs / "state-preparation.json", prep)
    return cfg_path, prep


def main():
    ap = argparse.ArgumentParser(description="784 I-only inner-layer topology-repair preflight")
    ap.add_argument("--mode", choices=("prepare-state",), default="prepare-state")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inner-layer-topology-repair", action="store_true", required=True)
    ap.add_argument("--max-operations", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--max-edge-length-m", type=float, default=EDGE_CAP_M)
    ap.add_argument("--max-edges", type=int, default=100000)
    a = ap.parse_args()
    cfg, prep = state_config_784(a.state, a.output,
        max_operations=a.max_operations, time_limit_s=a.seconds,
        max_edge_length_m=a.max_edge_length_m, max_edges=a.max_edges)
    prep_path = cfg.parent / "state-preparation.json"
    print(json.dumps({"status":prep["status"],
        "preparation":{"path":str(prep_path),"sha256":sha(prep_path)},
        "run_config":{"path":str(cfg),"sha256":sha(cfg)},
        "candidate_solver":"NOT_STARTED","physical_admission":False,"native_steps":0},
        indent=2, default=str))


if __name__ == "__main__":
    main()

