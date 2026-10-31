#!/usr/bin/env python3
"""State wrapper for coupled I-anchor collapse under verified 806 source evidence.

This version leaves 781 and its fixtures immutable. It validates the incoming
verified state with the pinned 806 recursive source-evidence verifier, keeps
current back/rim/route vertices out of collapse stars, and requires exact
surface-to-volume crosswalks only for face-referenced I rows. A missing mapping
is allowed only for a source point row unused by every I face.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMPL = HERE / "coupled_anchor_collapse_781.py"
IMPL_SHA256 = "739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb"
SOURCE_EVIDENCE = ROOT / "verify_current_source_evidence_807_root.py"
SOURCE_EVIDENCE_SHA256 = "9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62"
SOURCE_EVIDENCE_BASE = ROOT / "verify_current_source_evidence_807_validation_root.py"
SOURCE_EVIDENCE_BASE_SHA256 = "4769a77867e10fd86bedfe70b077b5c12eb5b0cf7bde6cec9fbbc84a6bbdb2d1"
MAX_SOURCE_I_BACK_MOVE_M = 5.0e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def import_pinned(path, expected, name):
    if not Path(path).is_file() or sha(path) != expected:
        raise RuntimeError(f"pinned helper changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = import_pinned(IMPL, IMPL_SHA256, "coupled_anchor_collapse_781_for_782")
E806 = import_pinned(SOURCE_EVIDENCE, SOURCE_EVIDENCE_SHA256, "source_evidence_806_for_782")


def _same_pin(a, b):
    return (isinstance(a, dict) and isinstance(b, dict) and
            Path(str(a.get("path", ""))).resolve() == Path(str(b.get("path", ""))).resolve() and
            str(a.get("sha256", "")) == str(b.get("sha256", "")))


def _coord_row_map(points):
    out = {}
    for row, xyz in enumerate(np.asarray(points, dtype=np.float32)):
        k = np.ascontiguousarray(xyz).view(np.uint32).tobytes()
        if k in out:
            raise ValueError("current I surface has duplicate F32 coordinate rows")
        out[k] = row
    return out


def exact_used_row_crosswalk(surface_points, surface_faces, volume_points, role):
    """Return surface-row -> volume-row map; allow -1 only on unused surface rows."""
    sp = np.asarray(surface_points, dtype=np.float32)
    sf = np.asarray(surface_faces, dtype=np.int64)
    vp = np.asarray(volume_points, dtype=np.float32)
    if sf.ndim != 2 or sf.shape[1] != 3 or (sf.size and (sf.min() < 0 or sf.max() >= len(sp))):
        raise ValueError(f"{role}: malformed source faces")
    cross = M.coord_crosswalk(sp, vp)
    if cross is None:
        raise ValueError(f"{role}: volume F32 coordinate keys are not unique")
    used = np.unique(sf.reshape(-1))
    if np.any(cross[used] < 0):
        missing = used[cross[used] < 0]
        raise ValueError(f"{role}: face-referenced I rows lack exact volume coordinate matches: {missing[:16].tolist()}")
    mapped = cross[used]
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError(f"{role}: face-referenced I rows do not map injectively")
    unused = np.setdiff1d(np.arange(len(sp), dtype=np.int64), used, assume_unique=True)
    missing_unused = unused[cross[unused] < 0]
    return cross, {"face_referenced_surface_rows": int(len(used)),
                   "unused_surface_rows": int(len(unused)),
                   "unmapped_unused_surface_rows": missing_unused.astype(np.int64).tolist(),
                   "used_rows_exact_unique": True}


def _source_evidence(state_path, state):
    proof = E806.verify_current_source_evidence(state_path)
    if not str(proof.get("status", "")).startswith("PASS_"):
        raise ValueError("806 source-evidence verifier did not pass")
    if proof.get("physical_admission") is not False or int(proof.get("native_steps", -1)) != 0:
        raise ValueError("source evidence unexpectedly carries physical/native admission")
    pins = proof.get("source_surfaces")
    if not isinstance(pins, dict) or set(pins) != {"O", "I", "D"}:
        raise ValueError("806 proof lacks complete O/I/D source pins")
    declared = state.get("source_surfaces")
    if not isinstance(declared, dict) or set(declared) != {"O", "I", "D"}:
        raise ValueError("verified state lacks current O/I/D source pins")
    for role in ("O", "I", "D"):
        if not _same_pin(pins[role], declared[role]):
            raise ValueError(f"806 source proof is not bound to current {role}")
    movement = proof.get("movement_by_role", {})
    # This workflow changes only I. The current outer O geometry must remain fixed.
    if int(movement.get("O", {}).get("moved_points", 0)) != 0:
        raise ValueError("806 source proof does not show a fixed outer O surface")
    # 807 validates either ordinary 775 movement (with the ordinary 0.5 mm
    # policy) or the explicit 805 inner-layer policy. The stricter 50 micrometre
    # check below applies to designated inner-back rows, not all I/D movement.
    return proof


def state_config(state_path, out_dir, *, move_cap_m=0.0005, max_operations=32,
                 time_limit_s=90, max_edge_length_m=0.0005, max_edges=100000):
    state_path = Path(state_path).resolve()
    out = Path(out_dir).resolve()
    state = json.loads(state_path.read_text())
    for key in ("source_surfaces", "Skin_current", "Fat_current"):
        if key not in state:
            raise ValueError("verified state is missing " + key)
    if state.get("physical_admission") is not False or int(state.get("native_steps", 0)) != 0:
        raise ValueError("state wrapper accepts construction-only verified states")
    proof = _source_evidence(state_path, state)

    surfaces = state["source_surfaces"]
    cfg_ref = proof.get("config")
    if cfg_ref is None and isinstance(proof.get("anchor"), dict):
        cfg_ref = proof["anchor"].get("config")
    if not isinstance(cfg_ref, dict) or sha(cfg_ref.get("path", "")) != cfg_ref.get("sha256"):
        raise ValueError("807 proof lacks a readable, hash-pinned source-boundary config")
    source_cfg = json.loads(Path(cfg_ref["path"]).read_text())
    baseline_refs = source_cfg.get("baseline_candidates")
    if not isinstance(baseline_refs, dict) or set(baseline_refs) != {"O", "I", "D"}:
        raise ValueError("807 source config lacks baseline O/I/D pins")
    current_role = {r:M.load_pinned_npz(surfaces[r], "current " + r) for r in ("O", "I", "D")}
    base_role = {r:M.load_pinned_npz(baseline_refs[r], "807 baseline " + r) for r in ("O", "I", "D")}
    # This workstream holds the complete outer O point geometry fixed.
    for lane in ("points_world_m_f32", "points_world_m"):
        if lane in current_role["O"] and lane in base_role["O"]:
            ca, ba = np.asarray(current_role["O"][lane]), np.asarray(base_role["O"][lane])
            if ca.shape != ba.shape or ca.dtype != ba.dtype or ca.tobytes(order="C") != ba.tobytes(order="C"):
                raise ValueError("outer O point bytes changed in the current source state")
    current_i = current_role["I"]
    ip = np.asarray(current_i["points_world_m_f32"], np.float32)
    faces = np.asarray(current_i["faces"], np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("current I source faces are malformed")
    if float(move_cap_m) <= 0 or float(move_cap_m) > 0.0005:
        raise ValueError("candidate anchor movement exceeds the unchanged 0.5 mm construction cap")

    # Keep authenticated back rows protected as operation endpoints/stars, but do
    # not compare their current coordinates to the pre-805 reference surface.
    back = M.load_pinned_npz({"path": str(M.BACK_SCOPE_PATH), "sha256": M.BACK_SCOPE_SHA}, "back412 scope")
    back_rows = np.asarray(back["I_vertex_rows"], np.int64)
    back_d_rows = np.asarray(back["D_vertex_rows"], np.int64)
    core_motion = {}
    for role, rows in (("I", back_rows), ("D", back_d_rows)):
        cp = np.asarray(current_role[role]["points_world_m_f32"], np.float32)
        bp = np.asarray(base_role[role]["points_world_m_f32"], np.float32)
        if cp.shape != bp.shape or np.any(rows < 0) or np.any(rows >= len(cp)):
            raise ValueError(f"current {role} row domain does not contain pinned inner-back rows")
        d = np.linalg.norm(cp[rows].astype(np.float64) - bp[rows].astype(np.float64), axis=1)
        max_move = float(np.max(d)) if len(d) else 0.0
        if "points_world_m" in current_role[role] and "points_world_m" in base_role[role]:
            cfull = np.asarray(current_role[role]["points_world_m"])
            bfull = np.asarray(base_role[role]["points_world_m"])
            if cfull.shape != bfull.shape or cfull.dtype != bfull.dtype:
                raise ValueError(f"current {role} full-precision point domain differs from 807 baseline")
            dfull = np.linalg.norm(cfull[rows].astype(np.float64) - bfull[rows].astype(np.float64), axis=1)
            max_move = max(max_move, float(np.max(dfull)) if len(dfull) else 0.0)
        if max_move > MAX_SOURCE_I_BACK_MOVE_M:
            raise ValueError(f"current {role} inner-back row movement exceeds 50 micrometres")
        core_motion[role] = {"rows":int(len(rows)),"maximum_displacement_m":max_move,
                             "limit_m":MAX_SOURCE_I_BACK_MOVE_M,"within_limit":True}
    if np.any(back_rows < 0) or np.any(back_rows >= len(ip)):
        raise ValueError("current I row domain does not contain the pinned back-scope rows")

    routes_obj = json.loads(M.A.PROT.read_text())
    route_records = routes_obj.get("surface_routes")
    if not isinstance(route_records, list) or len(route_records) != 149:
        raise ValueError("pinned 149 route inventory is malformed")
    coordmap = _coord_row_map(ip)
    route_rows = []
    for rec in route_records:
        if rec.get("role") != "I":
            continue
        key = np.ascontiguousarray(np.asarray(rec["target_world_m_f32"], np.float32)).view(np.uint32).tobytes()
        if key not in coordmap:
            raise ValueError("registered I route anchor is absent from current source")
        route_rows.append(coordmap[key])
    route_rows = np.asarray(sorted(set(map(int, route_rows))), np.int64)

    # Source-rim vertices stay fixed, including attachments that are not in the
    # back-scope or route lists.
    edge_rows = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1).reshape((-1, 2)), axis=1)
    unique_edges, edge_counts = np.unique(edge_rows, axis=0, return_counts=True)
    rim_rows = np.unique(unique_edges[edge_counts == 1].ravel()).astype(np.int64)
    protected = np.asarray(sorted(set(map(int, back_rows)) | set(map(int, route_rows)) |
                                  set(map(int, rim_rows))), np.int64)

    # Compile candidate I surface edges that touch bad Skin/Fat cells. Partial
    # crosswalk is valid only for source rows unused by every I face.
    face_edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    surface_edges, edge_counts = np.unique(face_edges, axis=0, return_counts=True)
    interior_edges = surface_edges[edge_counts == 2]
    edge_lookup = {tuple(map(int, e)): i for i, e in enumerate(interior_edges)}
    score = np.zeros(len(interior_edges), np.int64)
    role_recs = {}
    crosswalk_recs = {}
    for role in ("Skin", "Fat"):
        vol = M.load_pinned_npz(state[role + "_current"], "current " + role + " volume")
        vp = np.asarray(vol["points_world_m"], np.float32)
        tt = np.asarray(vol["tetrahedra"], np.int64)
        det, qv = M.measurements(vp, tt)
        bad = ~M.M.passes(det, qv)
        cross, cwrec = exact_used_row_crosswalk(ip, faces, vp, role)
        crosswalk_recs[role] = cwrec
        inv = np.full(len(vp), -1, np.int64)
        used_rows = np.unique(faces.reshape(-1))
        inv[cross[used_rows]] = used_rows
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
        role_recs[role] = {"path": state[role + "_current"]["path"],
            "sha256": state[role + "_current"]["sha256"], "cells": int(len(tt)),
            "native_bad": int(np.count_nonzero(bad)), "surface_volume_crosswalk": cwrec}

    lengths = np.linalg.norm(ip[interior_edges[:,0]].astype(np.float64) - ip[interior_edges[:,1]].astype(np.float64), axis=1)
    valid = (score > 0) & (lengths <= float(max_edge_length_m))
    protected_set = set(map(int, protected))
    eligible = []
    for ei in np.flatnonzero(valid):
        k, d = map(int, interior_edges[ei])
        rows = np.flatnonzero(np.any((faces == k) | (faces == d), axis=1))
        star_vertices = set(map(int, faces[rows].ravel()))
        if star_vertices & protected_set:
            continue
        eligible.append((int(ei), k, d))
    eligible.sort(key=lambda z: (-int(score[z[0]]), float(lengths[z[0]]), z[1], z[2]))
    eligible = eligible[:int(max_edges)]
    orientations = []
    for _, k, d in eligible:
        orientations.extend(((k, d), (d, k)))

    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    edgepath = inputs / "candidate-I-edges.npz"
    np.savez_compressed(edgepath,
        candidate_edges=np.asarray(orientations, np.int64).reshape((-1,2)),
        edge_length_m=np.asarray([lengths[ei] for ei,_,_ in eligible], np.float64),
        bad_tet_incidence=np.asarray([score[ei] for ei,_,_ in eligible], np.int64),
        undirected_edges=np.asarray([[k,d] for _,k,d in eligible], np.int64).reshape((-1,2)))
    protpath = inputs / "protected-I-rows.npz"
    np.savez_compressed(protpath, protected_surface_rows=protected,
        route_anchor_rows=route_rows, back412_rows=back_rows, source_rim_rows=rim_rows)
    cfg = {"schema":"h706.coupled-anchor-collapse-run-config.v1",
        "surface_I":surfaces["I"], "baseline_surfaces":surfaces,
        "volumes":{"Skin":state["Skin_current"], "Fat":state["Fat_current"]},
        "candidate_edges":{"path":str(edgepath),"sha256":sha(edgepath),"key":"candidate_edges"},
        "protected_surface_rows":{"path":str(protpath),"sha256":sha(protpath),"key":"protected_surface_rows"},
        "route_anchor_rows":{"path":str(protpath),"sha256":sha(protpath),"key":"route_anchor_rows"},
        "move_cap_m":float(move_cap_m), "max_operations":int(max_operations),
        "time_limit_s":float(time_limit_s)}
    cfgpath = inputs / "run-config.json"
    M._write_json(cfgpath, cfg)
    proof_path = Path(proof["state"]["path"]).resolve()
    prep = {"schema":"h706.coupled-anchor-collapse-state-preparation.v2",
        "source_state":{"path":str(state_path),"sha256":sha(state_path)},
        "source_evidence_807":{"path":str(proof_path),"sha256":proof["state"]["sha256"],
            "status":proof["status"],"mode":proof.get("audit_implementation",{}).get("mode"),
            "proof":proof},
        "current_surface_pins":surfaces,"volume_inputs":role_recs,
        "back_scope":{"path":str(M.BACK_SCOPE_PATH),"sha256":M.BACK_SCOPE_SHA,
            "back412_I_rows":int(len(back_rows)),"back412_D_rows":int(len(back_d_rows)),
            "current_core_motion_vs_807_baseline":core_motion,
            "rows_remain_fixed_for_this_collapse_batch":True},
        "source_evidence_implementation":{"path":str(SOURCE_EVIDENCE),"sha256":SOURCE_EVIDENCE_SHA256,
            "validation_helper":{"path":str(SOURCE_EVIDENCE_BASE),"sha256":SOURCE_EVIDENCE_BASE_SHA256}},
        "source_rim_rows":int(len(rim_rows)),"registered_I_route_rows":int(len(route_rows)),
        "protected_I_rows":int(len(protected)),"crosswalk_policy":"every face-used I row must map exactly and injectively; an unmapped row is permitted only if no current I face references it",
        "crosswalk_by_volume_role":crosswalk_recs,
        "candidate_I_interior_edges":int(len(interior_edges)),
        "bad_incident_eligible_edges_before_cap":int(np.count_nonzero(valid)),
        "scheduled_undirected_edges":int(len(eligible)),"scheduled_orientations":int(len(orientations)),
        "edge_length_cap_m":float(max_edge_length_m),"max_edges":int(max_edges),
        "edge_roster":{"path":str(edgepath),"sha256":sha(edgepath)},
        "run_config":{"path":str(cfgpath),"sha256":sha(cfgpath)},
        "native_q_gate_unchanged":M.NATIVE_Q,"minimum_cell_volume_m3_unchanged":M.MIN_CELL_VOLUME_M3,
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    M._write_json(inputs / "state-preparation.json", prep)
    return cfgpath, prep, protected, route_rows, rim_rows, back_rows


def run_verified_state(state_path, out_dir, *, move_cap_m=0.0005, max_operations=32,
                       time_limit_s=90, max_edge_length_m=0.0005, max_edges=100000):
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfgpath, prep, protected, route_rows, rim_rows, back_rows = state_config(state_path, out,
        move_cap_m=move_cap_m, max_operations=max_operations, time_limit_s=time_limit_s,
        max_edge_length_m=max_edge_length_m, max_edges=max_edges)
    proposal = out / "proposal"
    report = M.run_from_config(cfgpath, proposal)
    # Readback verifies this batch itself left all current back, rim, and route
    # coordinates byte-identical. Source proof 806 separately authenticates O and
    # any prior bounded I/D movement from its exact baseline.
    cfg = M.load_config(cfgpath)
    current = cfg["surface"]
    ip = np.asarray(current["points_world_m_f32"], np.float32)
    candidate_path = proposal / "candidate-I-typed-current.npz"
    readback = {"candidate_exists":candidate_path.is_file(),"back_rows_unchanged":None,
                "rim_rows_unchanged":None,"route_rows_unchanged":None}
    if candidate_path.is_file():
        candidate = M.load_pinned_npz({"path":str(candidate_path),"sha256":sha(candidate_path)},"candidate I readback")
        cp = np.asarray(candidate["points_world_m_f32"], np.float32)
        if cp.shape != ip.shape:
            raise ValueError("candidate I changed point row domain")
        readback["back_rows_unchanged"] = bool(np.array_equal(cp[back_rows].view(np.uint32), ip[back_rows].view(np.uint32)))
        readback["rim_rows_unchanged"] = bool(np.array_equal(cp[rim_rows].view(np.uint32), ip[rim_rows].view(np.uint32)))
        readback["route_rows_unchanged"] = bool(np.array_equal(cp[route_rows].view(np.uint32), ip[route_rows].view(np.uint32)))
        if not all(readback[k] for k in ("back_rows_unchanged","rim_rows_unchanged","route_rows_unchanged")):
            raise ValueError("candidate changed a protected back/rim/route coordinate")
    wrapper = {"schema":"h706.coupled-anchor-collapse-state-run.v2",
        "preparation":prep,
        "candidate_run":{"path":str(proposal / "state.json"),
            "sha256":sha(proposal / "state.json") if (proposal / "state.json").is_file() else None,
            "status":report.get("status"),"accepted_operations":report.get("accepted_operations",0)},
        "protected_coordinate_readback":readback,
        "candidate_exact_audit":"775_REQUIRED_FOR_THIS_NEW_I_GEOMETRY",
        "actual_697_217":"REQUIRED_FOR_TYPED_I_CANDIDATE",
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    M._write_json(out / "state-wrapper.json", wrapper)
    return wrapper


def crosswalk_fixture():
    sp = np.asarray([[0,0,0],[1,0,0],[0,1,0],[9,9,9]], np.float32)
    sf = np.asarray([[0,1,2]], np.int64)
    vp = sp[:3].copy()
    _, good = exact_used_row_crosswalk(sp, sf, vp, "fixture")
    assert good["unmapped_unused_surface_rows"] == [3], good
    try:
        exact_used_row_crosswalk(sp, sf, vp[:2], "fixture-bad")
    except ValueError as exc:
        bad = str(exc)
    else:
        raise AssertionError("face-used missing crosswalk row was accepted")
    return {"status":"PASS_USED_ROWS_EXACT_UNUSED_MISSING_ALLOWED",
            "unmapped_unused_rows":good["unmapped_unused_surface_rows"],
            "used_missing_rejected":bad}


def main():
    ap = argparse.ArgumentParser(description="Versioned coupled-anchor state runner using 806 source evidence and exact used-row crosswalks.")
    ap.add_argument("--mode", choices=("fixture","prepare-state","run","replay","context","prune-775","promote"), default="fixture")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--out","--output",dest="out",type=Path)
    ap.add_argument("--move-cap-m",type=float,default=0.0005)
    ap.add_argument("--max-operations",type=int,default=32)
    ap.add_argument("--seconds",type=float,default=90)
    ap.add_argument("--max-edge-length-m",type=float,default=0.0005)
    ap.add_argument("--max-edges",type=int,default=100000)
    ap.add_argument("--775-report",type=Path)
    ap.add_argument("--replay-report",type=Path)
    ap.add_argument("--697-receipt",type=Path)
    a = ap.parse_args()
    if a.state is not None:
        if a.out is None: ap.error("--state requires --output")
        if a.mode == "prepare-state":
            out = Path(a.out).resolve()
            out.mkdir(parents=True, exist_ok=False)
            cfgpath, prep, _, _, _, _ = state_config(a.state,out,move_cap_m=a.move_cap_m,
                max_operations=a.max_operations,time_limit_s=a.seconds,
                max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
            print(json.dumps({"status":"PASS_STATE_PREPARATION_ONLY","preparation":prep,
                "config":{"path":str(cfgpath),"sha256":sha(cfgpath)},
                "candidate_run":"NOT STARTED","physical_admission":False,"native_steps":0},indent=2,default=str)); return
        result = run_verified_state(a.state,a.out,move_cap_m=a.move_cap_m,
            max_operations=a.max_operations,time_limit_s=a.seconds,
            max_edge_length_m=a.max_edge_length_m,max_edges=a.max_edges)
        print(json.dumps(result,indent=2,default=str)); return
    if a.mode == "fixture":
        result = {"coupled_anchor_fixture":M.fixture(),"crosswalk_fixture":crosswalk_fixture()}
        if a.out:
            p=Path(a.out);p.mkdir(parents=True,exist_ok=False);f=p/"fixture-report.json"
            M._write_json(f,result);result["report_path"]=str(f);result["report_sha256"]=sha(f)
        print(json.dumps(result,indent=2,default=str)); return
    if a.mode == "run":
        if not a.config or not a.out:ap.error("run requires --config and --out")
        print(json.dumps(M.run_from_config(a.config,a.out),indent=2,default=str)); return
    if a.mode == "replay":
        if not a.config or not a.run_dir or not a.out:ap.error("replay requires --config --run-dir --out")
        print(json.dumps(M.independent_replay(a.config,a.run_dir,a.out),indent=2,default=str)); return
    if a.mode == "context":
        if not a.config or not a.run_dir or not a.out:ap.error("context requires --config --run-dir --out")
        x=M.load_config(a.config);r=json.loads((a.run_dir/"state.json").read_text())
        print(json.dumps(M.make_775_config(a.run_dir,x,r,a.out),indent=2)); return
    if a.mode == "prune-775":
        if not a.config or not a.run_dir or not a.out or not a.__dict__.get("775_report"):
            ap.error("prune-775 requires --config --run-dir --775-report --out")
        print(json.dumps(M.prune_from_775(a.config,a.run_dir,a.__dict__["775_report"],a.out),indent=2,default=str));return
    if a.mode == "promote":
        if not a.run_dir or not a.replay_report or not a.__dict__.get("775_report") or not a.__dict__.get("697_receipt") or not a.out:
            ap.error("promote requires --run-dir --replay-report --775-report --697-receipt --out")
        print(json.dumps(M.promote_candidate(a.run_dir,a.replay_report,a.__dict__["775_report"],a.__dict__["697_receipt"],a.out),indent=2,default=str));return


if __name__ == "__main__":
    main()
