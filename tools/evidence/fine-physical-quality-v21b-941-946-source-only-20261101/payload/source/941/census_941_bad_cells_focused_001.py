#!/usr/bin/env python3
"""Read-only focused census of the 12 native-bad cells in qualified state 941."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
WORK = ROOT / "quality-endpoint-collapse-001"
STATE = ROOT / "all-quality-941-b5-backtracking-root/qualified/verified-state.json"
STATE_SHA = "856a8850fa36bb06447bd667945e8241c65003fe1bb3f87c156079a76579bc6c"
C6_PATH = WORK / "fixed_boundary_cavity_884c6_connected_group_root_v2.py"
C6_SHA = "5ab0f901a6a1d74074f19bd677a09f46e76678f0d266c94f59325e99c5d0d8a4"
FIT_PATH = ROOT / "coupled-anchor-collapse-780-root/ordinary-allrole-v26b5-cowdiff-f32backtrack-250um/repair_coupled_quality_points_722_v26b5_cowdiff_f32backtrack_250um.py"
FIT_SHA = "53071ffde5f51eb65a2090d1e60931f552f0647fa8d5c4940af941d525fab946"
PROPOSAL = ROOT / "all-quality-941-b5-backtracking-root/proposal"
BACK_NPZ = PROPOSAL / "actual-back-fat-protected-rows.npz"
SCOPE_NPZ = ROOT / "current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
ROUTES_JSON = ROOT / "verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
OUT = WORK / "census-941-focused-001"

def sha(p):
    return hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()

def imp(name, path, expected):
    got = sha(path)
    if got != expected:
        raise RuntimeError(f"{name} hash mismatch: {got}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def rim_rows(faces):
    f = np.asarray(faces, np.int64)
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    e.sort(axis=1)
    ue, counts = np.unique(e, axis=0, return_counts=True)
    return np.unique(ue[counts == 1].reshape(-1))

def bad_components(tets, rows):
    rows = list(map(int, rows))
    parent = {r:r for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for r in rows:
        a,b,c,d = map(int, tets[r])
        for face in ((a,b,c),(a,d,b),(a,c,d),(b,d,c)):
            key = tuple(sorted(face))
            if key in owners:
                x,y = find(r),find(owners[key])
                if x != y: parent[max(x,y)] = min(x,y)
            else: owners[key] = r
    groups = {}
    for r in rows: groups.setdefault(find(r), []).append(r)
    return {r:i for i,g in enumerate(sorted(groups.values(), key=lambda x:(min(x),len(x)))) for r in g}

def main():
    t0 = time.monotonic()
    if sha(STATE) != STATE_SHA: raise RuntimeError("state pin mismatch")
    c6 = imp("c6_focused_census", C6_PATH, C6_SHA)
    fit = imp("v26b5_free_mask", FIT_PATH, FIT_SHA)
    state = json.loads(STATE.read_text())
    ctx = c6.P883.load_context(STATE, WORK, fixture=False, source_role="D")
    surfaces = ctx["surfaces"]
    caches = ctx["caches"]
    T = fit.T

    # Reconstruct the producer's exact shared point namespace; no geometry is changed.
    active_keys = {r: {T.key(surfaces[r]["points_world_m_f32"][int(i)]) for i in np.unique(surfaces[r]["faces"])} for r in ("O","I","D")}
    allpoints, gl, vmap = [], {}, {}
    for vr in ("Skin", "Fat"):
        mp=[]
        for point in np.asarray(caches[vr]["points"], np.float32):
            key=T.key(point)
            if vr=="Fat" and key in active_keys["I"] and key in gl:
                row=gl[key]
            else:
                row=len(allpoints); allpoints.append(point); gl.setdefault(key,row)
            mp.append(row)
        vmap[vr]=np.asarray(mp,np.int64)
    points=np.asarray(allpoints,np.float32)
    smap={r:np.asarray([gl.get(T.key(x),-1) for x in np.asarray(surfaces[r]["points_world_m_f32"],np.float32)],np.int64) for r in ("O","I","D")}
    tets=np.vstack([vmap[r][np.asarray(caches[r]["tets"],np.int64)] for r in ("Skin","Fat")])
    skin_n=len(caches["Skin"]["tets"])
    bad_by_role={r:np.flatnonzero(np.asarray(caches[r]["bad"],bool)).astype(np.int64) for r in ("Skin","Fat")}
    bad_rows=np.r_[bad_by_role["Skin"], skin_n+bad_by_role["Fat"]].astype(np.int64)
    bad_corners=np.unique(tets[bad_rows].reshape(-1))

    # Recreate exact fixed/free role masks used by the frozen 941 unified fit.
    with np.load(SCOPE_NPZ,allow_pickle=False) as z: scope={r:set(map(int,np.asarray(z[r+"_vertex_rows"],np.int64))) for r in ("O","I","D")}
    with np.load(BACK_NPZ,allow_pickle=False) as z: back={k:np.asarray(z[k],np.int64) for k in z.files if z[k].dtype.kind in "iu"}
    routes=json.loads(ROUTES_JSON.read_text())["surface_routes"]
    route_rows={r:set() for r in ("O","I","D")}
    for rec in routes:
        r=rec["role"]; target=np.asarray(rec["target_world_m_f32"],np.float32)
        hit=np.flatnonzero(np.all(np.asarray(surfaces[r]["points_world_m_f32"],np.float32)==target,axis=1))
        if len(hit)!=1: raise RuntimeError(f"route coordinate nonunique/absent: {r} {rec.get('canonical_protected_row')}")
        route_rows[r].add(int(hit[0]))
    rim={r:set(map(int,rim_rows(surfaces[r]["faces"]))) for r in ("O","I","D")}
    fixed=set(); ordinary_fixed=set(); source_categories={r:{"scope":scope[r],"rim":rim[r],"routes":route_rows[r]} for r in ("O","I","D")}
    for r in ("O","I","D"):
        mp=smap[r]
        ordinary_fixed.update(int(mp[i]) for i in scope[r] if i<len(mp) and mp[i]>=0)
        ordinary_fixed.update(int(mp[i]) for i in rim[r] if i<len(mp) and mp[i]>=0)
        ordinary_fixed.update(int(mp[i]) for i in route_rows[r] if i<len(mp) and mp[i]>=0)
        # Unified inner-back policy fixes O scope; I/D scope is released only through bad-star/ring eligibility.
        if r=="O": fixed.update(int(mp[i]) for i in scope[r] if i<len(mp) and mp[i]>=0)
        fixed.update(int(mp[i]) for i in rim[r] if i<len(mp) and mp[i]>=0)
        fixed.update(int(mp[i]) for i in route_rows[r] if i<len(mp) and mp[i]>=0)
    back_volume=np.asarray(back["current_plus_historical_fat_volume_rows"],np.int64)
    back_global=np.unique(vmap["Fat"][back_volume])
    back_i=np.asarray(back["current_i_surface_rows"],np.int64)
    back_d=np.asarray(back["current_d_surface_back_rows"],np.int64)
    back_i_global=smap["I"][back_i]
    back_d_global=smap["D"][back_d]
    ordinary_fixed.update(map(int,back_global))
    ordinary_fixed.update(map(int,back_i_global[back_i_global>=0]))

    faces={r:np.asarray(surfaces[r]["faces"],np.int64) for r in ("O","I","D")}
    ordinary,inner,free,ordinary_info,inner_info=fit.build_unified_free_sets(
        bad_rows,faces,smap,tets,back_global,ordinary_fixed,fixed,np.empty(0,np.int64))
    diag=json.loads((PROPOSAL/"lp-diagnostics.json").read_text())
    one=state["source_triangle_one_ring_expansion"]["unified_inner_back_ordinary"]
    expected={"bad_cells":12,"free_rows":399,"fixed_rows":326,"ordinary_free_rows":146}
    got={"bad_cells":int(len(bad_rows)),"free_rows":int(len(free)),"fixed_rows":int(len(fixed)),"ordinary_free_rows":int(len(ordinary))}
    if got!=expected: raise RuntimeError(f"reconstructed 941 mask differs from frozen diagnostics: {got} != {expected}")
    if got["free_rows"]!=int(one["inner_mask"]["free_global_rows"]): raise RuntimeError("inner free row count mismatch")

    # Only map and report source identities/protections for the 34 bad-cell corners.
    wanted=set(map(int,bad_corners)); src_by_g={g:[] for g in wanted}
    for r in ("O","I","D"):
        active=np.unique(faces[r].reshape(-1))
        mapped=smap[r][active]
        for sr,g in zip(active,mapped):
            if int(g) in wanted: src_by_g[int(g)].append({"role":r,"source_row":int(sr)})
    back_cat={"actual_Fat_back_global":set(map(int,back_global)),
              "actual_I_back_source":set(map(int,back_i)),"actual_D_back_source":set(map(int,back_d))}
    rows_out=[]
    for role in ("Skin","Fat"):
        cache=caches[role]; local_bad=bad_by_role[role]; comp=bad_components(np.asarray(cache["tets"],np.int64),local_bad)
        for cell in map(int,local_bad):
            tet=np.asarray(cache["tets"][cell],np.int64); gnodes=vmap[role][tet]
            xyz=np.asarray(cache["points"],np.float32)[tet]
            e=[]
            for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                e.append(float(np.linalg.norm(xyz[a].astype(np.float64)-xyz[b].astype(np.float64))))
            corners=[]
            for local,(volrow,grow,coord) in enumerate(zip(tet,gnodes,xyz)):
                tags=src_by_g.get(int(grow),[])
                categories=[]
                for tag in tags:
                    rr,sr=tag["role"],tag["source_row"]
                    if sr in scope[rr]:categories.append("scope:"+rr)
                    if sr in rim[rr]:categories.append("source_rim:"+rr)
                    if sr in route_rows[rr]:categories.append("route149:"+rr)
                    if rr=="I" and sr in back_cat["actual_I_back_source"]:categories.append("actual_back_I")
                    if rr=="D" and sr in back_cat["actual_D_back_source"]:categories.append("actual_back_D")
                if int(grow) in back_cat["actual_Fat_back_global"]:categories.append("actual_back_Fat")
                unique_categories=sorted(set(categories))
                corners.append({"tet_corner":local,"volume_point_row":int(volrow),"shared_global_point_row":int(grow),
                    "xyz_f32_m":[float(x) for x in coord],"source_rows":tags,"protection_categories":unique_categories,
                    "fit_status":"free_in_941_union" if int(grow) in set(map(int,free)) else ("explicitly_fixed_anchor" if int(grow) in fixed else "fixed_outside_941_free_region")})
            q=float(cache["q"][cell]);det=float(cache["det"][cell])
            minedge=min(e);maxedge=max(e)
            rows_out.append({"role":role,"cell_row":cell,"bad_component":int(comp[cell]),"material":int(cache["mat"][cell]),
                "native_q":q,"determinant":det,"exact_positive":bool(det>0),"edge_lengths_m_sorted":sorted(e),
                "min_edge_m":minedge,"max_edge_m":maxedge,"max_over_min_edge_ratio":maxedge/minedge if minedge else None,
                "free_corners":sum(c["fit_status"]=="free_in_941_union" for c in corners),"corners":corners})
    # Preserve concise provenance and exact counts; external digest is written separately.
    report={"schema":"h706.focused-native-bad-cell-census.v1","status":"PASS_READ_ONLY_FOCUSED_CENSUS",
      "input_state":{"path":str(STATE),"sha256":STATE_SHA},
      "input_volumes":{r:{"path":state[r+"_current"]["path"],"sha256":state[r+"_current"]["sha256"]} for r in ("Skin","Fat")},
      "source_surfaces":{r:state["source_surfaces"][r] for r in ("O","I","D")},
      "census_sources":[{"path":str(C6_PATH),"sha256":C6_SHA},{"path":str(FIT_PATH),"sha256":FIT_SHA}],
      "scope_mask":{"path":str(SCOPE_NPZ),"sha256":sha(SCOPE_NPZ)},
      "back_mask":{"path":str(BACK_NPZ),"sha256":sha(BACK_NPZ)},
      "routes":{"path":str(ROUTES_JSON),"sha256":sha(ROUTES_JSON),"records":len(routes)},
      "reconstructed_941_mask":got,"diagnostic_reference":{"path":str(PROPOSAL/"lp-diagnostics.json"),"sha256":sha(PROPOSAL/"lp-diagnostics.json"),
        "bad_before":diag["bad_before"],"bad_after":diag["bad_after"],"score_before":diag["bad_score_before"],"score_after":diag["bad_score_after"],
        "total_trust_cap_m":diag["total_trust_cap_m"],"max_free_node_l1_displacement_bound_m":diag["max_free_node_l1_displacement_bound_m"],
        "maximum_cumulative_path_L1_m":diag["maximum_cumulative_path_L1_m"],"changed_point_rows":len(diag["changed_point_rows"]),
        "patches":one["roster"]["patches"],"all_bad_cells_have_free_corner":one["roster"]["all_free_corner_bad_cells_covered_by_full_star"]},
      "mask_reconstruction_receipt":{"free_union_matches_399":len(free)==399,"fixed_anchor_count":len(fixed),
        "inner_policy_fixed_rows":len(fixed),"ordinary_fixed_rows_excluding_back_and_inner_scope":len(ordinary_fixed),
        "ordinary_free_rows":len(ordinary),"inner_free_rows":len(inner),"free_union_rows":len(free),
        "free_status_exact_for_bad_corners":True,"witness_frozen_rows":0},
      "bad_cells":rows_out,"elapsed_s":time.monotonic()-t0,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/"census.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"output":str(out),"bad_cells":len(rows_out),"mask":got,"elapsed_s":report["elapsed_s"]},separators=(",",":")),flush=True)

if __name__=="__main__": main()
