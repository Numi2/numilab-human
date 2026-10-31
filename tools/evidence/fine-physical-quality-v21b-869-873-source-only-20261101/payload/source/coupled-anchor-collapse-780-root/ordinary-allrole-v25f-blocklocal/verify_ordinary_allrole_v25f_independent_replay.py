#!/usr/bin/env python3
"""Independent v25f block-local replay with the full ordinary-mask and 775 gates."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BUILDER=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25f-blocklocal/repair_coupled_quality_points_722_v25f_ordinary_blocklocal.py"
BUILDER_SHA="e05d5662d64edd97d25c5a0e42e00a58ba06ef2850b3afc6fd9b23a3111fc376"
V24=ROOT/"continue_coupled_quality_points_v24c_ordinary_outer_o_local.py"
V24_SHA="f69ab97f020436c5f0afa997a770ad291bad9ee63942522d0ea0b3c49230074a"
AUDITOR=ROOT/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
AUDITOR_SHA="82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
BACK_DIR=ROOT/"coupled-anchor-collapse-780-root"
BACK=BACK_DIR/"coupled_anchor_collapse_795_outer_role.py"
BACK_SHA="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"
ZERO_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def pin(p):
    p=Path(p).resolve()
    return {"path":str(p),"sha256":sha(p)}

def imp(name,path,expected):
    if sha(path)!=expected: raise ValueError(name+" source hash mismatch")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import "+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

M=imp("v25_builder_for_replay",BUILDER,BUILDER_SHA)
V=imp("v24_replay_owner_for_v25",V24,V24_SHA)
R=M.BACK
if sha(AUDITOR)!=AUDITOR_SHA: raise ValueError("775 source auditor hash mismatch")
if sha(BACK)!=BACK_SHA: raise ValueError("795 authenticated back lineage source hash mismatch")

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def state(path):
    return V.load_state(Path(path).resolve())

def build_parent_domain(parent):
    surfaces={r:load_npz(parent["source_surfaces"][r]["path"]) for r in ("O","I","D")}
    volumes={r:load_npz(parent[r+"_current"]["path"]) for r in ("Skin","Fat")}
    spkeys={r:{M.T.key(surfaces[r]["points_world_m_f32"][i]) for i in np.unique(surfaces[r]["faces"])} for r in ("O","I","D")}
    lookup={};allpoints=[];vmap={}
    for vr in ("Skin","Fat"):
        rows=[]
        for x in np.asarray(volumes[vr]["points_world_m"]):
            key=M.T.key(x)
            if vr=="Fat" and key in spkeys["I"] and key in lookup: row=lookup[key]
            else: row=len(allpoints);allpoints.append(x);lookup.setdefault(key,row)
            rows.append(row)
        vmap[vr]=np.asarray(rows,np.int64)
    points=np.asarray(allpoints,np.float32)
    tets=np.vstack([vmap[r][np.asarray(volumes[r]["tetrahedra"],np.int64)] for r in ("Skin","Fat")])
    smap={r:np.asarray([lookup.get(M.T.key(x),-1) for x in surfaces[r]["points_world_m_f32"]],np.int64) for r in ("O","I","D")}
    for r in ("O","I","D"):
        if np.any(smap[r][np.unique(np.asarray(surfaces[r]["faces"],np.int64))]<0):
            raise ValueError(r+" active source row has no exact current volume point")
    return surfaces,volumes,points,tets,vmap,smap

def independent_back_lineage(parent,surfaces,volumes):
    scopep=Path(R.E.BACK_SCOPE_PATH).resolve()
    if sha(scopep)!=R.E.BACK_SCOPE_SHA: raise ValueError("back-scope mask source mismatch")
    scope=R.load_npz(R.pin(scopep),"back scope")
    reportp=scopep.parent/"report.json";report=json.loads(reportp.read_text())
    if report.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT" or report.get("protected_mask_asset",{}).get("sha256")!=sha(scopep):
        raise ValueError("back-scope report/asset mismatch")
    for role in ("O","I","D"):
        ref=report["current_sources"]["typed_roles"][role]["candidate"]
        if str(np.asarray(scope[role+"_candidate_sha256"]).reshape(-1)[0])!=ref["sha256"]:
            raise ValueError(role+" scope reference pin mismatch")
    iref=report["current_sources"]["typed_roles"]["I"]["candidate"]
    irows,lineage=R.map_rows_by_lineage(parent["source_surfaces"]["I"],iref,surfaces["I"],
        scope["I_vertex_rows"],"independent current I back lineage",allow_collapsed=True)
    if len(irows)!=790:raise ValueError("independent I back lineage count mismatch")
    fatpoints=np.asarray(volumes["Fat"].get("points_world_m_f32",volumes["Fat"]["points_world_m"]),np.float32)
    ix,_=R.S.exact_used_row_crosswalk(np.asarray(surfaces["I"]["points_world_m_f32"],np.float32),
        np.asarray(surfaces["I"]["faces"],np.int64),fatpoints,"independent current Fat back")
    current_fat=np.unique(ix[irows])
    if np.any(current_fat<0) or len(current_fat)!=len(np.unique(irows)):
        raise ValueError("independent current I-to-Fat back map mismatch")
    irefdata=R.load_npz(iref,"historical I back reference")
    lut=R.coordmap(fatpoints)
    historical=np.asarray([lut.get(R.fkey(irefdata["points_world_m_f32"][int(r)]),-1)
        for r in np.asarray(scope["I_vertex_rows"],np.int64)],np.int64)
    historical=historical[historical>=0]
    union=np.union1d(current_fat,historical)
    if len(union)!=790:raise ValueError("independent current+historical Fat back union mismatch")
    return {"fat_rows":union,"current_fat_rows":current_fat,"historical_fat_rows":np.unique(historical),
        "current_i_rows":np.asarray(irows,np.int64),"lineage":lineage,"scope_mask":R.pin(scopep),
        "scope_report":pin(reportp),"scope_reference_I":iref,"current_I":parent["source_surfaces"]["I"]}

def source_fixed_rows(parent,surfaces,smap,back_fat,vmap):
    scopep=Path(R.E.BACK_SCOPE_PATH).resolve()
    if sha(scopep)!=R.E.BACK_SCOPE_SHA: raise ValueError("back-scope mask source mismatch")
    scope=R.load_npz(R.pin(scopep),"back scope")
    reportp=scopep.parent/"report.json";report=json.loads(reportp.read_text())
    if report.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT" or report.get("protected_mask_asset",{}).get("sha256")!=sha(scopep):
        raise ValueError("back-scope report/asset mismatch")
    route_path=ROOT/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
    if sha(route_path)!="ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71":
        raise ValueError("registered route file hash mismatch")
    fixed=set();counts={}
    for role in ("O","I","D"):
        ref=report["current_sources"]["typed_roles"][role]["candidate"]
        got=str(np.asarray(scope[role+"_candidate_sha256"]).reshape(-1)[0])
        if got!=ref["sha256"]:raise ValueError(role+" scope reference pin mismatch")
        rows,lineage=R.map_rows_by_lineage(parent["source_surfaces"][role],ref,surfaces[role],
               scope[role+"_vertex_rows"],role+" protected source scope",allow_collapsed=True)
        if len(rows)!=len(scope[role+"_vertex_rows"]):raise ValueError(role+" scope row count mismatch")
        scope_rows=set(map(int,rows))
        faces=np.asarray(surfaces[role]["faces"],np.int64)
        edges=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
        ue,ec=np.unique(edges,axis=0,return_counts=True)
        rim=set(map(int,np.unique(ue[ec==1])))
        route_rows=set()
        for rec in json.loads(route_path.read_text())["surface_routes"]:
            if rec.get("role")!=role:continue
            hit=np.flatnonzero(np.all(np.asarray(surfaces[role]["points_world_m_f32"],np.float32)==np.asarray(rec["target_world_m_f32"],np.float32),axis=1))
            if len(hit)!=1:raise ValueError(role+" route does not map uniquely to the current parent")
            route_rows.add(int(hit[0]))
        fixed_rows=scope_rows|rim|route_rows
        mapped_fixed=np.asarray(smap[role][list(fixed_rows)],np.int64)
        if np.any(mapped_fixed<0):raise ValueError(role+" protected source rows are unmapped")
        fixed.update(map(int,mapped_fixed))
        counts[role]={"scope_rows":len(scope_rows),"rim_rows":len(rim),"route_rows":len(route_rows),
                      "lineage_steps":lineage["descendant_steps"]}
    back_global=set(map(int,vmap["Fat"][back_fat]))
    fixed.update(back_global)
    return fixed,counts

def admissible_bad_corner_ring(points,tets,surfaces,smap,fixed):
    det,q=M.M.measurements(points,tets)
    if np.any(det<=0):raise ValueError("baseline contains a nonpositive Float32 cell")
    passed=np.asarray(M.M.passes(det,q),bool)
    bad_corners=set(map(int,np.unique(tets[~passed])))
    ring=set()
    for role in ("D","O","I"):
        faces=np.asarray(surfaces[role]["faces"],np.int64)
        gr=np.asarray(smap[role],np.int64)
        seed_local=np.flatnonzero(np.isin(gr,np.fromiter(bad_corners-fixed,dtype=np.int64)) if bad_corners-fixed else np.zeros(len(gr),bool))
        if len(seed_local):
            incident=np.any(np.isin(faces,seed_local),axis=1)
            local=np.unique(faces[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
            mapped=gr[local]
            if np.any(mapped<0):raise ValueError(role+" one-ring source point is unmapped")
            ring.update(map(int,mapped))
    allowed=(bad_corners|ring)-fixed
    return allowed,{"baseline_bad_cells":int(np.count_nonzero(~passed)),
                    "baseline_bad_corners":len(bad_corners),"surface_ring_rows":len(ring),
                    "fixed_rows":len(fixed),"allowed_rows":len(allowed),
                    "baseline_nonpositive_cells":0}

def checked_volume_l1_changes(baseline, candidate, cap, label):
    bp=np.asarray(baseline,dtype=np.float32);cp=np.asarray(candidate,dtype=np.float32)
    if bp.ndim!=2 or bp.shape[1:]!=(3,) or cp.shape!=bp.shape or not np.isfinite(bp).all() or not np.isfinite(cp).all():
        raise ValueError(label+" volume point arrays have incompatible or nonfinite geometry")
    moved=np.flatnonzero(np.any(bp.view(np.uint32)!=cp.view(np.uint32),axis=1))
    l1=np.abs(cp[moved].astype(np.float64)-bp[moved].astype(np.float64)).sum(axis=1)
    maximum=float(l1.max(initial=0.0))
    if np.any(l1>float(cap)):raise ValueError(label+" volume point step exceeds the recorded L1 trust cap")
    return moved,maximum

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-state",type=Path,required=True)
    ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--audit-report",type=Path,required=True)
    ap.add_argument("--audit-config",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path)
    ap.add_argument("--rollback-plan",type=Path)
    ap.add_argument("--output-report",type=Path,required=True)
    args=ap.parse_args()
    candp=args.candidate_state.resolve();parentp=args.parent_state.resolve()
    cand=state(candp);parent=state(parentp)
    meta=cand.get("coupled_point_repair",{})
    producer=meta.get("producer",{})
    if producer.get("path")!=str(BUILDER.resolve()) or producer.get("sha256")!=BUILDER_SHA:
        raise ValueError("candidate is not pinned to the v25f block-local producer")
    if meta.get("ordinary_block_local_quality_fit") is not True:
        raise ValueError("candidate lacks the v25f bad-face block-local solver mode")
    expected_patch_policy="bad-face-connected patches run sequentially; overlapping free nodes and full stars are allowed only with current/baseline-good revalidation and cumulative per-node trust cap"
    if meta.get("ordinary_block_patch_policy")!=expected_patch_policy:
        raise ValueError("candidate lacks the sequential overlap/revalidation policy")
    if meta.get("ordinary_all_role_local_repair") is not True or meta.get("ordinary_all_role_policy")!="baseline-bad volume corners plus one direct source-triangle ring over O/I/D; current/historical actual Fat back, all scope, rims and registered routes remain fixed":
        raise ValueError("candidate lacks exact v25 policy metadata")
    cap=float(meta.get("maximum_displacement_m",0.0))
    if not 0<cap<=5e-4:raise ValueError("v25 cap is invalid")
    audit=json.loads(args.audit_report.read_text())
    if audit.get("status")!=ZERO_STATUS:raise ValueError("775 exact source audit did not pass")
    V.validate_report_config_pin(args.audit_report,args.audit_config.resolve())
    cfg=json.loads(args.audit_config.read_text())
    for r in ("O","I","D"):
        b=cfg.get("baseline_candidates",{}).get(r,{})
        c=cfg.get("candidate_surfaces",{}).get(r,{})
        if b.get("path")!=parent["source_surfaces"][r]["path"] or b.get("sha256")!=parent["source_surfaces"][r]["sha256"]:
            raise ValueError(r+" 775 baseline pin differs from parent")
        if c.get("path")!=cand["source_surfaces"][r]["path"] or c.get("sha256")!=cand["source_surfaces"][r]["sha256"]:
            raise ValueError(r+" 775 candidate pin differs from candidate state")
    base_replay=args.output_report.resolve().with_name(args.output_report.stem+"-base-volume-replay.json")
    proposal_path=args.proposal_state.resolve() if args.proposal_state else None
    rollback_path=args.rollback_plan.resolve() if args.rollback_plan else None
    if bool(proposal_path)!=bool(rollback_path):raise ValueError("proposal state and rollback plan must be supplied together")
    base_receipt=V.independent_replay(candp,parentp,base_replay,proposal_state_path=proposal_path,rollback_plan_path=rollback_path)
    if base_receipt.get("status")!="PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise ValueError("existing independent volume replay did not pass")
    surfaces,volumes,points,tets,vmap,smap=build_parent_domain(parent)
    back=independent_back_lineage(parent,surfaces,volumes)
    if len(back["fat_rows"])!=790:raise ValueError("complete current/historical Fat back union is not 790")
    fixed,protection=source_fixed_rows(parent,surfaces,smap,back["fat_rows"],vmap)
    back_global=np.asarray(vmap["Fat"][back["fat_rows"]],np.int64)
    current_i_global=np.asarray(smap["I"][back["current_i_rows"]],np.int64)
    if np.any(current_i_global<0):raise ValueError("current back I source rows do not map to volume points")
    fixed.update(map(int,back_global));fixed.update(map(int,current_i_global))
    recorded=meta.get("actual_back_Fat_lineage_protection",{})
    row_asset=recorded.get("row_asset")
    if not isinstance(row_asset,dict) or sha(row_asset.get("path",""))!=row_asset.get("sha256"):
        raise ValueError("candidate lacks authenticated producer back-lineage row asset")
    with np.load(row_asset["path"],allow_pickle=False) as z:
        comparisons=( ("current_i_surface_rows",back["current_i_rows"]),
          ("current_fat_volume_rows",back["current_fat_rows"]),
          ("historical_fat_volume_rows",back["historical_fat_rows"]),
          ("current_plus_historical_fat_volume_rows",back["fat_rows"]) )
        for key,expected in comparisons:
            if key not in z or not np.array_equal(np.asarray(z[key],np.int64),np.asarray(expected,np.int64)):
                raise ValueError("producer back lineage row asset differs from independent mapping: "+key)
    allowed,quality=admissible_bad_corner_ring(points,tets,surfaces,smap,fixed)
    changed_global=set()
    role_moved={}
    volume_moved={}
    for r in ("O","I","D"):
        bp=np.asarray(surfaces[r]["points_world_m_f32"],np.float32)
        cp=np.asarray(load_npz(cand["source_surfaces"][r]["path"])["points_world_m_f32"],np.float32)
        if cp.shape!=bp.shape:raise ValueError(r+" source point domain changed")
        moved=np.flatnonzero(np.any(bp.view(np.uint32)!=cp.view(np.uint32),axis=1))
        if len(moved):
            displacement=np.abs(cp[moved].astype(np.float64)-bp[moved].astype(np.float64)).sum(axis=1)
            if np.any(displacement>cap):raise ValueError(r+" source movement exceeds L1 cap")
            changed_global.update(map(int,smap[r][moved]))
        role_moved[r]=int(len(moved))
    for vr in ("Skin","Fat"):
        bp=np.asarray(volumes[vr]["points_world_m_f32"],np.float32)
        cp=np.asarray(load_npz(cand[vr+"_current"]["path"])["points_world_m_f32"],np.float32)
        if bp.shape!=cp.shape:raise ValueError(vr+" volume point domain changed")
        moved,max_volume_l1=checked_volume_l1_changes(bp,cp,cap,vr)
        volume_moved[vr]={"moved_point_rows":int(len(moved)),"maximum_L1_displacement_m":max_volume_l1}
        changed_global.update(map(int,vmap[vr][moved]))
    if not changed_global.issubset(allowed):
        raise ValueError("candidate moved rows outside baseline-bad plus one-ring free set: "+str(sorted(changed_global-allowed)[:20]))
    if changed_global&fixed:raise ValueError("candidate moved a scope/rim/route/back point")
    one_ring=cand.get("source_triangle_one_ring_expansion",{})
    block_roster=one_ring.get("ordinary_all_role_block_local",{}).get("roster")
    if not isinstance(block_roster,dict) or block_roster.get("schema")!="h706.ordinary-all-role-block-local-patches.v2":
        raise ValueError("candidate lacks v25f block roster receipt")
    if block_roster.get("overlapping_free_rows_allowed_and_sequentially_revalidated") is not True or block_roster.get("overlapping_affected_cells_allowed_and_sequentially_revalidated") is not True:
        raise ValueError("candidate block roster has unsafe overlap policy")
    receipt={"schema":"h706.ordinary-all-role-local-independent-replay.v25f",
        "status":"PASS_775_SOURCE_CONTEXT_VOLUME_QUALITY_AND_V25_FREE_SET",
        "candidate_state":pin(candp),"parent_state":pin(parentp),
        "block_local_producer":pin(BUILDER),"block_local_roster":block_roster,
        "exact_775_report":pin(args.audit_report.resolve()),"exact_775_config":pin(args.audit_config.resolve()),
        "rollback_plan":pin(rollback_path) if rollback_path else None,"original_proposal_state":pin(proposal_path) if proposal_path else None,
        "base_independent_volume_replay":pin(base_replay),"base_replay_status":base_receipt["status"],
        "back_lineage":{"producer_row_asset":row_asset,"independently_reconstructed_scope_mask":back["scope_mask"],
            "current_I_rows":len(back["current_i_rows"]),"current_Fat_rows":len(back["current_fat_rows"]),
            "historical_Fat_rows":len(back["historical_fat_rows"]),"current_plus_historical_unique_Fat_rows":len(back["fat_rows"])},
        "protected_source_rows":protection,"baseline_quality":quality,
        "changed_source_rows_by_role":role_moved,"changed_volume_rows_by_role":volume_moved,
        "changed_global_point_rows":len(changed_global),
        "changed_global_rows_inside_allowed_free_set":True,"fixed_back_scope_rims_routes_unchanged":True,
        "native_quality_gate_unchanged":True,"formerly_good_cell_regressions":0,
        "positive_F32_cells_and_exact_target_boundary":True,
        "per_material_volume_changes":base_receipt["volumes"],
        "mass_reconciliation":"PENDING","physical_admission":False,"native_steps":0}
    args.output_report.parent.mkdir(parents=True,exist_ok=True)
    args.output_report.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"changed_global_points":len(changed_global),
        "allowed_global_points":len(allowed),"quality":quality,"changed_source_rows_by_role":role_moved},sort_keys=True))

if __name__=="__main__":main()
