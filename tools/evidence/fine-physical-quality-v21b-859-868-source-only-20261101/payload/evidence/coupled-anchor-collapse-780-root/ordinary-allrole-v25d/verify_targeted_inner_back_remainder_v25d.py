#!/usr/bin/env python3
"""Targeted v25b independent point replay and saved 805/625 resume helper.

This tool consumes retained reports and archives only. It never invokes an LP,
805 auditor, 625 preparer, or rollback solver.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

H=Path(__file__).resolve().parents[2]
TARGET=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25d/repair_coupled_quality_points_722_v25d_inner_back_remainder_blocklocal.py"
TARGET_SHA="299653fb0979d60a9e3d506cfa31750c2f97e786b2e9e8010d9f1aa399e9bd72"
V24_PATH=H/"continue_coupled_quality_points_v24c_ordinary_outer_o_local.py"
V24_SHA="f69ab97f020436c5f0afa997a770ad291bad9ee63942522d0ea0b3c49230074a"
V25_REPLAY=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25/verify_ordinary_allrole_v25_independent_replay_v2.py"
V25_REPLAY_SHA="e38d383229087ff6b65fb8d741262f4e692f6f2216df127b0c95eff706fe037d"
MAX_CAP_M=5.e-5
ZERO_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
SCOPE_PATH=H/"current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
SCOPE_SHA="1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676"
ROUTES_PATH=H/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
ROUTES_SHA="ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def imp(name,path,expected):
    got=sha(path)
    if got!=expected:raise ValueError(name+" source SHA mismatch: "+got)
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import "+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

V=imp("v24c_owner_for_target_replay",V24_PATH,V24_SHA)
R=imp("v25_independent_owner_for_target_replay",V25_REPLAY,V25_REPLAY_SHA)
TGT=imp("target_builder_for_independent_replay",TARGET,TARGET_SHA)
BACK=R.R
T=R.M.T
if sha(SCOPE_PATH)!=SCOPE_SHA:raise ValueError("pinned current back scope changed")
if sha(ROUTES_PATH)!=ROUTES_SHA:raise ValueError("pinned route rows changed")

def state(path):
    return V.load_state(Path(path).resolve())

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def scalar_text(value,field,archive):
    return TGT._typed_archive_scalar_text(value,field,archive)

def _scope_and_routes(parent,surfaces,smap):
    scope=BACK.load_npz(BACK.pin(SCOPE_PATH),"target independent back scope")
    sr=json.loads((SCOPE_PATH.parent/"report.json").read_text())
    if sr.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT":
        raise ValueError("back scope receipt status mismatch")
    if sr.get("protected_mask_asset",{}).get("sha256")!=sha(SCOPE_PATH):
        raise ValueError("back scope report does not pin the current mask")
    for role in ("O","I","D"):
        ref=sr["current_sources"]["typed_roles"][role]["candidate"]
        got=scalar_text(scope[role+"_candidate_sha256"],role+"_candidate_sha256",SCOPE_PATH)
        if got!=ref["sha256"]:raise ValueError(role+" back scope source pin mismatch")
    routes=json.loads(ROUTES_PATH.read_text()).get("surface_routes")
    if not isinstance(routes,list):raise ValueError("protected route receipt has no route rows")
    scope_rows={};rim_route_rows={};route_counts={}
    for role in ("O","I","D"):
        a=surfaces[role];faces=np.asarray(a["faces"],np.int64);mp=np.asarray(smap[role],np.int64)
        rows=np.asarray(scope[role+"_vertex_rows"],np.int64)
        if rows.ndim!=1 or (rows.size and (rows.min()<0 or rows.max()>=len(mp))):
            raise ValueError(role+" protected scope row outside current typed domain")
        mapped=mp[rows]
        scope_rows[role]=set(map(int,mapped[mapped>=0]))
        edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
        ue,counts=np.unique(edges,axis=0,return_counts=True)
        rim_local=set(map(int,np.unique(ue[counts==1].reshape(-1))))
        local_routes=set()
        pp=np.asarray(a["points_world_m_f32"],np.float32)
        for rec in routes:
            if rec.get("role")!=role:continue
            hit=np.flatnonzero(np.all(pp==np.asarray(rec["target_world_m_f32"],np.float32),axis=1))
            if len(hit)!=1:raise ValueError(role+" route does not map to one current source point")
            local_routes.add(int(hit[0]))
        fixed_local=rim_local|local_routes
        fixed_global=mp[np.asarray(sorted(fixed_local),np.int64)] if fixed_local else np.empty(0,np.int64)
        if fixed_global.size and np.any(fixed_global<0):raise ValueError(role+" rim/route point has no parent volume mapping")
        rim_route_rows[role]=set(map(int,fixed_global))
        route_counts[role]={"scope_rows":len(scope_rows[role]),"rim_rows":len(rim_local),
                            "route_rows":len(local_routes)}
    return scope,scope_rows,rim_route_rows,route_counts

def _lineage(parent,surfaces,volumes):
    # v25's independent 795 replay reconstructs current+historical rows from the
    # pinned back-scope report, typed source ancestry, and exact F32 crosswalk.
    return R.independent_back_lineage(parent,surfaces,volumes)

def recompute_target_mask(parent_state_path):
    parent=state(parent_state_path)
    surfaces,volumes,points,tets,vmap,smap=R.build_parent_domain(parent)
    back=_lineage(parent,surfaces,volumes)
    if len(back["current_i_rows"])!=790 or len(back["fat_rows"])!=790:
        raise ValueError("independent 795 lineage does not reconstruct 790 actual back rows")
    scope,scope_rows,rim_route_rows,anchor_counts=_scope_and_routes(parent,surfaces,smap)

    # The reference mask is independently reconstructed with the same role
    # boundaries as the ordinary v25 pass: all scoped rows, all rims/routes,
    # and the current+historical actual Fat back are fixed.
    ordinary_fixed=set()
    for role in ("O","I","D"):
        ordinary_fixed.update(scope_rows[role])
        ordinary_fixed.update(rim_route_rows[role])
    current_i_global=np.asarray(smap["I"][back["current_i_rows"]],np.int64)
    if current_i_global.size and np.any(current_i_global<0):
        raise ValueError("current authenticated back I rows do not map into the parent volume")
    back_global=np.asarray(vmap["Fat"][back["fat_rows"]],np.int64)
    if back_global.size and np.any(back_global<0):
        raise ValueError("authenticated back Fat rows do not map into the parent volume")
    ordinary_fixed.update(map(int,current_i_global))
    ordinary_fixed.update(map(int,back_global))

    det,q=R.M.M.measurements(points,tets)
    passed=np.asarray(R.M.M.passes(det,q),bool)
    if np.any(np.asarray(det)<=0):raise ValueError("parent contains a nonpositive cell")
    bad_cells=np.flatnonzero(~passed)
    bad_corners=set(map(int,np.unique(tets[bad_cells].reshape(-1)))) if len(bad_cells) else set()
    mutable_seeds=bad_corners-ordinary_fixed
    ordinary_ring=set()
    role_faces={}
    for role in ("D","O","I"):
        faces=np.asarray(surfaces[role]["faces"],np.int64);mp=np.asarray(smap[role],np.int64)
        local_seed=np.flatnonzero(np.isin(mp,np.fromiter(mutable_seeds,dtype=np.int64))) if mutable_seeds else np.empty(0,np.int64)
        incident=np.any(np.isin(faces,local_seed),axis=1) if len(local_seed) else np.zeros(len(faces),bool)
        local_ring=np.unique(faces[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
        mapped=mp[local_ring]
        if mapped.size and np.any(mapped<0):raise ValueError(role+" ordinary reference ring is unmapped")
        ordinary_ring.update(map(int,mapped));role_faces[role]=faces
    ordinary_free=(bad_corners|ordinary_ring)-ordinary_fixed
    ordinary_free_arr=np.fromiter(ordinary_free,dtype=np.int64) if ordinary_free else np.empty(0,np.int64)
    if len(bad_cells):
        selected=bad_cells[~np.any(np.isin(tets[bad_cells],ordinary_free_arr),axis=1)]
    else:
        selected=np.empty(0,np.int64)
    selected_corners=set(map(int,np.unique(tets[selected].reshape(-1)))) if len(selected) else set()

    # Target mode keeps every O point fixed and also fixes I/D rims/routes.
    o_map=np.asarray(smap["O"],np.int64)
    o_all=set(map(int,o_map[o_map>=0]))
    target_fixed=o_all
    target_fixed.update(rim_route_rows["I"]);target_fixed.update(rim_route_rows["D"])
    target_fixed.update(rim_route_rows["O"])

    all_surface=set()
    inner_used=set()
    for role in ("O","I","D"):
        mp=np.asarray(smap[role],np.int64)
        all_surface.update(map(int,mp[mp>=0]))
        if role in ("I","D") and len(role_faces[role]):
            active=np.unique(role_faces[role].reshape(-1))
            inner_used.update(map(int,mp[active]))
    selected_inner_seed=(selected_corners&inner_used)-target_fixed
    direct_ring=set()
    ring_counts={}
    for role in ("I","D"):
        faces=role_faces[role];mp=np.asarray(smap[role],np.int64)
        local_seed=np.flatnonzero(np.isin(mp,np.fromiter(selected_inner_seed,dtype=np.int64))) if selected_inner_seed else np.empty(0,np.int64)
        incident=np.any(np.isin(faces,local_seed),axis=1) if len(local_seed) else np.zeros(len(faces),bool)
        local_ring=np.unique(faces[incident].reshape(-1)) if np.any(incident) else np.empty(0,np.int64)
        mapped=mp[local_ring]
        if mapped.size and np.any(mapped<0):raise ValueError(role+" target I/D ring is unmapped")
        direct_ring.update(map(int,mapped))
        ring_counts[role]={"selected_seed_source_rows":int(len(local_seed)),
            "incident_source_faces":int(np.count_nonzero(incident)),
            "direct_ring_global_rows":int(len(np.unique(mapped)))}
    interior=(selected_corners-all_surface)-target_fixed
    corner_arr=np.fromiter(selected_corners,dtype=np.int64) if selected_corners else np.empty(0,np.int64)
    star_cells=np.flatnonzero(np.any(np.isin(tets,corner_arr),axis=1)) if len(corner_arr) else np.empty(0,np.int64)
    star_nodes=set(map(int,np.unique(tets[star_cells].reshape(-1)))) if len(star_cells) else set()
    back_star=(set(map(int,back_global))&star_nodes)-target_fixed
    allowed=(selected_inner_seed|direct_ring|interior|back_star)-target_fixed

    group=np.concatenate((np.zeros(len(volumes["Skin"]["tetrahedra"]),np.int8),
                         np.ones(len(volumes["Fat"]["tetrahedra"]),np.int8)))
    groups={"Skin":int(np.count_nonzero(group[selected]==0)),"Fat":int(np.count_nonzero(group[selected]==1))}
    receipt={"status":"PASS_INDEPENDENT_TARGET_MASK_REPLAY",
        "parent_state":pin(parent_state_path),"bad_cells":int(len(bad_cells)),
        "ordinary_reference_free_rows":int(len(ordinary_free)),
        "ordinary_bad_cells_with_any_free_corner":int(len(bad_cells)-len(selected)),
        "ordinary_bad_cells_with_zero_free_corners":int(len(selected)),
        "selected_bad_cells_by_role":groups,"selected_bad_corner_rows":int(len(selected_corners)),
        "I_D_selected_seed_rows":int(len(selected_inner_seed)),
        "I_D_direct_ring_rows_before_exclusions":int(len(direct_ring)),
        "selected_interior_corner_rows":int(len(interior)),
        "actual_back_Fat_rows_in_selected_corner_star":int(len(back_star)),
        "selected_corner_star_tetrahedra":int(len(star_cells)),
        "target_allowed_global_rows":int(len(allowed)),
        "fixed_O_global_rows":int(len(o_all)),
        "fixed_I_D_rim_route_global_rows":int(len(rim_route_rows["I"]|rim_route_rows["D"])),
        "actual_back_lineage":{"current_I_rows":len(back["current_i_rows"]),
            "current_Fat_rows":len(back["current_fat_rows"]),
            "historical_Fat_rows":len(back["historical_fat_rows"]),
            "current_plus_historical_unique_Fat_rows":len(back["fat_rows"]),
            "lineage":back["lineage"]},
        "anchor_counts":anchor_counts,"role_ring_counts":ring_counts,
        "source_owner":{"builder":pin(TARGET),"v25_independent_replay":pin(V25_REPLAY),
                        "v24_replay_owner":pin(V24_PATH)}}
    return receipt,allowed,back,points,tets,vmap,smap,surfaces,volumes,scope

def _verify_back_asset(candidate,back):
    repair=candidate.get("coupled_point_repair",{})
    item=repair.get("actual_back_Fat_lineage_protection")
    if not isinstance(item,dict):raise ValueError("candidate has no actual-back lineage receipt")
    asset=item.get("row_asset")
    if not isinstance(asset,dict) or sha(asset.get("path",""))!=asset.get("sha256"):
        raise ValueError("candidate actual-back row asset pin mismatch")
    with np.load(asset["path"],allow_pickle=False) as z:
        for key,expected in (("current_i_surface_rows",back["current_i_rows"]),
            ("current_fat_volume_rows",back["current_fat_rows"]),
            ("historical_fat_volume_rows",back["historical_fat_rows"]),
            ("current_plus_historical_fat_volume_rows",back["fat_rows"])):
            if key not in z or not np.array_equal(np.asarray(z[key],np.int64),np.asarray(expected,np.int64)):
                raise ValueError("candidate actual-back asset disagrees with independent 795 mapping: "+key)
    return {"asset":asset,"current_i_rows":len(back["current_i_rows"]),
        "current_fat_rows":len(back["current_fat_rows"]),
        "historical_fat_rows":len(back["historical_fat_rows"]),
        "unique_Fat_back_rows":len(back["fat_rows"])}

def _compare_target_policy(candidate,mask):
    repair=candidate.get("coupled_point_repair",{})
    if repair.get("inner_back_remainder_repair") is not True:
        raise ValueError("candidate is not declared as the targeted remainder mode")
    if repair.get("block_local_target_repair") is not True:
        raise ValueError("candidate is not declared as the block-local target mode")
    if repair.get("inner_back_layer_repair") is not True:
        raise ValueError("candidate lacks existing 805 INNER_BACK_LAYER policy")
    cap=float(repair.get("maximum_displacement_m",0.0))
    if not 0<cap<=MAX_CAP_M:raise ValueError("candidate exceeds the 50 micrometre trust cap")
    producer=repair.get("producer",{})
    if Path(producer.get("path","")).resolve()!=TARGET.resolve() or producer.get("sha256")!=TARGET_SHA:
        raise ValueError("candidate producer is not the pinned targeted v25d builder")
    if repair.get("inner_back_remainder_policy")!="only native-bad cells with zero ordinary-v25-free corners; selected bad I/D corners plus one I/D source-face ring, selected volume-interior corners, and authenticated actual-back Fat rows in selected-corner tet star; O source/back, rims/routes fixed":
        raise ValueError("candidate lacks exact targeted remainder policy")
    witness=repair.get("exact_witness_conditioning",{})
    if isinstance(witness,dict) and witness.get("reports"):
        raise ValueError("this verifier only qualifies the first target pass without historical witness reports")
    info=candidate.get("source_triangle_one_ring_expansion",{}).get("inner_back_remainder")
    if not isinstance(info,dict):raise ValueError("candidate lacks target free-set receipt")
    pairs={
        "selected_bad_cells":mask["ordinary_bad_cells_with_zero_free_corners"],
        "selected_bad_corner_rows":mask["selected_bad_corner_rows"],
        "I_D_selected_seed_rows":mask["I_D_selected_seed_rows"],
        "I_D_direct_ring_rows_before_exclusions":mask["I_D_direct_ring_rows_before_exclusions"],
        "selected_interior_corner_rows":mask["selected_interior_corner_rows"],
        "authenticated_actual_back_Fat_rows_in_selected_corner_star":mask["actual_back_Fat_rows_in_selected_corner_star"],
        "selected_corner_star_tetrahedra":mask["selected_corner_star_tetrahedra"],
        "free_global_rows":mask["target_allowed_global_rows"],
        "ordinary_reference_free_rows":mask["ordinary_reference_free_rows"],
        "ordinary_reference_bad_cells_with_any_free_corner":mask["ordinary_bad_cells_with_any_free_corner"],
        "ordinary_reference_bad_cells_with_zero_free_corners":mask["ordinary_bad_cells_with_zero_free_corners"],
    }
    for key,expected in pairs.items():
        if int(info.get(key,-1))!=int(expected):
            raise ValueError("candidate target-mask count disagrees with independent replay: "+key)
    if info.get("selected_bad_cells_by_quality_role")!=mask["selected_bad_cells_by_role"]:
        raise ValueError("candidate Skin/Fat residual roster counts disagree")
    return {"maximum_displacement_m":cap,"independently_matched_target_mask_counts":pairs}

def _row_changes(base,candidate,cap,label):
    bp=np.asarray(base["points_world_m_f32"],np.float32)
    cp=np.asarray(candidate["points_world_m_f32"],np.float32)
    if bp.dtype!=np.float32 or cp.dtype!=np.float32 or bp.shape!=cp.shape or bp.ndim!=2 or bp.shape[1]!=3:
        raise ValueError(label+" F32 point domain mismatch")
    if not np.isfinite(bp).all() or not np.isfinite(cp).all():raise ValueError(label+" has nonfinite coordinates")
    moved=np.flatnonzero(np.any(bp.view(np.uint32)!=cp.view(np.uint32),axis=1))
    l1=np.abs(cp[moved].astype(np.float64)-bp[moved].astype(np.float64)).sum(axis=1)
    maximum=float(l1.max(initial=0.0))
    if np.any(l1>float(cap)):raise ValueError(label+" moved row exceeds 50 micrometre L1 cap")
    return moved,maximum

def assert_target_rows_allowed(changed_global_rows,allowed_global_rows):
    changed=set(map(int,changed_global_rows));allowed=set(map(int,allowed_global_rows))
    extra=changed-allowed
    if extra:raise ValueError("source/volume point changes exceed independently rebuilt target set: "+str(sorted(extra)[:24]))
    return True

def assert_outer_o_f32_unchanged(parent_o,candidate_o,parent_skin,candidate_skin,o_skin_rows):
    po=np.asarray(parent_o,dtype=np.float32);co=np.asarray(candidate_o,dtype=np.float32)
    ps=np.asarray(parent_skin,dtype=np.float32);cs=np.asarray(candidate_skin,dtype=np.float32)
    rows=np.asarray(o_skin_rows,dtype=np.int64)
    if po.shape!=co.shape or not np.array_equal(po.view(np.uint32),co.view(np.uint32)):
        raise ValueError("outer O source points are not bitwise unchanged")
    if ps.shape!=cs.shape or (rows.size and not np.array_equal(ps[rows].view(np.uint32),cs[rows].view(np.uint32))):
        raise ValueError("outer O volume points changed")
    return True

def verify_target_geometry(candidate_path,parent_path,volume_replay):
    candidate=state(candidate_path)
    parent=state(parent_path)
    mask,allowed,back,points,tets,vmap,smap,surfaces,volumes,scope=recompute_target_mask(parent_path)
    policy=_compare_target_policy(candidate,mask)
    back_receipt=_verify_back_asset(candidate,back)
    repair=candidate.get("coupled_point_repair",{})
    cap=float(repair["maximum_displacement_m"])
    changed_global=set();source_counts={};volume_counts={}
    candidate_surfaces={}
    for role in ("D","O","I"):
        e=candidate["source_surfaces"][role]
        if sha(e["path"])!=e["sha256"]:raise ValueError(role+" candidate source pin mismatch")
        ca=load_npz(e["path"]);candidate_surfaces[role]=ca
        ba=surfaces[role]
        moved,max_l1=_row_changes(ba,ca,cap,role+" source")
        if role=="O" and len(moved):
            raise ValueError("outer O source points are not bitwise unchanged")
        if len(moved):
            rows=np.asarray(smap[role][moved],np.int64)
            if np.any(rows<0):raise ValueError(role+" changed source row has no baseline volume identity")
            changed_global.update(map(int,rows))
        source_counts[role]={"moved_F32_rows":int(len(moved)),"maximum_L1_displacement_m":max_l1}
    # O geometry is fixed in both the typed O source and its current Skin volume
    # realization. The latter protects against a volume-only point mutation.
    o_global=np.asarray(smap["O"],np.int64)
    o_global=o_global[o_global>=0]
    skin_inverse={int(g):i for i,g in enumerate(np.asarray(vmap["Skin"],np.int64))}
    if any(int(g) not in skin_inverse for g in o_global):
        raise ValueError("an O source point is absent from the Skin volume domain")
    o_skin_rows=np.asarray([skin_inverse[int(g)] for g in o_global],np.int64)
    skin_candidate=load_npz(candidate["Skin_current"]["path"])
    skin_base=volumes["Skin"]
    assert_outer_o_f32_unchanged(
        surfaces["O"]["points_world_m_f32"],candidate_surfaces["O"]["points_world_m_f32"],
        skin_base["points_world_m_f32"],skin_candidate["points_world_m_f32"],o_skin_rows)
    for role in ("Skin","Fat"):
        e=candidate[role+"_current"]
        if sha(e["path"])!=e["sha256"]:raise ValueError(role+" candidate volume pin mismatch")
        ca=load_npz(e["path"]);ba=volumes[role]
        moved,max_l1=_row_changes(ba,ca,cap,role+" volume")
        rows=np.asarray(vmap[role][moved],np.int64)
        if rows.size and np.any(rows<0):raise ValueError(role+" changed volume row has no baseline identity")
        changed_global.update(map(int,rows))
        volume_counts[role]={"moved_F32_rows":int(len(moved)),"maximum_L1_displacement_m":max_l1}
    assert_target_rows_allowed(changed_global,allowed)
    # The native replay already proves exact positive determinants, no good-cell
    # regressions, full oriented boundary, typed aliases, and rollback ancestry.
    if volume_replay is not None and volume_replay.get("status")!="PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise ValueError("generic independent volume replay is not PASS")
    return {"status":("PASS_TARGETED_REMAINDER_MASK_AND_POINT_SCOPE_REPLAY" if volume_replay is not None
                      else "PASS_TARGETED_REMAINDER_SCOPE_ONLY_NO_GENERIC_REPLAY"),
        "target_mask":mask,"target_policy":policy,"authenticated_back_rows":back_receipt,
        "source_changes":source_counts,"volume_changes":volume_counts,
        "changed_global_rows":len(changed_global),"changed_rows_subset_of_target_mask":True,
        "outer_O_source_and_volume_F32_bitwise_unchanged":True,
        "rims_routes_and_unselected_actual_back_rows_unchanged":True,
        "native_and_oriented_boundary_gates":"inherited from pinned v24c independent replay",
        "mass_reconciliation":"PENDING","physical_admission":False,"native_steps":0}

def _check_candidate_pin(candidate,parent_path):
    meta=candidate.get("coupled_point_repair",{})
    if meta.get("inner_back_remainder_repair") is not True:raise ValueError("candidate lacks targeted remainder policy")
    V.validate_proposal_parent_pin(candidate,Path(parent_path).resolve())

def _audit_summary(report_path,config_path):
    V.validate_report_config_pin(Path(report_path).resolve(),Path(config_path).resolve())
    return V.audit_counts(Path(report_path).resolve())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path)
    ap.add_argument("--proposal-config",type=Path)
    ap.add_argument("--proposal-audit",type=Path)
    ap.add_argument("--candidate-state",type=Path)
    ap.add_argument("--candidate-scope-preflight",action="store_true")
    ap.add_argument("--rollback-state",type=Path)
    ap.add_argument("--rollback-config",type=Path)
    ap.add_argument("--rollback-plan",type=Path)
    ap.add_argument("--rollback-preparation-receipt",type=Path)
    ap.add_argument("--rollback-audit",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--mask-preflight-only",action="store_true")
    args=ap.parse_args()
    parent_path=args.parent_state.resolve()
    out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if args.mask_preflight_only:
        receipt,_,_,_,_,_,_,_,_,_=recompute_target_mask(parent_path)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        print(json.dumps({k:receipt[k] for k in ("status","bad_cells","ordinary_reference_free_rows",
            "ordinary_bad_cells_with_zero_free_corners","selected_bad_cells_by_role","target_allowed_global_rows")},sort_keys=True))
        return
    if args.candidate_scope_preflight:
        if args.candidate_state is None:raise ValueError("candidate scope preflight requires --candidate-state")
        candidate_path=args.candidate_state.resolve()
        candidate=state(candidate_path)
        _check_candidate_pin(candidate,parent_path)
        receipt=verify_target_geometry(candidate_path,parent_path,None)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\\n")
        print(json.dumps({"status":receipt["status"],"changed_global_rows":receipt["changed_global_rows"],
            "target_allowed_rows":receipt["target_mask"]["target_allowed_global_rows"],
            "source_changes":receipt["source_changes"],"volume_changes":receipt["volume_changes"]},sort_keys=True))
        return
    if not (args.proposal_state and args.proposal_config and args.proposal_audit):
        raise ValueError("full replay requires proposal state, config, and saved 805 audit")
    proposal_path=args.proposal_state.resolve()
    parent=state(parent_path);proposal=state(proposal_path)
    V.verify_source_input(parent_path)
    _check_candidate_pin(proposal,parent_path)
    V._validate_context_config_endpoints(args.proposal_config.resolve(),parent_path,proposal_path)
    proposal_summary,proposal_clean=_audit_summary(args.proposal_audit,args.proposal_config)
    if proposal_clean and args.rollback_state:
        raise ValueError("proposal source audit is clean; rollback artifacts are inconsistent with the saved flow")
    rollback_args=(args.rollback_state,args.rollback_config,args.rollback_plan,args.rollback_preparation_receipt,args.rollback_audit)
    rollback_mode=any(x is not None for x in rollback_args)
    if rollback_mode and not all(x is not None for x in rollback_args):
        raise ValueError("rollback state/config/plan/preparation receipt/audit must be supplied together")
    out.mkdir(parents=True)
    if rollback_mode:
        stage=out/"generic-625-replay-stage"
        V.replay_existing_rollback(parent_path,proposal_path,args.proposal_config.resolve(),
            args.proposal_audit.resolve(),args.rollback_state.resolve(),args.rollback_config.resolve(),
            args.rollback_plan.resolve(),args.rollback_preparation_receipt.resolve(),
            args.rollback_audit.resolve(),stage)
        candidate_path=args.rollback_state.resolve()
        clean_audit_path=args.rollback_audit.resolve()
        candidate=state(candidate_path)
        _check_candidate_pin(candidate,parent_path)
        replay_path=stage/"independent-point-replay.json"
        replay=json.loads(replay_path.read_text())
        rollback_receipt=pin(stage/"recovery-receipt.json")
        final_audit_summary,clean=V.audit_counts(clean_audit_path)
        if not clean:raise ValueError("rollback audit is not clean")
    else:
        if not proposal_clean:
            raise ValueError("proposal 805 audit failed; complete 625 rollback before promotion")
        candidate_path=proposal_path;candidate=proposal;clean_audit_path=args.proposal_audit.resolve()
        replay_path=out/"independent-point-replay.json"
        replay=V.independent_replay(candidate_path,parent_path,replay_path,
            proposal_state_path=None,rollback_plan_path=None)
        rollback_receipt=None;final_audit_summary=proposal_summary;clean=True
    target=verify_target_geometry(candidate_path,parent_path,replay)
    if not clean:raise ValueError("no clean 805 report for the final candidate")
    final_path=out/"verified-state.json"
    V.finalize_verified_state(candidate_path,clean_audit_path,replay_path,final_path)
    final_evidence=V.verify_source_input(final_path)
    receipt={"schema":"h706.targeted-remainder-v25d-blocklocal-independent-resume.v1",
        "status":"PASS_TARGETED_REMAINDER_805_807_625_AND_INDEPENDENT_REPLAY",
        "parent_state":pin(parent_path),"proposal_state":pin(proposal_path),
        "proposal_audit":pin(args.proposal_audit),"proposal_audit_summary":proposal_summary,
        "rollback_used":bool(rollback_mode),"rollback_stage_receipt":rollback_receipt,
        "candidate_state":pin(candidate_path),"final_clean_805_audit":pin(clean_audit_path),
        "final_805_summary":final_audit_summary,"generic_independent_point_replay":pin(replay_path),
        "target_independent_replay":target,"verified_state":pin(final_path),
        "final_807_source_evidence":final_evidence,"solver_rerun":False,"auditor_rerun":False,
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    (out/"target-replay-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"rollback_used":rollback_mode,
        "changed_global_rows":target["changed_global_rows"],
        "target_allowed_rows":target["target_mask"]["target_allowed_global_rows"],
        "verified_state":receipt["verified_state"]},sort_keys=True))

if __name__=="__main__":main()
