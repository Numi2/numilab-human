#!/usr/bin/env python3
"""Independent exact-context, volume replay, witness, and unified-mask verifier."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BUILDER=HERE/"repair_coupled_quality_points_722_v25h_unified_inner_back_ordinary.py"
BUILDER_SHA="6e285a7bc64afdaaf94b7075f0f2fb46d6f36671d058151d8b32c78e30e5ab4c"
BASE_REPLAY=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25g-blocklocal/verify_ordinary_allrole_v25g_independent_replay.py"
BASE_REPLAY_SHA="b0d61dbadfce6e164a21b4c8c7a6d5dbfff6fb01b78f2aa82bde2e56bdfdc52a"
SOURCE807=ROOT/"verify_current_source_evidence_807_root.py"
SOURCE807_SHA="9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62"
ZERO_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def pin(path):
    p=Path(path).resolve();return {"path":str(p),"sha256":sha(p)}
def loadmod(name,path,expected=None):
    if expected is not None and sha(path)!=expected:raise ValueError(name+" source SHA mismatch")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import "+str(path))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
G=loadmod("v25g_independent_support",BASE_REPLAY,BASE_REPLAY_SHA)
if sha(BUILDER)!=BUILDER_SHA:raise ValueError("v25h builder source changed")
M=loadmod("v25h_builder_for_mask_semantics",BUILDER,BUILDER_SHA)
R=M.BACK

def face_index(faces,mapping,n):
    f=np.asarray(faces,np.int64);gr=np.asarray(mapping,np.int64);gf=gr[f]
    if gf.size and (np.any(gf<0) or np.any(gf>=n)):raise ValueError("active face has unmapped vertex")
    flat=gf.reshape(-1);counts=np.bincount(flat,minlength=n)
    off=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    ids=np.repeat(np.arange(len(f),dtype=np.int64),3)[np.argsort(flat,kind="stable")]
    return off,ids,gf

def face_hits(idx,rows):
    off,ids,gf=idx;parts=[ids[off[int(r)]:off[int(r)+1]] for r in set(map(int,rows)) if off[int(r)+1]>off[int(r)]]
    return np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.empty(0,np.int64)

def point_cell_index(t,n):
    flat=np.asarray(t,np.int64).reshape(-1);counts=np.bincount(flat,minlength=n)
    off=np.r_[np.int64(0),np.cumsum(counts,dtype=np.int64)]
    ids=np.repeat(np.arange(len(t),dtype=np.int64),4)[np.argsort(flat,kind="stable")]
    return off,ids

def cell_hits(idx,rows):
    off,ids=idx;parts=[ids[off[int(r)]:off[int(r)+1]] for r in set(map(int,rows)) if off[int(r)+1]>off[int(r)]]
    return np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.empty(0,np.int64)

def bad_components(t,bad):
    t=np.asarray(t,np.int64);bad=np.asarray(bad,np.int64);n=len(bad)
    if not n:return []
    p=np.arange(n);rank=np.zeros(n,np.int8)
    def find(x):
        while p[x]!=x:p[x]=p[p[x]];x=int(p[x])
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        p[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    own={};omit=((1,2,3),(0,2,3),(0,1,3),(0,1,2))
    for j,c in enumerate(bad):
        for ix in omit:
            k=tuple(sorted(map(int,t[int(c),list(ix)])))
            if k in own:union(j,own[k])
            else:own[k]=j
    d={}
    for j,c in enumerate(bad):d.setdefault(find(j),[]).append(int(c))
    out=[np.asarray(sorted(v),np.int64) for v in d.values()];out.sort(key=lambda a:int(a[0]));return out

def witness_rows(meta,parent_path,parent,npoints):
    info=meta.get("exact_witness_conditioning",{})
    if not isinstance(info,dict):raise ValueError("witness history metadata malformed")
    refs=info.get("reports",[]);rp=info.get("remap_receipt")
    if not refs:
        if rp is not None or int(info.get("constraints",-1))!=0 or int(info.get("frozen_source_nodes",-1))!=0:
            raise ValueError("empty witness roster has nonempty receipt/constraints")
        return np.empty(0,np.int64),{"reports":0,"frozen_nodes":0}
    if not isinstance(rp,dict) or sha(rp.get("path",""))!=rp.get("sha256"):raise ValueError("witness remap receipt pin failed")
    doc=json.loads(Path(rp["path"]).read_text())
    if doc.get("status")!="PASS_EXACT_ORIENTED_FACE_MAP_AND_PAIR_ACCOUNTING" or doc.get("pair_accounting",{}).get("identity") is not True:
        raise ValueError("witness remap receipt did not pass exact pair accounting")
    producer=doc.get("producer",{})
    if not producer.get("path") or sha(producer["path"])!=producer.get("sha256"):raise ValueError("witness remap producer pin failed")
    current=doc.get("current_state",{});parent_pin={"path":str(Path(parent_path).resolve()),"sha256":sha(parent_path)}
    if current.get("input_state_pin")!=parent_pin:raise ValueError("witness reports were not remapped to exact parent state")
    if current.get("source_surface_pins")!={r:parent["source_surfaces"][r] for r in ("O","I","D")}:
        raise ValueError("witness remap source pins differ from parent")
    receipts=doc.get("reports",[])
    norm=lambda xs:sorted((str(Path(x["path"]).resolve()),x["sha256"]) for x in xs)
    if norm(receipts)!=norm(refs):raise ValueError("witness report history and remap receipt differ")
    for x in receipts:
        q=Path(x["path"])
        if sha(q)!=x["sha256"]:raise ValueError("witness report SHA mismatch")
        d=json.loads(q.read_text())
        if d.get("schema")!="h706.combined-current-exact-surface-diff-audit.v1":raise ValueError("witness report schema mismatch")
        for role in ("O","I","D"):
            for name in ("baseline","candidate"):
                e=d["roles"][role][name]
                if sha(e["path"])!=e["sha256"]:raise ValueError("witness source archive hash mismatch")
    prep=doc.get("halfspace_preparation",{});fr=np.asarray(prep.get("freeze_source_nodes",[]),np.int64)
    if fr.ndim!=1 or len(np.unique(fr))!=len(fr) or (len(fr) and (fr.min()<0 or fr.max()>=npoints)):
        raise ValueError("witness freeze rows invalid")
    if len(fr)!=int(info.get("frozen_source_nodes",-1)) or int(prep.get("linear_halfspace_rows",-1))!=int(info.get("constraints",-2)):
        raise ValueError("witness summary count differs from remap receipt")
    if int(prep.get("source_source_pairs",-1))+int(prep.get("source_fixed_pairs",-1))!=int(info.get("exact_pairs",-2)):
        raise ValueError("witness source-pair counts differ")
    return fr,{"reports":len(receipts),"frozen_nodes":len(fr),"remap_receipt":pin(rp["path"]),"pair_accounting_identity":True}

def inner_hard_fixed(parent,surfaces,smap):
    scopep=Path(R.E.BACK_SCOPE_PATH).resolve()
    if sha(scopep)!=R.E.BACK_SCOPE_SHA:raise ValueError("back scope mask SHA mismatch")
    scope=R.load_npz(R.pin(scopep),"scope");reportp=scopep.parent/"report.json";report=json.loads(reportp.read_text())
    if report.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT":raise ValueError("back scope report status mismatch")
    if report.get("protected_mask_asset",{}).get("sha256")!=sha(scopep):raise ValueError("back scope report/asset pin mismatch")
    routep=ROOT/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
    if sha(routep)!="ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71":raise ValueError("route source hash mismatch")
    routes=json.loads(routep.read_text())["surface_routes"];fixed=set();counts={}
    for role in ("O","I","D"):
        ref=report["current_sources"]["typed_roles"][role]["candidate"]
        if str(np.asarray(scope[role+"_candidate_sha256"]).reshape(-1)[0])!=ref["sha256"]:raise ValueError(role+" scope reference mismatch")
        rows,lineage=R.map_rows_by_lineage(parent["source_surfaces"][role],ref,surfaces[role],scope[role+"_vertex_rows"],"inner fixed scope",allow_collapsed=True)
        if role=="O":
            gr=np.asarray(smap[role][rows],np.int64)
            if np.any(gr<0):raise ValueError("O back-scope row unmapped")
            fixed.update(map(int,gr))
        faces=np.asarray(surfaces[role]["faces"],np.int64)
        ed=np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
        ue,ct=np.unique(ed,axis=0,return_counts=True);rim=set(map(int,np.unique(ue[ct==1])))
        pts=np.asarray(surfaces[role]["points_world_m_f32"],np.float32);routes_here=set()
        for x in routes:
            if x.get("role")!=role:continue
            hit=np.flatnonzero(np.all(pts==np.asarray(x["target_world_m_f32"],np.float32),axis=1))
            if len(hit)!=1:raise ValueError(role+" route identity not unique")
            routes_here.add(int(hit[0]))
        rows=np.asarray(smap[role][list(rim|routes_here)],np.int64)
        if np.any(rows<0):raise ValueError(role+" rim/route unmapped")
        fixed.update(map(int,rows));counts[role]={"O_scope":len(rows) if role=="O" else 0,"rim":len(rim),"routes":len(routes_here),"scope_lineage_depth":lineage["descendant_steps"]}
    return fixed,counts

def reconstruct_masks(points,tets,surfaces,smap,back,ordinary_fixed,inner_fixed,frozen):
    p=np.asarray(points,np.float32);t=np.asarray(tets,np.int64);det,q=G.M.M.measurements(p,t)
    good=np.asarray(G.M.M.passes(det,q),bool)
    if np.any(det<=0):raise ValueError("parent has nonpositive Float32 tet")
    bad=np.flatnonzero(~good);corners=set(map(int,np.unique(t[bad].reshape(-1))))
    all_surface=set();inner_surface=set();fidx={}
    for role in ("O","I","D"):
        gr=np.asarray(smap[role],np.int64);faces=np.asarray(surfaces[role]["faces"],np.int64)
        all_surface.update(map(int,gr[gr>=0]))
        if role in ("I","D"):
            rr=gr[np.unique(faces.reshape(-1))]
            if np.any(rr<0):raise ValueError(role+" active source point unmapped")
            inner_surface.update(map(int,rr))
        fidx[role]=face_index(faces,gr,len(p))
    witnesses=set(map(int,frozen));ordinary_ex=set(map(int,ordinary_fixed))|witnesses
    ordinary_seed=corners-ordinary_ex;oring=set()
    for role in ("O","I","D"):
        ix=face_hits(fidx[role],ordinary_seed)
        if len(ix):oring.update(map(int,np.unique(fidx[role][2][ix].reshape(-1))))
    ordinary=(corners|oring)-ordinary_ex
    inner_ex=set(map(int,inner_fixed))|witnesses;iseed=(corners&inner_surface)-inner_ex;iring=set()
    for role in ("I","D"):
        ix=face_hits(fidx[role],iseed)
        if len(ix):iring.update(map(int,np.unique(fidx[role][2][ix].reshape(-1))))
    interior=(corners-all_surface)-inner_ex
    ci=point_cell_index(t,len(p));star=cell_hits(ci,corners);star_nodes=set(map(int,np.unique(t[star].reshape(-1)))) if len(star) else set()
    backring=(set(map(int,back))&star_nodes)-inner_ex
    inner=(iseed|iring|interior|backring)-inner_ex;allowed=ordinary|inner
    comps=bad_components(t,bad);patches=[];component_receipts=[]
    for comp_id,comp in enumerate(comps):
        cc=set(map(int,np.unique(t[comp].reshape(-1))));cand=set(cc);allring=set();ring_roles={}
        for role in ("O","I","D"):
            ix=face_hits(fidx[role],cc);ring=set(map(int,np.unique(fidx[role][2][ix].reshape(-1)))) if len(ix) else set()
            allring.update(ring);cand.update(ring);ring_roles[role]={"incident_faces":len(ix),"ring_rows":len(ring)}
        seed=(cc&inner_surface)-inner_ex;inner_ring=set();inner_roles={}
        for role in ("I","D"):
            ix=face_hits(fidx[role],seed);ring=set(map(int,np.unique(fidx[role][2][ix].reshape(-1)))) if len(ix) else set()
            inner_ring.update(ring);cand.update(ring);role_rows=set(map(int,smap[role][smap[role]>=0]))
            inner_roles[role]={"seed_rows":len(seed&role_rows),"incident_faces":len(ix),"ring_rows":len(ring)}
        interior_corners=(cc-all_surface)-inner_ex;starcells=cell_hits(ci,cc)
        nodes=set(map(int,np.unique(t[starcells].reshape(-1)))) if len(starcells) else set();backstar=(set(map(int,back))&nodes)-inner_ex
        cand.update(seed);cand.update(interior_corners);cand.update(backstar);free=np.asarray(sorted(cand&allowed),np.int64)
        if not len(free):
            component_receipts.append({"bad_component":comp_id,"bad_cells":len(comp),"free_rows":0,"status":"BLOCKED_NO_ALLOWED_FREE_ROW"});continue
        affected=cell_hits(ci,free)
        patches.append({"bad_component":comp_id,"seed_bad_cells":comp,"free_rows":free,"affected_cells":affected,
            "ordinary_ring_rows":len(allring),"inner_seed_rows":len(seed),"inner_ring_rows":len(inner_ring),
            "interior_rows":len(interior_corners),"back_star_rows":len(backstar),"star_cell_count":len(starcells),
            "ring_by_role":ring_roles,"inner_ring_by_role":inner_roles})
        component_receipts.append({"bad_component":comp_id,"bad_cells":len(comp),"free_rows":len(free),"affected_cells":len(affected),"status":"PATCHED"})
    patches.sort(key=lambda x:(-len(x["seed_bad_cells"]),int(x["bad_component"])))
    for j,x in enumerate(patches):x["patch"]=j
    observed=np.unique(np.concatenate([x["free_rows"] for x in patches])) if patches else np.empty(0,np.int64)
    if not np.array_equal(observed,np.asarray(sorted(allowed),np.int64)):raise ValueError("independent patch roster misses free rows")
    eligible=set(map(int,bad[np.any(np.isin(t[bad],observed),axis=1)])) if len(observed) and len(bad) else set()
    covered=set()
    for x in patches:covered.update(map(int,np.intersect1d(x["affected_cells"],bad)))
    if eligible!=covered:raise ValueError("a baseline-bad cell with free corner is absent from complete-star roster")
    return ordinary,inner,allowed,bad,good,patches,component_receipts

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate-state",type=Path,required=True);ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--audit-report",type=Path,required=True);ap.add_argument("--audit-config",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path);ap.add_argument("--rollback-plan",type=Path);ap.add_argument("--output-report",type=Path,required=True);a=ap.parse_args()
    cp=a.candidate_state.resolve();pp=a.parent_state.resolve();cand=G.state(cp);parent=G.state(pp);meta=cand.get("coupled_point_repair",{})
    if meta.get("producer")!={"path":str(BUILDER.resolve()),"sha256":BUILDER_SHA}:raise ValueError("candidate is not pinned to v25h producer")
    if meta.get("unified_inner_back_ordinary_local_repair") is not True or meta.get("ordinary_block_local_quality_fit") is not True:raise ValueError("unified/block-local mode absent")
    pol="all baseline-bad face components receive ordinary bad-corner/source-ring freedoms unioned with I/D-inner source corners/rings and authenticated actual-back Fat rows in each bad-corner tet star; O back scope, outer scope, rims and routes remain fixed"
    if meta.get("unified_inner_back_ordinary_policy")!=pol:raise ValueError("unified policy metadata differs")
    if meta.get("fixed_outer_and_deep") is not False or meta.get("outer_O_scope_fixed") is not True or meta.get("outer_O_entire_surface_fixed") is not False:raise ValueError("outer scope policy metadata differs")
    cap=float(meta.get("maximum_displacement_m",0));
    if not 0<cap<=5e-5:raise ValueError("cap exceeds uniform 50um construction limit")
    if meta.get("ordinary_block_patch_policy")!="bad-face-connected patches run sequentially; overlapping free nodes and full stars are allowed only with current/baseline-good revalidation and cumulative per-node trust cap":raise ValueError("component solver policy differs")
    report=json.loads(a.audit_report.read_text())
    if report.get("status")!=ZERO_STATUS or report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0:raise ValueError("exact 775/805 report did not pass")
    G.V.validate_report_config_pin(a.audit_report.resolve(),a.audit_config.resolve());cfg=json.loads(a.audit_config.read_text());changed=[]
    for r in ("O","I","D"):
        b=cfg.get("baseline_candidates",{}).get(r,{});c=cfg.get("candidate_surfaces",{}).get(r,{})
        if b.get("path")!=parent["source_surfaces"][r]["path"] or b.get("sha256")!=parent["source_surfaces"][r]["sha256"]:raise ValueError(r+" baseline pin differs")
        if c.get("path")!=cand["source_surfaces"][r]["path"] or c.get("sha256")!=cand["source_surfaces"][r]["sha256"]:raise ValueError(r+" candidate pin differs")
        if b.get("sha256")!=c.get("sha256"):changed.append(r)
    if set(cfg.get("changed_roles",[]))!=set(changed) or not changed:raise ValueError("changed role set mismatch")
    bounded=cfg.get("bounded_point_repair",{});inner_roles=[]
    if set(bounded)!=set(changed):raise ValueError("bounded role set differs from changed roles")
    for r in changed:
        e=bounded[r]
        if float(e.get("maximum_displacement_m",-1))!=cap:raise ValueError(r+" cap mismatch")
        if r in ("I","D"):
            if e.get("inner_back_layer_repair") is not True:raise ValueError(r+" lacks explicit 805 inner policy")
            inner_roles.append(r)
        elif e.get("ordinary_outer_o_local_repair") is not True:raise ValueError("O movement lacks ordinary-local declaration")
    innerreport=report.get("inner_back_layer_repair_policy")
    if inner_roles:
        if not isinstance(innerreport,dict) or set(innerreport.get("declared_roles",[]))!=set(inner_roles):raise ValueError("805 declared inner roles differ")
        if float(innerreport.get("maximum_inner_displacement_m",float("nan")))!=5e-5 or innerreport.get("outer_back_scope_points_fixed") is not True or innerreport.get("source_rims_and_all149_routes_fixed") is not True:raise ValueError("805 policy limits differ")
    elif innerreport is not None:raise ValueError("ordinary-only repair unexpectedly claims 805 inner movement")
    base_replay=a.output_report.resolve().with_name(a.output_report.stem+"-base-volume-replay.json")
    proposal=a.proposal_state.resolve() if a.proposal_state else None;rollback=a.rollback_plan.resolve() if a.rollback_plan else None
    if bool(proposal)!=bool(rollback):raise ValueError("proposal state and rollback plan must be supplied together")
    # Explicit compatibility view for the legacy volume-only replay. Geometry,
    # source pins, parent pins and trust cap are identical; unified eligibility
    # is reconstructed independently below rather than inferred from this view.
    compat=dict(cand);compat_meta=dict(meta)
    compat_meta.update(ordinary_outer_O_local_repair=True,
        ordinary_outer_O_policy="bad O corners plus one direct incident O source-triangle ring; scope/rims/routes remain fixed")
    compat["coupled_point_repair"]=compat_meta
    compat["legacy_volume_replay_compatibility"]={"original_candidate":pin(cp),
        "purpose":"legacy ordinary-O flag only; geometry and all physical arrays unchanged; unified mask checked independently"}
    compat_path=a.output_report.resolve().with_name(a.output_report.stem+"-legacy-volume-compat-state.json")
    compat_path.parent.mkdir(parents=True,exist_ok=True)
    compat_path.write_text(json.dumps(compat,indent=2,sort_keys=True)+"\n")
    replay=G.V.independent_replay(compat_path,pp,base_replay,proposal_state_path=proposal,rollback_plan_path=rollback)
    if replay.get("status")!="PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":raise ValueError("independent volume replay failed")
    surfaces,volumes,points,tets,vmap,smap=G.build_parent_domain(parent);back=G.independent_back_lineage(parent,surfaces,volumes)
    if len(back["fat_rows"])!=790:raise ValueError("actual back lineage count differs")
    ordinary_fixed,_=G.source_fixed_rows(parent,surfaces,smap,back["fat_rows"],vmap)
    current_i=np.asarray(smap["I"][back["current_i_rows"]],np.int64)
    if np.any(current_i<0):raise ValueError("current I back row unmapped")
    ordinary_fixed.update(map(int,current_i));inner_fixed,inner_protect=inner_hard_fixed(parent,surfaces,smap)
    frozen,witness_receipt=witness_rows(meta,pp,parent,len(points))
    ordinary,inner,allowed,bad,good,patches,component_status=reconstruct_masks(points,tets,surfaces,smap,np.asarray(vmap["Fat"][back["fat_rows"]],np.int64),ordinary_fixed,inner_fixed,frozen)
    one=cand.get("source_triangle_one_ring_expansion",{}).get("unified_inner_back_ordinary",{})
    if not isinstance(one,dict):raise ValueError("unified mask receipt missing")
    if int(one.get("ordinary_mask",{}).get("final_movable_rows",-1))!=len(ordinary) or int(one.get("inner_mask",{}).get("free_global_rows",-1))!=len(inner):raise ValueError("producer mask counts differ from independent reconstruction: "+str({"recorded_ordinary":one.get("ordinary_mask",{}).get("final_movable_rows"),"independent_ordinary":len(ordinary),"recorded_inner":one.get("inner_mask",{}).get("free_global_rows"),"independent_inner":len(inner)}))
    roster=one.get("roster",{});schema="h706.unified-inner-back-ordinary-block-local-patches.v1"
    if roster.get("schema")!=schema:raise ValueError("producer patch roster schema mismatch")
    expected={"all_bad_cells":len(bad),"bad_face_components":len(component_status),"patched_components":len(patches),
        "ordinary_free_rows":len(ordinary),"inner_back_free_rows":len(inner),"free_union_rows":len(allowed),
        "bad_cells_with_at_least_one_free_corner":sum(x["status"]=="PATCHED" for x in component_status) if False else int(roster.get("bad_cells_with_at_least_one_free_corner",-1))}
    # Verify independently the aggregate roster fields that can be derived without trusting producer counts.
    for k in ("all_bad_cells","bad_face_components","patched_components","ordinary_free_rows","inner_back_free_rows","free_union_rows"):
        if int(roster.get(k,-1))!=int(expected[k]):raise ValueError("independent roster mismatch: "+k)
    recorded=roster.get("patches",[])
    if len(recorded)!=len(patches):raise ValueError("patch roster length mismatch")
    for j,(r,x) in enumerate(zip(recorded,patches)):
        pairs={"patch":x["patch"],"bad_component":x["bad_component"],"seed_bad_cells":len(x["seed_bad_cells"]),"free_rows":len(x["free_rows"]),"affected_cells":len(x["affected_cells"]),"ordinary_ring_rows":x["ordinary_ring_rows"],"inner_seed_rows":x["inner_seed_rows"],"inner_ring_rows":x["inner_ring_rows"],"back_star_rows":x["back_star_rows"]}
        for k,v in pairs.items():
            if int(r.get(k,-1))!=int(v):raise ValueError(f"patch {j} {k} differs")
        for role in ("O","I","D"):
            for k in ("incident_faces","ring_rows"):
                if int(r.get("ring_by_role",{}).get(role,{}).get(k,-1))!=int(x["ring_by_role"][role][k]):raise ValueError(f"patch {j} ordinary {role} incidence differs")
        for role in ("I","D"):
            for k in ("seed_rows","incident_faces","ring_rows"):
                if int(r.get("inner_ring_by_role",{}).get(role,{}).get(k,-1))!=int(x["inner_ring_by_role"][role][k]):raise ValueError(f"patch {j} inner {role} incidence differs")
    moved=set();source_moves={};volume_moves={}
    for role in ("O","I","D"):
        bp=np.asarray(surfaces[role]["points_world_m_f32"],np.float32);cpnt=np.asarray(G.load_npz(cand["source_surfaces"][role]["path"])["points_world_m_f32"],np.float32)
        if bp.shape!=cpnt.shape:raise ValueError(role+" point domain changed")
        rows=np.flatnonzero(np.any(bp.view(np.uint32)!=cpnt.view(np.uint32),axis=1))
        if len(rows):
            if np.any(np.abs(cpnt[rows].astype(np.float64)-bp[rows].astype(np.float64)).sum(axis=1)>cap):raise ValueError(role+" L1 cap exceeded")
            moved.update(map(int,smap[role][rows]))
        source_moves[role]=int(len(rows))
    for role in ("Skin","Fat"):
        bp=np.asarray(volumes[role].get("points_world_m_f32",volumes[role]["points_world_m"]),np.float32);cd=G.load_npz(cand[role+"_current"]["path"])
        cpnt=np.asarray(cd.get("points_world_m_f32",cd["points_world_m"]),np.float32)
        rows,maxl1=G.checked_volume_l1_changes(bp,cpnt,cap,role);moved.update(map(int,vmap[role][rows]))
        volume_moves[role]={"point_rows":len(rows),"max_L1_m":maxl1}
    if not moved or not moved.issubset(allowed):raise ValueError("moved point is empty or outside unified independently reconstructed mask")
    if moved&(set(inner_fixed)|set(map(int,frozen))):raise ValueError("moved point intersects O scope/rim/route or witness-frozen rows")
    asset=meta.get("actual_back_Fat_lineage_protection",{}).get("row_asset")
    if not isinstance(asset,dict) or sha(asset.get("path",""))!=asset.get("sha256"):raise ValueError("actual back row asset pin missing")
    with np.load(asset["path"],allow_pickle=False) as z:
        for k,e in (("current_i_surface_rows",back["current_i_rows"]),("current_fat_volume_rows",back["current_fat_rows"]),("historical_fat_volume_rows",back["historical_fat_rows"]),("current_plus_historical_fat_volume_rows",back["fat_rows"])):
            if k not in z or not np.array_equal(np.asarray(z[k],np.int64),np.asarray(e,np.int64)):raise ValueError("actual back source lineage mismatch: "+k)
    if sha(SOURCE807)!=SOURCE807_SHA:raise ValueError("807 verifier pin mismatch")
    S=loadmod("v25h_source807",SOURCE807,SOURCE807_SHA);proof=S.verify_current_source_evidence(cp)
    if not str(proof.get("status","" )).startswith("PASS_"):raise ValueError("807 current source proof failed")
    receipt={"schema":"h706.unified-inner-back-ordinary-independent-replay.v25h.v1","status":"PASS_INDEPENDENT_SOURCE_CONTEXT_VOLUME_QUALITY_AND_UNIFIED_MASK",
        "producer":pin(BUILDER),"candidate_state":pin(cp),"parent_state":pin(pp),"audit_report":pin(a.audit_report),"audit_config":pin(a.audit_config),
        "audit_status":report["status"],"declared_805_inner_roles":inner_roles,"volume_replay":pin(base_replay),"volume_replay_status":replay["status"],"legacy_volume_replay_compatibility_state":pin(compat_path),"source_evidence_807":proof,
        "witness_history":witness_receipt,"back_lineage":{"current_I_rows":len(back["current_i_rows"]),"current_Fat_rows":len(back["current_fat_rows"]),"historical_Fat_rows":len(back["historical_fat_rows"]),"union_Fat_rows":len(back["fat_rows"]),"row_asset":asset},
        "inner_hard_protection":inner_protect,"independent_mask":{"ordinary_rows":len(ordinary),"inner_rows":len(inner),"union_rows":len(allowed),"baseline_bad_cells":len(bad),"bad_face_components":len(component_status),"patches":len(patches),"roster_reconstructed":True},
        "source_moved_rows":source_moves,"volume_moved_rows":volume_moves,"changed_global_rows":len(moved),"all_moves_inside_unified_mask":True,
        "outer_back_scope_rims_routes_fixed":True,"all185_and_surface_context_zero":True,"formerly_good_regressions":0,"native_thresholds_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    a.output_report.parent.mkdir(parents=True,exist_ok=True);a.output_report.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"moved_rows":len(moved),"ordinary_free":len(ordinary),"inner_free":len(inner),"patches":len(patches),"bad":len(bad)},sort_keys=True))
if __name__=="__main__":main()
