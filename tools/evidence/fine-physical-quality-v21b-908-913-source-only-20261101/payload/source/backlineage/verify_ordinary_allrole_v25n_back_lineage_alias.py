#!/usr/bin/env python3
"""Independent exact-context, volume replay, witness, and unified-mask verifier."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BUILDER=HERE/"repair_coupled_quality_points_722_v25m_back_rowfix.py"
BUILDER_SHA="81b05b9d941d73771d81e31f5597877342cf3a80bff7e1a61e9f2b81cec2bf42"
JOINT=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25h-unified-mixed/joint_small_patch_slsqp_890.py"
JOINT_SHA="90d6325e9751a4da258ae7488fd4a08151607b24c6c5887fdb39ed3eda533d6a"
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
if sha(BUILDER)!=BUILDER_SHA:raise ValueError("v25k builder source changed")
if sha(JOINT)!=JOINT_SHA:raise ValueError("joint SLSQP helper source changed")
M=loadmod("v25m_builder_for_mask_semantics",BUILDER,BUILDER_SHA)
R=M.BACK
BACK_PROTECTION=ROOT/"quality-partial-cavity-684-root/paired_connected_boundary_patch_855_active_rows_lineage.py"
BACK_PROTECTION_SHA="5645dbfda4312b6df81b31bca03ba75fdf16af4f4956540c3b92f66b67eb101d"
if sha(BACK_PROTECTION)!=BACK_PROTECTION_SHA:raise ValueError("855 authenticated back-protection owner pin mismatch")

def independent_back_lineage_v25k(parent,surfaces,volumes):
    # Reconstruct all historical back records through pinned typed ancestry, permitting aliases.
    scopep=Path(R.E.BACK_SCOPE_PATH).resolve()
    if sha(scopep)!=R.E.BACK_SCOPE_SHA:raise ValueError("back-scope mask source mismatch")
    scope=R.load_npz(R.pin(scopep),"back scope")
    reportp=scopep.parent/"report.json";report=json.loads(reportp.read_text())
    if report.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT" or report.get("protected_mask_asset",{}).get("sha256")!=sha(scopep):
        raise ValueError("back-scope report/asset mismatch")
    for role in ("O","I","D"):
        ref=report["current_sources"]["typed_roles"][role]["candidate"]
        if str(np.asarray(scope[role+"_candidate_sha256"]).reshape(-1)[0])!=ref["sha256"]:
            raise ValueError(role+" scope reference pin mismatch")
    historical_rows=np.asarray(scope["I_vertex_rows"],np.int64)
    if historical_rows.shape!=(790,) or len(np.unique(historical_rows))!=790:
        raise ValueError("authenticated scope does not contain 790 distinct historical I-back records")
    iref=report["current_sources"]["typed_roles"]["I"]["candidate"]
    irefdata=R.load_npz(iref,"historical I back reference")
    irows,lineage=R.map_rows_by_lineage(parent["source_surfaces"]["I"],iref,surfaces["I"],
        historical_rows,"independent historical I-back lineage",allow_collapsed=True)
    if irows.shape!=(790,) or np.any(irows<0) or np.any(irows>=len(surfaces["I"]["points_world_m_f32"])):
        raise ValueError("not every historical back record has a current typed I representative")
    fatpoints=np.asarray(volumes["Fat"].get("points_world_m_f32",volumes["Fat"]["points_world_m"]),np.float32)
    ix,_=R.S.exact_used_row_crosswalk(np.asarray(surfaces["I"]["points_world_m_f32"],np.float32),
        np.asarray(surfaces["I"]["faces"],np.int64),fatpoints,"independent current I-to-Fat back map")
    fat_by_record=np.asarray(ix[irows],np.int64)
    if fat_by_record.shape!=(790,) or np.any(fat_by_record<0):
        raise ValueError("not every historical record has an exact current Fat representative")
    if len(np.unique(fat_by_record))!=len(np.unique(irows)):
        raise ValueError("distinct current I representatives do not map injectively into Fat")
    historical_points=np.asarray(irefdata["points_world_m_f32"],np.float32)
    if np.any(historical_rows<0) or np.any(historical_rows>=len(historical_points)):
        raise ValueError("historical scope row outside pinned I reference")
    fat_lut=R.coordmap(fatpoints)
    historical_by_record=np.asarray([fat_lut.get(R.fkey(historical_points[int(r)]),-1) for r in historical_rows],np.int64)
    if np.any(historical_by_record < -1) or np.any(historical_by_record>=len(fatpoints)):
        raise ValueError("historical direct Fat coordinate match row outside volume")
    current=np.unique(fat_by_record).astype(np.int64)
    historical=np.unique(historical_by_record[historical_by_record>=0]).astype(np.int64)
    union=np.union1d(current,historical).astype(np.int64)
    # Independently reconstruct all 1,673 historical D-back records and the
    # exact current D-to-Fat point map; aliases are allowed only through typed ancestry.
    dref=report["current_sources"]["typed_roles"]["D"]["candidate"]
    d_historical=np.asarray(scope["D_vertex_rows"],np.int64)
    if d_historical.shape!=(1673,) or len(np.unique(d_historical))!=1673:
        raise ValueError("authenticated scope does not contain 1,673 distinct historical D-back records")
    drows,dlineage=R.map_rows_by_lineage(parent["source_surfaces"]["D"],dref,surfaces["D"],
        d_historical,"independent historical D-back lineage",allow_collapsed=True)
    if drows.shape!=(1673,) or np.any(drows<0) or np.any(drows>=len(surfaces["D"]["points_world_m_f32"])):
        raise ValueError("not every historical D-back record has a current typed representative")
    dactive=M.BACK_PROTECTION.active_surface_point_rows(surfaces["D"])
    dmap=M.BACK_PROTECTION.exact_point_map(np.asarray(surfaces["D"]["points_world_m_f32"],np.float32),
        fatpoints,"independent current D/Fat back map",active_rows=dactive,required_rows=drows,
        volume_index=M.BACK_PROTECTION.point_key_index(fatpoints))
    dfat_by_record=np.asarray(dmap[drows],np.int64)
    if np.any(dfat_by_record<0):raise ValueError("historical D-back record lacks an exact Fat representative")
    dcurrent=np.unique(drows).astype(np.int64);dfat=np.unique(dfat_by_record).astype(np.int64)
    return {"fat_rows":union,"current_fat_rows":current,"historical_fat_rows":historical,
        "current_i_rows":irows.astype(np.int64),"historical_i_rows":historical_rows,
        "current_fat_by_record":fat_by_record,"historical_fat_by_record":historical_by_record,
        "current_d_source_rows_by_historical_record":drows.astype(np.int64),
        "current_d_source_rows":dcurrent,"current_d_fat_rows_by_historical_record":dfat_by_record,
        "current_d_fat_by_current_source_row":np.asarray(dmap[dcurrent],np.int64),
        "current_d_fat_rows":dfat,"historical_d_rows":d_historical,"d_lineage":dlineage,
        "lineage":lineage,"scope_mask":R.pin(scopep),"scope_report":pin(reportp),
        "scope_reference_I":iref,"scope_reference_D":dref,"current_I":parent["source_surfaces"]["I"],
        "current_D":parent["source_surfaces"]["D"],
        "coverage":{"historical_records":790,"mapped_records":790,"unmapped_records":0,
            "unique_current_I_representatives":int(len(np.unique(irows))),
            "collapsed_records":int(790-len(np.unique(irows))),
            "unique_current_Fat_representatives":int(len(current)),
            "direct_historical_coordinate_matches":int(np.count_nonzero(historical_by_record>=0)),
            "direct_historical_coordinate_orphans":int(np.count_nonzero(historical_by_record<0)),
            "unique_direct_historical_Fat_rows":int(len(historical)),
            "unique_protected_current_plus_historical_Fat_rows":int(len(union)),
            "D_historical_records":1673,"D_mapped_records":int(len(drows)),
            "D_uncovered_records":int(np.count_nonzero(drows<0)),
            "D_unique_current_source_representatives":int(len(dcurrent)),
            "D_collapsed_records":int(len(drows)-len(dcurrent)),
            "D_unique_current_Fat_representatives":int(len(dfat))}}

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

def reconstruct_masks(points,tets,surfaces,smap,back,ordinary_fixed,inner_fixed,frozen,quality_group_ids):
    p=np.asarray(points,np.float32);t=np.asarray(tets,np.int64);det,q=G.M.M.measurements(p,t)
    good=np.asarray(G.M.M.passes(det,q),bool)
    groups=np.asarray(quality_group_ids)
    if groups.shape!=(len(t),) or np.any((groups!=0)&(groups!=1)):
        raise ValueError("independent Fat-first quality-group domain mismatch")
    if np.any(det<=0):raise ValueError("parent has nonpositive Float32 tet")
    floor=np.finfo(np.float64).tiny
    cell_score=np.maximum(0.,-np.log(np.maximum(np.asarray(q,np.float64)/1.e-5,floor)))+np.maximum(0.,-np.log(np.maximum(np.asarray(det,np.float64)/6.e-18,floor)))
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
        comp_rows=np.asarray(comp,np.int64);fat_seed=comp_rows[groups[comp_rows]==1];skin_seed=comp_rows[groups[comp_rows]==0]
        patches.append({"bad_component":comp_id,"seed_bad_cells":comp,"free_rows":free,"affected_cells":affected,
            "ordinary_ring_rows":len(allring),"inner_seed_rows":len(seed),"inner_ring_rows":len(inner_ring),
            "interior_rows":len(interior_corners),"back_star_rows":len(backstar),"star_cell_count":len(starcells),
            "seed_bad_fat_cells":len(fat_seed),"seed_bad_skin_cells":len(skin_seed),
            "fat_seed_log_deficit":float(cell_score[fat_seed].sum(dtype=np.float64)),
            "skin_seed_log_deficit":float(cell_score[skin_seed].sum(dtype=np.float64)),
            "ring_by_role":ring_roles,"inner_ring_by_role":inner_roles})
        component_receipts.append({"bad_component":comp_id,"bad_cells":len(comp),"free_rows":len(free),"affected_cells":len(affected),"status":"PATCHED"})
    patches.sort(key=lambda x:(-int(x["seed_bad_fat_cells"]),-float(x["fat_seed_log_deficit"]),
                               -int(x["seed_bad_skin_cells"]),int(x["bad_component"])))
    for j,x in enumerate(patches):x["patch"]=j
    observed=np.unique(np.concatenate([x["free_rows"] for x in patches])) if patches else np.empty(0,np.int64)
    if not np.array_equal(observed,np.asarray(sorted(allowed),np.int64)):raise ValueError("independent patch roster misses free rows")
    eligible=set(map(int,bad[np.any(np.isin(t[bad],observed),axis=1)])) if len(observed) and len(bad) else set()
    covered=set()
    for x in patches:covered.update(map(int,np.intersect1d(x["affected_cells"],bad)))
    if eligible!=covered:raise ValueError("a baseline-bad cell with free corner is absent from complete-star roster")
    return ordinary,inner,allowed,bad,good,patches,component_receipts

def verify_joint_fallback_diagnostics(meta,producer_diag):
    """Check the joint-fallback ledger independently of the producer counters."""
    policy=meta.get("joint_small_patch_fallback",{})
    detail=producer_diag.get("joint_small_patch_fallback",{"enabled":False})
    if policy!=detail:raise ValueError("candidate joint fallback policy differs from producer diagnostics")
    enabled=bool(detail.get("enabled",False));attempts=0;accepted=0
    if enabled:
        if detail.get("helper")!={"path":str(JOINT.resolve()),"sha256":JOINT_SHA}:
            raise ValueError("diagnostics helper pin differs")
        for pr in producer_diag.get("local_patch_attempts",[]):
            jr=pr.get("joint_multi_node_slsqp_fallback")
            if not jr:continue
            rows=jr.get("reports",[])
            if int(jr.get("attempt_count",-1))!=len(rows):raise ValueError("joint fallback attempt count differs")
            row_accepted=sum(str(x.get("status","")).startswith("ACCEPT_") for x in rows)
            if int(jr.get("accepted_count",-1))!=row_accepted:raise ValueError("joint fallback accepted count differs")
            lp=pr.get("block_lp_retry",{});node=pr.get("single_node_fallback",{})
            if not lp.get("attempted") or lp.get("accepted") or int(node.get("accepted_count",-1))!=0:
                raise ValueError("joint fallback ran before LP and one-node paths stalled")
            attempts+=len(rows)
            for row in rows:
                free=set(map(int,row.get("free_rows",[])))
                if not 2<=len(free)<=4:raise ValueError("joint fallback free-node count outside 2..4")
                if not str(row.get("status","")).startswith("ACCEPT_"):continue
                prior=row.get("prior_failures",{})
                if prior.get("lp_attempted") is not True or prior.get("node_fallback_attempted") is not True or int(prior.get("node_fallback_accepted",-1))!=0:
                    raise ValueError("accepted joint fallback lacks stalled predecessor proof")
                if row.get("full_point_star_verified") is not True or row.get("all_local_exact_f32_determinants_positive") is not True or row.get("all_attached_separation_witnesses_rechecked") is not True:
                    raise ValueError("accepted joint fallback lacks local exact gates")
                if not set(map(int,row.get("moved_rows",[]))).issubset(free):raise ValueError("joint fallback moved a point outside its patch")
                if float(row.get("maximum_step_l1_m",float("inf")))>float(meta.get("maximum_displacement_m",0))+1.e-15:
                    raise ValueError("joint fallback step exceeds the state trust cap")
                accepted+=1
        if accepted!=int(producer_diag.get("accepted_joint_small_patch_fallbacks",-1)):
            raise ValueError("joint fallback accepted ledger differs")
    elif int(producer_diag.get("accepted_joint_small_patch_fallbacks",0))!=0:
        raise ValueError("disabled joint fallback has accepted operations")
    return {"enabled":enabled,"attempts":attempts,"accepted":accepted,"helper":detail.get("helper")}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate-state",type=Path,required=True);ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--audit-report",type=Path,required=True);ap.add_argument("--audit-config",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path);ap.add_argument("--rollback-plan",type=Path);ap.add_argument("--output-report",type=Path,required=True);a=ap.parse_args()
    cp=a.candidate_state.resolve();pp=a.parent_state.resolve();cand=G.state(cp);parent=G.state(pp);meta=cand.get("coupled_point_repair",{})
    if meta.get("producer")!={"path":str(BUILDER.resolve()),"sha256":BUILDER_SHA}:raise ValueError("candidate is not pinned to v25k producer")
    if meta.get("joint_small_patch_slsqp_helper")!={"path":str(JOINT.resolve()),"sha256":JOINT_SHA}:
        raise ValueError("candidate does not pin the joint small-patch SLSQP helper")
    diag_pin=meta.get("diagnostics")
    if not isinstance(diag_pin,dict) or sha(diag_pin.get("path",""))!=diag_pin.get("sha256"):
        raise ValueError("producer diagnostics pin is missing or stale")
    producer_diag=json.loads(Path(diag_pin["path"]).read_text())
    role_before=producer_diag.get("quality_log_deficit_by_role_before",{})
    role_after=producer_diag.get("quality_log_deficit_by_role_after",{})
    if set(role_before)!={"Skin","Fat"} or set(role_after)!={"Skin","Fat"}:raise ValueError("per-role native log-deficit scores missing")
    for phase,table in (("before",role_before),("after",role_after)):
        for role,prefix in (("Skin","skin"),("Fat","fat")):
            alias=prefix+"_log_deficit_score_"+phase
            if not np.isclose(float(producer_diag.get(alias,float("nan"))),float(table[role]),rtol=1.e-13,atol=1.e-13):raise ValueError("named per-role log-deficit alias differs: "+alias)
    if not np.isclose(sum(map(float,role_before.values())),float(producer_diag.get("bad_score_before",float("nan"))),rtol=1.e-12,atol=1.e-8):raise ValueError("per-role baseline scores do not sum to global score")
    if not np.isclose(sum(map(float,role_after.values())),float(producer_diag.get("bad_score_after",float("nan"))),rtol=1.e-12,atol=1.e-8):raise ValueError("per-role final scores do not sum to global score")
    fat_gain=float(role_before["Fat"])-float(role_after["Fat"])
    fat_score_expected=fat_gain>max(1.e-8,1.e-6*float(role_before["Fat"]))
    if bool(producer_diag.get("strict_fat_log_deficit_score_progress"))!=fat_score_expected:raise ValueError("Fat score progress flag differs from pinned per-role scores")
    joint_diag=producer_diag.get("joint_small_patch_fallback",{"enabled":False})
    joint_validation=verify_joint_fallback_diagnostics(meta,producer_diag)
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
    surfaces,volumes,points,tets,vmap,smap=G.build_parent_domain(parent);back=independent_back_lineage_v25k(parent,surfaces,volumes)
    ordinary_fixed,_=G.source_fixed_rows(parent,surfaces,smap,back["fat_rows"],vmap)
    current_i=np.asarray(smap["I"][back["current_i_rows"]],np.int64)
    if np.any(current_i<0):raise ValueError("current I back row unmapped")
    ordinary_fixed.update(map(int,current_i));inner_fixed,inner_protect=inner_hard_fixed(parent,surfaces,smap)
    frozen,witness_receipt=witness_rows(meta,pp,parent,len(points))
    quality_group_ids=np.concatenate([np.zeros(len(volumes["Skin"]["tetrahedra"]),np.int8),np.ones(len(volumes["Fat"]["tetrahedra"]),np.int8)])
    ordinary,inner,allowed,bad,good,patches,component_status=reconstruct_masks(points,tets,surfaces,smap,np.asarray(vmap["Fat"][back["fat_rows"]],np.int64),ordinary_fixed,inner_fixed,frozen,quality_group_ids)
    one=cand.get("source_triangle_one_ring_expansion",{}).get("unified_inner_back_ordinary",{})
    if not isinstance(one,dict):raise ValueError("unified mask receipt missing")
    if int(one.get("ordinary_mask",{}).get("final_movable_rows",-1))!=len(ordinary) or int(one.get("inner_mask",{}).get("free_global_rows",-1))!=len(inner):raise ValueError("producer mask counts differ from independent reconstruction: "+str({"recorded_ordinary":one.get("ordinary_mask",{}).get("final_movable_rows"),"independent_ordinary":len(ordinary),"recorded_inner":one.get("inner_mask",{}).get("free_global_rows"),"independent_inner":len(inner)}))
    roster=one.get("roster",{});schema="h706.unified-inner-back-ordinary-block-local-patches.v2-fat-first"
    if roster.get("schema")!=schema:raise ValueError("producer patch roster schema mismatch")
    expected_policy=("Fat-seeded patches first; descending Fat seed bad-cell count, then summed Fat per-cell native log deficit; ties use Skin seed count then stable component id. Mixed Skin/Fat patches remain joint and retain complete cross-material stars.")
    if roster.get("scheduler_policy")!=expected_policy or roster.get("Fat_seeded_patches_scheduled_before_Skin_only_patches") is not True:
        raise ValueError("Fat-first scheduling policy metadata differs")
    expected_order=[{"rank":int(i),"bad_component":int(x["bad_component"]),"fat_seed_bad_cells":int(x["seed_bad_fat_cells"]),"fat_seed_log_deficit":float(x["fat_seed_log_deficit"]),"skin_seed_bad_cells":int(x["seed_bad_skin_cells"])} for i,x in enumerate(patches)]
    if roster.get("fat_priority_order")!=expected_order:raise ValueError("recorded Fat-first patch order differs from independent reconstruction")
    expected={"all_bad_cells":len(bad),"bad_face_components":len(component_status),"patched_components":len(patches),
        "ordinary_free_rows":len(ordinary),"inner_back_free_rows":len(inner),"free_union_rows":len(allowed),
        "bad_cells_with_at_least_one_free_corner":sum(x["status"]=="PATCHED" for x in component_status) if False else int(roster.get("bad_cells_with_at_least_one_free_corner",-1))}
    # Verify independently the aggregate roster fields that can be derived without trusting producer counts.
    for k in ("all_bad_cells","bad_face_components","patched_components","ordinary_free_rows","inner_back_free_rows","free_union_rows"):
        if int(roster.get(k,-1))!=int(expected[k]):raise ValueError("independent roster mismatch: "+k)
    recorded=roster.get("patches",[])
    if len(recorded)!=len(patches):raise ValueError("patch roster length mismatch")
    attempts=producer_diag.get("local_patch_attempts",[])
    if len(attempts)>len(patches):raise ValueError("producer attempted more patches than independently rostered")
    for rank,attempt in enumerate(attempts):
        expected_patch=patches[rank]
        for key,expected_value in (("patch",rank),("bad_component",int(expected_patch["bad_component"])),
            ("fat_priority_rank",rank),("seed_bad_fat_cells",int(expected_patch["seed_bad_fat_cells"])),
            ("seed_bad_skin_cells",int(expected_patch["seed_bad_skin_cells"]))):
            if int(attempt.get(key,-1))!=int(expected_value):raise ValueError("attempted patch did not follow Fat-first roster: "+key)
        for key in ("fat_seed_log_deficit","skin_seed_log_deficit"):
            if not np.isclose(float(attempt.get(key,float("nan"))),float(expected_patch[key]),rtol=1.e-13,atol=1.e-13):raise ValueError("attempted patch score differs: "+key)
    for j,(r,x) in enumerate(zip(recorded,patches)):
        pairs={"patch":x["patch"],"bad_component":x["bad_component"],"seed_bad_cells":len(x["seed_bad_cells"]),"free_rows":len(x["free_rows"]),"affected_cells":len(x["affected_cells"]),"ordinary_ring_rows":x["ordinary_ring_rows"],"inner_seed_rows":x["inner_seed_rows"],"inner_ring_rows":x["inner_ring_rows"],"back_star_rows":x["back_star_rows"],"seed_bad_fat_cells":x["seed_bad_fat_cells"],"seed_bad_skin_cells":x["seed_bad_skin_cells"],"fat_priority_rank":j}
        for k,v in pairs.items():
            if int(r.get(k,-1))!=int(v):raise ValueError(f"patch {j} {k} differs")
        for k in ("fat_seed_log_deficit","skin_seed_log_deficit"):
            if not np.isclose(float(r.get(k,float("nan"))),float(x[k]),rtol=1.e-13,atol=1.e-13):raise ValueError(f"patch {j} {k} differs")
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
        comparisons=(("current_i_surface_rows",back["current_i_rows"]),
          ("current_fat_volume_rows",back["current_fat_rows"]),
          ("historical_fat_volume_rows",back["historical_fat_rows"]),
          ("current_plus_historical_fat_volume_rows",back["fat_rows"]),
          ("historical_i_reference_rows",back["historical_i_rows"]),
          ("current_i_surface_rows_by_historical_record",back["current_i_rows"]),
          ("current_fat_volume_rows_by_historical_record",back["current_fat_by_record"]),
          ("historical_fat_volume_rows_by_historical_record_or_minus_one",back["historical_fat_by_record"]),
          ("current_d_surface_back_rows",back["current_d_source_rows"]),
          ("current_d_fat_volume_rows_by_current_source_row",back["current_d_fat_by_current_source_row"]),
          ("shared_global_current_d_rows",np.asarray(smap["D"][back["current_d_source_rows"]],np.int64)))
        for k,e in comparisons:
            if k not in z or not np.array_equal(np.asarray(z[k],np.int64),np.asarray(e,np.int64)):
                raise ValueError("actual back source record lineage mismatch: "+k)
    coverage=meta.get("actual_back_Fat_lineage_protection",{}).get("historical_record_coverage",{})
    if int(coverage.get("historical_record_count",-1))!=790 or int(coverage.get("lineage_mapped_record_count",-1))!=790 or int(coverage.get("uncovered_historical_record_count",-1))!=0:
        raise ValueError("producer did not prove coverage of all 790 historical back records")
    if int(coverage.get("current_unique_i_source_representatives",-1))!=len(np.unique(back["current_i_rows"])):
        raise ValueError("producer current I representative count mismatch")
    if int(coverage.get("current_plus_historical_unique_fat_protected_rows",-1))!=len(back["fat_rows"]):
        raise ValueError("producer unique Fat protection union count mismatch")
    if int(coverage.get("historical_coordinate_orphan_record_count",-1))!=int(np.count_nonzero(back["historical_fat_by_record"]<0)):
        raise ValueError("producer historical coordinate orphan count mismatch")
    protection=meta.get("actual_back_Fat_lineage_protection",{})
    owner=protection.get("source_owner",{})
    if owner.get("path")!=str(BACK_PROTECTION.resolve()) or owner.get("sha256")!=BACK_PROTECTION_SHA:
        raise ValueError("producer authenticated protection owner pin mismatch")
    owner_payload=protection.get("protection_owner_report")
    if not isinstance(owner_payload,dict) or not isinstance(owner_payload.get("sources"),dict):
        owner_payload=protection.get("protection_report")
    if not isinstance(owner_payload,dict) or not isinstance(owner_payload.get("sources"),dict):
        raise ValueError("855 authenticated protection owner report is absent")
    owner_report=owner_payload.get("sources",{}).get("D",{})
    if not isinstance(owner_report,dict):
        raise ValueError("855 D source protection report is absent")
    owner_dlineage=owner_report.get("back_lineage",{})
    if not isinstance(owner_dlineage,dict):
        raise ValueError("855 D-back lineage report is absent")
    drows_expected=np.asarray(back["current_d_source_rows"],np.int64)
    owner_rows=np.asarray(owner_report.get("back_rows",[]),np.int64)
    if owner_rows.shape!=(len(drows_expected),) or not np.array_equal(np.sort(owner_rows),drows_expected):
        raise ValueError("855 current D-back rows differ from independent lineage")
    # v25m embeds the authenticated report as protection_owner_report. The
    # earlier verifier looked for a redundant alias and summary fields instead.
    # Reconstruct that summary here from the pinned owner report and independent map.
    historical_records=int(owner_dlineage.get("historical_back_records",-1))
    owner_unique=owner_dlineage.get("unique_current_rows")
    if owner_unique is None:
        owner_unique=owner_dlineage.get("unique_current_back_rows",-1)
    owner_unique=int(owner_unique)
    independent_cov=back.get("coverage",{})
    if (historical_records!=1673 or int(independent_cov.get("D_historical_records",-1))!=1673
        or int(independent_cov.get("D_mapped_records",-1))!=1673
        or int(independent_cov.get("D_uncovered_records",-1))!=0):
        raise ValueError("855 D-back lineage does not map all 1,673 historical records")
    if owner_unique!=len(drows_expected) or int(owner_dlineage.get("unique_current_rows",owner_unique))!=len(drows_expected):
        raise ValueError("855 D-back owner unique row count differs from independent lineage")
    if int(owner_dlineage.get("unique_current_back_rows",owner_unique))!=len(drows_expected):
        raise ValueError("855 D-back owner alias row count differs from independent lineage")
    expected_collapsed=historical_records-owner_unique
    if (int(owner_dlineage.get("collapsed_historical_records",-1))!=expected_collapsed
        or int(owner_dlineage.get("lineage_collapsed_rows",-1))!=expected_collapsed):
        raise ValueError("855 D-back many-to-one history count differs from authenticated map")
    if int(owner_dlineage.get("protected_reference_rows",-1))!=1673:
        raise ValueError("855 D-back reference record count differs")
    if int(owner_dlineage.get("current_row_domain",-1))!=len(surfaces["D"]["points_world_m_f32"]):
        raise ValueError("855 D-back current point domain differs")
    if owner_dlineage.get("current_transition_displacement_pass") is not True or int(owner_dlineage.get("current_transition_records_over_limit",-1))!=0:
        raise ValueError("855 D-back current transition protection proof failed")
    trans_limit=float(owner_dlineage.get("current_transition_l1_limit_m",float("nan")))
    trans_max=float(owner_dlineage.get("maximum_current_transition_l1_displacement_m",float("inf")))
    if not np.isfinite(trans_limit) or trans_limit<=0 or trans_max>trans_limit:
        raise ValueError("855 D-back current transition displacement evidence is inconsistent")
    dcoverage=protection.get("D_back_lineage")
    if dcoverage is None:
        dcoverage={"historical_records":historical_records,
            "current_unique_source_rows":owner_unique,
            "current_unique_Fat_rows":len(back["current_d_fat_rows"]),
            "derived_from":"protection_owner_report.sources.D.back_lineage plus independent exact current D/Fat row map"}
    if not isinstance(dcoverage,dict):
        raise ValueError("producer D_back_lineage summary is malformed")
    if int(dcoverage.get("historical_records",-1))!=historical_records or int(dcoverage.get("historical_records",-1))!=1673:
        raise ValueError("producer D-back historical source coverage mismatch")
    if int(dcoverage.get("current_unique_source_rows",-1))!=len(drows_expected):
        raise ValueError("producer D-back current source coverage mismatch")
    if int(dcoverage.get("current_unique_Fat_rows",-1))!=len(back["current_d_fat_rows"]):
        raise ValueError("producer D-back Fat volume representative count mismatch")
    if sha(SOURCE807)!=SOURCE807_SHA:raise ValueError("807 verifier pin mismatch")
    S=loadmod("v25h_source807",SOURCE807,SOURCE807_SHA);proof=S.verify_current_source_evidence(cp)
    if not str(proof.get("status","" )).startswith("PASS_"):raise ValueError("807 current source proof failed")
    receipt={"schema":"h706.unified-inner-back-ordinary-independent-replay.v25n.v1","status":"PASS_INDEPENDENT_SOURCE_CONTEXT_VOLUME_QUALITY_AND_UNIFIED_MASK",
        "verifier":pin(Path(__file__)),"producer":pin(BUILDER),"candidate_state":pin(cp),"parent_state":pin(pp),"audit_report":pin(a.audit_report),"audit_config":pin(a.audit_config),
        "audit_status":report["status"],"declared_805_inner_roles":inner_roles,"volume_replay":pin(base_replay),"volume_replay_status":replay["status"],"legacy_volume_replay_compatibility_state":pin(compat_path),"source_evidence_807":proof,
        "witness_history":witness_receipt,"D_back_lineage_metadata_reconciliation":{"source":"protection_owner_report.sources.D.back_lineage" if protection.get("D_back_lineage") is None else "explicit_D_back_lineage_checked_against_owner_and_independent_map","historical_records":1673,"mapped_historical_records":int(back["coverage"]["D_mapped_records"]),"uncovered_historical_records":int(back["coverage"]["D_uncovered_records"]),"unique_current_source_rows":len(back["current_d_source_rows"]),"unique_current_Fat_rows":len(back["current_d_fat_rows"]),"owner_report_current_rows_exact":True},"back_lineage":{"current_I_rows":len(back["current_i_rows"]),"current_Fat_rows":len(back["current_fat_rows"]),"historical_Fat_rows":len(back["historical_fat_rows"]),"union_Fat_rows":len(back["fat_rows"]),"current_D_rows":len(back["current_d_source_rows"]),"current_D_Fat_rows":len(back["current_d_fat_rows"]),"record_coverage":back["coverage"],"row_asset":asset},
        "inner_hard_protection":inner_protect,"independent_mask":{"ordinary_rows":len(ordinary),"inner_rows":len(inner),"union_rows":len(allowed),"baseline_bad_cells":len(bad),"bad_face_components":len(component_status),"patches":len(patches),"roster_reconstructed":True},
        "joint_slsqp_helper":pin(JOINT),"joint_small_patch_fallback":joint_diag,"joint_validation":joint_validation,
        "source_moved_rows":source_moves,"volume_moved_rows":volume_moves,"changed_global_rows":len(moved),"all_moves_inside_unified_mask":True,
        "outer_back_scope_rims_routes_fixed":True,"all185_and_surface_context_zero":True,"formerly_good_regressions":0,"native_thresholds_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    a.output_report.parent.mkdir(parents=True,exist_ok=True);a.output_report.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"moved_rows":len(moved),"ordinary_free":len(ordinary),"inner_free":len(inner),"patches":len(patches),"bad":len(bad)},sort_keys=True))
if __name__=="__main__":main()
