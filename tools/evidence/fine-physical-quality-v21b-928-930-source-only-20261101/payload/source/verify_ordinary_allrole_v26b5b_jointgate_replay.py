# v26b4 independent verifier: v26b3r3 cap-order fix plus replay of every accepted F32 backtracking record.
# b2 is the immutable v26b3r2 verifier with the extended-policy-aware 808 source verifier.
# Derived from frozen v26a verifier plus v25o count-stall ledger checks.
#!/usr/bin/env python3
"""Independent exact-context, volume, source-mask, and F32 backtracking verifier."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BUILDER=HERE/"repair_coupled_quality_points_722_v26b5_cowdiff_f32backtrack_250um.py"
BUILDER_SHA="53071ffde5f51eb65a2090d1e60931f552f0647fa8d5c4940af941d525fab946"
JOINT=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v26b4-f32backtrack-250um/joint_small_patch_slsqp_894_v26b4_f32backtrack.py"
JOINT_SHA="c6c4ae40c364485cb3fb65ee703d66af388f22a1a1842328ab97a6c323a416e2"
F32_BACKTRACK=ROOT/"quality-f32-backtracking-923-root/quality_f32_backtracking_923.py"
F32_BACKTRACK_SHA="8e7959e82ffc5adddf2dd24dcdcbb1b4550dbce81d62ca628f4f389688860a4c"
COW_NPZ=ROOT/"npz-cow-append-925-root/npz_cow_append_925.py"
COW_NPZ_SHA="4799f33d73a37072498565b6ff9c20d181916f0158e253fe5c7e9056ba2c0270"
COW_COMPLETE=ROOT/"npz-cow-append-926b-root/npz_cow_append_complete_926b.py"
COW_COMPLETE_SHA="98dd06cafb9a38af4a8c1c7a803ed967d6effff73fd7d1528df7bedcfb40418f"
BASE_REPLAY=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25g-blocklocal/verify_ordinary_allrole_v25g_independent_replay.py"
BASE_REPLAY_SHA="b0d61dbadfce6e164a21b4c8c7a6d5dbfff6fb01b78f2aa82bde2e56bdfdc52a"
SOURCE808=ROOT/"verify_current_source_evidence_808_v26a.py"
SOURCE808_SHA="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9"
SOURCE806=ROOT/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_806_extended_250um_construction.py"
SOURCE806_SHA="1c20de7dffa2cbf55abf195310af988aed9c5a87f035a8e79e099a3f00c74377"
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
if sha(BUILDER)!=BUILDER_SHA:raise ValueError("v26b5 producer source changed")
if sha(JOINT)!=JOINT_SHA:raise ValueError("v26b4 joint SLSQP helper source changed")
M=loadmod("v26b5_builder_for_mask_semantics",BUILDER,BUILDER_SHA)
R=M.BACK
BACK_PROTECTION=ROOT/"quality-partial-cavity-684-root/paired_connected_boundary_patch_856_extended_250um_construction.py"
BACK_PROTECTION_SHA="25a2bf8cbb2ff776745cb3a68bb3ffdfe4b294d8a555f247ce1bc53b476c4897"
if sha(BACK_PROTECTION)!=BACK_PROTECTION_SHA:raise ValueError("856 authenticated back-protection owner pin mismatch")

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

def _same_npz_array_independent(a,b):
    aa=np.asarray(a);bb=np.asarray(b)
    return aa.dtype==bb.dtype and aa.shape==bb.shape and aa.flags.f_contiguous==bb.flags.f_contiguous and aa.tobytes(order="A")==bb.tobytes(order="A")

def verify_cow_npz_receipts(candidate,parent):
    expected={"path":str(COW_COMPLETE.resolve()),"sha256":COW_COMPLETE_SHA}
    underlying={"path":str(COW_NPZ.resolve()),"sha256":COW_NPZ_SHA}
    if sha(COW_NPZ)!=COW_NPZ_SHA:raise ValueError("925 APFS NPZ writer pin mismatch")
    if sha(COW_COMPLETE)!=COW_COMPLETE_SHA:raise ValueError("926b complete-payload adapter pin mismatch")
    if candidate.get("npz_cow_append_writer")!=expected:raise ValueError("candidate does not pin the complete-payload APFS adapter")
    if candidate.get("npz_cow_underlying_npz_writer")!=underlying:raise ValueError("candidate does not pin the underlying 925 NPZ writer")
    policy="Complete desired NPZ payloads are diffed by the pinned 926b adapter using pinned 925 dtype/shape/order-A byte equality; only changed/new arrays and deletions are emitted through APFS clone-only updates; no full-copy fallback"
    if candidate.get("npz_cow_append_policy")!=policy:raise ValueError("APFS NPZ writer policy metadata differs")
    refs=candidate.get("npz_cow_writer_receipts",{})
    if not isinstance(refs,dict):raise ValueError("APFS NPZ writer receipts are malformed")
    checked=[]
    def check(key,parent_entry,child_entry):
        ref=refs.get(key)
        if not isinstance(ref,dict):raise ValueError("missing APFS NPZ writer receipt for "+key)
        rp=Path(ref.get("path",""));rh=ref.get("sha256")
        if not rp.is_file() or sha(rp)!=rh:raise ValueError("APFS NPZ writer receipt pin mismatch for "+key)
        r=json.loads(rp.read_text())
        expected_fields=(("schema","numi.standard-npz-apfs-clone-append-complete-payload.v1"),("status","PASS_VERIFIED_STANDARD_NPZ"),
            ("source_path",str(Path(parent_entry["path"]).resolve())),("source_sha256",parent_entry["sha256"]),
            ("output_path",str(Path(child_entry["path"]).absolute())),("output_sha256",child_entry["sha256"]))
        for field,want in expected_fields:
            if r.get(field)!=want:raise ValueError("APFS NPZ writer receipt field mismatch for "+key+":"+field)
        if (r.get("complete_payload_contract") is not True
            or r.get("input_payload_array_count")!=r.get("output_member_count")):
            raise ValueError("complete-payload COW contract metadata failed for "+key)
        if (r.get("parent_unchanged") is not True or r.get("apfs_clonefile_used") is not True
            or r.get("full_copy_fallback") is not False or r.get("readback_arrays_bitwise_verified")!=r.get("output_member_count")
            or r.get("zip_crc")!="PASS" or r.get("duplicate_central_names")!=0):
            raise ValueError("APFS NPZ writer integrity/clone policy failed for "+key)
        with np.load(parent_entry["path"],allow_pickle=False) as pa, np.load(child_entry["path"],allow_pickle=False) as ca:
            pkeys=set(pa.files);ckeys=set(ca.files)
            deleted=sorted(pkeys-ckeys);added=sorted(ckeys-pkeys);replaced=[];unchanged=[]
            for member in sorted(pkeys&ckeys):
                if _same_npz_array_independent(pa[member],ca[member]):unchanged.append(member)
                else:replaced.append(member)
        if (sorted(r.get("replaced_keys",[]))!=replaced or sorted(r.get("new_keys",[]))!=added
            or sorted(r.get("deleted_keys",[]))!=deleted):
            raise ValueError("COW receipt did not contain the exact member diff for "+key)
        if (r.get("source_member_count")!=len(pkeys) or r.get("output_member_count")!=len(ckeys)
            or r.get("unchanged_compressed_members_bitwise_verified")!=len(unchanged)):
            raise ValueError("COW receipt member accounting mismatch for "+key)
        summaries=candidate.get("npz_cow_payload_deltas",{})
        summary=summaries.get(key) if isinstance(summaries,dict) else None
        want_changed_or_new=sorted(replaced+added)
        if not isinstance(summary,dict) or summary.get("parent_member_count")!=len(pkeys) or summary.get("output_member_count")!=len(ckeys) or summary.get("unchanged_member_count")!=len(unchanged) or sorted(summary.get("changed_or_new_keys",[]))!=want_changed_or_new or sorted(summary.get("deleted_keys",[]))!=deleted:
            raise ValueError("producer COW delta summary differs from independent archive diff for "+key)
        checked.append({"key":key,"receipt":ref,"output_members":r["output_member_count"],
            "replaced_members":len(replaced),"new_members":len(added),"deleted_members":len(deleted),
            "unchanged_members_verified":len(unchanged),"full_copy_fallback":False})
    for role in ("O","I","D"):
        old=parent["source_surfaces"][role];new=candidate["source_surfaces"][role]
        if old["sha256"]==new["sha256"]:
            if old["path"]!=new["path"]:raise ValueError("unchanged source role path changed without a COW receipt: "+role)
            if "source_"+role in refs:raise ValueError("unchanged source role has an unexpected COW receipt: "+role)
        else:check("source_"+role,old,new)
    for role in ("Skin","Fat"):check("volume_"+role,parent[role+"_current"],candidate[role+"_current"])
    extra=set(refs)-{x["key"] for x in checked}
    if extra:raise ValueError("unused or unverified APFS COW receipts: "+repr(sorted(extra)))
    summaries=candidate.get("npz_cow_payload_deltas",{})
    if not isinstance(summaries,dict) or set(summaries)!={x["key"] for x in checked}:
        raise ValueError("COW delta summaries do not exactly cover changed outputs")
    return {"helper":expected,"verified_archive_count":len(checked),"receipts":checked,
        "parent_archives_unchanged":True,"apfs_clone_only":True,"standard_npz_readback_crc_and_arrays_verified":True}

def verify_f32_backtracking_diagnostics(meta,producer_diag,*,return_verified_record_ids=False):
    expected={"path":str(F32_BACKTRACK.resolve()),"sha256":F32_BACKTRACK_SHA}
    if sha(F32_BACKTRACK)!=F32_BACKTRACK_SHA:raise ValueError("923 F32 backtracking helper pin mismatch")
    if meta.get("f32_candidate_backtracking_helper")!=expected:
        raise ValueError("candidate does not pin the F32 backtracking helper")
    policy=meta.get("f32_candidate_backtracking_policy")
    required="raw float64 single-node and 2-4-node SLSQP candidates are checked at alpha 1,1/2,...,1/64 on complete incident stars; existing F32 native, exact sign, separation, cap, and role count/score gates remain unchanged"
    if policy!=required:raise ValueError("F32 backtracking policy metadata differs")
    Q=M.Q
    def replay(rec):
        if not isinstance(rec,dict) or rec.get("schema")!="h706.quality-f32-full-star-backtracking.v1":
            raise ValueError("F32 backtracking receipt schema mismatch")
        if rec.get("status")=="NO_ACCEPTED_BACKTRACKED_PROPOSAL":
            attempts=rec.get("attempts",[])
            if not attempts or len(attempts)>7 or any(x.get("status")=="ACCEPT" for x in attempts):
                raise ValueError("rejected F32 backtracking ledger is malformed")
            for k,attempt in enumerate(attempts):
                if not np.isclose(float(attempt.get("alpha",-1)),2.0**(-k),rtol=0,atol=0):
                    raise ValueError("rejected F32 backtracking alpha order differs")
            return False
        snap=rec.get("independent_replay")
        if rec.get("status")!="PASS_BOUNDED_F32_BACKTRACKED_PROPOSAL" or not isinstance(snap,dict) or snap.get("schema")!="h706.v26b4.backtrack-independent-replay.v1":
            raise ValueError("accepted F32 backtracking event lacks independent replay snapshot")
        point_rows=np.asarray(snap["global_point_rows"],np.int64)
        current=np.asarray(snap["current_points_f32_bits"],np.uint32).view(np.float32)
        baseline=np.asarray(snap["baseline_points_f32_bits"],np.uint32).view(np.float32)
        candidate64=np.asarray(snap["candidate64_points_m"],np.float64)
        accepted_bits=np.asarray(snap["accepted_points_f32_bits"],np.uint32)
        t=np.asarray(snap["tetrahedra_local"],np.int64);free=np.asarray(snap["free_local_rows"],np.int64)
        caps=np.asarray(snap["l1_caps_m"],np.float64);cur_recorded=np.asarray(snap["current_passed"],bool)
        base_recorded=np.asarray(snap["baseline_passed"],bool);seps=snap.get("separation_halfspaces",[])
        n=len(point_rows)
        if (current.shape!=(n,3) or baseline.shape!=(n,3) or candidate64.shape!=(n,3) or
            accepted_bits.shape!=(n,3) or t.ndim!=2 or t.shape[1]!=4 or
            np.any(t<0) or np.any(t>=n) or free.ndim!=1 or len(np.unique(free))!=len(free) or
            np.any(free<0) or np.any(free>=n) or caps.shape!=(len(free),) or
            np.any(~np.isfinite(candidate64)) or not np.isfinite(current).all() or not np.isfinite(baseline).all()):
            raise ValueError("F32 backtracking replay snapshot shape/domain invalid")
        d0,q0=Q._M.measurements(current,t);db,qb=Q._M.measurements(baseline,t)
        cur=np.asarray(Q._M.passes(d0,q0),bool);base=np.asarray(Q._M.passes(db,qb),bool)
        if not np.array_equal(cur,cur_recorded) or not np.array_equal(base,base_recorded):
            raise ValueError("F32 backtracking current/immutable-good masks fail independent measurement")
        allowed=np.zeros(n,bool);allowed[free]=True
        if not np.array_equal(candidate64[~allowed].view(np.uint64),current.astype(np.float64)[~allowed].view(np.uint64)):
            raise ValueError("F32 backtracking candidate64 changes a fixed local point")
        ref=snap.get("progress_reference",{});gb=int(ref.get("global_bad_before",-1));gs=float(ref.get("global_score_before",float("nan")))
        local_bad=int(np.count_nonzero(~cur));local_score=float(M._global_log_deficit(q0,d0))
        if (gb<local_bad or not np.isfinite(gs) or
            not np.isclose(float(ref.get("local_bad_before",-1)),local_bad,rtol=0,atol=0) or
            not np.isclose(float(ref.get("local_score_before",float("nan"))),local_score,rtol=1e-12,atol=1e-10)):
            raise ValueError("F32 backtracking global-progress reference is malformed")
        if local_score>gs+max(1.e-8,1.e-10*max(1.,abs(gs))):
            raise ValueError("F32 backtracking local score exceeds global score")
        attempts=rec.get("attempts",[])
        if not attempts or len(attempts)>7:raise ValueError("F32 backtracking alpha ledger has invalid length")
        accepted_alpha=None;accepted_trial=None;accepted_max_step=None
        for k,attempt in enumerate(attempts):
            alpha=2.0**(-k)
            if not np.isclose(float(attempt.get("alpha",-1)),alpha,rtol=0,atol=0):
                raise ValueError("F32 backtracking alpha order differs from 1,1/2,...,1/64")
            x64=current.astype(np.float64).copy();x64[free]=current[free].astype(np.float64)+alpha*(candidate64[free]-current[free].astype(np.float64))
            x32=x64.astype(np.float32);step=np.abs(x32[free].astype(np.float64)-current[free].astype(np.float64)).sum(axis=1,dtype=np.float64)
            status=None;d=q=None;good=None
            if np.any(step>caps+1.e-15):status="REJECT_L1_CAP"
            else:
                d,q=Q._M.measurements(x32,t);d=np.asarray(d,np.float64);q=np.asarray(q,np.float64)
                good=np.asarray(Q._M.passes(d,q),bool)
                if not np.isfinite(d).all() or not np.isfinite(q).all() or np.any(d<=0):status="REJECT_F32_NONPOSITIVE_OR_NONFINITE"
                elif np.any((cur|base)&~good):status="REJECT_F32_GOOD_CELL_REGRESSION"
                elif any(int(x)<=0 for x in Q._exact_positive(x32,t)):status="REJECT_EXACT_NONPOSITIVE"
                else:
                    sep_bad=[]
                    for si,sr in enumerate(seps):
                        nodes=np.asarray(sr["point_rows"],np.int64);vectors=np.asarray(sr["vectors"],np.float64)
                        lhs=float(np.einsum("ij,ij->",vectors,x32.astype(np.float64)[nodes]-np.asarray(sr["origin"],np.float64)[None,:]))
                        if lhs>float(sr["upper_bound"]):sep_bad.append(si)
                    if sep_bad:status="REJECT_SEPARATION"
                    else:
                        local_after_bad=int(np.count_nonzero(~good));local_after_score=float(M._global_log_deficit(q,d))
                        global_bad_after=gb-local_bad+local_after_bad;global_score_after=gs-local_score+local_after_score
                        tolerance=max(1.e-8,1.e-6*gs);count_gain=global_bad_after<gb
                        score_gain=(global_bad_after==gb and gs-global_score_after>tolerance)
                        progress=bool(count_gain or score_gain)
                        pd=attempt.get("progress")
                        if not isinstance(pd,dict):raise ValueError("F32 backtracking attempt lacks progress receipt")
                        if (int(pd.get("global_bad_before",-1))!=gb or int(pd.get("global_bad_candidate",-1))!=global_bad_after
                            or not np.isclose(float(pd.get("global_score_before",float("nan"))),gs,rtol=1e-12,atol=1e-10)
                            or not np.isclose(float(pd.get("global_score_candidate",float("nan"))),global_score_after,rtol=1e-12,atol=1e-10)
                            or bool(pd.get("accepted"))!=progress or bool(pd.get("count_gain"))!=count_gain
                            or bool(pd.get("score_gain"))!=score_gain):
                            raise ValueError("F32 backtracking global count/score progress replay differs")
                        status="ACCEPT" if progress else "REJECT_NO_ROLE_COUNT_OR_SCORE_PROGRESS"
            if attempt.get("status")!=status:
                raise ValueError("F32 backtracking alpha status differs from independent replay: "+str(attempt.get("status"))+" != "+str(status))
            if status=="ACCEPT":
                accepted_alpha=alpha;accepted_trial=x32;accepted_max_step=float(step.max(initial=0.));break
        if accepted_alpha is None:
            if rec.get("status")!="NO_ACCEPTED_BACKTRACKED_PROPOSAL":raise ValueError("F32 backtracking rejected ledger status differs")
            return False
        if not np.isclose(float(rec.get("alpha",-1)),accepted_alpha,rtol=0,atol=0) or not np.array_equal(accepted_trial.view(np.uint32),accepted_bits):
            raise ValueError("accepted F32 backtracking coordinates/alpha differ")
        if not np.isclose(float(rec.get("maximum_l1_step_m",float("nan"))),accepted_max_step,rtol=1.e-12,atol=1.e-15):
            raise ValueError("accepted F32 backtracking step summary differs from independently measured coordinates")
        return True
    seen=[]
    for patch in producer_diag.get("local_patch_attempts",[]):
        node=patch.get("single_node_fallback",{})
        for row in node.get("nodes",[]):
            if "f32_backtracking" in row:seen.append(row["f32_backtracking"])
        joint=patch.get("joint_multi_node_slsqp_fallback",{})
        for row in joint.get("reports",[]):
            if "f32_backtracking" in row:seen.append(row["f32_backtracking"])
    if int(producer_diag.get("f32_backtracking_event_count",-1))!=len(seen):
        raise ValueError("F32 backtracking event count differs")
    accepted=0;verified_record_ids=set()
    for rec in seen:
        ok=bool(replay(rec));accepted+=int(ok)
        if ok:verified_record_ids.add(id(rec))
    if accepted!=int(producer_diag.get("accepted_f32_backtracked_proposals",-1)):
        raise ValueError("F32 backtracking accepted event count differs")
    summary={"helper":expected,"events":len(seen),"accepted_events":accepted,
        "independent_alpha_full_star_replay":True,"unchanged_thresholds_caps_and_progress_policy":True}
    if return_verified_record_ids:return summary,verified_record_ids
    return summary

def verify_joint_fallback_diagnostics(meta,producer_diag,verified_f32_record_ids=None,full_tetrahedra=None):
    """Independently verify the count-stall trigger and unchanged joint acceptance gates."""
    if verified_f32_record_ids is None:
        _,verified_f32_record_ids=verify_f32_backtracking_diagnostics(meta,producer_diag,return_verified_record_ids=True)
    policy=meta.get("joint_small_patch_fallback",{})
    detail=producer_diag.get("joint_small_patch_fallback",{"enabled":False})
    if policy!=detail:raise ValueError("candidate joint fallback policy differs from producer diagnostics")
    enabled=bool(detail.get("enabled",False));attempts=0;accepted=0;independently_replayed_joint=0
    if enabled:
        if detail.get("helper")!={"path":str(JOINT.resolve()),"sha256":JOINT_SHA}:
            raise ValueError("diagnostics helper pin differs")
        eligibility="after LP and node attempts when the current bad-face patch has no native bad-count reduction; prior score-only LP/node gains are retained and 2-4 movable corners of 1-4 currently bad seed cells are refit from the current coordinates"
        if detail.get("eligibility")!=eligibility:raise ValueError("joint fallback eligibility policy differs")
        if int(detail.get("attempts_per_patch",-1))!=2 or not np.isclose(float(detail.get("reserve_seconds",-1)),.95,rtol=0,atol=1.e-15):
            raise ValueError("joint fallback attempt/reserve budget differs")
        if not np.isclose(float(detail.get("reserve_patch_budget_fraction",-1)),.65,rtol=0,atol=1.e-15):
            raise ValueError("joint fallback patch reserve fraction differs")
        for pr in producer_diag.get("local_patch_attempts",[]):
            jr=pr.get("joint_multi_node_slsqp_fallback")
            if not jr:continue
            rows=jr.get("reports",[])
            if int(jr.get("attempt_count",-1))!=len(rows):raise ValueError("joint fallback attempt count differs")
            row_accepted=sum(str(x.get("status","")).startswith("ACCEPT_") for x in rows)
            if int(jr.get("accepted_count",-1))!=row_accepted:raise ValueError("joint fallback accepted count differs")
            lp=pr.get("block_lp_retry",{});node=pr.get("single_node_fallback",{})
            if not lp.get("attempted") or not node.get("attempt_count"):
                raise ValueError("joint fallback ran before both LP and one-node attempts")
            patch_before=int(pr.get("native_bad_count_before_patch",-1))
            lp_after=int(pr.get("native_bad_count_after_block_lp",-2))
            node_after=int(pr.get("native_bad_count_after_one_node_fallback",-3))
            if not (patch_before>=0 and patch_before==lp_after==node_after):
                raise ValueError("joint fallback ran after native bad-count progress")
            node_rows=node.get("nodes",[])
            accepted_node_statuses=[str(x.get("status")) for x in node_rows if str(x.get("status","")).startswith("ACCEPT_")]
            if int(node.get("accepted_count",-1))!=len(accepted_node_statuses):
                raise ValueError("one-node accepted ledger differs")
            if any(x!="ACCEPT_TIED_SCORE_REDUCTION" for x in accepted_node_statuses):
                raise ValueError("joint fallback ran after one-node bad-count progress")
            attempts+=len(rows)
            for row in rows:
                free=set(map(int,row.get("free_rows",[])))
                if not 2<=len(free)<=4:raise ValueError("joint fallback free-node count outside 2..4")
                prior=row.get("prior_failures")
                if not isinstance(prior,dict):
                    if str(row.get("status","")).startswith("SKIP_REMAINING_FREE_CORNERS"):
                        continue
                    raise ValueError("joint fallback attempt lacks prior count-stall ledger")
                if prior.get("lp_attempted") is not True or prior.get("node_fallback_attempted") is not True:
                    raise ValueError("joint fallback lacks LP/node attempt proof")
                if prior.get("lp_status")!="NO_NATIVE_BAD_COUNT_REDUCTION":
                    raise ValueError("joint fallback prior status does not identify count stall")
                if prior.get("native_bad_count_reduction_since_patch_start") is not False:
                    raise ValueError("joint fallback prior shows native bad-count reduction")
                counts=(int(prior.get("native_bad_count_before_patch",-1)),
                    int(prior.get("block_lp_bad_count_after",-2)),
                    int(prior.get("node_bad_count_after",-3)),
                    int(prior.get("native_bad_count_before_joint",-4)))
                if counts!=(patch_before,patch_before,patch_before,patch_before):
                    raise ValueError("joint fallback prior counts differ from patch ledger")
                prior_node_statuses=list(prior.get("node_fallback_accepted_statuses",[]))
                if prior_node_statuses!=accepted_node_statuses:
                    raise ValueError("joint fallback accepted score-only node history differs")
                if int(prior.get("node_fallback_accepted",-1))!=len(accepted_node_statuses):
                    raise ValueError("joint fallback accepted node count differs")
                if prior.get("block_lp_count_reduction") is not False or prior.get("node_fallback_count_reduction") is not False:
                    raise ValueError("joint fallback prior does not prove count-stalled predecessors")
                if not str(row.get("status","")).startswith("ACCEPT_"):continue
                f32rec=row.get("f32_backtracking")
                if not isinstance(f32rec,dict) or id(f32rec) not in verified_f32_record_ids:
                    raise ValueError("accepted joint fallback is not linked to an independently replayed F32 event")
                snap=f32rec.get("independent_replay")
                if not isinstance(snap,dict):raise ValueError("accepted joint fallback lacks an independent F32 snapshot")
                global_points=np.asarray(snap.get("global_point_rows",[]),np.int64)
                free_local=np.asarray(snap.get("free_local_rows",[]),np.int64)
                global_cells=np.asarray(snap.get("global_cell_rows",[]),np.int64)
                local_tets=np.asarray(snap.get("tetrahedra_local",[]),np.int64)
                if (global_points.ndim!=1 or len(np.unique(global_points))!=len(global_points)
                    or free_local.ndim!=1 or np.any(free_local<0) or np.any(free_local>=len(global_points))
                    or local_tets.ndim!=2 or local_tets.shape[1]!=4):
                    raise ValueError("joint F32 snapshot point/cell map is malformed")
                mapped_free=set(map(int,global_points[free_local]))
                if mapped_free!=free:raise ValueError("joint outer free rows differ from independent F32 free rows")
                if set(map(int,row.get("affected_cells",[])))!=set(map(int,global_cells)):
                    raise ValueError("joint outer affected cells differ from independent F32 star rows")
                if full_tetrahedra is None:
                    raise ValueError("full global tetrahedra required for independent joint-star completeness")
                global_tets=np.asarray(full_tetrahedra,np.int64)
                if np.any(global_cells<0) or np.any(global_cells>=len(global_tets)):
                    raise ValueError("joint F32 global cell rows are outside current volume domain")
                actual_star=np.flatnonzero(np.any(np.isin(global_tets,np.asarray(sorted(free),np.int64)),axis=1))
                if set(map(int,actual_star))!=set(map(int,global_cells)):
                    raise ValueError("joint snapshot is not the complete global incident star of its free rows")
                global_to_local={int(g):i for i,g in enumerate(global_points)}
                try:
                    reconstructed=np.asarray([[global_to_local[int(g)] for g in tet] for tet in global_tets[global_cells]],np.int64)
                except KeyError as exc:raise ValueError("joint F32 point map omits a point used by its global star") from exc
                if not np.array_equal(reconstructed,local_tets):
                    raise ValueError("joint local tetrahedra do not match current global star connectivity")
                current_bits=np.asarray(snap.get("current_points_f32_bits",[]),np.uint32)
                accepted_bits=np.asarray(snap.get("accepted_points_f32_bits",[]),np.uint32)
                if current_bits.shape!=accepted_bits.shape or current_bits.shape!=(len(global_points),3):
                    raise ValueError("joint F32 accepted point arrays are malformed")
                moved_local=np.flatnonzero(np.any(current_bits!=accepted_bits,axis=1))
                moved_global=set(map(int,global_points[moved_local]))
                if set(map(int,row.get("moved_rows",[])))!=moved_global:
                    raise ValueError("joint moved-row ledger differs from independently replayed F32 coordinates")
                if not moved_global.issubset(free):raise ValueError("joint fallback moved a point outside its free set")
                attempts_f32=f32rec.get("attempts",[])
                accepted_f32=next((x for x in attempts_f32 if x.get("status")=="ACCEPT"),None)
                if not isinstance(accepted_f32,dict) or not isinstance(accepted_f32.get("progress"),dict):
                    raise ValueError("accepted joint fallback lacks an independently replayed accepted progress row")
                progress=accepted_f32["progress"]
                if int(row.get("global_bad_after",-1))!=int(progress.get("global_bad_candidate",-2)) or not np.isclose(float(row.get("global_score_after",float("nan"))),float(progress.get("global_score_candidate",float("nan"))),rtol=1.e-12,atol=1.e-10):
                    raise ValueError("joint outer count/score ledger differs from independent accepted F32 event")
                cap=float(meta.get("maximum_displacement_m",0))
                actual_steps=np.abs(accepted_bits.view(np.float32).astype(np.float64)[free_local]-current_bits.view(np.float32).astype(np.float64)[free_local]).sum(axis=1,dtype=np.float64)
                actual_max=float(actual_steps.max(initial=0.))
                if actual_max>cap+1.e-15 or not np.isclose(float(f32rec.get("maximum_l1_step_m",float("nan"))),actual_max,rtol=1.e-12,atol=1.e-15):
                    raise ValueError("joint F32 step exceeds cap or differs from independently replayed step")
                accepted+=1;independently_replayed_joint+=1
        if accepted!=int(producer_diag.get("accepted_joint_small_patch_fallbacks",-1)):
            raise ValueError("joint fallback accepted ledger differs")
    elif int(producer_diag.get("accepted_joint_small_patch_fallbacks",0))!=0:
        raise ValueError("disabled joint fallback has accepted operations")
    return {"enabled":enabled,"attempts":attempts,"accepted":accepted,"helper":detail.get("helper"),
        "count_stall_policy_verified":True,"accepted_joint_f32_events_independently_replayed":independently_replayed_joint,
        "complete_global_star_connectivity_verified":True}

def verify_806_construction_policy(meta,cfg,report,changed,cap):
    """Independently bind the producer's explicit trial mode to its 806 audit."""
    trial=meta.get("inner_back_layer_extended_construction_trial",False)
    if not isinstance(trial,bool):
        raise ValueError("producer extended construction policy must be boolean")
    declared_cap=meta.get("inner_back_layer_extended_construction_cap_m")
    if trial:
        if not 5.e-5 < cap <= 2.5e-4:
            raise ValueError("explicit extended construction cap is outside (50um,250um]")
        if declared_cap is None or float(declared_cap)!=cap:
            raise ValueError("producer extended construction cap metadata differs")
        if meta.get("inner_back_layer_repair") is not False:
            raise ValueError("extended construction cannot claim the legacy 50um policy")
    else:
        if cap>5.e-5:
            raise ValueError("legacy v26b3r2 construction cap exceeds 50um")
        if declared_cap is not None:
            raise ValueError("non-trial state declares an extended construction cap")
    bounded=cfg.get("bounded_point_repair",{})
    if set(bounded)!=set(changed):
        raise ValueError("bounded role set differs from changed roles")
    legacy_roles=[];extended_roles=[]
    for role in changed:
        entry=bounded[role]
        if float(entry.get("maximum_displacement_m",-1))!=cap:
            raise ValueError(role+" bounded point cap mismatch")
        legacy=entry.get("inner_back_layer_repair",False)
        extended=entry.get("inner_back_layer_extended_construction_trial",False)
        if not isinstance(legacy,bool) or not isinstance(extended,bool):
            raise ValueError(role+" inner-back flags must be boolean")
        if role in ("I","D"):
            if trial:
                if extended is not True or legacy is not False:
                    raise ValueError(role+" lacks the distinct extended construction declaration")
                extended_roles.append(role)
            else:
                if legacy is not True or extended is not False:
                    raise ValueError(role+" lacks the legacy explicit 805/806 inner policy")
                legacy_roles.append(role)
        else:
            if extended or legacy:
                raise ValueError("outer O cannot use an inner-back policy")
            if entry.get("ordinary_outer_o_local_repair") is not True:
                raise ValueError("O movement lacks ordinary-local declaration")
    if trial and not extended_roles:
        raise ValueError("extended construction trial did not change any I/D source role")
    expected_controls={"legacy_50um_unchanged":True,
        "extended_I_D_250um_explicit_only":True,"extended_O_rejected":True,
        "mixed_policy_rejected":True,"ordinary_outer_policy_keeps_existing_cap_rules":True}
    if report.get("construction_policy_controls")!=expected_controls:
        raise ValueError("source report does not carry the exact 806 policy-control receipt")
    legacy_report=report.get("inner_back_layer_repair_policy")
    if not isinstance(legacy_report,dict) or set(legacy_report.get("declared_roles",[]))!=set(legacy_roles):
        raise ValueError("806 legacy inner-role declaration differs")
    if (float(legacy_report.get("maximum_inner_displacement_m",float("nan")))!=5.e-5
        or legacy_report.get("outer_back_scope_points_fixed") is not True
        or legacy_report.get("source_rims_and_all149_routes_fixed") is not True):
        raise ValueError("806 legacy protection statement differs")
    extended_report=report.get("inner_back_layer_extended_construction_trial_policy")
    if not isinstance(extended_report,dict) or set(extended_report.get("declared_roles",[]))!=set(extended_roles):
        raise ValueError("806 extended inner-role declaration differs")
    expected_declared_cap=cap if extended_roles else 0.0
    if float(extended_report.get("maximum_declared_construction_displacement_m",float("nan")))!=expected_declared_cap:
        raise ValueError("806 extended construction cap report differs")
    if (extended_report.get("outer_back_scope_points_fixed") is not True
        or extended_report.get("source_rims_and_all149_routes_fixed") is not True
        or extended_report.get("physical_admission") is not False):
        raise ValueError("806 extended source protection report differs")
    if trial and set(extended_roles)!=set(role for role in changed if role in ("I","D")):
        raise ValueError("extended policy does not cover every changed I/D role")
    return {"extended_trial":trial,"cap_m":cap,
            "legacy_inner_roles":sorted(legacy_roles),
            "extended_inner_roles":sorted(extended_roles),
            "audit_implementation":pin(SOURCE806)}

def v26b3r2_policy_contract_controls():
    class Args:
        unified_inner_back_ordinary_local_repair=False
        unified_inner_back_ordinary_250um_construction_trial=True
        ordinary_outer_o_local_repair=False
        ordinary_all_role_local_repair=False
        allow_external_boundary=False
        inner_back_layer_repair=False
        inner_back_remainder_repair=False
        trust_cap_m=2.5e-4
    args=Args()
    meta,bits=M.serialized_policy_contract(args,["D","I","O"])
    assert meta["outer_O_entire_surface_fixed"] is False
    assert meta["ordinary_outer_O_local_repair"] is True
    assert meta["outer_O_scope_fixed"] is True and meta["outer_O_back_scope_fixed"] is True
    cfg={**bits,"bounded_point_repair":M.bounded_point_repair_entries(args,["D","I","O"])}
    report={"construction_policy_controls":{"legacy_50um_unchanged":True,
        "extended_I_D_250um_explicit_only":True,"extended_O_rejected":True,
        "mixed_policy_rejected":True,"ordinary_outer_policy_keeps_existing_cap_rules":True},
        "inner_back_layer_repair_policy":{"declared_roles":[],"maximum_inner_displacement_m":5.e-5,
          "outer_back_scope_points_fixed":True,"source_rims_and_all149_routes_fixed":True},
        "inner_back_layer_extended_construction_trial_policy":{"declared_roles":["I","D"],
          "maximum_declared_construction_displacement_m":2.5e-4,"outer_back_scope_points_fixed":True,
          "source_rims_and_all149_routes_fixed":True,"physical_admission":False}}
    # Exercise actual producer serialization, JSON roundtrip, and the independent 806 policy verifier.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="v26b3r2-policy-") as td:
        root=Path(td);state_path=root/"candidate-state.json";config_path=root/"context-config.json";report_path=root/"audit-report.json"
        state_path.write_text(json.dumps({"coupled_point_repair":meta},sort_keys=True)+"\n")
        config_path.write_text(json.dumps(cfg,sort_keys=True)+"\n")
        report_path.write_text(json.dumps(report,sort_keys=True)+"\n")
        loaded_meta=json.loads(state_path.read_text())["coupled_point_repair"]
        loaded_cfg=json.loads(config_path.read_text())
        loaded_report=json.loads(report_path.read_text())
        accepted=verify_806_construction_policy(loaded_meta,loaded_cfg,loaded_report,["D","I","O"],2.5e-4)
        assert accepted["extended_trial"] is True and accepted["extended_inner_roles"]==["D","I"]
        # Ordinary O must remain locally declared; no “entire O fixed” claim is allowed.
        assert loaded_cfg["bounded_point_repair"]["O"]["ordinary_outer_o_local_repair"] is True
        assert loaded_cfg["bounded_point_repair"]["O"]["maximum_displacement_m"]<=5.e-4
    def rejects(m,c,r,roles,cap):
        try: verify_806_construction_policy(m,c,r,roles,cap)
        except (ValueError,TypeError,KeyError): return True
        return False
    badcfg=json.loads(json.dumps(cfg));badcfg["bounded_point_repair"]["O"].pop("ordinary_outer_o_local_repair")
    assert rejects(meta,badcfg,report,["D","I","O"],2.5e-4)
    assert not M.validate_explicit_250um_construction_policy(2.5e-4,False)["outer_O_entire_surface_fixed"]
    for cap in (5.e-5,2.50001e-4):
        try:M.validate_explicit_250um_construction_policy(cap,False)
        except ValueError:pass
        else:raise AssertionError("invalid extended cap was accepted")
    try:M.validate_explicit_250um_construction_policy(2.5e-4,True)
    except ValueError:pass
    else:raise AssertionError("external-boundary release accepted")
    return {"producer_policy_metadata_serialized":True,"context_config_serialized":True,
        "independent_806_policy_verifier":"PASS","O_ordinary_local_cap_preserved":True,
        "outer_O_entire_surface_fixed_claim_rejected":True,"extended_I_D_only":True,
        "legacy_50um_unchanged":True,"external_boundary_release_rejected":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate-state",type=Path,required=True);ap.add_argument("--parent-state",type=Path,required=True)
    ap.add_argument("--audit-report",type=Path,required=True);ap.add_argument("--audit-config",type=Path,required=True)
    ap.add_argument("--proposal-state",type=Path);ap.add_argument("--rollback-plan",type=Path);ap.add_argument("--output-report",type=Path,required=True);a=ap.parse_args()
    cp=a.candidate_state.resolve();pp=a.parent_state.resolve();cand=G.state(cp);parent=G.state(pp);meta=cand.get("coupled_point_repair",{})
    cow_validation=verify_cow_npz_receipts(cand,parent)
    if meta.get("producer")!={"path":str(BUILDER.resolve()),"sha256":BUILDER_SHA}:raise ValueError("candidate is not pinned to v26b5 COW-delta producer")
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
    f32_backtracking_validation,verified_f32_record_ids=verify_f32_backtracking_diagnostics(
        meta,producer_diag,return_verified_record_ids=True)
    if meta.get("unified_inner_back_ordinary_local_repair") is not True or meta.get("ordinary_block_local_quality_fit") is not True:raise ValueError("unified/block-local mode absent")
    pol="all baseline-bad face components use ordinary bad-corner/source-ring freedoms; I/D additionally use declared inner-back construction freedoms and authenticated actual-back Fat rows in each bad-cell star; ordinary O freedom is limited to bad-corner/source-ring rows at the ordinary cap (at most 500um); O back scope, outer scope, source rims and all 149 routes remain fixed"
    if meta.get("unified_inner_back_ordinary_policy")!=pol:raise ValueError("unified policy metadata differs")
    if meta.get("fixed_outer_and_deep") is not False or meta.get("outer_O_scope_fixed") is not True or meta.get("outer_O_entire_surface_fixed") is not False:raise ValueError("outer scope policy metadata differs")
    if meta.get("ordinary_outer_O_local_repair") is not True or meta.get("outer_O_back_scope_fixed") is not True:raise ValueError("ordinary O local freedom/back-scope policy metadata differs")
    if float(meta.get("ordinary_outer_O_max_cap_m",-1))!=5.e-4:raise ValueError("ordinary O maximum policy cap differs from the existing 500um cap")
    cap=float(meta.get("maximum_displacement_m",0))
    if float(meta.get("ordinary_outer_O_maximum_displacement_m",-1))!=cap:raise ValueError("ordinary O declared displacement cap differs")
    trial=meta.get("inner_back_layer_extended_construction_trial",False)
    if not isinstance(trial,bool):raise ValueError("extended construction mode metadata must be boolean")
    if trial:
        if not 5.e-5 < cap <= 2.5e-4:raise ValueError("extended construction cap is not in (50um,250um]")
    elif not 0<cap<=5.e-5:
        raise ValueError("legacy construction cap exceeds 50um")
    if meta.get("ordinary_block_patch_policy")!="bad-face-connected patches run sequentially; overlapping free nodes and full stars are allowed only with current/baseline-good revalidation and cumulative per-node trust cap":raise ValueError("component solver policy differs")
    report=json.loads(a.audit_report.read_text())
    if report.get("status")!=ZERO_STATUS or report.get("physical_admission") is not False or int(report.get("native_steps",-1))!=0:raise ValueError("exact 775/806 report did not pass")
    G.V.validate_report_config_pin(a.audit_report.resolve(),a.audit_config.resolve());cfg=json.loads(a.audit_config.read_text());changed=[]
    for r in ("O","I","D"):
        b=cfg.get("baseline_candidates",{}).get(r,{});c=cfg.get("candidate_surfaces",{}).get(r,{})
        if b.get("path")!=parent["source_surfaces"][r]["path"] or b.get("sha256")!=parent["source_surfaces"][r]["sha256"]:raise ValueError(r+" baseline pin differs")
        if c.get("path")!=cand["source_surfaces"][r]["path"] or c.get("sha256")!=cand["source_surfaces"][r]["sha256"]:raise ValueError(r+" candidate pin differs")
        if b.get("sha256")!=c.get("sha256"):changed.append(r)
    if set(cfg.get("changed_roles",[]))!=set(changed) or not changed:raise ValueError("changed role set mismatch")
    if sha(SOURCE806)!=SOURCE806_SHA:raise ValueError("806 source auditor pin mismatch")
    policy_receipt=verify_806_construction_policy(meta,cfg,report,changed,cap)
    legacy_inner_roles=policy_receipt["legacy_inner_roles"]
    extended_inner_roles=policy_receipt["extended_inner_roles"]
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
    surfaces,volumes,points,tets,vmap,smap=G.build_parent_domain(parent)
    joint_validation=verify_joint_fallback_diagnostics(meta,producer_diag,verified_f32_record_ids,tets)
    back=independent_back_lineage_v25k(parent,surfaces,volumes)
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
        raise ValueError("856 authenticated protection owner report is absent")
    owner_report=owner_payload.get("sources",{}).get("D",{})
    if not isinstance(owner_report,dict):
        raise ValueError("856 D source protection report is absent")
    owner_dlineage=owner_report.get("back_lineage",{})
    if not isinstance(owner_dlineage,dict):
        raise ValueError("856 D-back lineage report is absent")
    drows_expected=np.asarray(back["current_d_source_rows"],np.int64)
    owner_rows=np.asarray(owner_report.get("back_rows",[]),np.int64)
    if owner_rows.shape!=(len(drows_expected),) or not np.array_equal(np.sort(owner_rows),drows_expected):
        raise ValueError("856 current D-back rows differ from independent lineage")
    # v26b3r2 embeds the authenticated report as protection_owner_report. The
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
        raise ValueError("856 D-back lineage does not map all 1,673 historical records")
    if owner_unique!=len(drows_expected) or int(owner_dlineage.get("unique_current_rows",owner_unique))!=len(drows_expected):
        raise ValueError("856 D-back owner unique row count differs from independent lineage")
    if int(owner_dlineage.get("unique_current_back_rows",owner_unique))!=len(drows_expected):
        raise ValueError("856 D-back owner alias row count differs from independent lineage")
    expected_collapsed=historical_records-owner_unique
    if (int(owner_dlineage.get("collapsed_historical_records",-1))!=expected_collapsed
        or int(owner_dlineage.get("lineage_collapsed_rows",-1))!=expected_collapsed):
        raise ValueError("856 D-back many-to-one history count differs from authenticated map")
    if int(owner_dlineage.get("protected_reference_rows",-1))!=1673:
        raise ValueError("856 D-back reference record count differs")
    if int(owner_dlineage.get("current_row_domain",-1))!=len(surfaces["D"]["points_world_m_f32"]):
        raise ValueError("856 D-back current point domain differs")
    if owner_dlineage.get("current_transition_displacement_pass") is not True or int(owner_dlineage.get("current_transition_records_over_limit",-1))!=0:
        raise ValueError("856 D-back current transition protection proof failed")
    trans_limit=float(owner_dlineage.get("current_transition_l1_limit_m",float("nan")))
    trans_max=float(owner_dlineage.get("maximum_current_transition_l1_displacement_m",float("inf")))
    if not np.isfinite(trans_limit) or trans_limit<=0 or trans_max>trans_limit:
        raise ValueError("856 D-back current transition displacement evidence is inconsistent")
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
    if sha(SOURCE808)!=SOURCE808_SHA:raise ValueError("808 verifier pin mismatch")
    S=loadmod("v26a_source808",SOURCE808,SOURCE808_SHA);proof=S.verify_current_source_evidence(cp)
    if not str(proof.get("status","" )).startswith("PASS_"):raise ValueError("808 current source proof failed")
    receipt={"schema":"h706.unified-inner-back-ordinary-independent-replay.v26b3r22.v1","status":"PASS_INDEPENDENT_SOURCE_CONTEXT_VOLUME_QUALITY_AND_UNIFIED_MASK",
        "verifier":pin(Path(__file__)),"producer":pin(BUILDER),"source_auditor_806":pin(SOURCE806),"candidate_state":pin(cp),"parent_state":pin(pp),"audit_report":pin(a.audit_report),"audit_config":pin(a.audit_config),
        "audit_status":report["status"],"declared_806_legacy_inner_roles":legacy_inner_roles,"declared_806_extended_inner_roles":extended_inner_roles,"construction_policy":policy_receipt,"volume_replay":pin(base_replay),"volume_replay_status":replay["status"],"legacy_volume_replay_compatibility_state":pin(compat_path),"source_evidence_808":proof,
        "witness_history":witness_receipt,"D_back_lineage_metadata_reconciliation":{"source":"protection_owner_report.sources.D.back_lineage" if protection.get("D_back_lineage") is None else "explicit_D_back_lineage_checked_against_owner_and_independent_map","historical_records":1673,"mapped_historical_records":int(back["coverage"]["D_mapped_records"]),"uncovered_historical_records":int(back["coverage"]["D_uncovered_records"]),"unique_current_source_rows":len(back["current_d_source_rows"]),"unique_current_Fat_rows":len(back["current_d_fat_rows"]),"owner_report_current_rows_exact":True},"back_lineage":{"current_I_rows":len(back["current_i_rows"]),"current_Fat_rows":len(back["current_fat_rows"]),"historical_Fat_rows":len(back["historical_fat_rows"]),"union_Fat_rows":len(back["fat_rows"]),"current_D_rows":len(back["current_d_source_rows"]),"current_D_Fat_rows":len(back["current_d_fat_rows"]),"record_coverage":back["coverage"],"row_asset":asset},
        "inner_hard_protection":inner_protect,"independent_mask":{"ordinary_rows":len(ordinary),"inner_rows":len(inner),"union_rows":len(allowed),"baseline_bad_cells":len(bad),"bad_face_components":len(component_status),"patches":len(patches),"roster_reconstructed":True},
        "joint_slsqp_helper":pin(JOINT),"joint_small_patch_fallback":joint_diag,"joint_validation":joint_validation,
        "f32_backtracking_validation":f32_backtracking_validation,"npz_cow_writer_validation":cow_validation,
        "source_moved_rows":source_moves,"volume_moved_rows":volume_moves,"changed_global_rows":len(moved),"all_moves_inside_unified_mask":True,
        "outer_back_scope_rims_routes_fixed":True,"all185_and_surface_context_zero":True,"formerly_good_regressions":0,"native_thresholds_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
    a.output_report.parent.mkdir(parents=True,exist_ok=True);a.output_report.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":receipt["status"],"moved_rows":len(moved),"ordinary_free":len(ordinary),"inner_free":len(inner),"patches":len(patches),"bad":len(bad)},sort_keys=True))
if __name__=="__main__":main()
