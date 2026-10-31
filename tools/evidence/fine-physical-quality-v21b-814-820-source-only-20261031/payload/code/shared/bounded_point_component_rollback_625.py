#!/usr/bin/env python3
"""Witness-driven rollback for bounded O/I/D point updates.

Pure import-safe helpers only. A 630 report's exact crossing face IDs are
mapped through its deterministic changed-record order (D, O, I). Only moved
point components incident to witnessed query faces are restored from the
authenticated baseline. Safe components remain untouched.

This module does not run a geometry audit at import or provide admission.
Callers must remeasure both affected volume layers and rerun the pinned 630
exact audit on the emitted re-audit config.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import copy
import numpy as np

ROLE_ORDER = ("D","O","I")
ROLE_VOLUMES = {"O":("Skin",),"I":("Skin","Fat"),"D":("Fat",)}
POINT_FIELDS = ("points_world_m_f32","points_world_m")


def point_array(payload: Mapping[str,Any], label: str) -> np.ndarray:
    if "points_world_m_f32" in payload:
        p=np.asarray(payload["points_world_m_f32"])
        if p.dtype!=np.float32: raise ValueError(label+": points_world_m_f32 must be Float32")
    elif "points_world_m" in payload:
        p=np.asarray(payload["points_world_m"],dtype=np.float32)
    else: raise ValueError(label+": missing point coordinates")
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError(label+": malformed points")
    return p


def _bytes_key(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(x,dtype="<f4")).view(np.uint32).tobytes()


def moved_point_rows(baseline_points: np.ndarray, candidate_points: np.ndarray) -> np.ndarray:
    b=np.asarray(baseline_points,dtype=np.float32);c=np.asarray(candidate_points,dtype=np.float32)
    if b.shape!=c.shape: raise ValueError("bounded candidate changed point domain")
    return np.flatnonzero(np.any(b.view(np.uint32)!=c.view(np.uint32),axis=1)).astype(np.int64)


def moved_point_components(faces: np.ndarray, moved_rows: Sequence[int]) -> list[list[int]]:
    """Connected components on the induced moved-vertex surface graph."""
    f=np.asarray(faces,np.int64);moved=sorted(set(map(int,moved_rows)))
    if f.ndim!=2 or f.shape[1]!=3 or (f.size and (f.min()<0 or f.max()>=max(moved+[int(f.max())]+[0])+1)):
        raise ValueError("malformed surface faces")
    parent={x:x for x in moved}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            if ra<rb:parent[rb]=ra
            else:parent[ra]=rb
    mset=set(moved)
    for face in f:
        hit=sorted(mset.intersection(map(int,face)))
        for x in hit[1:]:union(hit[0],x)
    groups={}
    for x in moved:groups.setdefault(find(x),[]).append(x)
    return sorted((sorted(v) for v in groups.values()),key=lambda x:x[0])


def changed_face_id_map(report: Mapping[str,Any], changed_face_id_base: int) -> dict[int,tuple[str,int]]:
    """Map 630's sequential candidate record IDs to (role,candidate face row)."""
    if not isinstance(changed_face_id_base,int) or changed_face_id_base<0: raise ValueError("explicit changed-face ID base required")
    counts=report.get("candidate_changed_record_count_by_role")
    roles=report.get("roles")
    if not isinstance(counts,Mapping) or not isinstance(roles,Mapping): raise ValueError("630 role/count schema missing")
    out={};cursor=changed_face_id_base
    for role in ROLE_ORDER:
        summary=roles.get(role)
        if not isinstance(summary,Mapping): raise ValueError("630 role missing: "+role)
        rows=list(map(int,summary.get("changed_candidate_face_rows",[])))
        if int(counts.get(role,-1))!=len(rows) or len(set(rows))!=len(rows):
            raise ValueError(role+": changed face row/count mismatch")
        for offset,row in enumerate(rows):out[cursor+offset]=(role,row)
        cursor+=len(rows)
    return out


def _pairs(section: Any, label: str) -> list[tuple[int,int]]:
    if not isinstance(section,Mapping): raise ValueError("630 "+label+" section missing")
    raw=section.get("triangle_pairs")
    count=section.get("count")
    if not isinstance(raw,list) or count!=len(raw): raise ValueError("630 "+label+" pair list/count mismatch")
    out=[]
    for pair in raw:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2: raise ValueError("malformed exact witness pair")
        a,b=map(int,pair)
        if a==b: raise ValueError("self-pair has identical face ID")
        out.append((a,b))
    return out


def build_rollback_plan(
    audit_report: Mapping[str,Any],
    baseline_surfaces: Mapping[str,Mapping[str,Any]],
    candidate_surfaces: Mapping[str,Mapping[str,Any]],
    *,
    changed_face_id_base: int,
) -> dict[str,Any]:
    """Map exact 630 witnesses to only the moved-point components they touch."""
    if set(baseline_surfaces)!=set(ROLE_ORDER) or set(candidate_surfaces)!=set(ROLE_ORDER):
        raise ValueError("baseline/candidate must contain exactly D, O, I")
    against=audit_report.get("changed_vs_complete_retained_OID_both_walls_all185")
    self_pairs=audit_report.get("changed_OID_joint_self_pairs")
    pairs=_pairs(against,"complete-context") + _pairs(self_pairs,"joint-self")
    if not pairs: raise ValueError("630 report has no exact crossing witnesses to roll back")
    face_map=changed_face_id_map(audit_report,changed_face_id_base)
    components={};moved={};face_components={}
    for role in ROLE_ORDER:
        b=baseline_surfaces[role];c=candidate_surfaces[role]
        bp=point_array(b,role+" baseline");cp=point_array(c,role+" candidate")
        bf=np.asarray(b["faces"],np.int64);cf=np.asarray(c["faces"],np.int64)
        if bf.shape!=cf.shape or not np.array_equal(bf,cf):
            raise ValueError(role+": rollback helper only accepts fixed-topology point candidates")
        if len(bf) and (bf.min()<0 or bf.max()>=len(bp)):raise ValueError(role+": face outside point domain")
        rows=moved_point_rows(bp,cp);moved[role]=rows
        receipt=audit_report["roles"][role].get("points",{})
        if int(receipt.get("bounded_moved_point_count",-1))!=len(rows):
            raise ValueError(role+": exact moved-point count differs from 630 receipt")
        expected_faces=np.flatnonzero(np.any(np.isin(cf,rows),axis=1)).astype(np.int64)
        role_report=audit_report["roles"][role]
        changed=np.asarray(role_report.get("changed_candidate_face_rows",[]),np.int64)
        removed=np.asarray(role_report.get("removed_baseline_face_rows",[]),np.int64)
        if not np.array_equal(changed,expected_faces) or not np.array_equal(removed,expected_faces):
            raise ValueError(role+": report contains topology changes or incomplete moved-face stars")
        comps=moved_point_components(cf,rows);components[role]=comps
        row_to_component={r:i for i,group in enumerate(comps) for r in group}
        for face_row in changed:
            hit=sorted(set(map(int,cf[int(face_row)]))&set(row_to_component))
            if not hit:raise ValueError(role+": changed face has no moved point")
            roots={row_to_component[x] for x in hit}
            if len(roots)!=1:raise ValueError(role+": moved-point components were not joined across their shared face")
            face_components[(role,int(face_row))]=next(iter(roots))
    offending=set();witnesses=[]
    for source,pair in (("complete-context",p) for p in _pairs(against,"complete-context")):
        witnesses.append((source,pair))
    for source,pair in (("joint-self",p) for p in _pairs(self_pairs,"joint-self")):
        witnesses.append((source,pair))
    for source,(a,b) in witnesses:
        mapped=[fid for fid in (a,b) if fid in face_map]
        if not mapped:raise ValueError(source+": witness has no changed candidate face ID in configured range")
        witness_components=[]
        for fid in mapped:
            role,row=face_map[fid]
            key=(role,row)
            if key not in face_components:raise ValueError(source+": changed query face is not incident to a moved component")
            comp=face_components[key];offending.add((role,comp))
            witness_components.append({"face_id":fid,"role":role,"candidate_face_row":row,"component":comp})
        witnesses[witnesses.index((source,(a,b)))] = {"source":source,"face_ids":[a,b],"mapped_changed_faces":witness_components}
    rollback={role:[] for role in ROLE_ORDER}
    for role,ci in sorted(offending):
        rollback[role].extend(components[role][ci])
    rollback={r:np.asarray(sorted(set(rows)),np.int64) for r,rows in rollback.items()}
    safe={r:np.setdiff1d(moved[r],rollback[r],assume_unique=True) for r in ROLE_ORDER}
    return {
        "status":"PASS_EXACT_WITNESS_COMPONENT_MAPPING",
        "changed_face_id_base":changed_face_id_base,
        "role_order":list(ROLE_ORDER),
        "components_by_role":{r:[list(x) for x in components[r]] for r in ROLE_ORDER},
        "moved_point_rows_by_role":{r:moved[r].tolist() for r in ROLE_ORDER},
        "rollback_component_ids_by_role":{r:sorted(ci for rr,ci in offending if rr==r) for r in ROLE_ORDER},
        "rollback_point_rows_by_role":{r:rollback[r].tolist() for r in ROLE_ORDER},
        "retained_safe_point_rows_by_role":{r:safe[r].tolist() for r in ROLE_ORDER},
        "witnesses":witnesses,
    }


def restore_surface_coordinates(
    baseline_surfaces: Mapping[str,Mapping[str,Any]],
    candidate_surfaces: Mapping[str,Mapping[str,Any]],
    plan: Mapping[str,Any],
) -> dict[str,dict[str,np.ndarray]]:
    """Return candidates with only witnessed component coordinate rows restored."""
    result={}
    rollback=plan["rollback_point_rows_by_role"]
    for role in ROLE_ORDER:
        b=baseline_surfaces[role];c=candidate_surfaces[role]
        bp=point_array(b,role+" baseline");cp=point_array(c,role+" candidate")
        if not np.array_equal(np.asarray(b["faces"]),np.asarray(c["faces"])):
            raise ValueError(role+": topology changed during coordinate rollback")
        rows=np.asarray(rollback[role],np.int64)
        if len(rows) and (rows.min()<0 or rows.max()>=len(bp)):raise ValueError(role+": rollback point row outside domain")
        payload={k:np.array(v,copy=True) for k,v in c.items()}
        found=False
        for field in POINT_FIELDS:
            if field not in b and field not in c:continue
            if field not in b or field not in c:raise ValueError(role+": baseline/candidate coordinate field mismatch")
            ba=np.asarray(b[field]);ca=np.asarray(c[field])
            if ba.shape!=ca.shape or ba.dtype!=ca.dtype:raise ValueError(role+": coordinate payload schema mismatch")
            if field=="points_world_m_f32" and ba.dtype!=np.float32:raise ValueError(role+": F32 field dtype changed")
            if len(rows):payload[field][rows]=ba[rows]
            found=True
        if not found:raise ValueError(role+": no restorable point-coordinate fields")
        final=point_array(payload,role+" rollback candidate")
        if not np.array_equal(final[rows].view(np.uint32),bp[rows].view(np.uint32)):raise ValueError(role+": rollback did not restore baseline F32 bytes")
        safe=np.setdiff1d(np.arange(len(cp),dtype=np.int64),rows,assume_unique=True)
        if not np.array_equal(final[safe].view(np.uint32),cp[safe].view(np.uint32)):raise ValueError(role+": safe point updates changed")
        result[role]=payload
    return result


def restore_volume_coordinate_rows(
    baseline_volume: Mapping[str,Any],
    trial_volume: Mapping[str,Any],
    baseline_surfaces: Mapping[str,Mapping[str,Any]],
    trial_surfaces: Mapping[str,Mapping[str,Any]],
    final_surfaces: Mapping[str,Mapping[str,Any]],
    plan: Mapping[str,Any],
    *,
    volume_role: str,
) -> tuple[dict[str,np.ndarray],dict[str,list[dict[str,int]]]]:
    """Restore corresponding volume nodes for rolled-back source-surface rows.

    O couples to Skin, I couples to Skin and Fat, and D couples to Fat. The
    map is exact F32 coordinate identity. Every moved row must exist in each
    expected layer and match the trial volume point before the update.
    """
    if volume_role not in ("Skin","Fat"):raise ValueError("volume_role must be Skin or Fat")
    bp=point_array(baseline_volume,volume_role+" baseline volume")
    tp=point_array(trial_volume,volume_role+" trial volume")
    if bp.shape!=tp.shape:raise ValueError("volume point domain changed")
    if not np.array_equal(np.asarray(baseline_volume["tetrahedra"]),np.asarray(trial_volume["tetrahedra"])):raise ValueError("volume topology changed")
    if not np.array_equal(np.asarray(baseline_volume["material_indices"]),np.asarray(trial_volume["material_indices"])):raise ValueError("volume materials changed")
    lookup={}
    for row,p in enumerate(bp):
        key=_bytes_key(p)
        if key in lookup:raise ValueError("ambiguous duplicate volume F32 point")
        lookup[key]=row
    rollback=plan["rollback_point_rows_by_role"];moved=plan["moved_point_rows_by_role"]
    out={k:np.array(v,copy=True) for k,v in trial_volume.items()};affected=[];row_receipts=[]
    for role in ROLE_ORDER:
        if volume_role not in ROLE_VOLUMES[role]:continue
        sb=point_array(baseline_surfaces[role],role+" baseline surface")
        st=point_array(trial_surfaces[role],role+" trial surface")
        sf=point_array(final_surfaces[role],role+" final surface")
        rrows=np.asarray(rollback[role],np.int64);mrows=np.asarray(moved[role],np.int64)
        role_lookup=[]
        inactive_rows={int(x["surface_point_row"]) for x in plan.get("component_closure",{}).get("inactive_unpaired_surface_moves",[]) if x["role"]==role and x["volume_role"]==volume_role}
        role_faces=np.asarray(trial_surfaces[role]["faces"],np.int64)
        for sr in mrows:
            key=_bytes_key(sb[int(sr)])
            vr=lookup.get(key)
            if vr is None:
                if int(sr) in inactive_rows and not np.any(role_faces==int(sr)):
                    role_lookup.append({"surface_point_row":int(sr),"volume_point_row":None,"rolled_back":False,"inactive_unpaired":True})
                    continue
                raise ValueError(role+"/"+volume_role+": moved active interface point missing from volume")
            rolled_back=int(sr) in set(map(int,rrows))
            if not np.array_equal(tp[vr].view(np.uint32),st[int(sr)].view(np.uint32)):
                is_inactive=(int(sr) in inactive_rows and not np.any(role_faces==int(sr)))
                if is_inactive:
                    pass
                elif not rolled_back or not np.array_equal(sf[int(sr)].view(np.uint32),bp[vr].view(np.uint32)):
                    raise ValueError(role+"/"+volume_role+": unrolled trial volume/surface coordinate crosswalk mismatch")
            if rolled_back:
                out["points_world_m"][vr]=np.asarray(baseline_volume["points_world_m"])[vr]
                if "points_world_m_f32" in out and "points_world_m_f32" in baseline_volume:
                    out["points_world_m_f32"][vr]=np.asarray(baseline_volume["points_world_m_f32"])[vr]
                affected.append(int(vr))
            expected=sf[int(sr)]
            got=np.asarray(out["points_world_m"],dtype=np.float32)[vr]
            if not np.array_equal(got.view(np.uint32),expected.view(np.uint32)):
                if int(sr) not in inactive_rows or np.any(role_faces==int(sr)):
                    raise ValueError(role+"/"+volume_role+": final volume/surface coordinate crosswalk mismatch")
            role_lookup.append({"surface_point_row":int(sr),"volume_point_row":int(vr),"rolled_back":bool(int(sr) in set(map(int,rrows)))})
        row_receipts.extend({"surface_role":role,**x} for x in role_lookup)
    extra_rows=np.asarray(plan.get("rollback_volume_point_rows_by_role",{}).get(volume_role,()),np.int64)
    if len(extra_rows):
        if extra_rows.min()<0 or extra_rows.max()>=len(bp):raise ValueError(volume_role+": expanded rollback volume row outside domain")
        out["points_world_m"][extra_rows]=np.asarray(baseline_volume["points_world_m"])[extra_rows]
        if "points_world_m_f32" in out and "points_world_m_f32" in baseline_volume:
            out["points_world_m_f32"][extra_rows]=np.asarray(baseline_volume["points_world_m_f32"])[extra_rows]
        affected.extend(map(int,extra_rows))
    if "points_world_m_f32" in out:
        if not np.array_equal(np.asarray(out["points_world_m_f32"],np.float32).view(np.uint32),np.asarray(out["points_world_m"],np.float32).view(np.uint32)):
            raise ValueError(volume_role+": paired volume point views disagree after rollback")
    return out,{"rows":row_receipts,"affected_volume_point_rows":sorted(set(affected))}


def verify_native_nonregression(owner, baseline_volume: Mapping[str,Any], final_volume: Mapping[str,Any]) -> dict[str,Any]:
    """Require positive F32 cells and zero regressions from baseline-passing cells."""
    bp=point_array(baseline_volume,"baseline volume");fp=point_array(final_volume,"final volume")
    bt=np.asarray(baseline_volume["tetrahedra"],np.int64);ft=np.asarray(final_volume["tetrahedra"],np.int64)
    bm=np.asarray(baseline_volume["material_indices"]);fm=np.asarray(final_volume["material_indices"])
    if not np.array_equal(bt,ft) or not np.array_equal(bm,fm):raise ValueError("quality rollback must retain exact volume topology and material labels")
    bd,bq=owner.measurements(bp,bt);fd,fq=owner.measurements(fp,ft)
    bg=np.asarray(owner.passes(bd,bq),bool);fg=np.asarray(owner.passes(fd,fq),bool)
    if np.any(fd<=0):raise ValueError("rollback creates Float32 nonpositive volume cell")
    regressions=np.flatnonzero(bg&~fg)
    if len(regressions):raise ValueError("rollback regresses baseline-native-good cells: "+str(regressions[:12].tolist()))
    return {
        "baseline_native_failures":int(np.count_nonzero(~bg)),
        "final_native_failures":int(np.count_nonzero(~fg)),
        "baseline_native_good_regressions":0,
        "positive_F32_all_cells":True,
        "native_quality_thresholds_unchanged":True,
    }


def require_zero_exact_audit(report: Mapping[str,Any]) -> dict[str,Any]:
    """Fail closed unless the repeated 630 exact full-context and joint-self audit is clean."""
    against=report.get("changed_vs_complete_retained_OID_both_walls_all185",{})
    joint=report.get("changed_OID_joint_self_pairs",{})
    if report.get("status")!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":
        raise ValueError("repeated exact 630 audit did not pass")
    if int(against.get("count",-1))!=0 or int(joint.get("count",-1))!=0:
        raise ValueError("repeated exact 630 audit has exact crossings")
    return {"status":"PASS_ZERO_EXACT_CONTEXT_AND_SELF_CROSSINGS","context_pairs":0,"joint_self_pairs":0}


def run_controls():
    # Pure synthetic face-ID/component mapping and selective restoration controls.
    def surface(offset=0.0):
        p=np.asarray(((0,0,0),(1,0,0),(0,1,0),(1,1,0),(3,0,0),(4,0,0),(3,1,0)),np.float32)
        p[:,0]+=np.float32(offset)
        return {"points_world_m_f32":p.copy(),"points_world_m":p.astype(np.float64),"faces":np.asarray(((0,1,2),(1,3,2),(4,5,6)),np.int64),"typed_marker":np.asarray((7,8),np.int64)}
    base={r:surface(i*10.0) for i,r in enumerate(ROLE_ORDER)}
    cand={r:{k:np.array(v,copy=True) for k,v in d.items()} for r,d in base.items()}
    cand["D"]["points_world_m_f32"][[0,1,4],1]+=np.float32(1e-6)
    cand["D"]["points_world_m"][[0,1,4],1]+=1e-6
    drows=[0,1,2]
    report={"status":"FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185",
      "candidate_changed_record_count_by_role":{"D":3,"O":0,"I":0},
      "roles":{r:{"changed_candidate_face_rows":drows if r=="D" else [],
                  "removed_baseline_face_rows":drows if r=="D" else [],
                  "points":{"bounded_moved_point_count":3 if r=="D" else 0}}
               for r in ROLE_ORDER},
      "changed_vs_complete_retained_OID_both_walls_all185":{"count":1,"triangle_pairs":[[901111,1901089]]},
      "changed_OID_joint_self_pairs":{"count":1,"triangle_pairs":[[1901089,1901090]]}}
    plan=build_rollback_plan(report,base,cand,changed_face_id_base=1901089)
    assert plan["rollback_point_rows_by_role"]["D"]==[0,1]
    assert plan["retained_safe_point_rows_by_role"]["D"]==[4]
    restored=restore_surface_coordinates(base,cand,plan)
    assert np.array_equal(restored["D"]["points_world_m_f32"][[0,1]].view(np.uint32),base["D"]["points_world_m_f32"][[0,1]].view(np.uint32))
    assert np.array_equal(restored["D"]["points_world_m_f32"][4].view(np.uint32),cand["D"]["points_world_m_f32"][4].view(np.uint32))
    assert np.array_equal(restored["D"]["typed_marker"],cand["D"]["typed_marker"])
    class Owner:
        @staticmethod
        def measurements(p,t):
            q=np.asarray(p,np.float64)[np.asarray(t,np.int64)]
            u=q[:,1:]-q[:,:1];d=np.einsum("ij,ij->i",u[:,0],np.cross(u[:,1],u[:,2]))
            e=np.max(np.stack([np.linalg.norm(q[:,i]-q[:,j],axis=1) for i in range(4) for j in range(i+1,4)]),axis=0)
            return d,d/np.maximum(e**3,np.finfo(float).tiny)
        @staticmethod
        def passes(d,q):return np.isfinite(d)&np.isfinite(q)&(d/6>1e-18)&(q>=1e-5)
    tetra_points=np.asarray(((0,0,0),(1,0,0),(0,1,0),(0,0,1)),np.float32)
    vol={"points_world_m":tetra_points,"tetrahedra":np.asarray(((0,1,2,3),),np.int64),"material_indices":np.asarray((16,),np.int32)}
    q=verify_native_nonregression(Owner,vol,vol)
    assert q["baseline_native_good_regressions"]==0 and q["positive_F32_all_cells"]
    try:require_zero_exact_audit({"status":"FAIL","changed_vs_complete_retained_OID_both_walls_all185":{"count":1},"changed_OID_joint_self_pairs":{"count":0}})
    except ValueError:negative=True
    else:raise AssertionError("failed exact audit must not pass")
    return {
      "status":"PASS_625_POINT_COMPONENT_ROLLBACK_CONTROLS",
      "controls":["630_D_O_I_face_id_range_mapping","shared-face moved-point component grouping","rollback only witnessed D component","safe D component retained bitwise","typed metadata retained","full native nonregression owner contract","nonzero repeated exact audit rejected"],
      "changed_face_id_base":1901089,
      "witness_ids":[1901089,1901090],
      "rollback_D_rows":[0,1],"retained_safe_D_rows":[4],
      "physical_admission":False,"native_steps":0,
    }


if __name__=="__main__":
    import json
    print(json.dumps(run_controls(),indent=2))



def expand_rollback_over_free_node_components(
    surface_plan: Mapping[str,Any],
    baseline_surfaces: Mapping[str,Mapping[str,Any]],
    trial_surfaces: Mapping[str,Mapping[str,Any]],
    baseline_volumes: Mapping[str,Mapping[str,Any]],
    trial_volumes: Mapping[str,Mapping[str,Any]],
) -> dict[str,Any]:
    """Expand witnessed surface seeds to complete coupled moved-node/tet components.

    Vertices in this graph are moved O/I/D surface rows and moved volume point
    rows. Surface faces join moved surface vertices; each Skin/Fat tet joins
    every moved volume vertex appearing in that tet; exact F32 coordinate
    crosswalks join the corresponding surface and volume vertices. This
    captures interior LP variables coupled to a witnessed boundary variable.
    """
    if set(baseline_surfaces)!=set(ROLE_ORDER) or set(trial_surfaces)!=set(ROLE_ORDER):
        raise ValueError("joint component closure requires D/O/I surface domains")
    if set(baseline_volumes)!={"Skin","Fat"} or set(trial_volumes)!={"Skin","Fat"}:
        raise ValueError("joint component closure requires Skin and Fat volumes")

    parent={}
    def add(x):
        if x not in parent:parent[x]=x
    def find(x):
        root=x
        while parent[root]!=root:root=parent[root]
        while parent[x]!=x:
            q=parent[x];parent[x]=root;x=q
        return root
    def union(a,b):
        add(a);add(b);ra,rb=find(a),find(b)
        if ra!=rb:
            if repr(ra)<repr(rb):parent[rb]=ra
            else:parent[ra]=rb

    moved_surface={}
    for role in ROLE_ORDER:
        bp=point_array(baseline_surfaces[role],role+" baseline surface")
        tp=point_array(trial_surfaces[role],role+" trial surface")
        bf=np.asarray(baseline_surfaces[role]["faces"],np.int64)
        tf=np.asarray(trial_surfaces[role]["faces"],np.int64)
        if bp.shape!=tp.shape or not np.array_equal(bf,tf):
            raise ValueError(role+": component closure requires fixed surface topology/domain")
        rows=moved_point_rows(bp,tp);moved_surface[role]=rows
        labels={int(r):("surface",role,int(r)) for r in rows}
        for node in labels.values():add(node)
        move_set=set(labels)
        for face in tf:
            hit=sorted(move_set.intersection(map(int,face)))
            for r in hit[1:]:union(labels[hit[0]],labels[r])

    moved_volume={}
    for vrole in ("Skin","Fat"):
        bp=point_array(baseline_volumes[vrole],vrole+" baseline volume")
        tp=point_array(trial_volumes[vrole],vrole+" trial volume")
        bt=np.asarray(baseline_volumes[vrole]["tetrahedra"],np.int64)
        tt=np.asarray(trial_volumes[vrole]["tetrahedra"],np.int64)
        bm=np.asarray(baseline_volumes[vrole]["material_indices"])
        tm=np.asarray(trial_volumes[vrole]["material_indices"])
        if bp.shape!=tp.shape or not np.array_equal(bt,tt) or not np.array_equal(bm,tm):
            raise ValueError(vrole+": component closure requires unchanged point/tet/material domains")
        rows=moved_point_rows(bp,tp);moved_volume[vrole]=rows
        row_to_node={int(r):("volume",vrole,int(r)) for r in rows}
        for node in row_to_node.values():add(node)
        move_set=set(row_to_node)
        for tet in tt:
            hit=[int(r) for r in tet if int(r) in move_set]
            for r in hit[1:]:union(row_to_node[hit[0]],row_to_node[r])

    # Couple shared exact-coordinate variables across both volume layers.
    shared_volume_keys={}
    for vrole in ("Skin","Fat"):
        tp=point_array(trial_volumes[vrole],vrole+" trial volume")
        for row in moved_volume[vrole]:
            key=_bytes_key(tp[int(row)])
            shared_volume_keys.setdefault(key,[]).append(("volume",vrole,int(row)))
    for nodes in shared_volume_keys.values():
        for node in nodes[1:]:union(nodes[0],node)

    # Couple each moved source-surface row to its exact volume point in every
    # material layer where that role participates.
    surface_volume_rows={}
    crosswalk_mismatch_seeds=[]
    inactive_unpaired_surface_moves=[]
    for role in ROLE_ORDER:
        for vrole in ROLE_VOLUMES[role]:
            bp=point_array(baseline_volumes[vrole],vrole+" baseline volume")
            tp=point_array(trial_volumes[vrole],vrole+" trial volume")
            lookup={}
            for row,p in enumerate(bp):
                key=_bytes_key(p)
                if key in lookup:raise ValueError(vrole+": ambiguous baseline F32 point crosswalk")
                lookup[key]=row
            sb=point_array(baseline_surfaces[role],role+" baseline surface")
            st=point_array(trial_surfaces[role],role+" trial surface")
            changed=set(map(int,moved_surface[role]))
            rows_by_surface={}
            role_faces=np.asarray(trial_surfaces[role]["faces"],np.int64)
            for sr in moved_surface[role]:
                sr=int(sr);vr=lookup.get(_bytes_key(sb[sr]))
                if vr is None:
                    if not np.any(role_faces==sr):
                        inactive_unpaired_surface_moves.append({"role":role,"surface_point_row":sr,"volume_role":vrole,"volume_point_row":None,"reason":"inactive_surface_point_absent_from_baseline_volume"})
                        continue
                    raise ValueError(role+"/"+vrole+": moved active surface point missing in baseline volume")
                if not np.array_equal(tp[vr].view(np.uint32),st[sr].view(np.uint32)):
                    # A one-way surface-only update is not silently accepted.
                    # Seed it for exact rollback if the paired volume stayed at
                    # its authenticated baseline coordinate; any other mismatch
                    # indicates a broken point-domain/crosswalk contract.
                    if not np.array_equal(tp[vr].view(np.uint32),bp[vr].view(np.uint32)):
                        raise ValueError(role+"/"+vrole+": trial surface/volume mismatch is not baseline-volume-local")
                    if not np.array_equal(bp[vr].view(np.uint32),sb[sr].view(np.uint32)):
                        raise ValueError(role+"/"+vrole+": baseline surface/volume crosswalk mismatch")
                    item={"role":role,"surface_point_row":sr,"volume_role":vrole,"volume_point_row":int(vr)}
                    if np.any(np.asarray(trial_surfaces[role]["faces"],np.int64)==sr):
                        crosswalk_mismatch_seeds.append(item)
                    else:
                        inactive_unpaired_surface_moves.append(item)
                    rows_by_surface[sr]=int(vr)
                    continue
                if vr not in set(map(int,moved_volume[vrole])):
                    raise ValueError(role+"/"+vrole+": moved surface point is not a moved volume variable")
                union(("surface",role,sr),("volume",vrole,int(vr)))
                rows_by_surface[sr]=int(vr)
            surface_volume_rows[f"{role}/{vrole}"]=rows_by_surface

    seed_nodes=[]
    for role in ROLE_ORDER:
        for row in surface_plan["rollback_point_rows_by_role"][role]:
            node=("surface",role,int(row))
            if node not in parent:raise ValueError("witness seed is not a moved surface variable")
            seed_nodes.append(node)
    for item in crosswalk_mismatch_seeds:
        seed_nodes.append(("surface",item["role"],int(item["surface_point_row"])))
    seed_roots={find(x) for x in seed_nodes}
    if not seed_roots:raise ValueError("no witnessed component seeds")
    surface_rollback={r:[] for r in ROLE_ORDER}
    volume_rollback={r:[] for r in ("Skin","Fat")}
    for role in ROLE_ORDER:
        for row in moved_surface[role]:
            node=("surface",role,int(row))
            if find(node) in seed_roots:surface_rollback[role].append(int(row))
    for vrole in ("Skin","Fat"):
        for row in moved_volume[vrole]:
            node=("volume",vrole,int(row))
            if find(node) in seed_roots:volume_rollback[vrole].append(int(row))
    expanded=dict(surface_plan)
    expanded["surface_seed_rollback_point_rows_by_role"]=surface_plan["rollback_point_rows_by_role"]
    expanded["rollback_point_rows_by_role"]={r:sorted(set(surface_rollback[r])) for r in ROLE_ORDER}
    expanded["rollback_volume_point_rows_by_role"]={r:sorted(set(volume_rollback[r])) for r in ("Skin","Fat")}
    expanded["component_closure"]={
        "method":"moved O/I/D surface-face plus Skin/Fat moved-node tetrahedron graph joined by exact Float32 coordinate identity",
        "seed_surface_rows":{r:list(map(int,surface_plan["rollback_point_rows_by_role"][r])) for r in ROLE_ORDER},
        "expanded_surface_rows":expanded["rollback_point_rows_by_role"],
        "expanded_volume_rows":expanded["rollback_volume_point_rows_by_role"],
        "moved_surface_row_counts":{r:int(len(moved_surface[r])) for r in ROLE_ORDER},
        "moved_volume_row_counts":{r:int(len(moved_volume[r])) for r in ("Skin","Fat")},
        "expanded_component_count":len(seed_roots),
        "surface_volume_crosswalk_counts":{k:len(v) for k,v in surface_volume_rows.items()},
        "baseline_volume_unpaired_surface_move_seed_rows":crosswalk_mismatch_seeds,
        "inactive_unpaired_surface_moves_preserved":inactive_unpaired_surface_moves,
        "inactive_unpaired_surface_moves":inactive_unpaired_surface_moves,
        "complete_free_node_tet_component_closure":True,
    }
    # The surface rollback set may enlarge across role/layer couplings. Safe rows
    # are all moved rows outside the expanded closure.
    expanded["retained_safe_point_rows_by_role"]={
        r:sorted(set(map(int,moved_surface[r]))-set(expanded["rollback_point_rows_by_role"][r]))
        for r in ROLE_ORDER
    }
    return expanded
