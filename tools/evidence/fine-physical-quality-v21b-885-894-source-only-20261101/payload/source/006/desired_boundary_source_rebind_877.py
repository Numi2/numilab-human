#!/usr/bin/env python3
"""Exact-coordinate rebinding for current desired-boundary source aliases.

Metadata-only: geometry and desired-boundary faces are read-only. Every target
face must resolve to exactly one current typed source face (or authenticated
rim face) with matching oriented Float32 coordinate triples. Ambiguities fail
closed; no nearest or row-order fallback exists.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse, hashlib, json
import numpy as np

SCHEMA="numi.human.current-desired-boundary-source-alias-rebind.v1"
ROLE_SOURCES={
    "Skin":(("O",0,False),("I",1,True)),
    "Fat":(("I",0,False),("D",1,True)),
}
SOURCE_ALIAS_KEYS=("desired_boundary_source_group","desired_boundary_source_rows","desired_boundary_source_face_rows",
    "proposed_current_desired_boundary_source_group","proposed_current_desired_boundary_source_rows",
    "proposed_current_desired_boundary_source_face_rows")
GEOMETRY_KEYS=("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces","desired_boundary_faces")
IDENTITY_KEYS=("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id",
    "face_source_identity0","face_source_identity1")

def _check(ok,msg):
    if not ok: raise ValueError(msg)
def _scalar(v):
    a=np.asarray(v)
    if a.size!=1: raise ValueError("expected scalar metadata")
    x=a.reshape(-1)[0]
    return x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x)
def _array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def _same(a,b):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.view(np.uint8).tobytes()==b.view(np.uint8).tobytes()
def _point_bytes(p):
    q=np.ascontiguousarray(np.asarray(p,dtype=np.float32))
    if q.shape!=(3,): raise ValueError("face coordinate must be one F32 xyz point")
    return q.view(np.uint32).tobytes()
def _directed_triangle_key(xyz):
    """Canonicalize cyclic starts while retaining orientation and exact F32 bits."""
    if np.asarray(xyz).shape!=(3,3): raise ValueError("triangle coordinates must be (3,3)")
    q=np.asarray(xyz,dtype=np.float32)
    a,b,c=(_point_bytes(q[i]) for i in range(3))
    return min(a+b+c,b+c+a,c+a+b)
def _unoriented_triangle_key(xyz):
    p=sorted(_point_bytes(xyz[i]) for i in range(3))
    return tuple(p)
def _identity(surface,row):
    n=len(np.asarray(surface.get("faces",[])))
    out={}
    for k in IDENTITY_KEYS:
        if k not in surface: continue
        a=np.asarray(surface[k])
        if a.ndim==0 or a.shape[0]!=n: continue
        v=a[row]
        if np.asarray(v).size==1:
            x=np.asarray(v).reshape(-1)[0]
            if isinstance(x,(bytes,np.bytes_)):x=x.decode(errors="replace")
            if isinstance(x,np.generic):x=x.item()
            out[k]=x
        else: out[k]=np.asarray(v).tolist()
    return out

def _source_points(surface,name):
    if "points_world_m_f32" not in surface:
        raise ValueError(f"{name}: typed source lacks points_world_m_f32")
    p=np.asarray(surface["points_world_m_f32"],np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)):
        raise ValueError(f"{name}: malformed/nonfinite F32 source points")
    if "points_world_m" in surface:
        f64=np.asarray(surface["points_world_m"])
        if f64.shape!=p.shape or not np.all(np.isfinite(f64)):
            raise ValueError(f"{name}: source F64 point lane shape/nonfinite mismatch")
        if not np.array_equal(f64.astype(np.float32).view(np.uint32),p.view(np.uint32)):
            raise ValueError(f"{name}: source F64/F32 rounded point lanes differ")
    return p

def _add_source_faces(index,ambiguous,surface,name,group,reverse):
    p=_source_points(surface,name)
    if "faces" not in surface: raise ValueError(f"{name}: typed source lacks faces")
    f=np.asarray(surface["faces"],np.int64)
    if f.ndim!=2 or f.shape[1]!=3 or (f.size and (f.min()<0 or f.max()>=len(p))):
        raise ValueError(f"{name}: malformed face rows")
    if f.size and np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,2]==f[:,0])):
        raise ValueError(f"{name}: face has duplicate point rows")
    for row,face in enumerate(f):
        xyz=p[face]
        if len({_point_bytes(x) for x in xyz})!=3:
            raise ValueError(f"{name}: face {row} has duplicate exact F32 coordinates")
        oriented=xyz[[0,2,1]] if reverse else xyz
        key=_directed_triangle_key(oriented)
        c={"group":int(group),"source_role":name,"source_row":int(row),"source_face_row":int(row),
           "source_identity":_identity(surface,row)}
        if key in ambiguous:
            if len(ambiguous[key])<4:ambiguous[key].append(c)
        elif key in index:
            ambiguous[key]=[index.pop(key),c]
        else:index[key]=c

def rebind_source_aliases(role,volume_arrays,typed_surfaces,wall_face_points_world_m_f32,
                          wall_faces_volume_rows,wall_source_rows,*,input_pins=None):
    """Return a patched NPZ-array dict and provenance report; never changes geometry.

    Group IDs and source-row domains match the existing 677/709 contract:
    Skin: O=0, reversed I=1, rim=2.
    Fat: I=0, reversed D=1, rim=2.
    Source face row is the current typed surface face row. Rim source row is the
    authenticated wall-provenance row supplied by the caller.
    """
    if role not in ROLE_SOURCES:raise ValueError("role must be Skin or Fat")
    if not isinstance(volume_arrays,dict):raise TypeError("volume_arrays must be an NPZ array mapping")
    if "points_world_m_f32" in volume_arrays:vp=np.asarray(volume_arrays["points_world_m_f32"],np.float32)
    elif "points_world_m" in volume_arrays:vp=np.asarray(volume_arrays["points_world_m"],np.float32)
    else:raise ValueError("volume lacks current F32 points")
    if vp.ndim!=2 or vp.shape[1]!=3 or not np.all(np.isfinite(vp)):raise ValueError("invalid current volume F32 points")
    if "desired_boundary_faces" not in volume_arrays:raise ValueError("volume lacks desired_boundary_faces")
    target=np.asarray(volume_arrays["desired_boundary_faces"],np.int64)
    if target.ndim!=2 or target.shape[1]!=3 or (target.size and (target.min()<0 or target.max()>=len(vp))):
        raise ValueError("desired boundary face index domain is invalid")
    if target.size and np.any((target[:,0]==target[:,1])|(target[:,1]==target[:,2])|(target[:,2]==target[:,0])):
        raise ValueError("desired boundary contains repeated vertex rows")
    target_xyz=vp[target]
    target_keys=[];seen_target={}
    for row,xyz in enumerate(target_xyz):
        und=_unoriented_triangle_key(xyz)
        if und in seen_target:
            raise ValueError(f"desired boundary has duplicate exact F32 face geometry at rows {seen_target[und]} and {row}")
        seen_target[und]=row
        target_keys.append(_directed_triangle_key(xyz))
    index={};ambiguous={}
    for source_name,group,reverse in ROLE_SOURCES[role]:
        if source_name not in typed_surfaces:raise ValueError(f"missing typed surface {source_name}")
        _add_source_faces(index,ambiguous,typed_surfaces[source_name],source_name,group,reverse)
    wall_xyz=np.asarray(wall_face_points_world_m_f32,np.float32)
    wall_rows=np.asarray(wall_faces_volume_rows,np.int64)
    wall_source_rows=np.asarray(wall_source_rows,np.int64)
    if wall_xyz.ndim!=3 or wall_xyz.shape[1:]!=(3,3):raise ValueError("wall face points must be (n,3,3)")
    if wall_rows.shape!=wall_xyz.shape[:2] or wall_source_rows.shape!=(len(wall_xyz),):
        raise ValueError("wall face row/source-row domains are not aligned")
    if wall_rows.size and (wall_rows.min()<0 or wall_rows.max()>=len(vp)):raise ValueError("wall volume row outside point domain")
    if len(np.unique(wall_source_rows))!=len(wall_source_rows):raise ValueError("duplicate authenticated wall source row")
    if not np.all(np.isfinite(wall_xyz)):raise ValueError("nonfinite wall face coordinate")
    if not _same(vp[wall_rows],wall_xyz):raise ValueError("existing wall row/coordinate rim bindings differ")
    for i,xyz in enumerate(wall_xyz):
        if len({_point_bytes(x) for x in xyz})!=3:raise ValueError(f"wall source face {int(wall_source_rows[i])} is degenerate")
        key=_directed_triangle_key(xyz)
        c={"group":2,"source_role":"wall","source_row":int(wall_source_rows[i]),
           "source_face_row":int(wall_source_rows[i]),"source_identity":{"wall_source_row":int(wall_source_rows[i])}}
        if key in ambiguous:
            if len(ambiguous[key])<4:ambiguous[key].append(c)
        elif key in index:ambiguous[key]=[index.pop(key),c]
        else:index[key]=c
    groups=np.empty(len(target),np.int8);rows=np.empty(len(target),np.int64);face_rows=np.empty(len(target),np.int64)
    used=[];missing=[];multi=[]
    for i,key in enumerate(target_keys):
        if key in ambiguous:
            multi.append({"target_face_row":i,"coordinate_key_sha256":hashlib.sha256(key).hexdigest(),
                          "typed_candidates":ambiguous[key]})
            continue
        c=index.get(key)
        if c is None:
            missing.append({"target_face_row":i,"coordinate_key_sha256":hashlib.sha256(key).hexdigest(),
                            "volume_node_rows":target[i].tolist()})
            continue
        groups[i]=c["group"];rows[i]=c["source_row"];face_rows[i]=c["source_face_row"]
        used.append(c)
    if multi:
        raise ValueError("ambiguous exact typed face identities; no group/row guessed: "+json.dumps(multi[:8],sort_keys=True))
    if missing:
        raise ValueError("desired boundary faces lack exact typed-source/rim match: "+json.dumps(missing[:8],sort_keys=True))
    used_wall={int(c["source_row"]) for c in used if c["group"]==2}
    expected_wall=set(map(int,wall_source_rows))
    if used_wall!=expected_wall:
        raise ValueError(f"rim coverage mismatch: unused={sorted(expected_wall-used_wall)[:12]}, unexpected={sorted(used_wall-expected_wall)[:12]}")
    usage=Counter(int(g) for g in groups)
    out=dict(volume_arrays)
    out["desired_boundary_source_group"]=groups
    out["desired_boundary_source_rows"]=rows
    out["desired_boundary_source_face_rows"]=face_rows
    out["proposed_current_desired_boundary_source_group"]=groups.copy()
    out["proposed_current_desired_boundary_source_rows"]=rows.copy()
    out["proposed_current_desired_boundary_source_face_rows"]=face_rows.copy()
    for k in GEOMETRY_KEYS:
        if k in volume_arrays:
            _check(k in out and _same(volume_arrays[k],out[k]),f"metadata rebinder changed geometry lane {k}")
    for k in ("desired_boundary_faces","mapped_current_boundary_faces","desired_boundary_degenerate_source_rows"):
        if k in volume_arrays:_check(_same(volume_arrays[k],out[k]),f"metadata rebinder changed protected boundary lane {k}")
    for k in SOURCE_ALIAS_KEYS:
        _check(np.asarray(out[k]).shape==(len(target),),f"rebuilt alias has wrong target-face row domain: {k}")
    report={
        "schema":SCHEMA,"status":"PASS_EXACT_CURRENT_TARGET_TYPED_SOURCE_ALIAS_REBIND",
        "role":role,"group_contract":{name:{"group":group,"reverse_orientation":reverse} for name,group,reverse in ROLE_SOURCES[role]},
        "target_face_count":len(target),"source_group_counts":{str(k):int(usage.get(k,0)) for k in (0,1,2)},
        "typed_source_face_counts":{name:int(len(np.asarray(typed_surfaces[name]["faces"]))) for name,_,_ in ROLE_SOURCES[role]},
        "wall_face_count":len(wall_xyz),"wall_rows_all_used_exactly_once":True,
        "target_faces_exact_oriented_F32_matched":len(target),"ambiguous_target_rows":0,"unmatched_target_rows":0,
        "unused_typed_source_faces":{name:int(len(np.asarray(typed_surfaces[name]["faces"]))-sum(1 for c in used if c["source_role"]==name)) for name,_,_ in ROLE_SOURCES[role]},
        "prior_aliases":{k:{"shape":list(np.asarray(volume_arrays[k]).shape),"sha256":_array_hash(volume_arrays[k])} for k in SOURCE_ALIAS_KEYS if k in volume_arrays},
        "new_alias_sha256":{k:_array_hash(out[k]) for k in SOURCE_ALIAS_KEYS},
        "input_pins":input_pins or {},"geometry_and_target_bytes_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
    }
    return out,report

def _surface(points,faces,identity):
    return {"points_world_m_f32":np.asarray(points,np.float32),"points_world_m":np.asarray(points,np.float32).astype(np.float64),
        "faces":np.asarray(faces,np.int64),"face_source_face_id":np.asarray(identity,np.int64)}

def run_controls():
    # Two disjoint role surfaces plus an authenticated rim; target row order is
    # deliberately interleaved and cyclic starts differ from source rows.
    pts=np.asarray(((0,0,0),(1,0,0),(0,1,0),
       (3,0,0),(4,0,0),(3,1,0),
       (6,0,0),(7,0,0),(6,1,0),
       (9,0,0),(10,0,0),(9,1,0)),np.float32)
    o=_surface(pts[:3],[[0,1,2]],[101])
    i=_surface(pts[3:6],[[0,1,2]],[201])
    d=_surface(pts[6:9],[[0,1,2]],[301])
    wallfaces=np.asarray([[9,10,11]],np.int64);wallxyz=pts[wallfaces]
    common={"points_world_m_f32":pts.copy(),"points_world_m":pts.astype(np.float64),"tetrahedra":np.asarray([[0,1,2,3]],np.int64),
        "material_indices":np.asarray([17],np.int32),"boundary_faces":np.asarray([[0,1,2]],np.int64)}
    skin_target=np.asarray([[1,2,0],[3,5,4],[9,10,11]],np.int64)
    skin=dict(common,desired_boundary_faces=skin_target.copy(),
        desired_boundary_source_group=np.full(4,99,np.int8),desired_boundary_source_rows=np.arange(4,dtype=np.int64),
        desired_boundary_source_face_rows=np.arange(4,dtype=np.int64),mapped_current_boundary_faces=np.asarray([[0,1,2]],np.int64),
        desired_boundary_degenerate_source_rows=np.empty(0,np.int64),
        proposed_current_desired_boundary_source_group=np.full(4,99,np.int8),proposed_current_desired_boundary_source_rows=np.arange(4,dtype=np.int64),
        proposed_current_desired_boundary_source_face_rows=np.arange(4,dtype=np.int64))
    out,rep=rebind_source_aliases("Skin",skin,{"O":o,"I":i},wallxyz,wallfaces,np.asarray([0],np.int64))
    assert rep["source_group_counts"]=={"0":1,"1":1,"2":1}
    assert np.array_equal(out["desired_boundary_source_group"],np.asarray([0,1,2],np.int8))
    assert np.array_equal(out["desired_boundary_source_rows"],np.asarray([0,0,0],np.int64))
    for k in GEOMETRY_KEYS:assert _same(skin[k],out[k]),k
    assert len(out["desired_boundary_source_group"])==len(skin_target)
    fat_target=np.asarray([[4,5,3],[6,8,7],[9,10,11]],np.int64)
    fat=dict(common,desired_boundary_faces=fat_target,
       desired_boundary_source_group=np.empty(0,np.int8),desired_boundary_source_rows=np.empty(0,np.int64),
       desired_boundary_source_face_rows=np.empty(0,np.int64),proposed_current_desired_boundary_source_group=np.empty(0,np.int8),
       proposed_current_desired_boundary_source_rows=np.empty(0,np.int64),proposed_current_desired_boundary_source_face_rows=np.empty(0,np.int64))
    fout,frep=rebind_source_aliases("Fat",fat,{"I":i,"D":d},wallxyz,wallfaces,np.asarray([90],np.int64))
    assert frep["source_group_counts"]=={"0":1,"1":1,"2":1}
    assert np.array_equal(fout["desired_boundary_source_group"],np.asarray([0,1,2],np.int8))
    def must_reject(name,fn):
        try:fn()
        except ValueError as e:return {"status":"PASS_REJECTED","reason":str(e)[:240]}
        raise AssertionError("control accepted invalid mapping: "+name)
    dup=_surface(pts[:3],[[0,1,2],[0,1,2]],[401,402])
    amb=must_reject("duplicate typed identities",lambda:rebind_source_aliases("Skin",skin,{"O":dup,"I":i},wallxyz,wallfaces,np.asarray([0],np.int64)))
    badtarget=dict(skin);badtarget["desired_boundary_faces"]=skin_target.copy();badtarget["desired_boundary_faces"][0]=[1,0,2]
    orient=must_reject("orientation mismatch",lambda:rebind_source_aliases("Skin",badtarget,{"O":o,"I":i},wallxyz,wallfaces,np.asarray([0],np.int64)))
    unmatched=dict(skin);unmatched["desired_boundary_faces"]=skin_target.copy();unmatched["desired_boundary_faces"][0]=[0,1,3]
    miss=must_reject("unmatched target",lambda:rebind_source_aliases("Skin",unmatched,{"O":o,"I":i},wallxyz,wallfaces,np.asarray([0],np.int64)))
    crossdup=_surface(pts[:3],[[0,1,2]],[501])
    crossi=_surface(pts[:3],[[0,2,1]],[601])
    cross=dict(common,desired_boundary_faces=np.asarray([[0,1,2]],np.int64))
    crossamb=must_reject("cross-role duplicate typed identity",lambda:rebind_source_aliases("Skin",cross,{"O":crossdup,"I":crossi},np.empty((0,3,3),np.float32),np.empty((0,3),np.int64),np.empty(0,np.int64)))
    return {"status":"PASS_DESIRED_BOUNDARY_SOURCE_ALIAS_REBIND_CONTROLS",
        "skin":{"status":"PASS","source_group_counts":rep["source_group_counts"],"stale_aliases_rebased":True},
        "fat":{"status":"PASS","source_group_counts":frep["source_group_counts"],"stale_aliases_rebased":True},
        "rejections":{"same_surface_duplicate_typed_face":amb,"orientation_mismatch":orient,"unmatched_target_face":miss,
            "cross_role_exact_identity_ambiguity":crossamb},
        "geometry_bytes_unchanged":True,"target_boundary_bytes_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--self-test",action="store_true");a=ap.parse_args()
    if a.self_test:print(json.dumps(run_controls(),indent=2));return
    ap.error("This import-safe helper exposes rebind_source_aliases(); the owner driver supplies pinned NPZ inputs/output paths.")
if __name__=="__main__":main()
