"""Lossless empty-run cache compaction for the existing Manifold authoring path.

This changes source-run bookkeeping only. Oriented world triangles, material
properties, source face IDs, per-run transforms and flags remain identical.
No as_original(), geometric simplification, or new source IDs are used.
"""
import hashlib
import numpy as np

def _expanded(mesh):
    raw_ri=np.asarray(mesh.run_index,np.int64)
    if np.any(raw_ri%3):raise ValueError("CSG source-run corner offsets are not divisible by three")
    ri=raw_ri//3
    ro=np.asarray(mesh.run_original_id,np.uint32)
    f=np.asarray(mesh.tri_verts,np.uint64)
    if len(ri)!=len(ro)+1 or ri[0]!=0 or ri[-1]!=len(f) or np.any(np.diff(ri)<0):
        raise ValueError("invalid CSG source-run table")
    valid=np.flatnonzero(np.diff(ri)>0);counts=np.diff(ri)[valid]
    run_for_face=np.repeat(valid,counts)
    return ri,ro,f,valid,counts,run_for_face

def oriented_surface_source_digest(mesh):
    _,ro,f,_,_,run=_expanded(mesh)
    props=np.asarray(mesh.vert_properties,dtype="<f8")
    triangles=props[f]
    if not len(f):return hashlib.sha256(b"empty-oriented-csg-surface").hexdigest()
    width=triangles.shape[1]*triangles.shape[2]*8
    rotations=np.stack([np.roll(triangles,-j,axis=1) for j in range(3)],axis=1)
    raw=np.ascontiguousarray(rotations.reshape(len(f),3,-1)).view(np.dtype("V"+str(width))).reshape(len(f),3)
    canonical=np.sort(raw,axis=1)[:,0]
    transforms=np.asarray(mesh.run_transform,dtype="<f8")
    flags=np.asarray(mesh.run_flags,dtype=np.uint8)
    if transforms.size and transforms.shape!=(len(ro),4,3):raise ValueError("CSG run transform domain mismatch")
    if flags.size and flags.shape!=(len(ro),):raise ValueError("CSG run flag domain mismatch")
    record=np.empty(len(f),dtype=[("origin","<u4"),("face","<u8"),("triangle","V"+str(width)),("transform","V96"),("flags","u1")])
    record["origin"]=ro[run];record["face"]=np.asarray(mesh.face_id,np.uint64);record["triangle"]=canonical
    if transforms.size:record["transform"]=np.ascontiguousarray(transforms[run]).reshape(len(f),12).view("V96").reshape(-1)
    else:
        identity=np.zeros((4,3),dtype="<f8");identity[:3,:3]=np.eye(3)
        record["transform"]=np.asarray([identity]).reshape(1,12).view("V96").reshape(-1)[0]
    record["flags"]=flags[run] if flags.size else 0
    return hashlib.sha256(np.sort(record,order=record.dtype.names).tobytes()).hexdigest()

def compact_mesh_empty_runs(mesh,verify=True):
    """Strip empty cache runs without reimporting or changing face order."""
    import manifold3d as mf
    ri,ro,f,valid,counts,run=_expanded(mesh)
    kwargs=dict(vert_properties=np.array(mesh.vert_properties,np.float64,order="C",copy=True),
        tri_verts=np.array(f,np.uint64,order="C",copy=True),
        run_index=np.asarray(np.r_[0,3*np.cumsum(counts)],np.uint64),
        run_original_id=np.asarray(ro[valid],np.uint32),
        face_id=np.asarray(mesh.face_id,np.uint64).copy(),
        tolerance=float(getattr(mesh,"tolerance",0.0)))
    for field,dtype in (("run_transform",np.float64),("run_flags",np.uint8)):
        value=np.asarray(getattr(mesh,field,[]),dtype)
        if value.size:kwargs[field]=np.array(value[valid],dtype,order="C",copy=True)
    for field,dtype in (("merge_from_vert",np.uint64),("merge_to_vert",np.uint64),("halfedge_tangent",np.float64)):
        value=np.asarray(getattr(mesh,field,[]),dtype)
        if value.size:kwargs[field]=np.array(value,dtype,order="C",copy=True)
    result=mf.Mesh64(**kwargs)
    before=oriented_surface_source_digest(mesh) if verify else None
    after=oriented_surface_source_digest(result) if verify else None
    if verify and before!=after:raise ValueError("empty CSG cache run removal changed source or geometry")
    return result,{"before_runs":len(ro),"after_runs":len(valid),"faces":len(f),
        "source_and_geometry_digest_before":before,"source_and_geometry_digest_after":after,
        "exact_cache_geometry_and_source_identity":bool(verify and before==after),
        "face_order_preserved":True,"manifold_reimported":False}
