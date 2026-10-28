"""Bounded local authoring repair on the pinned connected skin; no admission."""
import numpy as np

MAX_LOCAL_REPAIR_M = 50e-6

def _boundary(faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inv,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError("local repair boundary is nonmanifold")
    return {tuple(map(int,e)) for e in edges[count[inv]==1]}

def _ancestry(candidate, count):
    keys=("face_ancestry_parent_face_rows","face_ancestry_source_original_ids","face_ancestry_source_face_rows")
    off=np.asarray(candidate["face_ancestry_offsets"],np.int64)
    a=[np.asarray(candidate[k],np.int64) for k in keys]
    if off.shape!=(count+1,) or off[0]!=0 or np.any(np.diff(off)<=0) or any(x.ndim!=1 or len(x)!=off[-1] or np.any(x<0) for x in a):
        raise ValueError("local repair typed ancestry CSR is invalid")
    return [set(zip(*(x[off[i]:off[i+1]].tolist() for x in a))) for i in range(count)]

def validate_local_repair(reference, repair, reference_sha256):
    if str(np.asarray(repair["local_repair_parent_sha256"]).item())!=reference_sha256:
        raise ValueError("local repair parent hash mismatch")
    p=np.asarray(reference["points_world_m"],np.float32);q=np.asarray(repair["points_world_m"],np.float32)
    f=np.asarray(reference["source_O_faces"],np.int64);g=np.asarray(repair["source_O_faces"],np.int64)
    if p.ndim!=2 or p.shape[1]!=3 or q.shape!=p.shape or not np.isfinite(q).all():
        raise ValueError("local repair point domain or finite coordinates failed")
    if f.ndim!=2 or f.shape[1]!=3 or g.shape!=f.shape or not len(g) or np.any(g<0) or np.any(g>=len(p)):
        raise ValueError("local repair must preserve the bounded face-row domain")
    for key in ("candidate_vertex_to_O033_row","O033_to_candidate_vertex_row"):
        if not np.array_equal(reference[key],repair[key]):raise ValueError("local repair changed source node identity")
    allowed=np.asarray(repair["local_repair_allowed_vertex_rows"])
    if allowed.ndim!=1 or not np.issubdtype(allowed.dtype,np.integer) or len(np.unique(allowed))!=len(allowed) or np.any(allowed<0) or np.any(allowed>=len(p)):
        raise ValueError("local repair allowed vertex rows are invalid")
    mask=np.zeros(len(p),bool);mask[allowed]=True
    changed=np.any(q.view(np.uint32)!=p.view(np.uint32),axis=1)
    if np.any(changed&~mask):raise ValueError("local repair moved a vertex outside its declared connected patch")
    delta=np.linalg.norm(q.astype(float)-p.astype(float),axis=1)
    if np.any(delta>MAX_LOCAL_REPAIR_M):raise ValueError("local repair exceeds the 50 micrometer authoring bound")
    fb=_boundary(f);gb=_boundary(g)
    if fb!=gb:raise ValueError("local repair changed retained boundary edges")
    boundary_rows=np.array(sorted({n for e in fb for n in e}),np.int64)
    if np.any(changed[boundary_rows]):raise ValueError("local repair moved retained boundary coordinates")
    topo=np.any(f!=g,axis=1)
    if np.any(~mask[g[topo]]):raise ValueError("local repair changed topology outside its declared patch")
    old=_ancestry(reference,len(f));new=_ancestry(repair,len(g))
    changed_rows=np.flatnonzero(topo)
    for i in np.flatnonzero(~topo):
        if old[i]!=new[i]:raise ValueError("local repair changed unaffected face ancestry")
    before=set().union(*(old[i] for i in changed_rows)) if len(changed_rows) else set()
    after=set().union(*(new[i] for i in changed_rows)) if len(changed_rows) else set()
    if before!=after:raise ValueError("local topology repair lost or invented typed source ancestry")
    tri=q.astype(float)[g];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    if np.any(np.linalg.norm(n,axis=1)==0):raise ValueError("local repair creates a Float32 degenerate face")
    same=~topo;ot=p.astype(float)[f[same]];on=np.cross(ot[:,1]-ot[:,0],ot[:,2]-ot[:,0])
    if np.any(np.einsum("ij,ij->i",on,n[same])<=0):raise ValueError("local repair flips a retained face")
    return {"reference_sha256":reference_sha256,"moved_vertices":int(changed.sum()),"allowed_vertices":len(allowed),
            "topology_changed_faces":int(topo.sum()),"maximum_displacement_m":float(delta.max()),
            "outside_patch_exact":True,"retained_boundary_coordinates_exact":True,
            "typed_face_ancestry_preserved":True,"inferred_geometry_not_source_measurement":True,
            "geometry_intersection_qualification":False,"physical_steps":0,"admitted":False}
