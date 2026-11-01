#!/usr/bin/env python3
"""Paired I-endpoint quotient followed by exact connected cavity remesh.

Source-only construction proposal. Each paired operation collapses one
link-valid I source edge, then remeshes the complete drop/keep node stars plus
one complete same-material face-neighbor ring in Skin and Fat. Coordinates
remain bit-identical. Native quality may tie on one role only when the other
strictly improves in the combined Skin+Fat count, or on a tied count by a measured combined log-deficit reduction; neither role may regress. All original exact boundary,
positive-cell, per-material, paired-flux, and ancestry checks remain active.
"""
from pathlib import Path
from collections import Counter
import argparse, copy, hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BATCH=ROOT/"quality-partial-cavity-684-root"/"paired_connected_boundary_patch_batch_881_embedded_ring_fixture_root.py"
BATCH_SHA="94a81128448f2a923676c1a9cc48e90750ce9deb859a7818ceadfcc9caa7c5f3"
CORE=HERE/"inner_endpoint_quotient_002.py"
CORE_SHA="e6ce2c436ce0a4ca700e2a4b1be524403576a066c42efe6c945c6c5cb87cea36"
REBINDER=ROOT/"desired-boundary-source-rebind-001-nativepreflight"/"desired_boundary_source_rebind_877.py"
REBINDER_SHA="78f60687bda5f416604608ef877c7a8c1251bb74e3c53b6929385ce586000a34"
MAX_EDGE_M=50e-6
QMIN=1e-5
VMIN=1e-18
MAX_CELLS=64
MAX_POINTS=24
MAX_LOCAL_VOL_FRAC=.01
MAX_GLOBAL_VOL_FRAC=1e-4
PROTECTOR=ROOT/"quality-partial-cavity-684-root"/"paired_connected_boundary_patch_856_extended_250um_construction.py"
PROTECTOR_SHA="25a2bf8cbb2ff776745cb3a68bb3ffdfe4b294d8a555f247ce1bc53b476c4897"

def sha(p): return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def pin(p):
    p=Path(p).resolve()
    return {"path":str(p),"sha256":sha(p)}
def imp(name,path,expected):
    path=Path(path).resolve()
    got=sha(path)
    if got!=expected: raise RuntimeError(f"{name} pinned SHA mismatch {got} != {expected}")
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader: raise RuntimeError("cannot import "+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def ref(p):return {"path":str(Path(p).resolve()),"sha256":sha(p)}
def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=json_default)+"\n")
def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def chain(f):return ENG.chain(np.asarray(f,np.int64))
def detsum(points,tets):
    ts=np.asarray(tets,np.int64);used=np.unique(ts.reshape(-1));local={int(v):i for i,v in enumerate(used)}
    ip=[ENG.S.S.ip(x) for x in np.asarray(points,np.float32)[used]]
    return sum(ENG.S.S.det(*(ip[local[int(v)]] for v in tet)) for tet in ts)
def exact_positive(points,tets):
    ts=np.asarray(tets,np.int64);used=np.unique(ts.reshape(-1));local={int(v):i for i,v in enumerate(used)}
    ip=[ENG.S.S.ip(x) for x in np.asarray(points,np.float32)[used]]
    return [ENG.S.S.det(*(ip[local[int(v)]] for v in tet)) for tet in ts]
def deficit(det,q):
    r=np.minimum(np.minimum(np.asarray(q,np.float64)/QMIN,np.asarray(det,np.float64)/(6*VMIN)),1.)
    return float(-np.log(np.maximum(r,1e-300)).sum())
def native(points,tets):
    d,q=M.measurements(np.asarray(points,np.float32),np.asarray(tets,np.int64))
    good=np.asarray(M.passes(d,q),bool)
    return np.asarray(d),np.asarray(q),good

P=imp("postquotient_base_881",BATCH,BATCH_SHA)
B=P.B;ENG=P.ENG;M=P.M;T=P.T
core=imp("endpoint_quotient_core_882",CORE,CORE_SHA)
rebinder=imp("desired_boundary_source_rebind_882",REBINDER,REBINDER_SHA)
protection_owner=imp("active_row_protection_855",PROTECTOR,PROTECTOR_SHA)
_CONTEXT_CACHE={}

def exact_map(surface,cache,label):
    return P.active_map(surface,cache,label)

def build_target_update(cache,source_before,source_after,sres,source_map,drop,drop_v,keep_v):
    oldf=np.asarray(source_before["faces"],np.int64)
    target=np.asarray(cache["target"],np.int64)
    tl=cache["target_lookup"]
    drop_parent=np.flatnonzero(np.any(oldf==int(drop),axis=1)).astype(np.int64)
    if not len(drop_parent):raise ValueError("drop endpoint has no source faces")
    oldtarget=[];oldrows=[];relations=[]
    for pr in drop_parent:
        vf=np.asarray(source_map[oldf[int(pr)]],np.int64)
        if np.any(vf<0):raise ValueError("affected I face has no exact volume point map")
        hits=tl.get(tuple(sorted(map(int,vf))),[])
        if len(hits)!=1:raise ValueError("affected I face lacks a unique current target facet")
        row=int(hits[0]);tf=target[row]
        oldtarget.append(tf.copy());oldrows.append(row);relations.append(int(ENG.rel(tf,vf)))
    if len(set(oldrows))!=len(oldrows):raise ValueError("affected I faces alias the same target row")
    if len(set(relations))!=1:raise ValueError("I target/source orientation relation changes within endpoint star")
    # No non-I target facet may be incident to the retiring volume vertex.
    touching=set(map(int,np.flatnonzero(np.any(target==int(drop_v),axis=1))))
    if touching!=set(oldrows):
        raise ValueError("retired point is shared by target facets outside the current I source patch")
    q=np.arange(len(cache["points"]),dtype=np.int64);q[int(drop_v)]=int(keep_v)
    mapped=q[target];s=np.sort(mapped,axis=1)
    valid=~np.any(s[:,1:]==s[:,:-1],axis=1)
    target_q=mapped[valid]
    changed_rows=np.asarray(sres["changed_output_rows"],np.int64)
    parent_rows=np.asarray(sres["face_output_to_parent_rows"],np.int64)
    if not np.array_equal(np.sort(parent_rows[changed_rows]),np.sort(np.intersect1d(parent_rows[changed_rows],drop_parent))):
        raise ValueError("source changed-output parents do not belong to retired I star")
    newsource=np.asarray(sres["faces"],np.int64)[changed_rows]
    newvol=np.asarray(source_map[newsource],np.int64)
    if np.any(newvol<0):raise ValueError("new I face has no exact volume point map")
    rel_by_parent={int(pr):int(rel) for pr,rel in zip(drop_parent,relations)}
    # All affected parents share the same oriented I-to-volume boundary parity.
    rel=int(relations[0])
    if rel<0:newvol=newvol[:,[0,2,1]]
    if len(newvol)==0:raise ValueError("endpoint quotient emitted no surviving I facets")
    # Exact oriented target quotient must be precisely old target minus the retired
    # star plus the redirected surviving I facets.
    remove=np.zeros(len(target),bool);remove[np.asarray(oldrows,np.int64)]=True
    keep_target=mapped[valid & ~remove]
    constructed=np.vstack((keep_target,newvol))
    if chain(constructed)!=chain(target_q):
        raise ValueError("source patch replacement differs from direct target point quotient")
    if len(np.unique(np.sort(constructed,axis=1),axis=0))!=len(constructed):
        raise ValueError("target quotient patch contains duplicate facets")
    return {"target":target_q,"old_target":np.asarray(oldtarget,np.int64),
        "old_target_rows":np.asarray(oldrows,np.int64),"new_target_patch":newvol,
        "drop_parent_face_rows":drop_parent,"changed_output_face_rows":changed_rows,
        "face_parent_rows":parent_rows,"node_map":q,"orientation_relation":rel}

def cavity_boundary(cache,cellrows,update,solver_vertices):
    t=np.asarray(cache["tets"],np.int64)
    oldb=np.asarray(T.m.boundary(t[np.asarray(cellrows,np.int64)]),np.int64)
    oldchain=ENG.chain(oldb)
    for f in np.asarray(update["old_target"],np.int64):
        k=tuple(sorted(map(int,f)))
        if oldchain.get(k,0)!=ENG.fsign(f):
            raise ValueError("old I target facet is not on the oriented cavity boundary")
        oldchain.pop(k)
    for f in np.asarray(update["new_target_patch"],np.int64):
        k=tuple(sorted(map(int,f)))
        oldchain[k]=oldchain.get(k,0)+ENG.fsign(f)
        if oldchain[k]==0:oldchain.pop(k)
    newb_global=ENG.faces_from_chain(oldchain)
    closed,open_edges=P.checked_closed_boundary(newb_global,"post-quotient")
    topo=T.manifold_topology(newb_global)
    if not closed or topo["bad_vertex_links"] or topo["boundary_components"]!=1:
        raise ValueError("post-quotient cavity boundary is not one closed manifold")
    if np.any(newb_global==int(update["drop_volume_row"])):
        raise ValueError("dropped endpoint remains on replacement cavity boundary")
    qloc=np.full(len(cache["points"]),-1,np.int64)
    qloc[np.asarray(solver_vertices,np.int64)]=np.arange(len(solver_vertices),dtype=np.int64)
    newb=qloc[newb_global]
    if np.any(newb<0):raise ValueError("new cavity boundary uses point outside solver pool")
    return oldb,newb_global,newb,qloc,topo

def _face_components(tets):
    ts=np.asarray(tets,np.int64).reshape((-1,4))
    if len(ts)<=1:return int(len(ts))
    parent=list(range(len(ts)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[b]=a
    owner={}
    for i,tet in enumerate(ts):
        for omit in range(4):
            key=tuple(sorted(int(tet[j]) for j in range(4) if j!=omit))
            if key in owner:union(i,owner[key])
            else:owner[key]=i
    return len({find(i) for i in range(len(ts))})

def _cavity_size(cache,rows,drop):
    rows=np.asarray(rows,np.int64);t=np.asarray(cache["tets"],np.int64)
    verts=np.unique(t[rows]) if len(rows) else np.empty(0,np.int64)
    solver=verts[verts!=int(drop)]
    return {"rows":rows,"vertices":verts,"solver_vertices":solver,"cells":int(len(rows)),
        "solver_points_after_drop":int(len(solver)),"face_components":int(_face_components(t[rows])) if len(rows) else 0}

def select_endpoint_cavity(cache,seed,required_target_faces,material,drop,keep,max_cells=MAX_CELLS,max_points=MAX_POINTS):
    seed=np.unique(np.asarray(seed,np.int64))
    expected=np.unique(np.r_[B.incident_cells(cache,np.asarray([drop],np.int64)),
                             B.incident_cells(cache,np.asarray([keep],np.int64))]).astype(np.int64)
    if not len(seed) or not np.array_equal(seed,expected):raise ValueError("seed is not the complete two-endpoint star union")
    mats=np.unique(np.asarray(cache["mat"],np.int64)[seed])
    if len(mats)!=1 or int(mats[0])!=int(material):raise ValueError("endpoint-star seed crosses material")
    ring,ring_added,ringdiag=P.expand_complete_material_face_ring(cache,seed,required_target_faces,int(material))
    ring=np.asarray(ring,np.int64);ring_added=np.asarray(ring_added,np.int64)
    sr=_cavity_size(cache,seed,drop);rr=_cavity_size(cache,ring,drop)
    def fits(r):return r["cells"]<=int(max_cells) and r["solver_points_after_drop"]<=int(max_points) and r["face_components"]==1
    if fits(rr):chosen=rr;mode="complete_endpoint_star_plus_full_material_face_ring";added=ring_added;reasons=[]
    elif fits(sr):
        chosen=sr;mode="complete_endpoint_star_only_ring_exceeded_existing_cap";added=np.empty(0,np.int64);reasons=[]
        if rr["cells"]>int(max_cells):reasons.append("ring_cell_cap")
        if rr["solver_points_after_drop"]>int(max_points):reasons.append("ring_solver_point_cap_after_drop")
        if rr["face_components"]!=1:reasons.append("ring_not_face_connected")
    else:
        raise ValueError("neither full ring nor intact endpoint-star union fits existing caps: "+str({
            "ring_cells":rr["cells"],"ring_solver_points":rr["solver_points_after_drop"],
            "seed_cells":sr["cells"],"seed_solver_points":sr["solver_points_after_drop"],
            "caps":{"cells":int(max_cells),"solver_points_after_drop":int(max_points)}}))
    if not np.isin(expected,chosen["rows"]).all():raise ValueError("selected cavity omitted endpoint-star cells")
    selected_mat=np.unique(np.asarray(cache["mat"],np.int64)[chosen["rows"]])
    if len(selected_mat)!=1 or int(selected_mat[0])!=int(material):raise ValueError("selected cavity crosses material")
    meta={"policy":"prefer_full_ring_else_intact_endpoint_star_v1","mode":mode,"fallback_reasons":reasons,
        "cell_cap":int(max_cells),"solver_point_cap_after_drop":int(max_points),
        "complete_endpoint_star_cell_count":int(len(seed)),"ring_candidate_cell_count":int(len(ring)),
        "ring_candidate_added_cell_count":int(len(ring_added)),"selected_cell_count":int(chosen["cells"]),
        "selected_solver_point_count_after_drop":int(len(chosen["solver_vertices"])),
        "selected_face_connected_components":int(chosen["face_components"]),
        "complete_endpoint_star_rows":seed.copy(),"ring_candidate_rows":ring.copy(),
        "ring_added_rows":ring_added.copy(),"selected_rows":chosen["rows"].copy(),
        "selected_added_rows":np.asarray(added,np.int64),"ring_diagnostics":ringdiag}
    return chosen["rows"],np.asarray(added,np.int64),chosen["vertices"],chosen["solver_vertices"],meta

def cavity_selection_fixture_control():
    pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[-2,2,2]],np.float32)
    t=np.asarray(M.orient(pts,np.asarray([[0,1,2,3],[0,2,3,4],[2,3,4,5]],np.int64)),np.int64)
    target=np.asarray(T.m.boundary(t),np.int64)
    cache={"points":pts,"tets":t,"mat":np.full(3,16,np.int32),"target":target,
        "target_lookup":B.face_lookup(target),"incidence":B.cell_incidence(t,len(pts))}
    seed=np.unique(np.r_[B.incident_cells(cache,np.asarray([0],np.int64)),
                         B.incident_cells(cache,np.asarray([1],np.int64))]).astype(np.int64)
    required=np.asarray([f for f in target if tuple(sorted(map(int,f)))==(0,1,2)],np.int64)
    if seed.tolist()!=[0,1] or len(required)!=1:raise AssertionError("endpoint-star selector fixture malformed")
    fallback,_,_,solver,fd=select_endpoint_cavity(cache,seed,required,16,0,1,max_cells=2,max_points=24)
    ring,_,_,_,rd=select_endpoint_cavity(cache,seed,required,16,0,1,max_cells=64,max_points=24)
    if fallback.tolist()!=[0,1] or fd["mode"]!="complete_endpoint_star_only_ring_exceeded_existing_cap":
        raise AssertionError("ring overflow did not select the intact endpoint-star union")
    if ring.tolist()!=[0,1,2] or rd["mode"]!="complete_endpoint_star_plus_full_material_face_ring":
        raise AssertionError("complete ring was not preferred when it fit")
    if len(solver)!=len(np.unique(t[fallback]))-1:raise AssertionError("drop point was not removed before cap check")
    many=np.asarray([[0,1,2,3],[0,4,5,6],[0,7,8,9],[0,10,11,12],
        [0,13,14,15],[0,16,17,18],[0,19,20,21],[0,22,23,24]],np.int64)
    c24=_cavity_size({"tets":many},np.arange(len(many),dtype=np.int64),0)
    c27=_cavity_size({"tets":np.vstack((many,[[0,25,26,27]]))},np.arange(9,dtype=np.int64),0)
    if c24["solver_points_after_drop"]!=24 or c27["solver_points_after_drop"]!=27:
        raise AssertionError("point cap was not measured on the actual post-drop solver domain")
    return {"status":"PASS_SEED_STAR_FALLBACK_AND_POST_DROP_CAP","seed_rows":seed.tolist(),
        "fallback_rows":fallback.tolist(),"ring_rows":ring.tolist(),"solver_points_at_cap":24,
        "solver_points_over_cap":27,"complete_no_truncation":True}

def solve_role(vr,cache,source_before,sres,source_map,drop,keep,protections,inner_allowed,solve_seconds,source_role="I"):
    target=np.asarray(cache["target"],np.int64);points=np.asarray(cache["points"],np.float32)
    src_drop=int(drop);cross=np.asarray(source_map,np.int64)
    dv=int(cross[src_drop]);kv=int(cross[int(keep)])
    if min(dv,kv)<0 or dv==kv:raise ValueError(vr+" endpoint volume rows are missing or aliased")
    update=build_target_update(cache,source_before,{**source_before,"faces":sres["faces"]},sres,cross,src_drop,dv,kv)
    seed=np.unique(np.r_[B.incident_cells(cache,np.asarray([dv],np.int64)),
                        B.incident_cells(cache,np.asarray([kv],np.int64))]).astype(np.int64)
    if not len(seed):raise ValueError(vr+" endpoint node-star cavity is empty")
    mat=np.unique(np.asarray(cache["mat"])[seed])
    if len(mat)!=1:raise ValueError(vr+" endpoint node stars cross material indices")
    material=int(mat[0]);target_patch=update["old_target"]
    cellrows,added,verts,solveverts,cavity_selection=select_endpoint_cavity(
        cache,seed,target_patch,material,dv,kv,max_cells=MAX_CELLS,max_points=MAX_POINTS)
    oldt=np.asarray(cache["tets"][cellrows],np.int64);oldmat=np.asarray(cache["mat"][cellrows])
    # Preserve source-owned geometry and protected anatomy. Same-role back rows
    # may be released only through the pinned 881 inner-back policy.
    all_protected=set(map(int,protections[vr]["volume_rows"]))
    own_protected=P.own_protected_volume_rows(source_role,vr,protections[vr],cross,inner_allowed)
    conflict=P.protected_volume_star_conflicts(verts,all_protected,own_protected,True)
    if conflict:raise ValueError(vr+" cavity touches non-released protected source rows "+str(conflict[:8]))
    # The retired vertex is never offered to the replacement MILP.
    if len(solveverts)==0:raise ValueError(vr+" point pool is empty after removing the retired endpoint")
    cupdate=dict(update);cupdate["drop_volume_row"]=dv
    oldb,newb_global,newb,qloc,topo=cavity_boundary(cache,cellrows,cupdate,solveverts)
    # Old exact oriented cell sum and new boundary flux are authoritative.
    oldexact=exact_positive(points,oldt)
    if not oldexact or min(oldexact)<=0:raise ValueError(vr+" old cavity has nonpositive exact F32 tet")
    oldsum=sum(oldexact)
    lp=points[solveverts]
    newflux=ENG.exact_boundary_flux(lp,newb)
    pre_delta=int(newflux-oldsum)
    global_flux=int(ENG.exact_boundary_flux(points,target))
    if newflux<=0:raise ValueError(vr+" remeshed cavity boundary has nonpositive exact signed volume")
    if abs(pre_delta)/float(oldsum)>MAX_LOCAL_VOL_FRAC:raise ValueError(vr+" local exact volume delta exceeds 1% cap")
    if abs(pre_delta)/float(global_flux)>MAX_GLOBAL_VOL_FRAC:raise ValueError(vr+" global exact volume delta exceeds 1e-4 cap")
    report,newlocal=ENG.S.solve(lp,newb,seconds=min(3.,max(.02,float(solve_seconds))),
                                max_points=MAX_POINTS,material=material)
    if newlocal is None or report.get("status")!="PASS_PARTIAL_POSITIVE_CAVITY_MILP":
        raise ValueError(vr+" no positive native-feasible cavity MILP incumbent: "+str(report.get("status")))
    newlocal=np.asarray(newlocal,np.int64)
    if np.any(newlocal<0) or np.any(newlocal>=len(solveverts)):raise ValueError(vr+" MILP emitted out-of-domain point")
    newglobal=solveverts[newlocal]
    newexact=exact_positive(points,newglobal)
    if not newexact or min(newexact)<=0:raise ValueError(vr+" replacement has exact nonpositive F32 tet")
    newsum=sum(newexact)
    if newsum!=newflux:raise ValueError(vr+" exact replacement-cell sum differs from oriented boundary flux")
    delta=int(newsum-oldsum)
    if delta!=pre_delta:raise ValueError(vr+" exact cavity delta changed after MILP")
    if abs(delta)/float(oldsum)>MAX_LOCAL_VOL_FRAC or abs(delta)/float(global_flux)>MAX_GLOBAL_VOL_FRAC:
        raise ValueError(vr+" final exact volume delta exceeds configured cap")
    nd,nq,ngood=native(points,newglobal)
    oldd,oldq,ogood=native(points,oldt)
    oldbad=int(np.count_nonzero(~ogood));newbad=int(np.count_nonzero(~ngood))
    olddef=deficit(oldd,oldq);newdef=deficit(nd,nq)
    if newbad>oldbad:raise ValueError(vr+" cavity native bad count regresses")
    tol=max(1e-8,1e-6*olddef)
    progress=(newbad<oldbad) or (newbad==oldbad and newdef<olddef-tol)
    return {"role":vr,"cell_rows":cellrows,"ring_added_rows":np.asarray(added,np.int64),
      "ring_diagnostics":cavity_selection["ring_diagnostics"],"cavity_selection":cavity_selection,"old_tets":oldt,"old_materials":oldmat,
      "cavity_vertices":verts,"solver_vertices":solveverts,"old_boundary":oldb,
      "new_boundary_local":newb,"new_tets":newglobal,"new_material":material,
      "target":update["target"],"old_target_rows":update["old_target_rows"],
      "exact_old_det6_sum":int(oldsum),"exact_new_det6_sum":int(newsum),
      "exact_det6_delta":delta,"global_exact_flux":global_flux,
      "native_bad_before":oldbad,"native_bad_after":newbad,"local_progress":bool(progress),
      "log_deficit_before":olddef,"log_deficit_after":newdef,
      "old_q_range":[float( oldq.min()),float(oldq.max())],
      "new_q_range":[float(nq.min()),float(nq.max())],
      "exact_new_min_det6":str(min(newexact)),"solver":report,
      "replacement_parent_offsets":np.arange(len(newglobal)+1,dtype=np.int64)*len(cellrows),
      "drop_volume_row":dv,"keep_volume_row":kv,"target_update":update,
      "protected_volume_rows_touched":sorted(set(map(int,verts))&all_protected)}

def build_volume_result(cache,pr):
    parent=cache["arrays"];t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"])
    cellrows=np.asarray(pr["cell_rows"],np.int64);mask=np.ones(len(t),bool);mask[cellrows]=False
    retained=np.flatnonzero(mask).astype(np.int64)
    newt=np.asarray(pr["new_tets"],np.int64).reshape((-1,4))
    newmat=np.full(len(newt),int(pr["new_material"]),dtype=mat.dtype)
    outt=np.vstack((t[retained],newt));outmat=np.r_[mat[retained],newmat]
    actual=np.asarray(T.m.boundary(outt),np.int64)
    target=np.asarray(pr["target"],np.int64)
    if chain(actual)!=chain(target):raise ValueError(pr["role"]+" full output boundary differs from quotient current target")
    if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError(pr["role"]+" output has duplicate tetrahedra")
    q=np.arange(len(cache["points"]),dtype=np.int64);q[int(pr["drop_volume_row"])]=int(pr["keep_volume_row"])
    origin=np.r_[retained,np.full(len(newt),-1,np.int64)]
    inverse=np.full(len(t),-1,np.int64);inverse[retained]=np.arange(len(retained),dtype=np.int64)
    newrows=np.arange(len(retained),len(outt),dtype=np.int64)
    affected=cellrows
    rp_off=np.zeros(len(newrows)+1,np.int64)
    if len(newrows):rp_off[1:]=np.arange(1,len(newrows)+1,dtype=np.int64)*len(affected)
    vres={"tetrahedra":outt,"material_indices":outmat,"boundary_faces":actual,"desired_boundary_faces":target,
      "output_cell_to_input_cell_rows":origin,"retained_input_cell_rows":retained,
      "input_cell_to_output_cell_rows":inverse,"input_node_to_output_node_rows":q,
      "changed_output_cell_rows":newrows,"changed_input_cell_rows":affected,
      "affected_input_cell_rows":affected,"drop_volume_row":int(pr["drop_volume_row"]),
      "keep_volume_row":int(pr["keep_volume_row"]),
      "exact_signed_volume_delta_det6_integer":str(pr["exact_det6_delta"]),
      "replacement_tetrahedra":newt,"replacement_parent_offsets":rp_off,
      "replacement_parent_cell_rows":np.tile(affected,len(newt)).astype(np.int64)}
    return vres

def materialize_volume(parent_cache,pr,parent_ref,typed_surfaces,role,fixture=False):
    vres=build_volume_result(parent_cache,pr)
    out=core.endpoint_volume_arrays(parent_cache["arrays"],vres,parent_path=parent_ref["path"],parent_sha=parent_ref["sha256"])
    # Rebuild row-aligned cell data with explicit sentinel provenance on newly
    # generated cells; never inherit the final old row's metadata via index -1.
    nold=len(parent_cache["tets"]);ret=np.asarray(vres["retained_input_cell_rows"],np.int64)
    nnew=len(vres["changed_output_cell_rows"])
    special={"tetrahedra","material_indices","retained_input_cell_rows","output_cell_to_input_cell",
       "output_cell_to_input_cell_rows","input_cell_to_output_cell","input_cell_to_output_cell_rows",
       "replacement_tetrahedra","replacement_parent_offsets","replacement_parent_cell_rows"}
    sentinels={}
    for k,a0 in parent_cache["arrays"].items():
        a=np.asarray(a0)
        if k in special or a.ndim==0 or a.shape[0]!=nold:continue
        if a.dtype.kind in "iu":
            fill=-1 if a.dtype.kind=="i" else 0
        elif a.dtype.kind=="b":fill=False
        elif a.dtype.kind in "fc":fill=0
        elif a.dtype.kind in "US":fill=""
        else:raise ValueError("unknown cell-aligned metadata dtype for "+k)
        out[k]=np.concatenate((a[ret],np.full((nnew,)+a.shape[1:],fill,dtype=a.dtype)),axis=0)
        sentinels[k]={"dtype":str(a.dtype),"new_cell_fill":str(fill)}
    out["tetrahedra"]=vres["tetrahedra"];out["material_indices"]=vres["material_indices"]
    out["boundary_faces"]=vres["boundary_faces"];out["desired_boundary_faces"]=vres["desired_boundary_faces"]
    out["mapped_current_boundary_faces"]=vres["boundary_faces"]
    out["proposed_current_desired_boundary_faces"]=vres["desired_boundary_faces"]
    out["output_cell_to_input_cell"]=vres["output_cell_to_input_cell_rows"]
    out["output_cell_to_input_cell_rows"]=vres["output_cell_to_input_cell_rows"]
    out["retained_input_cell_rows"]=vres["retained_input_cell_rows"]
    out["input_cell_to_output_cell"]=vres["input_cell_to_output_cell_rows"]
    out["input_cell_to_output_cell_rows"]=vres["input_cell_to_output_cell_rows"]
    out["input_node_to_output_node_rows"]=vres["input_node_to_output_node_rows"]
    out["input_node_to_output_node"]=vres["input_node_to_output_node_rows"]
    out["output_node_to_input_node"]=np.arange(len(parent_cache["points"]),dtype=np.int64)
    out["output_node_to_input_node_rows"]=np.arange(len(parent_cache["points"]),dtype=np.int64)
    out["input_point_to_output_point_row"]=vres["input_node_to_output_node_rows"]
    out["input_point_to_output_point_rows"]=vres["input_node_to_output_node_rows"]
    out["output_point_to_input_point_row"]=np.arange(len(parent_cache["points"]),dtype=np.int64)
    out["output_point_to_input_point_rows"]=np.arange(len(parent_cache["points"]),dtype=np.int64)
    out["endpoint_quotient_input_node_map"]=vres["input_node_to_output_node_rows"]
    out["endpoint_quotient_changed_output_cell_rows"]=vres["changed_output_cell_rows"]
    out["endpoint_quotient_changed_input_cell_rows"]=vres["changed_input_cell_rows"]
    out["endpoint_quotient_affected_input_cell_rows"]=vres["affected_input_cell_rows"]
    out["replacement_tetrahedra"]=vres["replacement_tetrahedra"]
    out["replacement_parent_offsets"]=vres["replacement_parent_offsets"]
    out["replacement_parent_cell_rows"]=vres["replacement_parent_cell_rows"]
    if fixture:
        rebind={"synthetic_fixture":"source aliases not constructed"}
    else:
        prefix="skin" if role=="Skin" else "fat"
        wallpts=np.asarray(out[prefix+"_current_target_rebind_wall_face_points_world_m_f32"],np.float32)
        wallrows=np.asarray(out[prefix+"_current_target_rebind_wall_faces_volume_rows"],np.int64)
        wallids=np.arange(len(wallpts),dtype=np.int64)+(0 if role=="Skin" else len(wallpts))
        out,rebind=rebinder.rebind_source_aliases(role,out,typed_surfaces,wallpts,wallrows,wallids)
    return out,rebind,sentinels,vres

def update_text(arrays,key,value):
    if key not in arrays:return
    old=np.asarray(arrays[key]);dt=f"<U{max(1,len(str(value)))}"
    arrays[key]=np.asarray(str(value),dtype=dt) if old.ndim==0 else np.asarray([str(value)],dtype=dt)

def independent_replay(state_parent,source_parent,source_child,volume_parents,volume_children,source_result,prs,parent_refs,source_child_ref,volume_child_refs,source_role="I",volume_roles=("Skin","Fat"),fixture=False):
    # Recompute the source quotient and verify exact immediate typed ancestry.
    check=None
    verify_mod=imp("endpoint_quotient_batch_006_for_replay",
      HERE/"batch_inner_endpoint_quotient_006.py","595ee0252bf80451f9024ea7ab54e758e14182c02a67beb6626400b60f183c2d")
    verify_mod.core=core
    sr=core.collapse_surface(source_parent["points_world_m_f32"],source_parent["faces"],
       source_result["edge_rows"],drop=source_result["drop"],keep=source_result["keep"],
       fixed_endpoint_rows=source_result["fixed"],max_edge_m=source_result["max_edge_m"])
    ss=verify_mod.verify_surface_step(source_parent,source_child,sr,parent_refs[source_role],source_result["operation_id"])
    role_results={};deltas=[]
    for vr in volume_roles:
        before=volume_parents[vr];after=volume_children[vr];pr=prs[vr]
        bcache=B.volume_cache(parent_refs[vr]);t=np.asarray(bcache["tets"],np.int64);mat=np.asarray(bcache["mat"])
        rows=np.asarray(pr["cell_rows"],np.int64)
        sm=exact_map(source_parent,bcache,"independent I/"+vr)
        dv=int(sm[int(source_result["drop"])]);kv=int(sm[int(source_result["keep"])])
        if (dv,kv)!=(int(pr["drop_volume_row"]),int(pr["keep_volume_row"])):
            raise ValueError(vr+" volume endpoint rows do not replay from exact I point map")
        update=build_target_update(bcache,source_parent,source_parent,sr,sm,
                                  int(source_result["drop"]),dv,kv)
        seed=np.unique(np.r_[B.incident_cells(bcache,np.asarray([dv],np.int64)),
                             B.incident_cells(bcache,np.asarray([kv],np.int64))]).astype(np.int64)
        material=np.unique(np.asarray(bcache["mat"],np.int64)[seed])
        if len(material)!=1:raise ValueError(vr+" independent cavity seed crosses material")
        replay_rows,_,_,replay_solver,replay_selection=select_endpoint_cavity(
            bcache,seed,update["old_target"],int(material[0]),dv,kv,max_cells=MAX_CELLS,max_points=MAX_POINTS)
        if not np.array_equal(replay_rows,rows):raise ValueError(vr+" independent cavity selection differs from exact parent policy")
        recorded=pr.get("cavity_selection",{})
        if recorded.get("policy")!=replay_selection["policy"] or recorded.get("mode")!=replay_selection["mode"]:
            raise ValueError(vr+" cavity selection policy/mode does not replay")
        if not np.array_equal(np.asarray(pr.get("solver_vertices",[]),np.int64),replay_solver):
            raise ValueError(vr+" solver point domain does not replay after drop removal")
        qnode=np.arange(len(bcache["points"]),dtype=np.int64);qnode[dv]=kv
        mapped_target=qnode[np.asarray(bcache["target"],np.int64)]
        ts=np.sort(mapped_target,axis=1)
        valid=~np.any(ts[:,1:]==ts[:,:-1],axis=1)
        if not np.array_equal(np.asarray(after["desired_boundary_faces"],np.int64),mapped_target[valid]):
            raise ValueError(vr+" target boundary does not replay from exact parent node quotient")
        actual_ret=np.setdiff1d(np.arange(len(t),dtype=np.int64),rows,assume_unique=True)
        if not np.array_equal(after["tetrahedra"][:len(actual_ret)],t[actual_ret]):
            raise ValueError(vr+" retained cells differ from exact parent rows")
        if not np.array_equal(after["material_indices"][:len(actual_ret)],mat[actual_ret]):
            raise ValueError(vr+" retained material differs from exact parent rows")
        expected_origin=np.r_[actual_ret,np.full(len(after["tetrahedra"])-len(actual_ret),-1,np.int64)]
        if not np.array_equal(np.asarray(after["output_cell_to_input_cell_rows"],np.int64),expected_origin):
            raise ValueError(vr+" output-to-parent cell map does not preserve retained rows and mark new cells")
        expected_inverse=np.full(len(t),-1,np.int64);expected_inverse[actual_ret]=np.arange(len(actual_ret),dtype=np.int64)
        if not np.array_equal(np.asarray(after["input_cell_to_output_cell_rows"],np.int64),expected_inverse):
            raise ValueError(vr+" input-to-output cell map does not replay")
        newt=np.asarray(after["tetrahedra"][len(actual_ret):],np.int64)
        if not np.array_equal(newt,np.asarray(pr["new_tets"],np.int64)):
            raise ValueError(vr+" serialized cavity replacement differs from journal")
        if len(after["tetrahedra"])!=len(actual_ret)+len(newt):raise ValueError(vr+" output cell domain mismatch")
        if not np.array_equal(np.asarray(after["replacement_parent_cell_rows"],np.int64),
                              np.tile(rows,len(newt))):
            raise ValueError(vr+" replacement parent union does not include full old cavity")
        if not np.array_equal(np.asarray(after["replacement_parent_offsets"],np.int64),
                              np.asarray(pr["replacement_parent_offsets"],np.int64)):
            raise ValueError(vr+" replacement parent CSR offsets differ")
        if not np.array_equal(np.asarray(after["desired_boundary_faces"],np.int64),
                              np.asarray(pr["target"],np.int64)):
            raise ValueError(vr+" output target differs from directly quotiented parent target")
        actual=np.asarray(T.m.boundary(np.asarray(after["tetrahedra"],np.int64)),np.int64)
        if chain(actual)!=chain(after["desired_boundary_faces"]):raise ValueError(vr+" independent whole target boundary replay failed")
        exact=exact_positive(after["points_world_m"],newt)
        if not exact or min(exact)<=0:raise ValueError(vr+" exact changed F32 sign replay failed")
        oldt=np.asarray(t[rows],np.int64)
        oldexact=exact_positive(bcache["points"],oldt)
        if not oldexact or min(oldexact)<=0:raise ValueError(vr+" independent old cavity exact sign replay failed")
        oldsum=int(sum(oldexact));newsum=int(sum(exact))
        replay_update=dict(update);replay_update["drop_volume_row"]=dv
        oldb,newb_global,newb_local,_,_=cavity_boundary(bcache,rows,replay_update,replay_solver)
        if not np.array_equal(np.asarray(pr["new_boundary_local"],np.int64),newb_local):
            raise ValueError(vr+" replacement boundary differs from independent cavity chain")
        oldflux=int(ENG.exact_boundary_flux(bcache["points"],oldb))
        newflux=int(ENG.exact_boundary_flux(bcache["points"],newb_global))
        globalflux=int(ENG.exact_boundary_flux(bcache["points"],bcache["target"]))
        if oldflux!=oldsum:raise ValueError(vr+" old cavity exact cell sum differs from oriented flux")
        if newflux<=0 or globalflux<=0 or newsum!=newflux:
            raise ValueError(vr+" replacement exact cell sum differs from positive oriented flux")
        delta=int(newsum-oldsum)
        local_fraction=abs(delta)/float(oldsum);global_fraction=abs(delta)/float(globalflux)
        if local_fraction>MAX_LOCAL_VOL_FRAC or global_fraction>MAX_GLOBAL_VOL_FRAC:
            raise ValueError(vr+" independently recomputed exact volume delta exceeds cap")
        if oldsum!=int(pr["exact_old_det6_sum"]) or newsum!=int(pr["exact_new_det6_sum"]) or delta!=int(pr["exact_det6_delta"]):
            raise ValueError(vr+" journal exact sums/delta differ from array recomputation")
        det,q,good=native(after["points_world_m"],after["tetrahedra"])
        before_det,before_q,before_good=native(before["points_world_m"],before["tetrahedra"])
        if np.any(det<=0) or np.any(~np.isfinite(det)):raise ValueError(vr+" full mesh has nonpositive measured cell")
        oldbad=int(np.count_nonzero(~before_good));newbad=int(np.count_nonzero(~good))
        if newbad>oldbad:raise ValueError(vr+" full native bad count regressed")
        if not np.array_equal(after["points_world_m"].tobytes(),before["points_world_m"].tobytes()):
            raise ValueError(vr+" point bytes changed")
        if int(pr["exact_det6_delta"])!=int(pr["exact_new_det6_sum"])-int(pr["exact_old_det6_sum"]):
            raise ValueError(vr+" journal exact cavity delta invalid")
        role_results[vr]={"input_cells":len(t),"output_cells":len(after["tetrahedra"]),
            "cavity_old_cells":len(rows),"replacement_cells":len(newt),"old_bad":oldbad,"new_bad":newbad,
            "all_exact_positive_changed_cells":True,"complete_current_target_boundary":True,
            "full_parent_union_rows_per_new_cell":len(rows),
            "volume_alias_rebind":"NOT_APPLICABLE_SYNTHETIC" if fixture else "PASS"}
        role_results[vr].update({"independent_exact_old_det6_sum":str(oldsum),
            "independent_exact_new_det6_sum":str(newsum),"independent_exact_det6_delta":str(delta),
            "independent_old_cavity_flux_det6":str(oldflux),"independent_new_cavity_flux_det6":str(newflux),
            "independent_global_target_flux_det6":str(globalflux),
            "independent_local_delta_fraction":float(local_fraction),"independent_global_delta_fraction":float(global_fraction),
            "independent_volume_caps_pass":True,"cavity_selection_mode":replay_selection["mode"]})
        deltas.append(delta)
    if source_role=="I" and sum(deltas)!=0:raise ValueError("independent paired Skin/Fat exact det6 deltas do not cancel")
    return {"status":"PASS_INDEPENDENT_ENDPOINT_CAVITY_REPLAY","source_role":source_role,"source_surface":ss,"volume_roles":role_results,
        "paired_exact_det6_delta_sum":str(sum(deltas)) if source_role=="I" else "NOT_APPLICABLE_SINGLE_FAT_ROLE","all_point_bytes_unchanged":True,
        "native_bad_counts_no_role_regression":True,"physical_admission":False,"native_steps":0}

def load_context(state_path,protection_dir,fixture=False,source_role="I"):
    state_path=Path(state_path).resolve()
    key=(str(state_path),bool(fixture),str(source_role))
    if key in _CONTEXT_CACHE:return _CONTEXT_CACHE[key]
    root,st=P._source_state(state_path)
    if fixture:
        surfaces={r:load_npz(st["source_surfaces"][r]["path"]) for r in ("O","I","D")}
        caches={vr:B.volume_cache(st[vr+"_current"]) for vr in ("Skin","Fat")}
        protections={vr:{"surface_rows":np.empty(0,np.int64),"volume_rows":np.empty(0,np.int64),
            "report":{"synthetic_fixture_protection":True}} for vr in ("Skin","Fat")}
    else:
        surfaces={r:load_npz(B.check_ref(st["source_surfaces"][r],r+" typed source")) for r in ("O","I","D")}
        caches={vr:B.volume_cache(st[vr+"_current"]) for vr in ("Skin","Fat")}
        protections=None
    ctx={"root":root,"st":st,"surfaces":surfaces,"caches":caches,"protections":protections}
    _CONTEXT_CACHE[key]=ctx
    return ctx

def combined_quality_objective(role_pr,source_role):
    """Admit paired-I progress by total native debt, without role regression.

    Count reduction is primary. If total bad-cell count ties, the summed
    log-deficit must improve beyond the existing scale-aware tolerance.
    Fat-only D operations use the same rule on their single affected role.
    """
    if source_role not in ("I","D"):raise ValueError("unsupported source role for quality objective")
    if source_role=="I" and set(role_pr)!={"Skin","Fat"}:raise ValueError("paired-I objective requires both Skin and Fat")
    if source_role=="D" and set(role_pr)!={"Fat"}:raise ValueError("D objective requires Fat only")
    for role,metrics in role_pr.items():
        if int(metrics["native_bad_after"])>int(metrics["native_bad_before"]):
            raise ValueError(role+" native bad-cell count regresses")
    oldbad=sum(int(x["native_bad_before"]) for x in role_pr.values())
    newbad=sum(int(x["native_bad_after"]) for x in role_pr.values())
    olddef=sum(float(x["log_deficit_before"]) for x in role_pr.values())
    newdef=sum(float(x["log_deficit_after"]) for x in role_pr.values())
    tol=sum(max(1e-8,1e-6*float(x["log_deficit_before"])) for x in role_pr.values())
    if newbad>oldbad:raise ValueError("combined native bad-cell count regresses")
    if newbad<oldbad:
        progress_kind="combined_bad_count"
    elif newdef<olddef-tol:
        progress_kind="combined_log_deficit_tie"
    else:
        raise ValueError("candidate has no strict combined count or deficit progress")
    return {"native_bad_before":oldbad,"native_bad_after":newbad,
        "log_deficit_before":olddef,"log_deficit_after":newdef,
        "progress_kind":progress_kind,"deficit_tolerance":tol,
        "role_bad_counts":{r:{"before":int(x["native_bad_before"]),"after":int(x["native_bad_after"])} for r,x in role_pr.items()}}

def combined_candidate_rank(result,idx):
    q=result["quality_objective"]
    return (int(q["native_bad_after"]),float(q["log_deficit_after"]),
        float(result["source_edge_length_m"]),int(idx))

def run_one(state_path,edge,drop,keep,outdir,fixture=False,solve_seconds=3.,source_role="I"):
    if source_role not in ("I","D"):raise ValueError("only paired I and Fat-only D operations are supported")
    out=Path(outdir).resolve()
    if out.exists():raise FileExistsError("output path must be fresh")
    out.mkdir(parents=True)
    ctx=load_context(state_path,out,fixture=fixture,source_role=source_role)
    root,st,surfaces,caches=ctx["root"],ctx["st"],ctx["surfaces"],ctx["caches"]
    volume_roles=("Skin","Fat") if source_role=="I" else ("Fat",)
    protections=ctx["protections"]
    if not fixture and protections is None:
        protections=protection_owner.protected_scope(state_path,st,source_role,caches,out,fixture=False)
        ctx["protections"]=protections
    source=surfaces[source_role];source_ref=st["source_surfaces"][source_role]
    if sha(source_ref["path"])!=source_ref["sha256"]:raise ValueError("typed I source pin mismatch")
    fixed=np.unique(np.r_[np.asarray(protections["Fat"]["report"].get("sources",{}).get(source_role,{}).get("rim_rows",[]),np.int64),
                          np.asarray(protections["Fat"]["report"].get("sources",{}).get(source_role,{}).get("route_rows",[]),np.int64)]).astype(np.int64)
    if fixture:fixed=np.empty(0,np.int64)
    src_res=core.collapse_surface(source["points_world_m_f32"],source["faces"],edge,
        drop=drop,keep=keep,fixed_endpoint_rows=fixed,max_edge_m=MAX_EDGE_M)
    changed_source_faces=np.asarray(src_res["faces"],np.int64)[np.asarray(src_res["changed_output_rows"],np.int64)]
    if len(changed_source_faces) and np.any(ENG.J.score(np.asarray(source["points_world_m_f32"],np.float32),changed_source_faces)<1e-5):
        raise ValueError("post-quotient "+source_role+" source faces fail the unchanged joint-native apex gate")
    inner_allowed=P.allow_inner_back_only(source_role,protections,surfaces,caches) if not fixture else None
    maps={vr:exact_map(source,caches[vr],source_role+"/"+vr) for vr in volume_roles}
    role_pr={}
    for vr in volume_roles:
        role_pr[vr]=solve_role(vr,caches[vr],source,src_res,maps[vr],drop,keep,protections,inner_allowed,solve_seconds,source_role=source_role)
    if source_role=="I" and sum(int(x["exact_det6_delta"]) for x in role_pr.values())!=0:
        raise ValueError("paired I exact Skin/Fat cavity det6 deltas do not cancel")
    objective=combined_quality_objective(role_pr,source_role)
    oldbad=int(objective["native_bad_before"]);newbad=int(objective["native_bad_after"])
    olddef=float(objective["log_deficit_before"]);newdef=float(objective["log_deficit_after"])
    if getattr(run_one,"_evaluation_only",False):
        return {"status":"PASS_LOCAL_CANDIDATE_EVALUATION","source_role":source_role,"edge":list(map(int,edge)),
            "drop":int(drop),"keep":int(keep),"quality_objective":{**objective,"bad_reduction":int(oldbad-newbad)},
            "role_metrics":{vr:{k:v for k,v in pr.items() if k in ("native_bad_before","native_bad_after",
                "log_deficit_before","log_deficit_after","exact_det6_delta","ring_diagnostics","cavity_selection","solver")}
                for vr,pr in role_pr.items()},"source_edge_length_m":float(src_res["edge_length_m"]),
            "all_role_gates_passed":True,"geometry_written":False}
    opid=int(np.max(np.asarray(source.get("source_collapse_operation_rows",[-1]),np.int64)))+1
    sdict=core.endpoint_surface_arrays(source,src_res,parent_path=source_ref["path"],parent_sha=source_ref["sha256"],operation_id=opid)
    # Preserve only explicit source parent aliases until the final immutable file pin exists.
    spath=out/("candidate-"+source_role+"-typed-current.npz");np.savez_compressed(spath,**sdict);sref=ref(spath)
    np.savez_compressed(spath,**sdict);sref=ref(spath)
    child_source=load_npz(spath)
    new_surfaces=dict(surfaces);new_surfaces[source_role]=child_source
    volume_refs={};volume_children={};rebinds={};sentinels={}
    for vr in volume_roles:
        c=caches[vr]
        outv,rebind,sentinel,vres=materialize_volume(c,role_pr[vr],st[vr+"_current"],new_surfaces,vr,fixture=fixture)
        vp=out/(vr.lower()+"-cavity-remesh-volume-proposal.npz");np.savez_compressed(vp,**outv);vref=ref(vp)
        volume_children[vr]=load_npz(vp);volume_refs[vr]=vref;rebinds[vr]=rebind;sentinels[vr]=sentinel
    # Re-run target alias reconstruction from serialized arrays, not in-memory candidates.
    if not fixture:
        for vr in volume_roles:
            v=volume_children[vr];prefix="skin" if vr=="Skin" else "fat"
            wallpts=np.asarray(v[prefix+"_current_target_rebind_wall_face_points_world_m_f32"],np.float32)
            wallrows=np.asarray(v[prefix+"_current_target_rebind_wall_faces_volume_rows"],np.int64)
            wallids=np.arange(len(wallpts),dtype=np.int64)+(0 if vr=="Skin" else len(wallpts))
            recomputed,_=rebinder.rebind_source_aliases(vr,v,new_surfaces,wallpts,wallrows,wallids)
            for k in ("desired_boundary_source_group","desired_boundary_source_rows","desired_boundary_source_face_rows",
                      "proposed_current_desired_boundary_source_group","proposed_current_desired_boundary_source_rows",
                      "proposed_current_desired_boundary_source_face_rows"):
                if not np.array_equal(recomputed[k],v[k]):raise ValueError(vr+" serialized target alias does not reconstruct: "+k)
    source_result={"edge_rows":list(map(int,edge)),"drop":int(drop),"keep":int(keep),"fixed":fixed,
        "max_edge_m":MAX_EDGE_M,"operation_id":opid}
    parentrefs={source_role:dict(source_ref),**{vr:dict(st[vr+"_current"]) for vr in volume_roles}}
    replay=independent_replay(st,source,child_source,{vr:caches[vr]["arrays"] for vr in volume_roles},
        volume_children,source_result,role_pr,parentrefs,sref,volume_refs,source_role=source_role,volume_roles=volume_roles,fixture=fixture)
    outroot=copy.deepcopy(root);outst=outroot.get("current_verified_state_20261031",outroot)
    outst["source_state"]=ref(state_path);outst["source_surfaces"]=dict(st["source_surfaces"]);outst["source_surfaces"][source_role]=sref
    for vr in volume_roles:
        old=dict(st[vr+"_current"]);old.update(volume_refs[vr]);old.update({"cells":len(volume_children[vr]["tetrahedra"]),
            "nodes":len(volume_children[vr]["points_world_m"]),"native_quality_failures":int(role_pr[vr]["native_bad_before"]-role_pr[vr]["native_bad_before"]+np.count_nonzero(~native(volume_children[vr]["points_world_m"],volume_children[vr]["tetrahedra"])[2])),
            "nonpositive":0,"oriented_volume_target_boundary_equal":True,
            "boundary_source_binding_status":"PASS_LOCAL_CURRENT_SOURCE_ALIAS_REBIND_PENDING_EXTERNAL_CONTEXT"})
        outst[vr+"_current"]=old
    for k in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay","independent_target_replay",
              "source_evidence","source_audit","native_qualification","mass_mapping_receipt"):
        outst.pop(k,None)
    outst.update({"physical_admission":False,"native_steps":0,"mass_mapping":"PENDING","actual_697_217":"PENDING",
        "all185_775":"PENDING_FINAL_SOURCE_CONTEXT_AUDIT","construction_status":"POSTQUOTIENT_CAVITY_REMESH_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION",
        "postquotient_source_cavity_remesh":{"producer":pin(Path(__file__)),"role":source_role,"edge_rows":list(map(int,edge)),
          "drop_source_row":int(drop),"keep_source_row":int(keep),"source_coordinates_fixed":True,"affected_volume_roles":list(volume_roles),
          "accepted_ring_policy":"prefer full same-material face ring; on existing cap overflow use intact complete endpoint-star union only if it fits",
          "cavity_selection_policy":"prefer_full_ring_else_intact_endpoint_star_v1",
          "external_context":"PENDING","native_admission":False}})
    stateout=out/"candidate-state-proposal.json";write_json(stateout,outroot)
    journal={"schema":"h706.postquotient-source-connected-cavity-remesh.seed-star-fallback.journal.v1","source_role":source_role,"volume_roles":list(volume_roles),"status":"PASS_LOCAL_REPLAY_PENDING_EXTERNAL_CONTEXT",
        "parent_state":ref(state_path),"parent_source":ref(source_ref["path"]),"candidate_source":sref,"volume_candidates":volume_refs,
        "edge_rows":list(map(int,edge)),"drop_source_row":int(drop),"keep_source_row":int(keep),
        "source_operation_id":opid,"source_retired_parent_face_rows":np.asarray(src_res["retired_parent_face_rows"],np.int64).tolist(),
        "source_changed_output_rows":np.asarray(src_res["changed_output_rows"],np.int64).tolist(),
        "source_complete_operation_parent_face_rows":np.asarray(src_res["complete_operation_parent_face_rows"],np.int64).tolist(),
        "source_parent_union_complete":True,"source_parent_ancestry_complete":True,
        "paired_quality":{"bad_before":oldbad,"bad_after":newbad,"log_deficit_before":olddef,"log_deficit_after":newdef},
        "role_cavities":{vr:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in pr.items()
            if k in ("cell_rows","ring_added_rows","ring_diagnostics","cavity_selection","cavity_vertices","solver_vertices","new_tets","new_boundary_local",
               "old_target_rows","exact_old_det6_sum","exact_new_det6_sum","exact_det6_delta","native_bad_before","native_bad_after",
               "local_progress","log_deficit_before","log_deficit_after","old_q_range","new_q_range","exact_new_min_det6","drop_volume_row","keep_volume_row",
               "replacement_parent_offsets")}
            for vr,pr in role_pr.items()},
        "row_aligned_new_cell_metadata_sentinels":sentinels,"independent_replay":replay,
        "physical_admission":False,"native_steps":0,"mass_mapping":"PENDING"}
    write_json(out/"operation-journal.json",journal)
    report={"schema":"h706.postquotient-source-connected-cavity-remesh.seed-star-fallback.v1","source_role":source_role,"volume_roles":list(volume_roles),"status":"PASS_LOCAL_EXACT_CAVITY_REPLAY_PENDING_EXTERNAL_775_805_ALL185",
        "input_state":ref(state_path),"producer":pin(Path(__file__)),"protection_owner":pin(PROTECTOR),"core":pin(CORE),"ring_owner":pin(BATCH),"cavity_selection_policy":"prefer_full_ring_else_intact_endpoint_star_v1",
        "source_candidate":sref,"volume_candidates":volume_refs,"candidate_state":ref(stateout),
        "journal":ref(out/"operation-journal.json"),"independent_replay":replay,
        "source_edge":{"role":source_role,"rows":list(map(int,edge)),"drop":int(drop),"keep":int(keep),"length_m":float(src_res["edge_length_m"])},
        "quality_objective":dict(objective),
        "role_metrics":{vr:{k:v for k,v in pr.items() if k not in ("old_tets","old_materials","cavity_vertices","solver_vertices","old_boundary","new_boundary_local","new_tets","target","target_update")}
            for vr,pr in role_pr.items()},
        "all_point_bytes_unchanged":True,"paired_exact_det6_delta_sum":str(sum(int(x["exact_det6_delta"]) for x in role_pr.values())) if source_role=="I" else "NOT_APPLICABLE_SINGLE_FAT_ROLE",
        "full_current_OID_walls_all185":"PENDING_EXTERNAL_AUDIT","native_steps":0,"physical_admission":False,"mass_mapping":"PENDING"}
    write_json(out/"report.json",report)
    return report

def batch_directions(state_path,source_role,directions,outdir,solve_seconds=3.,fixture=False):
    """Evaluate each independent direction against one cached current state, then write only the best."""
    rootout=Path(outdir).resolve()
    if rootout.exists():raise FileExistsError("batch output path must be fresh")
    rootout.mkdir(parents=True)
    if not directions:raise ValueError("direction roster is empty")
    rows=[];best=None
    for idx,item in enumerate(directions):
        edge=tuple(map(int,item.get("edge",[])));drop=int(item["drop"]);keep=int(item["keep"])
        if len(edge)!=2 or set(edge)!={drop,keep}:raise ValueError(f"direction {idx} edge/drop/keep mismatch")
        attempt_dir=rootout/"attempts"/f"attempt-{idx:02d}"
        attempt_dir.parent.mkdir(parents=True,exist_ok=True)
        try:
            run_one._evaluation_only=True
            result=run_one(state_path,edge,drop,keep,attempt_dir,fixture=fixture,solve_seconds=solve_seconds,source_role=source_role)
            run_one._evaluation_only=False
            # Save the complete witness/result record; no candidate mesh is materialized yet.
            rows.append({"attempt":idx,"direction":dict(item),"result":result})
            attempt_dir.rmdir()
            rank=combined_candidate_rank(result,idx)
            if best is None or rank<best["rank"]:best={"rank":rank,"index":idx,"direction":dict(item),"result":result}
        except Exception as exc:
            run_one._evaluation_only=False
            fail={"status":"REJECTED","attempt":idx,"direction":dict(item),
                "error_type":type(exc).__name__,"error":str(exc)}
            rows.append(fail)
            attempt_dir.mkdir(parents=True,exist_ok=True)
            write_json(attempt_dir/"failure.json",fail)
    if best is None:
        summary={"schema":"h706.role-directed-cavity-direction-batch.v1","status":"NO_PASSING_DIRECTION",
            "input_state":ref(state_path),"source_role":source_role,"attempt_count":len(rows),"attempts":rows,
            "geometry_written":False,"physical_admission":False,"mass_mapping":"PENDING"}
        write_json(rootout/"batch-report.json",summary)
        return summary
    selected_dir=rootout/"selected"
    out=run_one(state_path,tuple(best["direction"]["edge"]),int(best["direction"]["drop"]),int(best["direction"]["keep"]),
        selected_dir,fixture=fixture,solve_seconds=solve_seconds,source_role=source_role)
    summary={"schema":"h706.role-directed-cavity-direction-batch.v1","status":"PASS_SELECTED_DIRECTION_PENDING_EXTERNAL_QUALIFICATION",
        "input_state":ref(state_path),"source_role":source_role,"attempt_count":len(rows),"attempts":rows,
        "selected_attempt":best["index"],"selected_direction":best["direction"],"selected_rank":list(best["rank"]),
        "selected_report":ref(selected_dir/"report.json"),"selected_state":ref(selected_dir/"candidate-state-proposal.json"),
        "selected_source":out["source_candidate"],"selected_volumes":out["volume_candidates"],
        "selected_quality_objective":out["quality_objective"],"independent_replay":out["independent_replay"],
        "all_point_bytes_unchanged":True,"native_steps":0,"physical_admission":False,"mass_mapping":"PENDING"}
    write_json(rootout/"batch-report.json",summary)
    return summary

def objective_policy_fixture():
    def m(before,after,dbefore,dafter):
        return {"native_bad_before":before,"native_bad_after":after,
                "log_deficit_before":dbefore,"log_deficit_after":dafter}
    # A Skin-only count gain is admissible when paired Fat does not regress.
    skin_only=combined_quality_objective({"Skin":m(4,3,4.,3.5),"Fat":m(3,3,3.,3.)},"I")
    if skin_only["progress_kind"]!="combined_bad_count" or skin_only["native_bad_after"]!=6:
        raise AssertionError("paired-I objective must admit Skin-only combined count progress")
    # With tied total count, only a real combined deficit reduction passes.
    deficit_tie=combined_quality_objective({"Skin":m(4,4,4.,3.8),"Fat":m(3,3,3.,3.)},"I")
    if deficit_tie["progress_kind"]!="combined_log_deficit_tie":
        raise AssertionError("paired-I objective must admit an honest deficit-only tie break")
    # Role regression and no-progress controls fail closed.
    try:combined_quality_objective({"Skin":m(4,3,4.,3.),"Fat":m(3,4,3.,2.)},"I")
    except ValueError:pass
    else:raise AssertionError("paired-I objective admitted per-role bad-count regression")
    try:combined_quality_objective({"Skin":m(4,4,4.,4.),"Fat":m(3,3,3.,3.)},"I")
    except ValueError:pass
    else:raise AssertionError("paired-I objective admitted no progress")
    # A lower total count outranks any deficit improvement at a tied/worse count.
    a={"quality_objective":{"native_bad_after":6,"log_deficit_after":99.},"source_edge_length_m":1.}
    b={"quality_objective":{"native_bad_after":7,"log_deficit_after":0.},"source_edge_length_m":1.}
    if not combined_candidate_rank(a,0)<combined_candidate_rank(b,1):
        raise AssertionError("batch ranking must prioritize total bad-cell count")
    c={"quality_objective":{"native_bad_after":6,"log_deficit_after":2.},"source_edge_length_m":1.}
    d={"quality_objective":{"native_bad_after":6,"log_deficit_after":3.},"source_edge_length_m":1.}
    if not combined_candidate_rank(c,0)<combined_candidate_rank(d,1):
        raise AssertionError("batch ranking must use combined deficit after count")
    return {"status":"PASS","skin_only_gain":"PASS","deficit_tie":"PASS",
        "per_role_regression_rejected":True,"no_progress_rejected":True,
        "rank_count_before_deficit":"PASS"}

def synthetic_fixture(outdir):
    objective_controls=objective_policy_fixture()
    cavity_controls=cavity_selection_fixture_control()
    out=Path(outdir).resolve()
    if out.exists():raise FileExistsError("fixture output must be fresh")
    inp=out/"inputs";inp.mkdir(parents=True)
    # A square disk with a short interior edge and two sliver triangles.
    eps=1e-7
    xy=np.asarray([[0,0],[1,0],[1,1],[0,1],[.5,.5-eps],[.5,.5+eps]],np.float32)
    pts2=np.c_[xy,np.zeros(len(xy),np.float32)]
    raw=np.asarray([[4,5,1],[5,4,3],[4,0,3],[4,1,0],[5,2,1],[5,3,2]],np.int64)
    faces=[]
    for f in raw:
        xyz=pts2[f].astype(np.float64);z=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])[2]
        faces.append(f if z>0 else f[[0,2,1]])
    faces=np.asarray(faces,np.int64)
    surf=ENG.fixture_surface(pts2,faces)
    surface_path=inp/"I.npz";np.savez_compressed(surface_path,**surf);sr=ref(surface_path)
    def make_volume(name,az):
        points=np.vstack((pts2,np.asarray([[.5,.5,az]],np.float32))).astype(np.float32)
        rawt=np.column_stack((np.repeat(6,len(faces)),faces))
        tets=M.orient(points,rawt)
        det,q=M.measurements(points,tets)
        if np.any(det<=0):raise AssertionError("fixture input exact signs must be positive")
        bf=np.asarray(T.m.boundary(tets),np.int64)
        arrays={"points_world_m":points,"points_world_m_f32":points.copy(),"tetrahedra":tets,
          "material_indices":np.full(len(tets),16,np.int32),"boundary_faces":bf,"desired_boundary_faces":bf.copy(),
          "retained_input_cell_rows":np.arange(len(tets),dtype=np.int64),
          "output_cell_to_input_cell":np.arange(len(tets),dtype=np.int64),
          "output_cell_to_input_cell_rows":np.arange(len(tets),dtype=np.int64),
          "input_cell_to_output_cell":np.arange(len(tets),dtype=np.int64),
          "input_cell_to_output_cell_rows":np.arange(len(tets),dtype=np.int64),
          "input_node_to_output_node":np.arange(len(points),dtype=np.int64),
          "output_node_to_input_node":np.arange(len(points),dtype=np.int64),
          "desired_boundary_source_group":np.zeros(len(bf),np.int8),
          "desired_boundary_source_rows":np.arange(len(bf),dtype=np.int64),
          "desired_boundary_source_face_rows":np.arange(len(bf),dtype=np.int64),
          "proposed_current_desired_boundary_faces":bf.copy()}
        path=inp/(name+".npz");np.savez_compressed(path,**arrays)
        return {"path":str(path),"sha256":sha(path),"cells":len(tets),"native_quality_failures":int(np.count_nonzero(~M.passes(det,q)))}
    skin=make_volume("skin",1.0);fat=make_volume("fat",-1.0)
    state={"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","status":"SYNTHETIC_POSTQUOTIENT_CAVITY_FIXTURE",
       "source_surfaces":{"O":sr,"I":sr,"D":sr},"Skin_current":skin,"Fat_current":fat}
    statep=inp/"state.json";write_json(statep,state)
    report=run_one(statep,(4,5),4,5,out/"run-I",fixture=True,solve_seconds=3.,source_role="I")
    dreport=run_one(statep,(4,5),4,5,out/"run-D",fixture=True,solve_seconds=3.,source_role="D")
    if report["status"]!="PASS_LOCAL_EXACT_CAVITY_REPLAY_PENDING_EXTERNAL_775_805_ALL185":
        raise AssertionError("synthetic endpoint cavity integration did not pass")
    if report["quality_objective"]["native_bad_after"]>=report["quality_objective"]["native_bad_before"]:
        raise AssertionError("fixture must show strict paired native bad-count reduction")
    if dreport["quality_objective"]["native_bad_after"]>=dreport["quality_objective"]["native_bad_before"]:
        raise AssertionError("fixture must show strict Fat-only D native bad-count reduction")
    six=[{"edge":[4,5],"drop":4,"keep":5,"direction_id":str(i)} for i in range(6)]
    batch=batch_directions(statep,"D",six,out/"batch-D",solve_seconds=3.,fixture=True)
    if batch["status"]!="PASS_SELECTED_DIRECTION_PENDING_EXTERNAL_QUALIFICATION" or batch["attempt_count"]!=6:
        raise AssertionError("six-direction cached batch fixture did not select a passing D operation")
    write_json(out/"fixture-receipt.json",{"schema":"h706.postquotient-source-cavity-remesh-fixture.v2","status":"PASS",
       "objective_policy_controls":objective_controls,"cavity_selection_controls":cavity_controls,
       "producer":pin(Path(__file__)),"input_state":ref(statep),
       "run_I_report":ref(out/"run-I"/"report.json"),"run_D_report":ref(out/"run-D"/"report.json"),
       "operation_journal_I":ref(out/"run-I"/"operation-journal.json"),"operation_journal_D":ref(out/"run-D"/"operation-journal.json"),
       "source_candidate_I":report["source_candidate"],"source_candidate_D":dreport["source_candidate"],
       "volume_candidates_I":report["volume_candidates"],"volume_candidates_D":dreport["volume_candidates"],
       "independent_replay_I":report["independent_replay"],"independent_replay_D":dreport["independent_replay"],
       "I_native_bad_before":report["quality_objective"]["native_bad_before"],"I_native_bad_after":report["quality_objective"]["native_bad_after"],
       "D_native_bad_before":dreport["quality_objective"]["native_bad_before"],"D_native_bad_after":dreport["quality_objective"]["native_bad_after"],
       "six_direction_cached_batch":ref(out/"batch-D"/"batch-report.json"),"selected_batch_attempt":batch["selected_attempt"],
       "paired_exact_det6_delta_sum":report["paired_exact_det6_delta_sum"],
       "all_point_bytes_unchanged":True,"external_all185":"NOT_RUN","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"})
    return {"status":"PASS","receipt":str(out/"fixture-receipt.json"),"report":report}

def main():
    ap=argparse.ArgumentParser(description="Role-directed endpoint quotient plus one-ring cavity remesh, source-only")
    ap.add_argument("--state",type=Path);ap.add_argument("--role",choices=("I","D"),default="I");ap.add_argument("--edge",help="single source role row pair a,b")
    ap.add_argument("--directions-json",type=Path,help="list or {directions:[...]} of independent edge/drop/keep directions")
    ap.add_argument("--drop",type=int);ap.add_argument("--keep",type=int)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--solve-seconds",type=float,default=3.)
    ap.add_argument("--fixture",action="store_true")
    a=ap.parse_args()
    if a.fixture:
        result=synthetic_fixture(a.output)
    else:
        if a.state is None:ap.error("--state is required unless --fixture is set")
        if a.directions_json:
            if a.edge is not None or a.drop is not None or a.keep is not None:ap.error("use either --directions-json or single --edge/--drop/--keep")
            doc=json.loads(a.directions_json.read_text());directions=doc.get("directions",doc) if isinstance(doc,dict) else doc
            if not isinstance(directions,list):ap.error("direction JSON must be a list or object with directions list")
            out=batch_directions(a.state,a.role,directions,a.output,solve_seconds=a.solve_seconds)
            result=out
            print(json.dumps({k:v for k,v in out.items() if k not in ("attempts",)},indent=2))
        else:
            if a.edge is None or a.drop is None or a.keep is None:ap.error("single mode requires --edge --drop --keep")
            e=tuple(int(x) for x in a.edge.split(","))
            if len(e)!=2:ap.error("--edge expects two comma-separated source role rows")
            out=run_one(a.state,e,a.drop,a.keep,a.output,fixture=False,solve_seconds=a.solve_seconds,source_role=a.role)
            result=out
            print(json.dumps({"status":out["status"],"source_role":out["source_role"],"source":out["source_candidate"],"volumes":out["volume_candidates"],
              "quality_objective":out["quality_objective"],"elapsed_s":{r:out["role_metrics"][r].get("solver",{}).get("elapsed_s") for r in out["volume_roles"]}},indent=2))
    return result

if __name__=="__main__":main()

