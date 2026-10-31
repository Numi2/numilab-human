#!/usr/bin/env python3
"""Connected O/I/D boundary 2-2 patch flips with exact local cavity MILP replay.

A connected set of source-face diagonal flips is applied as one patch. The union
of every complete material-local volume edge star is retired and solved once with
the pinned exact-boundary 785 MILP. This writes source ancestry and volume parent
unions, but does not perform the final all185/current OID context audit.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import permutations
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOLVER_PATH = HERE / "solve_complete_vertex_star_milp_785.py"
STAR_DRIVER_PATH = HERE / "batch_complete_vertex_star_quality_milp_785_root.py"
STAR_DRIVER_SHA = ""
TYPED_PATH = BASE / "fixed-boundary-quality-topology-196" / "prepare_current_earclip_adapter_024.py"
TOPOLOGY_PATH = BASE / "remesh_complete_surface_cavities_218_root.py"
APEX_PATH = BASE / "audit_joint_native_apex_272_root.py"
SOLVER_SHA = "20eae93eff31493702eda00adfd80cd812e31f673af6ee8b469444f7602fc251"
PAT = np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)), np.int64)
TSC = ("face_source_layer","face_source_body","face_source_bone","face_source_face_id",
       "face_source_original_id","face_source_identity0","face_source_identity1")
TUN = ("quality_star_patch_union_face_source_layer","quality_star_patch_union_face_source_body",
       "quality_star_patch_union_face_source_bone","quality_star_patch_union_face_source_face_id",
       "quality_star_patch_union_face_source_original_id","quality_star_patch_union_face_source_identity0",
       "quality_star_patch_union_face_source_identity1")
ANC = ("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")

def imp(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import pinned owner {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

S=imp("connected_boundary_solver785_850",SOLVER_PATH)
if hashlib.sha256(SOLVER_PATH.read_bytes()).hexdigest()!=SOLVER_SHA:
    raise RuntimeError("pinned exact-boundary solver785 changed")
P785=imp("connected_boundary_star_driver785_850",STAR_DRIVER_PATH)
STAR_DRIVER_SHA=hashlib.sha256(STAR_DRIVER_PATH.read_bytes()).hexdigest()
M=S.M
T=imp("connected_boundary_topology218_850",TOPOLOGY_PATH)
A=imp("connected_boundary_typed_rows024_850",TYPED_PATH)
J=imp("connected_boundary_apex272_850",APEX_PATH)

def sha(path): return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()
def pin(path): return {"path":str(path),"sha256":sha(path)}
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def fkey(x):
    q=np.asarray(x,np.float32)
    if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError("point key must be finite F32 xyz")
    return q.view(np.uint32).tobytes()
def fsign(face):
    a,b,c=map(int,face); inv=int(a>b)+int(a>c)+int(b>c)
    return 1 if inv%2==0 else -1
def chain(faces):
    c=Counter()
    for f in np.asarray(faces,np.int64):c[tuple(sorted(map(int,f)))]+=fsign(f)
    return {k:v for k,v in c.items() if v}
def edgechain(faces):
    c=Counter()
    for a,b,d in np.asarray(faces,np.int64):
        for u,v in ((int(a),int(b)),(int(b),int(d)),(int(d),int(a))):
            c[tuple(sorted((u,v)))]+=1 if u<v else -1
    return {k:v for k,v in c.items() if v}
def cyc(f):
    a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def flip_faces(old):
    old=np.asarray(old,np.int64)
    if old.shape!=(2,3):raise ValueError("2-2 patch requires two source faces")
    common=sorted(set(map(int,old[0]))&set(map(int,old[1])))
    opp=sorted((set(map(int,old[0]))|set(map(int,old[1])))-set(common))
    if len(common)!=2 or len(opp)!=2:raise ValueError("source faces do not form one quad")
    target=edgechain(old);sol={}
    for x in permutations((opp[0],opp[1],common[0])):
        for y in permutations((opp[0],opp[1],common[1])):
            q=np.asarray((x,y),np.int64)
            if edgechain(q)==target:sol[tuple(sorted((cyc(q[0]),cyc(q[1]))))]=q
    if len(sol)!=1:raise ValueError(f"oriented flip has {len(sol)} distinct solutions")
    return next(iter(sol.values())),common,opp
def topology_sig(faces):
    f=np.asarray(faces,np.int64)
    if len(np.unique(np.sort(f,axis=1),axis=0))!=len(f):raise ValueError("duplicate source faces")
    counts=Counter();signs=Counter();graph=defaultdict(set)
    for a,b,c in f:
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            k=tuple(sorted((u,v)));counts[k]+=1;signs[k]+=1 if u<v else -1
            graph[u].add(v);graph[v].add(u)
    if any(v>2 for v in counts.values()):raise ValueError("nonmanifold source edge")
    if any(counts[k]==2 and signs[k]!=0 for k in counts):raise ValueError("source edge orientation mismatch")
    used=np.unique(f);seen=set();components=0
    for v0 in map(int,used):
        if v0 in seen:continue
        components+=1;seen.add(v0);stack=[v0]
        while stack:
            u=stack.pop()
            for v in graph[u]:
                if v not in seen:seen.add(v);stack.append(v)
    bchain=tuple(sorted((k,int(signs[k])) for k,n in counts.items() if n==1 and signs[k]))
    return {"faces":len(f),"vertices":len(used),"edges":len(counts),
            "euler":int(len(used)-len(counts)+len(f)),"components":components,
            "boundary_edges":sum(n==1 for n in counts.values()),"boundary_chain":bchain}
def exact_cell_sum(points,tets):
    cells=np.asarray(tets,np.int64);used=np.unique(cells.reshape(-1));local={int(v):i for i,v in enumerate(used)}
    ip=[S.S.ip(x) for x in np.asarray(points,np.float32)[used]]
    return sum(S.S.det(*(ip[local[int(v)]] for v in tet)) for tet in cells)
def exact_boundary_flux(points,faces):
    fs=np.asarray(faces,np.int64);used=np.unique(fs.reshape(-1));local={int(v):i for i,v in enumerate(used)}
    ip=[S.S.ip(x) for x in np.asarray(points,np.float32)[used]]
    return sum(S.S.triple(*(ip[local[int(v)]] for v in face)) for face in fs)
def faces_from_chain(c):
    out=[]
    for k,s in sorted(c.items()):
        if s not in (-1,1):raise ValueError("boundary chain has invalid face multiplicity")
        f=list(map(int,k))
        if fsign(f)!=s:f[1],f[2]=f[2],f[1]
        out.append(f)
    return np.asarray(out,np.int64).reshape((-1,3))
def oriented_boundary_replace(boundary, pairs):
    c=chain(boundary)
    for old,new in pairs:
        k=tuple(sorted(map(int,old)));sgn=fsign(old)
        if c.get(k,0)!=sgn:raise ValueError("retired source face is not a cavity boundary facet")
        del c[k]
        nk=tuple(sorted(map(int,new)));c[nk]=c.get(nk,0)+fsign(new)
        if c[nk]==0:del c[nk]
    return faces_from_chain(c)
def required_surface(surface):
    f=np.asarray(surface.get("faces"),np.int64);p=np.asarray(surface.get("points_world_m_f32"),np.float32)
    req={"points_world_m","points_world_m_f32","quality_star_point_output_to_input_row",
         "quality_star_face_output_to_input_row","quality_star_patch_union_offsets","face_ancestry_offsets",*TSC,*TUN,*ANC}
    miss=req-set(surface)
    if miss:raise ValueError("typed source arrays missing: "+str(sorted(miss)))
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or p.ndim!=2 or p.shape[1]!=3:raise ValueError("bad source geometry shape")
    if np.asarray(surface["points_world_m"],np.float64).shape!=p.shape or not np.array_equal(np.asarray(surface["points_world_m"],np.float64).astype(np.float32).view(np.uint32),p.view(np.uint32)):raise ValueError("source coordinate views disagree")
    n=len(f)
    if f.min()<0 or f.max()>=len(p):raise ValueError("source face point row out of range")
    if np.asarray(surface["quality_star_point_output_to_input_row"]).shape!=(len(p),) or np.asarray(surface["quality_star_face_output_to_input_row"]).shape!=(n,):raise ValueError("source maps malformed")
    if any(np.asarray(surface[k]).shape!=(n,) for k in TSC):raise ValueError("typed face scalar arrays malformed")
    u=np.asarray(surface["quality_star_patch_union_offsets"],np.int64);a=np.asarray(surface["face_ancestry_offsets"],np.int64)
    if u.shape!=(n+1,) or a.shape!=(n+1,) or u[0]!=0 or a[0]!=0 or np.any(np.diff(u)<=0) or np.any(np.diff(a)<=0):raise ValueError("source ancestry offsets malformed")
    if any(len(surface[k])!=u[-1] for k in TUN) or any(len(surface[k])!=a[-1] for k in ANC):raise ValueError("source ancestry CSR lanes malformed")
    return f,p
def update_surface(parent,faces,changed,union,ancestry,parent_path):
    out=dict(parent);n=len(faces);changed=set(map(int,changed))
    uf=np.asarray(parent["quality_star_face_output_to_input_row"],np.int64).copy()
    for r in changed:uf[r]=-1
    ur=[];ar=[]
    for r in range(n):
        if r in changed:ur.append(union);ar.append(ancestry)
        else:ur.append(A.rows_union(parent,r));ar.append(A.rows_anc(parent,r))
    if any(not x for x in ur) or any(not x for x in ar):raise ValueError("empty face ancestry")
    uo,ud=A.csr(ur,7);ao,ad=A.csr(ar,3);direct=np.asarray([x[0] for x in ur],np.int64)
    out["faces"]=np.asarray(faces,np.int64);out["quality_star_face_output_to_input_row"]=uf
    out["quality_star_patch_union_offsets"]=uo
    for j,k in enumerate(TUN):out[k]=ud[:,j]
    out["face_ancestry_offsets"]=ao
    for j,k in enumerate(ANC):out[k]=ad[:,j]
    for j,k in enumerate(TSC):out[k]=direct[:,j]
    for k in ("face_parent_source_face_rows","face_exact_source_row"):
        if k in out:
            a=np.asarray(out[k]).copy();a[list(changed)]=-1;out[k]=a
    out["source_current_candidate_path"]=np.asarray([str(parent_path)])
    out["source_current_candidate_sha256"]=np.asarray([sha(parent_path)])
    out["typed_parent_candidate_path"]=np.asarray([str(parent_path)])
    out["typed_parent_candidate_sha256"]=np.asarray([sha(parent_path)])
    out["connected_boundary_patch_ancestry_semantics"]=np.asarray(["changed patch faces carry the full typed/source union of all retired faces in this connected patch; no exact overlap claim"])
    if "face_ancestry_method" in out:
        out["face_ancestry_method"]=np.asarray(["retained rows preserve full parent ancestry; changed rows carry complete connected-patch ancestry union"])
    return out
def target_key(v):
    for k in ("desired_boundary_faces","proposed_current_desired_boundary_faces","proposed_current_desired_boundary_faces_rows","boundary_faces"):
        if k in v:return k
    raise KeyError("current volume target boundary is absent")
def face_lookup(faces):
    q=defaultdict(list)
    for i,f in enumerate(np.asarray(faces,np.int64)):q[tuple(sorted(map(int,f)))].append(i)
    return q
def rel(a,b):
    if tuple(sorted(map(int,a)))!=tuple(sorted(map(int,b))):raise ValueError("orientation comparison on different faces")
    return fsign(a)*fsign(b)
def connect_pairs(pairs,faces):
    rows=[int(x) for pair in pairs for x in pair]
    if len(set(rows))!=len(rows):return False
    sets=[set(map(int,faces[r])) for r in rows];seen={0};todo=[0]
    while todo:
        i=todo.pop()
        for j in range(len(sets)):
            if j not in seen and sets[i]&sets[j]:seen.add(j);todo.append(j)
    return len(seen)==len(sets)
def edge_star(tets,materials,a,b):
    t=np.asarray(tets);mask=np.any(t==int(a),axis=1)&np.any(t==int(b),axis=1);rows=np.flatnonzero(mask)
    if not len(rows):raise ValueError("empty full volume edge star")
    if len(np.unique(np.asarray(materials)[rows]))!=1:raise ValueError("edge star crosses a material interface")
    return rows
def fixture_surface(points,faces):
    n=len(faces)
    base=np.column_stack((np.full(n,2),np.full(n,1),np.full(n,16),np.arange(n),np.full(n,600001),np.arange(n),np.full(n,-1))).astype(np.int64)
    out={"points_world_m":np.asarray(points,np.float64),"points_world_m_f32":np.asarray(points,np.float32),"faces":np.asarray(faces,np.int64),
         "quality_star_point_output_to_input_row":np.arange(len(points),dtype=np.int64),"quality_star_face_output_to_input_row":np.arange(n,dtype=np.int64),
         "quality_star_patch_union_offsets":np.arange(n+1,dtype=np.int64),"face_ancestry_offsets":np.arange(n+1,dtype=np.int64),
         "face_ancestry_method":np.asarray(["synthetic source rows"])}
    for j,k in enumerate(TSC):out[k]=base[:,j]
    for j,k in enumerate(TUN):out[k]=base[:,j]
    for j,k in enumerate(ANC):out[k]=np.arange(n,dtype=np.int64) if j!=1 else base[:,4]
    return out

def run(args):
    tic=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    vp=Path(args.volume_npz);sp=Path(args.surface_npz);v=load(vp);s=load(sp);sf,spoints=required_surface(s)
    p=np.asarray(v["points_world_m"],np.float32);t=np.asarray(v["tetrahedra"],np.int64);mat=np.asarray(v["material_indices"])
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or t.ndim!=2 or t.shape[1]!=4 or mat.shape!=(len(t),):raise ValueError("volume arrays malformed")
    if "points_world_m_f32" in v and not np.array_equal(p.view(np.uint32),np.asarray(v["points_world_m_f32"],np.float32).view(np.uint32)):raise ValueError("volume coordinate views disagree")
    det,q=M.measurements(p,t)
    if np.any(det<=0):raise ValueError("input cells must be measured positive")
    actual=np.asarray(T.m.boundary(t),np.int64);tk=target_key(v);target=np.asarray(v[tk],np.int64)
    if chain(actual)!=chain(target):raise ValueError("input target differs from complete actual volume boundary")
    global_flux=exact_boundary_flux(p,target)
    if global_flux<=0:raise ValueError("global target signed volume must be positive")
    pmap={}
    for i,x in enumerate(p):
        k=fkey(x)
        if k in pmap:raise ValueError("duplicate exact volume F32 point")
        pmap[k]=i
    s2v=np.asarray([pmap.get(fkey(x),-1) for x in spoints],np.int64)
    if np.any(s2v<0) or len(np.unique(s2v))!=len(s2v):raise ValueError("source-to-volume point crosswalk is not exact bijection")
    sfv=s2v[sf]
    spec=json.loads(Path(args.patches).read_text())
    patches=spec.get("patches")
    if not isinstance(patches,list) or len(patches)!=1 or not patches[0].get("face_pairs"):raise ValueError("one nonempty connected patch is required per run")
    pairs=[tuple(map(int,x)) for x in patches[0]["face_pairs"]]
    if not connect_pairs(pairs,sf):raise ValueError("flip pairs do not form one connected source patch")
    oldface_rows=[r for pair in pairs for r in pair]
    if min(oldface_rows)<0 or max(oldface_rows)>=len(sf):raise ValueError("source face row outside input")
    protected_surface_points=set();protected_volume_points=set();protected_volume_cells=set()
    if args.protected_npz:
        z=load(args.protected_npz)
        protected_surface_points=set(map(int,z.get("protected_surface_point_rows",np.empty(0,np.int64))))
        protected_volume_points=set(map(int,z.get("protected_volume_point_rows",z.get("protected_point_rows",np.empty(0,np.int64)))))
        protected_volume_cells=set(map(int,z.get("protected_volume_cell_rows",z.get("protected_cell_rows",np.empty(0,np.int64)))))
        if any(protected_surface_points.intersection(map(int,sf[r])) or protected_volume_points.intersection(map(int,sfv[r])) for r in oldface_rows):
            raise ValueError("patch touches a protected route/back/rim source point")
    before_sig=topology_sig(sf);expected_sf=sf.copy();flip_data=[]
    for pair in pairs:
        new,edge,opp=flip_faces(sf[list(pair)])
        expected_sf[list(pair)]=new
        flip_data.append({"rows":list(pair),"source_old":sf[list(pair)].copy(),"source_new":new,"edge_surface":edge,"opposite_surface":opp})
    after_sig=topology_sig(expected_sf)
    if before_sig!=after_sig:raise ValueError("2-2 patch changes source topology or outer edge chain")
    target_lu=face_lookup(target)
    stars=[];patch_material=[]
    for op in flip_data:
        old_source_volume=sfv[list(op["rows"])]
        newv=s2v[op["source_new"]]
        rels=[];rows_target=[];oldtarget=[]
        for face in old_source_volume:
            key=tuple(sorted(map(int,face)));hits=target_lu.get(key,[])
            if len(hits)!=1:raise ValueError("source patch face has no unique current target row")
            tr=int(hits[0]);rows_target.append(tr);oldtarget.append(target[tr].copy());rels.append(rel(target[tr],face))
        if len(set(rels))!=1:raise ValueError("source and physical target orientation mismatch inside flip")
        # Retire actual target-boundary orientation; inner interfaces can reverse source winding.
        oldv=np.asarray(oldtarget,np.int64)
        if rels[0]<0:newv=newv[:,[0,2,1]]
        op.update({"volume_old":oldv,"volume_new":newv,"target_rows":rows_target,"target_old":oldv.copy(),
                   "orientation_relation":rels[0],"source_volume_old":old_source_volume.copy()})
        stars.append(edge_star(t,mat,int(s2v[op["edge_surface"][0]]),int(s2v[op["edge_surface"][1]])))
    cellrows=np.unique(np.concatenate(stars)).astype(np.int64)
    if len(cellrows)>args.maximum_old_cells:raise ValueError("complete connected edge-star cavity exceeds cell cap")
    old=t[cellrows];oldmat=mat[cellrows];verts=np.unique(old)
    touched_protected_points=sorted(protected_volume_points.intersection(map(int,verts)))
    touched_protected_cells=sorted(protected_volume_cells.intersection(map(int,cellrows)))
    if touched_protected_points or touched_protected_cells:
        raise ValueError("complete volume cavity touches protected vertices/cells: "+json.dumps({"points":touched_protected_points[:16],"cells":touched_protected_cells[:16]}))
    if len(verts)>args.maximum_points:raise ValueError("complete connected edge-star cavity exceeds point cap")
    if len(np.unique(oldmat))!=1:raise ValueError("connected patch crosses material boundary")
    local_map=np.full(len(p),-1,np.int64);local_map[verts]=np.arange(len(verts),dtype=np.int64)
    lp=p[verts];lo=local_map[old]
    ip=[S.S.ip(x) for x in lp]
    old_exact=[S.S.det(*(ip[int(x)] for x in tet)) for tet in lo]
    if not old_exact or min(old_exact)<=0:raise ValueError("old cavity is not exact-positive")
    old_boundary=np.asarray(T.m.boundary(old),np.int64)
    # Confirm actual oriented target facets are part of the retired cavity boundary.
    local_old_chain=chain(local_map[old_boundary])
    for op in flip_data:
        for old_face in op["volume_old"]:
            local_face=local_map[np.asarray(old_face,np.int64)]
            if local_old_chain.get(tuple(sorted(map(int,local_face))),0)!=fsign(local_face):
                raise ValueError("mapped target facet is not the exact oriented facet of the retired cavity")
    closed,open_edges=T.closed(old_boundary);topo=T.manifold_topology(old_boundary)
    if not closed or topo["bad_vertex_links"] or topo["boundary_components"]!=1:raise ValueError("complete connected edge-star cavity shell is not one closed manifold")
    target_replacements=[]
    for op in flip_data:
        for j in range(2):
            oldlocal=local_map[op["volume_old"][j]];newlocal=local_map[op["volume_new"][j]]
            if np.any(oldlocal<0) or np.any(newlocal<0):raise ValueError("source patch point absent from complete cavity")
            target_replacements.append((oldlocal,newlocal))
    new_boundary=oriented_boundary_replace(local_map[old_boundary],target_replacements)
    closed2,open2=T.closed(new_boundary);topo2=T.manifold_topology(new_boundary)
    if not closed2 or topo2["bad_vertex_links"] or topo2["boundary_components"]!=1:raise ValueError("flipped patch does not produce a closed manifold cavity boundary")
    solver_report,newlocal=S.solve(lp,new_boundary,seconds=args.solve_seconds,max_points=args.maximum_points,material=int(oldmat[0]))
    if newlocal is None or solver_report.get("status")!="PASS_PARTIAL_POSITIVE_CAVITY_MILP":raise ValueError("exact-boundary MILP failed: "+str(solver_report.get("status")))
    newglobal=verts[np.asarray(newlocal,np.int64)]
    newip=[S.S.ip(x) for x in p]
    newexact=[S.S.det(*(newip[int(x)] for x in tet)) for tet in newglobal]
    if not newexact or min(newexact)<=0:raise ValueError("new volume cavity has nonpositive exact F32 determinant")
    oldbad=int(np.count_nonzero(~np.asarray(M.passes(det[cellrows],q[cellrows]),bool)))
    olddef=float(np.sum(-np.log(np.maximum(np.minimum(np.minimum(q[cellrows]/1e-5,det[cellrows]/6e-18),1.),1e-300))))
    nd,nq=M.measurements(p,newglobal);ngood=np.asarray(M.passes(nd,nq),bool)
    newbad=int(np.count_nonzero(~ngood));newdef=float(np.sum(-np.log(np.maximum(np.minimum(np.minimum(nq/1e-5,nd/6e-18),1.),1e-300))))
    if newbad>oldbad or (newbad==oldbad and newdef>=olddef-max(1e-8,1e-6*olddef)):raise ValueError("patch has no robust local native-quality progress")
    oldvol=sum(old_exact);newvol=sum(newexact);delta=newvol-oldvol
    if oldvol<=0 or abs(delta)/oldvol>args.maximum_local_volume_fraction:raise ValueError("local exact signed-volume budget exceeded")
    if abs(delta)/global_flux>args.maximum_global_volume_fraction:raise ValueError("global exact signed-volume budget exceeded")
    if exact_boundary_flux(lp,new_boundary)!=newvol:raise ValueError("MILP tetrahedra do not exactly realize changed shell flux")
    for op in flip_data:
        scores=np.asarray(J.score(spoints,op["source_new"]),np.float64)
        if len(scores)!=2 or np.any(scores<1e-5):raise ValueError("new source triangle fails exact joint-apex threshold")
    # Full source triangles and target rows get the same orientation-preserving 2-2 flips.
    outsf=expected_sf
    outtarget=target.copy()
    for op in flip_data:
        for j,tr in enumerate(op["target_rows"]):outtarget[int(tr)]=op["volume_new"][j]
    removed=np.asarray(sorted(map(int,cellrows)),np.int64);keep=np.ones(len(t),bool);keep[removed]=False
    outt=np.vstack((t[keep],newglobal));outmat=np.r_[mat[keep],np.full(len(newglobal),int(oldmat[0]),mat.dtype)]
    outboundary=np.asarray(T.m.boundary(outt),np.int64)
    if chain(outboundary)!=chain(outtarget):raise ValueError("global output boundary differs from updated complete target")
    if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("global output contains duplicate tetrahedra")
    od,oq=M.measurements(p,outt)
    if np.any(od<=0):raise ValueError("global output has nonpositive measured tetrahedra")
    before_good=np.asarray(M.passes(det,q),bool);after_good=np.asarray(M.passes(od,oq),bool)
    after_bad=int(np.count_nonzero(~after_good));before_bad=int(np.count_nonzero(~before_good))
    before_score=float(np.sum(-np.log(np.maximum(np.minimum(np.minimum(q/1e-5,det/6e-18),1.),1e-300))))
    after_score=float(np.sum(-np.log(np.maximum(np.minimum(np.minimum(oq/1e-5,od/6e-18),1.),1e-300))))
    if after_bad>before_bad or (after_bad==before_bad and after_score>=before_score-max(1e-8,1e-6*before_score)):raise ValueError("whole-role native quality did not improve")
    union=sorted(set(tuple(map(int,x)) for r in oldface_rows for x in A.rows_union(s,r)))
    ancestry=sorted(set(tuple(map(int,x)) for r in oldface_rows for x in A.rows_anc(s,r)))
    if not union or not ancestry:raise ValueError("connected source patch lacks complete parent unions")
    sout=update_surface(s,outsf,oldface_rows,union,ancestry,sp)
    # Emit cell maps and complete immediate-parent CSR, one whole-patch union per replacement cell.
    roff=[0];rrows=[]
    for _ in newglobal:rrows.extend(map(int,cellrows));roff.append(len(rrows))
    inverse=np.full(len(t),-1,np.int64);inverse[keep]=np.arange(int(keep.sum()),dtype=np.int64)
    origin=np.r_[np.flatnonzero(keep),np.full(len(newglobal),-1,np.int64)]
    vpout=dict(v);vpout.update({"tetrahedra":outt,"material_indices":outmat,"boundary_faces":outboundary,
       tk:outtarget,"output_cell_to_input_cell":origin,"input_cell_to_output_cell":inverse,
       "retained_input_cell_rows":np.flatnonzero(keep),"replacement_tetrahedra":newglobal,
       "replacement_parent_offsets":np.asarray(roff,np.int64),"replacement_parent_cell_rows":np.asarray(rrows,np.int64),
       "connected_boundary_patch_source_role":np.asarray([args.role]),
       "connected_boundary_patch_operation_cell_rows":cellrows,
       "connected_boundary_patch_operation_face_rows":np.asarray(oldface_rows,np.int64),
       "connected_boundary_patch_operation_face_parent_offsets":np.asarray([0,len(union)],np.int64),
       "connected_boundary_patch_operation_face_parent_union":np.asarray(union,np.int64).reshape((-1,7)),
       "connected_boundary_patch_operation_face_ancestry":np.asarray(ancestry,np.int64).reshape((-1,3)),
       "connected_boundary_patch_input_volume_path":np.asarray([str(vp)]),
       "connected_boundary_patch_input_volume_sha256":np.asarray([sha(vp)]),
       "connected_boundary_patch_input_surface_path":np.asarray([str(sp)]),
       "connected_boundary_patch_input_surface_sha256":np.asarray([sha(sp)]),
       "connected_boundary_patch_regional_volume_fraction_caps":np.asarray([args.maximum_local_volume_fraction,args.maximum_global_volume_fraction],np.float64)})
    for key in ("desired_boundary_faces","proposed_current_desired_boundary_faces","proposed_current_desired_boundary_faces_rows"):
        if key in vpout:vpout[key]=outtarget
    vpout.update(P785.unchanged_point_lineage_updates(v,len(p)))
    P785.assert_unchanged_point_lineage(vpout,len(p))
    vo=out/f"{args.role.lower()}-connected-boundary-patch-volume-proposal.npz"
    so=out/f"candidate-{args.role}-connected-boundary-patch-typed.npz"
    np.savez_compressed(vo,**vpout);np.savez_compressed(so,**sout)
    # Readback exact gates.
    vr=load(vo);sr=load(so)
    if not np.array_equal(vr["points_world_m"].view(np.uint32),p.view(np.uint32)):raise ValueError("volume point bytes changed on readback")
    if chain(T.m.boundary(vr["tetrahedra"]))!=chain(vr[tk]):raise ValueError("volume readback target mismatch")
    if not np.array_equal(sr["faces"],outsf):raise ValueError("typed surface readback geometry mismatch")
    required_surface(sr)
    rd,rq=M.measurements(vr["points_world_m"],vr["tetrahedra"])
    if np.any(rd<=0) or int(np.count_nonzero(~np.asarray(M.passes(rd,rq),bool)))!=after_bad:raise ValueError("volume readback quality mismatch")
    report={"schema":"h706.connected-boundary-patch-milp-proposal.v1","status":"PASS_CONNECTED_BOUNDARY_PATCH_PROPOSAL",
       "role":args.role,"input_volume":pin(vp),"input_typed_surface":pin(sp),"patch_spec":pin(args.patches),
       "protected_mask":pin(args.protected_npz) if args.protected_npz else None,
       "protected_surface_points_checked":len(protected_surface_points),"protected_volume_points_checked":len(protected_volume_points),"protected_volume_cells_checked":len(protected_volume_cells),
       "volume_candidate":pin(vo),"typed_surface_candidate":pin(so),
       "owners":{"solver785":pin(SOLVER_PATH),"star_driver785":pin(STAR_DRIVER_PATH),"typed_rows024":pin(TYPED_PATH),"topology218":pin(TOPOLOGY_PATH),"joint_apex272":pin(APEX_PATH)},
       "face_pairs":pairs,"changed_source_face_rows":oldface_rows,"edge_star_cell_rows":cellrows.tolist(),
       "source_to_target_orientation_relations":[int(x["orientation_relation"]) for x in flip_data],
       "retired_target_facets_verified_on_oriented_cavity_boundary":True,
       "replacement_cells":len(newglobal),"input_cells":len(t),"output_cells":len(outt),
       "native_bad_before":before_bad,"native_bad_after":after_bad,"log_deficit_before":before_score,"log_deficit_after":after_score,
       "source_surface_topology_before":before_sig,"source_surface_topology_after":topology_sig(outsf),"source_topology_preserved":True,
       "complete_source_typed_union":True,"complete_source_ancestry":True,"derived_faces_exact_parent_rows_claimed":False,
       "exact_old_cavity_positive":True,"exact_new_cavity_positive":True,"exact_modified_boundary_flux_matches_new_tets":True,
       "global_volume_boundary_matches_updated_target":True,"global_exact_volume_det6_before":str(global_flux),
       "global_exact_volume_det6_after":str(exact_boundary_flux(p,outtarget)),"global_exact_signed_volume_delta_det6":str(exact_boundary_flux(p,outtarget)-global_flux),
       "local_exact_signed_volume_delta_det6":str(delta),"local_volume_fraction":float(abs(delta)/oldvol),
       "global_volume_fraction":float(abs(exact_boundary_flux(p,outtarget)-global_flux)/global_flux),
       "regional_volume_budget_caps":{"local":float(args.maximum_local_volume_fraction),"global":float(args.maximum_global_volume_fraction)},
       "all_volume_points_bit_identical":True,"all_source_points_bit_identical":True,"native_thresholds_unchanged":True,
       "external_current_OID_wall_all185_audit":"REQUIRED_SEPARATE_FINAL_GATE_NOT_RUN",
       "mass_mapping":"NOT_RUN","physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-tic}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)
    return report

def fixture(outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=False)
    p=np.asarray(((0,0,0),(1,0,0),(.2,.2,0),(.8,-.2,0),(.5,0,1e-7)),np.float32)
    t=M.orient(p,((0,1,2,4),(1,0,3,4)));faces=np.asarray(T.m.boundary(t),np.int64)
    pair=[i for i,f in enumerate(faces) if 0 in f and 1 in f]
    if len(pair)!=2:raise AssertionError("synthetic boundary diagonal does not have two incident faces")
    inputs=outdir/"inputs";inputs.mkdir()
    runout=outdir/"synthetic-run"
    vp=inputs/"volume.npz";sp=inputs/"surface.npz";ps=inputs/"patches.json"
    np.savez_compressed(vp,points_world_m=p,tetrahedra=t,material_indices=np.full(len(t),16,np.int32),
        boundary_faces=faces,desired_boundary_faces=faces,input_node_to_output_node=np.arange(len(p),dtype=np.int64),
        output_node_to_input_node=np.arange(len(p),dtype=np.int64),output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),
        source_I_vertex_quotient=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),
        output_point_parent_input_cell_rows=np.empty(0,np.int64))
    np.savez_compressed(sp,**fixture_surface(p,faces))
    ps.write_text(json.dumps({"patches":[{"face_pairs":[pair]}]}))
    args=argparse.Namespace(volume_npz=vp,surface_npz=sp,patches=ps,protected_npz=None,role="D",output=runout,
         maximum_old_cells=16,maximum_points=12,solve_seconds=3.,maximum_local_volume_fraction=.01,maximum_global_volume_fraction=1e-4)
    report=run(args)
    if report["native_bad_after"]>report["native_bad_before"] or report["log_deficit_after"]>=report["log_deficit_before"]:
        raise AssertionError("synthetic patch did not satisfy the native lexicographic gain gate")
    vout=load(runout/"d-connected-boundary-patch-volume-proposal.npz")
    sout=load(runout/"candidate-D-connected-boundary-patch-typed.npz")
    if not np.array_equal(vout["points_world_m"].view(np.uint32),p.view(np.uint32)):raise AssertionError("fixture moved a point")
    if not np.all(np.asarray(sout["quality_star_face_output_to_input_row"])[pair]==-1):raise AssertionError("derived source faces claim exact parent rows")
    u=sout["quality_star_patch_union_offsets"]
    if any(int(u[r+1]-u[r])!=2 for r in pair):raise AssertionError("derived faces do not carry both source face parent unions")
    if report["global_exact_signed_volume_delta_det6"]!="0":raise AssertionError("planar fixture should have exact zero global signed-volume change")
    receipt={"status":"PASS_CONNECTED_BOUNDARY_PATCH_MILP_FIXTURE","producer":pin(Path(__file__)),
      "fixture_report":pin(runout/"report.json"),"fixture_volume_candidate":pin(runout/"d-connected-boundary-patch-volume-proposal.npz"),
      "fixture_typed_surface_candidate":pin(runout/"candidate-D-connected-boundary-patch-typed.npz"),
      "fixture_inputs":{"volume":pin(vp),"typed_surface":pin(sp),"patches":pin(ps)},
      "native_thresholds_unchanged":True,"full_typed_parent_union":True,
      "exact_surface_topology_preserved":True,"exact_global_volume_delta_det6":"0","mass_mapping":"NOT_RUN",
      "physical_admission":False,"native_steps":0}
    report_path=outdir/"fixture-report.json"
    report_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume-npz",type=Path);ap.add_argument("--surface-npz",type=Path);ap.add_argument("--patches",type=Path)
    ap.add_argument("--protected-npz",type=Path);ap.add_argument("--role",choices=("O","I","D"),default="D")
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--solve-seconds",type=float,default=5.)
    ap.add_argument("--maximum-old-cells",type=int,default=24);ap.add_argument("--maximum-points",type=int,default=16)
    ap.add_argument("--maximum-local-volume-fraction",type=float,default=.01)
    ap.add_argument("--maximum-global-volume-fraction",type=float,default=1e-4)
    ap.add_argument("--fixture",action="store_true");a=ap.parse_args()
    if a.fixture:
        print(json.dumps(fixture(a.output),indent=2,sort_keys=True),flush=True);return
    if not a.volume_npz or not a.surface_npz or not a.patches:ap.error("--volume-npz, --surface-npz, and --patches required")
    if not 1<=a.maximum_old_cells<=128:ap.error("--maximum-old-cells must be 1..128")
    if not 4<=a.maximum_points<=24:ap.error("--maximum-points must be 4..24")
    if a.solve_seconds<=0 or a.maximum_local_volume_fraction<0 or a.maximum_global_volume_fraction<0:ap.error("time and budgets must be nonnegative")
    run(a)

if __name__=="__main__":main()
