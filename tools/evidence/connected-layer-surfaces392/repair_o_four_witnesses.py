#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

PBASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
RUN= PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/minimum-thickness-minkowski-328/attempt-006-anchored-local-topology/run-030-global-derived-patch-remesh"
ATT=RUN/"attempt-002"
CAND=ATT/"candidate-O-open-f32-typed.npz"
AUD=ATT/"exact-self-current-pose-185-audit-001"
PREF=AUD/"connected-patch-preflight-001.json"
CW=PBASE/"canonical-O005-lineage-352/O005-canonical-face-lineage.npz"
REMESH=PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-043-abstract-cycle-coupled-remesh-fixed/remesh.py"
BOUNDARY=PBASE/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/remesh_d_witness_stars_rerun022.py"
VALIDATOR=PBASE/"derived-lineage-producer-260/export_authoring_registered_surface_sidecars_230.py"
NEW_ID=120000102
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path))
    if s is None or s.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def edge_map(F):
    em=collections.defaultdict(list)
    for r,(a,b,c) in enumerate(F):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            em[tuple(sorted((u,v)))].append((int(r),int(u),int(v)))
    return em
def boundary_directed(F):
    em=edge_map(F)
    return collections.Counter(vals[0][1:] for vals in em.values() if len(vals)==1)
def manifold_winding(F):
    em=edge_map(F);bad=[];nonman=[]
    for e,vals in em.items():
        if len(vals)>2: nonman.append((e,len(vals)))
        elif len(vals)==2 and vals[0][1:]!=(vals[1][2],vals[1][1]):
            bad.append((e,vals[0][0],vals[1][0]))
    return em,bad,nonman
def components(F):
    em=edge_map(F);adj=[set() for _ in range(len(F))]
    for vals in em.values():
        if len(vals)==2:
            a,b=vals[0][0],vals[1][0];adj[a].add(b);adj[b].add(a)
    todo=set(range(len(F)));n=0
    while todo:
        n+=1;seed=todo.pop();stack=[seed]
        while stack:
            r=stack.pop()
            take=adj[r]&todo
            todo-=take;stack.extend(take)
    return n
def cp_triangle(p,a,b,c):
    ab=b-a; ac=c-a; ap=p-a
    d1=float(ab@ap); d2=float(ac@ap)
    if d1<=0 and d2<=0:return a
    bp=p-b; d3=float(ab@bp); d4=float(ac@bp)
    if d3>=0 and d4<=d3:return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        v=d1/(d1-d3);return a+v*ab
    cp=p-c; d5=float(ab@cp); d6=float(ac@cp)
    if d6>=0 and d5<=d6:return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        w=d2/(d2-d6);return a+w*ac
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
        w=(d4-d3)/((d4-d3)+(d5-d6));return b+w*(c-b)
    den=1.0/(va+vb+vc);v=vb*den;w=vc*den
    return a+ab*v+ac*w
def directed_boundary_check(oldbd,newfaces):
    counts=collections.defaultdict(list)
    for a,b,c in newfaces:
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            counts[tuple(sorted((u,v)))].append((int(u),int(v)))
    bnew=collections.Counter(vals[0] for vals in counts.values() if len(vals)==1)
    if any(len(vals)>2 for vals in counts.values()):raise ValueError("replacement disk has nonmanifold internal edge")
    for vals in counts.values():
        if len(vals)==2 and vals[0]!=(vals[1][1],vals[1][0]):raise ValueError("replacement disk internal winding mismatch")
    if collections.Counter(oldbd)!=bnew:raise ValueError("replacement disk changes its directed boundary")
def load_exact():
    rem=module("remesh_d043",REMESH)
    return rem,rem.load_exact_owner()
REMESH_MOD=None
def build(seed_rows, P, F, pred, max_nodes=25000, max_seconds=10.0):
    boundary=module("boundary_cycles_d043",BOUNDARY)
    loops,bd=boundary.boundary_cycles(F,sorted(map(int,seed_rows)))
    if len(loops)!=1:raise ValueError(f"selected region has {len(loops)} boundary cycles")
    loop=np.asarray(loops[0],np.int64)
    if len(loop)<3:raise ValueError("boundary cycle too short")
    remove=set(map(int,seed_rows))
    keep=np.ones(len(F),bool); keep[list(remove)]=False
    baseF=F[keep]
    P32=np.asarray(P,np.float32)
    records,_=pred.float32_surface_records(P32,baseF,source_seam_groups=())
    base_index=pred._prepare_surface_aabb(records)
    ext=set(edge_map(baseF))
    lattice=tuple(pred.float32_point_lattice_key(p) for p in P32)
    tri,_,stats=REMESH_MOD.collision_aware_ear_clip(loop,P,P32,ext,base_index,pred,lattice,[],len(baseF)+100,max_nodes=max_nodes,max_seconds=max_seconds)
    tri=np.asarray(tri,np.int64)
    directed_boundary_check(bd,tri)
    return tri, loop.tolist(), stats, keep
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",type=Path,default=CAND)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    global REMESH_MOD
    REMESH_MOD,pred=load_exact()
    if a.self_test:
        P=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0],[2,0,1],[3,0,1],[2,1,1],[3,1,1]],np.float32)
        F=np.asarray([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[5,6,7],[6,8,7]],np.int64)
        tri,loop,stats,keep=build([0,1,2,3],P,F,pred,max_nodes=1000,max_seconds=5)
        assert tri.shape==(2,3) and sorted(loop)==[0,1,2,3] and len(keep)==6
        print(json.dumps({"self_test":"PASS","triangles":tri.tolist(),"stats":stats}))
        return
    outdir=a.output
    if outdir.exists():raise FileExistsError(f"refusing to overwrite {outdir}")
    outdir.mkdir(parents=True)
    t0=time.monotonic()
    if sha(a.candidate)!="831125d720b47f640de8bf455e8210b05f3d8f3142b7400346013237e333b58b":
        raise ValueError("candidate SHA mismatch; stop without repair")
    d0=np.load(a.candidate,allow_pickle=False);d={k:np.asarray(d0[k]).copy() for k in d0.files};d0.close()
    c0=np.load(CW,allow_pickle=False);cw={k:np.asarray(c0[k]).copy() for k in c0.files};c0.close()
    if sha(CW)!="3d6191d1dc5053086ae954288bb042bfaf80bbfd0abbf5b6ecb2a5fcef34f885":raise ValueError("canonical crosswalk SHA mismatch")
    pre=json.loads(PREF.read_text())
    if sha(AUD/"self.json")!="2b9df6d94203657a42bfde9df64d82eeb89ebfd5b253a73debc56e6599d604d8":raise ValueError("self witness report SHA mismatch")
    seed=np.asarray(pre["selected_face_rows"],np.int64)
    if len(seed)!=5 or pre["candidate_sha256"]!=sha(a.candidate):raise ValueError("patch seed preflight mismatch")
    P=np.asarray(d["points_world_m"],np.float32)
    F=np.asarray(d["faces"],np.int64)
    nfaces=len(F)
    for k in ("face_source_original_id","face_source_face_id","face_source_layer","face_source_body","face_source_bone","face_source_identity0","face_source_identity1"):
        if d[k].shape!=(nfaces,):raise ValueError(f"{k} face-row shape mismatch")
    sys.path.insert(0,str(VALIDATOR.parent))
    co=module("o_producer260",VALIDATOR)
    old_schema=json.loads(str(d["face_source_identity_schema_by_original_id_json"][0]))
    if str(NEW_ID) in old_schema:raise ValueError("new derived source ID already exists")
    if not np.array_equal(cw["source_cache_sha256"],d["source_cache_sha256"]):raise ValueError("candidate and canonical crosswalk source cache differ")
    off=np.asarray(d["face_ancestry_offsets"],np.int64)
    parent=np.asarray(d["face_ancestry_parent_face_rows"],np.int64)
    srcfid=np.asarray(d["face_ancestry_source_face_rows"],np.int64)
    srcorig=np.asarray(d["face_ancestry_source_original_ids"],np.int64)
    if off.shape!=(nfaces+1,) or off[0]!=0 or off[-1]!=len(parent) or len(srcfid)!=len(parent) or len(srcorig)!=len(parent):raise ValueError("input ancestry CSR shapes invalid")
    if np.any(parent<0) or np.any(parent>=len(cw["source_cache_rows"])):raise ValueError("parent O005 row outside crosswalk")
    # Confirm the existing typed ancestry matches the pinned canonical crosswalk on seed rows.
    seed_parent=set()
    for r in seed:
        s=int(off[r]);e=int(off[r+1]);pr=np.asarray(parent[s:e],np.int64);seed_parent.update(map(int,pr))
        expfid=np.asarray(cw["face_source_face_id"][pr],np.int64)
        exporig=np.asarray(cw["face_source_original_id"][pr],np.int64)
        if sorted(map(int,srcfid[s:e]))!=sorted(map(int,expfid)) or sorted(map(int,srcorig[s:e]))!=sorted(map(int,exporig)):
            raise ValueError(f"input source ancestry differs from canonical crosswalk at face {r}")
    seed_parent=np.asarray(sorted(seed_parent),np.int64)
    if len(seed_parent)!=9:raise ValueError(f"expected nine distinct O005 parents for patch, got {len(seed_parent)}")
    # Try the measured 5-face disk first; only expand one intact ring if no exact triangulation exists.
    try:
        tri,loop,stats,keep=build(seed,P,F,pred,max_nodes=25000,max_seconds=10.0)
        patch_rows=seed.copy();depth=0
    except Exception as e0:
        em=edge_map(F);patch=set(map(int,seed))
        for r in seed:
            a0,b0,c0=map(int,F[int(r)])
            for u0,v0 in ((a0,b0),(b0,c0),(c0,a0)):
                patch.update(int(x[0]) for x in em[tuple(sorted((u0,v0)))])
        patch_rows=np.asarray(sorted(patch),np.int64)
        tri,loop,stats,keep=build(patch_rows,P,F,pred,max_nodes=25000,max_seconds=10.0)
        depth=1
        (outdir/"ring0-failure.json").write_text(json.dumps({"error":repr(e0),"seed_rows":seed.tolist()},indent=2)+"\n")
        p0=set()
        for r in patch_rows:
            s=int(off[int(r)]);e=int(off[int(r)+1]);p0.update(map(int,parent[s:e]))
        seed_parent=np.asarray(sorted(p0),np.int64)
    # Candidate has fixed coordinates. Validate local disk and whole mesh incidence/winding.
    keepF=F[keep]
    Fnew=np.vstack((keepF,tri))
    if Fnew.shape[0] != nfaces-len(patch_rows)+len(tri):raise ValueError("output face count accounting mismatch")
    if not np.array_equal(boundary_directed(F),boundary_directed(Fnew)):raise ValueError("global open boundary changed")
    _,bad_winding,nonman=manifold_winding(Fnew)
    if bad_winding or nonman:raise ValueError(f"global topology/winding failed: {len(bad_winding)} winding, {len(nonman)} nonmanifold")
    if components(Fnew)!=components(F):raise ValueError("component count changed")
    tri32=P[Fnew]
    cr=np.cross(tri32[:,1].astype(np.float64)-tri32[:,0].astype(np.float64),tri32[:,2].astype(np.float64)-tri32[:,0].astype(np.float64))
    if np.any(np.linalg.norm(cr,axis=1)==0):raise ValueError("global F32 zero-area faces remain")
    # Sample bidirectional patch deviation using face vertices, edge midpoints, and centroids.
    oldpatch=F[patch_rows]; newpatch=tri
    def samples(faces):
        xyz=P[faces].astype(np.float64)
        return np.vstack((xyz.reshape(-1,3),(xyz[:,0]+xyz[:,1])/2,(xyz[:,1]+xyz[:,2])/2,(xyz[:,2]+xyz[:,0])/2,xyz.mean(axis=1)))
    def directed_deviation(samples_, target_faces):
        target=P[target_faces].astype(np.float64);vals=[]
        for q in samples_:
            vals.append(min(float(np.linalg.norm(q-cp_triangle(q,*t))) for t in target))
        return max(vals,default=0.0)
    dev=max(directed_deviation(samples(oldpatch),newpatch),directed_deviation(samples(newpatch),oldpatch))
    if dev>10e-6:raise ValueError(f"sampled patch deviation exceeds 10 um: {dev}")
    # Geometry checkpoint written before sidecar/schema packaging.
    geom=outdir/"geometry-only-candidate.npz"
    np.savez_compressed(geom,points_world_m=P,faces=Fnew,removed_face_rows=patch_rows,new_face_rows=np.arange(len(keepF),len(Fnew),dtype=np.int64),new_faces=tri,boundary_vertex_rows=np.asarray(loop,np.int64))
    # Full ancestry for each replacement face is the union of all canonical parents in the repaired disk.
    union=set()
    for r in patch_rows:
        s=int(off[int(r)]);e=int(off[int(r)+1]);union.update(map(int,parent[s:e]))
    union=np.asarray(sorted(union),np.int64)
    par_ids=np.asarray(cw["face_source_original_id"][union],np.int64)
    par_fids=np.asarray(cw["face_source_face_id"][union],np.int64)
    par_i0=np.asarray(cw["face_source_identity0"][union],np.int64)
    par_i1=np.asarray(cw["face_source_identity1"][union],np.int64)
    par_layer=np.asarray(cw["face_source_layer"][union],np.int64)
    par_body=np.asarray(cw["face_source_body"][union],np.int64)
    par_bone=np.asarray(cw["face_source_bone"][union],np.int64)
    # New typed source view; each face has explicit local identity plus full O005 parent CSR.
    sf=len(tri);ids=np.arange(sf,dtype=np.int64)
    srcpath=outdir/f"current-derived-source-{NEW_ID}.npz"
    so=np.arange(0,(sf+1)*len(union),len(union),dtype=np.int64);srows=np.tile(union,sf)
    source_arrays=dict(points_world_m_f32=P,faces=tri,face_id=ids,face_source_identity0=ids.copy(),face_source_identity1=ids.copy(),
        face_source_original_id=np.full(sf,NEW_ID,np.int64),face_source_layer=np.zeros(sf,np.int64),face_source_body=np.full(sf,-1,np.int64),face_source_bone=np.full(sf,-1,np.int64),
        parent_O005_offsets=so,parent_O005_rows=srows,parent_face_source_original_id=np.tile(par_ids,sf),parent_face_source_face_id=np.tile(par_fids,sf),
        parent_face_source_identity0=np.tile(par_i0,sf),parent_face_source_identity1=np.tile(par_i1,sf),parent_face_source_layer=np.tile(par_layer,sf),parent_face_source_body=np.tile(par_body,sf),parent_face_source_bone=np.tile(par_bone,sf),
        parent_O005_source_cache_sha256=np.asarray([str(cw["source_cache_sha256"][0])]),source_operation=np.asarray(["exact_aware_connected_patch_retriangulation_of_four_inherited_transverse_pairs"]))
    np.savez_compressed(srcpath,**source_arrays)
    srcsha=sha(srcpath)
    schema=old_schema
    parent_source_ids=sorted(set(map(int,par_ids)))
    schema[str(NEW_ID)]={
        "derived_operation":"exact_aware_connected_patch_retriangulation",
        "derived_parent_assets":[{"path":str(a.candidate),"sha256":sha(a.candidate)},{"path":str(CW),"sha256":sha(CW)}],
        "derived_parent_source_ids":parent_source_ids,"face_count":int(sf),"identity_face_id_key":"face_id",
        "identity_keys":["face_source_identity0","face_source_identity1"],"identity_original_id_key":"face_source_original_id",
        "identity_original_id":NEW_ID,"path":str(srcpath),"role":"O","sha256":srcsha,
        "source_geometry_scope":"single witness connected patch, fixed seven-vertex boundary",
        "temporary_construction_only":False
    }
    # Rebuild all face-row arrays and full O005 parent CSR. Retained ancestry is copied exactly.
    orig=np.asarray(d["face_source_original_id"],np.int64)[keep]
    fid=np.asarray(d["face_source_face_id"],np.int64)[keep]
    layer=np.asarray(d["face_source_layer"],np.int64)[keep]
    body=np.asarray(d["face_source_body"],np.int64)[keep]
    bone=np.asarray(d["face_source_bone"],np.int64)[keep]
    id0=np.asarray(d["face_source_identity0"],np.int64)[keep]
    id1=np.asarray(d["face_source_identity1"],np.int64)[keep]
    orig=np.r_[orig,np.full(sf,NEW_ID,np.int64)];fid=np.r_[fid,ids]
    layer=np.r_[layer,np.zeros(sf,np.int64)];body=np.r_[body,np.full(sf,-1,np.int64)];bone=np.r_[bone,np.full(sf,-1,np.int64)]
    id0=np.r_[id0,ids];id1=np.r_[id1,ids]
    new_parent_faces=np.tile(union,sf)
    new_src_fids=np.tile(par_fids,sf);new_src_orig=np.tile(par_ids,sf)
    new_offsets=np.r_[off[np.flatnonzero(keep)],off[-1]]
    # Build ancestry output explicitly, retaining each surviving row's exact prior CSR.
    ao=[0];a_parent=[];a_fid=[];a_orig=[]
    for r in np.flatnonzero(keep):
        s=int(off[r]);e=int(off[r+1]);a_parent.extend(parent[s:e]);a_fid.extend(srcfid[s:e]);a_orig.extend(srcorig[s:e]);ao.append(len(a_parent))
    for _ in range(sf):
        a_parent.extend(union.tolist());a_fid.extend(par_fids.tolist());a_orig.extend(par_ids.tolist());ao.append(len(a_parent))
    outdata={k:v for k,v in d.items()}
    outdata.update(points_world_m=P,faces=Fnew,face_source_original_id=orig,face_source_face_id=fid,face_source_layer=layer,face_source_body=body,face_source_bone=bone,face_source_identity0=id0,face_source_identity1=id1,
        face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True,separators=(",",":"))]),
        face_ancestry_offsets=np.asarray(ao,np.int64),face_ancestry_parent_face_rows=np.asarray(a_parent,np.int64),
        face_ancestry_source_face_rows=np.asarray(a_fid,np.int64),face_ancestry_source_original_ids=np.asarray(a_orig,np.int64),
        closed_face_rows=np.r_[np.asarray(d["closed_face_rows"],np.int64)[keep],np.full(sf,-1,np.int64)])
    candpath=outdir/"candidate-O-open-f32-typed.npz"
    np.savez_compressed(candpath,**outdata)
    val=co.validate_candidate(candpath,"O",require_f32=True)
    report={"schema":"o-connected-transverse-witness-retriangulation.v1","input_candidate":str(a.candidate),"input_sha256":sha(a.candidate),"source_crosswalk":str(CW),"source_crosswalk_sha256":sha(CW),"seed_faces":seed.tolist(),"repaired_patch_faces":patch_rows.tolist(),"repair_ring_depth":depth,"fixed_boundary_vertex_rows":loop,"replacement_faces":tri.tolist(),"exact_search":stats,"sampled_bidirectional_patch_deviation_m":dev,"points_bitwise_unchanged":True,"global_boundary_bitwise_unchanged":True,"global_nonmanifold_edges":0,"global_winding_failures":0,"global_F32_zero_area_faces":0,"component_count":components(Fnew),"source_parent_O005_rows":union.tolist(),"candidate":str(candpath),"candidate_sha256":sha(candpath),"derived_source":str(srcpath),"derived_source_sha256":srcsha,"producer260":{"status":"PASS","face_count":val["faceCount"],"vertex_count":val["vertexCount"],"source_original_face_ids":val["sourceOriginalFaceIDs"]},"full_exact_self":"NOT_RUN","native_185_banks":"NOT_RUN","physical_admission":False,"elapsed_s":time.monotonic()-t0}
    (outdir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="repaired_patch_faces"},indent=2),flush=True)
if __name__=="__main__":main()
