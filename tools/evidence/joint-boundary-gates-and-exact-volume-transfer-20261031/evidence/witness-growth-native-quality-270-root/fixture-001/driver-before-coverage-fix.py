"""Cell-disjoint cached native-quality kernel fans on material cavities.
Reuses determinant-kernel LP, exact boundary/sign/volume/intersection owners.
Adds declared inferred interior construction points only; no native admission.
"""
from pathlib import Path
from itertools import product,combinations
from collections import Counter
import hashlib
import numpy as np,time,argparse,json,importlib.util,sys
H=Path(__file__).resolve().parent
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
prev=load("pair237for245",H/"pair_star_native_defect_descent_237_root.py");m=prev.m;top=prev.top
kernel=load("kernel227for245",H/"choose_bounded_kernel_positions_227_root.py")
OWNER=H.parent/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(OWNER.parent.parent));exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])
def fan(p,old):
 b=m.boundary(old);vv=np.unique(b);lp=p[vv];ix=np.full(len(p),-1,np.int64);ix[vv]=np.arange(len(vv));lb=ix[b]
 if not top.closed(lb)[0]:return None,"nonclosed"
 tp=top.manifold_topology(lb)
 if tp["bad_vertex_links"] or tp["boundary_components"]!=1:return None,"nonmanifold"
 root=lp.astype(float).mean(0).astype(np.float32);ri=len(lp);pp=np.vstack([lp,root]);tt=np.c_[np.full(len(lb),ri),lb]
 lower=lp.min(0).astype(float);upper=lp.max(0).astype(float)
 # No old synthetic cone is privileged; existing accepted cells are certified
 # separately. First request all-native cones, then the max-clearance kernel.
 centers=kernel.kernel_centers(pp,tt,ri,lower,upper,np.zeros(len(tt),bool))
 if not centers:return None,"no_positive_kernel"
 oldbad=int(sum(~m.m.passes(*m.m.measurements(p,old))));best=None;seen=set()
 for mode,planned,radius in centers:
  center=planned.astype(np.float32);axes=[[np.nextafter(center[j],np.float32(-np.inf)),center[j],np.nextafter(center[j],np.float32(np.inf))] for j in range(3)]
  for values in product(*axes):
   point=np.asarray(values,np.float32);key=m.key(point)
   if key in seen:continue
   seen.add(key);pp[ri]=point;d,q=m.m.measurements(pp,tt)
   if np.any(d<=0):continue
   bad=int(sum(~m.m.passes(d,q)))
   if bad>=oldbad:continue
   score=(bad,-float(q.min()),-float(d.min()))
   if best is None or score<best[0]:best=(score,point.copy(),mode,radius)
 if best is None:return None,"no_F32_native_defect_descent"
 score,point,mode,radius=best;pp[ri]=point
 # Shared exact owner covers boundary and new internal fan faces. This rejects
 # a merely algebraic positive chain on a self-intersecting shell.
 faces=tt[:,m.m.PAT].reshape(-1,3);_,idx=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
 records,_=exact.float32_surface_records(pp,faces[idx]);gate=exact._audit_pair(records,records,same_surface=True)
 if gate["count"]:return None,"exact_internal_face_crossing"
 combined=np.vstack([p,point]);nt=np.c_[np.full(len(b),len(p)),b]
 ok,why=m.m.certify(combined,old,nt)
 if not ok:return None,"exact_"+why
 return (point,nt,{"old_native_failures":oldbad,"new_native_failures":score[0],"new_cells":len(nt),"kernel_mode":mode,"kernel_radius_m":radius,"exact_F32_positive":True,"exact_boundary_and_volume":True,"exact_internal_face_crossings":0}),None
def fixture(out):
 out.mkdir(exist_ok=False,parents=True)
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[.003,.003,1e-9]],np.float32);b=m.boundary(np.array([[0,1,2,3]],np.int64));old=np.c_[np.full(4,4),b]
 ans,why=fan(p,old);assert ans is not None,why
 point,nt,r=ans;assert r["new_native_failures"]==0 and r["old_native_failures"]==1
 r.update(status="PASS_KERNEL_FAN_ACTUAL_PATH_CONTROLS",physical_admission=False,native_steps=0,controls=["outside_quality_star_replaced","all_native_F32_fan","same_exact_oriented_boundary_and_volume","exact_internal_intersection_gate"])
 (out/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
def signature(p,t,material):
 vv=np.unique(t);st=np.sort(t,axis=1);st=st[np.lexsort(st[:,::-1].T)]
 hh=hashlib.sha256()
 hh.update(p[vv].astype("<f4",copy=False).tobytes());hh.update(st.astype("<i8",copy=False).tobytes())
 hh.update(np.asarray([material],"<i8").tobytes());hh.update(m.sha(Path(__file__)).encode())
 return hh.hexdigest()
def adjacent_fixture(out):
 out.mkdir(exist_ok=False,parents=True)
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[0,0,-.01],
 [.003,.003,1e-9],[.003,.003,-1e-9]],np.float32)
 upper=m.m.orient(p,[[0,1,2,3]]);lower=m.m.orient(p,[[0,2,1,4]])
 olda=np.c_[np.full(4,5),m.boundary(upper)]
 oldb=np.c_[np.full(4,6),m.boundary(lower)]
 aa,why=fan(p,olda);assert aa is not None,why
 bb,why=fan(p,oldb);assert bb is not None,why
 pa,ta,ra=aa;pb,tb,rb=bb
 assert set(olda.ravel())&set(oldb.ravel())
 tb[tb==len(p)]=len(p)+1
 pp=np.vstack([p,pa,pb]);tt=np.vstack([ta,tb]);old=np.vstack([olda,oldb])
 assert np.all(m.m.passes(*m.m.measurements(pp,tt)))
 assert m.m.certify(pp,old,tt)==(True,"PASS")
 assert np.array_equal(m.boundary_identity(m.boundary(old)),m.boundary_identity(m.boundary(tt)))
 assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
 assert signature(p,olda,15)==signature(p,olda[::-1],15)
 moved=p.copy();moved[3,2]=np.float32(.011)
 assert signature(moved,olda,15)!=signature(p,olda,15)
 r={"status":"PASS_CELL_DISJOINT_SHARED_BOUNDARY_KERNEL_FAN_CONTROLS",
 "physical_admission":False,"native_steps":0,"controls":["shared_boundary_vertices_and_face_allowed","disjoint_old_cells","combined_exact_oriented_boundary_volume","combined_native_quality","new_point_identity_distinct","order_invariant_cache","coordinate_invalidates_cache"],"bad_before":2,"bad_after":0}
 (out/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
adaptive=load("adaptive249for250",H/"remesh_adaptive_convex_kernel_fans_249_root.py")
# Identical cavity mathematics and signature reuse prior negative work safely;
# only the connected cavity enumeration grows to pair/triple corner-star unions.
fan=adaptive.extended_fan
signature=adaptive.signature

C=load("cavity247for270",H/"remesh_current_surface_cavities_247_root.py")
growth_stats=Counter()
def connected_growth_domains(p,t,mat,neighbor,seed,used,maximum_points):
    """Retire only face-connected cells. Absorb internal bad-face witnesses first."""
    domain={int(seed)};material=int(mat[seed]);PAT=m.m.PAT
    for step in range(64):
        rows=np.asarray(sorted(domain),np.int64)
        if set(rows)&used:return
        if len(np.unique(t[rows]))>maximum_points:
            growth_stats["point_budget_stop"]+=1;return
        ff=[];neighbors=[]
        for ci in rows:
            for k,pat in enumerate(PAT):
                ni=int(neighbor[ci,k])
                if ni in domain:continue
                f=t[ci,pat];ff.append(f);neighbors.append(ni)
        if not ff:return
        ff=np.asarray(ff,np.int64);nb=np.asarray(neighbors,np.int64);xyz=p[ff].astype(float)
        u=xyz[:,1]-xyz[:,0];v=xyz[:,2]-xyz[:,0];w=xyz[:,2]-xyz[:,1]
        area=np.linalg.norm(np.cross(u,v),axis=1)
        length2=np.maximum.reduce((np.einsum('ij,ij->i',u,u),np.einsum('ij,ij->i',v,v),np.einsum('ij,ij->i',w,w)))
        upper=np.sqrt(3)/2*area/length2
        impossible=upper<1e-5*(1-1e-10)
        if np.any(impossible&(nb<0)):
            growth_stats["fixed_boundary_no_apex_witness"]+=1;return
        eligible=np.asarray([n>=0 and n not in used and int(mat[n])==material for n in nb],bool)
        if np.any(impossible&~eligible):
            growth_stats["unavailable_neighbor_no_apex_witness"]+=1;return
        # A hard cut-face obstruction cannot be solved by a center search.
        if not np.any(impossible) and (step<4 or step in (7,15,31,63)):
            growth_stats["yielded_domains"]+=1
            yield rows
        if not np.any(eligible):return
        hard=np.flatnonzero(impossible&eligible)
        if len(hard):
            chosen=int(hard[np.argmin(upper[hard])]);growth_stats["absorbed_internal_no_apex_faces"]+=1
        else:
            center=p[np.unique(t[rows])].astype(float).mean(0)
            a=xyz[:,0]-center;b=xyz[:,1]-center;c=xyz[:,2]-center
            det=np.einsum('ij,ij->i',a,np.cross(b,c))
            score=det/np.maximum(length2,1e-60)**1.5
            eligible_rows=np.flatnonzero(eligible)
            chosen=int(eligible_rows[np.argmin(score[eligible_rows])])
        domain.add(int(nb[chosen]));growth_stats["added_face_neighbors"]+=1
def growth_fixture(out):
    out.mkdir(parents=True,exist_ok=False)
    p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[.003,.003,1e-9]],np.float32)
    bd=m.boundary(np.array([[0,1,2,3]],np.int64));t=np.c_[np.full(4,4),bd];mat=np.full(4,15,np.int32)
    neighbor=C.base.adjacency(t)
    # Verify actual adjacency slots correspond to the shared face used for growth.
    for i in range(len(t)):
        for k,ni in enumerate(neighbor[i]):
            if ni>=0:assert len(set(t[i,m.m.PAT[k]])&set(t[ni]))==3
    d,q=m.m.measurements(p,t);seed=int(np.flatnonzero(~m.m.passes(d,q))[0])
    domains=list(connected_growth_domains(p,t,mat,neighbor,seed,set(),100))
    assert any(len(r)==4 for r in domains),'growth did not reach actual connected fan shell'
    ans,why=fan(p,t[next(r for r in domains if len(r)==4)]);assert ans is not None,why
    point,nt,report=ans;pp=np.vstack([p,point]);assert m.m.certify(pp,t,nt)==(True,'PASS')
    assert np.all(m.m.passes(*m.m.measurements(pp,nt)))
    # A physical impossible boundary must be retained as an explicit failure.
    thin=np.array([[0,0,0],[1,0,0],[.5,1.05e-5,0],[.3,.2,.5]],np.float32)
    tt=m.m.orient(thin,[[0,1,2,3]]);assert not list(connected_growth_domains(thin,tt,np.array([15]),C.base.adjacency(tt),0,set(),100))
    r={'status':'PASS_CONNECTED_FACE_GROWTH_ACTUAL_PATH_CONTROLS','physical_admission':False,'native_steps':0,'controls':['actual_adjacency_face_slots','internal_no_apex_cut_absorbed','connected_full_fan_reached','exact_boundary_and_volume','all_native_quality_replacement','fixed_impossible_boundary_not_ignored'],'growth_stats':dict(growth_stats)}
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
def run(a):
 tic=time.monotonic();a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"];d,q=m.m.measurements(p,t);good=m.m.passes(d,q)
 if np.any(d<=0):raise ValueError("positive input required")
 neighbor=C.base.adjacency(t)
 bad=np.flatnonzero(~good);seeds=bad[np.lexsort((bad,q[bad]))]
 rawcache=v.get("kernel_fan_no_progress_sha256",np.empty((0,32),np.uint8))
 if rawcache.dtype!=np.uint8 or rawcache.ndim!=2 or rawcache.shape[1]!=32:raise ValueError("invalid kernel cache")
 cache={row.tobytes().hex() for row in rawcache};cache_before=len(cache);cache_hits=0
 seen_vertices=set();shared_reuse=0
 used=set();removed=[];added=[];newm=[];newp=[];ops=[];rejections=Counter();seen=set();attempts=0;keys={m.key(x) for x in p}
 for seed in seeds:
  if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts:break
  if int(seed) in used:continue
  domains=connected_growth_domains(p,t,mat,neighbor,int(seed),used,a.maximum_points)
  for rows in domains:
   sig=tuple(rows.tolist())
   if sig in seen:continue
   seen.add(sig);vertices=set(map(int,t[rows].ravel()))
   if set(map(int,rows))&used or len(vertices)>a.maximum_points:continue
   if len(np.unique(mat[rows]))!=1:rejections["material_interface"]+=1;continue
   sigbytes=signature(p,t[rows],int(mat[rows[0]]))
   if sigbytes in cache:cache_hits+=1;continue
   attempts+=1;ans,why=fan(p,t[rows])
   if ans is None:
    rejections[why]+=1;cache.add(sigbytes);continue
   point,nt,r=ans
   if m.key(point) in keys:rejections["existing_coordinate"]+=1;continue
   nt[nt==len(p)]=len(p)+len(newp)
   keys.add(m.key(point));newp.append(point);used.update(map(int,rows));shared_reuse+=int(bool(vertices&seen_vertices));seen_vertices.update(vertices);removed.extend(rows.tolist());added.append(nt);newm.append(np.full(len(nt),int(mat[rows[0]]),np.int32))
   case=a.output/("pass-%04d"%len(ops));case.mkdir()
   np.savez_compressed(case/"replacement.npz",old_cell_rows=rows,new_tetrahedra=nt,new_point_world_m=point,material_indices=newm[-1])
   r.update(seed_cell=int(seed),old_cell_rows=rows.tolist(),old_cells=len(rows),vertices=len(vertices),new_global_point_row=len(p)+len(newp)-1,point_world_m=point.tolist(),inferred_construction=True)
   (case/"report.json").write_text(json.dumps(r,indent=2)+"\n");ops.append(r);break
  if time.monotonic()-tic>a.seconds:break
 keep=np.ones(len(t),bool);keep[removed]=False;pp=np.vstack([p,np.array(newp,np.float32).reshape(-1,3)]);tt=np.vstack([t[keep],*(added or [np.empty((0,4),np.int64)])]);mm=np.concatenate([mat[keep],*newm]);bd=m.boundary(tt)
 assert np.array_equal(m.boundary_identity(bd),m.boundary_identity(m.boundary(t)))
 nd,nq=m.m.measurements(pp,tt);assert np.all(nd>0)
 newbad=int(sum(~m.m.passes(nd,nq)));assert newbad==int(sum(~good))-sum(o["old_native_failures"]-o["new_native_failures"] for o in ops)
 assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
 out=a.output/"kernel-fan-volume-proposal.npz";np.savez_compressed(out,points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bd,desired_boundary_faces=v.get("desired_boundary_faces",bd),solver_infeasible_cavity_sha256=v.get("solver_infeasible_cavity_sha256",np.empty((0,32),np.uint8)),kernel_fan_no_progress_sha256=np.asarray([list(bytes.fromhex(x)) for x in sorted(cache)],np.uint8).reshape(-1,32))
 report={"status":"PASS_NATIVE_QUALITY_KERNEL_FAN_BATCH" if ops else "NO_NATIVE_QUALITY_KERNEL_FAN_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"source_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"bad_before":int(sum(~good)),"bad_after":newbad,"nonpositive":int(sum(nd<=0)),"input_cells":len(t),"output_cells":len(tt),"new_inferred_interior_points":len(newp),"enumeration":"face_connected_witness_growth","growth_statistics":dict(growth_stats),"identical_cavity_algorithm_source_sha256":m.sha(Path(adaptive.__file__)),"adaptive_kernel":adaptive.stats,"cell_disjoint_cavities":True,"shared_boundary_vertex_reuse_operations":shared_reuse,"cache_entries_before":cache_before,"cache_entries_after":len(cache),"cache_hits":cache_hits,"attempts":attempts,"passing_operations":len(ops),"rejections":dict(rejections),"oriented_boundary_bit_identical":True,"original_points_bit_identical":True,"operations":ops,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=60);ap.add_argument("--maximum-attempts",type=int,default=1000);ap.add_argument("--maximum-points",type=int,default=150);a=ap.parse_args();growth_fixture(a.output) if a.fixture else run(a)
