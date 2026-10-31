"""Native-quality kernel fans on bounded connected material cavities.
Reuses determinant-kernel LP, exact boundary/sign/volume/intersection owners.
Adds declared inferred interior construction points only; no native admission.
"""
from pathlib import Path
from itertools import product
from collections import Counter
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
def run(a):
 tic=time.monotonic();a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"];d,q=m.m.measurements(p,t);good=m.m.passes(d,q)
 if np.any(d<=0):raise ValueError("positive input required")
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 bad=np.flatnonzero(~good);seeds=bad[np.lexsort((bad,-q[bad]))]
 used=set();removed=[];added=[];newm=[];newp=[];ops=[];rejections=Counter();seen=set();attempts=0;keys={m.key(x) for x in p}
 for seed in seeds:
  if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts:break
  if set(map(int,t[seed]))&used:continue
  roots=t[seed];domains=[]
  # Full corner stars, then their union: increasing geometric freedom has
  # linear face cost, instead of the combinatorial candidate cost of MILP.
  for root in roots:domains.append(np.unique(order[off[root]:off[root+1]]//4))
  domains.append(np.unique(np.concatenate(domains)))
  for rows in domains:
   sig=tuple(rows.tolist())
   if sig in seen:continue
   seen.add(sig);vertices=set(map(int,t[rows].ravel()))
   if vertices&used or len(vertices)>a.maximum_points:continue
   if len(np.unique(mat[rows]))!=1:rejections["material_interface"]+=1;continue
   attempts+=1;ans,why=fan(p,t[rows])
   if ans is None:rejections[why]+=1;continue
   point,nt,r=ans
   if m.key(point) in keys:rejections["existing_coordinate"]+=1;continue
   nt[nt==len(p)]=len(p)+len(newp)
   keys.add(m.key(point));newp.append(point);used.update(vertices);removed.extend(rows.tolist());added.append(nt);newm.append(np.full(len(nt),int(mat[rows[0]]),np.int32))
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
 out=a.output/"kernel-fan-volume-proposal.npz";np.savez_compressed(out,points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bd,desired_boundary_faces=v.get("desired_boundary_faces",bd),solver_infeasible_cavity_sha256=v.get("solver_infeasible_cavity_sha256",np.empty((0,32),np.uint8)))
 report={"status":"PASS_NATIVE_QUALITY_KERNEL_FAN_BATCH" if ops else "NO_NATIVE_QUALITY_KERNEL_FAN_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"source_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"bad_before":int(sum(~good)),"bad_after":newbad,"nonpositive":int(sum(nd<=0)),"input_cells":len(t),"output_cells":len(tt),"new_inferred_interior_points":len(newp),"attempts":attempts,"passing_operations":len(ops),"rejections":dict(rejections),"oriented_boundary_bit_identical":True,"original_points_bit_identical":True,"operations":ops,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=60);ap.add_argument("--maximum-attempts",type=int,default=1000);ap.add_argument("--maximum-points",type=int,default=150);a=ap.parse_args();fixture(a.output) if a.fixture else run(a)
