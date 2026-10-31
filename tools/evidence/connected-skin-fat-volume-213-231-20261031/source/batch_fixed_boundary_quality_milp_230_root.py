"""Bounded quality repair of complete fixed-coordinate vertex cavities.
Existing native-quality, exact F32, manifold and intersection owners remain gates.
Compiler proposal only: no physical-state replacement or mass assignment.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import argparse,importlib.util,json,numpy as np,time
H=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
top=load("top218for230",H/"remesh_complete_surface_cavities_218_root.py")
m=top.m
solver=load("milp220for230",H/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py")
def coverage(p,b):
 # A missing native-admissible boundary face proves infeasibility before MILP.
 ct=np.asarray(list(combinations(range(len(p)),4)),np.int64)
 if not len(ct):return False,{"candidate_count":0,"missing_faces":len(b)}
 d,qq=m.m.measurements(p,ct);ct=m.m.orient(p,ct[d!=0])
 good=solver.native_quality_mask(p,ct)[0];ct=ct[good]
 facekeys={tuple(sorted(map(int,f))) for f in ct[:,m.m.PAT].reshape(-1,3)}
 missing=[f.tolist() for f in b if tuple(sorted(map(int,f))) not in facekeys]
 return not missing,{"candidate_count":len(ct),"missing_faces":len(missing),"first_missing_faces":missing[:4]}
def fixture():
 p=np.array([[-.01,-.01,0],[.01,-.01,0],[.01,.01,0],[-.01,.01,0],[0,0,.01],[0,0,-.01],[0,-.009999999,0]],np.float32)
 old=m.m.orient(p,[[4,6,0,1],[4,6,1,2],[4,6,2,3],[4,6,3,0],[5,6,1,0],[5,6,2,1],[5,6,3,2],[5,6,0,3]])
 b=m.boundary(old);ok,cr=coverage(p,b);assert ok,cr
 r,new=solver.solve(p,b,seconds=3,max_points=32,material=15)
 assert new is not None,r
 assert np.all(m.m.passes(*m.m.measurements(p,new)))
 assert m.m.certify(p,old,new)==(True,"PASS")
 pp=np.array([[0,0,0],[.01,0,0],[0,1e-12,0],[0,0,.01]],np.float32)
 bb=m.boundary(np.array([[0,1,2,3]],np.int64));assert not coverage(pp,bb)[0]
 assert top.manifold_topology(b)["boundary_components"]==1
 return {"status":"PASS_FIXED_BOUNDARY_QUALITY_MILP_CONTROLS","controls":["coverage_admission","unattainable_boundary_rejected","strict_native_cell_quality","exact_F32_boundary_volume","exact_internal_face_crossing_owner"],"physical_admission":False,"native_steps":0}
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"]
 d,qq=m.m.measurements(p,t);good=m.m.passes(d,qq)
 if np.any(d<=0):raise ValueError("requires positive baseline; orientation repairs have separate owner")
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 badhits=np.bincount(t[~good].ravel(),minlength=len(p));degree=np.diff(off)
 nodes=np.flatnonzero(badhits)
 nodes=sorted(map(int,nodes),key=lambda n:(-float(badhits[n])/max(int(degree[n]),1),-int(badhits[n]),n))
 used=set();removed=[];added=[];mats=[];ops=[];rej=Counter();attempts=0;trials=[]
 for node in nodes:
  if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts or len(ops)>=a.maximum_operations:break
  rows=np.unique(order[off[node]:off[node+1]]//4);ot=t[rows];vertices=np.unique(ot);vs=set(map(int,vertices))
  if vs&used:rej["conflict"]+=1;continue
  if len(vertices)>a.maximum_points:rej["point_budget"]+=1;continue
  if len(np.unique(mat[rows]))!=1:rej["material_interface"]+=1;continue
  b=m.boundary(ot);topo=top.manifold_topology(b)
  if not top.closed(b)[0] or topo["bad_vertex_links"] or topo["boundary_components"]!=1:rej["nonmanifold_cavity"]+=1;continue
  ix=np.full(len(p),-1,np.int64);ix[vertices]=np.arange(len(vertices));lp=p[vertices];lb=ix[b]
  attempts+=1;ok,pre=coverage(lp,lb)
  if not ok:rej["missing_native_boundary_candidate"]+=1;continue
  case=a.output/("solve-%04d"%len(trials));case.mkdir()
  rr,nt=solver.solve(lp,lb,seconds=a.solve_seconds,max_points=a.maximum_points,material=int(mat[rows[0]]))
  trial={"node":node,"old_cell_rows":rows.tolist(),"vertices":len(vertices),"old_native_failures":int(sum(~good[rows])),"coverage":pre,"solve":rr}
  trials.append(trial);(case/"report.json").write_text(json.dumps(trial,indent=2)+"\n")
  if nt is None:rej[rr["status"]]+=1;continue
  new=vertices[nt];ok,why=m.m.certify(p,ot,new)
  if not ok:raise ValueError("exact local certificate "+why)
  assert np.all(m.m.passes(*m.m.measurements(p,new)))
  used.update(vs);removed.extend(rows.tolist());added.append(new);mats.append(np.full(len(new),int(mat[rows[0]]),np.int32))
  np.savez_compressed(case/"replacement.npz",old_cell_rows=rows,new_tetrahedra=new,material_indices=mats[-1])
  ops.append({"node":node,"old_cell_rows":rows.tolist(),"new_cell_count":len(new),"bad_before":int(sum(~good[rows])),"bad_after":0,"certificate":rr})
 keep=np.ones(len(t),bool);keep[removed]=False
 tt=np.vstack([t[keep],*(added or [np.empty((0,4),np.int64)])]);mm=np.concatenate([mat[keep],*mats])
 nd,nq=m.m.measurements(p,tt);newbad=int(sum(~m.m.passes(nd,nq)))
 assert np.all(nd>0) and newbad==int(sum(~good))-sum(o["bad_before"] for o in ops)
 ib=m.boundary(t);ob=m.boundary(tt)
 assert np.array_equal(m.boundary_identity(ib),m.boundary_identity(ob))
 ss=np.sort(tt,axis=1);assert len(np.unique(ss,axis=0))==len(tt)
 out=a.output/"fixed-boundary-quality-volume-proposal.npz"
 np.savez_compressed(out,points_world_m=p,tetrahedra=tt,material_indices=mm,boundary_faces=ob,desired_boundary_faces=v.get("desired_boundary_faces",ib),retained_input_cell_rows=np.flatnonzero(keep))
 report={"status":"PASS_BOUNDED_FIXED_BOUNDARY_QUALITY_MILP_BATCH" if ops else "NO_VALID_BOUNDED_QUALITY_MILP_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"solver_sha256":m.sha(Path(solver.__file__)),"output":str(out),"output_sha256":m.sha(out),"input_cells":len(t),"output_cells":len(tt),"bad_before":int(sum(~good)),"bad_after":newbad,"point_coordinates_bit_identical":True,"oriented_boundary_bit_identical":True,"same_material_and_exact_F32_cavity_volume":True,"attempts":attempts,"passing_operations":len(ops),"solve_count":len(trials),"rejections":dict(rej),"operations":ops,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--seconds",type=float,default=90);ap.add_argument("--solve-seconds",type=float,default=1);ap.add_argument("--maximum-points",type=int,default=32);ap.add_argument("--maximum-attempts",type=int,default=300);ap.add_argument("--maximum-operations",type=int,default=100);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture(),indent=2))
 else:run(a)
