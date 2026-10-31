"""Marginal-defect-first connected cavity remeshing using existing exact owners.
No coordinate movement, source-mass change, threshold relaxation or native admission.
"""
from pathlib import Path
from collections import Counter
import argparse,importlib.util,sys,time,json,numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("pair237for242",H/"pair_star_native_defect_descent_237_root.py");prev=importlib.util.module_from_spec(s);s.loader.exec_module(prev)
m=prev.m;top=prev.top;solver=prev.solver
def run(a):
 tic=time.monotonic();a.output.mkdir(exist_ok=False,parents=True)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"];t=v["tetrahedra"];mat=v["material_indices"];d,q=m.m.measurements(p,t);g=m.m.passes(d,q)
 if np.any(d<=0):raise ValueError("positive baseline required")
 neighbors=top.base.adjacency(t)
 bad=np.flatnonzero(~g);rank=np.lexsort((bad,-q[bad]));seeds=bad[rank]
 used=set();removed=[];added=[];mats=[];ops=[];rej=Counter();attempts=0;seen=set();infeasible={row.tobytes().hex() for row in v.get("solver_infeasible_cavity_sha256",np.empty((0,32),np.uint8))};cache_initial=len(infeasible);hits=0
 for seed in seeds:
  if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts:break
  if set(map(int,t[seed]))&used:continue
  rows={int(seed)};vertices=set(map(int,t[seed]));front=set(map(int,neighbors[seed][neighbors[seed]>=0]))
  # Grow by adjacent physical cells. Favor a new non-coplanar direction, not
  # nearest surface vertices, so a one-sided sliver can reach the other layer.
  for step in range(a.maximum_cells-1):
   choices=[]
   for r in front:
    vv=vertices|set(map(int,t[r]))
    if len(vv)>a.maximum_points or vv&used or mat[r]!=mat[seed]:continue
    xyz=p[sorted(vv)].astype(float);center=xyz.mean(0);sv=np.linalg.svd(xyz-center,compute_uv=False)
    spread=float(sv[-1]/max(sv[0],1e-300)) if len(sv)==3 else 0.
    choices.append((not bool(g[r]),-spread,-float(d[r]),r))
   if not choices:break
   r=min(choices)[-1];front.remove(r);rows.add(r);vertices.update(map(int,t[r]));front.update(int(x) for x in neighbors[r] if x>=0 and int(x) not in rows)
   # Do not ask the same MILP on every growth step; a short connected path
   # and then bounded larger neighborhoods offer different admissible apexes.
   if step not in (0,1,3,7,15,23,31,47):continue
   rr=np.array(sorted(rows),np.int64);ot=t[rr];budget=int(sum(~g[rr]))-1
   if budget<0:continue
   sig=prev.cavity_signature(p,ot,int(mat[seed]),budget)
   if sig in infeasible or sig in seen:hits+=1;continue
   seen.add(sig)
   b=m.boundary(ot);tp=top.manifold_topology(b)
   if not top.closed(b)[0] or tp["bad_vertex_links"] or tp["boundary_components"]!=1:rej["nonmanifold_cavity"]+=1;continue
   vv=np.array(sorted(vertices),np.int64);ix=np.full(len(p),-1,np.int64);ix[vv]=np.arange(len(vv));lp=p[vv];lb=ix[b]
   attempts+=1
   case=a.output/("solve-%04d"%attempts);case.mkdir()
   cert,nt=solver.solve(lp,lb,seconds=a.solve_seconds,max_points=a.maximum_points,material=int(mat[seed]),maximum_native_failures=budget)
   rec={"seed_cell":int(seed),"growth_cells":len(rr),"vertices":len(vv),"old_native_failures":int(sum(~g[rr])),"solve":cert}
   (case/"report.json").write_text(json.dumps(rec,indent=2)+"\n")
   if nt is None:
    if cert.get("milp_status")==2:infeasible.add(sig)
    rej[cert["status"]]+=1
   else:
    new=vv[nt];ok,why=m.m.certify(p,ot,new)
    if not ok:raise ValueError("exact local certificate:"+why)
    nd,nq=m.m.measurements(p,new);newbad=int(sum(~m.m.passes(nd,nq)))
    assert np.all(nd>0) and newbad<int(sum(~g[rr]))
    np.savez_compressed(case/"replacement.npz",old_cell_rows=rr,new_tetrahedra=new,material_indices=np.full(len(new),int(mat[seed]),np.int32))
    rec.update(old_cell_rows=rr.tolist(),new_cells=len(new),new_native_failures=newbad,exact_F32_boundary_and_volume=True)
    (case/"report.json").write_text(json.dumps(rec,indent=2)+"\n")
    removed.extend(rr.tolist());added.append(new);mats.append(np.full(len(new),int(mat[seed]),np.int32));used.update(vertices);ops.append(rec);break
   if time.monotonic()-tic>a.seconds or attempts>=a.maximum_attempts:break
 keep=np.ones(len(t),bool);keep[removed]=False;tt=np.vstack([t[keep],*(added or [np.empty((0,4),np.int64)])]);mm=np.concatenate([mat[keep],*mats]);nb=m.boundary(tt)
 assert np.array_equal(m.boundary_identity(nb),m.boundary_identity(m.boundary(t)))
 nd,nq=m.m.measurements(p,tt);assert np.all(nd>0)
 newbad=int(sum(~m.m.passes(nd,nq)));assert newbad==int(sum(~g))-sum(o["old_native_failures"]-o["new_native_failures"] for o in ops)
 assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
 out=a.output/"through-layer-cavity-volume-proposal.npz"
 np.savez_compressed(out,points_world_m=p,tetrahedra=tt,material_indices=mm,boundary_faces=nb,desired_boundary_faces=v.get("desired_boundary_faces",nb),solver_infeasible_cavity_sha256=np.asarray([list(bytes.fromhex(x)) for x in sorted(infeasible)],np.uint8).reshape(-1,32))
 report={"status":"PASS_WITNESS_CONNECTED_NATIVE_DEFECT_DESCENT" if ops else "NO_WITNESS_CONNECTED_NATIVE_DEFECT_DESCENT_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"source_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"bad_before":int(sum(~g)),"bad_after":newbad,"nonpositive":int(sum(nd<=0)),"input_cells":len(t),"output_cells":len(tt),"attempts":attempts,"passing_operations":len(ops),"cache_entries_before":cache_initial,"cache_entries_after":len(infeasible),"cache_hits":hits,"rejections":dict(rej),"point_coordinates_bit_identical":True,"oriented_boundary_bit_identical":True,"operations":ops,"elapsed_s":time.monotonic()-tic}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="operations"},indent=2))
def fixture(a):
 a.output.mkdir(exist_ok=False,parents=True);p=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1e-7],[.2,.2,-1]],np.float32)
 t=m.m.orient(p,[[0,1,2,3],[0,2,1,4]])
 for name,mt in (("same_material",np.array([15,15],np.int32)),("material_interface",np.array([15,16],np.int32))):
  src=a.output/(name+".npz");np.savez_compressed(src,points_world_m=p,tetrahedra=t,material_indices=mt,boundary_faces=m.boundary(t))
  aa=argparse.Namespace(**vars(a));aa.input=src;aa.output=a.output/name;run(aa)
  r=json.loads((aa.output/"report.json").read_text())
  assert r["nonpositive"]==0 and r["oriented_boundary_bit_identical"]
  assert r["bad_after"]==(0 if name=="same_material" else 1)
 r={"status":"PASS_WITNESS_CONNECTED_CAVITY_CONTROLS","controls":["actual_growth_and_MILP_path","strict_native_defect_descent","material_interface_preserved","exact_fixed_boundary","exact_positive_volume"],"physical_admission":False,"native_steps":0}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=60);ap.add_argument("--solve-seconds",type=float,default=.25);ap.add_argument("--maximum-attempts",type=int,default=1000);ap.add_argument("--maximum-points",type=int,default=26);ap.add_argument("--maximum-cells",type=int,default=32);a=ap.parse_args();fixture(a) if a.fixture else run(a)
