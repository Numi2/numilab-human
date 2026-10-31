"""Quality-targeted existing star-kernel cavity construction.
Uses the 062 kernel LP and 198 exact Float32 predicates. Fixed complete boundary;
same material; all new cells must pass the unchanged native-quality thresholds.
Offline proposal only. Never a runtime physics fallback.
"""
from pathlib import Path
from collections import Counter
import numpy as np,json,time,argparse,importlib.util
from scipy.optimize import linprog
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("base209",H/"transfer_surface_contractions_to_volume_209_root.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def adjacency(t):
 f=t[:,m.m.PAT].reshape(-1,3);sf=np.sort(f,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));ss=sf[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[starts,len(ss)])
 if np.any(ct>2):raise ValueError("nonmanifold input")
 ids=starts[ct==2];pairs=order[np.c_[ids,ids+1]];neigh=np.full(4*len(t),-1,np.int64)
 neigh[pairs[:,0]]=pairs[:,1]//4;neigh[pairs[:,1]]=pairs[:,0]//4
 return neigh.reshape(-1,4)
def cavity_boundary(t):
 b=m.boundary(t)
 e=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1)
 _,counts=np.unique(e,axis=0,return_counts=True)
 if np.any(counts!=2):return None,"nonmanifold_boundary_edges"
 if len(np.unique(b))-len(counts)+len(b)!=2:return None,"non_spherical_boundary"
 return b,"PASS"
def propose(p,old):
 b,why=cavity_boundary(old)
 if b is None:return None,why
 q=p[b].astype(float);v=p[np.unique(b)].astype(float)
 normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(normal,axis=1)
 if np.any(area<=0):return None,"zero_boundary_face"
 normal/=area[:,None];origin=(v.max(0)+v.min(0))/2;scale=float(np.max(np.ptp(v,axis=0)))
 if not scale>0:return None,"zero_scale"
 # Existing star-kernel construction, translated/scaled to avoid world-coordinate conditioning.
 sol=linprog([0,0,0,-1],A_ub=np.c_[normal,np.ones(len(b))],
  b_ub=np.einsum("ij,ij->i",normal,(q[:,0]-origin)/scale),
  bounds=[(-1,1)]*3+[(0,None)],method="highs",
  options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
 if not sol.success or sol.x[3]<=0:return None,"no_strict_kernel"
 center=(origin+scale*sol.x[:3]).astype(np.float32)
 pp=np.vstack([p,center]);fan=np.c_[np.full(len(b),len(p),np.int64),b]
 det,nq=m.m.measurements(pp,fan)
 if not np.all(m.m.passes(det,nq)):return None,"fan_native_quality"
 if m.m.certify(pp,old,fan)!=(True,"PASS"):return None,"exact_cavity_certificate"
 return (center,fan,b),"PASS"
def fixture():
 # A nearly flat artificial internal separator in a convex bipyramid.
 p=np.array([[-.01,-.01,0],[.01,-.01,0],[.01,.01,0],[-.01,.01,0],[0,0,.01],[0,0,-.01],[0,-.009999999,0]],np.float32)
 old=m.m.orient(p,[[4,6,0,1],[4,6,1,2],[4,6,2,3],[4,6,3,0],[5,6,1,0],[5,6,2,1],[5,6,3,2],[5,6,0,3]])
 assert sum(~m.m.passes(*m.m.measurements(p,old)))>0
 result,why=propose(p,old);assert result is not None,why
 center,fan,b=result;pp=np.vstack([p,center])
 assert np.all(m.m.passes(*m.m.measurements(pp,fan)))
 assert m.m.certify(pp,old,fan)==(True,"PASS")
 assert np.array_equal(m.boundary_identity(b),m.boundary_identity(m.boundary(fan)))
 # An open two-tetra shape still has a closed cavity surface; zero/inverted fan rejected by exact cert.
 bad=fan.copy();bad[0,[0,1]]=bad[0,[1,0]]
 assert m.m.certify(pp,old,bad)[0] is False
 assert propose(p,old[:1])[0] is None
 return {"status":"PASS_QUALITY_STAR_CAVITY_FIXTURES","controls":["native_quality_improvement","unchanged_oriented_boundary","exact_F32_volume","inverted_fan_rejected","unrepairable_small_cavity_rejected"],"physical_admission":False,"native_steps":0}
def run(a):
 start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.input) as z:p=z["points_world_m"];t=z["tetrahedra"];mat=z["material_indices"]
 det,q=m.m.measurements(p,t);good=m.m.passes(det,q)
 if np.any(det<=0):raise ValueError("nonpositive baseline")
 neigh=adjacency(t);used=set();deleted=[];newpoints=[];newtets=[];newmat=[];ops=[];rej=Counter();attempts=0
 for seed in np.flatnonzero(~good):
  if time.monotonic()-start>a.seconds or len(ops)>=a.maximum_operations or attempts>=a.maximum_attempts:break
  if set(map(int,t[seed]))&used:rej["conflict"]+=1;continue
  found=False
  rows={int(seed)}
  for radius in range(1,a.maximum_radius+1):
   for r in list(rows):rows.update(int(x) for x in neigh[r] if x>=0)
   rr=np.asarray(sorted(rows),np.int64);old=t[rr];verts=set(map(int,old.ravel()))
   if verts&used:rej["conflict"]+=1;break
   if len(np.unique(mat[rr]))!=1:rej["mixed_material"]+=1;break
   attempts+=1;prop,why=propose(p,old)
   if prop is None:rej[why]+=1;continue
   center,fan,b=prop
   # Fan boundary contains no interior old node; all other interfaces remain exact.
   newnode=len(p)+len(newpoints);fan=fan.copy();fan[:,0]=newnode
   newpoints.append(center);newtets.append(fan);newmat.extend([int(mat[rr[0]])]*len(fan));deleted.extend(rr.tolist());used.update(verts)
   ops.append({"seed":int(seed),"radius":radius,"old_rows":rr.tolist(),"new_node_row":newnode,"new_point_world_m":center.tolist(),"new_cell_count":len(fan),"bad_before":int(sum(~good[rr])),"bad_after":0})
   found=True;break
 keep=np.ones(len(t),bool);keep[deleted]=False
 pp=np.vstack([p,np.asarray(newpoints,np.float32).reshape(-1,3)])
 tt=np.vstack([t[keep],*(newtets or [np.empty((0,4),np.int64)])])
 mm=np.r_[mat[keep],np.asarray(newmat,np.int32)]
 dd,qq=m.m.measurements(pp,tt)
 assert np.all(dd>0) and sum(~m.m.passes(dd,qq))==sum(~good)-sum(o["bad_before"] for o in ops)
 ib=m.boundary_identity(m.boundary(t));ob=m.boundary_identity(m.boundary(tt))
 if not np.array_equal(ib,ob):raise ValueError("global oriented boundary mismatch")
 output=a.output/"fixed-boundary-volume-proposal.npz"
 np.savez_compressed(output,points_world_m=pp,tetrahedra=tt,material_indices=mm,retained_input_cell_rows=np.flatnonzero(keep),new_interior_points_world_m=np.asarray(newpoints,np.float32).reshape(-1,3),replacement_parent_offsets=np.r_[0,np.cumsum([len(o["old_rows"]) for o in ops])],replacement_parent_cell_rows=np.asarray(deleted,np.int64))
 r={"status":"PASS_QUALITY_STAR_CAVITY_PROPOSAL" if ops else "NO_VALID_QUALITY_STAR_PROGRESS","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"output":str(output),"output_sha256":m.sha(output),"input_nodes":len(p),"output_nodes":len(pp),"input_cells":len(t),"output_cells":len(tt),"passing_operations":len(ops),"attempts":attempts,"bad_before":int(sum(~good)),"bad_after":int(sum(~m.m.passes(dd,qq))),"boundary_bit_identical":True,"exact_F32_cavity_volume":True,"material_preserved":True,"new_cells_all_native_quality_pass":True,"rejections":dict(rej),"elapsed_s":time.monotonic()-start,"operations":ops}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="operations"}))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--seconds",type=float,default=60);ap.add_argument("--maximum-radius",type=int,default=2);ap.add_argument("--maximum-operations",type=int,default=10000);ap.add_argument("--maximum-attempts",type=int,default=1000);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture()))
 else:run(a)
