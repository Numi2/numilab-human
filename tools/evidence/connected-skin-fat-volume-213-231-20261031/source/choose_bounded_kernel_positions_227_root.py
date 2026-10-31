"""Bounded local determinant-kernel correction of unresolved D construction nodes.
This changes inferred construction coordinates only. Exact anatomical intersection,
typed ancestry, material mass and native admission remain mandatory afterward.
"""
from pathlib import Path
from itertools import combinations,product
from collections import Counter
import numpy as np,json,argparse,time,importlib.util
from scipy.optimize import linprog
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("prev223",H/"choose_volume_aware_endpoints_223_root.py");prev=importlib.util.module_from_spec(s);s.loader.exec_module(prev);m=prev.m
def kernel_centers(lp,lt,ri,lower,upper,oldgood):
 root=lp[ri].astype(float);scale=float(np.max(upper-lower))
 if not scale>0:return []
 slots=np.argmax(lt==ri,axis=1);opp=np.asarray([tet[m.m.PAT[k]] for tet,k in zip(lt,slots)],np.int64)
 xyz=lp[opp].astype(float);normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);norm=np.linalg.norm(normal,axis=1)
 if np.any(norm<=0):return []
 d0=np.einsum("ij,ij->i",normal,xyz[:,0]-root)
 # Opposite face normal gives d(x)=d(root)-normal dot(x-root).
 unit=normal/norm[:,None];maxedge=np.zeros(len(lt))
 q=lp[lt].astype(float)
 for i,j in combinations(range(4),2):maxedge=np.maximum(maxedge,np.linalg.norm(q[:,i]-q[:,j],axis=1))
 for j in range(4):
  delta=np.maximum(np.abs(lp[lt[:,j]].astype(float)-lower),np.abs(lp[lt[:,j]].astype(float)-upper));maxedge=np.maximum(maxedge,np.linalg.norm(delta,axis=1))
 native=np.maximum(6e-18*(1+1e-6),1e-5*maxedge**3*(1+1e-6));positive=np.maximum(6e-27,64*np.finfo(float).eps*maxedge**3)
 bounds=[(float((lower[j]-root[j])/scale),float((upper[j]-root[j])/scale)) for j in range(3)]+[(0,None)]
 candidates=[]
 for mode,req in (("all_native",native),("preserve_native_and_positive",np.where(oldgood,native,positive))):
  sol=linprog([0,0,0,-1],A_ub=np.c_[unit,np.ones(len(lt))],b_ub=(d0-req)/norm/scale,bounds=bounds,method="highs",options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
  if sol.success and sol.x[3]>0:candidates.append((mode,root+scale*sol.x[:3],float(sol.x[3]*scale)))
 return candidates
def fixture():
 # Interior root moved outside a closed four-tet star; a bounded kernel restores it.
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[.006,.006,.006]],np.float32)
 boundary=m.boundary(np.array([[0,1,2,3]],np.int64));t=np.c_[np.full(4,4),boundary]
 od,oq=m.m.measurements(p,t);assert np.any(od<0)
 cs=kernel_centers(p,t,4,np.zeros(3),np.full(3,.01),m.m.passes(od,oq));assert cs
 for mode,point,radius in cs:
  pp=p.copy();pp[4]=point.astype(np.float32)
  assert np.all(m.m.measurements(pp,t)[0]>0) and radius>0
 return {"status":"PASS_BOUNDED_DETERMINANT_KERNEL_CONTROLS","controls":["outside_star_restored","positive_radius","F32_positive_emission"],"native_steps":0,"physical_admission":False}
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"].copy();source=v["source_points_before_segment"];t=v["tetrahedra"];b=v["desired_boundary_faces"];mp=v["input_node_quotient"]
 lookup={m.key(x):i for i,x in enumerate(source)}
 with np.load(a.d_surface) as z:dp=z["points_world_m"];df=z["faces"]
 with np.load(a.i_surface) as z:ii=z["points_world_m"];iff=z["faces"]
 eligible={lookup[m.key(x)] for x in dp[np.unique(df)]}-{lookup[m.key(x)] for x in ii[np.unique(iff)]}
 used=np.unique(t);active={m.key(p[i]):int(i) for i in used}
 if len(active)!=len(used):raise ValueError("duplicate active coordinates")
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 sort=np.argsort(mp,kind="stable");counts=np.bincount(mp,minlength=len(p));coff=np.r_[0,np.cumsum(counts)]
 d,q=m.m.measurements(p,t);initial=int(sum(d<=0));initialq=int(sum(~m.m.passes(d,q)));roots=[int(x) for x in np.unique(t[d<=0]) if x in eligible and counts[x]>1]
 ops=[];rej=Counter();trials=0
 for root in roots:
  rows=np.unique(order[off[root]:off[root+1]]//4);br=np.flatnonzero(np.any(b==root,axis=1));nodes=np.unique(np.r_[t[rows].ravel(),b[br].ravel()]);loc=np.full(len(p),-1,np.int64);loc[nodes]=np.arange(len(nodes))
  lp=p[nodes].copy();lt=loc[t[rows]];lb=loc[b[br]];ri=int(loc[root]);od,oq=m.m.measurements(lp,lt);og=m.m.passes(od,oq);oneg=od<=0
  if not np.any(oneg):continue
  fq,fn=prev.face_measure(lp,lb);fg=fq>=1e-5;members=sort[coff[root]:coff[root+1]];src=source[members].astype(float)
  diam=float(max(np.linalg.norm(a-b) for a in src for b in src));ulp=float(np.max(np.abs(np.nextafter(p[root],np.float32(np.inf))-p[root])));pad=max(2*ulp,.1*diam)
  lower=src.min(0)-pad;upper=src.max(0)+pad;best=None;seen=set()
  centers=kernel_centers(lp,lt,ri,lower,upper,og)
  if not centers:rej["no_bounded_kernel"]+=1;continue
  for mode,planned,radius in centers:
   center=planned.astype(np.float32);axes=[[np.nextafter(center[j],np.float32(-np.inf)),center[j],np.nextafter(center[j],np.float32(np.inf))] for j in range(3)]
   for values in product(*axes):
    point=np.asarray(values,np.float32);key=m.key(point)
    if key in seen:continue
    seen.add(key)
    if np.any(point<lower) or np.any(point>upper):continue
    if key in active and active[key]!=root:rej["active_coordinate_collision"]+=1;continue
    lp[ri]=point;trials+=1;nd,nq=m.m.measurements(lp,lt);ng=m.m.passes(nd,nq);nneg=nd<=0
    if np.any(~oneg&nneg) or sum(nneg)>=sum(oneg):rej["nonpositive_not_reduced"]+=1;continue
    if np.any(og&~ng) or sum(~ng)>sum(~og):rej["native_quality_regression"]+=1;continue
    nf,nn=prev.face_measure(lp,lb)
    if np.any(nf<=0) or np.any(fg&(nf<1e-5)) or np.any(np.einsum("ij,ij->i",fn,nn)<=0):rej["boundary_shape_regression"]+=1;continue
    ip=m.m.integer_points(lp);ss=np.asarray([(lambda d:(d>0)-(d<0))(m.m.exact_det(ip,tet)) for tet in lt],np.int8)
    if np.any(~nneg&(ss<=0)) or np.any(~oneg&(ss<=0)):rej["exact_sign"]+=1;continue
    score=(int(sum(nneg)),int(sum(~ng)),-float(nq.min()),float(np.linalg.norm(point.astype(float)-p[root])))
    if best is None or score<best[0]:best=(score,point.copy(),mode,planned,radius,nd,nq)
  if best is None:continue
  score,point,mode,planned,radius,nd,nq=best;old=p[root].copy();del active[m.key(old)];active[m.key(point)]=root;p[root]=point
  ops.append({"current_node":root,"position_before_world_m":old.tolist(),"position_after_world_m":point.tolist(),"source_cluster_nodes":members.tolist(),"source_cluster_diameter_m":diam,"construction_pad_m":pad,"bounded_lower_world_m":lower.tolist(),"bounded_upper_world_m":upper.tolist(),"LP_mode":mode,"kernel_radius_m":radius,"planned_double_point_world_m":planned.tolist(),"displacement_m":float(np.linalg.norm(point.astype(float)-old)),"affected_cell_rows":rows.tolist(),"affected_desired_face_rows":br.tolist(),"old_nonpositive":int(sum(oneg)),"new_nonpositive":int(sum(nd<=0)),"old_quality_failures":int(sum(~og)),"new_quality_failures":int(sum(~m.m.passes(nd,nq)))})
 d,q=m.m.measurements(p,t);out=a.output/"bounded-kernel-volume-and-boundary-proposal.npz";v["source_points_before_kernel"]=v["points_world_m"];v["points_world_m"]=p;v["kernel_changed_point_rows"]=np.asarray([o["current_node"] for o in ops],np.int64);np.savez_compressed(out,**v)
 r={"status":"BOUNDED_KERNEL_PROPOSAL_REQUIRES_EXACT_GEOMETRY_AND_ANCESTRY","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"clusters":len(roots),"trials":trials,"passing_operations":len(ops),"nonpositive_before":initial,"nonpositive_after":int(sum(d<=0)),"quality_failures_before":initialq,"quality_failures_after":int(sum(~m.m.passes(d,q))),"rejections":dict(rej),"elapsed_s":time.monotonic()-tic,"operations":ops}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--d-surface",type=Path);ap.add_argument("--i-surface",type=Path);ap.add_argument("--output",type=Path);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture()))
 else:run(a)
