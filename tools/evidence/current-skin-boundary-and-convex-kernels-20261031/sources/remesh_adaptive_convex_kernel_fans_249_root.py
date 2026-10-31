"""Convex adaptive kernel centers for fixed-boundary F32 cavity fans.
Boundary geometry, native thresholds, exact sign/volume and intersection gates
are inherited unchanged. This is static compiler work, never per-step physics.
"""
from pathlib import Path
import importlib.util,sys,numpy as np,argparse,json,time,hashlib
from scipy.optimize import linprog
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("fan248for249",H/"remesh_cell_disjoint_kernel_fans_248_root.py")
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
oldfan=m.fan
oldcenters=m.kernel.kernel_centers
calls=0;stats={"adaptive_attempts":0,"adaptive_native_centers":0,"adaptive_passing_fans":0}
def adaptive_centers(lp,lt,ri,lower,upper,oldgood):
 root=lp[ri].astype(float);scale=float(np.max(upper-lower))
 if scale<=0:return []
 slots=np.argmax(lt==ri,axis=1);opp=np.asarray([tet[m.m.m.PAT[k]] for tet,k in zip(lt,slots)],np.int64)
 xyz=lp[opp].astype(float);normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);norm=np.linalg.norm(normal,axis=1)
 if np.any(norm<=0):return []
 unit=normal/norm[:,None];d0=np.einsum("ij,ij->i",normal,xyz[:,0]-root)
 fixededge=np.maximum.reduce([np.linalg.norm(xyz[:,i]-xyz[:,j],axis=1) for i,j in ((0,1),(0,2),(1,2))])
 coeff=1e-5*(1+1e-6)/norm
 minimum=np.maximum(6e-18*(1+1e-6)/norm,coeff*fixededge**3)
 A=[np.c_[unit,np.ones(len(lt))]];B=[(d0/norm-minimum)/scale]
 bounds=[(float((lower[j]-root[j])/scale),float((upper[j]-root[j])/scale)) for j in range(3)]+[(0,None)]
 u=np.zeros(3)
 for iteration in range(12):
  # ||x-v||^3 is convex. Its tangent is a necessary linear constraint;
  # iterate outer approximations, then verify the original inequalities.
  delta=(root+scale*u)[None,None,:]-xyz;length=np.linalg.norm(delta,axis=2)
  val=coeff[:,None]*length**3/scale
  grad=3*coeff[:,None,None]*length[:,:,None]*delta
  rows=unit[:,None,:]+grad
  A.append(np.c_[rows.reshape(-1,3),np.ones(len(lt)*3)])
  B.append(np.repeat(d0/norm/scale,3)+np.einsum("fkj,j->fk",grad,u).ravel()-val.ravel())
  sol=linprog([0,0,0,-1],A_ub=np.vstack(A),b_ub=np.concatenate(B),bounds=bounds,method="highs",options={"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10})
  if not sol.success:return []
  u=sol.x[:3];point=root+scale*u
  edge=np.maximum(fixededge,np.linalg.norm(point[None,None,:]-xyz,axis=2).max(axis=1))
  req=np.maximum(6e-18*(1+1e-6)/norm,coeff*edge**3)
  height=d0/norm-scale*(unit@u)
  if np.all(height>=req):
   return [("adaptive_convex_all_native",point,float(np.min(height-req)))]
 return []
def extended_fan(p,old):
 global calls
 result,why=oldfan(p,old)
 if result is not None or why!="no_F32_native_defect_descent":return result,why
 calls+=1;stats["adaptive_attempts"]+=1
 def centers(*args):
  cs=adaptive_centers(*args)
  if cs:stats["adaptive_native_centers"]+=1
  return cs
 m.kernel.kernel_centers=centers
 try:ans,reason=oldfan(p,old)
 finally:m.kernel.kernel_centers=oldcenters
 if ans is not None:stats["adaptive_passing_fans"]+=1;return ans,None
 return None,"adaptive_"+reason
oldsignature=m.signature
def signature(p,t,material):
 return hashlib.sha256((oldsignature(p,t,material)+m.m.sha(Path(__file__))).encode()).hexdigest()
m.fan=extended_fan;m.signature=signature
def fixture(out):
 out.mkdir(exist_ok=False,parents=True)
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,9e-7],[.002,.002,2e-7]],np.float32)
 old=np.c_[np.full(4,4),m.m.boundary(m.m.m.orient(p,[[0,1,2,3]]))]
 vv=np.unique(m.m.boundary(old));lower=p[vv].astype(float).min(0);upper=p[vv].astype(float).max(0)
 ordinary=oldcenters(p,old,4,lower,upper,np.zeros(4,bool))
 assert not any(x[0]=="all_native" for x in ordinary)
 cs=adaptive_centers(p,old,4,lower,upper,np.zeros(4,bool));assert cs
 pp=p.copy();pp[4]=cs[0][1].astype(np.float32)
 assert np.all(m.m.m.passes(*m.m.m.measurements(pp,old)))
 # Parent volume cannot accommodate four native minimum-volume children.
 tiny=p*np.float32(1e-4)
 assert not adaptive_centers(tiny,old,4,tiny[vv].min(0).astype(float),tiny[vv].max(0).astype(float),np.zeros(4,bool))
 ans,why=extended_fan(p,old);assert ans is not None,why
 assert ans[2]["new_native_failures"]==0
 r={"status":"PASS_ADAPTIVE_CONVEX_KERNEL_ACTUAL_PATH","physical_admission":False,"native_steps":0,"controls":["fixed_box_bound_false_negative_recovered","authoritative_F32_native_check","minimum_child_volume_infeasible_rejected","inherited_exact_boundary_volume_self_intersection"]}
 (out/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=50);ap.add_argument("--maximum-attempts",type=int,default=16000);ap.add_argument("--maximum-points",type=int,default=200);a=ap.parse_args()
 if a.fixture:fixture(a.output)
 else:
  m.run(a)
  f=a.output/"report.json";r=json.loads(f.read_text());r["adaptive_kernel"]=stats;r["adaptive_source_sha256"]=m.m.sha(Path(__file__));f.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(stats))
