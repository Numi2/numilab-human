"""Bounded cluster-segment construction for unresolved D endpoint transfers.
All positions remain within original contracted endpoint bounds. Exact geometry,
typed ancestry, registration/mass and native admission are separate gates.
"""
from pathlib import Path
from itertools import combinations,product
from collections import Counter
import numpy as np,json,argparse,time,importlib.util
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prev=load("prev223",H/"choose_volume_aware_endpoints_223_root.py");m=prev.m
def segment_interval(d0,d1,req):
 lo,hi=0.,1.
 for a,b,r in zip(d0,d1,req):
  slope=b-a
  if slope>0:lo=max(lo,float((r-a)/slope))
  elif slope<0:hi=min(hi,float((r-a)/slope))
  elif a<r:return None
  if lo>=hi:return None
 return max(lo,0.),min(hi,1.)
def fixture():
 assert segment_interval(np.array([-1.,2.]),np.array([2.,-1.]),np.zeros(2))==(1/3,2/3)
 assert segment_interval(np.array([-1.,-1.]),np.array([-2.,-1.]),np.zeros(2)) is None
 assert segment_interval(np.array([1.]),np.array([1.]),np.array([2.])) is None
 return {"status":"PASS_CLUSTER_SEGMENT_INTERVAL_CONTROLS","native_steps":0,"physical_admission":False}
def run(a):
 a.output.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v["points_world_m"].copy();original=p.copy();t=v["tetrahedra"];b=v["desired_boundary_faces"];mp=v["input_node_quotient"]
 lookup={m.key(x):i for i,x in enumerate(p)}
 with np.load(a.d_surface) as z:dp=z["points_world_m"];df=z["faces"]
 with np.load(a.i_surface) as z:ii=z["points_world_m"];iff=z["faces"]
 eligible={lookup[m.key(x)] for x in dp[np.unique(df)]}-{lookup[m.key(x)] for x in ii[np.unique(iff)]}
 flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 sort=np.argsort(mp,kind="stable");counts=np.bincount(mp,minlength=len(p));coff=np.r_[0,np.cumsum(counts)]
 dd,qq=m.m.measurements(p,t);before=int(sum(dd<=0));beforeq=int(sum(~m.m.passes(dd,qq)))
 roots=[int(x) for x in np.unique(t[dd<=0]) if x in eligible and counts[x]>1]
 ops=[];rej=Counter();trials=0
 for root in roots:
  rows=np.unique(order[off[root]:off[root+1]]//4);br=np.flatnonzero(np.any(b==root,axis=1));nodes=np.unique(np.r_[t[rows].ravel(),b[br].ravel()])
  lk={int(v):i for i,v in enumerate(nodes)};lt=np.asarray([[lk[int(x)] for x in tet] for tet in t[rows]],np.int64);lb=np.asarray([[lk[int(x)] for x in f] for f in b[br]],np.int64)
  lp=p[nodes].copy();ri=lk[root];od,oq=m.m.measurements(lp,lt);og=m.m.passes(od,oq);oneg=od<=0
  if not np.any(oneg):continue
  fq,fn=prev.face_measure(lp,lb);fg=fq>=1e-5;members=sort[coff[root]:coff[root+1]]
  pairs=list(combinations(map(int,members),2));best=None;seen=set()
  lower=original[members].min(0);upper=original[members].max(0)
  for u,w in pairs:
   lp[ri]=original[u];d0,q0=m.m.measurements(lp,lt)
   lp[ri]=original[w];d1,q1=m.m.measurements(lp,lt)
   # Norms along a segment are convex, so endpoint edge maxima bound every trial.
   xyz0=lp[lt].astype(float);lp[ri]=original[u];xyz1=lp[lt].astype(float)
   maxedge=np.zeros(len(lt))
   for i,j in combinations(range(4),2):maxedge=np.maximum(maxedge,np.maximum(np.linalg.norm(xyz0[:,i]-xyz0[:,j],axis=1),np.linalg.norm(xyz1[:,i]-xyz1[:,j],axis=1)))
   req=np.where(og,np.maximum(6e-18*(1+1e-6),1e-5*maxedge**3*(1+1e-6)),np.maximum(6e-27,64*np.finfo(float).eps*maxedge**3))
   interval=segment_interval(d0,d1,req)
   if interval is None:rej["empty_positive_segment_interval"]+=1;continue
   lo,hi=interval
   for frac in (.5,.25,.75):
    alpha=lo+(hi-lo)*frac;planned=original[u].astype(float)*(1-alpha)+original[w].astype(float)*alpha;center=planned.astype(np.float32)
    axes=[[np.nextafter(center[j],np.float32(-np.inf)),center[j],np.nextafter(center[j],np.float32(np.inf))] for j in range(3)]
    for values in product(*axes):
     point=np.asarray(values,np.float32);key=m.key(point)
     if key in seen:continue
     seen.add(key)
     if np.any(point<lower) or np.any(point>upper):continue
     trials+=1;lp[ri]=point;nd,nq=m.m.measurements(lp,lt);ng=m.m.passes(nd,nq);nneg=nd<=0
     if np.any(~oneg&nneg) or sum(nneg)>=sum(oneg):rej["nonpositive_not_reduced"]+=1;continue
     if np.any(og&~ng) or sum(~ng)>sum(~og):rej["volume_quality_regression"]+=1;continue
     nf,nn=prev.face_measure(lp,lb)
     if np.any(nf<=0) or np.any(fg&(nf<1e-5)) or np.any(np.einsum("ij,ij->i",fn,nn)<=0):rej["boundary_shape_regression"]+=1;continue
     ip=m.m.integer_points(lp)
     signs=np.asarray([(lambda d:(d>0)-(d<0))(m.m.exact_det(ip,tet)) for tet in lt],np.int8)
     if np.any((~nneg)&(signs<=0)) or np.any(~oneg&(signs<=0)):rej["exact_sign"]+=1;continue
     score=(int(sum(nneg)),int(sum(~ng)),-float(nq.min()),float(np.linalg.norm(point.astype(float)-p[root])))
     if best is None or score<best[0]:best=(score,point.copy(),u,w,float(alpha),planned.copy(),nd,nq)
  if best is None:continue
  score,point,u,w,alpha,planned,nd,nq=best;old=p[root].copy();p[root]=point
  ops.append({"current_node":root,"position_before_world_m":old.tolist(),"position_after_world_m":point.tolist(),"source_segment_original_nodes":[u,w],"source_segment_alpha":alpha,"planned_double_point_world_m":planned.tolist(),"F32_rounding_m":float(np.linalg.norm(point.astype(float)-planned)),"displacement_m":float(np.linalg.norm(point.astype(float)-old)),"affected_cell_rows":rows.tolist(),"affected_desired_face_rows":br.tolist(),"old_nonpositive":int(sum(oneg)),"new_nonpositive":int(sum(nd<=0)),"old_quality_failures":int(sum(~og)),"new_quality_failures":int(sum(~m.m.passes(nd,nq)))})
 dd,qq=m.m.measurements(p,t);out=a.output/"cluster-segment-volume-and-boundary-proposal.npz"
 v["points_world_m"]=p;v["source_points_before_segment"]=original;v["changed_point_rows"]=np.asarray([o["current_node"] for o in ops],np.int64)
 np.savez_compressed(out,**v)
 r={"status":"SEGMENT_POSITION_PROPOSAL_REQUIRES_EXACT_GEOMETRY_AND_ANCESTRY","physical_admission":False,"native_steps":0,"input":str(a.input),"input_sha256":m.sha(a.input),"script_sha256":m.sha(Path(__file__)),"output":str(out),"output_sha256":m.sha(out),"clusters":len(roots),"trials":trials,"passing_operations":len(ops),"nonpositive_before":before,"nonpositive_after":int(sum(dd<=0)),"quality_failures_before":beforeq,"quality_failures_after":int(sum(~m.m.passes(dd,qq))),"rejections":dict(rej),"elapsed_s":time.monotonic()-tic,"operations":ops}
 (a.output/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="operations"},indent=2))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--fixture",action="store_true");ap.add_argument("--input",type=Path);ap.add_argument("--d-surface",type=Path);ap.add_argument("--i-surface",type=Path);ap.add_argument("--output",type=Path);a=ap.parse_args()
 if a.fixture:print(json.dumps(fixture()))
 else:run(a)
