#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
from itertools import product
import importlib.util,hashlib,json,sys,time
import numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent
P=R/'envelope-005-all18-stable/candidate.npz';OUT=R/'envelope-006-local-lattice';OUT.mkdir(exist_ok=False)
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
A=mod('seam_audit',R.parent.parent/'audit_skin_bone_surfaces_159.py');ex=A.exact_module();t=time.monotonic();z=np.load(P,allow_pickle=False);payload={k:z[k].copy() for k in z.files};X=payload['points_world_m'];Y=X.copy();F=payload['source_O_faces'];src=np.load(R.parent/'publication-206-stage-001/inputs/baseline-806.npz')['points_world_m'];pins={tuple(v.view(np.uint32)) for v in src}
a=json.load(open(R/'boolean-prototype-001/audit-envelope-005-self.json'));pairs=np.array([r['face_rows'] for r in a['triangle_pairs']],dtype=np.int64);core=np.unique(F[pairs.ravel()]);close=list(cKDTree(X[core].astype(float)).query_pairs(5e-7));nodes=sorted({int(core[i]) for pair in close for i in pair if tuple(X[core[i]].view(np.uint32)) not in pins})
bb=X[F[pairs.ravel()]].astype(float);lo=bb.min((0,1))-.001;hi=bb.max((0,1))+.001;tri=X[F].astype(float);focus=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1));f=F[focus]
points=[tuple(Fraction.from_float(float(v)) for v in p) for p in Y]
def check():
 records=ex._records(points,f);r=ex._audit_self_pairs_touching_faces(records,set(range(len(f))));return {(int(i),int(j)) for i,j in r['triangle_pairs']}
current=check();print(json.dumps({'stage':'initial','pairs':len(current),'focus_faces':len(focus),'nodes':nodes,'time':time.monotonic()-t}),flush=True)
trials=[];changes=[]
for sweep in range(4):
 improved=False
 for node in nodes:
  base=Y[node].copy();best=len(current);choice=None;opts=[]
  for steps in product([-1,0,1],repeat=3):
   if steps==(0,0,0):continue
   v=base.copy()
   for axis,s in enumerate(steps):
    if s:v[axis]=np.nextafter(v[axis],np.float32(np.inf if s>0 else -np.inf))
   if any(np.array_equal(v,Y[j]) for j in nodes if j!=node):continue
   opts.append((float(np.sum((v.astype(float)-base.astype(float))**2)),steps,v))
  for _,steps,v in sorted(opts,key=lambda x:(x[0],x[1])):
   Y[node]=v;points[node]=tuple(Fraction.from_float(float(x)) for x in v);res=check();count=len(res)
   trials.append({'sweep':sweep,'node':node,'steps':steps,'pairs':count})
   if count<best:best=count;choice=(v.copy(),res,steps)
   if best==0:break
  Y[node]=base;points[node]=tuple(Fraction.from_float(float(x)) for x in base)
  if choice:
   v,current,steps=choice;Y[node]=v;points[node]=tuple(Fraction.from_float(float(x)) for x in v);changes.append({'node':node,'steps':steps,'pairs':len(current),'distance_m':float(np.linalg.norm(v.astype(float)-X[node].astype(float)))});improved=True
   print(json.dumps({'stage':'improved','node':node,'pairs':len(current),'elapsed_s':time.monotonic()-t}),flush=True)
  if not current:break
 if not current or not improved:break
payload['points_world_m']=Y;np.savez_compressed(OUT/'candidate.npz',**payload)
report={'input_candidate_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((OUT/'candidate.npz').read_bytes()).hexdigest(),'baseline_local_self_pairs':len(pairs),'final_local_self_pairs':len(current),'changes':changes,'trials':trials,'focus_faces':focus.tolist(),'movable_nodes':nodes,'native_admission':False,'full_audit_pending':True,'elapsed_s':time.monotonic()-t};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['trials','focus_faces']}),flush=True)
