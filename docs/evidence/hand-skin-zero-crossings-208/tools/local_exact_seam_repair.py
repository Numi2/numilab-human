from pathlib import Path
from fractions import Fraction
from itertools import product
import sys,json,time,hashlib,argparse,numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from verify_candidate import module
ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seconds',type=float,default=120);a=ap.parse_args();a.out.mkdir(exist_ok=False);start=time.monotonic()
z=np.load(a.candidate);data={k:z[k].copy()for k in z.files};X=data['points_world_m'];Y=X.copy();F=data['source_O_faces'];ar=json.load(open(a.audit));pairs=np.asarray([p['face_rows']for p in ar['triangle_pairs']],dtype=int);core=np.unique(F[pairs]);src=np.load(R.parent/'publication-206-stage-001/inputs/baseline-806.npz')['points_world_m'];pins={tuple(p.view(np.uint32))for p in src};near=cKDTree(X[core].astype(float)).query_pairs(1e-6);near_nodes={int(core[i])for p in near for i in p};involvement=np.bincount(F[pairs].ravel(),minlength=len(Y));nodes=sorted([int(i)for i in core if tuple(X[i].view(np.uint32))not in pins],key=lambda i:(i not in near_nodes,-involvement[i],i));pred=module('seampredicate',R/'boolean-prototype-001/audit_self_envelope006.py').load_exact();points=[tuple(Fraction.from_float(float(v))for v in p)for p in Y];incident=[[]for _ in Y]
for i,f in enumerate(F):
 for v in f:incident[v].append(i)
changes=[];log=[]
for sweep in range(3):
 improved=0
 for node in nodes:
  if time.monotonic()-start>a.seconds:break
  touch=np.array(incident[node],dtype=int);tri=Y[F].astype(float);lo=tri[touch].min((0,1))-3e-6;hi=tri[touch].max((0,1))+3e-6;scope=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1));sf=F[scope];selected=set(np.flatnonzero(np.isin(scope,touch)).tolist());ref=np.cross(tri[touch,1]-tri[touch,0],tri[touch,2]-tri[touch,0]);base=Y[node].copy()
  def check():
   records=pred._records(points,sf);return pred._audit_self_pairs_touching_faces(records,selected)['count']
  before=check()
  if before==0:continue
  normal=ref.sum(0);normal/=max(np.linalg.norm(normal),1e-30);opts=[]
  for scale in [2e-7,-2e-7,1e-6,-1e-6]:opts.append((base.astype(float)+scale*normal).astype(np.float32))
  for distance in [1,4,16]:
   for steps in product([-distance,0,distance],repeat=3):
    if steps==(0,0,0):continue
    v=base.copy()
    for axis,s in enumerate(steps):
     for _ in range(abs(s)):v[axis]=np.nextafter(v[axis],np.float32(np.inf if s>0 else -np.inf))
    opts.append(v)
  best=before;chosen=None
  for value in opts:
   if time.monotonic()-start>a.seconds:break
   if any(np.array_equal(value,Y[j])for j in np.unique(sf)if j!=node):continue
   test=Y[F[touch]].astype(float);test[F[touch]==node]=value.astype(float);N=np.cross(test[:,1]-test[:,0],test[:,2]-test[:,0]);dots=np.einsum('ij,ij->i',N,ref)
   if not np.all(dots>0):continue
   Y[node]=value;points[node]=tuple(Fraction.from_float(float(v))for v in value);count=check()
   if count<best:best=count;chosen=value.copy()
   if best==0:break
  Y[node]=base;points[node]=tuple(Fraction.from_float(float(v))for v in base)
  if chosen is not None:
   Y[node]=chosen;points[node]=tuple(Fraction.from_float(float(v))for v in chosen);improved+=1;row={'node':node,'before':before,'after':best,'move_um':float(np.linalg.norm(chosen.astype(float)-base))*1e6};changes.append(row);print(json.dumps(row),flush=True)
 log.append({'sweep':sweep,'improved_nodes':improved,'elapsed_s':time.monotonic()-start})
 if not improved or time.monotonic()-start>a.seconds:break
data['points_world_m']=Y;np.savez_compressed(a.out/'candidate.npz',**data);report={'candidate_sha256':hashlib.sha256((a.out/'candidate.npz').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'changes':changes,'sweeps':log,'potential_nodes':len(nodes),'max_movement_m':float(np.linalg.norm(Y.astype(float)-X.astype(float),axis=1).max()),'full_exact_audit_required':True,'native_admission':False,'physical_steps':0};(a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='changes'}),flush=True)
