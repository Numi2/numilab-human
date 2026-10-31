"""Defect-budgeted exact-boundary cavity construction using existing SciPy/HiGHS.
MILP proposes a connectivity; exact F32 signs, chain boundary, volume and the
retained exact intersection owner certify it. Does not allocate physical mass."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys,time,importlib.util
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
QUALITY_OWNER_PATH=P/'fine-physical-volume-706/fixed_boundary_volume_flip_198_root.py'
_QUALITY_SPEC=importlib.util.spec_from_file_location('quality_owner_198_for_cavity220',QUALITY_OWNER_PATH)
if _QUALITY_SPEC is None or _QUALITY_SPEC.loader is None:raise RuntimeError('cannot load pinned 198 native-quality owner')
QUALITY_OWNER=importlib.util.module_from_spec(_QUALITY_SPEC)
sys.modules[_QUALITY_SPEC.name]=QUALITY_OWNER
_QUALITY_SPEC.loader.exec_module(QUALITY_OWNER)
def native_quality_mask(points,tetrahedra):
 det,normalized=QUALITY_OWNER.measurements(points,tetrahedra)
 return QUALITY_OWNER.passes(det,normalized),det,normalized
PAT=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED_QUALITY_OWNER_SHA256='650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0'
if sha(QUALITY_OWNER_PATH)!=EXPECTED_QUALITY_OWNER_SHA256:raise RuntimeError('native-quality owner hash changed; refusing unpinned thresholds')
def ip(x):
 return tuple(int(float(v).as_integer_ratio()[0])<<(149-(float(v).as_integer_ratio()[1].bit_length()-1))for v in x)
def triple(a,b,c):
 return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
def det(a,b,c,d):return triple(tuple(b[k]-a[k]for k in range(3)),tuple(c[k]-a[k]for k in range(3)),tuple(d[k]-a[k]for k in range(3)))
def signs(f):return np.where(((f[:,0]>f[:,1]).astype(int)+(f[:,0]>f[:,2])+(f[:,1]>f[:,2]))%2==0,1,-1)
def codes(f,n):
 f=np.sort(f,axis=1);return (f[:,0]*n+f[:,1])*n+f[:,2]
def cyclic(f):return min(tuple(f),tuple(np.roll(f,1)),tuple(np.roll(f,2)))
def write_proposal(path,p,t,b,material):
 np.savez_compressed(path,points_world_m=p,tetrahedra=t,boundary_faces=b,material_indices=np.full(len(t),int(material),np.int32))
 return sha(path)
def solve(p,b,seconds=20,max_points=36,material=15,maximum_native_failures=None):
 if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite F32 points required')
 n=len(p)
 if not n or n>max_points:raise ValueError('bounded vertex limit '+str(n))
 b=np.asarray(b,np.int64)
 if b.ndim!=2 or b.shape[1]!=3 or np.any(b<0)or np.any(b>=n):raise ValueError('bad faces')
 edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1)
 if np.any(np.unique(edges,axis=0,return_counts=True)[1]!=2):raise ValueError('closed manifold boundary required')
 xyz=[ip(x)for x in p];boundary_volume=sum(triple(*(xyz[int(v)]for v in f))for f in b)
 if boundary_volume<=0:raise ValueError('positive oriented cavity boundary required')
 candidates=np.array(list(itertools.combinations(range(n),4)),np.int64)
 q=p[candidates].astype(np.float64);u,v,w=(q[:,j]-q[:,0]for j in (1,2,3))
 ds=np.einsum('ij,ij->i',u,np.cross(v,w))
 uncertain=np.flatnonzero(np.abs(ds)<=np.finfo(float).eps*64*np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1))
 for k in uncertain:
  d=det(*(xyz[int(v)]for v in candidates[k]));ds[k]=1e-300 if d>0 else -1e-300 if d<0 else 0.
 keep=ds!=0;candidates=candidates[keep];ds=ds[keep]
 neg=ds<0;candidates[neg,0],candidates[neg,1]=candidates[neg,1].copy(),candidates[neg,0].copy()
 generated_candidate_count=len(candidates)
 native_ok,native_det,native_normalized=native_quality_mask(p,candidates)
 rejected_native_quality=int(np.count_nonzero(~native_ok))
 if maximum_native_failures is not None:
  if type(maximum_native_failures) is not int or maximum_native_failures<0:raise ValueError("nonnegative integer defect budget required")
  retained=(native_det>0)&np.isfinite(native_det)&np.isfinite(native_normalized)
 else:retained=native_ok
 defect=(~native_ok)[retained]
 candidates=candidates[retained];ds=ds[retained]
 if len(candidates)==0:
  return {'status':'NO_NATIVE_QUALITY_ADMISSIBLE_CANDIDATE_TETS',
          'generated_candidate_tetrahedra':generated_candidate_count,
          'native_quality_admissible_candidates':0,
          'native_quality_rejected_candidates':rejected_native_quality if maximum_native_failures is None else 0,
         'native_quality_ineligible_candidates':rejected_native_quality,
          'native_quality_owner_sha256':sha(QUALITY_OWNER_PATH),
          'physical_admission':False,'runtime':0},None
 # Build the matrix only after filtering, so its columns exactly match MILP variables.
 tris=np.array(list(itertools.combinations(range(n),3)),np.int64);tc=codes(tris,n)
 f=candidates[:,PAT].reshape(-1,3);fr=np.searchsorted(tc,codes(f,n));sg=signs(f)
 A=coo_matrix((sg.astype(float),(fr,np.repeat(np.arange(len(candidates)),4))),shape=(len(tris),len(candidates))).tocsc()
 target=np.zeros(len(tris));br=np.searchsorted(tc,codes(b,n))
 if len(np.unique(br))!=len(b):raise ValueError('duplicate source boundary face')
 target[br]=signs(b)
 q=p[candidates].astype(np.float64)
 edge=max(np.max(np.linalg.norm(q[:,i]-q[:,j],axis=1))for i,j in itertools.combinations(range(4),2))
 quality=np.abs(ds)/max(edge**3,1e-300);cost=1+1e-3/(1+1e3*quality)+1000*defect
 constraints=[LinearConstraint(A,target,target)]
 if maximum_native_failures is not None:
  constraints.append(LinearConstraint(defect.astype(float)[None,:],0,maximum_native_failures))
 start=time.monotonic()
 r=milp(cost,integrality=np.ones(len(candidates),np.uint8),bounds=Bounds(0,1),constraints=constraints,options={'time_limit':seconds,'mip_rel_gap':.05,'presolve':True})
 report={'milp_status':int(r.status),'milp_message':str(r.message),
         'generated_candidate_tetrahedra':generated_candidate_count,
         'native_quality_rejected_candidates':rejected_native_quality if maximum_native_failures is None else 0,
         'native_quality_ineligible_candidates':rejected_native_quality,
         'native_quality_admissible_candidates':int(sum(~defect)),
         'candidate_tetrahedra':len(candidates),'constraint_faces':len(tris),'material_index':int(material),
         'native_quality_owner_path':str(QUALITY_OWNER_PATH),
         'native_quality_owner_sha256':sha(QUALITY_OWNER_PATH),
         'native_quality_filter':('strict all-native prefilter' if maximum_native_failures is None else 'positive candidates; explicit native-defect-count MILP constraint; final native admission unchanged'),
         'construction_maximum_native_failures':maximum_native_failures,
         'candidate_native_defects':int(sum(defect)),
         'strict_native_candidate_count':int(sum(~defect)),
         'milp_wall_s':time.monotonic()-start,'physical_admission':False,'runtime':0}
 if r.x is None:return dict(report,status='NO_FEASIBLE_CONNECTIVITY_WITHIN_BOUND'),None
 selected=np.rint(r.x).astype(np.int8)
 if np.any(np.abs(selected-r.x)>1e-6)or not np.array_equal(A@selected,target):return dict(report,status='REJECT_NONINTEGRAL_OR_FACE_BALANCE'),None
 t=candidates[selected!=0]
 selected_native_failures=int(sum(~native_quality_mask(p,t)[0]))
 if maximum_native_failures is None:
  if selected_native_failures:return dict(report,status='REJECT_STRICT_NATIVE_QUALITY'),None
 elif selected_native_failures>maximum_native_failures:return dict(report,status='REJECT_NATIVE_DEFECT_BUDGET'),None
 report['selected_native_failures']=selected_native_failures
 values=[det(*(xyz[int(v)]for v in tet))for tet in t]
 if not values or min(values)<=0 or sum(values)!=boundary_volume:return dict(report,status='REJECT_EXACT_SIGN_OR_VOLUME'),None
 faces=t[:,PAT].reshape(-1,3);_,ix,ct=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
 if np.any(ct>2)or sorted(cyclic(x)for x in faces[ix[ct==1]])!=sorted(cyclic(x)for x in b):return dict(report,status='REJECT_EXACT_ORIENTED_BOUNDARY'),None
 owner=P/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py'
 sys.path.insert(0,str(owner.parent.parent))
 exact=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
 rec,_=exact.float32_surface_records(p,faces[ix]);gate=exact._audit_pair(rec,rec,same_surface=True)
 if gate['count']:return dict(report,status='REJECT_EXACT_INTERNAL_FACE_CROSSING',crossings=gate['count']),None
 return dict(report,status='PASS_EXACT_ORIENTED_CAVITY_TETRAHEDRALIZATION',output_tetrahedra=len(t),exact_volume_conserved=True,exact_boundary_preserved=True,exact_internal_face_crossings=0,exact_owner_sha256=sha(owner)),t
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--shell',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seconds',type=float,default=20);ap.add_argument('--max-points',type=int,default=36);ap.add_argument('--material',type=int,required=True);a=ap.parse_args()
 a.out.mkdir(exist_ok=False)
 with np.load(a.shell,allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
 p=arrays['points_world_m'];b=arrays['faces'];start=time.monotonic()
 try:r,t=solve(p,b,a.seconds,a.max_points,a.material)
 except Exception as e:r={'status':'CAVITY_SOLVE_FAILED','reason':repr(e),'physical_admission':False,'runtime':0};t=None
 r.update(source_path=str(a.shell),source_sha256=sha(a.shell),source_vertices=len(p),source_faces=len(b),material_index=a.material,
          native_quality_owner_path=str(QUALITY_OWNER_PATH),native_quality_owner_sha256=sha(QUALITY_OWNER_PATH),
          elapsed_s=time.monotonic()-start)
 if t is not None:
  output=a.out/'cavity-positive-proposal.npz'
  r['proposal_path']=str(output);r['proposal_sha256']=write_proposal(output,p,t,b,a.material)
 (a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
