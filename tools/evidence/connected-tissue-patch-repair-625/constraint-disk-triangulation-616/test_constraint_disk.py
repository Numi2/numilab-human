from pathlib import Path
from functools import lru_cache
import importlib.util,sys,numpy as np,itertools,json,hashlib,time
P=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("dp616_test",P/"constraint_disk.py")
r=load("ears616_test",P.parent/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=r.load_exact_owner();empty=pred._prepare_surface_aabb(())
rows=[]
def run(name,xyz,external=set()):
 xyz=np.asarray(xyz,np.float32);loop=list(range(len(xyz)));keys=tuple(pred.float32_point_lattice_key(v)for v in xyz)
 tris,recs,info=h.collision_aware_dp_disk(loop,xyz.astype(float),xyz,external,empty,pred,keys,[],1000000)
 assert pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)["count"]==0
 assert h.validate_disk(tris,loop,external)["euler_characteristic"]==1
 rows.append({"fixture":name,"status":"PASS","exact_self_pairs":0,"details":info})
 return xyz,keys,tris,info
run("square",[[0,0,0],[2,0,0],[2,2,0],[0,2,0]])
run("concave",[[0,0,0],[3,0,0],[3,3,0],[1.5,1,0],[0,3,0]])
_,_,tris,_=run("retained_diagonal",[[0,0,0],[2,0,0],[2,2,0],[0,2,0]],{(0,2)})
assert not any(0 in f and 2 in f for f in tris)
fixture=json.loads((P/"branch-fixture.json").read_text())
xyz,keys,tris,info=run("nonplanar_conflict",fixture["points"])
assert info["statistics"]["branches_popped"]>1 and info["statistics"]["self_pair_conflicts"]>0
@lru_cache(None)
def alltri(i,j):
 if j-i<2:return ((),)
 return tuple(left+right+((i,k,j),)for k in range(i+1,j)for left in alltri(i,k)for right in alltri(k,j))
scale=float(np.ptp(xyz.astype(float),axis=0).max())
def score(ts):
 total=0.
 for t in ts:
  v=xyz[list(t)].astype(float);a=v[1]-v[0];b=v[2]-v[0];area2=float(np.linalg.norm(np.cross(a,b)))
  lens=float(np.dot(a,a)+np.dot(b,b)+np.dot(a-b,a-b))
  total+=area2/(scale*scale)+1e-6*min(lens/(area2+1e-300),1e6)
 return total
valid=[]
for ci,ts in enumerate(alltri(0,len(xyz)-1)):
 recs=pred._records(keys,ts)
 if not pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)["count"]:valid.append(score(ts))
assert len(alltri(0,7))==132 and valid
assert abs(score(tris)-min(valid))<1e-12,(score(tris),min(valid))
rows[-1]["exhaustive_triangulations"]=132;rows[-1]["exhaustive_valid"]=len(valid);rows[-1]["optimal_cost_parity"]=True
# A distinct retained sheet occupying the disk center must make every triangulation fail.
xyz=np.asarray([[0,0,0],[2,0,0],[2,2,0],[0,2,0]],np.float32);keys=tuple(pred.float32_point_lattice_key(v)for v in xyz)
sheet=np.asarray([[.9,.9,0],[1.1,.9,0],[1.,1.1,0]],np.float32)
rec=pred._records(tuple(pred.float32_point_lattice_key(v)for v in sheet),[(0,1,2)])[0]
barrier=(rec[0],rec[1],rec[2],-1,(4,5,6))
try:h.collision_aware_dp_disk([0,1,2,3],xyz,xyz,set(),pred._prepare_surface_aabb([barrier]),pred,keys,[],1000000)
except ValueError:rows.append({"fixture":"fully_blocked_distinct_surface","status":"PASS","rejection_expected":True})
else:raise AssertionError("occupied disk accepted")
for loop in ([0,1],[0,1,1,3]):
 try:h.collision_aware_dp_disk(loop,xyz,xyz,set(),empty,pred,keys,[],1000000)
 except ValueError:pass
 else:raise AssertionError("invalid boundary accepted")
result={"schema":"exact-constraint-disk-regression.v1","status":"PASS","helper_sha256":hashlib.sha256((P/"constraint_disk.py").read_bytes()).hexdigest(),
 "test_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"exact_owner_sha256":hashlib.sha256(Path(pred.__file__).read_bytes()).hexdigest(),
 "fixtures":rows,"invalid_boundary_negative_controls":2,"physical_admission":False}
(P/"regression.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":"PASS","fixtures":len(rows),"exhaustive_triangulations":132,"valid":len(valid),"conflict_dp_solves":info["statistics"]["dp_solves"]}))
