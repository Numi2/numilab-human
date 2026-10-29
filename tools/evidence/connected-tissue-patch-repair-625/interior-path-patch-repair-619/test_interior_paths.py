from pathlib import Path
import importlib.util,sys,numpy as np,json,hashlib
P=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("paths619",P/"interior_paths.py")
m=load("multi619",P.parent/"multiple-boundary-patch-repair-617/multi_boundary.py")
dp=load("dp619",P.parent/"constraint-disk-triangulation-616/constraint_disk.py")
c=load("chord619",P.parent/"exact-chord-pruning-618/chord_pruning.py")
ears=load("ears619",P.parent/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=ears.load_exact_owner()
points=np.asarray([[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0],[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]],np.float32)
loops=[[0,1,2,3],[4,7,6,5]];old=[]
for a,b in [([0,1,2,3],[8,9,10,11]),([8,9,10,11],[4,5,6,7])]:
 for i in range(4):
  j=(i+1)%4;old.extend([(a[i],a[j],b[j]),(a[i],b[j],b[i])])
boundary={tuple(sorted((a,b)))for l in loops for a,b in zip(l,l[1:]+l[:1])}
keys=tuple(pred.float32_point_lattice_key(v)for v in points)
tris,recs,info=h.triangulate_with_interior_paths(loops,np.asarray(old),points,boundary,pred._prepare_surface_aabb(()),pred,keys,1000000,chord_pruning=c,dp_disk=dp.collision_aware_dp_disk,identity_proxy=m.PhysicalIdentityPredicate,max_seconds=10)
assert info["topology"]["interior_vertices"]>=1 and info["topology"]["euler_characteristic"]==0
area=sum(float(np.cross(points[b]-points[a],points[d]-points[a])[2])*.5 for a,b,d in tris)
assert area==32,(area,tris)
try:h.validate_patch(tris+[tris[0]],loops,set(range(12)),boundary)
except ValueError:pass
else:raise AssertionError("duplicate physical face accepted")
result={"schema":"certified-interior-path-patch-regression.v1","status":"PASS","helper_sha256":hashlib.sha256((P/"interior_paths.py").read_bytes()).hexdigest(),"fixture":"annulus-with-two-edge-graph-only-route","oriented_area":area,"expected_area":32,"details":info,"duplicate_negative_control":True,"physical_admission":False}
(P/"regression.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":"PASS","seconds":info["elapsed_s"],"paths":info["paths"],"topology":info["topology"]}))
