from pathlib import Path
import importlib.util,sys,numpy as np,itertools,json,hashlib,time
P=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("dp616",P/"constraint_disk.py")
r=load("ears616",P.parent/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=r.load_exact_owner()
empty=pred._prepare_surface_aabb(())
rows=[];branchfixture=None
for seed in range(128):
 xyz=np.random.default_rng(seed).uniform(-1,1,size=(8,3)).astype(np.float32)
 lattice=tuple(pred.float32_point_lattice_key(v)for v in xyz)
 try:
  tris,recs,info=h.collision_aware_dp_disk(list(range(8)),xyz.astype(float),xyz,set(),empty,pred,lattice,[],1000000,max_seconds=1,max_branches=128)
 except (ValueError,TimeoutError):continue
 if info["statistics"]["self_pair_conflicts"]:
  branchfixture={"seed":seed,"points":xyz.tolist(),"triangles":tris,"info":info};break
assert branchfixture is not None,"could not find physical 3D-conflict fixture"
(P/"branch-fixture.json").write_text(json.dumps(branchfixture,indent=2)+"\n")
print(json.dumps({"seed":branchfixture["seed"],"stats":branchfixture["info"]["statistics"],"seconds":branchfixture["info"]["elapsed_s"]}))
