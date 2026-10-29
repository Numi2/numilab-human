from pathlib import Path
import importlib.util,sys,json,hashlib,itertools,numpy as np
P=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("chord618test",P/"chord_pruning.py")
r=load("ears618test",P.parent/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=r.load_exact_owner()
T=((0,0,0),(4,0,0),(0,4,0))
cases=[
 ("transverse",(1,1,-1),(1,1,1),{(1,1,0)}),
 ("coplanar_cross",(-1,1,0),(4,1,0),{(0,1,0),(3,1,0)}),
 ("coplanar_contained",(1,1,0),(2,1,0),{(1,1,0),(2,1,0)}),
 ("collinear_edge_overlap",(-1,0,0),(2,0,0),{(0,0,0),(2,0,0)}),
 ("supporting_line_outside",(5,0,0),(6,0,0),set()),
 ("parallel_offset",(1,1,1),(2,1,1),set()),
 ("touch_vertex",(-1,0,0),(0,0,0),{(0,0,0)})]
for name,p,q,expected in cases:
 actual=set(h.segment_triangle_points(pred,p,q,T))
 assert actual==expected,(name,actual,expected)
# Segment pruning must never reject a triangle that the exact geometric owner admits.
rng=np.random.default_rng(618);samples=0;detected=0
for case in range(256):
 allpts=rng.integers(-4,5,size=(6,3)).astype(np.float32)
 if case%2==0:allpts[:,2]=0
 lattice=tuple(pred.float32_point_lattice_key(v)for v in allpts)
 try:qs=pred._records(lattice,[(0,1,2)])[0];br=pred._records(lattice,[(3,4,5)])[0]
 except Exception:continue
 br=(br[0],br[1],br[2],-1,br[4])
 idx=pred._prepare_surface_aabb([br])
 blocked,info=h.prune_chords([0,1,2],lattice,idx,pred)
 exact=pred._audit_record_pairs([(qs,br)],same_surface=True)
 if blocked:assert exact["count"]>0,(case,blocked,exact)
 samples+=1;detected+=bool(blocked)
assert samples>=200 and detected>20
# A true shared edge is permitted. Identical coordinates with distinct IDs are contact.
points=np.asarray([[0,0,0],[2,0,0],[0,2,0],[2,2,0]],np.float32)
lattice=tuple(pred.float32_point_lattice_key(v)for v in points)
br=pred._records(lattice,[(1,3,2)])[0];br=(br[0],br[1],br[2],-1,br[4])
blocked,info=h.prune_chords([0,1,2],lattice,pred._prepare_surface_aabb([br]),pred)
assert not blocked,(blocked,info)
separate=(br[0],br[1],br[2],-1,tuple(i+10 for i in br[4]))
blocked,info=h.prune_chords([0,1,2],lattice,pred._prepare_surface_aabb([separate]),pred)
assert (1,2)in blocked
result={"schema":"exact-chord-necessary-constraint-regression.v1","status":"PASS","helper_sha256":hashlib.sha256((P/"chord_pruning.py").read_bytes()).hexdigest(),
 "segment_cases":[r[0]for r in cases],"random_exact_triangle_conservativeness_samples":samples,"blocked_samples":detected,
 "shared_edge_allowed":True,"distinct_identity_contact_detected":True,"physical_admission":False}
(P/"regression.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
