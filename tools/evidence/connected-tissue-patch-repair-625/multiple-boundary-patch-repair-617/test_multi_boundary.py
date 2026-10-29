from pathlib import Path
import importlib.util,sys,numpy as np,itertools,json,hashlib,time
P=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("multi617",P/"multi_boundary.py")
dp=load("dp617",P.parent/"constraint-disk-triangulation-616/constraint_disk.py")
ears=load("ears617",P.parent/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=ears.load_exact_owner();index=pred._prepare_surface_aabb(())
fixtures=[
 ("one_hole",[[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0],[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2,3],[4,7,6,5]]),
 ("two_holes",[[-5,-3,0],[5,-3,0],[5,3,0],[-5,3,0],[-3,-1,0],[-1,-1,0],[-1,1,0],[-3,1,0],[1,-1,0],[3,-1,0],[3,1,0],[1,1,0]],[[0,1,2,3],[4,7,6,5],[8,11,10,9]]),
 ("three_holes",[[-6,-4,0],[6,-4,0],[6,4,0],[-6,4,0],[-4,-1,0],[-2,-1,0],[-2,1,0],[-4,1,0],[2,-1,0],[4,-1,0],[4,1,0],[2,1,0],[-1,2,0],[1,2,0],[1,3,0],[-1,3,0]],[[0,1,2,3],[4,7,6,5],[8,11,10,9],[12,15,14,13]])]
rows=[]
for name,pts,loops in fixtures:
 pts=np.asarray(pts,np.float32)
 bd={tuple(sorted((a,b)))for l in loops for a,b in zip(l,l[1:]+l[:1])}
 tris,recs,info=h.collision_aware_multi_boundary(loops,pts.astype(float),pts,bd,index,pred,1000000,dp_disk=dp.collision_aware_dp_disk,max_seconds=10,max_cuts=8,max_branches=4096)
 expected=sum(float(np.sum(pts[l,0]*np.roll(pts[l,1],-1)-pts[l,1]*np.roll(pts[l,0],-1)))*.5 for l in loops)
 actual=sum(float(np.cross(pts[b]-pts[a],pts[c]-pts[a])[2])*.5 for a,b,c in tris)
 assert actual==expected,(actual,expected)
 assert pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)["count"]==0
 assert h.validate_surface(tris,loops,bd)["euler_characteristic"]==2-len(loops)
 try:h.validate_surface(tris+[tris[0]],loops,bd)
 except ValueError:pass
 else:raise AssertionError("duplicate physical face accepted")
 rows.append({"fixture":name,"status":"PASS","oriented_area":actual,"expected_area":expected,"details":info})
# Virtual cut copies must never hide duplicate physical faces or wrong edge orientation.
pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,1,0]],np.float32)
proxy=h.PhysicalIdentityPredicate(pred,[0,1,2,0,1,2])
recs=proxy._records(tuple(pred.float32_point_lattice_key(v)for v in pts),[(0,1,2),(3,5,4)])
assert proxy._audit_record_pairs([(recs[0],recs[1])],same_surface=True)["count"]>=1
pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],np.float32)
proxy=h.PhysicalIdentityPredicate(pred,[0,1,2,3])
recs=proxy._records(tuple(pred.float32_point_lattice_key(v)for v in pts),[(0,1,2),(0,1,3)])
assert pred._audit_record_pairs([(recs[0],recs[1])],same_surface=True)["count"]==0
assert proxy._audit_record_pairs([(recs[0],recs[1])],same_surface=True)["count"]==1
print(json.dumps({"status":"PASS","rows":[{"fixture":r["fixture"],"seconds":r["details"]["elapsed_s"],"branches":r["details"]["dp"]["statistics"]["branches_popped"]}for r in rows]}))
result={"schema":"exact-multiple-boundary-constraint-DP-regression.v1","status":"PASS","helper_sha256":hashlib.sha256((P/"multi_boundary.py").read_bytes()).hexdigest(),"dp_helper_sha256":hashlib.sha256((P.parent/"constraint-disk-triangulation-616/constraint_disk.py").read_bytes()).hexdigest(),"fixtures":rows,"duplicate_physical_face_control":"PASS","cooriented_physical_edge_control":"PASS","physical_admission":False}
(P/"regression.json").write_text(json.dumps(result,indent=2)+"\n")
