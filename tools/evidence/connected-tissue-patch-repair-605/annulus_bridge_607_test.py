from pathlib import Path
import importlib.util,sys,json,itertools,hashlib,time
import numpy as np
p=Path(__file__).resolve().parent
P=p.parent
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("annulus607",p/"annulus_bridge.py")
r=load("ears607",P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=r.load_exact_owner()
fixtures=[
("square",np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],np.float32),[[0,1,2,3],[4,7,6,5]]),
("concave_outer",np.array([[-3,-3,0],[3,-3,0],[3,3,0],[0,1,0],[-3,3,0],[-.5,-1,0],[.5,-1,0],[.5,0,0],[-.5,0,0]],np.float32),[[0,1,2,3,4],[5,8,7,6]])]
rows=[]
def polyarea(points,loop):
 t=points[loop].astype(np.float64)
 return float(np.sum(t[:,0]*np.roll(t[:,1],-1)-t[:,1]*np.roll(t[:,0],-1))*.5)
for name,points,loops in fixtures:
 lattice=tuple(pred.float32_point_lattice_key(v) for v in points)
 boundary={tuple(sorted((u,v))) for loop in loops for u,v in zip(loop,loop[1:]+loop[:1])}
 start=time.monotonic()
 tris,recs,detail=h.collision_aware_bridged_annulus(loops,points.astype(float),points,boundary,
  pred._prepare_surface_aabb(()),pred,lattice,[],1000000000,ear_clip=r.collision_aware_ear_clip,
  max_nodes=20000,max_seconds=8,max_bridges=24)
 exact=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
 if exact["count"]:raise RuntimeError("fixture exact self failed")
 area=sum(float(np.cross(points[b].astype(float)-points[a],points[c].astype(float)-points[a])[2])*.5 for a,b,c in tris)
 expected=sum(polyarea(points,l) for l in loops)
 if abs(area-expected)>1e-12:raise RuntimeError("fixture oriented area mismatch")
 rejects=[]
 try:h.validate_annulus(tris+[tris[0]],loops,boundary)
 except ValueError:rejects.append("duplicate_triangle")
 else:raise RuntimeError("duplicate triangle accepted")
 interior={tuple(sorted((u,v))) for t in tris for u,v in zip(t,t[1:]+t[:1])}-boundary
 try:h.validate_annulus(tris,loops,boundary|{min(interior)})
 except ValueError:rejects.append("retained_interior_edge")
 else:raise RuntimeError("retained interior edge accepted")
 rows.append({"fixture":name,"status":"PASS","exact_self_pairs":0,"oriented_area":area,"expected_area":expected,
  "negative_controls":rejects,"seconds":time.monotonic()-start,"details":detail})
out={"schema":"bounded-annulus-bridge-regression.v1","source_sha256":hashlib.sha256((p/"annulus_bridge.py").read_bytes()).hexdigest(),"fixtures":rows,"physical_admission":False}
(p/"regression.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"status":"PASS","fixtures":len(rows),"rows":[{"fixture":r["fixture"],"seconds":r["seconds"],"same_boundary_only_triangles":r["details"]["topology"]["same_boundary_only_triangles"]}for r in rows]}))
