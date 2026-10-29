from pathlib import Path
import json,hashlib,importlib.util,sys,time,itertools,collections
import numpy as np
P=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent/"run-001"
E=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh"
B=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union"
REPLAY=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
MAN=B/"exact-set-union-remaining-013/all185-bank-operand-manifest.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 t0=time.monotonic();OUT.mkdir(exist_ok=False)
 inv=json.loads((OUT.parent/"collar-inventory.json").read_text())
 base=Path(inv["candidate_path"])
 assert sha(base)==inv["candidate_sha256"]=="9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a"
 z=np.load(base);points=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["faces"],np.int64)
 original=np.load(E.parent/"candidate-f32.npz")
 assert np.array_equal(points,original["points_world_m"]), "current candidate frame changed"
 ears=load("ears614",B/"D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
 bones=load("bones614",P/"bone-barrier-remeshing-609/bone_barrier.py")
 pred=ears.load_exact_owner()
 assert sha(pred.__file__)=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
 assert sha(REPLAY)=="e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5"
 assert sha(MAN)=="0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c"
 fineI=B/"dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
 assert sha(fineI)=="5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410"
 # Every possible replacement triangle lies in the box of its fixed vertices.
 # Min/max use exact representable input values; outward nextafter is conservative.
 allrows=sorted(set(r for plan in inv["plans"]for piece in plan["pieces"]for r in piece["face_rows"]))
 allowed=np.unique(F[allrows]);lo=np.nextafter(points[allowed].astype(float).min(axis=0),-np.inf);hi=np.nextafter(points[allowed].astype(float).max(axis=0),np.inf)
 selected_counts={};barriers=[];negative=[-1]
 def overlap(xyz,faces):
  tris=xyz[faces]
  return np.flatnonzero(np.all(tris.max(axis=1)>=lo,axis=1)&np.all(tris.min(axis=1)<=hi,axis=1))
 def add_records(label,xyz,faces,vertexoffset,exact64=False):
  rows=overlap(xyz,faces);selected_counts[label]={"total_faces":int(len(faces)),"selected_faces":int(len(rows))}
  if not len(rows):return
  ff=faces[rows];used=np.unique(ff);mapping=np.full(len(xyz),-1,np.int64);mapping[used]=np.arange(len(used))
  lattice=tuple(bones.exact_common_lattice_key(v)if exact64 else pred.float32_point_lattice_key(v)for v in xyz[used])
  raw=pred._records(lattice,mapping[ff])
  for rec in raw:
   barriers.append((rec[0],rec[1],rec[2],negative[0],tuple(int(used[v])+vertexoffset for v in rec[4])))
   negative[0]-=1
 with np.load(REPLAY)as b:
  add_records("all185-native-F32",b["points_world_f32"].astype(float),b["faces"].astype(np.int64),len(points))
 vertexoffset=len(points)+1000000
 manifest=json.loads(MAN.read_text())
 for bank in manifest["operands"]:
  path=Path(bank["operand_path"]);assert sha(path)==bank["operand_sha256"]
  with np.load(path)as b:
   xyz=b["points_world_m_f64"];ff=b["faces_f64"]
   add_records("F64-%d-%d"%(bank["body"],bank["bone"]),xyz,ff,vertexoffset,True)
   vertexoffset+=len(xyz)
 with np.load(fineI)as b:
  add_records("fine-I",b["points_world_m"].astype(float),b["faces"].astype(np.int64),vertexoffset)
 currentnear=overlap(points.astype(float),F)
 lattice=tuple(pred.float32_point_lattice_key(v)for v in points)
 originalrecords={}
 for row in currentnear:
  r=pred._records(lattice,[tuple(map(int,F[row]))])[0]
  originalrecords[int(row)]=(r[0],r[1],r[2],int(row),r[4])
 # Retained edge identity gate covers the whole surface, even beyond the exact-query box.
 em=collections.defaultdict(list)
 for row,f in enumerate(F):
  for a,b in zip(f,np.roll(f,-1)):em[tuple(sorted((int(a),int(b))))].append(row)
 attempts=[]
 result={"schema":"current025-expanded-exact-witness-patch.v1","candidate":str(base),"candidate_sha256":sha(base),
  "script_sha256":sha(__file__),"exact_owner_sha256":sha(pred.__file__),"barrier_query_box_m":[lo.tolist(),hi.tolist()],
  "selection":"conservative vertex min/max AABB for all attempted patches, widened by one float64 ulp; all185 source operands hash checked",
  "barrier_counts":selected_counts,"current_near_face_count":len(currentnear),"attempts":attempts,
  "physical_admission":False,"runtime":0,"accepted_steps":0}
 for ring in [4,2,1]:
  plan=next(r for r in inv["plans"]if r["ring"]==ring)
  if any(piece["kind"]!="disk"for piece in plan["pieces"]):continue
  removed=set(r for piece in plan["pieces"]for r in piece["face_rows"])
  fixed=[rec for row,rec in originalrecords.items()if row not in removed]+barriers
  fixedidx=pred._prepare_surface_aabb(fixed)
  external={edge for edge,rows in em.items()if any(row not in removed for row in rows)}
  facesnew=[];prior=[];details=[]
  try:
   for pi,piece in enumerate(plan["pieces"]):
    tris,recs,info=ears.collision_aware_ear_clip(piece["boundary_loops"][0],points.astype(float),points,external,fixedidx,pred,lattice,prior,
       1000000000000+ring*1000000+pi*100000,max_nodes=20000,max_seconds=8)
    facesnew.extend(tris);prior=list(zip(recs,[*facesnew]));details.append(info)
    for f in tris:
     external.update(tuple(sorted((int(a),int(b))))for a,b in zip(f,np.roll(f,-1)))
   # Independent exact recheck of fresh faces against all retained near faces/structures and one another.
   recs=[x[0]for x in prior]
   against=pred._audit_pair_prepared_first(fixedidx,recs,same_surface=True)
   selfaudit=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
   assert against["count"]==0 and selfaudit["count"]==0
   old=np.asarray(F[sorted(removed)],np.int64);new=np.asarray(facesnew,np.int64)
   def edges(fs):
    c=collections.Counter((int(a),int(b))for f in fs for a,b in zip(f,np.roll(f,-1)))
    return c
   oldedges=edges(old);newedges=edges(new)
   def boundary(c):
    out={}
    for a,b in c:
     net=c[(a,b)]-c[(b,a)]
     if net>0:out[(a,b)]=net
    return out
   assert boundary(oldedges)==boundary(newedges),"directed collar boundary changed"
   assert all(n<=1 for n in newedges.values()),"duplicate directed edge"
   assert len({tuple(sorted(f))for f in facesnew})==len(facesnew),"duplicate face"
   def measure(fs):
    tri=points[fs].astype(float);x=tri[:,0];u=tri[:,1]-x;v=tri[:,2]-x
    area=np.linalg.norm(np.cross(u,v),axis=1)/2
    origin=points[np.unique(old)].astype(float).mean(axis=0);tri-=origin
    volume=np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    return {"area_sum_m2":float(area.sum()),"min_area_m2":float(area.min()),"oriented_volume_about_common_origin_m3":float(volume),
      "max_edge_m":float(max(np.linalg.norm(points[fs[:,k]].astype(float)-points[fs[:,(k+1)%3]].astype(float),axis=1).max()for k in range(3)))}
   op={"schema":"pair-linked-connected-remesh-compact-operations.v1","candidate_sha256":sha(base),"transaction_id":65,"patch_index":0,
     "parent_face_rows":sorted(removed),"new_faces":facesnew,"ring":ring,"old_measurements":measure(old),"new_measurements":measure(new),
     "coordinate_edits":0,"source_ancestry":"every replacement face retains full union support of removed current025 faces; no unique measured source face asserted",
     "new_original_id":62016500,"exact_changed_vs_current_and_structures_pairs":0,"exact_new_self_pairs":0,"physical_admission":False}
   result["operation"]=op;result["status"]="LOCAL_EXACT_GATES_PASS";attempts.append({"ring":ring,"status":"PASS","details":details})
   (OUT/"compact-operation.json").write_text(json.dumps(op,indent=2)+"\n")
   break
  except Exception as ex:
   attempts.append({"ring":ring,"status":"FAIL","error":repr(ex),"details":details})
 else:result["status"]="BOUNDED_SEARCH_FAILED"
 result["elapsed_s"]=time.monotonic()-t0
 (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"status":result["status"],"elapsed_s":result["elapsed_s"],"attempts":[{k:v for k,v in r.items()if k!="details"}for r in attempts],
  "operation":result.get("operation")}),flush=True)
if __name__=="__main__":main()
