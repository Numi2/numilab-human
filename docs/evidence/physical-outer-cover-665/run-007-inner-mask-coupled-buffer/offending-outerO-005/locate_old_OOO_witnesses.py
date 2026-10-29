from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=P/"reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer"
R=A/"run-007-inner-mask-coupled-buffer"
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
MAP=P/"conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv"
REG=P/"full-current-surface-registration-399/sidecars"
O_CSV=REG/"registered-O.csv"; OF_CSV=REG/"registered-O-faces.csv"
REGISTRY=REG/"source-domain-registry.json"
BANK=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/bank-envelopes/bank-023-032-cube2um.npz"
PREV=R/"source-only-audit-004/report.json"
INV=A/"raw-fat-skin-source-role-inventory.json"
OUT=R/"offending-outerO-005"
LOCATOR=P/"reduced-physical-boundary-400/attempt-042-current006-d23-bone32-cavity/attempt-005-bone32-subtraction/continuation-005/source330-registration-001/register_cavity_points_to_source330.py"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def main():
 t0=time.monotonic()
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/"report.json").exists():raise FileExistsError(OUT/"report.json")
 spec=importlib.util.spec_from_file_location("source330_locator_044",str(LOCATOR)); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 with np.load(BASE,allow_pickle=False) as z:b={k:np.asarray(z[k]) for k in z.files}
 p=np.asarray(b["points_reference_world_m_f64"],np.float32).astype(np.float64)
 T=np.asarray(b["volume_tetrahedra"],np.int64)
 pinned=np.asarray(b["pinned_volume_node_rows"],np.int64)
 if len(pinned)!=149: raise ValueError(f"expected 149 pinned rows, got {len(pinned)}")
 with np.load(R/"mask-inputs.npz",allow_pickle=False) as z: Fs=np.asarray(z["source_skin23_faces_global_node_rows"],np.int64); bonep=np.asarray(z["bone32_points_world_m_f64"],np.float64); bonef=np.asarray(z["bone32_faces"],np.int64)
 prev=json.load(open(PREV));inventory=json.load(open(INV));inv_ids=set(map(int,inventory["histogram"]["OOO"]["source_face_rows"]))
 records={int(x["source_face_row"]):x for x in prev["old_pure_OOO"]["per_source_face"]}
 bad=sorted(i for i in inv_ids if i in records and records[i]["coverage_area_ratio"]<1-1e-10)
 if len(bad)!=13:raise ValueError(f"expected 13 partial inventory faces, got {len(bad)}")
 frows=[Fs[i] for i in bad]
 node_rows=np.unique(np.asarray(frows,np.int64))
 if node_rows.min()<0 or node_rows.max()>=len(p):raise ValueError("source face node outside canonical volume row domain")
 # Edge-connected face components.
 edge_faces=defaultdict(list)
 for k,face in enumerate(frows):
  for a,b0 in ((0,1),(1,2),(2,0)): edge_faces[tuple(sorted((int(face[a]),int(face[b0]))))].append(k)
 adj=[set() for _ in bad]
 for vals in edge_faces.values():
  for i in vals:
   adj[i].update(j for j in vals if j!=i)
 unseen=set(range(len(bad)));comps=[]
 while unseen:
  seed=min(unseen);stack=[seed];unseen.remove(seed);comp=[]
  while stack:
   i=stack.pop();comp.append(i)
   for j in sorted(adj[i]&unseen):unseen.remove(j);stack.append(j)
  comps.append(sorted(comp))
 comps.sort(key=lambda x:x[0])
 # Full combined-node role mapping and protected149 incidence.
 node_records=readcsv(MAP); node_map={int(x["joint_node_id"])-1:x for x in node_records}
 direct=sorted(set(map(int,node_rows))&set(map(int,pinned)))
 incidence=np.any(np.isin(T,node_rows),axis=1)
 one_ring=np.unique(T[incidence].reshape(-1))
 pin_ring=sorted(set(map(int,one_ring))&set(map(int,pinned)))
 # Load accepted fine O points/faces from frozen registered 399 sidecars.
 orows=readcsv(O_CSV); fcsv=readcsv(OF_CSV)
 if len(orows)!=56054 or len(fcsv)!=112063:raise ValueError("fine-O sidecar row count differs from authenticated registry")
 if [int(r["vertex"]) for r in orows]!=list(range(len(orows))):raise ValueError("fine-O vertex row order is not canonical")
 if [int(r["face"]) for r in fcsv]!=list(range(len(fcsv))):raise ValueError("fine-O face row order is not canonical")
 finep=np.array([[float(r["wx"]),float(r["wy"]),float(r["wz"])] for r in orows],np.float32).astype(np.float64)
 finef=np.array([[int(r["v0"]),int(r["v1"]),int(r["v2"])] for r in fcsv],np.int64)
 if finef.min()<0 or finef.max()>=len(finep):raise ValueError("fine-O face index outside source domain")
 # Bank closest-triangle geometry, all quantities in metres.
 with np.load(BANK,allow_pickle=False) as z: bp=np.asarray(z["points_world_m_f64"],np.float64);bf=np.asarray(z["faces_f64"],np.int64)
 points=p[node_rows]
 def nearest_surface(target, surfp, surff):
  tri=np.asarray(surfp,np.float64)[np.asarray(surff,np.int64)]
  valid=np.isfinite(tri).all(axis=(1,2)) & np.any(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])!=0,axis=1)
  original_rows=np.flatnonzero(valid); tri=tri[valid]
  q,bary,d2=mod.closest_triangles(np.asarray(target,np.float64),tri[:,0],tri[:,1],tri[:,2])
  k=int(np.argmin(d2));row=int(original_rows[k]);n=np.cross(tri[k,1]-tri[k,0],tri[k,2]-tri[k,0]);nn=float(np.linalg.norm(n))
  return {"face_row":row,"distance_m":float(np.sqrt(max(0.0,d2[k]))),"point_m":q[k].astype(float).tolist(),"barycentric":bary[k].astype(float).tolist(),"unit_normal":(n/nn).astype(float).tolist() if nn else None,"valid_triangle_count":int(len(tri)),"dropped_degenerate_count":int(len(surff)-len(tri))}
 node_witness=[]
 fby={int(r["face"]):r for r in fcsv}
 for vr in map(int,node_rows):
  fi=nearest_surface(p[vr],finep,finef); bi=nearest_surface(p[vr],bp,bf)
  of=fby[fi["face_row"]]
  node_witness.append({"volume_node_row":vr,"world_reference_f32_m":p[vr].astype(float).tolist(),"role":node_map.get(vr,{}).get("role"),"combined_node_record":node_map.get(vr),
   "base_full86_dominant_column":int(np.argmax(b["full86_weights"][vr])),"base_full86_dominant_weight":float(np.max(b["full86_weights"][vr])),
   "nearest_fineO":{**fi,"face_source_layer":int(of["face_source_layer"]),"face_source_body":int(of["face_source_body"]),"face_source_bone":int(of["face_source_bone"]),"face_source_face_id":int(of["face_source_face_id"]),"face_source_original_id":int(of["face_source_original_id"]),"vertex_rows":finef[fi["face_row"]].tolist()},
   "nearest_bone32":bi})
 face_witness=[]
 for i,face in zip(bad,frows):
  face_witness.append({"source_skin23_face_row":int(i),"global_volume_node_rows":face.astype(int).tolist(),"world_reference_f32_m":p[face].astype(float).tolist(),"source_area_m2":float(records[i]["original_area_m2"]),"candidate_skin_tagged_area_m2":float(records[i]["output_skin_area_m2"]),"retained_area_fraction":float(records[i]["coverage_area_ratio"]),"source330_stable_source_face_id_per_incident_tet":sorted(set(map(int,b["stable_source_face_id"][np.flatnonzero(np.any(np.isin(T,face),axis=1))]))),"full86_dominant_columns_by_corner":[int(np.argmax(b["full86_weights"][int(n)])) for n in face]})
 result={"schema":"numi.run007-old-OOO-source-patch-witness.v1","status":"SOURCE_ONLY_LOCAL_WITNESS","inputs_sha256":{str(x):sha(x) for x in [BASE,R/"receipt.json",R/"mask-inputs.npz",R/"source-view-maps.npz",R/"candidate-skin-region23-f64.npz",R/"candidate-fat-region23-f64.npz",PREV,INV,O_CSV,OF_CSV,REGISTRY,BANK,MAP,LOCATOR]},
  "accepted_fine_O_source_pin":"2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a","fine_O_registered_rows":{"vertices":len(finep),"faces":len(finef),"registered_O_csv_sha256":sha(O_CSV),"registered_O_faces_csv_sha256":sha(OF_CSV),"domain_registry_sha256":sha(REGISTRY)},
  "offending_inventory_source_faces":face_witness,"edge_connected_face_components":[{"face_ids":[bad[k] for k in c],"vertex_rows":sorted(set(int(n) for k in c for n in frows[k]))} for c in comps],
  "offending_node_witnesses":node_witness,"source_node_patch":{"unique_volume_nodes":len(node_rows),"volume_node_rows":node_rows.astype(int).tolist(),"protected149_total":len(pinned),"protected149_direct_intersection":direct,"protected149_direct_count":len(direct),"protected149_one_tet_ring_intersection":pin_ring,"protected149_one_tet_ring_count":len(pin_ring),"one_tet_ring_volume_node_count":len(one_ring),"one_tet_ring_volume_node_rows":one_ring.astype(int).tolist()},
  "scope":{"distance_units":"metres","fineO_projection":"Euclidean nearest triangle on frozen registered O sidecar; inferred geometric witness only, not identity map","bone_clearance":"Euclidean nearest triangle on pinned bank32 surface; does not establish inside/outside or contact status","runtime_created":0,"accepted_steps":0,"physical_admission":False},
  "elapsed_s":time.monotonic()-t0}
 (OUT/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":result["status"],"offending_source_face_rows":bad,"connected_components":result["edge_connected_face_components"],"unique_source_nodes":len(node_rows),"protected149_direct":direct,"protected149_one_tet_ring":pin_ring,"node_witnesses":[{"volume_node_row":x["volume_node_row"],"role":x["role"],"fineO_face":x["nearest_fineO"]["face_row"],"fineO_distance_mm":x["nearest_fineO"]["distance_m"]*1000,"bone32_face":x["nearest_bone32"]["face_row"],"bone_distance_mm":x["nearest_bone32"]["distance_m"]*1000} for x in node_witness],"output":str(OUT),"elapsed_s":result["elapsed_s"]},indent=2),flush=True)
if __name__=="__main__":main()
