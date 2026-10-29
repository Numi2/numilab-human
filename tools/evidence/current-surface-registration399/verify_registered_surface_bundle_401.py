#!/usr/bin/env python3
"""Independently verify current authoring CSV rows; this does not admit geometry/physics."""
from pathlib import Path
from contextlib import ExitStack
import csv,hashlib,importlib.util,json,math,sys,time,collections
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
S=P/"full-current-surface-registration-399/sidecars";O=P/"full-current-surface-registration-401";O.mkdir(exist_ok=False)
M=P/"streamed-layer-registration-398/run-002";sys.path.insert(0,str(M))
spec=importlib.util.spec_from_file_location("registered401",M/"export_authoring_registered_surface_sidecars_230.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
t=time.monotonic()
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
receipt=json.loads((S/"receipt.json").read_text());launch=json.loads((S.parent/"launch.json").read_text())
for name,want in receipt["outputSHA256"].items():assert sha(S/name)==want,name
parent,index,material_index,parent_meta=m.load_parent_current(Path(receipt["parentCurrent"]["massPath"]))
weights,weight_meta=m.load_parent_full86_weights(Path(receipt["parentFull86Weights"]["path"]),parent,parent_meta)
frames,pose_meta,binding=m.load_frames(Path(receipt["poseSnapshot"]["path"]))
results={};whole=0
for layer in ["O","I","D"]:
 src=receipt["candidateSurfaces"][layer];assert sha(src["path"])==src["sha256"]
 z=np.load(src["path"],allow_pickle=False);points=np.asarray(z["points_world_m"],np.float64)
 regions=collections.defaultdict(list);donors=collections.Counter();counts=collections.Counter()
 maxima={"parent_reconstruction_m":0.0,"frame_reconstruction_m":0.0,"native_F32_local_reconstruction_m":0.0,"weight_interpolation_abs":0.0,"weight_sum_abs":0.0}
 with ExitStack() as stack:
  tables=[csv.DictReader(stack.enter_context((S/n).open(newline=""))) for n in [f"registered-{layer}.csv",f"parent-tet-full-weights-{layer}.csv",f"parent-tet-full-weight-owners-{layer}.csv"]]
  for vertex,(row,wrow,owner) in enumerate(zip(*tables,strict=True)):
   assert int(row["vertex"])==int(wrow["vertex"])==int(owner["vertex"])==vertex
   assert row["source_mapping_kind"]=="vertex" and int(row["source_vertex_row"])==vertex and int(row["source_face_row"])==-1
   world=np.array([float(row[k]) for k in ["wx","wy","wz"]]);local=np.array([float(row[k]) for k in ["lx","ly","lz"]])
   assert np.array_equal(world,points[vertex]),(layer,vertex,"direct source point")
   parent_row=int(row["parent_tet_row"]);bary=np.array([float(row[f"parent_bary{i}"]) for i in range(4)])
   assert parent_row==int(owner["parent_tet_row"]) and np.isfinite(bary).all() and bary.min()>=-1e-8 and abs(bary.sum()-1)<1e-8
   tet=parent["tets"][parent_row];material=int(parent["materials"][parent_row]);assert material==m.MATERIAL[layer]==int(row["parent_material_index"])
   residual=np.array([float(row[f"parent_residual_{a}_m"]) for a in "xyz"])
   e=float(np.linalg.norm(bary@parent["points"][tet]+residual-world));maxima["parent_reconstruction_m"]=max(maxima["parent_reconstruction_m"],e);assert e<1e-12
   w=np.array([float(wrow[f"w{i}"]) for i in range(86)]);assert np.isfinite(w).all() and w.min()>=-1e-8
   we=float(np.max(np.abs(w-bary@weights[tet])));se=abs(float(w.sum())-1)
   maxima["weight_interpolation_abs"]=max(maxima["weight_interpolation_abs"],we);maxima["weight_sum_abs"]=max(maxima["weight_sum_abs"],se)
   assert we<1e-12 and se<1e-4
   lane=int(np.argmax(w));donor=int(row["donor"]);assert donor==binding[lane]==int(owner["dominant_binding_body"]) and lane==int(owner["dominant_binding_index"])
   frame=frames[donor];reconstructed=frame["currentR"]@local+frame["currentOrigin"]
   fe=float(np.linalg.norm(reconstructed-world));maxima["frame_reconstruction_m"]=max(maxima["frame_reconstruction_m"],fe);assert fe<1e-12
   f32e=float(np.linalg.norm(frame["currentR"]@local.astype(np.float32).astype(np.float64)+frame["currentOrigin"]-world))
   maxima["native_F32_local_reconstruction_m"]=max(maxima["native_F32_local_reconstruction_m"],f32e)
   reg=int(row["parent_anatomical_region_id"]);distance=float(row["parent_registration_distance_m"])
   assert distance<=receipt["limits"]["maximumParentRegistrationDistanceM"] and math.isfinite(distance)
   regions[reg].append(distance);donors[donor]+=1;counts["rows"]+=1
  assert counts["rows"]==len(points)
 regstats={str(reg):{"count":len(a),"maximum_m":float(np.max(a)),"p95_m":float(np.percentile(a,95)),"rms_m":float(np.sqrt(np.mean(np.square(a))))} for reg,a in sorted(regions.items())}
 results[layer]={"nodes":len(points),"face_count":src["faces"],"candidate_sha256":src["sha256"],"maxima":maxima,"parent_geometric_error_by_anatomical_region":regstats,"frame_owner_row_counts":dict(donors)}
 whole+=len(points)
r={"status":"PASS_ALL_REGISTERED_AUTHORING_ROWS","nodes_verified":whole,"input_receipt":{"path":str(S/"receipt.json"),"sha256":sha(S/"receipt.json")},"source":{"path":__file__,"sha256":sha(__file__)},"output_files_hash_verified":len(receipt["outputSHA256"]),"parent_input_pins":parent_meta,"full86_input_pins":weight_meta,"pose_input_pins":pose_meta,"layers":results,"elapsed_s":time.monotonic()-t,"physical_admission":False,"geometry_admission":False,"native_steps":0,"limits":["Parent residuals are retained geometric discrepancies, not physical-boundary qualification.","Native F32 local reconstruction measures cast-local error using the captured matrix; it does not reproduce native fused quaternion emission."]}
(O/"receipt.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":r["status"],"nodes":whole,"elapsed_s":r["elapsed_s"],"maxima":{k:v["maxima"] for k,v in results.items()}}),flush=True)
