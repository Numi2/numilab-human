from pathlib import Path
import hashlib, json, math, struct, sys
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
ROOT=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005")
OUT=Path(__file__).resolve().parent
NHTISS=ROOT/"output/current-anatomy-20261005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=NHTISS.with_name("bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json")
REG=ROOT/"Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json"
ART=ROOT/"Build/skin-source-fit-recovery-20261004"
OBJROOT=ROOT/"output/nhtiss-source-prep-20261005/project/Sources"
CAPS=[
 (R/"native-lung1178-thumb1187-smoke-1191/native-run",0),
 (R/"native-lung1178-thumb1187-smoke-1191/native-run",10000),
 (R/"native-flat-reference-40s-1201/native-run",20000),
]
EDGE=(2597,3054)
FACE_ROWS=(5197,5198)
TARGET=64
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human import model, common_atlas_skin_clearance as clearance
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology

def sha(p):
 p=Path(p); h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p); return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def qrot(q):
 x,y,z,w=map(float,q); n=x*x+y*y+z*z+w*w; s=2.0/n
 return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                  [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                  [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]],dtype=np.float64)
def xyz_key(p): return np.asarray(p,dtype="<f4").tobytes()
def quotient(v,f):
 key_to_id={}; qv=[]; qid=np.empty(len(v),dtype=np.int64)
 for i,p in enumerate(v):
  k=xyz_key(p)
  if k not in key_to_id:
   key_to_id[k]=len(qv); qv.append(np.asarray(p,dtype="<f4").copy())
  qid[i]=key_to_id[k]
 qf=qid[np.asarray(f,dtype=np.int64)]
 return np.asarray(qv,dtype="<f4"),qf,key_to_id

def audit(v,f,label):
 qv,qf,_=quotient(v,f)
 topology=analyze_topology(qv.astype(float).tolist(),qf.astype(int).tolist())
 keys=[ci.float32_point_lattice_key(p) for p in qv]
 records=ci._records(keys,qf.tolist())
 result=ci._audit_pair(records,records,same_surface=True)
 return {"label":label,"vertex_count":int(len(v)),"face_count":int(len(f)),"coordinate_quotient_vertex_count":int(len(qv)),"topology":topology,"exact_self_intersections":result}

def triangle_area(a,b,c): return 0.5*float(np.linalg.norm(np.cross(b-a,c-a)))
def signed_volume(v,f):
 v=np.asarray(v,dtype=np.float64); total=0.0
 for a,b,c in np.asarray(f,dtype=np.int64): total+=float(np.dot(v[a],np.cross(v[b],v[c])))/6.0
 return total
def surface_area(v,f):
 v=np.asarray(v,dtype=np.float64)
 return sum(triangle_area(v[a],v[b],v[c]) for a,b,c in np.asarray(f,dtype=np.int64))

# Decode exact NHTISS row bytes without writing or changing source assets.
raw=NHTISS.read_bytes()
magic,abi,nr,nb,nv,ni,fingerprint,source_hash=struct.unpack_from("<8s6I32s",raw)
assert magic==b"NHTISS4\0" and abi==5
rec_table=np.frombuffer(raw,dtype="<u4",count=nr*8,offset=64).reshape(nr,8)
rec=rec_table[rec_table[:,6]==TARGET][0]
first_bind,bind_count,first_v,vertex_count,first_i,index_count,stable,layer=map(int,rec)
bind_offset=64+nr*32; vertex_offset=bind_offset+nb*36; index_offset=vertex_offset+nv*56
binds=np.frombuffer(raw,dtype=[("body","<u4"),("v","<f4",(8,))],count=bind_count,offset=bind_offset+first_bind*36)
source_vertices=np.frombuffer(raw,dtype=[("p","<f4",(3,)),("n","<f4",(3,)),("idx","<u4",(4,)),("w","<f4",(4,))],count=vertex_count,offset=vertex_offset+first_v*56).copy()
source_faces=np.frombuffer(raw,dtype="<u4",count=index_count,offset=index_offset+first_i*4).reshape(-1,3).astype(np.int64)-first_v
assert len(source_faces)==6688 and source_vertices[2597]["p"].tobytes()!=source_vertices[3054]["p"].tobytes()
manifest=json.loads(MAN.read_text())
surface=next(s for s in manifest["source"]["surfaces"] if int(s["stable_id"])==TARGET)
assert surface["nhtiss_vertex_count"]==vertex_count if "nhtiss_vertex_count" in surface else True
route_manifest=json.loads(MAN.read_text())
reg=json.loads(REG.read_text()); M=np.asarray(reg["coordinate_system"]["global_source_mm_to_myosim_world_m"],dtype=np.float64)
_,routes,_=model._myosim_surface_route_context(ART,manifest["source"]["myosim_source_archive_sha256"])
route=routes["vaslat_l"]; route_points=route["route_points"]
body_order=[x["myosim_body"] for x in surface["body_bindings"]]
body_to_slot={int(b["body"]):i for i,b in enumerate(binds)}
assert len(body_order)==bind_count==3
assert [int(b["body"]) for b in binds]==[int(x["core_body_index"]) for x in surface["body_bindings"]]
# Existing owner uses per-body nearest route node, inverse-square normalized.
def route_weights(p_local_m):
 pmm=np.asarray(p_local_m,dtype=np.float64)*1000.0
 route_space=M[:3,:3]@pmm+M[:3,3]
 d2=[]
 for body in body_order:
  d2.append(min(float(np.dot(route_space-np.asarray(node["world_m"],dtype=np.float64),route_space-np.asarray(node["world_m"],dtype=np.float64))) for node in route_points if node["body"]==body))
 vals=[1.0/(x+9e-6) for x in d2]; s=sum(vals)
 return np.asarray([x/s for x in vals],dtype=np.float64)

# Verify one conforming indexed/source-coordinate edge star, not a nearest-coordinate weld.
fa,fb=EDGE
edge_ids=set(EDGE)
incident=[i for i,t in enumerate(source_faces) if edge_ids.issubset(set(map(int,t)))]
edge_bytes=tuple(sorted((xyz_key(source_vertices[fa]["p"]),xyz_key(source_vertices[fb]["p"]))))
coordinate_incident=[]
for i,t in enumerate(source_faces):
 for a,b in zip(t,np.roll(t,-1)):
  if tuple(sorted((xyz_key(source_vertices[int(a)]["p"]),xyz_key(source_vertices[int(b)]["p"]))))==edge_bytes:
   coordinate_incident.append((i,int(a),int(b)))
assert incident==[5197,5198] and {x[0] for x in coordinate_incident}==set(incident)
assert source_faces[5197].tolist()==[2597,2599,3054]
assert source_faces[5198].tolist()==[2597,3054,2788]

# Make a source-space Float32 midpoint and compute its weight from the pinned route owner.
p_a=source_vertices[fa]["p"].astype(np.float64); p_b=source_vertices[fb]["p"].astype(np.float64)
midpoint_exact=(p_a+p_b)*0.5
midpoint_f32=midpoint_exact.astype("<f4")
midpoint_weights64=route_weights(midpoint_f32.astype(np.float64))
midpoint_weights=midpoint_weights64.astype("<f4")
order=np.argsort(-midpoint_weights,kind="stable")
midpoint_idx=np.full(4,0xffffffff,dtype="<u4"); midpoint_w=np.zeros(4,dtype="<f4")
for lane,body_order_index in enumerate(order):
 body_name=body_order[int(body_order_index)]
 slot=body_to_slot[int(surface["body_bindings"][int(body_order_index)]["core_body_index"])]
 midpoint_idx[lane]=slot; midpoint_w[lane]=midpoint_weights[int(body_order_index)]
midpoint_normal=source_vertices[fa]["n"].astype(np.float64)+source_vertices[fb]["n"].astype(np.float64)
midpoint_normal/=np.linalg.norm(midpoint_normal)
midpoint_normal=midpoint_normal.astype("<f4")
assert abs(float(midpoint_w.sum())-1.0)<2e-7
# Construct child faces with preserved winding. Replace first child at each source row; append second child.
new_vertex_id=len(source_vertices)
refined_faces=source_faces.copy()
refined_faces[5197]=[2597,2599,new_vertex_id]
refined_faces[5198]=[2597,new_vertex_id,2788]
refined_faces=np.vstack([refined_faces,[new_vertex_id,2599,3054],[new_vertex_id,3054,2788]]).astype(np.int64)
origins=np.concatenate([np.arange(len(source_faces),dtype=np.int64),np.array([5197,5198],dtype=np.int64)])
assert len(refined_faces)==6690 and origins[5197]==5197 and origins[5198]==5198
# Source geometry arrays remain byte-identical for every pre-existing vertex; only midpoint appended.
positions=np.vstack([source_vertices["p"],midpoint_f32[None,:]]).astype("<f4")
normals=np.vstack([source_vertices["n"],midpoint_normal[None,:]]).astype("<f4")
indices=np.vstack([source_vertices["idx"],midpoint_idx[None,:]]).astype("<u4")
weights=np.vstack([source_vertices["w"],midpoint_w[None,:]]).astype("<f4")
assert np.array_equal(positions[:-1].view("u1"),source_vertices["p"].view("u1"))
assert np.array_equal(normals[:-1].view("u1"),source_vertices["n"].view("u1"))
assert np.array_equal(indices[:-1].view("u1"),source_vertices["idx"].view("u1"))
assert np.array_equal(weights[:-1].view("u1"),source_vertices["w"].view("u1"))

# Save compact immutable offline candidate and operation provenance.
candidate=OUT/"left-vastus-lateralis-one-edge-refinement.npz"
np.savez(candidate,positions=positions,normals=normals,indices=indices,weights=weights,faces=refined_faces,source_face_ids=origins)
# Face geometry area/volume delta is solely Float32 midpoint packing; no old coordinate moved.
base_area=surface_area(source_vertices["p"],source_faces); new_area=surface_area(positions,refined_faces)
base_vol=signed_volume(source_vertices["p"],source_faces); new_vol=signed_volume(positions,refined_faces)
source_audit=audit(positions,refined_faces,"raw-source-NHTISS-F32-with-one-conforming-edge-bisect")

# Load pose geometry/receipts; reconstruct the body-weighted map for the new midpoint.
def make_affines(receipt):
 pose_rows={int(x["body_index"]):x for x in receipt["accepted_registered_body_poses"]}
 aff=[]
 for bind in binds:
  body=int(bind["body"]); row=pose_rows[body]
  qr=qrot(row["quaternion_xyzw"]); local=bind["v"].astype(np.float64); ql=qrot(local[3:7])
  A=qr@(ql*float(local[7])); off=np.asarray(row["position_m"],dtype=np.float64)+qr@local[:3]
  aff.append((A,off))
 return aff

def map_weighted(p,idx,w,aff):
 total=np.zeros(3,dtype=np.float64)
 for slot,weight in zip(idx,w,strict=True):
  slot=int(slot); weight=float(weight)
  if slot==0xffffffff or weight==0: continue
  total+=weight*(aff[slot][0]@np.asarray(p,dtype=np.float64)+aff[slot][1])
 return total

def face_local_to_capture(local_faces,capture_faces):
 assert local_faces.shape==capture_faces.shape
 local_to_capture=np.full(len(source_vertices),-1,dtype=np.int64)
 capture_to_local={}
 for row,(a,b) in enumerate(zip(local_faces,capture_faces,strict=True)):
  for li,gi in zip(a,b,strict=True):
   li=int(li); gi=int(gi)
   if local_to_capture[li] not in (-1,gi): raise RuntimeError(f"inconsistent local-to-capture mapping at local vertex {li}")
   if gi in capture_to_local and capture_to_local[gi]!=li: raise RuntimeError(f"capture vertex reused by distinct local vertices {li}/{capture_to_local[gi]}")
   local_to_capture[li]=gi; capture_to_local[gi]=li
 if np.any(local_to_capture<0): raise RuntimeError("some source vertices are not referenced by captured face topology")
 return local_to_capture

pose_results=[]
for run,step in CAPS:
 capdir=run/"accepted-geometry"; pack=capdir/f"step-{step}.mrvpack"; receipt_path=capdir/f"step-{step}.receipt.json"
 receipt=json.loads(receipt_path.read_text()); pack_pos, surfaces, _=clearance._pack_surfaces(pack,{(51005,TARGET)})
 captured_faces=np.asarray(surfaces[(51005,TARGET)]["faces"],dtype=np.int64)
 assert captured_faces.shape==source_faces.shape
 local_to_capture=face_local_to_capture(source_faces,captured_faces)
 actual_old=np.asarray(pack_pos,dtype="<f4")[local_to_capture]
 aff=make_affines(receipt)
 predicted_old=np.empty_like(actual_old)
 for vi in range(len(source_vertices)):
  predicted_old[vi]=map_weighted(source_vertices[vi]["p"],source_vertices[vi]["idx"],source_vertices[vi]["w"],aff).astype("<f4")
 residual=np.linalg.norm(actual_old.astype(np.float64)-predicted_old.astype(np.float64),axis=1)
 predicted_mid=map_weighted(midpoint_f32,midpoint_idx,midpoint_w,aff).astype("<f4")
 endpoint_linear=((actual_old[fa].astype(np.float64)+actual_old[fb].astype(np.float64))*0.5)
 chord=float(np.linalg.norm(predicted_mid.astype(np.float64)-endpoint_linear))
 pose_v=np.vstack([actual_old,predicted_mid[None,:]]).astype("<f4")
 pose_audit=audit(pose_v,refined_faces,f"captured-step-{step}-plus-owner-predicted-refined-midpoint")
 pose_results.append({"step":step,"pack":pin(pack),"receipt":pin(receipt_path),"local_to_capture_vertex_map_verified":True,"old_vertex_runtime_map_residual_m":{"max":float(residual.max()),"p95":float(np.quantile(residual,.95)),"mean":float(residual.mean()),"count":int(len(residual))},"new_midpoint_model_equivalent_capture_prediction_f32_m":predicted_mid.tolist(),"captured_endpoint_linear_midpoint_m":endpoint_linear.tolist(),"midpoint_chord_deviation_from_captured_linear_edge_m":chord,"audit":pose_audit})

report={
 "schema":"numi.human.vastus-lateralis-conforming-edge-refinement-diagnostic.v1",
 "status":"one-edge-offline-candidate-and-exact-source_pose_audits_complete",
 "claim_scope":"Offline diagnostic candidate only. No NHTISS, runtime, candidate asset admission, or captured pose was changed. Existing source route-weight and local binding owners were reused. New midpoint poses are counterfactual predictions; they are not native captures.",
 "operation":{"method":"bisect one existing indexed and coordinate-unique source edge with two incident triangles; append one midpoint; split both incident triangles with orientation preserved; evaluate the existing per-body nearest-route-node inverse-square formula only for the new midpoint","edge_local_vertex_ids":list(EDGE),"edge_endpoints_source_m":[p_a.tolist(),p_b.tolist()],"source_edge_length_m":float(np.linalg.norm(p_b-p_a)),"incident_face_rows":incident,"coordinate_equal_incident_faces":[int(x[0]) for x in coordinate_incident],"children_at_original_face_rows":{"5197":[[2597,2599,new_vertex_id]],"5198":[[2597,new_vertex_id,2788]]},"appended_children":[{"face":[new_vertex_id,2599,3054],"source_face_id":5197},{"face":[new_vertex_id,3054,2788],"source_face_id":5198}],"new_vertex_id":new_vertex_id,"new_vertex_source_m":midpoint_f32.tolist(),"exact_midpoint_before_float32_m":midpoint_exact.tolist(),"float32_midpoint_rounding_error_m":float(np.linalg.norm(midpoint_f32.astype(np.float64)-midpoint_exact)),"new_vertex_route_body_order":body_order,"new_vertex_route_weights_double":midpoint_weights64.tolist(),"new_vertex_route_weights_float32":midpoint_w.tolist(),"new_vertex_binding_slots_by_lane":midpoint_idx.tolist(),"existing_vertices_byte_identical":True,"old_vertex_count":int(len(source_vertices)),"new_vertex_count":int(len(positions)),"old_face_count":int(len(source_faces)),"new_face_count":int(len(refined_faces)),"face_origin_array":str(candidate)},
 "source_mesh":{"topology_and_self_audit":source_audit,"area_before_m2":base_area,"area_after_m2":new_area,"area_delta_m2":new_area-base_area,"signed_volume_before_m3":base_vol,"signed_volume_after_m3":new_vol,"signed_volume_delta_m3":new_vol-base_vol,"max_original_coordinate_delta_m":0.0},
 "captured_poses":pose_results,
 "inputs":[pin(NHTISS),pin(MAN),pin(REG),pin(OBJROOT/"isa_BP3D_4.0_obj_99.zip"),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/model.py")),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/cardiac_cavity_intersections.py")),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/cardiac_cavity_geometry.py")),pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/common_atlas_skin_clearance.py"))],
 "candidate":pin(candidate),"script":pin(Path(__file__))
}
for pp in pose_results:
 report["inputs"].extend([pp["pack"],pp["receipt"]])
out=OUT/"report.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(out),"report_sha256":sha(out),"candidate":pin(candidate),"source_intersections":source_audit["exact_self_intersections"]["count"],"pose_counts":[{"step":x["step"],"self":x["audit"]["exact_self_intersections"]["count"],"candidate_pairs":x["audit"]["exact_self_intersections"]["aabb_candidate_pairs"],"chord_deviation_m":x["midpoint_chord_deviation_from_captured_linear_edge_m"],"baseline_residual_max_m":x["old_vertex_runtime_map_residual_m"]["max"]} for x in pose_results],"topology":source_audit["topology"]},indent=2,sort_keys=True))
