#!/usr/bin/env python3
"""Read-only validation for 625's paired target metadata derivative."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def text(a):return str(np.asarray(a).reshape(-1)[0])
def chain_key_sign(f):
 f=np.asarray(f,np.int64)
 k=np.sort(f,axis=1)
 inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)
 s=np.where(inv%2==0,1,-1).astype(np.int8)
 o=np.lexsort((s,k[:,2],k[:,1],k[:,0]))
 return k[o],s[o]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--report",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 rep=json.loads(a.report.read_text())
 if rep.get("status")!="PASS_PAIRED_TYPED_TARGET_METADATA_REBOUND_SOURCE_ONLY":raise ValueError("wrong finalizer receipt")
 inp=json.loads(Path(rep["input_state"]["path"]).read_text())
 outstate=json.loads(Path(rep["output_state"]["path"]).read_text())
 srcstate=json.loads(Path(rep["source_state"]["path"]).read_text())
 result={"schema":"numi.quality677.metadata-finalizer-readback.v1","status":"PASS","finalizer_report":{"path":str(a.report),"sha256":sha_file(a.report)},"regions":{}}
 for role,e in inp["source_surfaces"].items():
  p=Path(e["path"])
  if sha_file(p)!=e["sha256"]:raise ValueError(f"{role} input source pin mismatch")
  z=load(p);n=len(z["points_world_m_f32"]);nf=len(z["faces"])
  q=np.arange(n,dtype=np.int64)
  if not np.array_equal(z.get("source_input_vertex_quotient",np.empty(0,np.int64)),q):raise ValueError(f"{role} point quotient is not identity")
  if not np.array_equal(z.get("source_input_face_rows",np.empty(0,np.int64)),np.arange(nf,dtype=np.int64)):raise ValueError(f"{role} input face map is not identity")
  cr=np.asarray(z.get("source_collapse_operation_rows",np.empty(0,np.int64)))
  if cr.shape!=(nf,) or np.any(cr!=-1):raise ValueError(f"{role} current-step collapse ledger is not empty")
 for role in ("Skin","Fat"):
  entry=rep["regions"][role];outpath=Path(entry["output_candidate"]["path"]);parent=Path(entry["parent_candidate"]["path"])
  if sha_file(outpath)!=entry["output_candidate"]["sha256"] or sha_file(parent)!=entry["parent_candidate"]["sha256"]:raise ValueError(f"{role} candidate pin mismatch")
  z=load(outpath);p=load(parent);n=len(z["points_world_m_f32"]);nc=len(z["tetrahedra"]);nb=len(z["boundary_faces"]);nd=len(z["desired_boundary_faces"])
  for k in ("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces"):
   if arrsha(z[k])!=arrsha(p[k]):raise ValueError(f"{role} geometry changed: {k}")
  for k in ("replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets","replacement_tetrahedra"):
   if k in p and arrsha(z[k])!=arrsha(p[k]):raise ValueError(f"{role} cell ancestry changed: {k}")
  if not np.array_equal(z["desired_boundary_faces"],z["proposed_current_desired_boundary_faces"]):raise ValueError(f"{role} desired aliases differ")
  if not np.array_equal(z["mapped_current_boundary_faces"],z["boundary_faces"]):raise ValueError(f"{role} actual boundary alias differs")
  for k in ("desired_boundary_source_rows","desired_boundary_source_group","desired_boundary_source_face_rows","proposed_current_desired_boundary_source_rows","proposed_current_desired_boundary_source_group","proposed_current_desired_boundary_source_face_rows"):
   if z[k].shape!=(nd,):raise ValueError(f"{role} source metadata length mismatch: {k}")
  if not np.array_equal(z["desired_boundary_source_rows"],z["desired_boundary_source_face_rows"]):raise ValueError(f"{role} source row and face row disagree")
  if not np.array_equal(z["desired_boundary_source_rows"],z["proposed_current_desired_boundary_source_rows"]):raise ValueError(f"{role} proposed source-row alias stale")
  if not np.array_equal(z["desired_boundary_source_group"],z["proposed_current_desired_boundary_source_group"]):raise ValueError(f"{role} proposed source-group alias stale")
  if not np.array_equal(z["desired_boundary_source_face_rows"],z["proposed_current_desired_boundary_source_face_rows"]):raise ValueError(f"{role} proposed source-face alias stale")
  for k in ("input_node_to_output_node","output_node_to_input_node","output_point_to_input_point_row","source_interior_vertex_quotient"):
   if not np.array_equal(z[k],np.arange(n,dtype=np.int64)):raise ValueError(f"{role} immediate point map not identity: {k}")
  if not np.array_equal(z["output_point_parent_offsets"],np.zeros(n+1,np.int64)) or len(z["output_point_parent_input_cell_rows"])!=0:raise ValueError(f"{role} unexpected new-point parent links")
  bk,bs=chain_key_sign(z["boundary_faces"]);dk,ds=chain_key_sign(z["desired_boundary_faces"])
  if not (np.array_equal(bk,dk) and np.array_equal(bs,ds)):raise ValueError(f"{role} target oriented chain differs from actual")
  result["regions"][role]={"parent_sha256":entry["parent_candidate"]["sha256"],"output_sha256":entry["output_candidate"]["sha256"],"points":n,"cells":nc,"actual_faces":nb,"target_faces":nd,"target_equals_actual_oriented":True,"geometry_and_cell_ancestry_unchanged":True,"target_aliases_aligned":True,"immediate_point_maps_identity":True,"new_point_parents":0}
 result["source_maps"]={r:{"candidate_sha256":inp["source_surfaces"][r]["sha256"],"point_quotient_identity":True,"face_input_map_identity":True,"collapse_operations":0} for r in ("O","I","D")}
 result["physical_admission"]=False;result["native_steps"]=0;result["mass_mapping"]="NOT_RUN";result["surface_crossings"]="PENDING_SEPARATE_678_CONTEXT_AUDIT"
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":result["status"],"output":str(a.output),"sha256":sha_file(a.output),"regions":result["regions"]},indent=2))
if __name__=="__main__":main()

