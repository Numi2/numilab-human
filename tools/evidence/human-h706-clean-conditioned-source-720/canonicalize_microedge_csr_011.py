"""Restore existing scalar CSR rank after a microedge writer serialization defect."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
FIELDS={
 "O":{"reference_vertex_parent_face_rows","face_ancestry_source_face_rows","face_ancestry_parent_face_rows","face_ancestry_source_original_ids"},
 "I":{"reference_vertex_parent_face_rows","reference_original_input_vertex_rows","edge_flip_parent_face_rows","face_parent_rows"},
 "D":{"union_face_ancestry_rows","union_face_ancestry_run_original_ids","candidate_face_ancestry_rows","union_face_ancestry_face_ids","operational_output_face_parent_rows","output_vertex_input_parent_rows"}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def canonicalize(parent,current,role):
 changed={k for k in current.keys()&parent.keys() if current[k].ndim!=parent[k].ndim}
 if changed!=FIELDS[role]:raise ValueError("unexpected CSR rank-difference inventory")
 out=dict(current);record={}
 for key in sorted(changed):
  a=current[key];b=parent[key]
  if b.ndim!=1 or a.ndim!=2 or a.shape[1]!=1 or a.dtype!=b.dtype:
   raise ValueError("member is not the authenticated scalar CSR rank defect: "+key)
  c=a.reshape(-1)
  if c.tobytes()!=a.tobytes():raise ValueError("scalar CSR content changed")
  out[key]=c
  record[key]={"before_shape":list(a.shape),"after_shape":list(c.shape),"dtype":a.dtype.str,
               "flat_value_bytes_sha256":hashlib.sha256(a.tobytes()).hexdigest()}
 for key in set(current)-changed:
  a=current[key];b=out[key]
  if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
   raise ValueError("non-CSR member changed: "+key)
 return out,record
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir()
 manifest=json.loads((H/"plan-003/manifest.json").read_text())
 sources={"O":(H/"local-microedge-condition-001/candidate-O-microcollapsed.npz","1b3820a1a7aa46c3bed0f1475c0f68a9929a2e1353097a3cc4a28240b39844fe"),
 "I":(H/"local-microedge-condition-I-retry-001/candidate-I-microcollapsed.npz","fee20f3d7a1697158398c73a8a771302625e122b52ee23840101c552e3832205"),
 "D":(H/"local-microedge-condition-D-retry-007/candidate-D-microcollapsed.npz","5503033cf67b3adb6533abc7711f2cde5a612fe4275833ef15b8ae1aba202115")}
 result={"schema":"numi.fine-physical-volume.microedge-condition-batch.v1",
  "status":"PASS_SCALAR_CSR_SHAPE_CANONICALIZATION_GEOMETRY_EQUIVALENT_ONLY",
  "operation":"restore inherited scalar CSR members from rank (n,1) to existing rank (n)",
  "owner_sha256":sha(__file__),"physical_admission":False,"runtime_created":0,"accepted_physical_steps":0,
  "inherited_operation_metadata_row_space":"immediate parent; use microedge row maps for candidate rows","roles":{}}
 for role,label in [("O","outer"),("I","inner"),("D","deep")]:
  path,digest=sources[role];ref=manifest["inputs"][label]
  if sha(path)!=digest or sha(ref["path"])!=ref["sha256"]:raise ValueError("source pin mismatch")
  old=load(path);parent=load(ref["path"]);out,changes=canonicalize(parent,old,role)
  folder=a.output/role;folder.mkdir();target=folder/("candidate-"+role+"-csr-canonical.npz")
  np.savez_compressed(target,**out);actual=load(target)
  if set(actual)!=set(old):raise ValueError("candidate member set changed")
  for key in actual:
   if actual[key].shape!=out[key].shape or actual[key].dtype!=out[key].dtype or actual[key].tobytes()!=out[key].tobytes():
    raise ValueError("canonical output readback differs: "+key)
  result["roles"][role]={"input_path":str(path),"input_sha256":digest,"output_path":str(target),"output_sha256":sha(target),
   "immediate_parent_path":ref["path"],"immediate_parent_sha256":ref["sha256"],"changed_members":changes,
   "member_inventory_unchanged":True,"all_flat_value_bytes_identical":True,"all_nonchanged_members_bit_identical":True,
   "points_dtype_shape_bits_identical":True,"faces_dtype_shape_bits_identical":True,
   "geometry_predicates_rerun":False,"no_current_row_metadata_reinterpretation":True}
 (a.output/"certificate.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":result["status"],"roles":{k:{x:v[x] for x in ("output_path","output_sha256")} for k,v in result["roles"].items()}}),flush=True)
if __name__=="__main__":main()
