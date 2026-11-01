from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK=ROOT/"current-native-1024-root/current-433-1044"
LAUNCH=ROOT/"current-native-1024-root/current-433-1043/launch-inputs.json"
SHELL=ROOT/"current-shell-1022-native/actual-001"
RECEIPT=SHELL/"receipt.json"
EXPECTED={"receipt":"b5dd9f69a5a61331141ad69f06466225349526b26916de9d57c7bbbbd71ec10f","skin":"1d29d3a07147ab7c7bcf45e017ee6515e0a4e7015dd236dbc7d07407c894c9cb","fat":"2a97964d38c945836a4090a7c9580624b427a0163eda9932370cc6f77277ef66"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def can(tri):
 x=tuple(map(int,tri));return min(x[i:]+x[:i] for i in range(3))
def orient(a,b):
 if can(a)==can(b):return 1
 if can(a)==can((int(b[0]),int(b[2]),int(b[1]))):return -1
 return 0
def check_asset(role, pin, skin, fat):
 path=Path(pin["path"]);digest=pin["sha256"]
 if sha(path)!=digest:raise ValueError(f"{role} current typed SHA mismatch")
 shell=skin if role in ("O","I") else fat
 prefix=f"registered_{role}_"
 srcv=np.asarray(shell[prefix+"source_vertex_rows"],dtype=np.int64)
 shellv=np.asarray(shell[prefix+"shell_node_rows"],dtype=np.int64)
 srctri_shell=np.asarray(shell[prefix+"source_faces"],dtype=np.int64)
 shellrows=np.asarray(shell[prefix+"shell_face_rows"],dtype=np.int64)
 shelltri_stored=np.asarray(shell[prefix+"shell_faces"],dtype=np.int64) if prefix+"shell_faces" in shell.files else np.asarray(shell["faces"],dtype=np.int64)[shellrows]
 if len(np.unique(srcv))!=len(srcv) or len(np.unique(shellv))!=len(shellv):raise ValueError(f"{role} shell vertex map is not injective")
 node_to_source={int(sn):int(vr) for vr,sn in zip(srcv,shellv)}
 with np.load(path,allow_pickle=False) as a:
  afaces=np.asarray(a["faces"],dtype=np.int64)
  identity={k:np.asarray(a[k],dtype=np.int64) for k in ("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id") if k in a.files}
  if not np.array_equal(srctri_shell,afaces):raise ValueError(f"{role} 1022 registered source triangles are not exact typed row order")
  if role=="I":
   source_face_rows=np.asarray(shell["registered_I_source_face_rows"],dtype=np.int64)
   if not np.array_equal(source_face_rows,np.arange(len(afaces),dtype=np.int64)):raise ValueError("I 1022 face-row mapping is not identity")
   if not np.array_equal(np.asarray(shell["registered_I_source_face_ids"],dtype=np.int64),identity["face_source_face_id"]):raise ValueError("I shell face ID row map differs from typed source")
  else:
   source_face_rows=np.arange(len(afaces),dtype=np.int64)
  mapped_shell=np.asarray([[node_to_source[int(n)] for n in tri] for tri in shelltri_stored],dtype=np.int64)
  signs=np.asarray([orient(a,b) for a,b in zip(afaces,mapped_shell)],dtype=np.int8)
  fields_ok={}
  for key,vals in identity.items():
   shell_key=key
   if shell_key in shell.files:
    fields_ok[key]=bool(np.array_equal(vals, np.asarray(shell[shell_key])[shellrows]))
  shell_role=np.asarray(shell["face_source_role"]).astype(str)[shellrows]
  shell_marker=np.asarray(shell["face_marker"],dtype=np.int64)[shellrows]
  shell_face_rows_actual=np.asarray(shell["faces"],dtype=np.int64)[shellrows]
  if not np.array_equal(shell_face_rows_actual,shelltri_stored):raise ValueError(f"{role} stored shell triangles differ from actual shell face rows")
  return {"role":role,"asset":{"path":str(path),"sha256":digest,"faces":len(afaces),"points":len(a["points_world_m_f32"])},"shell_face_rows":len(shellrows),"source_triangle_row_order_exact":True,"source_face_rows_identity":bool(np.array_equal(source_face_rows,np.arange(len(afaces)))),"identity_lanes_equal":fields_ok,"shell_face_roles":{str(x):int(np.count_nonzero(shell_role==x)) for x in np.unique(shell_role)},"shell_face_markers":{str(x):int(np.count_nonzero(shell_marker==x)) for x in np.unique(shell_marker)},"orientation_counts":{"same":int(np.count_nonzero(signs==1)),"reverse":int(np.count_nonzero(signs==-1)),"unmatched":int(np.count_nonzero(signs==0))},"first_source_triangle":afaces[0].tolist(),"first_shell_triangle_as_source_rows":mapped_shell[0].tolist(),"first_face_identity":{k:int(v[0]) for k,v in identity.items()}}
def main():
 if sha(RECEIPT)!=EXPECTED["receipt"] or sha(SHELL/"skin-shell.npz")!=EXPECTED["skin"] or sha(SHELL/"fat-shell.npz")!=EXPECTED["fat"]:raise ValueError("1022 current shell source pin mismatch")
 launch=json.loads(LAUNCH.read_text()); receipt=json.loads(RECEIPT.read_text()); roles={}
 with np.load(SHELL/"skin-shell.npz",allow_pickle=False) as skin, np.load(SHELL/"fat-shell.npz",allow_pickle=False) as fat:
  for role in ("O","I","D"):roles[role]=check_asset(role,launch["role_pins"][role],skin,fat)
 orient=receipt["assembly"]["orientation"]
 expected={"O":1,"I":-1,"D":-1}
 ok=(roles["O"]["orientation_counts"]=={"same":roles["O"]["shell_face_rows"],"reverse":0,"unmatched":0}
  and roles["I"]["orientation_counts"]=={"same":0,"reverse":roles["I"]["shell_face_rows"],"unmatched":0}
  and roles["D"]["orientation_counts"]=={"same":0,"reverse":roles["D"]["shell_face_rows"],"unmatched":0}
  and all(all(x["identity_lanes_equal"].values()) for x in roles.values())
  and orient["O_flipped_to_wall"] is False and orient["I_skin_flipped_to_wall"] is True
  and orient["D_flipped_to_wall"] is True and orient["fat_global_positive_volume_flip"] is False
  and orient["skin_global_positive_volume_flip"] is False)
 report={"schema":"numi.h706.1022-role-surface-orientation-audit.v1","status":"PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS" if ok else "FAIL_ROLE_ORIENTATION_OR_IDENTITY","receipt":{"path":str(RECEIPT),"sha256":sha(RECEIPT)},"shell_assets":{"skin":{"path":str(SHELL/"skin-shell.npz"),"sha256":sha(SHELL/"skin-shell.npz")},"fat":{"path":str(SHELL/"fat-shell.npz"),"sha256":sha(SHELL/"fat-shell.npz")}},"1022_orientation_contract":orient,"source_to_material_outward_sign":expected if ok else None,"roles":roles,"physical_admission":False,"native_steps":0}
 out=TASK/"1022-role-surface-orientation-audit.json";out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n");print(json.dumps(report,sort_keys=True))
 if not ok:raise SystemExit(2)
if __name__=="__main__":main()
