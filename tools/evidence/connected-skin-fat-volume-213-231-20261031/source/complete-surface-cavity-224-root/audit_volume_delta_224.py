from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
IN=H/"complete-surface-cavity-222-root/run-001/skin-composed-candidate.npz"
OUT=H/"complete-surface-cavity-224-root/run-002/skin-component4-dp-cluster-composed-candidate.npz"
SHELL=H/"complete-surface-cavity-223-root/run-001/component4-dp-cluster-11face-shell.npz"
REPORT=H/"complete-surface-cavity-224-root/run-002/report.json"
OUTDIR=H/"complete-surface-cavity-224-root/run-003-volume-delta-audit"
PINS={
 str(IN):"cb4c57dd4f281edfbcf9361f1b848dc52a91793978d138c19540eff50efa079d",
 str(OUT):"eb90c2ac505f426886f03ca7867335794618f791e88cad3b0e67e5165f914f74",
 str(SHELL):"8c297fe0279603d0ee96042c3d1342e64db6e003b6bf87e994820861560c1d07",
}
def sha(p): return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module
for path,pin in PINS.items():
 if sha(path)!=pin: raise SystemExit("input hash mismatch: "+path)
if OUTDIR.exists(): raise SystemExit("refusing to overwrite "+str(OUTDIR))
owner=load("boundary_delta_owner224",H/"transfer_surface_contractions_to_volume_209_root.py")
q=load("quality_delta_owner224",H/"fixed_boundary_volume_flip_198_root.py")
with np.load(IN,allow_pickle=False) as z: src={k:z[k] for k in z.files}
with np.load(OUT,allow_pickle=False) as z: dst={k:z[k] for k in z.files}
with np.load(SHELL,allow_pickle=False) as z: sh={k:z[k] for k in z.files}
old_t=src["tetrahedra"].astype(np.int64)
out_t=dst["tetrahedra"].astype(np.int64)
ret=sh["source_222_output_tet_rows"].astype(np.int64)
g=sh["global_point_rows"].astype(np.int64)
patch_rows=np.flatnonzero(dst["tetrahedron_output_patch_ids"]==0)
new_t=out_t[patch_rows]
old_patch_t=old_t[ret]
target_global=g[sh["faces"].astype(np.int64)]
old_boundary=owner.boundary(old_patch_t)
new_boundary=owner.boundary(new_t)
def chain_map(faces):
 ident=owner.boundary_identity(np.asarray(faces,np.int64))
 return {tuple(map(int,row[:3])):int(row[3]) for row in ident}
def delta(a,b):
 keys=set(a)|set(b)
 return {k:b.get(k,0)-a.get(k,0) for k in keys if b.get(k,0)!=a.get(k,0)}
old_chain=chain_map(old_boundary)
new_chain=chain_map(new_boundary)
target_chain=chain_map(target_global)
global_delta=delta(chain_map(src["boundary_faces"]),chain_map(dst["boundary_faces"]))
local_delta=delta(old_chain,new_chain)
used=np.unique(np.concatenate((old_patch_t.ravel(),new_t.ravel(),old_boundary.ravel(),new_boundary.ravel(),target_global.ravel())))
lookup={int(v):i for i,v in enumerate(used)}
ip=q.integer_points(src["points_world_m"][used])
def tet_volume_det(tets):
 return sum(q.exact_det(ip,[lookup[int(v)] for v in tet]) for tet in tets)
def boundary_flux_det(faces):
 p=[(0,0,0)]+ip
 return sum(q.exact_det(p,[0]+[lookup[int(v)]+1 for v in face]) for face in faces)
old_v=tet_volume_det(old_patch_t)
new_v=tet_volume_det(new_t)
old_flux=boundary_flux_det(old_boundary)
new_flux=boundary_flux_det(new_boundary)
target_flux=boundary_flux_det(target_global)
input_report=json.loads(REPORT.read_text())
checks={
 "new_patch_boundary_equals_target_shell":new_chain==target_chain,
 "source_cell_volume_equals_source_patch_boundary_flux":old_v==old_flux,
 "new_cell_volume_equals_target_shell_flux":new_v==target_flux,
 "exact_cell_volume_delta_equals_target_boundary_flux_delta":new_v-old_v==target_flux-old_flux,
 "global_boundary_delta_equals_local_patch_boundary_delta":global_delta==local_delta,
}
report={
 "schema":"numi.skin.component4-dp-cluster-volume-delta-audit.v1",
 "status":"PASS_EXACT_LOCAL_REPLACEMENT_BOUNDARY_DELTA" if all(checks.values()) else "REJECT_EXACT_LOCAL_REPLACEMENT_BOUNDARY_DELTA",
 "physical_admission":False,"native_steps":0,
 "input_222_path":str(IN),"input_222_sha256":sha(IN),
 "output_224_path":str(OUT),"output_224_sha256":sha(OUT),
 "shell_path":str(SHELL),"shell_sha256":sha(SHELL),
 "composition_report_sha256":sha(REPORT),
 "retired_tet_rows":len(old_patch_t),"new_patch_tet_rows":len(new_t),
 "old_patch_boundary_faces":len(old_boundary),"new_patch_boundary_faces":len(new_boundary),"target_shell_faces":len(target_global),
 "global_boundary_delta_equals_local_patch_boundary_delta":checks["global_boundary_delta_equals_local_patch_boundary_delta"],
 "global_boundary_delta_face_count":len(global_delta),"local_patch_boundary_delta_face_count":len(local_delta),
 "old_patch_tet_volume_det_numerator":str(old_v),"old_patch_boundary_flux_det_numerator":str(old_flux),
 "new_patch_tet_volume_det_numerator":str(new_v),"new_patch_boundary_flux_det_numerator":str(new_flux),
 "target_shell_flux_det_numerator":str(target_flux),
 "exact_volume_delta_det_numerator":str(new_v-old_v),
 "exact_boundary_flux_delta_det_numerator":str(target_flux-old_flux),
 "checks":checks,
 "global_boundary_missing_after":input_report["boundary_missing_after"],
 "global_boundary_extra_after":input_report["boundary_extra_after"],
 "global_exact_positive_cells":input_report["exact_F32_sign_and_native_quality"]["sign_counts"]["positive"],
 "global_exact_nonpositive_cells":input_report["exact_F32_sign_and_native_quality"]["sign_counts"]["negative"]+input_report["exact_F32_sign_and_native_quality"]["sign_counts"]["zero"],
 "native_quality_failures":input_report["exact_F32_sign_and_native_quality"]["native_quality_failures"],
}
OUTDIR.mkdir(parents=True)
(OUTDIR/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
