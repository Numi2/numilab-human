"""Compose the bounded component-4 DP-cluster patch without touching 222."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[1]
IN=H/"complete-surface-cavity-222-root/run-001/skin-composed-candidate.npz"
IN_SHA="cb4c57dd4f281edfbcf9361f1b848dc52a91793978d138c19540eff50efa079d"
IN_REPORT=H/"complete-surface-cavity-222-root/run-001/report.json"
SHELL=H/"complete-surface-cavity-223-root/run-001/component4-dp-cluster-11face-shell.npz"
SHELL_SHA="8c297fe0279603d0ee96042c3d1342e64db6e003b6bf87e994820861560c1d07"
SHELL_REPORT=H/"complete-surface-cavity-223-root/run-001/shell-preflight.json"
MILP=H/"complete-surface-cavity-223-root/run-001/milp-run-001/component4-dp-cluster-11face-milp-proposal.npz"
MILP_SHA="a1b92e2d40d80be9e632f40722b9b6a71eb2ecb2d0efa65f422b44cd5962f69e"
MILP_REPORT=H/"complete-surface-cavity-223-root/run-001/milp-run-001/report.json"
SOURCE214=H/"complete-surface-quotient-214-root/run002/skin-complete-surface-quotient-remesh-input.npz"
SOURCE214_SHA="42d438fba30b43243cafa1e9f33fa2b247404c1b979757b724c7e51132f62ec8"
OWNER=H/"transfer_surface_contractions_to_volume_209_root.py"
QUALITY=H/"fixed_boundary_volume_flip_198_root.py"
QUALITY_SHA="650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
OUT=H/"complete-surface-cavity-224-root/run-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
for path,pin in ((IN,IN_SHA),(SHELL,SHELL_SHA),(MILP,MILP_SHA),(SOURCE214,SOURCE214_SHA),(QUALITY,QUALITY_SHA)):
 if sha(path)!=pin:raise SystemExit("input hash mismatch: "+str(path))
if OUT.exists():raise SystemExit("refusing to overwrite "+str(OUT))
in_report=json.loads(IN_REPORT.read_text());shell_report=json.loads(SHELL_REPORT.read_text());milp_report=json.loads(MILP_REPORT.read_text())
if in_report.get("output_sha256")!=IN_SHA or shell_report.get("shell_sha256")!=SHELL_SHA or milp_report.get("proposal_sha256")!=MILP_SHA or milp_report.get("status")!="PASS_EXACT_ORIENTED_CAVITY_TETRAHEDRALIZATION":raise SystemExit("source report pins/status mismatch")
OUT.mkdir(parents=True)
q=load("quality198_224",QUALITY);m=load("boundary209_224",OWNER);audit=load("audit222_224",H/"complete-surface-cavity-220-root/compose_and_audit_skin_222.py")
with np.load(IN,allow_pickle=False) as z:src={k:z[k] for k in z.files}
with np.load(SHELL,allow_pickle=False) as z:sh={k:z[k] for k in z.files}
with np.load(MILP,allow_pickle=False) as z:patch={k:z[k] for k in z.files}
with np.load(SOURCE214,allow_pickle=False) as z:base={k:z[k] for k in z.files}
p=src["points_world_m"];old_t=src["tetrahedra"].astype(np.int64);old_m=src["material_indices"].astype(np.int32)
ret=sh["source_222_output_tet_rows"].astype(np.int64);base_rows=sh["source_input_tet_rows"].astype(np.int64)
if p.dtype!=np.float32 or not np.array_equal(p,src["points_world_m"]):raise SystemExit("source point bytes changed")
if not np.array_equal(old_t[ret],base["tetrahedra"][base_rows]):raise SystemExit("retired patch parent rows are not the pinned 214 source connectivity")
if not np.array_equal(old_m[ret],np.full(len(ret),15,np.int32)) or len(ret)!=150:raise SystemExit("retired patch material or count mismatch")
g=sh["global_point_rows"].astype(np.int64);local_t=patch["tetrahedra"].astype(np.int64);global_t=patch["global_tetrahedra"].astype(np.int64);new_m=patch["material_indices"].astype(np.int32)
if not np.array_equal(patch["points_world_m"],p[g]) or not np.array_equal(global_t,g[local_t]):raise SystemExit("patch point map mismatch")
if not np.array_equal(patch["source_222_output_tet_rows"],ret) or len(global_t)!=122 or not np.all(new_m==15):raise SystemExit("proposal row/material contract mismatch")
keep=np.ones(len(old_t),bool);keep[ret]=False;kept_rows=np.flatnonzero(keep)
out_t=np.concatenate((old_t[kept_rows],global_t));out_m=np.concatenate((old_m[kept_rows],new_m))
out_to_in=np.concatenate((kept_rows,np.full(len(global_t),-1,np.int64)))
out_patch=np.concatenate((np.full(len(kept_rows),-1,np.int32),np.zeros(len(global_t),np.int32)))
if out_t.shape!=(len(old_t)-150+122,4) or out_m.shape!=(len(out_t),):raise SystemExit("output row counts invalid")
sorted_t=np.sort(out_t,axis=1);_,dup_counts=np.unique(sorted_t,axis=0,return_counts=True);dups=int(np.sum(dup_counts>1))
try:out_b=m.boundary(out_t);boundary_error=None
except Exception as exc:out_b=np.empty((0,3),np.int64);boundary_error=repr(exc)
source_b=m.boundary(old_t)
if not np.array_equal(m.boundary_identity(source_b),m.boundary_identity(src["boundary_faces"])):raise SystemExit("222 source boundary pin mismatch")
source_closed,source_edge_fail=audit.boundary_closed(source_b);out_closed,out_edge_fail=audit.boundary_closed(out_b)
desired=src["desired_boundary_faces"].astype(np.int64)
key=lambda f:tuple(sorted(map(int,f)))
desired_keys={key(f) for f in desired};source_keys={key(f) for f in source_b};out_keys={key(f) for f in out_b}
missing_before=desired_keys-source_keys;extra_before=source_keys-desired_keys;missing_after=desired_keys-out_keys;extra_after=out_keys-desired_keys
# All 11 intended DP faces must now occur on the oriented global boundary; all 17 replaced extras must be gone.
want=sh["desired_dp_faces_global"].astype(np.int64);retired_extra=sh["selected_extra_boundary_faces_global"].astype(np.int64)
out_identity=m.boundary_identity(out_b);identity_rows={tuple(map(int,row[:3])):int(row[3]) for row in out_identity}
def oriented_sign(f):
 a,b,c=map(int,f);inv=(a>b)+(a>c)+(b>c);return 1 if inv%2==0 else -1
want_present=sum(key(f) in out_keys for f in want);want_oriented=sum(identity_rows.get(tuple(sorted(map(int,f))),0)==oriented_sign(f) for f in want)
extras_left=sum(key(f) in out_keys for f in retired_extra)
# Exact signed source-cavity volume equals exact patch volume and exact shell boundary volume.
local_ip=q.integer_points(sh["points_world_m"])
local_lookup={int(v):i for i,v in enumerate(g)}
old_local=old_t[ret]
old_local_idx=np.asarray([[local_lookup[int(x)] for x in tet] for tet in old_local],np.int64)
source_volume_det=sum(q.exact_det(local_ip,tet) for tet in old_local_idx)
new_volume_det=sum(q.exact_det(local_ip,tet) for tet in local_t)
shell_volume_det=sum(q.exact_det([(0,0,0)]+local_ip,[0,*map(int,f)]) for f in sh["faces"])
volume_equal=(source_volume_det==new_volume_det==shell_volume_det)
# Full candidate exact Float32 sign and native quality audit; retained rows are unchanged and new patch was exact-certified by the 220 owner.
quality=audit.exact_audit(p,out_t,q)
material_counts={str(int(k)):int(v) for k,v in zip(*np.unique(out_m,return_counts=True))}
checks={"retired_rows_exact_source_connectivity":True,"point_coordinates_unchanged":True,"new_patch_material_15":bool(np.all(new_m==15)),"duplicate_tetrahedra_zero":dups==0,"source_and_output_closed_oriented":bool(source_closed and out_closed and not source_edge_fail and not out_edge_fail),"all_11_dp_target_facets_present":want_present==len(want),"all_11_dp_target_facets_oriented":want_oriented==len(want),"all_17_selected_old_extras_removed":extras_left==0,"exact_cavity_volume_conserved":volume_equal,"all_F32_cells_exact_positive":quality["sign_counts"]["positive"]==len(out_t) and quality["sign_counts"]["negative"]==0 and quality["sign_counts"]["zero"]==0}
status="PASS_LOCAL_COMPONENT4_DP_CLUSTER_COMPOSITION" if all(checks.values()) and boundary_error is None else "REJECT_LOCAL_COMPONENT4_DP_CLUSTER_COMPOSITION"
# Retain even a failed composition candidate separately for review.
candidate=OUT/"skin-component4-dp-cluster-composed-candidate.npz"
np.savez_compressed(candidate,points_world_m=p,tetrahedra=out_t,material_indices=out_m,boundary_faces=out_b,desired_boundary_faces=desired,retained_input_candidate_tet_rows=kept_rows,retired_input_candidate_tet_rows=ret,tetrahedron_output_to_input_candidate_row=out_to_in,tetrahedron_output_patch_ids=out_patch,patch_global_point_rows=g,patch_output_tetrahedra_global_node_rows=global_t,patch_material_indices=new_m)
report={"schema":"numi.skin.component4-dp-cluster-composition-224.v1","status":status,"physical_admission":False,"native_steps":0,"input_candidate_path":str(IN),"input_candidate_sha256":sha(IN),"input_candidate_report_sha256":sha(IN_REPORT),"source_214_path":str(SOURCE214),"source_214_sha256":sha(SOURCE214),"shell_path":str(SHELL),"shell_sha256":sha(SHELL),"shell_preflight_report_sha256":sha(SHELL_REPORT),"milp_proposal_path":str(MILP),"milp_proposal_sha256":sha(MILP),"milp_report_sha256":sha(MILP_REPORT),"native_quality_owner_sha256":sha(QUALITY),"boundary_owner_sha256":sha(OWNER),"retired_source_222_output_tet_rows":len(ret),"new_patch_tetrahedra":len(global_t),"output_tetrahedra":len(out_t),"output_material_counts":material_counts,"duplicate_tetrahedra":dups,"boundary_extraction_error":boundary_error,"source_boundary_faces":len(source_b),"output_boundary_faces":len(out_b),"source_boundary_closed_oriented_chain":bool(source_closed),"source_boundary_edge_failures":source_edge_fail,"output_boundary_closed_oriented_chain":bool(out_closed),"output_boundary_edge_failures":out_edge_fail,"desired_boundary_faces":len(desired),"boundary_missing_before":len(missing_before),"boundary_extra_before":len(extra_before),"boundary_missing_after":len(missing_after),"boundary_extra_after":len(extra_after),"dp_target_facets_added":int(want_present),"dp_target_facets_oriented":int(want_oriented),"selected_extra_boundary_faces_removed":int(len(retired_extra)-extras_left),"exact_cavity_volume_det_numerator":{"source":str(source_volume_det),"patch":str(new_volume_det),"shell":str(shell_volume_det),"equal":bool(volume_equal)},"exact_F32_sign_and_native_quality":quality,"checks":checks,"candidate_path":str(candidate),"candidate_sha256":sha(candidate)}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="exact_F32_sign_and_native_quality"},indent=2));print(json.dumps({"exact_F32_sign_and_native_quality":quality},indent=2))
