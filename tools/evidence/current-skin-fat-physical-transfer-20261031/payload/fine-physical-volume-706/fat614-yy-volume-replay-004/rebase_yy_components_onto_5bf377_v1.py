#!/usr/bin/env python3
"""Rebase the seven audited 004 Fat boundary fills onto the frozen 5bf19 D307/I377 branch.

This is a source-only topology proposal. It does not claim source330 mass mapping,
native execution, or physical admission.
"""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys, time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
P5=H/"paired-I-interface-377-root/run002/fat-coupled-I-quotient-volume-proposal.npz"
P5R=H/"paired-I-interface-377-root/run002/report.json"
V4=H/"fat614-yy-volume-replay-004/run-010-composed-boundary-components-v4/fat-d-run005-boundary-transfer-volume-proposal-v4.npz"
V4R=H/"fat614-yy-volume-replay-004/run-010-composed-boundary-components-v4/report.json"
D4906=H/"fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz"
D4906R=H/"fat614-yy-volume-replay-002/report.json"
OUT=H/"fat614-yy-volume-replay-004/run-011-rebased-377-007-components"
EXPECTED={
 "parent_5bf":("5bf19e5c8eeccc9c081beb8753d19e3727a12b067e0a0774c4e951ae39a9fb91",P5),
 "parent_5bf_report":("baa09bce6e6995268ec5a932be530573b8de69bb202025568ad41dd9e5325486",P5R),
 "v4_candidate":("4075370b75cbabf0dbad2cf9abba2da7b69d85f8ed0a09dd326180eac5cbe024",V4),
 "v4_report":("009e163ce496fd1ed365362a4d97e3e8ff70653f963f811216d80ae82b7c96e1",V4R),
 "d4906_parent":("d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f",D4906),
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
for name,(want,p) in EXPECTED.items():
 got=sha(p)
 if got!=want: raise ValueError(f"{name} SHA mismatch: {got}")
p5r=json.loads(P5R.read_text());v4r=json.loads(V4R.read_text())
if p5r.get("status")!="PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT" or p5r.get("physical_admission") is not False or p5r.get("native_steps")!=0:
 raise ValueError("5bf parent is not the pinned source-only I377 pass")
if v4r.get("whole_candidate",{}).get("candidate_sha256")!=EXPECTED["v4_candidate"][0] or v4r.get("physical_admission") is not False:
 raise ValueError("v4 report does not bind the pinned source-only candidate")
r=loadmod("fat614_remesh296_007_for_377_rebase",H/"remesh_fat614_compact_cavities_296_007.py")
m=r.m
caps=loadmod("fat614_caps288_for_377_rebase",H/"direct_connected_skin_caps_288_root.py")
z5=load_npz(P5);zv=load_npz(V4);zd=load_npz(D4906)
p5=np.asarray(z5["points_world_m"],np.float32);t5=np.asarray(z5["tetrahedra"],np.int64);mat5=np.asarray(z5["material_indices"],np.int32)
pd=np.asarray(zd["points_world_m"],np.float32);td=np.asarray(zd["tetrahedra"],np.int64);md=np.asarray(zd["material_indices"],np.int32)
pv=np.asarray(zv["points_world_m"],np.float32);tv=np.asarray(zv["tetrahedra"],np.int64);mv=np.asarray(zv["material_indices"],np.int32)
new_flat=np.asarray(zv["fat614_boundary_transfer_004_tetgen_output_tets_global_rows"],np.int64)
old_d=np.asarray(zv["fat614_boundary_transfer_004_deleted_d4906_output_rows"],np.int64)
components=np.asarray(zv["fat614_boundary_transfer_004_component_ids"],np.int64)
comp_offsets=np.asarray(zv["fat614_boundary_transfer_004_partial_4602_parent_rows_by_component_offsets"],np.int64)
tet_offsets=np.asarray(zv["fat614_boundary_transfer_004_tetgen_output_tet_offsets"],np.int64)
node_offsets=np.asarray(zv["fat614_boundary_transfer_004_tetgen_output_node_offsets"],np.int64)
node_rows=np.asarray(zv["fat614_boundary_transfer_004_tetgen_node_global_rows"],np.int64)
if not np.array_equal(p5,pd) or not np.array_equal(pv[:len(pd)],pd):
 raise ValueError("5bf/d4906/v4 base point coordinate prefixes are not bitwise identical")
if len(pv)-len(pd)!=21: raise ValueError("unexpected v4 added-point suffix")
# Compose 5bf output cell rows -> d4906 source rows through its two frozen maps.
mid=np.asarray(z5["retained_input_cell_rows"],np.int64)
mid_to_d=np.asarray(z5["output_cell_to_input_cell"],np.int64)
if len(mid)!=len(t5) or mid.min()<0 or mid.max()>=len(mid_to_d):raise ValueError("5bf cell map shape/domain invalid")
out5_to_d=mid_to_d[mid]
if np.any(out5_to_d<0) or len(np.unique(out5_to_d))!=len(out5_to_d):raise ValueError("5bf composed cell-source map is not a unique total map")
d_to_out5=np.full(len(td),-1,np.int64);d_to_out5[out5_to_d]=np.arange(len(out5_to_d),dtype=np.int64)
if len(old_d)!=9645 or len(np.unique(old_d))!=len(old_d):raise ValueError("v4 selected D4906 parent rows malformed")
old5=d_to_out5[old_d]
if np.any(old5<0) or len(np.unique(old5))!=len(old5):raise ValueError("v4 selected source cavities are missing/colliding in 5bf")
if not np.array_equal(t5[old5],td[old_d]):raise ValueError("selected source tet corner order changed in 5bf")
if not np.array_equal(mat5[old5],md[old_d]):raise ValueError("selected source material changed in 5bf")
if len(new_flat)!=10590 or np.any(new_flat<0) or int(new_flat.max())>=len(pv):raise ValueError("v4 replacement tet rows outside v4 point domain")
if not all(int(v["material"])==16 for v in v4r["components"].values()):raise ValueError("v4 component receipts do not all declare Fat material 16")
# For candidate construction, v4's point namespace is the exact d4906 prefix plus its 21 added points.
pnew=np.vstack([p5,pv[len(pd):]]).astype(np.float32,copy=False)
if not np.array_equal(pnew,pv):raise ValueError("composed point array differs from frozen v4 point array")
keep=np.ones(len(t5),bool);keep[old5]=False
kept_rows=np.flatnonzero(keep).astype(np.int64)
newt=new_flat.copy()
newm=np.full(len(newt),16,np.int32)
tnew=np.vstack([t5[keep],newt]).astype(np.int64,copy=False)
mnew=np.concatenate([mat5[keep],newm]).astype(np.int32,copy=False)
if len(tnew)!=len(t5)-len(old5)+len(newt):raise ValueError("cell count arithmetic failure")
# Verify complete point-index validity and no duplicate tetrahedra.
if int(tnew.min())<0 or int(tnew.max())>=len(pnew):raise ValueError("new tet node index outside point array")
canon=np.sort(tnew,axis=1)
if len(np.unique(canon,axis=0))!=len(canon):raise ValueError("duplicate tetrahedron after composition")
# Import the same canonical incidence/quality/exact-geometry owners used by v4.
actual5=m.boundary(t5)
stored5=np.asarray(z5["boundary_faces"],np.int64)
def facecounter(f):
 a=m.boundary_identity(np.asarray(f,np.int64).reshape(-1,3))
 return Counter(tuple(map(int,row)) for row in a)
if facecounter(actual5)!=facecounter(stored5):raise ValueError("5bf stored boundary does not replay from tets")
bnew=m.boundary(tnew)
desired=np.asarray(z5["desired_boundary_faces"],np.int64)
# Check each component's new mesh is exactly its previously audited shell.
shell_all=np.asarray(zv["fat614_boundary_transfer_004_expected_local_shell_faces"],np.int64)
shell_off=np.asarray(zv["fat614_boundary_transfer_004_expected_local_shell_offsets"],np.int64)
if len(components)!=7 or len(comp_offsets)!=8 or len(tet_offsets)!=8 or len(node_offsets)!=8 or len(shell_off)!=8:
 raise ValueError("component CSR map lengths are inconsistent")
component_local=[]
for i,c in enumerate(components):
 c=int(c);a,b=int(tet_offsets[i]),int(tet_offsets[i+1])
 ta,tb=int(shell_off[i]),int(shell_off[i+1])
 out_patch=new_flat[a:b]
 got=facecounter(m.boundary(out_patch))
 want=facecounter(shell_all[ta:tb])
 if got!=want:raise ValueError(f"component {c} output boundary differs from frozen audited local shell")
 p0,p1=int(comp_offsets[i]),int(comp_offsets[i+1])
 old_rows=old_d[p0:p1]
 old_out=old5[p0:p1]
 if len(old_rows)!=len(old_out):raise ValueError("component old-row map mismatch")
 component_local.append({"component_id":c,"d4906_parent_rows":old_rows,"parent_5bf_rows":old_out,"new_tets":out_patch,"new_global_tet_rows":np.arange(len(t5)-len(old5)+a,len(t5)-len(old5)+b,dtype=np.int64)})
# Bidirectional cut identity and opposition against the actual new 5bf retained shell.
old_cut=np.asarray(zv["fat614_boundary_transfer_004_old_internal_cut_faces"],np.int64)
new_cut=np.asarray(zv["fat614_boundary_transfer_004_new_internal_cut_faces"],np.int64)
if facecounter(old_cut)!=facecounter(new_cut):raise ValueError("v4 old/new internal cut sets differ")
retained_boundary=m.boundary(t5[keep])
retained_cut=facecounter(retained_boundary)
new_cut_counter=facecounter(new_cut)
for key,count in new_cut_counter.items():
 opp=key[:3]+(-key[3],)
 if retained_cut.get(opp,0)<count:raise ValueError(f"new internal cut not opposed by retained 5bf tets: {key}")
# Ensure cavity source rows disjoint from any 5bf nonidentity rewrite (ordered tet tuple).
changed_parent_rows=[]
for drow,frow in zip(old_d,old5):
 if not np.array_equal(t5[frow],td[drow]):changed_parent_rows.append((int(drow),int(frow)))
if changed_parent_rows:raise ValueError(f"cavity rows overlap changed 5bf operations: {changed_parent_rows[:4]}")
# Exact SAT between the replacements and all current retained 5bf cells, plus all patches.
tree=caps.TetBVH(p5,t5)
sat=caps.check_intersections(pnew,t5,new_flat,old5,[],tree,max_pairs=20_000_000)
if not sat.get("pass"):raise ValueError("exact retained/interpatch SAT failed: "+str(sat))
# Exact sign validation: inherited parent cells are unchanged; validate every cell in the final namespace.
ip=m.m.integer_points(pnew)
exact_signs=[m.m.exact_det(ip,[int(x) for x in tet]) for tet in tnew]
nonpositive=int(sum(v<=0 for v in exact_signs))
if nonpositive:raise ValueError(f"final rebased candidate has {nonpositive} exact nonpositive F32 tets")
# Native measurements are diagnostic only, not an admission gate.
det_before,q_before=m.m.measurements(p5,t5)
det_after,q_after=m.m.measurements(pnew,tnew)
quality_before=int(np.sum(~m.m.passes(det_before,q_before)))
quality_after=int(np.sum(~m.m.passes(det_after,q_after)))
# Exact global boundary-flux delta must equal the sum of the seven audited local volume deltas.
flux5=int(r.exact_volume(p5,actual5));fluxnew=int(r.exact_volume(pnew,bnew))
sum_local=sum(int(v["exact_local_volume_delta_det6"]) for v in v4r["components"].values())
if fluxnew-flux5!=sum_local:raise ValueError("whole exact boundary-flux delta does not equal audited seven-component volume deltas")
# Report against the *latest* 5bf desired surface; do not substitute the old v4 target.
def counter_delta(actual,wanted):
 got=facecounter(actual);want=facecounter(wanted);missing=want-got;extra=got-want
 return {"missing_oriented_records":int(sum(missing.values())),"extra_oriented_records":int(sum(extra.values())),
 "missing_samples":[list(k) for k in sorted(missing)[:20]],"extra_samples":[list(k) for k in sorted(extra)[:20]]}
diff_before=counter_delta(actual5,desired);diff_after=counter_delta(bnew,desired)
# Explicit maps and changed-row masks for downstream composition.
parent5_to_output=np.full(len(t5),-1,np.int64);parent5_to_output[kept_rows]=np.arange(len(kept_rows),dtype=np.int64)
out_to_parent5=np.concatenate([kept_rows,np.full(len(newt),-1,np.int64)])
d4906_to_output=np.full(len(td),-1,np.int64)
# Only claim direct D4906 identity for retained ordered tets; replaced/rewritten rows remain -1.
old5_mask=np.zeros(len(t5),bool);old5_mask[old5]=True
identity_rows=(np.all(t5==td[out5_to_d],axis=1)&(mat5==md[out5_to_d]))&~old5_mask
d4906_to_output[out5_to_d[identity_rows]]=parent5_to_output[identity_rows]
# Compose source row masks per component.
comp_source_rows=np.asarray(old_d,np.int64)
comp_parent_rows=np.asarray(old5,np.int64)
comp_new_rows=np.arange(len(kept_rows),len(tnew),dtype=np.int64)
new_global_nodes=np.unique(newt[newt>=len(pd)]).astype(np.int64)
# Validate that all added points are actually used and each point is from v4 suffix.
if not np.array_equal(new_global_nodes,np.arange(len(pd),len(pnew),dtype=np.int64)):
 raise ValueError("the 21 appended v4 points are not exactly the points used by new cavity tets")
# Save output only after all validations pass.
if OUT.exists():raise FileExistsError(f"refusing to overwrite {OUT}")
OUT.mkdir(parents=True)
outnpz=OUT/"fat-D307-I377-plus-YY-components-source-only.npz"
payload={}
# Preserve the exact source candidate's metadata/maps with explicit namespace.
for k,v in z5.items():
 if k in {"points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces","desired_boundary_faces","desired_boundary_source_rows","desired_boundary_degenerate_source_rows"}:continue
 payload["source_5bf_"+k]=v
# Preserve v4 operation provenance/diagnostics separately, avoiding stale top-level row domains.
for k,v in zv.items():
 if k in {"points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces","desired_boundary_faces","retained_input_cell_rows","replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets","replacement_tetrahedra"}:continue
 payload["source_v4_"+k]=v
payload.update({
 "points_world_m":pnew,"points_world_m_f32":pnew.copy(),"tetrahedra":tnew,"material_indices":mnew,
 "boundary_faces":bnew,"desired_boundary_faces":desired.copy(),
 "desired_boundary_source_rows":np.asarray(z5["desired_boundary_source_rows"],np.int64).copy(),
 "desired_boundary_degenerate_source_rows":np.asarray(z5["desired_boundary_degenerate_source_rows"],np.int64).copy(),
 "fat_boundary_rebase_377_schema":np.asarray("numi.fat614.d307-i377-plus-audited-yy-cavities.v1"),
 "fat_boundary_rebase_377_parent5bf_cell_row_to_output":parent5_to_output,
 "fat_boundary_rebase_377_output_to_parent5bf_cell_row":out_to_parent5,
 "fat_boundary_rebase_377_d4906_cell_to_output_if_exact_ordered_identity":d4906_to_output,
 "fat_boundary_rebase_377_d4906_cell_to_5bf_source_row":out5_to_d,
 "fat_boundary_rebase_377_retired_parent5bf_cell_rows":comp_parent_rows,
 "fat_boundary_rebase_377_retired_d4906_cell_rows":comp_source_rows,
 "fat_boundary_rebase_377_new_output_cell_rows":comp_new_rows,
 "fat_boundary_rebase_377_component_ids":components.copy(),
 "fat_boundary_rebase_377_component_parent_offsets":comp_offsets.copy(),
 "fat_boundary_rebase_377_component_new_tet_offsets":tet_offsets.copy(),
 "fat_boundary_rebase_377_component_retired_parent5bf_cell_rows":comp_parent_rows.copy(),
 "fat_boundary_rebase_377_component_retired_d4906_cell_rows":comp_source_rows.copy(),
 "fat_boundary_rebase_377_component_new_tetrahedra_global_rows":new_flat.copy(),
 "fat_boundary_rebase_377_component_added_point_global_rows":new_global_nodes,
 "fat_boundary_rebase_377_v4_expected_local_shell_offsets":shell_off.copy(),
 "fat_boundary_rebase_377_v4_expected_local_shell_faces":shell_all.copy(),
 "fat_boundary_rebase_377_old_internal_cut_faces":old_cut.copy(),
 "fat_boundary_rebase_377_new_internal_cut_faces":new_cut.copy(),
})
np.savez_compressed(outnpz,**payload)
report={
 "schema":"numi.fat614.d307-i377-plus-audited-yy-cavities.v1",
 "status":"PASS_REBASED_AUDITED_YY_BOUNDARY_COMPONENTS_ON_D307_I377_SOURCE_ONLY",
 "source_only":True,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "producer_script_path":str(Path(__file__)),"producer_script_sha256":sha(Path(__file__)),
 "inputs":{k:{"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size} for k,(_,p) in EXPECTED.items()},
 "parent_5bf":{
   "points":len(p5),"cells":len(t5),"cell_map_is_unique_total_to_d4906":True,
   "source_cell_map_sha256":array_sha(out5_to_d),"point_coordinates_bitwise_equal_to_d4906":True,
   "candidate_role":p5r["status"]
 },
 "composed_components":{
   str(int(c)):{
    "retired_d4906_rows":int(comp_offsets[i+1]-comp_offsets[i]),
    "retired_parent5bf_rows_sha256":array_sha(comp_parent_rows[comp_offsets[i]:comp_offsets[i+1]]),
    "retired_d4906_rows_sha256":array_sha(comp_source_rows[comp_offsets[i]:comp_offsets[i+1]]),
    "new_tets":int(tet_offsets[i+1]-tet_offsets[i]),
    "new_output_rows":[int(len(kept_rows)+tet_offsets[i]),int(len(kept_rows)+tet_offsets[i+1])],
    "new_points":[int(x) for x in np.unique(new_flat[tet_offsets[i]:tet_offsets[i+1]][new_flat[tet_offsets[i]:tet_offsets[i+1]]>=len(pd)])],
    "local_boundary_matches_pinned_v4_shell":True
   } for i,c in enumerate(components)
 },
 "whole_candidate":{
   "path":str(outnpz),"sha256":sha(outnpz),"size_bytes":outnpz.stat().st_size,
   "points":len(pnew),"cells":len(tnew),"parent_5bf_retired_cells":len(old5),
   "new_cells":len(newt),"added_points":len(pnew)-len(p5),
   "exact_F32_nonpositive_cells":nonpositive,"duplicate_tetrahedra":0,
   "stored_boundary_matches_incidence":True,
   "oriented_boundary_mismatch_before_vs_latest_desired":diff_before,
   "oriented_boundary_mismatch_after_vs_latest_desired":diff_after,
   "full_boundary_matches_latest_desired":diff_after["missing_oriented_records"]==0 and diff_after["extra_oriented_records"]==0,
   "old_new_internal_cut_face_counts":[len(old_cut),len(new_cut)],
   "internal_cut_sets_bidirectionally_identical":True,
   "new_cut_faces_opposed_by_untouched_5bf_cells":True,
   "retained_and_interpatch_exact_F32_SAT":sat,
   "exact_global_boundary_flux_delta_det6":fluxnew-flux5,
   "sum_audited_local_delta_det6":sum_local,
   "global_flux_delta_equals_sum_local":True,
   "quality_failures_before_after":[quality_before,quality_after],
   "quality_failure_delta":quality_after-quality_before,
   "F64_metric_nonpositive_after":int(np.sum(det_after<=0)),
 },
 "unresolved":{
   "known_pending_boundary_components":[0,9,12],
   "boundary_mismatch_preserved":True,
   "source330_mass_mapping":"NOT_RUN","native_execution":"NOT_RUN","physical_admission":False
 },
 "limitations":"This rebases seven already audited local cavity fills onto 5bf19 without changing its points or retained cells outside the seven selected cavities. It remains a source-only partial boundary proposal. Components 0, 9, and 12 remain unresolved; current full target boundary mismatch is reported, not waived. No mass mapping, native execution, or physical admission is claimed."
}
report_path=OUT/"report.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"candidate_path":str(outnpz),"candidate_sha256":sha(outnpz),"report_path":str(report_path),"report_sha256":sha(report_path),"components":report["composed_components"],"boundary_before_after":[diff_before,diff_after],"sat":sat,"quality_failures":[quality_before,quality_after],"flux_delta":fluxnew-flux5},indent=2))

