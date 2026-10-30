#!/usr/bin/env python3
import hashlib,json,pathlib
import numpy as np
H=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
T=H/"fat-construction-edge-controls-065-guarded"
src=H/"fat009-batch-cavity-064/fat-partial-proposal.npz"
old=H/"fat009-construction-node-quotient-065/fat-quotient-proposal.npz"
new=H/"fat009-construction-node-quotient-065-guarded/fat-quotient-proposal.npz"
oldr=json.loads((old.parent/"report.json").read_text());newr=json.loads((new.parent/"report.json").read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(src)as z:a={k:z[k]for k in z.files}
with np.load(old)as z:o={k:z[k]for k in z.files}
with np.load(new)as z:n={k:z[k]for k in z.files}
same_keys=set(o)==set(n)
arrays={k:bool(same_keys and o[k].dtype==n[k].dtype and o[k].shape==n[k].shape and np.array_equal(o[k],n[k]))for k in o}
node_map=n["source_candidate_node_to_quotient_node"];ret=n["retained_source_candidate_cell_rows"];rem=n["removed_exact_zero_source_candidate_cell_rows"]
mapped_ret=node_map[a["tetrahedra"][ret]]
mapped_rem=node_map[a["tetrahedra"][rem]]
partition=np.sort(np.r_[ret,rem])
partition_ok=np.array_equal(partition,np.arange(len(a["tetrahedra"])))
rem_zero=bool(np.all(np.any(np.diff(np.sort(mapped_rem,axis=1),axis=1)==0,axis=1)))
rem_zero_material=bool(np.all(a["material_indices"][rem]==16))
ret_map_ok=bool(np.array_equal(mapped_ret,n["tetrahedra"]) and np.array_equal(a["material_indices"][ret],n["material_indices"]))
original_prefix=bool(np.array_equal(n["points_world_m"][:245996].view(np.uint32),a["points_world_m"][:245996].view(np.uint32)))
original_used=bool(np.all(np.isin(np.arange(245996),np.unique(n["tetrahedra"]))))
logical_keys=["source_sha256","source_cells","candidate_cells","removed_cells","source_nodes","candidate_nodes","all_original_source_nodes_and_F32_bits_preserved","no_nonzero_cell_removed","F32_zero_cells","F32_negative_cells","duplicate_cells","nonmanifold_tet_faces","boundary_edges_not_two_incident","boundary_faces"]
logical={k:oldr.get(k)==newr.get(k) for k in logical_keys}
step_checks=len(newr["steps"])==4 and all(x["simplicial_link_condition"] and x["link_intersection_size"]==x["edge_link_size"] for x in newr["steps"])
gates=newr["fail_closed_global_gates"]
checks={"source_hash_matches_064":sha(src)==newr["source_sha256"],"all_npz_keys_equal":same_keys,"all_npz_arrays_byte_value_equal":all(arrays.values()),"logical_report_equal_except_guard_metadata":all(logical.values()),"all_four_link_conditions_pass_and_sizes_equal":step_checks,"source_cell_partition_exact":partition_ok,"removed_cells_have_repeated_quotient_vertices":rem_zero,"removed_cells_material16_only":rem_zero_material,"retained_cells_map_exactly":ret_map_ok,"all_original_points_bitwise_preserved":original_prefix,"all_original_points_still_cell_used":original_used,"all_global_prewrite_gates_true":all(gates.values()),"partial_not_admission":newr["physical_admission"] is False and newr["F32_negative_cells"]==70 and newr["original_PLC_facets"]=="PENDING_CONNECTED_RESTORATION"}
report={"schema":"guarded-construction-quotient-replay-verification-v1","status":"PASS" if all(checks.values())else"FAIL","source_064_path":str(src),"source_064_sha256":sha(src),"frozen_065_candidate_path":str(old),"frozen_065_candidate_sha256":sha(old),"guarded_065_candidate_path":str(new),"guarded_065_candidate_sha256":sha(new),"candidate_archive_hashes_may_differ":True,"array_comparison":arrays,"logical_report_comparison":logical,"checks":checks,"source_cells":len(a["tetrahedra"]),"candidate_cells":len(n["tetrahedra"]),"removed_cells":len(rem),"candidate_points":len(n["points_world_m"]),"remaining_f32_negative_cells":newr["F32_negative_cells"],"physical_admission":False}
(T/"guarded-replay-verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"checks":checks,"source_sha256":report["source_064_sha256"],"old_npz_sha256":report["frozen_065_candidate_sha256"],"guarded_npz_sha256":report["guarded_065_candidate_sha256"],"files":{"report":str(T/"guarded-replay-verification.json")}},sort_keys=True))
if report["status"]!="PASS":raise SystemExit(1)
