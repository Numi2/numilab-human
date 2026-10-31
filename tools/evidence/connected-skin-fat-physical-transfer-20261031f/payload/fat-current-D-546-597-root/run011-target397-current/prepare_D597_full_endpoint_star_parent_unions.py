#!/usr/bin/env python3
"""Freeze full sequential D endpoint stars and canonical Fat597 immediate-parent unions."""
from pathlib import Path
import hashlib, json, time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
RUN=H/"fat-current-D-546-597-root"
OUT=RUN/"run011-target397-current"
BASE=H/"current-paired-fixed-boundary-composition-597-root/fat-current-connected-volume-proposal.npz"
D=RUN/"run009-typed-bind-D597-current-D558-v1/candidate-D-typed-current.npz"
TRANSFER=RUN/"run007-transfer-379-001/report.json"
TYPED=RUN/"run009-typed-bind-D597-current-D558-v1"
BASE_SHA="6677a4bbc53637617cebaf9e74d7b0dfd9c775d0d5e64d7050bf4741f8e35054"
D_SHA="97eb4720ebb09d62659aa03f6c4f86482f9cf0e949101f636f6d1fff0e1daa89"
TRANSFER_SHA="2ecb1c6dbaf081a9716e4f993b36afc4a3485a35969a73df12a2a35fa79d020a"
OUT_NPZ=OUT/"D597-full-endpoint-star-canonical-parent-unions.npz"
OUT_REPORT=OUT/"D597-full-endpoint-star-canonical-parent-unions.json"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def key(x):return np.ascontiguousarray(x,dtype="<f4").tobytes()
def csr(rows):
 offsets=np.zeros(len(rows)+1,dtype=np.int64)
 for i,x in enumerate(rows): offsets[i+1]=offsets[i]+len(x)
 vals=np.concatenate([np.asarray(x,np.int64) for x in rows]) if offsets[-1] else np.empty(0,np.int64)
 return offsets,vals

t0=time.monotonic()
for p,want in ((BASE,BASE_SHA),(D,D_SHA),(TRANSFER,TRANSFER_SHA)):
 got=sha(p)
 if got!=want:raise ValueError(f"source pin mismatch {p}: {got}")
base=load(BASE); d=load(D); tr=json.loads(TRANSFER.read_text())
if tr.get("status")!="PASS_D597_INDEXED379_VOLUME_QUOTIENT_PROPOSAL" or tr.get("physical_admission") is not False or tr.get("native_steps")!=0:
 raise ValueError("unexpected D597 transfer source status")
T=np.asarray(base["tetrahedra"],np.int64); M=np.asarray(base["material_indices"],np.int32)
P=np.asarray(base["points_world_m_f32"],np.float32); DP=np.asarray(d["points_world_m_f32"],np.float32)
if T.shape!=(789327,4) or M.shape!=(len(T),) or len(P)!=253712 or np.any(M!=16):raise ValueError("unexpected Fat597 row domain")
if not np.array_equal(np.asarray(base["points_world_m"],np.float32),P):raise ValueError("Fat597 point views differ")
vol={key(x):i for i,x in enumerate(P)}
if len(vol)!=len(P):raise ValueError("Fat597 has duplicate coordinate keys")
dmap=np.asarray([vol[key(x)] for x in DP],np.int64)
if len(np.unique(dmap))!=len(dmap):raise ValueError("typed D point map is not injective")
stored=np.asarray(base["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)
if stored.shape!=dmap.shape or not np.array_equal(stored,dmap):raise ValueError("Fat597 stored D crosswalk differs from exact F32 coordinate map")
parent_path=Path(str(np.asarray(base["current_geometry_parent_path"]).item()))
parent_sha=str(np.asarray(base["current_geometry_parent_sha256"]).item())
if sha(parent_path)!=parent_sha:raise ValueError("Fat597 canonical parent asset SHA mismatch")
parent=load(parent_path)
parent_T=np.asarray(parent["tetrahedra"],np.int64); parent_M=np.asarray(parent["material_indices"],np.int32)
out2in=np.asarray(base["output_cell_to_input_cell"],np.int64)
if out2in.shape!=(len(T),):raise ValueError("Fat597 output-to-immediate-parent row map shape mismatch")
rep_tets=np.asarray(base["replacement_tetrahedra"],np.int64)
rep_par_off=np.asarray(base["replacement_parent_offsets"],np.int64)
rep_par_rows=np.asarray(base["replacement_parent_cell_rows"],np.int64)
rep_tet_off=np.asarray(base["replacement_tet_offsets"],np.int64)
if rep_tet_off.shape!=(2,) or rep_tet_off[0]!=0 or rep_tet_off[-1]!=len(rep_tets):raise ValueError("unexpected replacement tet CSR")
if rep_par_off.shape!=(len(rep_tets)+1,) or rep_par_off[0]!=0 or rep_par_off[-1]!=len(rep_par_rows):raise ValueError("replacement immediate-parent CSR malformed")
if np.any(rep_par_rows<0) or np.any(rep_par_rows>=len(parent_T)):raise ValueError("replacement parent row outside canonical parent")
# Resolve every current Fat597 tet to its canonical immediate parent row union.
def canon_tet(t):return tuple(sorted(map(int,t)))
rep_lookup={}
for j,t in enumerate(rep_tets):
 k=canon_tet(t)
 if k in rep_lookup:raise ValueError("duplicate replacement tetrahedron key")
 rep_lookup[k]=j
parents_by_cell=[]
replacement_seen=[]
for row,t in enumerate(T):
 pr=int(out2in[row])
 if pr>=0:
  if pr>=len(parent_T) or tuple(map(int,t))!=tuple(map(int,parent_T[pr])) or int(M[row])!=int(parent_M[pr]):
   raise ValueError(f"retained current cell {row} does not match ordered immediate parent {pr}")
  parents_by_cell.append(np.asarray([pr],np.int64))
 else:
  j=rep_lookup.get(canon_tet(t))
  if j is None:raise ValueError(f"replacement current cell {row} has no exact replacement-tet map")
  if int(M[row])!=16:raise ValueError("replacement material changed")
  q=np.asarray(rep_par_rows[rep_par_off[j]:rep_par_off[j+1]],np.int64)
  if len(q)==0 or len(np.unique(q))!=len(q):raise ValueError("replacement immediate parent union is empty or duplicated")
  parents_by_cell.append(np.sort(q))
  replacement_seen.append(j)
if len(replacement_seen)!=len(rep_tets) or len(set(replacement_seen))!=len(rep_tets):raise ValueError("replacement tetrahedra do not map one-to-one into current Fat rows")
# Build full cell incidence once, then replay the ordered 152 accepted contractions against exact source rows.
inc=[[] for _ in range(len(P))]
for ci,t in enumerate(T):
 for node in t:inc[int(node)].append(ci)
ops=tr.get("transferred_operations",[])
if len(ops)!=152 or tr.get("transferred_operation_count")!=len(ops):raise ValueError("D597 transfer operation count mismatch")
full_stars=[]; active_stars=[]; keep_stars=[]; drop_stars=[]; deleted=[]; parent_unions=[]; typed_unions=[]; ancestry_unions=[]; op_meta=[]
removed=set(); total_deleted=0
for i,op in enumerate(ops):
 sidx=int(op["surface_operation_index"])
 keep_s=int(op["keep_surface_row"]); drop_s=int(op["drop_surface_row"])
 keep_v=int(dmap[keep_s]); drop_v=int(dmap[drop_s])
 if keep_v!=int(op["keep_volume_row"]) or drop_v!=int(op["drop_volume_row"]):raise ValueError(f"operation {i}: endpoint row crosswalk mismatch")
 pj=TYPED/f"operation-D-{sidx:04d}.json"
 if not pj.is_file():raise ValueError(f"operation {i}: typed operation lineage missing {pj}")
 td=json.loads(pj.read_text())
 if td.get("patch_id")!=op.get("patch_id") or int(td.get("retained_input_point_row"))!=keep_s or int(td.get("collapsed_input_point_row"))!=drop_s:
  raise ValueError(f"operation {i}: typed D operation record does not match transfer endpoints")
 # The transfer owner rewrites precisely the active drop star and removes its edge cells.
 ks=set(inc[keep_v])-removed; ds=set(inc[drop_v])-removed; expected_deleted=ks&ds
 got_active=set(map(int,op["affected_input_cell_rows"]))
 got_deleted=set(map(int,op["deleted_input_cell_rows"]))
 if ds!=got_active:raise ValueError(f"operation {i}: active full drop-endpoint star differs from transfer affected rows")
 if expected_deleted!=got_deleted:raise ValueError(f"operation {i}: exact shared-edge deleted-cell star mismatch")
 full=ks|ds
 if not got_deleted<=full:raise ValueError("deleted cell outside full endpoint star")
 full_stars.append(np.asarray(sorted(full),np.int64)); active_stars.append(np.asarray(sorted(full),np.int64))
 keep_stars.append(np.asarray(sorted(ks),np.int64)); drop_stars.append(np.asarray(sorted(ds),np.int64)); deleted.append(np.asarray(sorted(got_deleted),np.int64))
 union=sorted({int(p) for cell in full for p in parents_by_cell[cell]})
 parent_unions.append(np.asarray(union,np.int64))
 tu=np.asarray(td.get("complete_typed_parent_union",[]),np.int64).reshape(-1,7)
 au=np.asarray(td.get("source_face_ancestry_union",[]),np.int64).reshape(-1,3)
 if len(tu)==0 or len(au)==0:raise ValueError(f"operation {i}: typed source parent/ancestry union missing")
 typed_unions.append(tu); ancestry_unions.append(au)
 op_meta.append({"transfer_operation_index":i,"surface_operation_index":sidx,"patch_id":op["patch_id"],"keep_surface_row":keep_s,"drop_surface_row":drop_s,"keep_volume_row":keep_v,"drop_volume_row":drop_v,"operation_json_path":str(pj),"operation_json_sha256":sha(pj),"full_keep_star_count":len(ks),"full_drop_star_count":len(ds),"full_endpoint_star_count":len(full),"deleted_edge_star_count":len(got_deleted),"canonical_immediate_parent_row_union_count":len(union),"typed_source_parent_tuple_count":len(tu),"typed_source_face_ancestry_count":len(au)})
 removed |= got_deleted; total_deleted+=len(got_deleted)
if total_deleted!=567 or len(removed)!=567 or tr.get("removed_cell_count")!=567:raise ValueError("ordered D597 deletion count does not match source transfer")
# Flatten full endpoint-star rows and the three parent-identity domains as CSR arrays.
star_off,star_rows=csr(full_stars); keep_off,keep_rows=csr(keep_stars); drop_off,drop_rows=csr(drop_stars); del_off,del_rows=csr(deleted)
parent_off,parent_rows=csr(parent_unions); tu_off,tu_flat=csr([x.reshape(-1) for x in typed_unions]); au_off,au_flat=csr([x.reshape(-1) for x in ancestry_unions])
# Keep row-shaped records so every immediate-parent cell identity is reconstructible.
parent_keys=np.asarray([np.sort(parent_T[int(i)]) for i in parent_rows],np.int64)
parent_mats=np.asarray([parent_M[int(i)] for i in parent_rows],np.int32)
cell_keys=np.sort(T[star_rows],axis=1) if len(star_rows) else np.empty((0,4),np.int64)
cell_ordered=T[star_rows] if len(star_rows) else np.empty((0,4),np.int64)
cell_mats=M[star_rows] if len(star_rows) else np.empty(0,np.int32)
if OUT_NPZ.exists() or OUT_REPORT.exists():raise FileExistsError("run011 sidecar already exists")
np.savez_compressed(OUT_NPZ,
 transfer_operation_index=np.arange(len(ops),dtype=np.int64),surface_operation_index=np.asarray([x["surface_operation_index"] for x in ops],np.int64),
 keep_surface_rows=np.asarray([x["keep_surface_row"] for x in ops],np.int64),drop_surface_rows=np.asarray([x["drop_surface_row"] for x in ops],np.int64),
 keep_volume_rows=np.asarray([x["keep_volume_row"] for x in ops],np.int64),drop_volume_rows=np.asarray([x["drop_volume_row"] for x in ops],np.int64),
 endpoint_star_offsets=star_off,endpoint_star_current_cell_rows=star_rows,endpoint_star_ordered_tet_rows=cell_ordered,endpoint_star_canonical_tet_rows=cell_keys,endpoint_star_material_indices=cell_mats,
 keep_star_offsets=keep_off,keep_star_current_cell_rows=keep_rows,drop_star_offsets=drop_off,drop_star_current_cell_rows=drop_rows,
 deleted_edge_star_offsets=del_off,deleted_edge_star_current_cell_rows=del_rows,
 canonical_parent_union_offsets=parent_off,canonical_parent_union_parent_rows=parent_rows,canonical_parent_union_parent_tet_corner_rows=parent_keys,canonical_parent_union_parent_material_indices=parent_mats,
 typed_source_parent_union_offsets=tu_off,typed_source_parent_union_lanes=tu_flat,
 typed_source_face_ancestry_offsets=au_off,typed_source_face_ancestry_rows=au_flat,
 source_cell_row_domain_sha256=np.asarray(BASE_SHA),source_D_sha256=np.asarray(D_SHA),source_transfer_sha256=np.asarray(TRANSFER_SHA),source_canonical_parent_path=np.asarray(str(parent_path)),source_canonical_parent_sha256=np.asarray(parent_sha),
 D_surface_point_to_Fat597_volume_point_rows=dmap,removed_current_input_cell_rows=np.asarray(sorted(removed),np.int64))
report={"schema":"h706.D597-full-endpoint-star-canonical-parent-unions.v1","status":"PASS_D597_ENDPOINT_STAR_CANONICAL_PARENT_UNIONS_SOURCE_ONLY","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "inputs":{"Fat597":{"path":str(BASE),"sha256":BASE_SHA,"points":len(P),"cells":len(T)},"typed_D597":{"path":str(D),"sha256":D_SHA,"points":len(DP)},"indexed379_report":{"path":str(TRANSFER),"sha256":TRANSFER_SHA,"status":tr["status"]},"canonical_immediate_parent":{"path":str(parent_path),"sha256":parent_sha}},
 "operations":{"count":len(ops),"replayed_in_order":True,"full_endpoint_stars_recomputed_from_Fat597_tets":True,"active_drop_stars_match_transfer_affected_rows":True,"deleted_edge_stars_match_endpoint_intersections":True,"total_deleted_cells":total_deleted,"unique_deleted_cells":len(removed),"full_endpoint_star_cell_references":int(len(star_rows)),"canonical_parent_row_references":int(len(parent_rows)),"typed_source_parent_union_rows":int(len(tu_flat)//7),"typed_source_face_ancestry_rows":int(len(au_flat)//3),"records":op_meta},
 "crosswalk":{"all_D_points_map_exactly_to_unique_Fat597_F32_rows":True,"stored_Fat597_D_crosswalk_matches":True,"all_op_endpoints_match_exact_transfer_rows":True},
 "output":{"path":str(OUT_NPZ),"sha256":sha(OUT_NPZ),"size_bytes":OUT_NPZ.stat().st_size},"elapsed_s":time.monotonic()-t0,
 "limitations":"This sidecar freezes exact Fat597 full endpoint stars, sequential deleted-edge stars, and canonical immediate-parent unions for the source-only D597 transfer. It makes no additional geometry, mass, native, or physical-admission claim."}
OUT_REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"npz":str(OUT_NPZ),"npz_sha256":sha(OUT_NPZ),"report":str(OUT_REPORT),"report_sha256":sha(OUT_REPORT),"operations":len(ops),"full_star_rows":len(star_rows),"parent_union_rows":len(parent_rows),"typed_parent_rows":len(tu_flat)//7,"face_ancestry_rows":len(au_flat)//3,"removed":total_deleted,"elapsed_s":report["elapsed_s"]},indent=2))
