#!/usr/bin/env python3
"""Canonicalize D597 transfer lineage to Fat597 immediate parents without changing geometry."""
from pathlib import Path
import hashlib, json, time, importlib.util
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
RUN=H/"fat-current-D-546-597-root"
OUTDIR=RUN/"run011-target397-current"
BASE=H/"current-paired-fixed-boundary-composition-597-root/fat-current-connected-volume-proposal.npz"
CANON=H/"paired-I-local-cavity-527-root/run055-current-I50-D264-source-I-baseline-context/canonical-parent-unions-v3/fat/fat-current-I50-D264-canonical-parent-unions.npz"
TRANSFER=RUN/"run007-transfer-379-001/fat-D597-current-D558-positive-subset-short-edge-quotient-volume-proposal.npz"
TRANSFER_REPORT=RUN/"run007-transfer-379-001/report.json"
INDEPENDENT=RUN/"run008-independent-379-001/independent-verification.json"
TARGET=OUTDIR/"rebound-v4/fat-current-typed-target-rebound.npz"
TARGET_REPORT=OUTDIR/"rebound-v4/report.json"
SIDE=OUTDIR/"D597-full-endpoint-star-canonical-parent-unions.npz"
SIDE_REPORT=OUTDIR/"D597-full-endpoint-star-canonical-parent-unions.json"
CONTEXT=H/"joint-current-O612-D597-context-613-root/audit354/report.json"
OUTDIR2=OUTDIR/"canonical-v3"
OUT_NPZ=OUTDIR2/"fat-current-O612-D597-I50-canonical-parent-unions.npz"
OUT_REPORT=OUTDIR2/"receipt.json"

EXPECTED={
 "base":"6677a4bbc53637617cebaf9e74d7b0dfd9c775d0d5e64d7050bf4741f8e35054",
 "canon":"4798310f9fc4257ef27618d5323ff1350b1df334ab3f59680665aa8beee4e396",
 "transfer":"36143b2efe2f91af985a74182293ab5c9faaaa3f02b5053af4577ad1eed3fd31",
 "transfer_report":"2ecb1c6dbaf081a9716e4f993b36afc4a3485a35969a73df12a2a35fa79d020a",
 "target":"7cab00129163160a5b8ce2e860d098798aca6ac30ca1255429d23d80c0ef30d0",
 "side":"d54622d4193da1b37d97982c784eb03ff77274ed8314bf6e0541d4130bb7a08c",
 "side_report":"f297f2e69fa0a2c3b3cf87aa5241ae2aaeb5f244b00d46f3d99cd698559b00ee",
 "context":"4f457bbbab1d858061ac141b383cff5648ff5d2080a2a39cc4b39853b6fdb73c",
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def tetkey(t):return tuple(sorted(map(int,t)))
def csr(rows):
 off=np.zeros(len(rows)+1,np.int64)
 for i,x in enumerate(rows):off[i+1]=off[i]+len(x)
 flat=np.concatenate([np.asarray(x,np.int64) for x in rows]) if off[-1] else np.empty(0,np.int64)
 return off,flat

started=time.monotonic()
for name,p in (("base",BASE),("canon",CANON),("transfer",TRANSFER),("transfer_report",TRANSFER_REPORT),("target",TARGET),("side",SIDE),("side_report",SIDE_REPORT),("context",CONTEXT)):
 got=sha(p)
 if got!=EXPECTED[name]:raise ValueError(f"{name} pin mismatch: {got}")
base=load(BASE); parent=load(CANON); transfer=load(TRANSFER); out=load(TARGET); side=load(SIDE)
tr=json.loads(TRANSFER_REPORT.read_text()); ir=json.loads(INDEPENDENT.read_text())
target_report=json.loads(TARGET_REPORT.read_text()); side_report=json.loads(SIDE_REPORT.read_text()); context=json.loads(CONTEXT.read_text())
if tr.get("status")!="PASS_D597_INDEXED379_VOLUME_QUOTIENT_PROPOSAL" or tr.get("physical_admission") is not False or tr.get("native_steps")!=0:raise ValueError("D597 transfer status/claims changed")
if ir.get("status")!="PASS_INDEPENDENT_D597_CURRENT_D558_POSITIVE_SUBSET_VOLUME_REPLAY" or ir.get("physical_admission") is not False or ir.get("native_steps")!=0:raise ValueError("D597 independent replay status/claims changed")
if ir["candidate"]["sha256"]!=EXPECTED["transfer"] or ir["transfer_report"]["sha256"]!=EXPECTED["transfer_report"]:raise ValueError("independent replay is not bound to transfer")
if target_report.get("status")!="PASS_TYPED_CURRENT_TARGET_REBOUND_SOURCE_ONLY" or target_report.get("physical_admission") is not False or target_report.get("native_steps")!=0:raise ValueError("target397 receipt status/claims changed")
if target_report["output_candidate"]["sha256"]!=EXPECTED["target"] or target_report["actual_vs_typed_target"]["missing_oriented_records"]!=0 or target_report["actual_vs_typed_target"]["extra_oriented_records"]!=0:raise ValueError("target397 binding differs")
if side_report.get("status")!="PASS_D597_ENDPOINT_STAR_CANONICAL_PARENT_UNIONS_SOURCE_ONLY":raise ValueError("endpoint-star source sidecar status changed")
if context.get("status")!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":raise ValueError("joint O612/D597/I50 context is not passing")
if context.get("physical_admission") is not False or context.get("native_steps")!=0:raise ValueError("source context claim changed")
if OUTDIR2.exists():raise FileExistsError(OUTDIR2)
OUTDIR2.mkdir(parents=True)

P=np.asarray(base["points_world_m_f32"],np.float32); T=np.asarray(base["tetrahedra"],np.int64); M=np.asarray(base["material_indices"],np.int32)
pt=np.asarray(parent["tetrahedra"],np.int64); pm=np.asarray(parent["material_indices"],np.int32)
ot=np.asarray(out["tetrahedra"],np.int64); om=np.asarray(out["material_indices"],np.int32)
if len(P)!=253712 or T.shape!=(789327,4) or len(M)!=len(T) or np.any(M!=16):raise ValueError("unexpected Fat597 base domain")
if not np.array_equal(np.asarray(base["points_world_m"],np.float32).view("u4"),P.view("u4")):raise ValueError("base point views differ")
if not np.array_equal(np.asarray(parent["points_world_m_f32"],np.float32).view("u4"),P.view("u4")):raise ValueError("canonical parent point bytes differ")
if not np.array_equal(np.asarray(out["points_world_m"],np.float32).view("u4"),P.view("u4")) or not np.array_equal(np.asarray(out["points_world_m_f32"],np.float32).view("u4"),P.view("u4")):raise ValueError("D597 target rebound changed point bytes")
if out["material_indices"].shape!=(len(ot),) or np.any(om!=16):raise ValueError("unexpected output material domain")

# Resolve each Fat597 source cell to its immediate parent union exactly as the upstream
# canonical asset records it (direct retained row or an explicit replacement CSR row).
base_map=np.asarray(base["output_cell_to_input_cell"],np.int64)
if base_map.shape!=(len(T),):raise ValueError("Fat597 canonical source map shape mismatch")
base_rep=np.asarray(base["replacement_tetrahedra"],np.int64)
base_rep_off=np.asarray(base["replacement_parent_offsets"],np.int64)
base_rep_rows=np.asarray(base["replacement_parent_cell_rows"],np.int64)
if base_rep_off.shape!=(len(base_rep)+1,) or base_rep_off[0]!=0 or base_rep_off[-1]!=len(base_rep_rows):raise ValueError("Fat597 replacement CSR malformed")
if np.any(base_rep_rows<0) or np.any(base_rep_rows>=len(pt)):raise ValueError("Fat597 immediate parent row out of range")
rep_lookup={}
for j,t in enumerate(base_rep):
 k=tetkey(t)
 if k in rep_lookup:raise ValueError("duplicate Fat597 replacement tetrahedron key")
 rep_lookup[k]=j
parents_by_base=[]; replacement_indexes=[]; direct_count=0
for row,t in enumerate(T):
 q=int(base_map[row])
 if q>=0:
  if q>=len(pt) or not np.array_equal(t,pt[q]) or int(M[row])!=int(pm[q]):raise ValueError(f"Fat597 direct parent mismatch at row {row}")
  parents_by_base.append(np.asarray([q],np.int64)); replacement_indexes.append(-1); direct_count+=1
 else:
  j=rep_lookup.get(tetkey(t))
  if j is None:raise ValueError(f"Fat597 replacement row {row} absent from replacement-tet map")
  q=base_rep_rows[base_rep_off[j]:base_rep_off[j+1]]
  if len(q)==0 or len(np.unique(q))!=len(q):raise ValueError("empty or duplicated Fat597 parent union")
  parents_by_base.append(np.sort(q)); replacement_indexes.append(j)
if direct_count!=788697 or len(rep_lookup)!=630 or sum(x>=0 for x in replacement_indexes)!=630:raise ValueError("Fat597 direct/replacement counts changed")
if len(set(x for x in replacement_indexes if x>=0))!=630:raise ValueError("Fat597 replacement map is not one-to-one")

# Reconstruct the accepted D full-star partition and independently check the sidecar’s
# per-operation canonical parent union against the pinned Fat597 immediate lineage.
so=np.asarray(side["endpoint_star_offsets"],np.int64); sr=np.asarray(side["endpoint_star_current_cell_rows"],np.int64)
uo=np.asarray(side["canonical_parent_union_offsets"],np.int64); ur=np.asarray(side["canonical_parent_union_parent_rows"],np.int64)
if so.shape!=(153,) or uo.shape!=(153,):raise ValueError("D597 sidecar operation count mismatch")
owner=np.full(len(T),-1,np.int32); op_parent=[]
for k in range(len(so)-1):
 rows=sr[so[k]:so[k+1]]
 if len(rows)==0 or np.any(rows<0) or np.any(rows>=len(T)) or len(np.unique(rows))!=len(rows):raise ValueError(f"invalid D operation full star {k}")
 if np.any(owner[rows]>=0):raise ValueError(f"overlapping D full endpoint stars at operation {k}")
 owner[rows]=k
 derived=np.unique(np.concatenate([parents_by_base[int(x)] for x in rows]))
 claimed=np.asarray(ur[uo[k]:uo[k+1]],np.int64)
 if len(claimed)==0 or not np.array_equal(np.sort(claimed),derived):raise ValueError(f"D op {k} parent union does not equal reconstructed Fat597 full-star ancestry")
 op_parent.append(derived)
if int(np.sum(so[1:]-so[:-1]))!=3130 or np.count_nonzero(owner>=0)!=3130:raise ValueError("D endpoint-star partition count changed")

# The target397 candidate carries the exact indexed379 output-to-Fat597 row map.
src=np.asarray(out["output_cell_to_input_cell"],np.int64)
if src.shape!=(len(ot),) or np.any(src<0) or np.any(src>=len(T)) or len(np.unique(src))!=len(src):raise ValueError("D597 output-to-current row map invalid")
if len(T)-len(src)!=567 or not np.array_equal(np.setdiff1d(np.arange(len(T),dtype=np.int64),src),np.asarray(side["removed_current_input_cell_rows"],np.int64)):raise ValueError("D597 removed source cell rows disagree with full-star sidecar")
if not np.array_equal(om,M[src]):raise ValueError("D597 output material does not map to source rows")
inside=owner[src]>=0
changed=np.any(ot!=T[src],axis=1)
if np.any(changed & ~inside):raise ValueError("D597 changed tetrahedron lies outside authenticated endpoint stars")
if np.count_nonzero(changed)!=1120 or np.count_nonzero(inside)!=2563:raise ValueError("D597 changed/full-star cell counts differ from independent transfer domain")
if np.any(om[changed]!=M[src[changed]]) or np.any(om!=16):raise ValueError("material changed during D transfer")

# Compose output row -> current geometry parent row, preserving inherited replacements.
out_map=np.full(len(ot),-1,np.int64)
out_parent_lists=[None]*len(ot)
for row,brow0 in enumerate(src):
 brow=int(brow0); k=int(owner[brow])
 if k>=0:
  out_parent_lists[row]=op_parent[k]
 else:
  q=int(base_map[brow])
  if q>=0:out_map[row]=q
  else:out_parent_lists[row]=parents_by_base[brow]
replacement_rows=np.flatnonzero(out_map<0)
parent_offsets,parent_rows=csr([out_parent_lists[int(i)] for i in replacement_rows])
if len(parent_offsets)!=len(replacement_rows)+1 or np.any(parent_rows<0) or np.any(parent_rows>=len(pt)):raise ValueError("composed immediate parent CSR malformed")
direct=out_map[out_map>=0]
if len(np.unique(direct))!=len(direct):raise ValueError("duplicate direct immediate-parent cell mapping")
reverse=np.full(len(pt),-1,np.int64)
reverse[direct]=np.flatnonzero(out_map>=0)
if not np.array_equal(reverse[direct],np.flatnonzero(out_map>=0)):raise ValueError("direct parent reverse map mismatch")
for row in np.flatnonzero(out_map>=0):
 q=int(out_map[row])
 if not np.array_equal(ot[row],pt[q]) or int(om[row])!=int(pm[q]):raise ValueError(f"unmodified direct cell does not equal immediate parent at output {row}")
for i,row in enumerate(replacement_rows):
 q=parent_rows[parent_offsets[i]:parent_offsets[i+1]]
 if len(q)==0 or len(np.unique(q))!=len(q):raise ValueError(f"replacement output {row} has empty/duplicate parents")
 if np.any(pm[q]!=16):raise ValueError(f"replacement output {row} crosses material")

# Independently verify exact oriented boundary identity against the current typed target.
srcmod=H/"transfer_surface_contractions_indexed_379_root.py"
spec=importlib.util.spec_from_file_location("transfer379_canonical",srcmod)
Tmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(Tmod)
actual=Tmod.boundary(ot)
identity=Tmod.boundary_identity
if not np.array_equal(identity(actual),identity(out["boundary_faces"])):raise ValueError("canonicalization would disagree with current actual boundary")
if not np.array_equal(identity(actual),identity(out["desired_boundary_faces"])):raise ValueError("D597 target397 boundary is not complete against actual volume")
if not np.array_equal(identity(actual),identity(out["proposed_current_desired_boundary_faces"])):raise ValueError("D597 proposed/current boundary field mismatch")
if target_report["actual_vs_typed_target"]["missing_oriented_records"] or target_report["actual_vs_typed_target"]["extra_oriented_records"]:raise ValueError("target397 receipt is not exact")

# Update only source-lineage arrays and canonical proof metadata; preserve every geometry,
# target, typed crosswalk, and source-context payload byte-for-byte.
a={k:np.array(v,copy=True) for k,v in out.items()}
changes={}
def replace(k,v):
 changes[k]={"before_shape":list(a[k].shape) if k in a else None}
 a[k]=np.asarray(v)
replace("output_cell_to_input_cell",out_map)
replace("input_cell_to_output_cell",reverse)
replace("retained_input_cell_rows",direct)
replace("replacement_parent_offsets",parent_offsets)
replace("replacement_parent_cell_rows",parent_rows)
replace("replacement_tet_offsets",np.asarray([0,len(replacement_rows)],np.int64))
replace("replacement_tetrahedra",ot[replacement_rows])
replace("current_geometry_parent_path",np.asarray([str(CANON)]))
replace("current_geometry_parent_sha256",np.asarray([EXPECTED["canon"]]))
replace("output_node_to_input_node",base["output_node_to_input_node"])
replace("input_node_to_output_node",base["input_node_to_output_node"])
replace("output_point_to_input_point_row",base["output_point_to_input_point_row"])
replace("output_point_parent_offsets",base["output_point_parent_offsets"])
replace("output_point_parent_input_cell_rows",base["output_point_parent_input_cell_rows"])
replace("canonical_parent_union_schema",np.asarray("numi.run011.D597-endpoint-star-immediate-parent-unions.v1"))
replace("canonical_parent_union_operation_order",np.asarray("I50_then_D264_then_disjoint_fixed_boundary_then_D597_152"))
replace("canonical_D597_full_endpoint_star_sidecar_path",np.asarray([str(SIDE)]))
replace("canonical_D597_full_endpoint_star_sidecar_sha256",np.asarray([EXPECTED["side"]]))
replace("canonical_D597_full_endpoint_star_report_path",np.asarray([str(SIDE_REPORT)]))
replace("canonical_D597_full_endpoint_star_report_sha256",np.asarray([EXPECTED["side_report"]]))
replace("canonical_D597_transfer_report_path",np.asarray([str(TRANSFER_REPORT)]))
replace("canonical_D597_transfer_report_sha256",np.asarray([EXPECTED["transfer_report"]]))
replace("canonical_D597_independent_replay_path",np.asarray([str(INDEPENDENT)]))
replace("canonical_D597_independent_replay_sha256",np.asarray([sha(INDEPENDENT)]))
replace("canonical_target397_report_path",np.asarray([str(TARGET_REPORT)]))
replace("canonical_target397_report_sha256",np.asarray([sha(TARGET_REPORT)]))
replace("canonical_joint_context_report_path",np.asarray([str(CONTEXT)]))
replace("canonical_joint_context_report_sha256",np.asarray([EXPECTED["context"]]))
replace("canonical_parent_union_physical_status",np.asarray("SOURCE_ONLY_PHYSICAL_ADMISSION_FALSE"))

np.savez_compressed(OUT_NPZ,**a)
written=load(OUT_NPZ)
if set(written)!=set(a) or any(not np.array_equal(written[k],v) for k,v in a.items()):raise ValueError("canonical output readback differs")
for k,v in out.items():
 if k not in changes and not np.array_equal(written[k],v):raise ValueError(f"non-lineage field changed: {k}")
if sha(OUT_NPZ)==EXPECTED["target"]:raise ValueError("unexpected candidate hash collision")
# Ensure target and all geometry arrays are bit-identical to target397 source.
for k in ("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces","desired_boundary_faces","mapped_current_boundary_faces","proposed_current_desired_boundary_faces"):
 if not np.array_equal(written[k],out[k]):raise ValueError(f"geometry/target changed: {k}")
receipt={
 "schema":"numi.run011.D597-canonical-parent-unions.v1",
 "status":"PASS_CANONICAL_O612_D597_I50_FAT_PARENT_UNIONS_SOURCE_ONLY",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "candidate":pin(OUT_NPZ),"target397_candidate":pin(TARGET),"target397_report":pin(TARGET_REPORT),
 "Fat597_base":pin(BASE),"canonical_immediate_parent":pin(CANON),
 "D597_transfer":pin(TRANSFER),"D597_transfer_report":pin(TRANSFER_REPORT),
 "D597_independent_replay":pin(INDEPENDENT),"D597_full_endpoint_star_sidecar":pin(SIDE),
 "D597_endpoint_star_report":pin(SIDE_REPORT),"joint_O612_D597_I50_context":pin(CONTEXT),
 "geometry":{"points":len(P),"tets":len(ot),"material16_only":True,"point_bytes_unchanged":True,"tetrahedra_bytes_unchanged":True,"target_arrays_unchanged":True,"actual_boundary_faces":len(actual),"actual_equals_target_oriented":True,"flux_replay":"Inherited from exact D597 indexed379 transfer and target397; no geometry changed"},
 "lineage":{"current_geometry_parent":pin(CANON),"direct_parent_mapped_cells":int(np.count_nonzero(out_map>=0)),"replacement_cells":int(len(replacement_rows)),"replacement_parent_row_references":int(len(parent_rows)),"full_endpoint_star_operations":int(len(op_parent)),"full_endpoint_star_source_rows":int(len(sr)),"output_cells_in_full_endpoint_stars":int(np.count_nonzero(inside)),"changed_output_cells":int(np.count_nonzero(changed)),"removed_source_cells":567,"parent_unions_rederived_and_sidecar_matched":True,"all_unrelated_inherited_replacement_unions_preserved":True,"point_lineage_inherited_from_Fat597":True},
 "quality":{"input_native_bad":int(ir["quality"]["input_native_bad"]),"output_native_bad":int(ir["quality"]["output_native_bad"]),"gain":int(ir["quality"]["native_bad_cell_gain"]),"nonpositive":0,"source":"pinned independent indexed379 replay"},
 "changed_fields":changes,"all_other_candidate_arrays_unchanged":True,
 "limitations":"Lineage-only canonicalization of a source-only, exact-positive D597 transfer. No source mass mapping, native execution, physical admission, or anatomy qualification."
}
OUT_REPORT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":receipt["status"],"candidate":str(OUT_NPZ),"candidate_sha256":sha(OUT_NPZ),"report":str(OUT_REPORT),"report_sha256":sha(OUT_REPORT),"direct_parent_cells":receipt["lineage"]["direct_parent_mapped_cells"],"replacement_cells":receipt["lineage"]["replacement_cells"],"parent_refs":receipt["lineage"]["replacement_parent_row_references"],"stars":receipt["lineage"]["full_endpoint_star_operations"],"cells_in_stars":receipt["lineage"]["output_cells_in_full_endpoint_stars"],"changed_cells":receipt["lineage"]["changed_output_cells"],"quality":receipt["quality"],"elapsed_s":time.monotonic()-started},indent=2))
