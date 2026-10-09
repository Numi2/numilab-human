#!/usr/bin/env python3
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

E=Path("/Users/n/numi-human-retained-delivery-20261009")
R=E/"distributed-resting-support-1199"
OUT=R/"source-region-coverage-1199.json"
MANIFEST=R/"resting-supine-scene-distributed-support.manifest.json"
SUPPORT=R/"myosim-fullbody-distributed-rigid-digit-support.nhcnt"
ROOT1196=E/"contoured-bed-reference-1196"
PARENT=ROOT1196/"resting-supine-scene-contoured-5mm.manifest.json"
RUNTIME_AUDIT=ROOT1196/"heightfield-runtime-audit-5mm.json"
RUNTIME_ARRAYS=ROOT1196/"heightfields-runtime-f32-5mm.npz"
REGION_BASELINE=ROOT1196/"source-voronoi-region-gaps-5mm.json"
PACK=Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run/accepted-geometry/step-0.mrvpack")
RECEIPT=PACK.with_name("step-0.receipt.json")
SKIN=Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin")
RIGID=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
BODY_MANIFEST=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json")
GEN=ROOT1196/"build_runtime_manifest.py"
AUDIT=ROOT1196/"audit_fixed_bed_full_body_captures_1198.py"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def require(c,m):
 if not c: raise RuntimeError(m)

manifest=read(MANIFEST); parent=read(PARENT)
support_out=manifest["outputs"]["support_contact"]
require(Path(support_out["path"])==SUPPORT,"manifest support-contact path differs")
require(support_out["sha256"]==sha(SUPPORT),"manifest support-contact hash differs")
require(manifest["source"]["skin"]["sha256"]==sha(SKIN),"scene skin identity differs from pinned skin")
require(manifest["source"]["rigid"]["sha256"]==sha(RIGID),"scene rigid identity differs")
require(manifest["bed"]["contact_distribution"]["parent_manifest"]==str(PARENT),"parent manifest path differs")
require(manifest["bed"]["contact_distribution"]["parent_manifest_sha256"]==sha(PARENT),"parent manifest hash differs")
require(manifest["bed"]["minimum_source_skin_gap_m"]==parent["bed"]["minimum_source_skin_gap_m"],"source gap differs from 1196")
require(manifest["source"]["skin"]["path"]==parent["source"]["skin"]["path"] and manifest["source"]["skin"]["sha256"]==parent["source"]["skin"]["sha256"],"skin source differs from 1196")
require(manifest["bed"]["heightfield"]==parent["bed"]["heightfield"],"heightfield object differs from 1196 parent")

builder=load_module(GEN,"bedgen1196")
audit=load_module(AUDIT,"fixedbedaudit1196")
from numilab_human.resting_scene import load_rigid,load_skin,NHCNT_HEADER,NHCNT_RECORD,source_shell_world
rigid=load_rigid(RIGID); skin=load_skin(SKIN,rigid)
raw=SUPPORT.read_bytes()
header=NHCNT_HEADER.unpack_from(raw,0)
require(header[0]==b"NHCNT1\0\0" and int(header[3])==32,"NHCNT header/magic/count mismatch")
require(len(raw)==NHCNT_HEADER.size+32*NHCNT_RECORD.size,"NHCNT byte length mismatch")
witnesses=manifest["bed"]["support_witnesses"]
require(len(witnesses)==32 and int(manifest["bed"]["support_witness_count"])==32,"scene does not declare 32 witnesses")
rows=[]
for i,w in enumerate(witnesses):
 row=NHCNT_RECORD.unpack_from(raw,NHCNT_HEADER.size+i*NHCNT_RECORD.size)
 rows.append(row)
 require(int(row[0])==int(w["core_body_index"]),"body binding differs at region {}".format(i))
 require(int(row[1])-1==int(w["vertex_index"]),"seed vertex differs at region {}".format(i))

seeds=[int(row[1])-1 for row in rows]
require(len(set(seeds))==32,"duplicate support seed vertices")
require(all(0<=x<int(skin["vertex_count"]) for x in seeds),"support seed outside NHSKIN vertex range")

# Reconstruct exact owner Float32 weighted source-rest coordinates and strict-distance Voronoi labels.
n=int(skin["vertex_count"]); pts=builder.f32(skin["vertices"]); acc=builder.f32(np.zeros((n,3),dtype=np.float32))
for j,(body,t,q,scale) in enumerate(skin["bindings"]):
 weights=builder.f32(skin["weights"][:,j]); mask=weights>0
 if not mask.any(): continue
 local=builder.f32(builder.f32(t).reshape(1,3)+builder.f32(builder.rotate(q,pts[mask])*builder.f32(scale)))
 bp=builder.f32(rigid["core_poses"][int(body)])
 world=builder.f32(builder.f32(bp[:3]).reshape(1,3)+builder.rotate(bp[3:7],local))
 acc[mask]=builder.f32(acc[mask]+builder.f32(world*weights[mask,None]))
labels=np.full(n,-1,np.int32); best=np.full(n,np.float32(np.inf),np.float32)
for rid,seed in enumerate(seeds):
 d=builder.f32(acc-builder.f32(acc[seed]))
 ds=builder.f32(builder.f32(builder.f32(d[:,0]*d[:,0])+builder.f32(d[:,1]*d[:,1]))+builder.f32(d[:,2]*d[:,2]))
 take=ds<best; labels[take]=rid; best[take]=ds[take]
require(np.all((labels>=0)&(labels<32)),"some skin vertices were not assigned exactly once")
counts=np.bincount(labels,minlength=32)
require(int(counts.sum())==n,"partition cardinalities do not cover each source skin vertex exactly once")
require(np.all(counts>0),"at least one support region has no NHSKIN vertices")

# The first eight anatomical seeds must remain unchanged from the 1196 parent.
p_w=parent["bed"]["support_witnesses"]
first8=[]
for i in range(8):
 same=(int(witnesses[i]["vertex_index"])==int(p_w[i]["vertex_index"]) and int(witnesses[i]["core_body_index"])==int(p_w[i]["core_body_index"]) and witnesses[i]["region"]==p_w[i]["region"])
 require(same,"first-eight source support seed changed at {}".format(i))
 first8.append({"index":i,"region":witnesses[i]["region"],"body_index":int(witnesses[i]["core_body_index"]),"vertex_index":int(witnesses[i]["vertex_index"]),"unchanged_from_1196":True})

# Verify exact F32 runtime grid against saved 1196 arrays, then query every rest-world skin vertex.
grid=audit.build_runtime_grid(manifest); audit.validate_runtime_grid(grid)
with np.load(RUNTIME_ARRAYS,allow_pickle=False) as saved:
 grid_identical=bool(np.array_equal(grid["x"],saved["x_nodes_f32"]) and np.array_equal(grid["y"],saved["y_nodes_f32"]) and np.array_equal(grid["z"],saved["heights_yx_f32"]))
require(grid_identical,"1199 runtime Float32 grid differs from exact 1196 grid")
pack_positions,pack_surfaces,_=builder._pack_surfaces(PACK,{(51999,1)})
skin_faces=pack_surfaces[(51007,1)]["faces"]
skin_used=np.unique(skin_faces)
skin_start=int(skin_used.min())
require(int(skin_used.min())==skin_start and int(skin_used.max())<skin_start+n and len(pack_positions)>=skin_start+n,"accepted pack skin vertex range is not the pinned contiguous payload range")
skin_capture=builder.f32(pack_positions[skin_start:skin_start+n])
bed_query=audit.query_bed(skin_capture,grid)
require(bed_query["outside_count"]==0,"accepted source skin capture has vertices outside finite bed bounds")
signed=np.full(n,np.nan,dtype=np.float64)
signed[bed_query["selected_rows"]]=bed_query["signed_gap_m"].astype(np.float64)
vertical=(skin_capture[:,2].astype(np.float64)-builder.height_at(grid["x"],grid["y"],grid["z"],skin_capture))
require(np.isfinite(signed).all() and np.isfinite(vertical).all(),"nonfinite source bed gap")
regions=[]; names=[]
for i,w in enumerate(witnesses):
 ids=np.flatnonzero(labels==i)
 k_s=int(ids[np.argmin(signed[ids])]); k_v=int(ids[np.argmin(vertical[ids])])
 names.append(str(w["region"]))
 regions.append({"region_index":i,"region":w["region"],"body_index":int(w["core_body_index"]),"body":str(w.get("region")),"seed_skin_vertex_index":seeds[i],"source_skin_vertex_count":int(counts[i]),"minimum_signed_point_to_facet_gap_m":float(signed[k_s]),"minimum_signed_gap_vertex_index":k_s,"minimum_vertical_gap_m":float(vertical[k_v]),"minimum_vertical_gap_vertex_index":k_v})
require(len(set(names[8:]))==24,"distributed axial-half region names are not unique")
groups={}
for x in manifest["bed"]["contact_distribution"]["seed_groups"]:
 groups[x["region"]]={"body_indices":[int(v) for v in x["body_indices"]],"bodies":x["bodies"],"seed_skin_vertices":[int(v) for v in x["seeds"]],"source_skin_vertices_claimed":int(x["source_skin_vertices"])}
expected_bases={part+"_"+side for side in ("r","l") for part in ("upper_arm","forearm","rigid_hand","thigh","shin","foot")}
require(len(groups)==12 and set(groups)==expected_bases,"distributed semantic groups differ from six bilateral limb/hand regions")
for i in range(8,32):
 region=names[i]
 require("_axial_half_" in region,"distributed region lacks proximal/distal axial label")
 base,half=region.rsplit("_axial_half_",1)
 require(base in groups and half in ("0","1"),"distributed region does not bind an expected semantic group")
 g=groups[base]
 require(seeds[i]==g["seed_skin_vertices"][int(half)],"NHCNT seed differs from declared axial-half group")
 require(int(witnesses[i]["core_body_index"]) in g["body_indices"],"seed body is outside its declared group binding")
body_order=read(BODY_MANIFEST)["core_tree"]["body_order"]
for i,r in enumerate(regions):
 r["body"]=body_order[r["body_index"]]

baseline_runtime=read(RUNTIME_AUDIT); baseline_regions=read(REGION_BASELINE)
require(baseline_runtime["runtime_mesh_audit"]["intersections"]==0,"1196 retained exact mesh audit is not intersection-free")
require(baseline_runtime["candidate_skin"]["sha256"]==sha(SKIN),"1196 exact mesh audit used different skin")
require(baseline_runtime["runtime_grid"]["nx_nodes"]==grid["nx"] and baseline_runtime["runtime_grid"]["ny_nodes"]==grid["ny"],"1196 runtime audit grid dimensions differ")
require(baseline_runtime["source_capture"]["sha256"]==sha(PACK),"source capture identity differs")
report={
 "schema":"numi.human.resting.distributed-source-region-coverage.v1",
 "status":"source_rest_partition_complete_and_bed_grid_identity_verified",
 "scope":"CPU-only source-rest ownership partition and per-region source-pose gap calculation for manifest 1199. No native run, contact-force response, or dynamic clearance claim.",
 "inputs":{str(p):sha(p) for p in [MANIFEST,SUPPORT,PARENT,RUNTIME_AUDIT,RUNTIME_ARRAYS,REGION_BASELINE,PACK,RECEIPT,SKIN,RIGID,BODY_MANIFEST,GEN,AUDIT]},
 "source_skin":{"path":str(SKIN),"sha256":sha(SKIN),"vertex_count":n,"triangle_count":int(skin["index_count"]//3)},
 "support_contact":{"path":str(SUPPORT),"sha256":sha(SUPPORT),"row_count":32,"unique_seed_count":len(set(seeds)),"first_eight_unchanged_from_1196":True,"first_eight":first8},
 "partition":{"algorithm":"1196 source_labels Float32 weighted source-rest skin coordinates; nearest ordered NHCNT seed by sequential Float32 squared distance; strict less-than update means first seed wins exact ties","assigned_skin_vertex_count":n,"unassigned_count":int(np.count_nonzero(labels<0)),"multiply_assigned_count":0,"region_count":32,"nonempty_region_count":int(np.count_nonzero(counts)),"cardinality_sum":int(counts.sum()),"minimum_region_vertex_count":int(counts.min()),"maximum_region_vertex_count":int(counts.max()),"regions":regions,"distributed_groups":groups},
 "heightfield":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"parent_grid_manifest":str(PARENT),"same_heightfield_object_as_1196":True,"runtime_float32_nodes_equal_retained_1196_arrays":grid_identical,"nx_nodes":grid["nx"],"ny_nodes":grid["ny"],"spacing_xy_f32_m":grid["spacing"].astype(float).tolist(),"source_vertices_outside_bed_bounds":bed_query["outside_count"],"skin_capture_vertices_mapped_to_rest_voronoi_by_index":True,"source_pose_min_signed_point_to_facet_gap_m":float(signed.min()),"source_pose_min_vertical_gap_m":float(vertical.min()),"1196_exact_skin_heightfield_triangle_intersection_count":baseline_runtime["runtime_mesh_audit"]["intersections"],"exact_mesh_result_scope":"Transferred from 1196 only after exact same skin hash and exact runtime-grid arrays were verified; no new skin-triangle mesh scan was performed."},
 "native_support_counts":{"status":"not_available_for_1199_before_native_run","1196_counts_not_reused":True},
 "interpretation":["The 32-region owner partition covers each of the 54,949 full-weight NHSKIN source vertices once, with no empty regions; first eight seeds are preserved from 1196.","The fixed bed and skin payloads are byte-identical to the 1196 source-pose geometry inputs; its exact mesh-intersection-free audit transfers by exact identity. Per-region gaps here use new 1199 seed assignments, not the old 1196 Voronoi labels.","This check does not establish dynamic support distribution, contact transitions, reduced drift, or equilibrium."],
}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT),"report_sha256":sha(OUT),"manifest_sha256":sha(MANIFEST),"support_sha256":sha(SUPPORT),"regions":len(regions),"vertices":n,"counts_min":int(counts.min()),"counts_max":int(counts.max()),"min_signed_gap_m":float(signed.min()),"min_vertical_gap_m":float(vertical.min()),"grid_identical":grid_identical,"transferred_exact_skin_bed_intersection_count":baseline_runtime["runtime_mesh_audit"]["intersections"]},indent=2))
