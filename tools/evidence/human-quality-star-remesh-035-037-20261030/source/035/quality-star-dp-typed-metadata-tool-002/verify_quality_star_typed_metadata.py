import csv,hashlib,json
from pathlib import Path
import numpy as np
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=BASE/"quality-star-dp-batch-035-typed-metadata-006"
SOURCES={
"O":BASE/"microedge-csr-canonical-011/O/candidate-O-csr-canonical.npz",
"I":BASE/"local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz",
"D":BASE/"current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary={}
for r,sp in SOURCES.items():
 cp=OUT/f"candidate-{r}-quality-star-typed.npz"
 mp=OUT/f"row-maps-{r}.npz"
 up=OUT/f"patch-unions-{r}.json"
 src=np.load(sp,allow_pickle=False); srcd={k:np.asarray(src[k]).copy() for k in src.files}; src.close()
 cand=np.load(cp,allow_pickle=False); cd={k:np.asarray(cand[k]).copy() for k in cand.files}; cand.close()
 maps=np.load(mp,allow_pickle=False); md={k:np.asarray(maps[k]).copy() for k in maps.files}; maps.close()
 unions=json.loads(up.read_text())
 pt=md["output_point_to_input_row"]; pmap=md["input_point_to_output_row"]
 fo=md["output_face_to_input_row"]; fmap=md["input_face_to_output_row"]
 assert cd["points_world_m"].tobytes()==srcd["points_world_m"][pt].tobytes()
 assert np.array_equal(pmap[pt],np.arange(len(pt)))
 assert np.array_equal(fo[fo>=0].shape, np.asarray([np.count_nonzero(fo>=0)]))
 oldf=np.flatnonzero(fo>=0)
 assert set(map(int,fo[oldf]))==set(range(len(srcd["faces"])))-set(map(int,md["retired_input_face_rows"]))
 for outrow in oldf:
  srow=int(fo[outrow])
  expected=pmap[srcd["faces"][srow]]
  assert np.array_equal(cd["faces"][outrow],expected)
  for key in ("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1"):
   assert cd[key][outrow]==srcd[key][srow],(r,key,outrow)
 offs=cd["quality_star_patch_union_offsets"]
 assert len(offs)==len(cd["faces"])+1
 assert int(offs[-1])==len(cd["quality_star_patch_union_face_source_layer"])
 patch_by_row={int(row):patch for patch in unions["patches"] for row in patch["new_output_face_rows"]}
 assert set(patch_by_row)==set(map(int,np.flatnonzero(fo<0)))
 schema=json.loads(str(cd["face_source_identity_schema_by_original_id_json"][0]))
 for row,patch in patch_by_row.items():
  expect=np.asarray(patch["complete_typed_parent_union"],dtype=np.int64)
  a,b=map(int,offs[row:row+2])
  got=np.column_stack([cd[f"quality_star_patch_union_face_source_{lane}"][a:b] for lane in ("layer","body","bone","face_id","original_id","identity0","identity1")])
  assert np.array_equal(got,expect),(r,row,"typed union mismatch")
  sid=int(cd["face_source_original_id"][row])
  assert schema[str(sid)]["parent_union_tuple_sha256"]==patch["typed_union_sha256"]
  assert int(cd["face_source_face_id"][row])==-1
 if r=="D":
  for key in ("point_preservation_source_asset_index","point_preservation_source_vertex_row"):
   assert np.array_equal(cd[key],srcd[key][pt])
 summary[r]={"source_sha256":sha(sp),"candidate_sha256":sha(cp),"maps_sha256":sha(mp),
             "points":[len(srcd["points_world_m"]),len(cd["points_world_m"])],
             "faces":[len(srcd["faces"]),len(cd["faces"])],
             "new_faces":len(patch_by_row),"typed_union_tuple_records":int(offs[-1]),
             "point_dtype":str(cd["points_world_m"].dtype)}
for role in ("O","I"):
 cand=np.load(OUT/f"candidate-{role}-quality-star-typed.npz",allow_pickle=False)
 offsets=np.asarray(cand["face_ancestry_offsets"],dtype=np.int64)
 lane_names=("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")
 assert offsets.shape==(len(cand["faces"])+1,) and offsets[0]==0 and offsets[-1]==len(cand[lane_names[0]])
 assert np.all(np.diff(offsets)>0)
 assert all(len(cand[k])==offsets[-1] for k in lane_names)
 if role=="I":
  new_rows=np.flatnonzero(fo==-1)
  input_face_count=len(np.load(SOURCES["I"],allow_pickle=False)["faces"])
  for outrow in new_rows:
   a,b=map(int,offsets[outrow:outrow+2])
   assert a<b
   assert np.all((cand["face_ancestry_source_face_rows"][a:b]>=0)&(cand["face_ancestry_source_face_rows"][a:b]<input_face_count))
 cand.close()
if True:
 cand=np.load(OUT/"candidate-D-quality-star-typed.npz",allow_pickle=False)
 offsets=np.asarray(cand["union_face_ancestry_offsets"],dtype=np.int64)
 lanes=("union_face_ancestry_rows","union_face_ancestry_run_original_ids","union_face_ancestry_face_ids")
 assert offsets.shape==(len(cand["faces"])+1,) and offsets[0]==0 and offsets[-1]==len(cand[lanes[0]])
 assert np.all(np.diff(offsets)>0)
 assert all(len(cand[k])==offsets[-1] for k in lanes)
 cand.close()
with (OUT/"protected149-reindexed.csv").open(newline="") as f: rows=list(csv.DictReader(f))
assert len(rows)==149
assert set(int(row["quality_star_output_surface_vertex_row"])>=0 for row in rows)=={True}
receipt=json.loads((OUT/"metadata-receipt.json").read_text())
assert receipt["status"]=="PASS_METADATA_REINDEX_ONLY" and not receipt["geometry_or_physics_admission"]
print(json.dumps({"status":"PASS_INDEPENDENT_STATIC_REINDEX_CHECK","roles":summary,"protected149_rows":len(rows),
"protected149_role_counts":{r:sum(x["role"]==r for x in rows) for r in ("O","I","D")}},sort_keys=True))

