import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
O=P/"protected-149-current-surface-crosswalk-001/run-009-correct-witness-operand-packet"
def sha(x):
 h=hashlib.sha256()
 with open(x,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
rec=json.loads((O/"receipt.json").read_text())
assert sha(O/"protected-149-targets.csv")==rec["output_csv_sha256"]
assert sha(O/"deep-target-details.json")==rec["deep_details_sha256"]
for x,h in rec["input_sha256"].items():
 assert sha(Path(x))==h,(x,h,sha(Path(x)))
with (O/"protected-149-targets.csv").open(newline="") as f: rows=list(csv.DictReader(f))
assert len(rows)==149
assert [int(x["canonical_protected_row"]) for x in rows]==list(range(149))
head=[x for x in rows if x["mapping_kind"]=="head_ring_boundary_target"]
deep=[x for x in rows if x["stable_attachment_id"]]
assert len(head)==135 and len(deep)==14
assert {x["role"] for x in head}=={"O","I","D"}
assert {int(x["stable_attachment_id"]) for x in deep}=={8467,11704,13555,14506,21455,29464,32578,33621,34352,37264,39356,40968,41702,46481}
byid={int(x["stable_attachment_id"]):x for x in deep}
assert int(byid[29464]["current_surface_vertex_row"])==83239
assert int(byid[14506]["current_surface_vertex_row"])==26444
det=json.loads((O/"deep-target-details.json").read_text())
details={int(x["stable_attachment_id"]):x for x in det["remapped_deep_attachment_targets"]}
assert set(details)=={29464,14506}
search=json.loads((P/"protected-149-current-surface-crosswalk-001/run-007-corrected-parent-face-targets/corrected-search.json").read_text())
qsearch={int(x["attachment_id"]):x for x in search["results"]}
replay=np.load(P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz",allow_pickle=False)
cand=np.load(P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz",allow_pickle=False)
banks={23:P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-007/bank-operands/bank-23-33.npz",139:P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-017/bank-operands/bank-139-132.npz"}
target_summary=[]
for aid,d in details.items():
 srow=qsearch[aid]; assert int(d["candidate_surface_vertex_row"])==int(srow["selected_vertex_row"]) if "selected_vertex_row" in srow else int(d["candidate_surface_vertex_row"]) in {int(z["vertex_row"]) for z in srow["candidates_ranked"]}
 g=int(srow["native_replay_global_face_row"]); operand=int(srow["native_replay_operand_face_row"]); stable=int(srow["native_replay_stable_face_row"]); tag=int(srow["canonical_witness_face_id_tag"])
 bi=65 if aid==29464 else 146
 assert int(replay["bank_face_offsets"][bi])<=g<int(replay["bank_face_offsets"][bi+1])
 assert int(replay["face_source_operand_face_row"][g])==operand
 assert int(replay["face_source_stable_row"][g])==stable
 bank=np.load(banks[int(srow["body"])],allow_pickle=False)
 faces=np.asarray(bank["faces_f64"],dtype=int); ids=np.asarray(bank["face_id"],dtype=int)
 assert int(ids[operand])==tag
 path=list(map(int,d["source_face_path_operand_rows"]))
 tags=list(map(int,d["source_face_path_face_id_tags"]))
 assert path[0]==operand and path[-1]==int(srow["selected_source_operand_face_row"] if "selected_source_operand_face_row" in srow else next(z["source_operand_face_row"] for z in srow["candidates_ranked"] if int(z["vertex_row"])==int(d["candidate_surface_vertex_row"])))
 assert len(path)==len(tags)==int(d["source_face_path_hops"])+1
 assert [int(ids[x]) for x in path]==tags
 for a,b in zip(path,path[1:]): assert len(set(faces[a])&set(faces[b]))>=2
 assert float(d["old_reference_to_target_displacement_m"])<=0.010
 assert float(d["target_to_parent_source_face_residual_m"])<=4e-6*(3.0**0.5/2.0)
 assert float(d["full86_confidence_weight"])>=0.85
 assert int(d["full86_confidence_body"])==int(d["canonical_donor_body"])
 assert float(d["candidate_surface_quality"]["min_edge_m"])>0
 assert float(d["candidate_surface_quality"]["min_area_m2"])>0
 assert float(d["candidate_surface_quality"]["min_altitude_m"])>0
 target_summary.append({"attachment_id":aid,"target_vertex_row":int(d["candidate_surface_vertex_row"]),"witness_global_row":g,"witness_operand_row":operand,"witness_stable_row":stable,"witness_face_id_tag":tag,"target_face_id_tag":int(d["source_target_face_id_tag"]),"hops":int(d["source_face_path_hops"]),"centroid_path_m":float(d["source_face_centroid_path_length_m"]),"reference_offset_m":float(d["old_reference_to_target_displacement_m"]),"parent_face_residual_m":float(d["target_to_parent_source_face_residual_m"]),"full86_weight":float(d["full86_confidence_weight"]),"min_edge_m":float(d["candidate_surface_quality"]["min_edge_m"]),"min_area_m2":float(d["candidate_surface_quality"]["min_area_m2"]),"min_altitude_m":float(d["candidate_surface_quality"]["min_altitude_m"])})
# Verify head-ring ancestry against the actual role-local source maps.
head_path=P/"protected-149-current-surface-crosswalk-001/head-ring-135-root-008/head-ring-135-targets.csv"
with head_path.open(newline="") as f: head=list(csv.DictReader(f))
assert len(head)==135
sm=np.load(P/"paired-fine-headwall-rim-635/run-013/output/closed-skin-source-map.npz",allow_pickle=False)
fm=np.load(P/"closed-fat-source-map-649/closed-fat-source-map.npz",allow_pickle=False)
o_ids=np.asarray(sm["boundary_O_fine_vertex_rows"],dtype=int)
i_ids=np.asarray(sm["boundary_I_fine_vertex_rows"],dtype=int)
dlayer=np.asarray(fm["output_vertex_source_layer"],dtype=int)
dsource=np.asarray(fm["output_vertex_source_row"],dtype=int)
dlookup={}
for j,v in enumerate(dsource):
 if int(dlayer[j])==2:dlookup.setdefault(int(v),[]).append(j)
bycanonical={int(x["canonical_protected_row"]):x for x in rows}
for h in head:
 cr=int(h["canonical_protected_row"]); out=bycanonical[cr]; role=h["role"]; v=int(h["current_surface_vertex_row"])
 assert int(out["current_surface_vertex_row"])==v and out["role"]==role
 if role=="O":
  mrow=int(out["source_map_row"]);assert int(o_ids[mrow])==v
 elif role=="I":
  mrow=int(out["source_map_row"]);assert int(i_ids[mrow])==v
 else:
  j=int(out["source_map_row"]); lineage=json.loads(out["source330_parent_star_tets_json"])
  assert int(dlayer[j])==2 and int(dsource[j])==int(lineage["fine_D_source_vertex_row"])
  assert len(dlookup[int(dsource[j])])==1
# Every immutable deep row retains the measured full-86 lane-to-body binding.
for aid,x in byid.items():
 assert int(x["full86_confidence_body"])==int(x["canonical_donor_body"])
 assert float(x["full86_confidence_weight"])>=0.85,(aid,x["full86_confidence_weight"])
# Verify every retained attachment still points to its original canonical volume node.
base=np.load(P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz",allow_pickle=False)
for aid,x in byid.items():
 if aid in details: continue
 cr=int(x["canonical_protected_row"])
 assert int(x["base_volume_node_row"])==int(base["pinned_volume_node_rows"][cr])

# Independently recompute every target coordinate and every immutable full86
# binding, rather than treating the packet's descriptive values as evidence.
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
with (CAP/"parent-deep-attachments.csv").open(newline="") as f:
 arows=[r for r in csv.DictReader(f) if r["role"]=="D"]
attachment={int(r["source_vertex_row"]):r for r in arows}
with (CAP/"compiled-parent-cell-mass.csv.weights.csv").open(newline="") as f:
 weight_by_node={int(r["vertex"]):r for r in csv.DictReader(f)}
pose=json.loads((CAP/"registered-skin-poses-100.json").read_text())
lane_body=[int(r["body_index"]) for r in pose["bodies"]]
with (CAP/"compiled-parent-cell-mass.csv.tets.csv").open(newline="") as f:
 tet_rows=list(csv.DictReader(f))
with (CAP/"compiled-parent-cell-mass.csv.labels.csv").open(newline="") as f:
 label_by_tet={int(r["parent_tet_row"]):r for r in csv.DictReader(f)}
source_star={}
for r in tet_rows:
 ti=int(r["parent_tet_row"])
 for k in range(4):source_star.setdefault(int(r["parent_node"+str(k)]),[]).append(ti)
basepoints=np.asarray(base["points_current_world_m_f32"],dtype=np.float32)
baseparent=np.asarray(base["parent_volume_node_row"],dtype=int)
pins=np.asarray(base["pinned_volume_node_rows"],dtype=int)
head_rows=[r for r in rows if r["mapping_kind"]=="head_ring_boundary_target"]
assert len(head_rows)==135 and len({int(r["canonical_protected_row"]) for r in head_rows})==135
assert {int(r["canonical_protected_row"]) for r in head_rows}.isdisjoint(int(r["canonical_protected_row"]) for r in deep)
o=np.load(P/"surface-transfer-checkpoint-392/payload/O/candidate.npz",allow_pickle=False)
i=np.load(P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz",allow_pickle=False)
role_points={"O":np.asarray(o["points_world_m"],dtype=np.float32),"I":np.asarray(i["points_world_m"],dtype=np.float32),"D":np.asarray(cand["points_world_m"],dtype=np.float32)}
coordinate_rows=0;star_rows=0
for r in rows:
 cr=int(r["canonical_protected_row"]);bn=int(r["base_volume_node_row"]);pn=int(r["source330_parent_node_row"])
 assert pins[cr]==bn and baseparent[bn]==pn
 p=np.asarray([float(r["target_world_"+x+"_m"]) for x in ("x","y","z")])
 assert np.isfinite(p).all()
 if r["stable_attachment_id"]:
  aid=int(r["stable_attachment_id"]);a=attachment[aid];node=int(a["volume_node_row"]);donor=int(a["donor_body"])
  assert node==pn and donor==int(r["canonical_donor_body"])
  wr=weight_by_node[node];ww=np.asarray([float(wr["w"+str(k)]) for k in range(86)])
  lane=int(np.argmax(ww));assert lane==int(r["full86_confidence_lane"])
  assert lane_body[lane]==donor==int(r["full86_confidence_body"])
  assert ww[lane]>=.85 and abs(ww[lane]-float(r["full86_confidence_weight"]))<1e-15
  star=json.loads(r["source330_parent_star_tets_json"])
  assert sorted(int(x["parent_tet_row"]) for x in star)==sorted(source_star[node])
  for x in star:
   ti=int(x["parent_tet_row"]);lab=label_by_tet[ti];tr=tet_rows[ti]
   assert int(x["material_index"])==int(tr["material_index"])==16
   assert int(x["donor_body"])==int(tr["donor_body"])
   assert int(x["geometric_donor_body"])==int(lab["geometric_donor_body"])
   assert int(x["anatomical_region_id"])==int(lab["anatomical_region_id"])
   assert int(x["stable_source_face_id"])==int(lab["stable_source_face_id"])
  star_rows+=len(star)
  if aid not in details:
   assert int(r["current_surface_vertex_row"])==-1
   assert np.array_equal(p.astype(np.float32),basepoints[bn])
   original=np.asarray([float(a["current_"+x+"_m"]) for x in ("x","y","z")])
   assert np.array_equal(p.astype(np.float32),original.astype(np.float32))
  else:
   assert np.array_equal(p.astype(np.float32),role_points["D"][int(r["current_surface_vertex_row"])])
 else:
  assert r["mapping_kind"]=="head_ring_boundary_target"
  assert np.array_equal(p.astype(np.float32),role_points[r["role"]][int(r["current_surface_vertex_row"])])
 coordinate_rows+=1
assert coordinate_rows==149

assert rec["physical_admission"] is False and rec["runtime"]==0 and rec["accepted_steps"]==0
report={"independently_recomputed_coordinate_rows":coordinate_rows,"independently_checked_parent_star_entries":star_rows,"head_classification":"mapping_kind, canonical indices are interleaved","schema":"numi.protected-149-target-packet-independent-check.v1","packet_receipt_sha256":sha(O/"receipt.json"),"csv_sha256":sha(O/"protected-149-targets.csv"),"script_sha256":sha(O/"build_packet.py"),"validation_script_sha256":sha(Path(__file__)),"rows":149,"head_rows":135,"deep_attachment_rows":14,"remapped_attachment_rows":2,"exact_retained_attachment_rows":12,"targets":target_summary,"physical_admission":False,"runtime":0,"accepted_steps":0}
(Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/protected-149-current-surface-crosswalk-001/root-010-independent-check')/"packet-validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
