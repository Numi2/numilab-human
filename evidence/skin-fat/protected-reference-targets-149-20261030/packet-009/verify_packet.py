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
for x,h in rec["input_sha256"].items():
 assert sha(Path(x))==h,(x,h,sha(Path(x)))
with (O/"protected-149-targets.csv").open(newline="") as f: rows=list(csv.DictReader(f))
assert len(rows)==149
assert [int(x["canonical_protected_row"]) for x in rows]==list(range(149))
head=[x for x in rows if int(x["canonical_protected_row"])<135]
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
assert rec["physical_admission"] is False and rec["runtime"]==0 and rec["accepted_steps"]==0
report={"schema":"numi.protected-149-target-packet-independent-check.v1","packet_receipt_sha256":sha(O/"receipt.json"),"csv_sha256":sha(O/"protected-149-targets.csv"),"script_sha256":sha(O/"build_packet.py"),"validation_script_sha256":sha(O/"verify_packet.py"),"rows":149,"head_rows":135,"deep_attachment_rows":14,"remapped_attachment_rows":2,"exact_retained_attachment_rows":12,"targets":target_summary,"physical_admission":False,"runtime":0,"accepted_steps":0}
(O/"packet-validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
