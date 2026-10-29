#!/usr/bin/env python3
"""Read-only lineage, typed-source and gate-pin validator for D run006."""
import hashlib, json
from pathlib import Path
import numpy as np

P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
RUN=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006"
CHAIN=RUN/"source-chain"
BASE=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
SELF622=P/"complete-locality-self-replay-622/joint15-run-002/report.json"
GATE626=P/"boundary-feasibility-612/current025-collar-rings-615/independent-changed-face-gate-626/result.json"
EXPECTED_CANDIDATE="c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83"
EXPECTED_BASE="9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a"
EXPECTED_SELF622="5661751e02258fdfa5696afa2619b07e425a699f3790af3f5e9ef28a0595ce3d"
EXPECTED_GATE626="01855313385f216c14bd2f08009d51bcd8e553f8d8be34aabd718dd76291bc0b"

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def cached_npz(path):
 with np.load(path,allow_pickle=False) as z:
  return {k:np.array(z[k],copy=True) for k in z.files}
def csr(offsets,flat):
 o=np.asarray(offsets,dtype=np.int64); f=np.asarray(flat,dtype=np.int64)
 assert len(o)>=1 and int(o[0])==0 and int(o[-1])==len(f)
 assert np.all(np.diff(o)>=0)
 return [f[int(o[i]):int(o[i+1])].tolist() for i in range(len(o)-1)]
def write_json(path,obj):
 p=Path(path); tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); tmp.replace(p)

candidate=RUN/"candidate-f32-linked-remesh-typed.npz"
result=json.loads((RUN/"result.json").read_text())
accepted=json.loads((RUN/"accepted-patch-operations.json").read_text())
replay=json.loads((CHAIN/"replay-operations.json").read_text())
assert sha(candidate)==EXPECTED_CANDIDATE==result["typed_candidate_sha256"]
assert sha(BASE)==EXPECTED_BASE==result["candidate_input_sha256"]
assert result["accepted_transaction_count"]==15 and result["failed_transaction_count"]==0
assert result["input_faces"]==258217 and result["output_faces"]==257827
assert result["input_vertices"]==129546 and result["output_vertices"]==129546
assert result["new_faces"]==154 and result["removed_faces"]==544
assert result["exact_self_gate"]==result["all185_gate"]==result["D_minus_I_gate"]=="NOT_RUN"
assert result["physical_admission"] is False
assert len(accepted["operations"])==15 and len(accepted["accepted_transactions"])==15
assert accepted["failed_transactions"]==[]

ops_by_key={(int(x["transaction_id"]),int(x["patch_index"])):x for x in accepted["operations"]}
replay_by_key={(int(x["transaction_id"]),int(x["patch_index"])):x for x in replay["operations"]}
expected_source_ids={}
for key,op in ops_by_key.items():
 saved=replay_by_key.get(key,{})
 expected_source_ids[key]=int(saved.get("new_original_id",63000000+key[0]*100+key[1]))
assert len(expected_source_ids)==15 and len(set(expected_source_ids.values()))==15
assert expected_source_ids[(65,0)]==62016500 and expected_source_ids[(227,0)]==62032700
assert sum(len(x["new_faces"]) for x in ops_by_key.values())==154
assert sum(len(x["removed_face_rows"]) for x in ops_by_key.values())==544

out=cached_npz(candidate); base=cached_npz(BASE)
pts=np.asarray(out["points_world_m"]); bpts=np.asarray(base["points_world_m"])
faces=np.asarray(out["faces"],dtype=np.int64); bfaces=np.asarray(base["faces"],dtype=np.int64)
assert pts.shape==bpts.shape==(129546,3) and pts.dtype==bpts.dtype
assert np.array_equal(pts,bpts),"run006 must preserve every input point bitwise"
assert faces.shape==(257827,3) and np.all((faces>=0)&(faces<len(pts)))
for name in ("face_source_original_id","face_source_face_id","face_source_identity0","face_source_identity1","face_source_layer","face_source_body","face_source_bone"):
 assert len(out[name])==len(faces),(name,len(out[name]),len(faces))
orig=np.asarray(out["face_source_original_id"],dtype=np.int64)
fids=np.asarray(out["face_source_face_id"],dtype=np.int64)
i0=np.asarray(out["face_source_identity0"],dtype=np.int64)
i1=np.asarray(out["face_source_identity1"],dtype=np.int64)
schema=json.loads(str(out["face_source_identity_schema_by_original_id_json"][0]))
base_csr=csr(base["candidate_face_ancestry_offsets"],base["candidate_face_ancestry_rows"])
base_union_run=csr(base["union_face_ancestry_offsets"],base["union_face_ancestry_run_original_ids"])
base_union_fid=csr(base["union_face_ancestry_offsets"],base["union_face_ancestry_face_ids"])
base_union_row=csr(base["union_face_ancestry_offsets"],base["union_face_ancestry_rows"])
base_d289=csr(base["d289_parent_row_indptr"],base["d289_parent_rows"])
out_csr=csr(out["candidate_face_ancestry_offsets"],out["candidate_face_ancestry_rows"])
out_union_run=csr(out["union_face_ancestry_offsets"],out["union_face_ancestry_run_original_ids"])
out_union_fid=csr(out["union_face_ancestry_offsets"],out["union_face_ancestry_face_ids"])
out_union_row=csr(out["union_face_ancestry_offsets"],out["union_face_ancestry_rows"])
out_d289=csr(out["d289_parent_row_indptr"],out["d289_parent_rows"])
assert all(len(x)==len(faces) for x in (out_csr,out_union_run,out_union_fid,out_union_row,out_d289))
sidecar_checks=[]
for key,op in sorted(ops_by_key.items()):
 sid=expected_source_ids[key]; rows=np.flatnonzero(orig==sid)
 assert len(rows)==len(op["new_faces"]),(key,sid,len(rows),len(op["new_faces"]))
 side=RUN/"derived-source-views"/f"D-derived-source-{sid}.npz"
 assert side.exists(),(key,sid,str(side))
 s=cached_npz(side); n=len(rows); tri=np.asarray(s["source_faces"],dtype=np.int64)
 assert tri.shape==(n,3) and np.all((tri>=0)&(tri<len(s["points_world_m"])))
 assert np.all(np.asarray(s["face_source_original_id"])==sid)
 assert np.array_equal(s["face_source_face_id"],np.arange(n,dtype=np.int64))
 assert np.array_equal(s["face_source_identity0"],np.arange(n,dtype=np.int64))
 assert np.all(np.asarray(s["face_source_identity1"])==-1)
 assert np.array_equal(pts[faces[rows]],np.asarray(s["points_world_m"])[tri]),(key,sid,"oriented geometry mismatch")
 assert np.all(fids[rows]==np.arange(n,dtype=np.int64))
 assert np.all(i0[rows]==fids[rows]) and np.all(i1[rows]==-1)
 assert str(sid) in schema
 sm=schema[str(sid)]
 assert sm["role"]=="D" and sm["identity_face_id_key"]=="face_source_face_id"
 assert sm["identity_keys"]==["face_source_identity0","face_source_identity1"]
 assert sm["path"]==str(side) and sm["sha256"]==sha(side)
 parent=sorted(set(map(int,op["parent_face_rows"])))
 so=np.asarray(s["parent_candidate_face_offsets"],dtype=np.int64); sf=np.asarray(s["parent_candidate_face_rows"],dtype=np.int64)
 assert len(so)==n+1 and int(so[0])==0 and int(so[-1])==len(sf) and np.all(np.diff(so)>0)
 assert np.array_equal(np.asarray(s["parent_candidate_face_support_input_rows"],dtype=np.int64),np.asarray(parent,dtype=np.int64))
 for j in range(n): assert list(map(int,sf[int(so[j]):int(so[j+1])]))==parent
 expected_cand=sorted(set(int(x) for r in parent for x in base_csr[r]))
 expected_union=sorted(set((int(base_union_run[r][j]),int(base_union_fid[r][j]),int(base_union_row[r][j])) for r in parent for j in range(len(base_union_run[r]))))
 expected_d289=sorted(set(int(x) for r in parent for x in base_d289[r]))
 uo=np.asarray(s["parent_union_source_offsets"],dtype=np.int64)
 assert len(uo)==n+1 and int(uo[0])==0 and int(uo[-1])==len(s["parent_union_source_rows"]) and np.all(np.diff(uo)>0)
 sur=list(map(int,s["parent_union_source_run_original_ids"])); suf=list(map(int,s["parent_union_source_face_ids"])); suw=list(map(int,s["parent_union_source_rows"]))
 side_union=sorted(set(zip(sur,suf,suw)))
 assert side_union==expected_union
 side_cand=sorted(set(map(int,sf[:int(so[1])])))
 assert side_cand==parent and expected_cand
 for r in rows:
  assert sorted(set(map(int,out_csr[int(r)])))==expected_cand
  out_union=sorted(set((int(out_union_run[int(r)][j]),int(out_union_fid[int(r)][j]),int(out_union_row[int(r)][j])) for j in range(len(out_union_run[int(r)]))))
  assert out_union==expected_union
  assert sorted(set(map(int,out_d289[int(r)])))==expected_d289
 sidecar_checks.append({"transaction_id":key[0],"patch_index":key[1],"source_id":sid,"replacement_face_count":n,"removed_parent_faces":len(parent),"candidate_parent_support_count":len(expected_cand),"union_parent_support_count":len(expected_union),"d289_parent_support_count":len(expected_d289),"sidecar_sha256":sha(side)})

selfr=json.loads(SELF622.read_text()); self_sha=sha(SELF622)
assert self_sha==EXPECTED_SELF622
assert selfr["candidate_sha256"]==EXPECTED_CANDIDATE and selfr["status"]=="PASS_SELF_GATE"
assert selfr["complete"] is True and selfr["exact_triangle_pair_count"]==0
assert selfr["physical_admission"] is False
bgr=json.loads(GATE626.read_text()); gate_sha=sha(GATE626)
assert gate_sha==EXPECTED_GATE626
assert bgr["candidate_sha256"]==EXPECTED_CANDIDATE
assert bgr["status"]=="PASS_CHANGED_FACE_STRUCTURE_AND_WHOLE_BOUNDARY_ANCESTRY_ONLY"
assert bgr["whole_candidate"]["face_count"]==257827 and bgr["whole_candidate"]["vertex_count"]==129546
assert bgr["whole_candidate"]["coordinates_bitwise_unchanged"] is True
assert bgr["whole_candidate"]["directed_boundary_equal"] is True
assert bgr["whole_candidate"]["new_face_source_ancestry_equals_removed_patch_ancestry_union"] is True
assert bgr["new_face_cross_operation_self_pairs"]==0
assert bgr["physical_admission"] is False
reported_ids={int(tx):int(sid) for tx,sid in bgr["derived_source_ids_by_transaction"].items()}
expected_tx_ids={int(tx):sid for (tx,_),sid in expected_source_ids.items()}
assert reported_ids==expected_tx_ids
assert bgr["whole_candidate_self_intersection_audit"]=="NOT_RUN_HERE; coordinated single full-self audit remains separate"
receipt={"schema":"run006-typed-source-lineage-validation.v1","candidate_path":str(candidate),"candidate_sha256":sha(candidate),"source_chain_manifest_path":str(CHAIN/"source-chain-manifest.json"),"source_chain_manifest_sha256":sha(CHAIN/"source-chain-manifest.json"),"validator_path":str(Path(__file__).resolve()),"validator_sha256":sha(__file__),"input_candidate":{"path":str(BASE),"sha256":sha(BASE),"point_coordinates_bitwise_preserved":True},"accepted_operations_path":str(RUN/"accepted-patch-operations.json"),"accepted_operations_sha256":sha(RUN/"accepted-patch-operations.json"),"accepted_transaction_ids":accepted["accepted_transactions"],"source_ids_by_transaction":{str(tx):sid for tx,sid in sorted(expected_tx_ids.items())},"replacement_faces":int(result["new_faces"]),"removed_faces":int(result["removed_faces"]),"candidate_face_count":int(result["output_faces"]),"candidate_vertex_count":int(result["output_vertices"]),"all_15_sidecars_and_candidate_CSR_validated":True,"sidecar_validation_summary":sidecar_checks,"separate_gate_pins":{"complete_locality_self":{"path":str(SELF622),"sha256":self_sha,"status":selfr["status"],"complete":selfr["complete"],"remaining_exact_triangle_pairs":selfr["exact_triangle_pair_count"],"physical_admission":selfr["physical_admission"]},"changed_face_structure_and_whole_boundary":{"path":str(GATE626),"sha256":gate_sha,"status":bgr["status"],"whole_candidate":bgr["whole_candidate"],"changed_vs_retained_or_structures_pair_archive":bgr["changed_vs_retained_or_structures_pair_archive"],"new_face_cross_operation_self_pairs":bgr["new_face_cross_operation_self_pairs"],"whole_candidate_self_intersection_audit":bgr["whole_candidate_self_intersection_audit"],"physical_admission":bgr["physical_admission"]}},"engine_run_gates":{"exact_self_gate":result["exact_self_gate"],"all185_gate":result["all185_gate"],"D_minus_I_gate":result["D_minus_I_gate"],"physical_admission":result["physical_admission"]},"scope":"Typed identity and full source-ancestry verification only. The 622 complete-locality self gate and 626 changed-face/whole-boundary gate are distinct pinned evidence. Run006 result remains NOT_RUN for full exact self/all185/D-minus-I and does not claim physical admission."}
receipt_path=CHAIN/"typed-lineage-validation.json"
write_json(receipt_path,receipt)
print(json.dumps({"status":"PASS_TYPED_SOURCE_LINEAGE","receipt":str(receipt_path),"receipt_sha256":sha(receipt_path),"validator":str(Path(__file__).resolve()),"validator_sha256":sha(__file__),"candidate_sha256":sha(candidate),"transactions":accepted["accepted_transactions"],"source_ids":receipt["source_ids_by_transaction"],"replacement_faces":result["new_faces"],"removed_faces":result["removed_faces"],"self622_sha256":self_sha,"self622_status":selfr["status"],"gate626_sha256":gate_sha,"gate626_status":bgr["status"],"all_sidecars_and_CSR":True,"run006_whole_gates":{"self":result["exact_self_gate"],"all185":result["all185_gate"],"D_minus_I":result["D_minus_I_gate"]}},sort_keys=True))
