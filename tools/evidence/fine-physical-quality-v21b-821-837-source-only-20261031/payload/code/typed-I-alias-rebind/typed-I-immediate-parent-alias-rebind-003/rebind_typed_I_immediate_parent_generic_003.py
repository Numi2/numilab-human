#!/usr/bin/env python3
"""Generic source-only typed-I immediate-parent alias rebind.

Replay source rows/quotients/unions against the run's actual I input. If a
canonical parent alias is supplied, require a receipt-authenticated exact
non-alias array equivalence before using that canonical file as the alias.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
ENGINE=BASE/"coupled-anchor-collapse-780-root/coupled_anchor_collapse_781.py"
ENGINE_SHA="739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb"
ALIAS={"typed_parent_candidate_path","typed_parent_candidate_sha256"}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def eq(a,b):
 a,b=np.asarray(a),np.asarray(b)
 return a.shape==b.shape and a.dtype==b.dtype and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def pinned(entry,label):
 if not isinstance(entry,dict) or not {"path","sha256"}<=set(entry): raise ValueError(label+" missing path/SHA")
 p=Path(entry["path"]).resolve();got=sha(p)
 if got!=entry["sha256"]: raise ValueError(label+" SHA mismatch: "+got)
 return p,got
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--state",required=True,type=Path)
 ap.add_argument("--config",required=True,type=Path)
 ap.add_argument("--output",required=True,type=Path)
 ap.add_argument("--canonical-parent",type=Path)
 ap.add_argument("--canonical-parent-receipt",type=Path)
 a=ap.parse_args()
 state_path=a.state.resolve(); state_sha=sha(state_path); state=json.loads(state_path.read_text())
 if state.get("status")!="CANDIDATE_PENDING_775_ALL185" or state.get("physical_admission") is not False or int(state.get("native_steps",-1))!=0:
  raise ValueError("input run is not a source-only candidate pending 775")
 cfg_path,cfg_sha=pinned(state["run_config"],"state run-config")
 supplied_cfg=a.config.resolve()
 if cfg_path!=supplied_cfg: raise ValueError("CLI config differs from state run-config")
 cfg=json.loads(cfg_path.read_text())
 actual_entry=state["source_inputs"]["baseline_surfaces"]["I"]
 if actual_entry!=cfg["baseline_surfaces"]["I"] or actual_entry!=state["source_inputs"]["surface_I"] or actual_entry!=cfg["surface_I"]:
  raise ValueError("state/config disagree on actual I input")
 actual_path,actual_sha=pinned(actual_entry,"actual I input")
 candidate_entry=state["outputs"]["candidate-I-typed-current.npz"]
 candidate_path,candidate_sha=pinned(candidate_entry,"candidate I")
 parent_alias_path=actual_path; parent_alias_sha=actual_sha; alias_mode="DIRECT_RUN_INPUT"
 equivalence=None
 if bool(a.canonical_parent)!=bool(a.canonical_parent_receipt):
  raise ValueError("canonical-parent and receipt must be supplied together")
 actual=load(actual_path); candidate=load(candidate_path)
 if a.canonical_parent:
  parent_alias_path=a.canonical_parent.resolve(); parent_alias_sha=sha(parent_alias_path)
  receipt_path=a.canonical_parent_receipt.resolve(); receipt=json.loads(receipt_path.read_text()); receipt_sha=sha(receipt_path)
  if receipt.get("schema")!="h706.typed-I-immediate-parent-alias-rebind.v1" or receipt.get("status")!="PASS_SOURCE_ONLY_TYPED_I_IMMEDIATE_PARENT_ALIAS_REBOUND":
   raise ValueError("canonical parent receipt is not an accepted alias-rebind receipt")
  inrec=receipt["input_candidate"]; outrec=receipt["output_candidate"]
  if Path(inrec["path"]).resolve()!=actual_path or inrec["sha256"]!=actual_sha:
   raise ValueError("canonical receipt input is not this run's actual parent")
  if Path(outrec["path"]).resolve()!=parent_alias_path or outrec["sha256"]!=parent_alias_sha:
   raise ValueError("canonical receipt does not bind the requested parent file")
  if receipt.get("replay",{}).get("all_other_typed_arrays_match_replay") is not True or receipt.get("replay",{}).get("non_alias_arrays_byte_identical_after_serialization") is not True:
   raise ValueError("canonical receipt lacks complete typed replay/equivalence claims")
  canon=load(parent_alias_path)
  if set(canon)!=set(actual): raise ValueError("canonical parent array key-domain differs from actual input")
  changed=[]
  for key in sorted(actual):
   if key in ALIAS: continue
   if not eq(actual[key],canon[key]): raise ValueError("canonical parent differs outside alias fields: "+key)
  if not {**{k:np.asarray(v).item() for k,v in ((k,canon[k]) for k in ALIAS)}}:
   raise AssertionError("unreachable")
  if str(np.asarray(canon["typed_parent_candidate_path"]).item())!=str(receipt["actual_immediate_parent"]["path"]) or str(np.asarray(canon["typed_parent_candidate_sha256"]).item())!=receipt["actual_immediate_parent"]["sha256"]:
   raise ValueError("canonical parent aliases do not bind to receipt-authenticated parent")
  equivalence={"actual_run_input":{"path":str(actual_path),"sha256":actual_sha},
    "canonical_parent_alias":{"path":str(parent_alias_path),"sha256":parent_alias_sha},
    "receipt":{"path":str(receipt_path),"sha256":receipt_sha},
    "non_alias_array_key_count":len(actual)-len(ALIAS),"non_alias_arrays_exactly_equal":True}
  alias_mode="RECEIPT_AUTHENTICATED_ARRAY_EQUIVALENT_PARENT"
 p=np.asarray(actual["points_world_m_f32"],np.float32); faces=np.asarray(actual["faces"],np.int64)
 q=np.arange(len(p),dtype=np.int64); points=p.copy(); ops=[]; rowop={}; seen_nodes=set(); seen_faces=set()
 for i,r in enumerate(state.get("operation_records",[])):
  k,d=int(r["keep"]),int(r["drop"])
  if min(k,d)<0 or max(k,d)>=len(p) or k==d or q[k]!=k or q[d]!=d or k in seen_nodes or d in seen_nodes:
   raise ValueError(f"operation {i}: invalid/overlapping endpoint")
  q[d]=k; anchor=np.asarray(r["anchor_f32"],np.float32)
  if anchor.shape!=(3,): raise ValueError(f"operation {i}: invalid F32 anchor")
  points[k]=anchor
  endpoint=np.union1d(np.flatnonzero(np.any(faces==k,axis=1)),np.flatnonzero(np.any(faces==d,axis=1)))
  listed=np.asarray(r["surface_parent_faces"],np.int64)
  if not np.array_equal(endpoint,listed): raise ValueError(f"operation {i}: complete endpoint star mismatch")
  if seen_faces.intersection(map(int,listed)): raise ValueError(f"operation {i}: overlapping endpoint face rows")
  seen_nodes.update((k,d)); seen_faces.update(map(int,listed))
  for row in listed: rowop[int(row)]=i
  op=dict(r);op["surface_full_endpoint_face_rows"]=listed.tolist()
  ops.append(op)
 mapped=q[faces]; valid=~np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1)
 parent_rows=np.flatnonzero(valid).astype(np.int64); output_faces=mapped[valid].astype(np.int64)
 owner_sha=sha(ENGINE)
 if owner_sha!=ENGINE_SHA: raise ValueError("pinned 781 CSR owner changed")
 spec=importlib.util.spec_from_file_location("typed_alias_owner_781",ENGINE)
 owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
 unions=[];ancestry=[];op_rows=[]
 for row in parent_rows:
  oi=rowop.get(int(row),-1)
  if oi>=0:
   rows=ops[oi]["surface_full_endpoint_face_rows"]
   u=owner.B.union_tuples(actual,rows);anc=owner.B.ancestry_union(actual,rows)
  else:
   u=owner.A.rows_union(actual,int(row));anc=owner.A.rows_anc(actual,int(row))
  if not u or not anc:raise ValueError(f"empty typed source provenance at parent face {int(row)}")
  unions.append(u);ancestry.append(anc);op_rows.append(oi)
 uoff,ud=owner.A.csr(unions,7);aoff,ad=owner.A.csr(ancestry,3)
 expected={k:np.array(v,copy=True) for k,v in actual.items()}
 for key in list(expected):
  if key.startswith("source_") and key.endswith("_operation_rows"):expected.pop(key)
 expected["points_world_m_f32"]=points
 for key,val in list(expected.items()):
  arr=np.asarray(val)
  if key!="points_world_m_f32" and ("points_world_m" in key or key=="points") and arr.shape==points.shape and np.issubdtype(arr.dtype,np.floating):
   alias=arr.copy()
   for r in ops:alias[int(r["keep"])]=np.asarray(r["anchor_f32"],dtype=arr.dtype)
   expected[key]=alias
 expected["faces"]=output_faces
 fmap=np.asarray(actual.get("quality_star_face_output_to_input_row",np.arange(len(faces))),np.int64)[parent_rows].copy()
 fmap[np.asarray(op_rows)>=0]=-1
 expected["quality_star_face_output_to_input_row"]=fmap
 expected["quality_star_patch_union_offsets"]=uoff;expected["face_ancestry_offsets"]=aoff
 expected["source_input_face_rows"]=parent_rows;expected["source_input_vertex_quotient"]=q
 expected["source_collapse_operation_rows"]=np.asarray(op_rows,np.int64)
 for j,key in enumerate(owner.A.T):expected[key]=np.asarray([u[0][j] for u in unions],np.int64)
 for j,key in enumerate(owner.A.U):expected[key]=ud[:,j]
 for j,key in enumerate(owner.A.A):expected[key]=ad[:,j]
 expected["coupled_anchor_operations_json"]=np.asarray([json.dumps({
  "role":"I","keep_parent_point_row":r["keep"],"drop_parent_point_row":r["drop"],
  "moved_anchor_world_m":r["anchor_f32"],"move_cap_m":float(cfg["move_cap_m"]),
  "complete_parent_face_rows":r["surface_full_endpoint_face_rows"],
  "retired_parent_face_rows":r["retired_surface_faces"],
  "physical_admission":False,"native_steps":0},sort_keys=True) for r in ops])
 if set(expected)!=set(candidate):raise ValueError("typed output array key-domain differs from lineage replay")
 for key in sorted(set(expected)-ALIAS):
  if not eq(candidate[key],expected[key]):raise ValueError("typed lineage replay differs at "+key)
 if not np.array_equal(candidate["source_input_face_rows"],parent_rows) or not np.array_equal(candidate["source_input_vertex_quotient"],q):
  raise ValueError("immediate face-row map or vertex quotient mismatch")
 oldpath=str(np.asarray(candidate["typed_parent_candidate_path"]).item());oldsha=str(np.asarray(candidate["typed_parent_candidate_sha256"]).item())
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);out_npz=out/"candidate-I-typed-current.npz"
 final={k:np.array(v,copy=True) for k,v in candidate.items()}
 final["typed_parent_candidate_path"]=np.asarray([str(parent_alias_path)],dtype=candidate["typed_parent_candidate_path"].dtype)
 final["typed_parent_candidate_sha256"]=np.asarray([parent_alias_sha],dtype=candidate["typed_parent_candidate_sha256"].dtype)
 np.savez_compressed(out_npz,**final);out_sha=sha(out_npz);saved=load(out_npz)
 for key in sorted(set(candidate)-ALIAS):
  if not eq(candidate[key],saved[key]):raise ValueError("serialization changed non-alias array "+key)
 if str(np.asarray(saved["typed_parent_candidate_path"]).item())!=str(parent_alias_path) or str(np.asarray(saved["typed_parent_candidate_sha256"]).item())!=parent_alias_sha:
  raise ValueError("serialized parent aliases mismatch")
 receipt={"schema":"h706.typed-I-immediate-parent-alias-rebind.v2","status":"PASS_SOURCE_ONLY_TYPED_I_IMMEDIATE_PARENT_ALIAS_REBOUND",
  "state":{"path":str(state_path),"sha256":state_sha},"run_config":{"path":str(cfg_path),"sha256":cfg_sha},
  "actual_run_input":{"role":"I","path":str(actual_path),"sha256":actual_sha},
  "typed_parent_alias":{"path":str(parent_alias_path),"sha256":parent_alias_sha,"mode":alias_mode,"canonical_equivalence":equivalence},
  "input_candidate":{"path":str(candidate_path),"sha256":candidate_sha,"prior_typed_parent_path":oldpath,"prior_typed_parent_sha256":oldsha},
  "output_candidate":{"path":str(out_npz),"sha256":out_sha},
  "owner":{"path":str(ENGINE),"sha256":ENGINE_SHA},
  "replay":{"operations":len(ops),"input_points":len(p),"input_faces":len(faces),"output_faces":len(output_faces),
   "exact_endpoint_stars":True,"exact_immediate_face_rows":True,"exact_vertex_quotient":True,
   "all_7_semantic_union_lanes_and_offsets_replayed":True,"all_3_face_ancestry_lanes_and_offsets_replayed":True,
   "all_scalar_lanes_equal_sorted_union_primary_tuple":True,"all_other_typed_arrays_match_replay":True,
   "non_alias_arrays_unchanged_in_output":True},
  "required_followup":"Run fresh 775 all-185 source audit against the new NPZ SHA before volume metadata rebind.",
  "mass_transport":"NOT_RUN","physical_admission":False,"native_steps":0}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
 print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=="__main__":main()

