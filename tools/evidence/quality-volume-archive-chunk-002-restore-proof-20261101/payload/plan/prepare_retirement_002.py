#!/usr/bin/env python3
"""Fresh terminal-input proof and exact-12 retirement intent for archive chunk 002."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, re, subprocess

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-volume-archive-chunk-002-plan"
PLAN=D/"selection-plan.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def read_json(p): return json.loads(Path(p).read_text())
plan=read_json(PLAN); plan_sha=sha(PLAN)
if plan["preparation_status"]!="READY_FOR_REVIEW_LIVE_PAYLOAD_HASH_PENDING": raise RuntimeError("unexpected plan status")
build=read_json(D/"archive-build-receipt.json"); remote_path=D/"archive-remote-verification.json"; remote=read_json(remote_path)
if build["status"]!="PASS_LOCAL_STREAMED_ARCHIVE_BUILD" or remote["status"]!="PASS_LOCAL_BUILD_AND_GITHUB_ASSET_RESTORE_VERIFICATION": raise RuntimeError("archive verification receipts are not passing")
if build["selection_plan_sha256"]!=plan_sha: raise RuntimeError("selection plan no longer matches archive build")
archive=Path(plan["release_target"]["local_archive_path"])
download=Path(plan["release_target"]["independent_download_directory"])/plan["release_target"]["asset_name"]
if not archive.is_file() or not download.is_file(): raise RuntimeError("local or independent archive copy is missing")
if sha(archive)!=build["archive_sha256"] or sha(download)!=build["archive_sha256"]: raise RuntimeError("archive copies no longer match the build receipt")
if remote["github_asset"]["digest"]!="sha256:"+build["archive_sha256"]: raise RuntimeError("GitHub digest mismatch")
if sha(D/"archive-index.json")!=build["index_sha256"]: raise RuntimeError("archive index hash mismatch")
readback=read_json(D/"npz-readback-report.json")
if readback["status"]!="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" or readback["npz_count"]!=12 or readback["archive_sha256"]!=build["archive_sha256"]: raise RuntimeError("NPZ readback receipt mismatch")
restore_root=Path(remote["restore_tool"]["restore_root"])
restored=restore_root/plan["files"][0]["source_relative_path"]
if not restored.is_file() or sha(restored)!=plan["files"][0]["expected_sha256_from_source_receipt"]: raise RuntimeError("restore smoke no longer matches source pin")
state_specs=[
("all-quality-824-root/round001/flips/verified-state.json","canonical824-flips","roles"),
("all-quality-824-root/round001/points/verified-state.json","canonical824-points","roles"),
("all-quality-824-root/round002/flips/state.json","canonical824-round002-flips","roles"),
("all-quality-847-root/round-01/v24d-preflight-only/verified-state.json","847-current","roles"),
("all-quality-850-root/round-02/verified-state.json","850-current","roles"),
("all-quality-851-qualified-root/qualified/quality-working-state.json","851-qualified","roles"),
("all-quality-851-root/proposal/state.json","851-proposal","none"),
("all-quality-854-root/round-01/proposal/state.json","854-round01","roles"),
("all-quality-854-root/round-02/verified-state.json","854-round02","roles"),
("all-quality-855-root/proposal/state.json","855-preparation","inputs"),
("all-quality-855-root/state-wrapper.json","855-wrapper","none"),
]
state_rows=[]; direct_records={}
def add_asset(node,state_path,role):
 if not isinstance(node,dict): return
 path=node.get("path"); digest=node.get("sha256")
 if not isinstance(path,str) or not path.endswith(".npz"): return
 p=Path(path)
 if not p.is_absolute(): p=H/p
 if not p.is_file() or p.is_symlink(): raise RuntimeError("current input missing/nonregular: "+str(p))
 if not isinstance(digest,str) or len(digest)!=64: raise RuntimeError("current input missing declared sha256: "+str(p))
 actual=sha(p)
 if actual!=digest: raise RuntimeError("current input digest changed: "+str(p))
 r=direct_records.setdefault(str(p),{"path":str(p),"sha256":actual,"size_bytes":p.stat().st_size,"state_receipts":[],"roles":[]})
 if r["sha256"]!=actual: raise RuntimeError("conflicting current-input hash for "+str(p))
 if state_path not in r["state_receipts"]: r["state_receipts"].append(state_path)
 if role not in r["roles"]: r["roles"].append(role)
def walk_direct_inputs(node,state_path,prefix="855.source_inputs"):
 if isinstance(node,dict):
  if isinstance(node.get("path"),str) and node["path"].endswith(".npz"):
   add_asset(node,state_path,prefix); return
  for k,v in node.items(): walk_direct_inputs(v,state_path,prefix+"."+str(k))
 elif isinstance(node,list):
  for i,v in enumerate(node): walk_direct_inputs(v,state_path,prefix+"."+str(i))
for rel,label,kind in state_specs:
 p=H/rel
 if not p.is_file(): raise RuntimeError("required state receipt missing: "+str(p))
 raw=p.read_bytes(); state=json.loads(raw)
 state_rows.append({"label":label,"path":str(p),"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)})
 if kind=="roles":
  for role in ("Fat_current","Skin_current"): add_asset(state.get(role),str(p),label+"."+role)
 elif kind=="inputs":
  walk_direct_inputs(state.get("source_inputs",{}),str(p))
# Revalidate exact selected paths against their own receipt, size, and fresh payload hash.
selected=[]; total=0; direct_paths=set(direct_records)
for row in plan["files"]:
 p=Path(row["path"]); receipt_path=Path(row["source_receipt_path"]); receipt=json.loads(receipt_path.read_text())
 found=[]
 def visit(node):
  if isinstance(node,dict):
   if node.get("path")==str(p) and node.get("sha256")==row["expected_sha256_from_source_receipt"]: found.append(True)
   for v in node.values(): visit(v)
  elif isinstance(node,list):
   for v in node: visit(v)
 visit(receipt)
 if not found: raise RuntimeError("source receipt no longer pins selected path: "+str(p))
 if not p.is_file() or p.is_symlink() or p.stat().st_size!=row["size_bytes_from_stat"]: raise RuntimeError("selected source stat changed: "+str(p))
 actual=sha(p)
 if actual!=row["expected_sha256_from_source_receipt"]: raise RuntimeError("selected source payload changed: "+str(p))
 m=re.search(r"all-quality-(\d+)-root",str(p))
 if not m or int(m.group(1))>=824: raise RuntimeError("selected path is not strictly pre824: "+str(p))
 if str(p) in direct_paths: raise RuntimeError("selected file is a direct current input: "+str(p))
 selected.append({"path":str(p),"sha256":actual,"size_bytes":p.stat().st_size,"archive_member":row["archive_member"],"source_receipt_path":str(receipt_path)})
 total+=p.stat().st_size
if len(selected)!=12 or total!=282529137: raise RuntimeError("exact selection count/bytes mismatch")
fresh={"schema":"numi-human.quality-volume-archive-current-input-proof.v1","status":"PASS_DIRECT_CURRENT_INPUTS_DISJOINT; PASS_SELECTED_PAYLOADS_MATCH_ARCHIVE; PASS_TERMINAL_DIRECT_ASSETS_REHASHED","created_utc":datetime.now(timezone.utc).isoformat(),"fresh_process_pid":os.getpid(),"method":"Fresh process rehashed explicit canonical/current state nodes, their direct Fat_current/Skin_current role arrays, 855 proposal source_inputs only, and the exact selected 12 source files. No historical 144-node DAG replay.","selection_plan":{"path":str(PLAN),"sha256":plan_sha},"current_terminal":{"state_nodes_rehashed_unchanged":state_rows,"direct_npz_inputs_rehashed_unchanged":sorted(direct_records.values(),key=lambda x:x["path"]),"direct_path_count":len(direct_records),"selected_path_overlap_count":0,"current_latest_state":"855 is preparation-only; it directly binds current volume assets from 854 round02. No 855 result volume was selected."},"selected_file_count":len(selected),"selected_original_bytes":total,"selected_payload_hashing_by_this_proof":"All 12 were rehashed in this fresh process immediately before retirement and matched their source receipt/archive blob SHA.","selected_paths_pre_retirement":selected,"protected_824_plus_rule":"All selected paths match 794/799/807 roots (<824); canonical824 and every post824 input remain protected.","archive_verification":{"path":str(remote_path),"sha256":sha(remote_path),"archive_sha256":build["archive_sha256"],"github_asset_digest":remote["github_asset"]["digest"],"npz_readback_path":str(D/"npz-readback-report.json"),"npz_readback_sha256":sha(D/"npz-readback-report.json"),"restore_smoke_path":str(restored),"restore_smoke_sha256":sha(restored),"local_archive_and_download_present":True},"scope_limit":"Does not rehash the historical 144-node ancestry set; chunk001 proof remains preserved and unchanged. Does not scan the whole source tree."}
proof_path=D/"current-input-proof-pre-retirement.json"
proof_path.write_text(json.dumps(fresh,indent=2,sort_keys=True)+"\n")
targets=list(selected)
intent={"schema":"numi-human.quality-volume-archive-retirement-intent.v1","status":"PRECHECK_PASS_EXACT_12_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),"current_input_proof_path":str(proof_path),"current_input_proof_sha256":sha(proof_path),"archive_remote_verification_path":str(remote_path),"archive_remote_verification_sha256":sha(remote_path),"no_other_files_authorized":True,"target_count":12,"retired_original_bytes":total,"targets":targets,"protected":{"all_json_reports_states":True,"all_source_and_witness_npz":True,"all_canonical824_plus_and_post824_volumes":True,"all_unlisted_historical_files":True,"archive_release":"human-quality-volume-archive-20261031-chunk-002"}}
intent_path=D/"retirement-intent.json"
if intent_path.exists(): raise RuntimeError("refusing existing retirement intent")
intent_path.write_text(json.dumps(intent,indent=2,sort_keys=True)+"\n")
print(json.dumps({"proof_path":str(proof_path),"proof_sha256":sha(proof_path),"proof_status":fresh["status"],"direct_npz_inputs_rehashed":len(direct_records),"states_rehashed":len(state_rows),"selected_rehashed":len(selected),"selected_bytes":total,"intent_path":str(intent_path),"intent_sha256":sha(intent_path),"remote_verification_sha256":sha(remote_path)},sort_keys=True))
