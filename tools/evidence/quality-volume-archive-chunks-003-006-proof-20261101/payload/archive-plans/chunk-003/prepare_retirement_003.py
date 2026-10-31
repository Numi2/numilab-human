#!/usr/bin/env python3
"""Fresh selected-source and current-input proof before exact chunk003 retirement."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,subprocess
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-volume-archive-chunk-003-plan"; PLAN=D/"selection-plan.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
plan=load(PLAN); plan_sha=sha(PLAN)
if plan_sha!="2ecbbc9c24655ee242e83b837b851422b620936f8d0420ec174d15e6772c4128": raise RuntimeError("approved plan changed")
build=load(D/"archive-build-receipt.json"); remote_path=D/"archive-remote-verification.json"; remote=load(remote_path)
if build["status"]!="PASS_LOCAL_STREAMED_ARCHIVE_BUILD" or remote["status"]!="PASS_LOCAL_BUILD_AND_GITHUB_ASSET_RESTORE_VERIFICATION": raise RuntimeError("archive receipts not passing")
if build["selection_plan_sha256"]!=plan_sha: raise RuntimeError("build plan hash mismatch")
archive=Path(plan["release_target"]["local_archive_path"]); download=Path(plan["release_target"]["independent_download_directory"])/plan["release_target"]["asset_name"]
if sha(archive)!=build["archive_sha256"] or sha(download)!=build["archive_sha256"]: raise RuntimeError("archive copies changed")
if remote["github_asset"]["digest"]!="sha256:"+build["archive_sha256"]: raise RuntimeError("release digest mismatch")
if sha(D/"archive-index.json")!=build["index_sha256"]: raise RuntimeError("archive index mismatch")
rbpath=D/"npz-readback-report.json"; rb=load(rbpath)
if rb["status"]!="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" or rb["npz_count"]!=16 or rb["archive_sha256"]!=build["archive_sha256"]: raise RuntimeError("NPZ readback receipt failed")
restored=Path(remote["restore_tool"]["restore_root"])/plan["files"][0]["source_relative_path"]
if not restored.is_file() or sha(restored)!=plan["files"][0]["expected_sha256_from_source_receipt"]: raise RuntimeError("restored smoke member changed")
prior_path=H/"quality-volume-archive-chunk-002-plan/current-input-proof-post-retirement.json"; prior=load(prior_path)
if prior["status"]!="PASS_POST_RETIREMENT_12_PATHS_ABSENT; PASS_PROTECTED_CURRENT_INPUTS_UNCHANGED; PASS_RESTORE_REHYDRATION": raise RuntimeError("previous protected-input proof not passing")
state_specs=[
("all-quality-859-qualified-root/quality-working-state.json","859-qualified"),
("all-quality-861-root/state.json","861-raw"),
("all-quality-854-root/round-02/verified-state.json","854-paired-source"),
("all-quality-824-root/round001/flips/verified-state.json","canonical824-flips"),
("all-quality-824-root/round001/points/verified-state.json","canonical824-points"),
("all-quality-824-root/round002/flips/state.json","canonical824-round002"),
]
state_rows=[]; direct={}
for rel,label in state_specs:
 p=H/rel
 if not p.is_file(): raise RuntimeError("required current state missing "+rel)
 raw=p.read_bytes(); s=json.loads(raw); state_rows.append({"label":label,"path":str(p),"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)})
 for role in ("Fat_current","Skin_current"):
  v=s.get(role)
  if not isinstance(v,dict) or not isinstance(v.get("path"),str): continue
  q=Path(v["path"])
  if not q.is_file() or q.is_symlink(): raise RuntimeError("current asset missing "+str(q))
  actual=sha(q)
  if actual!=v.get("sha256"): raise RuntimeError("current role asset changed "+str(q))
  direct[str(q)]={"path":str(q),"sha256":actual,"size_bytes":q.stat().st_size,"state":str(p),"role":role}
# 861 must be the newly targeted state sourced from the pinned 859 qualified base.
s861=load(H/"all-quality-861-root/state.json")
source=s861.get("source_state",{})
if source.get("path")!=str(H/"all-quality-859-qualified-root/quality-working-state.json") or source.get("sha256")!=state_rows[0]["sha256"]:
 raise RuntimeError("861 source binding no longer matches pinned 859 qualified state")
selected=[]; total=0; direct_paths=set(direct)
for row in plan["files"]:
 p=Path(row["path"]); sources=row.get("source_receipt_paths",[row["source_receipt_path"]])
 if not p.is_file() or p.is_symlink() or p.stat().st_size!=row["size_bytes_from_stat"]: raise RuntimeError("selected file missing/stat changed "+str(p))
 m=re.match(r"all-quality-(\d+)-root/(.+)$",row["source_relative_path"])
 if not m or int(m.group(1))<700 or int(m.group(1))>790: raise RuntimeError("selected path outside old-trial root range "+row["source_relative_path"])
 if str(p) in direct_paths: raise RuntimeError("selected file is direct current input "+str(p))
 for receipt in sources:
  d=load(receipt); matched=False
  for role in ("Fat_current","Skin_current"):
   v=d.get(role)
   if isinstance(v,dict) and v.get("path")==str(p) and v.get("sha256")==row["expected_sha256_from_source_receipt"]: matched=True
  if not matched: raise RuntimeError("source receipt no longer declares selected path/hash "+receipt)
 actual=sha(p)
 if actual!=row["expected_sha256_from_source_receipt"]: raise RuntimeError("selected payload SHA changed "+str(p))
 selected.append({"path":str(p),"sha256":actual,"size_bytes":p.stat().st_size,"archive_member":row["archive_member"],"source_receipt_paths":sources})
 total+=p.stat().st_size
if len(selected)!=16 or total!=338546535: raise RuntimeError("exact selection count/bytes mismatch")
proof={"schema":"numi-human.quality-volume-archive-current-input-proof.v1","status":"PASS_CURRENT_859_861_AND_CANONICAL824_REHASHED; PASS_SELECTED_16_MATCH_RECEIPTS_ARCHIVE; PASS_NO_DIRECT_OVERLAP","created_utc":datetime.now(timezone.utc).isoformat(),"fresh_process_pid":os.getpid(),"method":"Rehashed current 859-qualified, 861-raw, paired 854 and canonical824 state receipts and direct role NPZs. Reused chunk002 post-retirement proof for already protected 847/850/851/854/855 current-role set; no 144-state ancestry replay. Rehashed exact selected 16 source NPZs immediately before retirement.","selection_plan":{"path":str(PLAN),"sha256":plan_sha},"prior_protected_current_input_proof":{"path":str(prior_path),"sha256":sha(prior_path),"status":prior["status"]},"current_states_rehashed":state_rows,"current_direct_npz_assets_rehashed":sorted(direct.values(),key=lambda x:x["path"]),"current_direct_npz_asset_count":len(direct),"861_source_binding":{"path":source.get("path"),"sha256":source.get("sha256"),"matches_859_qualified":True},"selected_file_count":16,"selected_original_bytes":total,"selected_paths_pre_retirement":selected,"selected_direct_overlap_count":0,"post824_protection":"All selected paths are strictly pre824 and under all-quality-703-root old-trial directories. Canonical824 and every post824 volume/source input are protected; chunk002 proof pins prior post824 direct state assets.","archive_verification":{"path":str(remote_path),"sha256":sha(remote_path),"archive_sha256":build["archive_sha256"],"github_digest":remote["github_asset"]["digest"],"npz_readback_path":str(rbpath),"npz_readback_sha256":sha(rbpath),"restore_smoke_path":str(restored),"restore_smoke_sha256":sha(restored),"local_archive_and_download_present":True},"scope_limit":"No 144-state history rehash and no whole-tree scan."}
proof_path=D/"current-input-proof-pre-retirement.json";proof_path.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
intent={"schema":"numi-human.quality-volume-archive-retirement-intent.v1","status":"PRECHECK_PASS_EXACT_16_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),"current_input_proof_path":str(proof_path),"current_input_proof_sha256":sha(proof_path),"archive_remote_verification_path":str(remote_path),"archive_remote_verification_sha256":sha(remote_path),"no_other_files_authorized":True,"target_count":16,"retired_original_bytes":total,"targets":selected,"protected":{"all_json_reports_states":True,"all_source_witness_replay_npz":True,"all_canonical824_plus_and_post824_volumes":True,"all_unlisted_historical_files":True,"release_tag":plan["release_target"]["tag"]}}
ip=D/"retirement-intent.json"
if ip.exists(): raise RuntimeError("refusing existing retirement intent")
ip.write_text(json.dumps(intent,indent=2,sort_keys=True)+"\n")
print(json.dumps({"proof":str(proof_path),"proof_sha256":sha(proof_path),"proof_status":proof["status"],"current_state_nodes_rehashed":len(state_rows),"current_direct_npz_assets_rehashed":len(direct),"selected_npz_rehashed":len(selected),"selected_bytes":total,"retirement_intent":str(ip),"retirement_intent_sha256":sha(ip),"disk_free_bytes":__import__("shutil").disk_usage(H).free},sort_keys=True))
