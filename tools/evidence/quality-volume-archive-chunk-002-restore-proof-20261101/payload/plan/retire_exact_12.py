#!/usr/bin/env python3
"""Retire exactly the 12 archive-verified chunk-002 historical volume NPZs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-volume-archive-chunk-002-plan"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
intent_path=D/"retirement-intent.json"; intent=load(intent_path); proof_path=Path(intent["current_input_proof_path"]); proof=load(proof_path); remote_path=Path(intent["archive_remote_verification_path"]); remote=load(remote_path)
if intent["status"]!="PRECHECK_PASS_EXACT_12_ONLY" or proof["status"]!="PASS_DIRECT_CURRENT_INPUTS_DISJOINT; PASS_SELECTED_PAYLOADS_MATCH_ARCHIVE; PASS_TERMINAL_DIRECT_ASSETS_REHASHED": raise RuntimeError("precheck receipt not passing")
if sha(proof_path)!=intent["current_input_proof_sha256"] or sha(remote_path)!=intent["archive_remote_verification_sha256"]: raise RuntimeError("precheck/remote receipt changed")
plan=load(D/"selection-plan.json"); build=load(D/"archive-build-receipt.json")
if sha(D/"selection-plan.json")!=build["selection_plan_sha256"]: raise RuntimeError("selection plan changed")
archive=Path(plan["release_target"]["local_archive_path"]); download=Path(plan["release_target"]["independent_download_directory"])/plan["release_target"]["asset_name"]
if sha(archive)!=build["archive_sha256"] or sha(download)!=build["archive_sha256"]: raise RuntimeError("verified local archive copy changed")
# Fresh recheck of the direct terminal JSON nodes and the precise direct NPZ inputs.
for row in proof["current_terminal"]["state_nodes_rehashed_unchanged"]:
 if sha(row["path"])!=row["sha256"]: raise RuntimeError("terminal state changed after proof: "+row["path"])
for row in proof["current_terminal"]["direct_npz_inputs_rehashed_unchanged"]:
 if sha(row["path"])!=row["sha256"]: raise RuntimeError("terminal direct input changed after proof: "+row["path"])
# Fresh source-content recheck immediately before exact retirement.
targets=intent["targets"]
if len(targets)!=12 or len({x["path"] for x in targets})!=12: raise RuntimeError("intent is not exactly 12 unique targets")
if {x["path"] for x in targets}!={x["path"] for x in proof["selected_paths_pre_retirement"]}: raise RuntimeError("intent and pre-retirement proof selections differ")
if sum(x["size_bytes"] for x in targets)!=282529137: raise RuntimeError("retirement byte total differs")
for x in targets:
 p=Path(x["path"])
 if not p.is_file() or p.is_symlink() or p.stat().st_size!=x["size_bytes"] or sha(p)!=x["sha256"]: raise RuntimeError("selected source changed immediately before retirement: "+str(p))
start={"schema":"numi-human.quality-volume-archive-retirement-progress.v1","status":"RETIREMENT_STARTED_EXACT_12_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"intent_path":str(intent_path),"intent_sha256":sha(intent_path),"verified_archive_sha256":remote["github_asset"]["digest"],"targets_expected":targets,"retired_so_far":[]}
progress_path=D/"retirement-progress.json"; progress_path.write_text(json.dumps(start,indent=2,sort_keys=True)+"\n")
for x in targets:
 p=Path(x["path"])
 if sha(p)!=x["sha256"]: raise RuntimeError("source changed at unlink boundary: "+str(p))
 p.unlink()
 start["retired_so_far"].append({"path":x["path"],"sha256":x["sha256"],"size_bytes":x["size_bytes"]})
 start["updated_utc"]=datetime.now(timezone.utc).isoformat()
 progress_path.write_text(json.dumps(start,indent=2,sort_keys=True)+"\n")
missing=[x["path"] for x in targets if Path(x["path"]).exists()]
if missing: raise RuntimeError("one or more exact targets remain: "+repr(missing))
receipt={"schema":"numi-human.quality-volume-archive-retirement-receipt.v1","status":"PASS_RETIRED_EXACT_12_LISTED_VOLUME_NPZ_FILES","created_utc":datetime.now(timezone.utc).isoformat(),"intent_path":str(intent_path),"intent_sha256":sha(intent_path),"current_input_proof_path":str(proof_path),"current_input_proof_sha256":sha(proof_path),"archive_remote_verification_path":str(remote_path),"archive_remote_verification_sha256":sha(remote_path),"archive_sha256":remote["github_asset"]["digest"].removeprefix("sha256:"),"retired_file_count":12,"retired_original_bytes":sum(x["size_bytes"] for x in targets),"retired":targets,"selected_paths_absent":True,"preserved_json_reports_states_sources_nonvolume_npz_and_post824_volume_inputs":True,"local_archive_and_download_removed":False}
rp=D/"retirement-receipt.json"; rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
start["status"]="PASS_RETIRED_EXACT_12_ONLY"; start["retirement_receipt_path"]=str(rp); start["retirement_receipt_sha256"]=sha(rp); start["updated_utc"]=datetime.now(timezone.utc).isoformat()
progress_path.write_text(json.dumps(start,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":receipt["status"],"retired_files":12,"retired_bytes":receipt["retired_original_bytes"],"receipt_path":str(rp),"receipt_sha256":sha(rp),"progress_path":str(progress_path),"progress_sha256":sha(progress_path)},sort_keys=True))
