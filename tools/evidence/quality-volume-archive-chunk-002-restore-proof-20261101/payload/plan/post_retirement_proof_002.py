#!/usr/bin/env python3
"""Post-retirement direct-input, archive, and restoration proof for chunk 002."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-volume-archive-chunk-002-plan"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
pre_path=D/"current-input-proof-pre-retirement.json"; pre=load(pre_path)
intent_path=D/"retirement-intent.json"; intent=load(intent_path)
ret_path=D/"retirement-receipt.json"; ret=load(ret_path)
remote_path=D/"archive-remote-verification.json"; remote=load(remote_path)
plan=load(D/"selection-plan.json"); build=load(D/"archive-build-receipt.json")
for r in pre["current_terminal"]["state_nodes_rehashed_unchanged"]:
 if sha(r["path"])!=r["sha256"]: raise RuntimeError("protected terminal state changed after retirement: "+r["path"])
for r in pre["current_terminal"]["direct_npz_inputs_rehashed_unchanged"]:
 if sha(r["path"])!=r["sha256"]: raise RuntimeError("protected direct input changed after retirement: "+r["path"])
selected=[x["path"] for x in intent["targets"]]
if any(Path(p).exists() for p in selected): raise RuntimeError("selected source still exists")
archive=Path(plan["release_target"]["local_archive_path"])
download=Path(plan["release_target"]["independent_download_directory"])/plan["release_target"]["asset_name"]
if sha(archive)!=build["archive_sha256"] or sha(download)!=build["archive_sha256"]: raise RuntimeError("verified archive copies changed before cleanup")
restored=Path(remote["restore_tool"]["restore_root"])/plan["files"][0]["source_relative_path"]
if not restored.is_file() or sha(restored)!=remote["restore_tool"]["restored_member_sha256"]: raise RuntimeError("restore smoke changed before cleanup")
if ret["status"]!="PASS_RETIRED_EXACT_12_LISTED_VOLUME_NPZ_FILES" or ret["retired_file_count"]!=12: raise RuntimeError("retirement receipt not passing")
if remote["github_asset"]["digest"]!="sha256:"+build["archive_sha256"]: raise RuntimeError("GitHub release digest mismatch")
disk=shutil.disk_usage(H)
out={"schema":"numi-human.quality-volume-archive-current-input-proof.v1","status":"PASS_POST_RETIREMENT_12_PATHS_ABSENT; PASS_PROTECTED_CURRENT_INPUTS_UNCHANGED; PASS_RESTORE_REHYDRATION","created_utc":datetime.now(timezone.utc).isoformat(),"fresh_process_pid":os.getpid(),"current_terminal":{"state_nodes_rehashed_unchanged":pre["current_terminal"]["state_nodes_rehashed_unchanged"],"direct_npz_inputs_rehashed_unchanged":pre["current_terminal"]["direct_npz_inputs_rehashed_unchanged"],"state_rehash_passed":True,"direct_asset_rehash_passed":True},"selected_paths_absent":selected,"retired_file_count":12,"retired_original_bytes":ret["retired_original_bytes"],"source_files_outside_exact_selection_modified":False,"archive_remote_verification_path":str(remote_path),"archive_remote_verification_sha256":sha(remote_path),"independent_restored_member":{"path":str(restored),"sha256":sha(restored),"size_bytes":restored.stat().st_size},"local_archive_and_download_still_present_for_postproof_cleanup":True,"github_release_asset":{"url":remote["github_asset"]["url"],"digest":remote["github_asset"]["digest"],"size_bytes":remote["github_asset"]["size_bytes"]},"disk_free_bytes_before_local_copy_cleanup":disk.free,"scope_limit":"No 144-state historical ancestry rehash and no whole-tree scan; source deletion was restricted to the exact 12 listed volume NPZ paths."}
outpath=D/"current-input-proof-post-retirement.json"; outpath.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
ret["post_retirement_proof_path"]=str(outpath); ret["post_retirement_proof_sha256"]=sha(outpath)
retpath=D/"retirement-receipt.json"; retpath.write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":out["status"],"proof_path":str(outpath),"proof_sha256":sha(outpath),"retirement_receipt_path":str(retpath),"retirement_receipt_sha256":sha(retpath),"selected_absent":len(selected),"protected_npz_rehashed":len(out["current_terminal"]["direct_npz_inputs_rehashed_unchanged"]),"disk_free_bytes":disk.free},sort_keys=True))
