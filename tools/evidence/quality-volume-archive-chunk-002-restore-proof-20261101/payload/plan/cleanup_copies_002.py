#!/usr/bin/env python3
"""Remove only the verified local archive, independent download, and one restore smoke."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
D=H/"quality-volume-archive-chunk-002-plan"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
post_path=D/"current-input-proof-post-retirement.json"; post=load(post_path)
ret_path=D/"retirement-receipt.json"; ret=load(ret_path)
remote_path=D/"archive-remote-verification.json"; remote=load(remote_path)
plan=load(D/"selection-plan.json"); build=load(D/"archive-build-receipt.json")
if post["status"]!="PASS_POST_RETIREMENT_12_PATHS_ABSENT; PASS_PROTECTED_CURRENT_INPUTS_UNCHANGED; PASS_RESTORE_REHYDRATION": raise RuntimeError("post-retirement proof not passing")
if sha(post_path)!=ret["post_retirement_proof_sha256"]: raise RuntimeError("post-retirement proof hash mismatch")
if ret["status"]!="PASS_RETIRED_EXACT_12_LISTED_VOLUME_NPZ_FILES": raise RuntimeError("selected-source retirement not passing")
if remote["github_asset"]["digest"]!="sha256:"+build["archive_sha256"]: raise RuntimeError("GitHub release is not archive-pinned")
archive=Path(plan["release_target"]["local_archive_path"])
download_dir=Path(plan["release_target"]["independent_download_directory"])
download=download_dir/plan["release_target"]["asset_name"]
restore_root=Path(remote["restore_tool"]["restore_root"])
restored=restore_root/plan["files"][0]["source_relative_path"]
expected=[(archive,build["archive_sha256"]),(download,build["archive_sha256"]),(restored,plan["files"][0]["expected_sha256_from_source_receipt"])]
for p,digest in expected:
 if not p.is_file() or p.is_symlink() or sha(p)!=digest: raise RuntimeError("cleanup target is not the verified copy: "+str(p))
# Fail closed if the two temporary roots contain any other file or symlink.
download_files=[p for p in download_dir.rglob("*") if p.is_file() or p.is_symlink()]
if download_files!=[download]: raise RuntimeError("download directory contains unexpected entries")
restore_files=[p for p in restore_root.rglob("*") if p.is_file() or p.is_symlink()]
if restore_files!=[restored]: raise RuntimeError("restore smoke contains unexpected entries")
tmp=D/".chunk002-npz-readback.tmp"
if tmp.exists(): raise RuntimeError("unexpected NPZ readback temporary remains")
intent={"schema":"numi-human.quality-volume-archive-local-copy-cleanup-intent.v1","status":"PRECHECK_PASS_EXACT_VERIFIED_COPIES_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),"post_retirement_proof_path":str(post_path),"post_retirement_proof_sha256":sha(post_path),"retirement_receipt_path":str(ret_path),"retirement_receipt_sha256":sha(ret_path),"remote_verification_path":str(remote_path),"remote_verification_sha256":sha(remote_path),"no_other_files_authorized":True,"targets":[{"path":str(p),"sha256":d,"size_bytes":p.stat().st_size} for p,d in expected],"directory_cleanup":[str(download_dir),str(restore_root)]}
intent_path=D/"local-copy-cleanup-intent.json"
if intent_path.exists(): raise RuntimeError("refusing existing cleanup intent")
intent_path.write_text(json.dumps(intent,indent=2,sort_keys=True)+"\n")
before=shutil.disk_usage(H)
progress={"schema":"numi-human.quality-volume-archive-local-copy-cleanup-progress.v1","status":"CLEANUP_STARTED_EXACT_VERIFIED_COPIES_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),"intent_path":str(intent_path),"intent_sha256":sha(intent_path),"removed":[]}
progress_path=D/"local-copy-cleanup-progress.json"; progress_path.write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n")
for p,digest in expected:
 if sha(p)!=digest: raise RuntimeError("cleanup target changed at unlink boundary: "+str(p))
 size=p.stat().st_size; p.unlink(); progress["removed"].append({"path":str(p),"sha256":digest,"size_bytes":size})
 progress["updated_utc"]=datetime.now(timezone.utc).isoformat(); progress_path.write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n")
# Remove only now-empty directories created for this independent download and restore smoke.
for p in sorted([x for x in restore_root.rglob("*") if x.is_dir()],key=lambda x:len(x.parts),reverse=True): p.rmdir()
restore_root.rmdir(); download_dir.rmdir()
if any(p.exists() for p,_ in expected) or restore_root.exists() or download_dir.exists(): raise RuntimeError("verified local copies/directories remain")
after=shutil.disk_usage(H)
# Update the source-retirement receipt only after the three verified local copies are gone.
ret=load(ret_path); ret["local_archive_and_download_removed"]=True
ret["local_copy_cleanup_intent_path"]=str(intent_path); ret["local_copy_cleanup_intent_sha256"]=sha(intent_path)
ret["local_archive_and_download_removed_utc"]=datetime.now(timezone.utc).isoformat()
ret_path.write_text(json.dumps(ret,indent=2,sort_keys=True)+"\n")
receipt={"schema":"numi-human.quality-volume-archive-local-copy-cleanup-receipt.v1","status":"PASS_REMOVED_LOCAL_ARCHIVE_DOWNLOAD_AND_RESTORE_SMOKE","created_utc":datetime.now(timezone.utc).isoformat(),"intent_path":str(intent_path),"intent_sha256":sha(intent_path),"retained_github_release":plan["release_target"]["tag"],"github_asset_digest":remote["github_asset"]["digest"],"removed":intent["targets"],"removed_file_count":len(expected),"removed_bytes":sum(x["size_bytes"] for x in intent["targets"]),"disk_free_bytes_before":before.free,"disk_free_bytes_after":after.free,"observed_disk_free_delta_bytes":after.free-before.free,"directory_cleanup":intent["directory_cleanup"],"source_npz_retirement_receipt_path":str(ret_path),"source_npz_retirement_receipt_sha256":sha(ret_path),"post_retirement_proof_path":str(post_path)}
rp=D/"local-copy-cleanup-receipt.json";rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
progress["status"]="PASS_REMOVED_EXACT_VERIFIED_COPIES";progress["cleanup_receipt_path"]=str(rp);progress["cleanup_receipt_sha256"]=sha(rp);progress["updated_utc"]=datetime.now(timezone.utc).isoformat();progress_path.write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n")
lane={"schema":"numi-human.quality-volume-archive-lane-release-check.v1","status":"PASS_LANE_RELEASED","created_utc":datetime.now(timezone.utc).isoformat(),"fresh_process_pid":os.getpid(),"archive_tag":plan["release_target"]["tag"],"github_archive_retained":True,"archive_sha256":build["archive_sha256"],"selected_source_files_retired":12,"selected_source_bytes_retired":ret["retired_original_bytes"],"post_retirement_proof_path":str(post_path),"post_retirement_proof_sha256":sha(post_path),"retirement_receipt_path":str(ret_path),"retirement_receipt_sha256":sha(ret_path),"local_copy_cleanup_receipt_path":str(rp),"local_copy_cleanup_receipt_sha256":sha(rp),"local_archive_download_restore_bytes_removed":receipt["removed_bytes"],"local_archive_download_restore_files_removed":receipt["removed_file_count"],"disk_free_bytes_before_local_copy_cleanup":before.free,"disk_free_bytes_after_local_copy_cleanup":after.free,"observed_disk_free_delta_bytes":after.free-before.free,"current_inputs_unchanged":True,"source_retirement_performed":True}
lp=D/"lane-release-check.json";lp.write_text(json.dumps(lane,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":lane["status"],"retired_source_bytes":lane["selected_source_bytes_retired"],"removed_verified_copy_bytes":lane["local_archive_download_restore_bytes_removed"],"files_removed":lane["local_archive_download_restore_files_removed"],"disk_free_before":before.free,"disk_free_after":after.free,"disk_free_delta":after.free-before.free,"lane_receipt":str(lp),"lane_receipt_sha256":sha(lp),"cleanup_receipt_sha256":sha(rp)},sort_keys=True))
