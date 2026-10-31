import hashlib,json,os,datetime,pathlib,subprocess
from typing import Any
base=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
d=base/"quality-volume-archive-chunk-009-plan"
def load(n): return json.loads((d/n).read_text())
def sha(path):
 h=hashlib.sha256(); n=0
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b); n+=len(b)
 return h.hexdigest(),n
plan=load("selection-plan.json"); direct=load("current-direct-path-proof-009.json")
build=load("archive-build-receipt.json"); readback=load("npz-readback-report.json")
release=load("github-release-metadata.json"); index=load("archive-index.json")
assert plan["chunk"]["id"]=="quality-volume-chunk-009"
assert len(plan["files"])==14 and plan["chunk"]["original_bytes"]==315679208
assert direct["status"].startswith("PASS_CURRENT_STATES_AND_DIRECT_ROLE_ASSETS_REHASHED")
assert direct["861_source_binding_matches_859"] is True and direct["selected_direct_overlap_count"]==0
archive=pathlib.Path(plan["release_target"]["local_archive_path"])
download=pathlib.Path(plan["release_target"]["independent_download_directory"])/plan["release_target"]["asset_name"]
assert build["status"]=="PASS_LOCAL_STREAMED_ARCHIVE_BUILD"
assert build["archive_sha256"]==readback["archive_sha256"]==release["assets"][0]["digest"].removeprefix("sha256:")
assert build["archive_size_bytes"]==release["assets"][0]["size"]
assert len(release["assets"])==1 and release["assets"][0]["name"]==plan["release_target"]["asset_name"]
assert release["assets"][0]["state"]=="uploaded" and not release["isDraft"] and release["tagName"]==plan["release_target"]["tag"]
for p in (archive,download):
 got,n=sha(p); assert got==build["archive_sha256"] and n==build["archive_size_bytes"],(p,got,n)
assert readback["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and readback["npz_count"]==14 and readback["npy_array_headers_read"]==1145
assert index["unique_blob_count"]==14 and index["unique_blob_bytes"]==315679208
restore_root=pathlib.Path("/Users/numi/Work/.quality-volume-archive-chunk-009-restore-smoke")
restore_path=pathlib.Path(plan["files"][0]["path"])
restore_file=restore_root/plan["files"][0]["source_relative_path"]
assert restore_file.exists()
restore_sha,restore_size=sha(restore_file)
assert restore_sha==plan["files"][0]["expected_sha256_from_source_receipt"] and restore_size==plan["files"][0]["size_bytes_from_stat"]
archive_index_by_path={r["path"]:r for r in index["files"]}
selected=[]; receipt_records={}
for r in plan["files"]:
 p=pathlib.Path(r["path"]); got,n=sha(p)
 assert got==r["expected_sha256_from_source_receipt"] and n==r["size_bytes_from_stat"],r["path"]
 assert str(p) in archive_index_by_path and archive_index_by_path[str(p)]["sha256"]==got
 for receipt_path in r.get("source_receipt_paths",[r["source_receipt_path"]]):
  rp=pathlib.Path(receipt_path); rh,rn=sha(rp); obj=json.loads(rp.read_text())
  role=obj.get(r["role"])
  assert role and role.get("path")==str(p) and role.get("sha256")==got,(str(rp),r["role"],str(p))
  receipt_records[str(rp)]={"sha256":rh,"size_bytes":rn}
 selected.append({"path":str(p),"role":r["role"],"root":r["root"],"sha256":got,"size_bytes":n,"source_relative_path":r["source_relative_path"],"source_receipts":r.get("source_receipt_paths",[r["source_receipt_path"]])})
assert len(selected)==14 and len({x["path"] for x in selected})==14
assert all(int(x["root"])<824 and "/all-quality-"+x["root"]+"-root/" in x["path"] for x in selected)
assert not any(x["current_direct_input_overlap"] for x in plan["files"])
states=[]
for r in direct["current_states_rehashed"]:
 got,n=sha(r["path"]); assert got==r["sha256"] and n==r["size_bytes"],r["path"]
 states.append({"label":r["label"],"path":r["path"],"sha256":got,"size_bytes":n})
direct_assets=[]
for r in direct["current_direct_npz_assets_rehashed"]:
 got,n=sha(r["path"]); assert got==r["sha256"] and n==r["size_bytes"],r["path"]
 direct_assets.append({"path":r["path"],"role":r["role"],"state":r["state"],"sha256":got,"size_bytes":n})
assert len(states)==6 and len(direct_assets)==12
prior={"path":str(d/"quality-volume-archive-chunk-008-plan/lane-release-check.json"),"sha256":hashlib.sha256((base/"quality-volume-archive-chunk-008-plan/lane-release-check.json").read_bytes()).hexdigest(),"status":json.loads((base/"quality-volume-archive-chunk-008-plan/lane-release-check.json").read_text())["status"]}
assert prior["status"]=="PASS_LANE_RELEASED"
remote_verify={"release_tag":release["tagName"],"release_url":release["url"],"asset_id":release["assets"][0]["id"],"asset_name":release["assets"][0]["name"],"asset_digest":release["assets"][0]["digest"],"asset_size_bytes":release["assets"][0]["size"],"asset_state":release["assets"][0]["state"],"independent_download_path":str(download),"independent_archive_sha256":readback["archive_sha256"],"archive_index_sha256":hashlib.sha256(json.dumps(index,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"unique_blob_count":index["unique_blob_count"],"uncompressed_blob_bytes":index["unique_blob_bytes"],"npz_crc_and_npy_headers":"PASS","npy_array_headers_read":readback["npy_array_headers_read"],"one_npz_restore_path":str(restore_path),"one_npz_restore_sha256":restore_sha,"one_npz_restore_size_bytes":restore_size}
remote_path=d/"archive-remote-verification.json"; remote_path.write_text(json.dumps(remote_verify,indent=2,sort_keys=True)+"\n")
proof={"schema":"numi-human.quality-volume-archive-current-input-proof.v1","status":"PASS_CURRENT_859_861_854_AND_CANONICAL824_REHASHED; PASS_DIRECT_ASSETS_UNCHANGED; PASS_SELECTED_14_MATCH_RECEIPTS_ARCHIVE; PASS_NO_DIRECT_OVERLAP","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"fresh_process_pid":os.getpid(),"roster":{"path":plan["source_inventory"]["followup_roster_plan_path"],"sha256":plan["source_inventory"]["followup_roster_plan_sha256"],"slice":plan["source_inventory"]["roster_chunk_id"]},"selection_plan":{"path":str(d/"selection-plan.json"),"sha256":sha(d/"selection-plan.json")[0]},"archive_verification":{"path":str(remote_path),"sha256":sha(remote_path)[0],**remote_verify},"current_states_rehashed":states,"current_direct_npz_assets_rehashed":direct_assets,"current_direct_npz_asset_count":len(direct_assets),"861_source_binding_matches_859":True,"selected_paths_pre_retirement":selected,"selected_file_count":len(selected),"selected_original_bytes":sum(x["size_bytes"] for x in selected),"selected_direct_overlap_count":0,"source_receipt_files_rehashed":receipt_records,"prior_chunk007_lane_release":prior,"scope_limit":"No broad tree scan or historical-state replay; exact selected old pre-824 full-volume NPZ files only. Current state and direct assets were freshly rehashed.","physical_admission":False,"native_steps":0}
out=d/"current-input-proof-pre-retirement.json"
if out.exists(): raise RuntimeError("refusing existing proof; preserve and use successor")
out.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":proof["status"],"selected_files":len(selected),"selected_bytes":proof["selected_original_bytes"],"receipt_files":len(receipt_records),"states":len(states),"direct_assets":len(direct_assets),"fresh_process_pid":os.getpid(),"proof":str(out),"proof_sha256":sha(out)[0],"archive_verification":str(remote_path),"archive_verification_sha256":sha(remote_path)[0]},sort_keys=True))
