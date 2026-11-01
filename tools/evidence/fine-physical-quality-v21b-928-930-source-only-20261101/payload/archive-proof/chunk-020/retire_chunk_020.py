from pathlib import Path
import datetime, hashlib, json, subprocess, zipfile
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-020-plan"
PLAN=OUT/"selection-plan.json"
PROOF=H/"quality-volume-failed-proposals-current-path-proof-020.json"
RESTORE=OUT/"one-npz-restore-receipt.json"
READBACK=OUT/"npz-readback-report.json"
GHMETA=OUT/"github-release-metadata.json"
STATE928=H/"all-quality-928-b5-backtracking-root/qualified/verified-state.json"
TARGETS={
 str(H/"all-quality-858-root/fat-coupled-quality-point-proposal.npz"):"178df55695b71746125a0190bbe8f55c8c49e649f8218a9aedaebddc96c072f1",
 str(H/"all-quality-858-root/skin-coupled-quality-point-proposal.npz"):"1fdbaf0bbef7efca6947279d1b71eef412ff2cd09ebf84c186b25b88369a4340",
}
def sha(b): return hashlib.sha256(b).hexdigest()
def pin(p):
 b=p.read_bytes(); st=p.stat()
 return {"path":str(p),"sha256":sha(b),"size_bytes":len(b),"mtime_ns":st.st_mtime_ns,"link_count":st.st_nlink,"is_symlink":p.is_symlink()}
plan=json.loads(PLAN.read_text()); proof=json.loads(PROOF.read_text()); rr=json.loads(RESTORE.read_text()); rb=json.loads(READBACK.read_text()); gh=json.loads(GHMETA.read_text()); s928=json.loads(STATE928.read_text())
assert sha(PROOF.read_bytes())=="92ff17843ce1172e9ea54028b2d841c4e0c0f69078604efba4635f1014764b53"
assert proof["status"]=="PASS_METADATA_SCAN_COMPLETE_HASH_ALIASES_RESOLVED"
assert proof["summary"]["exact_target_path_hits"]==0 and proof["summary"]["scan_error_count"]==0
assert proof["summary"]["all_hash_hits_resolve_to_live_exact_sha_alias"] is True
assert proof["summary"]["target_sha_hits"]==4
assert rr["status"]=="PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK" and not rr["filesystem_restore_performed"]
assert rb["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and rb["all_zip_crcs_passed"]
assert rr["archive_sha256"]=="ed0ef16587af1bbd58ca4b01ab951b87e6d978da3193f5a816dc1dca95bee027"
asset=next(a for a in gh["assets"] if a["name"]=="quality-volume-archive-chunk-020.tar.zst")
assert asset["id"]=="RA_kwDOUElluM4pjLd6" and asset["digest"]=="sha256:"+rr["archive_sha256"]
assert s928["status"]=="SOURCE_ONLY_BATCH_PROPOSAL_PENDING_CURRENT_ALL185_AND_NATIVE_GATES"
needles=[b"all-quality-858-root/fat-coupled-quality-point-proposal.npz",b"all-quality-858-root/skin-coupled-quality-point-proposal.npz",b"178df55695b71746125a0190bbe8f55c8c49e649f8218a9aedaebddc96c072f1",b"1fdbaf0bbef7efca6947279d1b71eef412ff2cd09ebf84c186b25b88369a4340"]
state928_pin=pin(STATE928); assert all(n not in STATE928.read_bytes() for n in needles)
npz_rows=[]
for p in sorted((H/"all-quality-928-b5-backtracking-root").rglob("*.npz")):
    members=[]
    with zipfile.ZipFile(p) as z:
        for name in z.namelist():
            low=Path(name).name.lower()
            if any(k in low for k in ("path","sha256","parent")):
                data=z.read(name); members.append({"member":name,"member_sha256":sha(data)})
                assert all(n not in data for n in needles), f"858 target ref in {p}:{name}"
    npz_rows.append({**pin(p),"metadata_members":members})
assert len(npz_rows)==6
ctx=proof["current_context_pins"]
qpath=Path(next(k for k in ctx if "all-quality-924-joint-role-equal-root/qualified/verified-state.json" in k))
qpin=pin(qpath); assert qpin["sha256"]==ctx[str(qpath)]["sha256"]
plan_rows={r["path"]:r["expected_sha256_from_source_receipt"] for r in plan["files"]}
assert plan_rows==TARGETS
inventory={r["path"]:r for r in proof["target_inventory"]}
before=[]
for path,digest in TARGETS.items():
 p=Path(path); assert p.is_file() and not p.is_symlink()
 row=pin(p); assert row["sha256"]==digest and row["link_count"]==1
 assert inventory[path]["exists"] is True and inventory[path]["sha256"]==digest
 before.append(row)
ps=subprocess.run(["/bin/ps","-Ao","pid,command"],check=True,stdout=subprocess.PIPE,text=True).stdout
active=[line for line in ps.splitlines() if any(k in line for k in ("fat-coupled-quality-point-proposal.npz","skin-coupled-quality-point-proposal.npz")) and "retire_chunk_020.py" not in line]
assert not active, repr(active[:5])
pre={"schema":"numi-human.failed-volume-proposal-retirement-preflight.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS","chunk":"020","ancestry_proof":{"path":str(PROOF),"sha256":sha(PROOF.read_bytes()),"exact_target_path_hits":0,"target_sha_hits":4,"all_hash_hits_resolve_to_live_exact_sha_alias":True},"archive_verification":{"asset_id":asset["id"],"tag":gh["tagName"],"archive_sha256":rr["archive_sha256"],"restore_status":rr["status"],"restore_method":rr["restore_method"],"readback_status":rb["status"]},"qualified_context_924":qpin,"newer_source_only_context_928":{"pin":state928_pin,"status":s928["status"],"target_string_hits":0,"scanned_npz_count":len(npz_rows),"npz_metadata_scans":npz_rows},"target_files_before":before,"active_process_matches":active,"retirement_performed":False,"physical_admission":False,"native_steps":0}
(OUT/"retirement-preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
removed=[]
for path,digest in TARGETS.items():
 p=Path(path); assert sha(p.read_bytes())==digest
 p.unlink(); removed.append({"path":path,"sha256":digest,"bytes_retired":before[len(removed)]["size_bytes"]})
receipt={"schema":"numi-human.failed-volume-proposal-retirement-receipt.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE","chunk":"020","preflight_sha256":sha((OUT/"retirement-preflight.json").read_bytes()),"archive_sha256":rr["archive_sha256"],"archive_index_sha256":sha((OUT/"archive-index.json").read_bytes()),"github_metadata_sha256":sha(GHMETA.read_bytes()),"files_removed":removed,"total_bytes_retired":sum(r["bytes_retired"] for r in removed),"sources_reports_and_witnesses_preserved":True,"physical_admission":False,"native_steps":0}
(OUT/"retirement-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":receipt["status"],"files_removed":removed,"total_bytes_retired":receipt["total_bytes_retired"],"preflight":str(OUT/"retirement-preflight.json"),"receipt":str(OUT/"retirement-receipt.json")},sort_keys=True))
