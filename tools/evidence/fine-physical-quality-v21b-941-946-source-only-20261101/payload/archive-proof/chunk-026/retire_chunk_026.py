from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-026-plan"
PLAN=OUT/"selection-plan.json"; PREV=H/"quality-volume-failed-proposals-current-path-proof-025.json"; PROOF=H/"quality-volume-failed-proposals-current-path-proof-026.json"
PREV_PREF=H/"quality-volume-archive-chunk-025-plan/retirement-preflight.json"; PREV_REC=H/"quality-volume-archive-chunk-025-plan/retirement-receipt.json"
RESTORE=OUT/"one-npz-restore-receipt.json"; READBACK=OUT/"npz-readback-report.json"; GHMETA=OUT/"github-release-metadata.json"; STATE=H/"all-quality-875-root/round-01/proposal/state.json"
plan=json.loads(PLAN.read_text()); prev=json.loads(PREV.read_text()); proof=json.loads(PROOF.read_text()); rr=json.loads(RESTORE.read_text()); rb=json.loads(READBACK.read_text()); gh=json.loads(GHMETA.read_text()); sx=json.loads(STATE.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 b=p.read_bytes();st=p.stat();return {"path":str(p),"sha256":sha(b),"size_bytes":len(b),"mtime_ns":st.st_mtime_ns,"link_count":st.st_nlink,"is_symlink":p.is_symlink()}
TARGETS={r["path"]:r["expected_sha256_from_source_receipt"] for r in plan["files"]}
assert plan["chunk"]["id"]=="quality-volume-chunk-026" and plan["chunk"]["file_count"]==2 and plan["chunk"]["original_bytes"]==66831361
assert prev["status"]=="PASS_INHERITED_CURRENT_PASSING_ANCESTRY_AND_NEWER_ATTEMPTS_SCANNED_WITH_LIVE_ALIAS" and prev["newer_attempt_scan"]["direct_original_path_hits"]==[] and prev["newer_attempt_scan"]["unresolved_hash_hits"]==[]
assert json.loads(PREV_PREF.read_text())["status"]=="PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS"
assert json.loads(PREV_REC.read_text())["status"]=="PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE"
assert proof["status"]=="PASS_INHERITED_QUALIFIED_CONTEXT_AND_CURRENT_TREES_SCANNED_NO_TARGET_REFERENCES" and proof["target_path_hits"]==0 and proof["target_hash_hits"]==0
assert sha(PROOF.read_bytes())==plan["current_direct_path_proof"]["sha256"]
assert proof["newer_attempt_scan"]["direct_original_path_hits"]==[] and proof["newer_attempt_scan"]["unresolved_hash_hits"]==[]
assert rr["status"]=="PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK" and rr["archive_sha256"]=="0d4045617d31ea1df2b8b5918a15160cd6e2ecda37dfe20b3daf6e1757c24c5e" and not rr["filesystem_restore_performed"]
assert rb["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and rb["all_zip_crcs_passed"] is True
asset=next(a for a in gh["assets"] if a["name"]=="quality-volume-archive-chunk-026.tar.zst")
assert gh["tagName"]=="human-quality-volume-failed-unadmitted-20261101-chunk-026" and asset["id"]=="RA_kwDOUElluM4pjUbw" and asset["digest"]=="sha256:"+rr["archive_sha256"] and asset["size"]==39917957
assert sx.get("physical_admission") is False and sx.get("native_steps")==0 and sx["Fat_current"]["native_quality_failures"]>0 and sx["Skin_current"]["native_quality_failures"]>0
q931state=H/"all-quality-931-b5-backtracking-root/qualified/verified-state.json";q931=H/"all-quality-931-b5-backtracking-root/qualified/receipt.json";q931replay=H/"all-quality-931-b5-backtracking-root/qualified/independent-v26b5b-replay.json"
ctx=proof["current_qualified_source_context_931"]
assert pin(q931state)["sha256"]==ctx["verified_state"]["sha256"] and pin(q931)["sha256"]==ctx["receipt"]["sha256"] and pin(q931replay)["sha256"]==ctx["independent_replay"]["sha256"]
assert json.loads(q931state.read_text()).get("physical_admission") is False and json.loads(q931state.read_text()).get("native_steps")==0
path_needles=[p.encode() for p in TARGETS]+[Path(p).relative_to(H).as_posix().encode() for p in TARGETS]
hash_needles=[d.encode() for d in TARGETS.values()]
roots=[]
for p in H.iterdir():
 if p.is_dir() and p.name.startswith("all-quality-"):
  try:n=int(p.name.split("-",2)[2].split("-",1)[0])
  except (ValueError,IndexError):continue
  if n>924:roots.append(p)
roots.sort(key=lambda p:(int(p.name.split("-",2)[2].split("-",1)[0]),p.name))
scanned_json=[];scanned_npz=[];hits=[]
for root in roots:
 for p in sorted(root.rglob("*.json")):
  data=p.read_bytes()
  if any(x in data for x in path_needles+hash_needles):hits.append({"path":str(p),"kind":"json","matches":[x.decode() for x in path_needles+hash_needles if x in data]})
  scanned_json.append({"path":str(p),"sha256":sha(data),"size_bytes":len(data)})
 for p in sorted(root.rglob("*.npz")):
  members=[]
  with zipfile.ZipFile(p) as z:
   for name in z.namelist():
    if any(k in Path(name).name.lower() for k in ("path","sha256","parent")):
     data=z.read(name)
     if any(x in data for x in path_needles+hash_needles):hits.append({"path":str(p),"member":name,"kind":"npz-metadata","matches":[x.decode() for x in path_needles+hash_needles if x in data]})
     members.append({"member":name,"sha256":sha(data)})
  scanned_npz.append({**pin(p),"metadata_members":members})
assert not hits,hits[:10]
inv={x["path"]:x for x in proof["target_inventory"]}
before=[]
for path,digest in TARGETS.items():
 p=Path(path);assert p.is_file() and not p.is_symlink()
 row=pin(p);assert row["sha256"]==digest and row["link_count"]==1 and row["size_bytes"]==next(x["size_bytes_from_stat"] for x in plan["files"] if x["path"]==path)
 assert inv[path]["sha256"]==digest
 before.append(row)
lsof_rows=[]
for path in TARGETS:
 cp=subprocess.run(["/usr/sbin/lsof","-t",path],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if cp.returncode not in (0,1):raise RuntimeError("lsof failed for "+path+": "+cp.stderr[-500:])
 if cp.stdout.strip():lsof_rows.append({"path":path,"pids":cp.stdout.strip().splitlines()})
assert not lsof_rows,lsof_rows
ps=subprocess.run(["/bin/ps","-Ao","pid,command"],check=True,stdout=subprocess.PIPE,text=True).stdout
active=[line for line in ps.splitlines() if any(n.decode() in line for n in path_needles+hash_needles) and "retire_chunk_026.py" not in line]
assert not active,repr(active[:5])
pre={"schema":"numi-human.failed-volume-proposal-retirement-preflight.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS","chunk":"026","previous_path_proof":pin(PREV),"previous_retirement_preflight":pin(PREV_PREF),"previous_retirement_receipt":pin(PREV_REC),"fresh_attempt_scan":{"proof_path":str(PROOF),"proof_sha256":sha(PROOF.read_bytes()),"roots_scanned":[p.name for p in roots],"json_count":len(scanned_json),"npz_count":len(scanned_npz),"target_reference_hits":len(hits),"json_pins":scanned_json,"npz_metadata_pins":scanned_npz},"source_receipt":pin(STATE),"current_qualified_source_context_931":{"verified_state":pin(q931state),"verified_state_status":"STALE_PENDING_SOURCE_ONLY_STRING; receipt and independent replay are authoritative","receipt":pin(q931),"independent_replay":pin(q931replay),"physical_admission":False,"native_steps":0},"archive_verification":{"tag":gh["tagName"],"asset_id":asset["id"],"archive_sha256":rr["archive_sha256"],"restore_status":rr["status"],"restore_method":rr["restore_method"],"readback_status":rb["status"],"filesystem_restore_performed":False},"target_files_before":before,"lsof_open_handles":lsof_rows,"active_process_matches":active,"retirement_performed":False,"physical_admission":False,"native_steps":0}
pref=OUT/"retirement-preflight.json";pref.write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
for path,digest in TARGETS.items():assert sha(Path(path).read_bytes())==digest
removed=[]
for path,digest in TARGETS.items():
 Path(path).unlink();removed.append({"path":path,"sha256":digest,"bytes_retired":before[len(removed)]["size_bytes"]})
rec={"schema":"numi-human.failed-volume-proposal-retirement-receipt.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE","chunk":"026","preflight_sha256":sha(pref.read_bytes()),"archive_sha256":rr["archive_sha256"],"archive_index_sha256":sha((OUT/"archive-index.json").read_bytes()),"github_metadata_sha256":sha(GHMETA.read_bytes()),"files_removed":removed,"total_bytes_retired":sum(r["bytes_retired"] for r in removed),"sources_reports_and_witnesses_preserved":True,"physical_admission":False,"native_steps":0}
rp=OUT/"retirement-receipt.json";rp.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":rec["status"],"files_removed":removed,"total_bytes_retired":rec["total_bytes_retired"],"roots_scanned":[p.name for p in roots],"target_reference_hits":len(hits),"lsof_open_handles":lsof_rows,"preflight":str(pref),"receipt":str(rp)},sort_keys=True))
