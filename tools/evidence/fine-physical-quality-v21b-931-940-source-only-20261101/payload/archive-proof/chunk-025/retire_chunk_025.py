from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-025-plan"
PLAN=OUT/"selection-plan.json"; PROOF20=H/"quality-volume-failed-proposals-current-path-proof-020.json"; PROOF21=H/"quality-volume-failed-proposals-current-path-proof-025.json"
RESTORE=OUT/"one-npz-restore-receipt.json"; READBACK=OUT/"npz-readback-report.json"; GHMETA=OUT/"github-release-metadata.json"; STATE=H/"all-quality-873-root/round-02/proposal/state.json"
TARGETS={row["path"]:row["expected_sha256_from_source_receipt"] for row in json.loads(PLAN.read_text())["files"]}
def sha(b): return hashlib.sha256(b).hexdigest()
def pin(p):
 b=p.read_bytes(); st=p.stat(); return {"path":str(p),"sha256":sha(b),"size_bytes":len(b),"mtime_ns":st.st_mtime_ns,"link_count":st.st_nlink,"is_symlink":p.is_symlink()}
plan=json.loads(PLAN.read_text()); proof20=json.loads(PROOF20.read_text()); proof21=json.loads(PROOF21.read_text()); rr=json.loads(RESTORE.read_text()); rb=json.loads(READBACK.read_text()); gh=json.loads(GHMETA.read_text()); sx=json.loads(STATE.read_text())
assert sha(PROOF20.read_bytes())==proof21["scope"]["inherited_full_scan_sha256"]
assert proof20["status"]=="PASS_METADATA_SCAN_COMPLETE_HASH_ALIASES_RESOLVED" and proof20["summary"]["exact_target_path_hits"]==0 and proof20["summary"]["scan_error_count"]==0
assert sha(PROOF21.read_bytes())==plan["current_direct_path_proof"]["sha256"]
assert proof21["status"]=="PASS_INHERITED_CURRENT_PASSING_ANCESTRY_AND_NEWER_ATTEMPTS_SCANNED_WITH_LIVE_ALIAS" and proof21["newer_attempt_scan"]["direct_original_path_hits"]==[] and proof21["newer_attempt_scan"]["unresolved_hash_hits"]==[] and proof21["newer_attempt_scan"]["resolved_live_alias_reference_count"]>0 and proof21["inherited_target_path_hits"]==0 and proof21["inherited_target_hash_hits_resolved"]["Skin"]==1
assert proof21["target_inventory"] and all(x["sha256"] in TARGETS.values() for x in proof21["target_inventory"])
assert rr["status"]=="PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK" and rr["archive_sha256"]=="42001c612a4cacdebab1b12bfa6eeefb036626850d460c31ada4f0149d6d9f8a" and not rr["filesystem_restore_performed"]
assert rb["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and rb["all_zip_crcs_passed"] is True
asset=next(a for a in gh["assets"] if a["name"]=="quality-volume-archive-chunk-025.tar.zst")
assert gh["tagName"]=="human-quality-volume-failed-unadmitted-20261101-chunk-025" and asset["id"]=="RA_kwDOUElluM4pjS-Q" and asset["digest"]=="sha256:"+rr["archive_sha256"] and asset["size"]==39936238
assert sx.get("physical_admission") is False and sx.get("native_steps")==0 and sx["Fat_current"]["native_quality_failures"]>0 and sx["Skin_current"]["native_quality_failures"]>0
alias=H/"all-quality-873-root/round-02/witness-component-rollback/skin-coupled-quality-point-rollback-proposal.npz"
assert alias.is_file() and not alias.is_symlink() and sha(alias.read_bytes())==TARGETS[str(H/"all-quality-873-root/round-02/proposal/skin-coupled-quality-point-proposal.npz")]
alias_rows=[x for x in proof20["hash_alias_resolution"] if x.get("target")==str(H/"all-quality-873-root/round-02/proposal/skin-coupled-quality-point-proposal.npz")]
assert len(alias_rows)==1 and alias_rows[0]["alias_path"]==str(alias) and alias_rows[0]["alias_matches"] is True
q931state=H/"all-quality-931-b5-backtracking-root/qualified/verified-state.json"; q931=H/"all-quality-931-b5-backtracking-root/qualified/receipt.json"; q931replay=H/"all-quality-931-b5-backtracking-root/qualified/independent-v26b5b-replay.json"; ERR21=H/"quality-volume-failed-proposals-current-path-proof-021-erratum-001.json"
assert hashlib.sha256(ERR21.read_bytes()).hexdigest()=="75a09846639c0702e3de4268136767f1d4c5b8a0d3a981e43869060516126b7b"
assert sha(q931.read_bytes())==proof21["qualified_source_context_931"]["receipt"]["sha256"]
assert sha(q931replay.read_bytes())==proof21["qualified_source_context_931"]["independent_replay"]["sha256"]
assert json.loads(q931state.read_text()).get("physical_admission") is False and json.loads(q931state.read_text()).get("native_steps")==0
path_needles=[p.encode() for p in TARGETS]+[Path(p).relative_to(H).as_posix().encode() for p in TARGETS]
hash_to_target={d:p for p,d in TARGETS.items()}
alias=H/"all-quality-873-root/round-02/witness-component-rollback/skin-coupled-quality-point-rollback-proposal.npz"
alias_by_target={str(H/"all-quality-873-root/round-02/proposal/skin-coupled-quality-point-proposal.npz"):str(alias)}
needles=path_needles+[d.encode() for d in TARGETS.values()]
roots=[]
for p in H.iterdir():
 if p.is_dir() and p.name.startswith("all-quality-"):
  try:n=int(p.name.split("-",2)[2].split("-",1)[0])
  except (ValueError,IndexError):continue
  if n>924:roots.append(p)
roots.sort(key=lambda p:(int(p.name.split("-",2)[2].split("-",1)[0]),p.name))
scanned_json=[];scanned_npz=[];direct_hits=[];unresolved_hash_hits=[];resolved_alias_refs=[]
def inspect_json(value,file_path,loc="$"):
 if isinstance(value,dict):
  local_path=value.get("path")
  for key,item in value.items():
   if isinstance(item,str) and item in hash_to_target:
    target_path=hash_to_target[item]
    expected_alias=alias_by_target.get(target_path)
    if expected_alias is not None and local_path==expected_alias and key in ("sha256","source_sha256","candidate_sha256"):
     resolved_alias_refs.append({"file":str(file_path),"json_path":loc+"."+key,"target":target_path,"sha256":item,"alias_path":expected_alias})
    else:
     unresolved_hash_hits.append({"file":str(file_path),"json_path":loc+"."+key,"target":target_path,"sha256":item,"containing_path":local_path})
   inspect_json(item,file_path,loc+"."+str(key))
 elif isinstance(value,list):
  for i,item in enumerate(value):inspect_json(item,file_path,f"{loc}[{i}]")
for root in roots:
 for p in sorted(root.rglob("*.json")):
  data=p.read_bytes()
  if any(q in data for q in path_needles):direct_hits.append({"path":str(p),"kind":"json","matches":[q.decode() for q in path_needles if q in data]})
  inspect_json(json.loads(data),p)
  scanned_json.append({"path":str(p),"sha256":sha(data),"size_bytes":len(data)})
 for p in sorted(root.rglob("*.npz")):
  members={}
  with zipfile.ZipFile(p) as z:
   for name in z.namelist():
    low=Path(name).name.lower()
    if any(k in low for k in ("path","sha256","parent")):
     data=z.read(name);members[name]=data
     if any(q in data for q in path_needles):direct_hits.append({"path":str(p),"member":name,"kind":"npz-metadata","matches":[q.decode() for q in path_needles if q in data]})
  for member,data in members.items():
   for digest,target_path in hash_to_target.items():
    if digest.encode() in data:
     expected_alias=alias_by_target.get(target_path)
     alias_present=expected_alias is not None and any(expected_alias.encode() in val for name,val in members.items() if "path" in Path(name).name.lower())
     if alias_present:resolved_alias_refs.append({"file":str(p),"member":member,"target":target_path,"sha256":digest,"alias_path":expected_alias})
     else:unresolved_hash_hits.append({"file":str(p),"member":member,"target":target_path,"sha256":digest,"containing_path":None})
  scanned_npz.append({**pin(p),"metadata_members":[{"member":n,"sha256":sha(data)} for n,data in sorted(members.items())]})
hits=direct_hits+unresolved_hash_hits
assert not direct_hits,direct_hits[:10]
assert not unresolved_hash_hits,unresolved_hash_hits[:10]
assert resolved_alias_refs and all(x["target"]==str(H/"all-quality-873-root/round-02/proposal/skin-coupled-quality-point-proposal.npz") for x in resolved_alias_refs)
assert alias.is_file() and not alias.is_symlink() and sha(alias.read_bytes())==TARGETS[str(H/"all-quality-873-root/round-02/proposal/skin-coupled-quality-point-proposal.npz")]
# Keep current qualified context 924 pinned; current unqualified attempts after it
# have just been inspected by exact path/hash scan above.
ctx=proof20["current_context_pins"]
qpath=Path(next(k for k in ctx if "all-quality-924-joint-role-equal-root/qualified/verified-state.json" in k))
qpin=pin(qpath); assert qpin["sha256"]==ctx[str(qpath)]["sha256"]
plan_rows={r["path"]:r["expected_sha256_from_source_receipt"] for r in plan["files"]}; assert plan_rows==TARGETS
inv={r["path"]:r for r in proof20["target_inventory"]}
before=[]
for path,digest in TARGETS.items():
 p=Path(path); assert p.is_file() and not p.is_symlink()
 row=pin(p); assert row["sha256"]==digest and row["link_count"]==1
 assert inv[path]["exists"] and inv[path]["sha256"]==digest
 before.append(row)
ps=subprocess.run(["/bin/ps","-Ao","pid,command"],check=True,stdout=subprocess.PIPE,text=True).stdout
active=[line for line in ps.splitlines() if any(n.decode() in line for n in needles) and "retire_chunk_025.py" not in line]
assert not active,repr(active[:5])
pre={"schema":"numi-human.failed-volume-proposal-retirement-preflight.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS","chunk":"025","inherited_current_path_proof":{"path":str(PROOF20),"sha256":sha(PROOF20.read_bytes()),"state_count":proof20["summary"]["state_count"],"npz_node_count":proof20["summary"]["npz_node_count"],"exact_target_path_hits":0,"target_sha_hits":proof20["summary"]["target_sha_hits"],"sha_aliases_resolved":True},"fresh_attempt_scan":{"proof_path":str(PROOF21),"proof_sha256":sha(PROOF21.read_bytes()),"roots_scanned":[p.name for p in roots],"json_count":len(scanned_json),"npz_count":len(scanned_npz),"target_reference_hits":len(hits),"resolved_alias_reference_count":len(resolved_alias_refs),"resolved_alias_references":resolved_alias_refs,"json_pins":scanned_json,"npz_metadata_pins":scanned_npz},"source_receipt":pin(STATE),"proof021_status_erratum":pin(ERR21),"current_qualified_context_924":qpin,"current_qualified_source_context_931":{"verified_state":pin(q931state),"verified_state_status":"STALE_PENDING_SOURCE_ONLY_STRING; receipt and independent replay are authoritative","receipt":pin(q931),"independent_replay":pin(q931replay),"physical_admission":False,"native_steps":0},"preserved_hash_alias":pin(alias),"archive_verification":{"tag":gh["tagName"],"asset_id":asset["id"],"archive_sha256":rr["archive_sha256"],"restore_status":rr["status"],"restore_method":rr["restore_method"],"readback_status":rb["status"]},"target_files_before":before,"active_process_matches":active,"retirement_performed":False,"physical_admission":False,"native_steps":0}
(OUT/"retirement-preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
removed=[]
for path,digest in TARGETS.items():
 p=Path(path); assert sha(p.read_bytes())==digest
 p.unlink(); removed.append({"path":path,"sha256":digest,"bytes_retired":before[len(removed)]["size_bytes"]})
receipt={"schema":"numi-human.failed-volume-proposal-retirement-receipt.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE","chunk":"025","preflight_sha256":sha((OUT/"retirement-preflight.json").read_bytes()),"archive_sha256":rr["archive_sha256"],"archive_index_sha256":sha((OUT/"archive-index.json").read_bytes()),"github_metadata_sha256":sha(GHMETA.read_bytes()),"files_removed":removed,"total_bytes_retired":sum(r["bytes_retired"] for r in removed),"sources_reports_and_witnesses_preserved":True,"physical_admission":False,"native_steps":0}
(OUT/"retirement-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":receipt["status"],"files_removed":removed,"total_bytes_retired":receipt["total_bytes_retired"],"roots_scanned":[r.name for r in roots],"target_reference_hits":len(hits),"resolved_alias_reference_count":len(resolved_alias_refs),"resolved_alias_references":resolved_alias_refs,"preflight":str(OUT/"retirement-preflight.json"),"receipt":str(OUT/"retirement-receipt.json")},sort_keys=True))
