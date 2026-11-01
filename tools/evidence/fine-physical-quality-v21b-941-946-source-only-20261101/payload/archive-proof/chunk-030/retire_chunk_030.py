from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,zipfile
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-030-plan";PLAN=OUT/"selection-plan.json";PROOF=H/"quality-volume-failed-proposals-current-path-proof-030.json"
PREV=H/"quality-volume-failed-proposals-current-path-proof-029.json";PREVPREF=H/"quality-volume-archive-chunk-029-plan/retirement-preflight.json";PREVREC=H/"quality-volume-archive-chunk-029-plan/retirement-receipt.json";P20=H/"quality-volume-failed-proposals-current-path-proof-020.json"
STATE=H/"all-quality-904-fat-only-root/round-03/proposal/candidate-state-proposal.json";RESTORE=OUT/"one-npz-restore-receipt.json";READBACK=OUT/"npz-readback-report.json";SUP=OUT/"all-files-exact-restore-supplement.json";BUILD=OUT/"archive-build-receipt.json";GHMETA=OUT/"github-release-metadata.json"
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
 b=p.read_bytes();s=p.stat();return {"path":str(p),"sha256":sha(b),"size_bytes":len(b),"mtime_ns":s.st_mtime_ns,"link_count":s.st_nlink,"is_symlink":p.is_symlink()}
def num(p):
 try:return int(p.name.split("-",2)[2].split("-",1)[0])
 except (ValueError,IndexError):return None
plan=json.loads(PLAN.read_text());proof=json.loads(PROOF.read_text());rr=json.loads(RESTORE.read_text());rb=json.loads(READBACK.read_text());sup=json.loads(SUP.read_text());build=json.loads(BUILD.read_text());gh=json.loads(GHMETA.read_text());state=json.loads(STATE.read_text());p20=json.loads(P20.read_text())
assert plan["chunk"]["id"]=="quality-volume-chunk-030" and len(plan["files"])==1 and plan["chunk"]["original_bytes"]==38447022
assert proof["status"]=="PASS_INHERITED_QUALIFIED_CONTEXT_AND_CURRENT_TREES_SCANNED_NO_EXTERNAL_TARGET_REFERENCES" and proof["target_path_hits"]==proof["target_hash_hits"]==0
assert proof["scope"]["target_root"]==str(H/"all-quality-904-fat-only-root")
assert proof["scope"]["inherited_current_path_proof_020"]["sha256"]==sha(P20.read_bytes())
assert p20["status"]=="PASS_METADATA_SCAN_COMPLETE_HASH_ALIASES_RESOLVED" and p20["summary"]["scan_error_count"]==0
assert json.loads(PREVPREF.read_text())["status"]=="PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS"
assert json.loads(PREVREC.read_text())["status"]=="PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE"
assert json.loads(PREV.read_text())["status"] in ("PASS_INHERITED_CURRENT_PASSING_ANCESTRY_AND_NEWER_ATTEMPTS_SCANNED_WITH_LIVE_ALIAS","PASS_INHERITED_QUALIFIED_CONTEXT_AND_CURRENT_TREES_SCANNED_NO_EXTERNAL_TARGET_REFERENCES")
assert rr["status"]=="PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK" and rb["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and rb["all_zip_crcs_passed"] is True
assert sup["status"]=="PASS_INDEPENDENT_GH_ARCHIVE_STREAM_AND_EXACT_RESTORE_ALL_SELECTED_FILES" and sup["archive_sha256"]==rr["archive_sha256"]
assert rr["filesystem_restore_performed"] is False and state["physical_admission"] is False and state["native_steps"]==0
assert state["Fat_current"]["native_quality_failures"]>0 and state["Skin_current"]["native_quality_failures"]>0
assert pin(STATE)["sha256"]==proof["scope"]["target_state"]["sha256"]
targets={r["path"]:r["expected_sha256_from_source_receipt"] for r in plan["files"]};assert len(targets)==1
readback_by_digest={r["sha256"]:r for r in rb["files"]}
for row in plan["files"]:
 p=Path(row["path"]);d=row["expected_sha256_from_source_receipt"]
 assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_size==row["size_bytes_from_stat"] and sha(p.read_bytes())==d
 assert p.resolve().is_relative_to(H/"all-quality-904-fat-only-root")
 assert d in readback_by_digest and readback_by_digest[d]["zip_crc"]=="PASS" and readback_by_digest[d]["npy_header_count"]>0
 assert str(p.relative_to(H)) in readback_by_digest[d]["source_paths"]
assert rr["archive_sha256"]==build["archive_sha256"] and rr["archive_sha256"]==sup["archive_sha256"]
asset=next(a for a in gh["assets"] if a["name"]=="quality-volume-archive-chunk-030.tar.zst")
assert gh["tagName"]=="human-quality-volume-failed-unadmitted-20261101-chunk-030"
assert rr["archive_sha256"]==asset["digest"].removeprefix("sha256:") and asset["size"]==build["archive_size_bytes"]
assert not (OUT/"quality-volume-archive-chunk-030.tar.zst").exists()
ctx=proof["current_qualified_source_context_931"]
for key in ("verified_state","receipt","independent_replay"):assert pin(Path(ctx[key]["path"]))["sha256"]==ctx[key]["sha256"]
assert ctx["physical_admission"] is False and ctx["native_steps"]==0
root=H/"all-quality-904-fat-only-root";pn=[s.encode() for s in targets]+[Path(s).relative_to(H).as_posix().encode() for s in targets];hn=[s.encode() for s in targets.values()]
allowed={STATE.resolve()}
for p in (STATE.parent/"verified-state.json",STATE.parent/"report.json"):
 if p.is_file():allowed.add(p.resolve())
allowed.update(p.resolve() for p in STATE.parent.glob("*npz-clone-receipt.json"))
same=[];external=[];jp=[];np=[]
roots=[]
for q in H.iterdir():
 n=num(q)
 if q.is_dir() and q.name.startswith("all-quality-") and n is not None and (n>=925 or q.resolve()==root.resolve()):roots.append(q)
roots.sort(key=lambda p:(num(p),p.name))
for qr in roots:
 for p in sorted(qr.rglob("*.json")):
  b=p.read_bytes();m=[x.decode() for x in pn+hn if x in b]
  if m:
   hit={"path":str(p),"matches":m}
   if qr.resolve()==root.resolve():
    if p.resolve() not in allowed:same.append(hit)
   else:external.append(hit)
  jp.append({"path":str(p),"sha256":sha(b),"size_bytes":len(b)})
 for p in sorted(qr.rglob("*.npz")):
  members=[]
  with zipfile.ZipFile(p) as z:
   for name in z.namelist():
    if any(k in Path(name).name.lower() for k in ("path","sha256","parent")):
     b=z.read(name);m=[x.decode() for x in pn+hn if x in b]
     if m:
      hit={"path":str(p),"member":name,"matches":m}
      (same if qr.resolve()==root.resolve() else external).append(hit)
     members.append({"member":name,"sha256":sha(b)})
  np.append({"path":str(p),"sha256":sha(p.read_bytes()),"size_bytes":p.stat().st_size,"metadata_members":members})
assert not same,same[:10]
assert not external,external[:10]
lsof=[]
for path in targets:
 cp=subprocess.run(["/usr/sbin/lsof","-t",path],capture_output=True,text=True)
 if cp.returncode not in (0,1):raise RuntimeError("lsof error "+cp.stderr[-300:])
 if cp.stdout.strip():lsof.append({"path":path,"pids":cp.stdout.strip().splitlines()})
assert not lsof,lsof
ps=subprocess.run(["/bin/ps","-Ao","pid,command"],capture_output=True,text=True,check=True).stdout
active=[l for l in ps.splitlines() if any(x.decode() in l for x in pn+hn) and "retire_chunk_030.py" not in l]
assert not active,active[:8]
free0=shutil.disk_usage(H).free
pre={"schema":"numi-human.failed-volume-proposal-retirement-preflight.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS","chunk":"030","plan_sha256":sha(PLAN.read_bytes()),"previous_chunk_029":{"proof":pin(PREV),"preflight":pin(PREVPREF),"receipt":pin(PREVREC)},"inherited_path_proof_020":pin(P20),"current_path_proof":pin(PROOF),"current_path_proof_status":proof["status"],"target_source_state":pin(STATE),"target_unqualified_failures":{"Fat":state["Fat_current"]["native_quality_failures"],"Skin":state["Skin_current"]["native_quality_failures"],"physical_admission":False,"native_steps":0},"archive_verification":{"tag":gh["tagName"],"asset_id":asset["id"],"archive_sha256":rr["archive_sha256"],"zip_npy_status":rb["status"],"all_exact_restoration_status":sup["status"],"all_selected_npz_ram_restored":True,"filesystem_restore_performed":False,"local_archive_removed_after_verification":True},"qualified_context_931":{"verified_state":ctx["verified_state"],"receipt":ctx["receipt"],"independent_replay":ctx["independent_replay"],"physical_admission":False,"native_steps":0},"fresh_ownership_scan":{"roots":[x.name for x in roots],"json_count":len(jp),"npz_count":len(np),"same_root_nonreceipt_reference_hits":same,"external_path_or_hash_hits":external,"json_pins":jp,"npz_metadata_pins":np},"lsof_open_handles":lsof,"active_process_matches":active,"target_files_before":[pin(Path(p)) for p in targets],"df_free_bytes_before":free0,"retirement_performed":False,"physical_admission":False,"native_steps":0}
pref=OUT/"retirement-preflight.json";pref.write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
removed=[]
for path,d in targets.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 and sha(p.read_bytes())==d
 size=p.stat().st_size;p.unlink();removed.append({"path":path,"sha256":d,"logical_bytes_retired":size})
free1=shutil.disk_usage(H).free
rec={"schema":"numi-human.failed-volume-proposal-retirement-receipt.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE","chunk":"030","preflight_sha256":sha(pref.read_bytes()),"archive_sha256":rr["archive_sha256"],"archive_index_sha256":sha((OUT/"archive-index.json").read_bytes()),"github_metadata_sha256":sha(GHMETA.read_bytes()),"readback_receipt_sha256":sha(READBACK.read_bytes()),"all_files_exact_restore_supplement_sha256":sha(SUP.read_bytes()),"files_removed":removed,"logical_bytes_retired":sum(x["logical_bytes_retired"] for x in removed),"df_free_bytes_before":free0,"df_free_bytes_after":free1,"measured_df_free_increase_bytes":free1-free0,"physical_recovery_reported_from_df_only":True,"sources_reports_states_witnesses_preserved":True,"physical_admission":False,"native_steps":0}
(OUT/"retirement-receipt.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":rec["status"],"files_removed":removed,"logical_bytes_retired":rec["logical_bytes_retired"],"df_free_before":free0,"df_free_after":free1,"measured_df_delta":free1-free0,"root_count":len(roots),"json_count":len(jp),"npz_count":len(np),"path_hash_hits":len(same)+len(external),"lsof_open_handles":lsof,"preflight":str(pref),"receipt":str(OUT/"retirement-receipt.json")},sort_keys=True))
