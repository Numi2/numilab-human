from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-022-plan"
PLAN=OUT/"selection-plan.json"; PROOF20=H/"quality-volume-failed-proposals-current-path-proof-020.json"; PROOF21=H/"quality-volume-failed-proposals-current-path-proof-022.json"
RESTORE=OUT/"one-npz-restore-receipt.json"; READBACK=OUT/"npz-readback-report.json"; GHMETA=OUT/"github-release-metadata.json"; STATE=H/"all-quality-867-root/state.json"
TARGETS={str(H/"all-quality-867-root/fat-coupled-quality-point-proposal.npz"):"34a7e78b0c30bfd2f543ce74383de5de2ef8cc466b4d2396f4e1e3176df5e4dc",str(H/"all-quality-867-root/skin-coupled-quality-point-proposal.npz"):"787d9b70ccf51677a355468d3d84bd0d2cefb2a6a4bb3e01709269b6e0d710c7"}
def sha(b): return hashlib.sha256(b).hexdigest()
def pin(p):
 b=p.read_bytes(); st=p.stat(); return {"path":str(p),"sha256":sha(b),"size_bytes":len(b),"mtime_ns":st.st_mtime_ns,"link_count":st.st_nlink,"is_symlink":p.is_symlink()}
plan=json.loads(PLAN.read_text()); proof20=json.loads(PROOF20.read_text()); proof21=json.loads(PROOF21.read_text()); rr=json.loads(RESTORE.read_text()); rb=json.loads(READBACK.read_text()); gh=json.loads(GHMETA.read_text()); sx=json.loads(STATE.read_text())
assert sha(PROOF20.read_bytes())=="92ff17843ce1172e9ea54028b2d841c4e0c0f69078604efba4635f1014764b53"
assert proof20["status"]=="PASS_METADATA_SCAN_COMPLETE_HASH_ALIASES_RESOLVED" and proof20["summary"]["exact_target_path_hits"]==0 and proof20["summary"]["scan_error_count"]==0
assert sha(PROOF21.read_bytes())=="4552b65f2993f139caa0ff073208ba56e0034a5b91d0c5fa598ccfc2e4f158c3"
assert proof21["status"]=="PASS_INHERITED_CURRENT_PASSING_ANCESTRY_AND_NEWER_ATTEMPTS_SCANNED" and proof21["newer_attempt_scan"]["exact_path_or_hash_hits"]==0
assert proof21["target_inventory"] and all(x["sha256"] in TARGETS.values() for x in proof21["target_inventory"])
assert rr["status"]=="PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK" and rr["archive_sha256"]=="726a884e3f37cda61914d2dc245a7ed48e779be12b8ad7639913c48b2dd48b13" and not rr["filesystem_restore_performed"]
assert rb["status"]=="PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS" and rb["all_zip_crcs_passed"] is True
asset=next(a for a in gh["assets"] if a["name"]=="quality-volume-archive-chunk-022.tar.zst")
assert gh["tagName"]=="human-quality-volume-failed-unadmitted-20261101-chunk-022" and asset["id"]=="RA_kwDOUElluM4pjOoc" and asset["digest"]=="sha256:"+rr["archive_sha256"] and asset["size"]==39954449
assert sx.get("physical_admission") is False and sx.get("native_steps")==0 and sx["Fat_current"]["native_quality_failures"]>0 and sx["Skin_current"]["native_quality_failures"]>0
needles=[p.encode() for p in TARGETS]+[Path(p).relative_to(H).as_posix().encode() for p in TARGETS]+[d.encode() for d in TARGETS.values()]
# Recheck any later quality roots that appeared after preparation while the
# serial lock was held by geometry. Metadata only; no geometry arrays are loaded.
roots=[]
for p in H.iterdir():
 if p.is_dir() and p.name.startswith("all-quality-"):
  try: n=int(p.name.split("-",2)[2].split("-",1)[0])
  except (ValueError,IndexError): continue
  if n>924: roots.append(p)
roots.sort(key=lambda p:(int(p.name.split("-",2)[2].split("-",1)[0]),p.name))
scanned_json=[]; scanned_npz=[]; hits=[]
for root in roots:
 for p in sorted(root.rglob("*.json")):
  data=p.read_bytes(); found=[x.decode() for x in needles if x in data]
  if found: hits.append({"path":str(p),"needles":found})
  scanned_json.append({"path":str(p),"sha256":sha(data),"size_bytes":len(data)})
 for p in sorted(root.rglob("*.npz")):
  members=[]
  with zipfile.ZipFile(p) as z:
   for name in z.namelist():
    low=Path(name).name.lower()
    if any(k in low for k in ("path","sha256","parent")):
     data=z.read(name); found=[x.decode() for x in needles if x in data]
     if found: hits.append({"path":str(p),"member":name,"needles":found})
     members.append({"member":name,"sha256":sha(data)})
  scanned_npz.append({**pin(p),"metadata_members":members})
assert not hits, hits[:10]
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
active=[line for line in ps.splitlines() if any(n.decode() in line for n in needles) and "retire_chunk_022.py" not in line]
assert not active,repr(active[:5])
pre={"schema":"numi-human.failed-volume-proposal-retirement-preflight.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_READY_TO_RETIRE_EXACT_ARCHIVED_FAILED_VOLUME_PROPOSALS","chunk":"022","inherited_current_path_proof":{"path":str(PROOF20),"sha256":sha(PROOF20.read_bytes()),"state_count":proof20["summary"]["state_count"],"npz_node_count":proof20["summary"]["npz_node_count"],"exact_target_path_hits":0,"target_sha_hits":proof20["summary"]["target_sha_hits"],"sha_aliases_resolved":True},"fresh_attempt_scan":{"proof_path":str(PROOF21),"proof_sha256":sha(PROOF21.read_bytes()),"roots_scanned":[p.name for p in roots],"json_count":len(scanned_json),"npz_count":len(scanned_npz),"target_reference_hits":len(hits),"json_pins":scanned_json,"npz_metadata_pins":scanned_npz},"source_receipt":pin(STATE),"current_qualified_context_924":qpin,"archive_verification":{"tag":gh["tagName"],"asset_id":asset["id"],"archive_sha256":rr["archive_sha256"],"restore_status":rr["status"],"restore_method":rr["restore_method"],"readback_status":rb["status"]},"target_files_before":before,"active_process_matches":active,"retirement_performed":False,"physical_admission":False,"native_steps":0}
(OUT/"retirement-preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
removed=[]
for path,digest in TARGETS.items():
 p=Path(path); assert sha(p.read_bytes())==digest
 p.unlink(); removed.append({"path":path,"sha256":digest,"bytes_retired":before[len(removed)]["size_bytes"]})
receipt={"schema":"numi-human.failed-volume-proposal-retirement-receipt.v1","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),"status":"PASS_ARCHIVED_FAILED_VOLUME_PROPOSALS_RETIRED_AFTER_INDEPENDENT_GH_RESTORE","chunk":"022","preflight_sha256":sha((OUT/"retirement-preflight.json").read_bytes()),"archive_sha256":rr["archive_sha256"],"archive_index_sha256":sha((OUT/"archive-index.json").read_bytes()),"github_metadata_sha256":sha(GHMETA.read_bytes()),"files_removed":removed,"total_bytes_retired":sum(r["bytes_retired"] for r in removed),"sources_reports_and_witnesses_preserved":True,"physical_admission":False,"native_steps":0}
(OUT/"retirement-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":receipt["status"],"files_removed":removed,"total_bytes_retired":receipt["total_bytes_retired"],"roots_scanned":[r.name for r in roots],"target_reference_hits":len(hits),"preflight":str(OUT/"retirement-preflight.json"),"receipt":str(OUT/"retirement-receipt.json")},sort_keys=True))
