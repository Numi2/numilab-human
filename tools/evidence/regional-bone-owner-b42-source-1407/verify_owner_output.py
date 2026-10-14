from pathlib import Path
import json,hashlib,time
base=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/bilateral-b42-native-bone-owner-001")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
failed=json.loads((base/"compile-report.json").read_text())
assert failed["phase"]=="postcompile_checks" and "coverage" in failed["error"]
manifest=base/"owner-base/bodyparts3d-myosim-major-bones.manifest.json"
j=json.loads(manifest.read_text()); p=manifest.parent/j["payload"]["file"]
assert sha(p)==j["payload"]["sha256"]
assert j["payload"]["bone_count"]==185 and j["payload"]["registration_fingerprint32"]=="31c77660"
assert j["source"]["registration"]["sha256"]==failed["candidate_sha256"]
expected_registration=json.loads(Path(failed["candidate_path"]).read_text())
assert j["source"]["bodyparts"]==expected_registration["source"]["bodyparts"]
for key,h in failed["input_hashes_before"].items(): assert sha(Path(key))==h,key
for row in failed["myosim_source_tree_file_manifest"]:
 path=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim")/row["path"]
 assert sha(path)==row["sha256"],path
report={"status":"OWNER_COMPILED_SOURCE_OUTPUT_VERIFIED_AFTER_REPORT_FINALIZATION_FAILURE",
"failed_compile_report_path":str(base/"compile-report.json"),"failed_compile_report_sha256":sha(base/"compile-report.json"),
"owner_output_payload_path":str(p),"owner_output_payload_sha256":sha(p),"owner_output_manifest_sha256":sha(manifest),
"payload":j["payload"],"runtime_reference":j["source"]["runtime_reference"],
"input_hashes_unchanged":True,"myosim_source_tree_file_count_verified":len(failed["myosim_source_tree_file_manifest"]),
"owner_compile_repeated":False,"preparation_erratum":"Borrowed tissue report finalization expected coverage key; bone manifest has separate continuity/extent fields. Owner returned and payload/source checks completed; original failure retained.",
"anatomical_or_native_admission":False,"elapsed_owner_and_report_attempt_s":failed["finished_unix"]-failed["started_unix"],
"postcheck_unix":time.time()}
out=base/"postcompile-output-proof-002.json";assert not out.exists();out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"payload_sha256":sha(p),"proof_sha256":sha(out),"elapsed_s":report["elapsed_owner_and_report_attempt_s"],"payload":j["payload"]},indent=2))
