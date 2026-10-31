from pathlib import Path
import sys,subprocess,hashlib,json,importlib.util
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-871-confirmation-root";out.mkdir(exist_ok=False)
state=H/"all-quality-871-root/round-01/finalized-current-state.json"
config=H/"all-quality-871-root/round-01/witness-component-rollback/context-config.json"
logs=out/"proposal-audit-logs";logs.mkdir()
aud=H/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
cmd=[sys.executable,str(aud),"--config",str(config),"--out",str(out/"proposal-exact-775")]
with (logs/"audit775.stdout.log").open("w") as f,(logs/"audit775.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(logs/"audit775.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode},indent=2)+"\n")
if r.returncode:raise SystemExit(r.returncode)
report=out/"proposal-exact-775/report.json";rd=json.loads(report.read_text());assert rd["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
d=json.loads(state.read_text());d["previous_exact_context_before_invocation_namespace_confirmation"]=d["exact_context"]
d["exact_context"]=pin(report);d["source_invocation_confirmation_producer"]=pin(Path(__file__))
dest=out/"verified-state.json";dest.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
owner=H/"verify_current_source_evidence_807_root.py";sp=importlib.util.spec_from_file_location("root871proof",owner);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
(out/"source-proof-807.json").write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
receipt={"status":"PASS_SAVED_INDEPENDENT_NATIVE_QUALITY_AND_FRESH_SOURCE_CONFIRMATION","state":pin(dest),"input_state":pin(state),"fresh_source_audit":pin(report),"counts":{r:d[r+"_current"]["native_quality_failures"] for r in("Skin","Fat")},"reason":"unchanged 807 bounded775 validator requires actual invocation in proposal-audit-logs beside current report; original rollback audit and logs preserved","physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)

