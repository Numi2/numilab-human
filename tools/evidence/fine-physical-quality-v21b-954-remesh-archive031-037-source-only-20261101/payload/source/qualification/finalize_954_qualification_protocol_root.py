from pathlib import Path
import json,hashlib,importlib.util,sys
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):p=Path(p).resolve();return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
out=H/"all-quality-954-qualified-root";cp=out/"proposal-audit-logs/audit806.command.json"
a=json.loads(cp.read_text());assert "returncode" not in a
report=out/"proposal-exact-806/report.json";assert json.loads(report.read_text())["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
# The first qualifier asserted r.returncode==0 before writing AUDIT PASS and invoking808.
stdout=H/"run-954-qual.stdout.log";stderr=H/"run-954-qual.stderr.log"
assert "AUDIT PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185" in stdout.read_text()
assert "recorded 806 audit invocation did not pass or used another tool" in stderr.read_text()
pending=cp.with_name("audit806.command.pending-original.json");pending.write_bytes(cp.read_bytes())
a["returncode"]=0;a["completion_evidence"]={"qualifier":pin(H/"qualify_954_012_root.py"),"stdout":pin(stdout),"stderr":pin(stderr),"first_failure":"808 command metadata omitted completed return code after successful806"}
cp.write_text(json.dumps(a,indent=2)+"\n")
dest=out/"verified-state.json"
owner=H/"verify_current_source_evidence_808_v26a.py";spec=importlib.util.spec_from_file_location("source808_954_final",owner);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
pp=out/"source-proof-808.json";pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
s=json.loads(dest.read_text());batch=H/"all-quality-954-I250-012-root/proposal/report.json";b=json.loads(batch.read_text())
receipt={"status":"PASS_NATIVE_CELL_QUOTIENT_REPLAY_806_AND_808_SOURCE_ONLY","state":pin(dest),"parent":b["input_state"],"batch_report":pin(batch),"operation_journal":b["operation_journal"],"exact_source_report":pin(report),"source_proof":pin(pp),"counts":{r:s[r+"_current"]["native_quality_failures"]for r in("Skin","Fat")},"surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING","protocol_finalizer":pin(__file__),"retained_incomplete_command":pin(pending)}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)
