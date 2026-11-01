from pathlib import Path
import json,subprocess,sys,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");o=H/"all-quality-884-qualified-root";o.mkdir(exist_ok=True)
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
p=H/"all-quality-884-unified-root/state.json";s=json.loads(p.read_text());audit=H/"all-quality-884-audit805-root/proposal-exact-805/report.json";a=json.loads(audit.read_text());assert a["status"].startswith("PASS_")
s.update(exact_context=pin(audit),surface_crossings=0,source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
dest=o/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
v=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25h-unified-mixed/verify_ordinary_allrole_v25h1_legacy_volume_adapter_root.py";assert pin(v)["sha256"]=="8c19a5e458a198d43b8db24c58cc38c5deae9c6041f651db63700a512fe86f54"
cmd=[sys.executable,str(v),"--candidate-state",str(dest),"--parent-state",str(H/"all-quality-877-qualified-root/verified-state.json"),"--audit-report",str(audit),"--audit-config",str(H/"all-quality-884-unified-root/context-config.json"),"--output-report",str(o/"independent-unified-replay.json")]
t=time.monotonic()
with (o/"verifier.stdout.log").open("w") as f,(o/"verifier.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(o/"verifier.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n")
print("RETURN",r.returncode,flush=True)
if r.returncode:print((o/"verifier.stderr.log").read_text());raise SystemExit(r.returncode)
proof=json.loads((o/"independent-unified-replay.json").read_text());assert proof["status"].startswith("PASS_")
receipt={"status":"PASS_INDEPENDENT_UNIFIED_MASK_NATIVE_CELL_VOLUME_AND_805_807_SOURCE_ONLY","state":pin(dest),"raw_proposal":pin(p),"independent_replay":pin(o/"independent-unified-replay.json"),"exact_audit":pin(audit),"counts":{r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")},"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
(o/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)
