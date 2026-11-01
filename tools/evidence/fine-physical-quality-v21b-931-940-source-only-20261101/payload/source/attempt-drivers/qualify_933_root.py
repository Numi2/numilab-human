from pathlib import Path
import sys,json,hashlib,importlib.util,subprocess
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-933-fixed-boundary-complete-cow-root/flips";sp=out/"state.json";parent=H/"all-quality-931-b5-backtracking-root/qualified/verified-state.json"
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
v=H/"verify_fixed_boundary_journal_752_root.py";m=mod("replay923",v);proof=m.verify_step(out/"report.json",sp,parent)
rp=out/"independent-replay.json";rp.write_text(json.dumps({"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_REPLAY","replay":proof,"verifier":pin(v),"physical_admission":False,"native_steps":0},indent=2)+"\n")
s=json.loads(sp.read_text());s.update(surface_crossings=0,source_audit_status="PASS_INHERITED_UNCHANGED_SOURCE_BOUNDARY",independent_replay=pin(rp),physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
dest=out/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
sv=mod("src923",H/"verify_current_source_evidence_808_v26a.py");pp=sv.verify_current_source_evidence(dest);assert pp["status"].startswith("PASS_")
(out/"source-proof-808.json").write_text(json.dumps(pp,indent=2)+"\n")
counts={r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")}
(out/"receipt.json").write_text(json.dumps({"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_NATIVE_CELL_REPLAY_AND_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"independent_replay":pin(rp),"source_proof":pin(out/"source-proof-808.json"),"counts":counts,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"},indent=2)+"\n")
subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)
print("QUALIFIED",counts,pin(dest),flush=True)
