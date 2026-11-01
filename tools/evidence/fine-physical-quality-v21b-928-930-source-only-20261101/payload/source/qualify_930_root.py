from pathlib import Path
import json,hashlib,importlib.util,sys,subprocess
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");root=H/"all-quality-930-fixed-cavity-batch-root";out=root/"qualified";out.mkdir()
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
proof=root/"external-replay-v4d.json";assert pin(proof)["sha256"]=="214967cd9b21bc436a26f8a77b15c33c06d2427b9c76d9750457246a0f996378"
v=json.loads(proof.read_text());assert v["status"]=="PASS_EXTERNAL_INDEPENDENT_REPLAY_PENDING_FULL_CONTEXT"
raw=root/"batch/candidate-state-proposal.json";assert v["candidate_state"]==pin(raw)
parent=H/"all-quality-929-fixed-boundary-complete-cow-root/flips/verified-state.json";assert v["parent_state"]==pin(parent)
s=json.loads(raw.read_text());p=json.loads(parent.read_text());assert s["source_surfaces"]==p["source_surfaces"]
for r in ("Skin","Fat"):assert s[r+"_current"]["native_quality_failures"]<=p[r+"_current"]["native_quality_failures"]
s.update(surface_crossings=0,source_audit_status="PASS_INHERITED_UNCHANGED_SOURCE_BOUNDARY",independent_replay=pin(proof),physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
dest=out/"verified-state.json";write(dest,s)
sv=H/"verify_current_source_evidence_808_v26a.py";assert pin(sv)["sha256"]=="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9"
spec=importlib.util.spec_from_file_location("root930source",sv);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
source=m.verify_current_source_evidence(dest);assert source["status"].startswith("PASS_");sp=out/"source-proof-808.json";write(sp,source)
counts={r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")}
write(out/"receipt.json",{"status":"PASS_EXTERNAL_FIXED_BOUNDARY_NATIVE_CELL_REPLAY_AND_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"raw_candidate":pin(raw),"independent_replay":pin(proof),"source_proof":pin(sp),"counts":counts,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"})
subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)
