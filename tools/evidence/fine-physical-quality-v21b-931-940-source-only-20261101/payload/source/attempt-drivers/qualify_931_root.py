from pathlib import Path
import subprocess,sys,json,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");root=H/"all-quality-931-b5-backtracking-root";out=root/"qualified";out.mkdir()
parent=H/"all-quality-930-fixed-cavity-batch-root/qualified/verified-state.json";raw=root/"proposal/state.json";s=json.loads(raw.read_text());p=json.loads(parent.read_text())
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
def run(label,cmd,d):
 d.mkdir(exist_ok=True,parents=True);t=time.monotonic()
 with (d/(label+".stdout.log")).open("w") as f,(d/(label+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
 write(d/(label+".command.json"),{"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t})
 if r.returncode:print((d/(label+".stderr.log")).read_text()[-7000:],flush=True);raise SystemExit(r.returncode)
counts={r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")}
assert all(counts[r]<=p[r+"_current"]["native_quality_failures"] for r in counts)
audit=H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_806_extended_250um_construction.py"
assert pin(audit)["sha256"]=="1c20de7dffa2cbf55abf195310af988aed9c5a87f035a8e79e099a3f00c74377"
cfg=root/"proposal/context-config.json";ar=out/"proposal-exact-806/report.json"
run("audit806",[sys.executable,str(audit),"--config",str(cfg),"--out",str(ar.parent)],out/"proposal-audit-logs")
a=json.loads(ar.read_text());print("EXACT806",counts,a["status"],flush=True)
if a["status"]!="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":raise SystemExit(3)
dest=out/"verified-state.json"
s.update(exact_context=pin(ar),surface_crossings=0,source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
write(dest,s)
ver=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v26b5-cowdiff-f32backtrack-250um/verify_ordinary_allrole_v26b5b_jointgate_replay.py"
assert pin(ver)["sha256"]=="7a30e106c38b964f716f78cab6be20805bc754723c82329d9623c74e3b24546f"
proof=out/"independent-v26b5b-replay.json"
run("verifier",[sys.executable,str(ver),"--candidate-state",str(dest),"--parent-state",str(parent),"--audit-report",str(ar),"--audit-config",str(cfg),"--output-report",str(proof)],out/"verification-logs")
replay=json.loads(proof.read_text());assert replay["status"]=="PASS_INDEPENDENT_SOURCE_CONTEXT_VOLUME_QUALITY_AND_UNIFIED_MASK"
write(out/"receipt.json",{"status":"PASS_INDEPENDENT_EXPLICIT_250UM_UNIFIED_MASK_NATIVE_CELL_VOLUME_AND_806_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"raw_proposal":pin(raw),"independent_replay":pin(proof),"exact_audit":pin(ar),"counts":counts,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"})
run("checkpoint",[sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],out/"checkpoint-logs")
print("QUALIFIED",counts,pin(dest),flush=True)
