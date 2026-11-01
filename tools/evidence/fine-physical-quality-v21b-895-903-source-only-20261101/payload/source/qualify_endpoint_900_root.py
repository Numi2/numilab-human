from pathlib import Path
import sys,subprocess,json,hashlib,importlib.util
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-900-qualified-root";out.mkdir(exist_ok=True)
parent=H/"all-quality-897-qualified-root/verified-state.json";proposal=H/"all-quality-900b-D-endpoint007a-root/proposal/candidate-state-proposal.json";batch=proposal.parent/"report.json"
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
b=json.loads(batch.read_text());s=json.loads(proposal.read_text());p=json.loads(parent.read_text())
assert b["status"]=="PASS_BOUNDED_ENDPOINT_QUOTIENT_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION"
assert b["accepted_count"]>0
jp=Path(b["operation_journal"]["path"]);assert pin(jp)==b["operation_journal"]
j=json.loads(jp.read_text());assert j["status"]=="PASS_FINAL_READBACK_PENDING_EXTERNAL_QUALIFICATION"
assert j["independent_surface_replay"]["source_point_bytes_identical"] is True
for vr in ("Fat",):
 for key in ("final_target_chain_exact","final_tets_replayed_from_parent","full_native_gate_replayed","all_exact_positive_changed_cells","full_replacement_parent_union_replayed","source_aliases_reconstructed"):
  assert j["independent_volume_replay"][vr][key] is True
assert j["input_state"]==pin(parent) and b["input_state"]==pin(parent)
assert b["candidate_state_proposal"]==pin(proposal)
assert s["source_state"]==pin(parent)
cfg={"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":p["source_surfaces"],"candidate_surfaces":s["source_surfaces"],"changed_roles":["D"],"fixed_coordinate_topology_only":{"declared":True,"edited_role":"D","all_point_bytes_fixed":True,"maximum_inner_back_surface_hausdorff_m":5e-5,"independent_replay":pin(batch),"endpoint_quotient_maximum_surface_distance_m":5e-5,"operation_journal":pin(jp)}}
cp=out/"context-config.json";cp.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
aud=H/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py";report=out/"proposal-exact-775/report.json";logs=out/"proposal-audit-logs";logs.mkdir()
cmd=[sys.executable,str(aud),"--config",str(cp),"--out",str(report.parent)]
with (logs/"audit775.stdout.log").open("w") as f,(logs/"audit775.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(logs/"audit775.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode},indent=2)+"\n")
assert r.returncode==0
a=json.loads(report.read_text());print("AUDIT",a["status"],flush=True)
if not a["status"].startswith("PASS_"):raise SystemExit(3)
s.update(surface_crossings=0,exact_context=pin(report),source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",independent_batch_replay=pin(batch),physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False},quality_working_state_qualifier=pin(Path(__file__)))
dest=out/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
owner=H/"verify_current_source_evidence_807_root.py";sp=importlib.util.spec_from_file_location("root877proof",owner);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
(out/"source-proof-807.json").write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
receipt={"status":"PASS_NATIVE_CELL_BATCH_REPLAY_775_AND_807_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"batch_report":pin(batch),"exact_source_report":pin(report),"counts":{r:s[r+"_current"]["native_quality_failures"] for r in("Skin","Fat")},"source_coordinates":"ALL_F32_BYTES_FIXED; SOURCE_ENDPOINT_QUOTIENT_AND_VOLUME_CONNECTIVITY_REPAIRED_WITHIN_50UM","physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)

subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)

