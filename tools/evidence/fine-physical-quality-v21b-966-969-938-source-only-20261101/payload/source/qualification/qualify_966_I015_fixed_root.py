from pathlib import Path
import sys,subprocess,json,hashlib,importlib.util,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):p=Path(p).resolve();return {"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
out=H/"all-quality-966-I015-fixed-775-qualified-root";out.mkdir(exist_ok=False)
parent=H/"all-quality-959-remesh-883h-fixed-775-qualified-root/verified-state.json"
proposal=H/"all-quality-959-I015-paired-root/run001/candidate-state-proposal.json"
batch=proposal.parent/"report.json";b=json.loads(batch.read_text());s=json.loads(proposal.read_text());p=json.loads(parent.read_text())
assert b["status"]=="PASS_BOUNDED_ENDPOINT_QUOTIENT_PROPOSAL_PENDING_EXTERNAL_QUALIFICATION" and b["accepted_count"]==2
assert b["input_state"]==pin(parent) and s["source_state"]==pin(parent)
jp=Path(b["operation_journal"]["path"]);assert pin(jp)==b["operation_journal"]
j=json.loads(jp.read_text());assert j["status"]=="PASS_FINAL_READBACK_PENDING_EXTERNAL_QUALIFICATION"
for k in("full_typed_union_and_ancestry_replayed","source_face_quotient_replayed","source_point_coordinates_replayed_from_anchors"):
 assert j["independent_surface_replay"][k] is True,k
for vr in("Skin","Fat"):
 for key in("final_target_chain_exact","final_tets_replayed_from_parent","full_native_gate_replayed","all_exact_positive_changed_cells","full_replacement_parent_union_replayed","source_aliases_reconstructed"):
  assert j["independent_volume_replay"][vr][key] is True,(vr,key)
 assert s[vr]==s[vr+"_current"]
assert b["paired_anchor_independent_replay"]["maximum_keep_anchor_l1_m"]==0
cfg={"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":p["source_surfaces"],"candidate_surfaces":s["source_surfaces"],"changed_roles":["I"],"fixed_coordinate_topology_only":{"declared":True,"edited_role":"I","all_point_bytes_fixed":True,"maximum_inner_back_surface_hausdorff_m":250e-6,"endpoint_quotient_maximum_surface_distance_m":250e-6,"independent_replay":pin(batch),"operation_journal":pin(jp)}}
cp=out/"context-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
aud=H/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
assert pin(aud)["sha256"]=="82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
report=out/"proposal-exact-775/report.json";logs=out/"proposal-audit-logs";logs.mkdir()
cmd=[sys.executable,str(aud),"--config",str(cp),"--out",str(report.parent)]
with(logs/"audit775.stdout.log").open("w")as f,(logs/"audit775.stderr.log").open("w")as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(logs/"audit775.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode},indent=2)+"\n")
assert r.returncode==0,("audit exit",r.returncode,(logs/"audit775.stderr.log").read_text()[-2000:])
a=json.loads(report.read_text());print("AUDIT",a["status"],flush=True)
assert a["status"].startswith("PASS_"),a
s.update(surface_crossings=0,exact_context=pin(report),source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",independent_batch_replay=pin(batch),physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False},quality_working_state_qualifier=pin(__file__))
dest=out/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
owner=H/"verify_current_source_evidence_808_v26a.py"
spec=importlib.util.spec_from_file_location("source808_954",owner);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
pp=out/"source-proof-808.json";pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
receipt={"status":"PASS_NATIVE_CELL_QUOTIENT_REPLAY_775_AND_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"batch_report":pin(batch),"operation_journal":pin(jp),"exact_source_report":pin(report),"source_proof":pin(pp),"counts":{r:s[r+"_current"]["native_quality_failures"]for r in("Skin","Fat")},"surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)

subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)
