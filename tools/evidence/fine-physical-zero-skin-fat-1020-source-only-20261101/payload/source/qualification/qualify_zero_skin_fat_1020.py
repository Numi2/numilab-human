#!/usr/bin/env python3
"""Qualify the current paired point repair with unchanged 806/808 source owners."""
from pathlib import Path
import json,hashlib,subprocess,sys,importlib.util
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):
 p=Path(p).resolve()
 with p.open("rb")as f:h=hashlib.file_digest(f,"sha256").hexdigest()
 return {"path":str(p),"sha256":h}
def checked(r):
 assert pin(r["path"])=={"path":str(Path(r["path"]).resolve()),"sha256":r["sha256"]},r
 return Path(r["path"]).resolve()
def main():
 batch=H/"all-quality-1009-point-update-1018-root/actual-001/point-update-1018-report.json"
 assert pin(batch)["sha256"]=="ce8e9544f3a0e285cb17fd654619e8f55360b7e6675dbf87b9edfb4191f3baf2"
 b=json.loads(batch.read_text());assert b["status"]=="PASS_INDEPENDENT_POINT_UPDATE_REPLAY_QUALITY_ONLY"
 assert b["physical_admission"]is False and b["native_steps"]==0
 checked(b["producer"]);checked(b["fit_producer"]);checked(b["fit_report"])
 parent=checked(b["input_state"]);p=json.loads(parent.read_text())
 assert pin(parent)["sha256"]=="413d7ecbf624c0c89c13f3f63905d7c7baf9d222d34d7ce11b706e95a0e4ec2a"
 proposal=checked(b["verified_state"]);s=json.loads(proposal.read_text())
 assert s["source_state"]==pin(parent)
 r=b["independent_replay"]
 assert r["paired_exact_volume_cancel"] is True and r["paired_exact_det6_delta_integer"]=="0"
 assert r["source_facets_and_volume_topology_unchanged"]is True
 for role in("Skin","Fat"):
  rr=r["roles"][role]
  assert rr["candidate_bad_cells"]==rr["old_good_regressions"]==0
  assert rr["candidate_min_q"]>=1e-5 and rr["exact_det6_floor_m3"]==6e-18
  assert rr["changed_star_exact_nonpositive_cells"]==rr["changed_star_exact_det6_floor_failures"]==[]
  assert rr["topology_material_boundary_arrays_bit_identical_to_parent"]is True
  assert rr["target_actual_oriented_chain_equal"]is True
  assert rr["local_exact_volume_delta_fraction"]<=.01 and rr["global_exact_volume_delta_fraction"]<=1e-4
  assert s[role+"_current"]["native_quality_failures"]==s[role+"_current"]["nonpositive"]==0
  s[role]=dict(s[role+"_current"]);checked(s[role])
 for role in("O","D"):assert s["source_surfaces"][role]==p["source_surfaces"][role]
 for role in("O","I","D"):checked(s["source_surfaces"][role])
 face=b["source_face_screen"];assert face["area_ratio_min"]>=.25 and face["area_ratio_max"]<=4 and face["orientation_cosine_min"]>=.1
 out=H/"all-quality-1020-zero-skin-fat-source-qualified-root";out.mkdir(exist_ok=False)
 cfg={"schema":"h706.combined-current-exact-surface-diff-config.v1",
 "baseline_candidates":p["source_surfaces"],"candidate_surfaces":s["source_surfaces"],"changed_roles":["I"],
 "bounded_point_repair":{"I":{"maximum_displacement_m":50e-6}},
 "independent_point_update_replay":pin(batch)}
 cp=out/"context-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
 aud=H/"combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_806_extended_250um_construction.py"
 assert pin(aud)["sha256"]=="1c20de7dffa2cbf55abf195310af988aed9c5a87f035a8e79e099a3f00c74377"
 report=out/"proposal-exact-806/report.json";logs=out/"proposal-audit-logs";logs.mkdir()
 cmd=[sys.executable,str(aud),"--config",str(cp),"--out",str(report.parent)]
 with(logs/"audit806.stdout.log").open("w")as f,(logs/"audit806.stderr.log").open("w")as e:
  done=subprocess.run(cmd,stdout=f,stderr=e)
 (logs/"audit806.command.json").write_text(json.dumps({"command":cmd,"returncode":done.returncode},indent=2)+"\n")
 assert done.returncode==0,(done.returncode,(logs/"audit806.stderr.log").read_text()[-3000:])
 a=json.loads(report.read_text());print("AUDIT",a["status"],flush=True);assert a["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
 # Current source evidence replaces pending aliases only after the unchanged exact auditor passes.
 s.update(status="PASS_ZERO_SKIN_FAT_SOURCE_QUALITY_MASS_PENDING",surface_crossings=0,
 exact_context=pin(report),all185_775=pin(report),all185_806=pin(report),
 source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",
 source_crossing_audit_status="PASS_CURRENT_806_EXACT_SOURCE_CONTEXT",
 independent_batch_replay=pin(batch),quality_working_state_qualifier=pin(__file__),
 physical_admission=False,native_steps=0,mass_mapping="NOT_RUN",
 mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False})
 if "source_crossing_evidence_inherited_from_unchanged_surfaces"in s:
  s["historical_source_crossing_inheritance_before_fresh_1020"]=s.pop("source_crossing_evidence_inherited_from_unchanged_surfaces")
 s["source_crossing_evidence_mode"]="FRESH_CURRENT_POINT_MOVEMENT_AUDIT"
 dest=out/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
 ver=H/"verify_current_source_evidence_808_v26a.py"
 assert pin(ver)["sha256"]=="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9"
 spec=importlib.util.spec_from_file_location("source808_zero1020",ver);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
 pp=out/"source-proof-808.json";pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
 receipt={"status":"PASS_ZERO_SKIN_FAT_QUALITY_806_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),
 "point_update_replay":pin(batch),"exact_source_report":pin(report),"source_proof":pin(pp),
 "counts":{"Skin":0,"Fat":0},"surface_crossings":0,"physical_admission":False,"native_steps":0,
 "mass_mapping":"NOT_RUN","mass_reconciliation":"PENDING"}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)
 subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)
if __name__=="__main__":main()
