from pathlib import Path
import argparse,subprocess,json,hashlib,sys,importlib.util
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def pin(p):
 p=Path(p).resolve()
 return {"path":str(p),"sha256":hashlib.file_digest(p.open("rb"),"sha256").hexdigest()}
def check(ref):
 assert pin(ref["path"])==ref
 return Path(ref["path"])
ap=argparse.ArgumentParser()
ap.add_argument("--report",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
ap.add_argument("--skin",type=int,required=True);ap.add_argument("--fat",type=int,required=True)
args=ap.parse_args()
out=args.out.resolve();assert not out.exists()
batch=args.report.resolve();b=json.loads(batch.read_text())
assert b["status"]=="PASS_INDEPENDENT_REPLAY_PENDING_FRESH_775_808"
parent=check(b["parent_state"]);p=json.loads(parent.read_text())
assert p["source_audit_status"]=="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY"
assert b["physical_admission"] is False and b["native_steps"]==0
assert b["paired_exact_signed_delta_det6"]=="0"
rp=check(b["independent_replay"]);r=json.loads(rp.read_text())
assert r["status"]=="PASS_INDEPENDENT_BACKSKIN_CAP_REPLAY"
for key in ("source_topology_preserved","source_point_bytes_unchanged","full_typed_source_face_ancestry","paired_exact_volume_cancellation","full_global_target_chains","native_thresholds_unchanged"):
 assert r[key] is True,key
check(b["operation_journal"])
proposal=check(b["candidate_state"]);s=json.loads(proposal.read_text())
assert s["source_state"]==pin(parent)
for vr,expected in (("Skin",args.skin),("Fat",args.fat)):
 role=r["roles"][vr]
 for key in ("new_seed_cells_native_pass","retained_formerly_good_cells_nonregressing","full_target_chain","full_domain_native_pass"):
  assert role[key] is True,(vr,key)
 assert role["local_volume_fraction"]<=.01 and role["global_volume_fraction"]<=1e-4
 assert role["native_bad_after"]==expected and role["native_bad_after"]<=role["native_bad_before"]
 assert role["native_bad_before"]==p[vr+"_current"]["native_quality_failures"]
 assert s[vr+"_current"]["native_quality_failures"]==expected and s[vr+"_current"]["nonpositive"]==0
 s[vr]=dict(s[vr+"_current"])
assert args.skin+args.fat < sum(p[vr+"_current"]["native_quality_failures"]for vr in ("Skin","Fat"))
assert s["source_surfaces"]["O"]==p["source_surfaces"]["O"]
assert s["source_surfaces"]["D"]==p["source_surfaces"]["D"]
proofpath=H/"all-quality-992-backskin-cap-candidate-root/independent-3d-distance-replay-994.json"
assert pin(proofpath)["sha256"]=="ec10bce966ab1c13ee9527587c18bdc2d89a3d1d1d1d00bcbd51db49740e8e05"
owner=H/"quality-partial-cavity-684-root/verify_backskin_3d_distance_994_root.py"
assert pin(owner)["sha256"]=="cab873e238cc3e4942c7f9ba4c49679a09f77de3f4513b5c107ba736cef727b3"
d=json.loads(proofpath.read_text())
assert d["status"]=="PASS" and d["source_point_arrays_bitwise_identical"] is True
assert d["method"]["cap_m"]==50e-6
assert all(d["method"]["directional_results"][x]["upper_bound_m"]<=50e-6 for x in ("old_to_new","new_to_old"))
with np.load(check(p["source_surfaces"]["I"]),allow_pickle=False)as z:
 pfaces=np.asarray(z["faces"],np.int64)
 points32=np.asarray(z["points_world_m_f32"],np.float32);points64=np.asarray(z["points_world_m"],np.float64)
with np.load(check(s["source_surfaces"]["I"]),allow_pickle=False)as z:
 sfaces=np.asarray(z["faces"],np.int64)
 assert np.array_equal(points32.view(np.uint32),np.asarray(z["points_world_m_f32"],np.float32).view(np.uint32))
 assert np.array_equal(points64.view(np.uint64),np.asarray(z["points_world_m"],np.float64).view(np.uint64))
assert hashlib.sha256(points32.tobytes()).hexdigest()==d["points_world_m_f32_sha256"]
assert hashlib.sha256(points64.tobytes()).hexdigest()==d["points_world_m_f64_sha256"]
rows=np.asarray(d["face_rows"],np.int64)
assert np.array_equal(np.flatnonzero(np.any(pfaces!=sfaces,axis=1)),rows)
assert np.array_equal(pfaces[rows],np.asarray(d["old_faces"],np.int64))
assert np.array_equal(sfaces[rows],np.asarray(d["new_faces"],np.int64))
out.mkdir()
cfg={"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":p["source_surfaces"],"candidate_surfaces":s["source_surfaces"],"changed_roles":["I"],"fixed_coordinate_topology_only":{"declared":True,"edited_role":"I","all_point_bytes_fixed":True,"maximum_inner_back_surface_hausdorff_m":50e-6,"endpoint_quotient_maximum_surface_distance_m":50e-6,"independent_replay":pin(batch),"operation_journal":b["operation_journal"],"independent_3d_distance_replay":pin(proofpath)}}
cp=out/"context-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
aud=H/"combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
assert pin(aud)["sha256"]=="82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
report=out/"proposal-exact-775/report.json";logs=out/"proposal-audit-logs";logs.mkdir()
cmd=[sys.executable,str(aud),"--config",str(cp),"--out",str(report.parent)]
with(logs/"audit775.stdout.log").open("w")as f,(logs/"audit775.stderr.log").open("w")as e:
 result=subprocess.run(cmd,stdout=f,stderr=e)
(logs/"audit775.command.json").write_text(json.dumps({"command":cmd,"returncode":result.returncode},indent=2)+"\n")
assert result.returncode==0,(result.returncode,(logs/"audit775.stderr.log").read_text()[-2000:])
a=json.loads(report.read_text());print("AUDIT",a["status"],flush=True);assert a["status"].startswith("PASS_")
s.update(status="PASS_SOURCE_QUALIFIED_RETAINED_CELL_TRANSFER_MASS_PENDING",surface_crossings=0,exact_context=pin(report),all185_775=pin(report),source_audit_status="PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",source_crossing_audit_status="PASS_CURRENT_775_EXACT_SOURCE_CONTEXT",independent_batch_replay=pin(batch),independent_3d_distance_replay=pin(proofpath),physical_admission=False,native_steps=0,mass_reconciliation={"status":"PENDING_VOLUME_CHANGE_RECONCILIATION","mass_conservation_claimed":False},quality_working_state_qualifier=pin(__file__))
dest=out/"verified-state.json";dest.write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
ver=H/"verify_current_source_evidence_808_v26a.py"
assert pin(ver)["sha256"]=="8c7f6e78218cb7f3db62d657ea23dd074d77cc58bd5514362678047549ac74a9"
spec=importlib.util.spec_from_file_location("source808_back1004",ver);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
proof=m.verify_current_source_evidence(dest);assert proof["status"].startswith("PASS_")
pp=out/"source-proof-808.json";pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
receipt={"status":"PASS_RETAINED_BACKSKIN_TRANSFER_775_808_SOURCE_ONLY","state":pin(dest),"parent":pin(parent),"batch_report":pin(batch),"operation_journal":b["operation_journal"],"exact_source_report":pin(report),"source_proof":pin(pp),"independent_3d_distance_replay":pin(proofpath),"distance_cap_m":50e-6,"counts":{vr:s[vr+"_current"]["native_quality_failures"]for vr in ("Skin","Fat")},"surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_reconciliation":"PENDING"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)
subprocess.run([sys.executable,str(H/"select_verified_quality_checkpoint_root.py"),str(dest),pin(dest)["sha256"]],check=True)
