from pathlib import Path
import sys,json,hashlib,importlib.util,csv,time
import numpy as np
import gzip
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");C=A/"skin-native-integration-1292-001";P=A/"thirty-surface-native-composition-001";S=A/"skin-candidate-30-pose-preparation-001/result-005";O=C/"native-verification-002";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca
H=A/"skin-candidate-30-pose-preparation-001/prepare_skin_candidate_30pose_005.py";sp=importlib.util.spec_from_file_location("prep",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
N=C/"baseline/native-run";B=P/"baseline/native-run";rp=S/"preparation-report.json";r=json.loads(rp.read_text())
assert r["all_declared_inputs_unchanged"]
for row in r["input_postcheck"]:assert sha(row["path"])==row["sha256"]
execution=json.loads((C/"baseline/execution.json").read_text());assert execution["returncode"]==0 and not execution["changed_inputs"] and execution["native_argv_matches_prepared_cli_preview"] and execution["native_environment_matches_prepared_cli_preview"]
preview=S/"run-declaration-preview.json"
pins={str(p):sha(p) for p in [Path(__file__),H,Path(ca.__file__),rp,preview,C/"baseline/execution.json",N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv"]}
pins.update(json.loads(preview.read_text())["immutable_assets"])
_,skin=h.load_skin(Path(r["candidate_skin"]["path"]));ref=np.unique(skin["faces"]);compact=np.searchsorted(ref,skin["faces"]);assert len(ref)==54663
keys,*_=ca._load_target_inventory(h.INV);assert len(keys)==859
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()));assert len(trace)==len(base)==1000
tables_path=Path(r["target_refresh"]["candidate_tables_path"]);pins[str(tables_path)]=sha(tables_path)
with gzip.open(tables_path,"rt") as stream:tables=json.load(stream)
assert len(tables)==8 and all(len(table)==861 and all(x["count"]==0 for x in table.values()) for table in tables)
pins[str(h.FIT_EVENTS)]=sha(h.FIT_EVENTS);events=[json.loads(x) for x in h.FIT_EVENTS.read_text().splitlines()];fit_event=next(x for x in reversed(events) if x.get("status")=="accepted")
assert len(fit_event["self_audits_by_pose"])==8
out={"scope":"Native16s skin integration on30 repaired muscle scene, no newSCM rows. Fresh exact incremental self and859-rendered-target checks from hash-bound prior full self/target proof; every native F32-changed skin face is rechecked.2SCM targets remain offline prospective anatomy. Complete unchanged geometry and coupled-state comparison. Not whole-body or finalfive-minute qualification.","pins":pins,"trace_rows":len(trace),"trace_columns":len(trace[0]),"all_coupled_trace_rows_exact_parent":trace==base,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
for i,step in enumerate((0,4767,5023,5599,6207,6815,7423,8000)):
 pack=N/"accepted-geometry"/f"step-{step}.mrvpack";bp=B/"accepted-geometry"/f"step-{step}.mrvpack"
 receipts=[]
 for p in (pack,bp):
  rr=p.with_suffix(".receipt.json");pins[str(p)]=sha(p);pins[str(rr)]=sha(rr);rc=json.loads(rr.read_text());assert rc["pack_file_sha256"]==pins[str(p)] and rc["physical_endpoint"]=="accepted";receipts.append(rc)
 rc,brc=receipts
 assert rc["accepted_body_poses"]==brc["accepted_body_poses"] and len(rc["accepted_body_poses"])==157
 assert rc["accepted_registered_body_poses"]==brc["accepted_registered_body_poses"]
 assert rc["accepted_respiratory_motion"]==brc["accepted_respiratory_motion"]
 assert rc["initial_anatomical_registration"]==brc["initial_anatomical_registration"]
 xyz,surfs,_=ca._pack_surfaces(pack,keys);bx,bs,_=ca._pack_surfaces(bp,keys);assert set(surfs)==set(bs)==set(keys)|{(51007,1)}
 for key in sorted(keys):
  f=np.asarray(surfs[key]["faces"],int);bf=np.asarray(bs[key]["faces"],int);assert f.shape==bf.shape and np.array_equal(xyz[f],bx[bf]),(step,key)
 sf=np.asarray(surfs[(51007,1)]["faces"],int);offset=int(sf.min());assert np.array_equal(sf-offset,skin["faces"])
 world=xyz[offset:offset+skin["nv"]][ref];digest=ca._float32_xyz_sha256(world);expected=r["captures"]["candidate_world_f32_sha256_by_pose"][i]
 pr={"step":step,"unchanged_anatomical_surfaces_exact_parent":859,"complete157_body_poses_exact_parent":True,"respiratory_state_exact_parent":True,"native_skin_world_f32_sha256":digest,"expected_proved_skin_world_f32_sha256":expected,"native_skin_exact_proved_world":digest==expected,"skin_face_index_sha256":ca._face_index_sha256(compact),"rendered_target_count":859}
 out["poses"].append(pr);save()
 dp=C/"native-forward-diagnostic-002"/f"step-{step}.npz";pins[str(dp)]=sha(dp);zz=np.load(dp)
 pred=zz["expected"];assert np.array_equal(zz["actual"],world) and ca._float32_xyz_sha256(pred)==expected
 baseline_self=fit_event["self_audits_by_pose"][i];assert baseline_self["count"]==0 and baseline_self["candidate_world_f32_sha256"]==expected and baseline_self["face_index_sha256"]==ca._face_index_sha256(compact)
 sa=ca.audit_incremental_skin_self_intersections(baseline_world_positions=pred,candidate_world_positions=world,baseline_faces=compact,candidate_faces=compact,baseline_self_audit=baseline_self,expected_baseline_world_f32_sha256=expected,expected_face_index_sha256=ca._face_index_sha256(compact),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(baseline_self,len(compact)))
 tfs={key:np.asarray(surfs[key]["faces"],int) for key in keys};btfs={key:np.asarray(bs[key]["faces"],int) for key in keys}
 target_sha=ca._target_geometry_f32_sha256(xyz,tfs);assert target_sha==ca._target_geometry_f32_sha256(bx,btfs)
 bt={f"{key[0]}:{key[1]}":tables[i][f"{key[0]}:{key[1]}"] for key in keys}
 ta=ca.audit_incremental_skin_target_intersections(baseline_world_positions=pred,candidate_world_positions=world,baseline_faces=compact,candidate_faces=compact,baseline_target_audits=bt,target_faces_by_key=tfs,target_positions=xyz,expected_baseline_world_f32_sha256=expected,expected_face_index_sha256=ca._face_index_sha256(compact),expected_target_geometry_f32_sha256=target_sha,expected_baseline_target_audits_sha256=ca._baseline_target_pair_table_sha256(bt))
 with gzip.open(O/f"step-{step}-exact-audits.json.gz","wt") as stream:json.dump({"self":sa,"targets":ta},stream,sort_keys=True)
 pr.update(self_audit=sa,target_audit_summary={k:v for k,v in ta.items() if k not in ("audits_by_target","target_audits","triangle_pairs")},maximum_forward_difference_m=float(np.linalg.norm(world.astype(float)-pred,axis=1).max()),proof_transfer_basis="Unchanged faces retain pinned exact pair witnesses; every bitwise-changed native skin face is exactly checked against fullskin and all859 rendered targets; no positional tolerance.")
 save();print(json.dumps({"step":step,"self":sa.get("count"),"target_result_keys":list(ta),"forward_max_m":pr["maximum_forward_difference_m"]}),flush=True)
 assert sa["count"]==0

out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()),all_native_skin_exact_checks_pass=all(p["self_audit"]["count"]==0 for p in out["poses"]),all_prior30_geometry_exact=True)
save();assert out["inputs_unchanged"] and out["all_coupled_trace_rows_exact_parent"] and out["all_native_skin_exact_checks_pass"]
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}),flush=True)

