from pathlib import Path
import json,hashlib,gzip,csv,subprocess
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"right-scm-native-composition-001";O=A/"right-scm-native-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_checks_pass"] and d["coupled_trace_exact_parent"]
assert len(d["poses"])==8
for p in d["poses"]:
 assert p["self_count"]==0 and p["external_targets_scanned"]==860 and p["skin_ica_ijv_clear"] and p["external_total_matches_offline_proof"]
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==860 and p["complete157_body_poses_exact_parent"]
ex=read(N/"baseline/execution.json"); assert ex["returncode"]==0 and not ex["changed_inputs"] and ex["native_argv_matches_prepared_cli_preview"] and ex["native_environment_matches_prepared_cli_preview"]
movie=(N/"movie-inspection.log").read_text(); assert "frames=252 timing_markers=5" in movie and all("frame="+k+" " in movie for k in ("skin","muscles","skeleton","organs","lungs","heart","vessels"))
audit=list(csv.DictReader((N/"baseline/native-run/resting-surface-audit.csv").open()))
assert int(audit[-1]["step"])==8000 and all(float(r[k])==0 for r in audit for k in ("mesh_zero_area_triangles","mesh_nonfinite_area_triangles","vertices_below_1mm","nonfinite_skin_vertices"))
O.mkdir(exist_ok=False);inventory={}
def put(src,rel,compress=False):
 src=Path(src);dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes()
 if not compress and src.suffix==".py" and raw.endswith(b"\n\n"):compress=True;rel+=".gz";dest=O/rel
 dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
 inventory[rel]={"bytes":dest.stat().st_size,"encoding":"gzip" if compress else "identity","retained_path":str(src),"retained_sha256":hashlib.sha256(raw).hexdigest(),"sha256":sha(dest)}
for f in ("launch_arm.py","launch-guard.json","composition-result.json"):put(N/f,"native/"+f)
for f in ("source-reconstruction-proof.json","source-resolution-bounds.json"):put(N/f,"native/"+f+".gz",True)
for f in ("run-declaration.json","execution-start.json","execution.json"):put(N/"baseline"/f,"native/baseline/"+f)
for f in ("invocation.json","run-metadata.json","resting-coupled.csv"):put(N/"baseline/native-run"/f,"native/baseline/native-run/"+f)
put(N/"baseline/native-run/resting-surface-audit.csv","checks/resting-surface-audit.csv.gz",True)
for f in ("frame-skin.png","frame-muscles.png","frame-skeleton.png","frame-organs.png","frame-lungs.png","frame-heart.png","frame-vessels.png","native-viewer.mov"):put(N/"baseline/native-run"/f,"viewer/"+f)
put(N/"movie-inspection.log","viewer/movie-inspection.log");put(N/"native-verification-001/report.json","checks/native-verification.json")
put(N/"right-scm-row-patch.npz","patches/stable-151.npz")
RP=A/"scm-vessel-clearance-001/run-072-scm-ica-offset-source-lift-001/result-001/report.json"
put(RP,"derivation/source-and-eight-poses.json.gz",True)
put(RP.parent/"unadmitted-scm-minus-ica-ijv-source-lift.npz","patches/source-lift.npz")
for p in read(RP)["input_pins"]:
 p=Path(p)
 if str(p).startswith(str(A)) and p.suffix in (".py",".json",".npz") and "accepted-geometry" not in p.parts:
  rel="derivation/"+str(p.relative_to(A));put(p,rel+(".gz" if p.suffix==".json" else ""),p.suffix==".json")
for f in ("prepare_right_scm_native_001.py","verify_right_scm_native_001.py"):
 put(A/f,"scripts/"+f)
runner=A/"scm-vessel-clearance-001/run-072-scm-ica-offset-source-lift-001.py";put(runner,"scripts/"+runner.name)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json")
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":16.0,"owner_wall_seconds":runtime["wall_seconds"],"real_time_factor":16/runtime["wall_seconds"],"performance_optimization_paused":True,"new_passive_stable_id":151,"label":"right sternocleidomastoid","member_id":"FJ1595","candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"fifty-seven-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"source_revisions":{"Numi2/numilab-human":"6a5596c2038ea98f56bb63b2994b04a98452c9b4","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"device":"SSH macmini, Apple M4 Pro, 24GB, macOS26.6","native_pose_count":8,"native_prior_surfaces_exact":860,"physiology_rows_exact":1000,"physiology_columns_exact":60,"continuously_audited_steps":8000,"audit_sample_rows":len(audit),"movie_frames":252,"movie_timing_markers":5,"parent_qualification":"../native-anatomy-fifty-seven-surfaces-1325/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","right-scm"))
(O/"README.md").write_text("""# Right sternocleidomastoid in the native resting human

The native whole-body scene now includes a source-derived passive right sternocleidomastoid (BodyParts3D FJ1595, stable151), beside the previous 57 repaired surfaces and corrected skin. This is a verified incremental scene, not completed whole-body anatomy or final five-minute qualification. Performance work remains deferred.

The existing append/composition owners preserve all prior rows, physical mass, MyoSim routes and force ownership. The SCM is inspection geometry bound to the accepted body state, not an independently simulated neck actuator. Its source identity and upstream rights remain in the pinned source manifests. Mixed-source reference anatomy is not a measured individual.

The reconstruction uses a declared local reference shape followed by source-derived Boolean clearance from 0.10 mm expanded carotid and jugular shells. That offset is an engineering construction, not a measured tissue gap. The final source has 5,309 vertices and 10,622 faces, one closed oriented component and no exact self-intersections. It retains 5,119 original vertices of the local reference and all 1,561 one-hot attachment proxies exactly. There is no certified attachment footprint. Generated cap vertices can be several millimetres from source support; the derivation records those offsets and the 0.7693 mL source volume removed relative to that local reference.

Eight retained parent poses and eight new actual native captures show no SCM self-intersections or crossings with the carotid, jugular or skin. Every native pose checks all 860 surrounding surfaces and matches the declared candidate's interface pairs. Existing pairs with the manubrium, right temporal, occipital, right clavicle and right pectoralis remain reported. Their attachment interpretation is not a general anatomical-interface exemption or final interface qualification.

All 860 prior rendered surfaces, all 157 accepted body poses and all 1,000 rows / 60 columns of coupled physiology match the parent exactly. The native invalid-triangle and bed-contact audit reaches step8,000, and a 252-frame continuous recording includes all seven inspection layers. Exact self/inter-tissue geometry checks use the eight exported poses, not every physical step.

The run covers 16 simulated seconds in 161.03 seconds on the SSH Mac mini. It adds no evidence of five-minute stability for this final anatomy. Left neck, head and back coverage, eight original defective rows, and remaining interface qualification continue to block the full deliverable.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-right-scm-1326/reproduce.py

The pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify inputs without starting the GPU. The run declaration retains exact revisions, asset hashes, configuration and launch command; native timing, trace, verification and recording are included. Preparation reports saying native verification was pending are immutable historical records; checks/native-verification.json is the subsequent result.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 assert hashlib.sha256(gzip.decompress(raw) if x["encoding"]=="gzip" else raw).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"files":len(inventory),"summary":summary},indent=2))
