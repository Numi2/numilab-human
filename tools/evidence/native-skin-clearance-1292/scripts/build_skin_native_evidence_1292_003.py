from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"skin-native-integration-1292-001";S=A/"skin-candidate-30-pose-preparation-001/result-005";O=A/"skin-native-publication-1292-003"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
r=read(N/"native-verification-003/report.json");assert r["complete"] and r["inputs_unchanged"] and r["all_native_skin_exact_checks_pass"] and r["all_coupled_trace_rows_exact_parent"]
O.mkdir(exist_ok=False);inventory={}
def put(src,rel,compress=False):
 src=Path(src);dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes();dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
 inventory[rel]={"bytes":dest.stat().st_size,"encoding":"gzip" if compress else "identity","retained_path":str(src),"retained_sha256":hashlib.sha256(raw).hexdigest(),"sha256":sha(dest)}
for f in ("launch_arm.py","launch-guard.json"):put(N/f,"native/"+f)
for f in ("run-declaration.json","execution-start.json","execution.json"):put(N/"baseline"/f,"native/baseline/"+f)
for f in ("invocation.json","run-metadata.json","resting-coupled.csv"):put(N/"baseline/native-run"/f,"native/baseline/native-run/"+f)
for f in ("frame-skin.png","frame-muscles.png","frame-skeleton.png","frame-organs.png","native-viewer.mov"):put(N/"baseline/native-run"/f,"viewer/"+f)
put(N/"movie-inspection.log","viewer/movie-inspection.log")
for v in ("001","002","003"):put(N/f"native-verification-{v}/report.json",f"checks/native-verification-{v}.json.gz",True)
put(N/"native-forward-diagnostic-002/report.json","checks/native-forward-rounding.json")
for p in sorted((N/"native-verification-002").glob("step-*-exact-audits.json.gz")):put(p,"checks/"+p.name)
for f in ("preparation-report.json","target-refresh-report.json"):
 p=S/f
 if p.exists():put(p,"derivation/"+f+".gz",True)
put(S/"candidate-target-pair-tables.json.gz","derivation/candidate-target-pair-tables.json.gz")
fit=A/"stable152-skin-fit-001/result-014-resume-004"
put(fit/"candidate-report.json","derivation/skin-fit-report.json.gz",True)
put(fit/"candidate-source-positions-f32.npy","assets/candidate-source-positions-f32.npy.gz",True)
for p in [A/"prepare_skin_native_integration_1292_001.py",A/"verify_skin_native_integration_1292_002.py",A/"finalize_skin_native_verification_1292_003.py",A/"diagnose_skin_native_forward_1292_002.py",A/"skin-candidate-30-pose-preparation-001/prepare_skin_candidate_30pose_005.py",A/"stable152-skin-fit-001/fit_skin_outward_017_resume.py",Path(__file__)]:put(p,"scripts/"+p.name)
runtime=read(N/"baseline/native-run/run-metadata.json");execution=read(N/"baseline/execution.json")
trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()));sim=float(trace[-1][next(iter(trace[-1]))])
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":runtime["wall_seconds"],"launch_wall_seconds":execution["wall_seconds"],"real_time_factor":sim/runtime["wall_seconds"],"performance_optimization_paused":True,"skin_sha256":sha(S/"bodyparts3d-myosim-skinned-shell.nhskin"),"maximum_source_offset_m":0.0027802606,"exact_native_self_pairs":0,"exact_native_target_pairs":0,"rendered_targets_per_pose":859,"native_poses_checked":8,"trace_rows_exact_parent":1000,"trace_columns_exact_parent":60,"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"}}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-touching-loop-arrangement-1290/tools/evidence/native-anatomy-thirty-surfaces-1291/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("thirty-surface","skin-clearance"))
(O/"README.md").write_text("""# Native whole-body skin clearance increment

The existing native resting scene now uses an inferred skin correction that clears all 859 surrounding anatomical structures at eight captured accepted states. The scene includes the preceding 30 muscle-surface repairs. **This is a verified 16-second increment, not completion of whole-body anatomy or final five-minute qualification.**

The maximum source displacement is 2.781 mm. All 32 bed-contact witnesses, 115 intentional ocular-boundary vertices, skin topology and anatomical binding weights remain exactly fixed. This is a bounded reference correction, not measured-person anatomy or a claim of millimetre anatomical accuracy. Existing source provenance and rights remain bound in the anatomy receipt.

## Exact native checks

At every captured pose the rendered skin has zero exact self-intersection pairs and zero intersections with all 859 surrounding structures. Two prospective sternocleidomastoid surfaces were checked offline but are not included in this native scene.

The initial attempt to transfer the offline proof by byte identity failed: native Float32 arithmetic produced coordinate differences up to 0.300 micrometres. That failed report is retained. The successful check instead rechecks every bitwise-changed face with the existing exact predicates against the complete skin and target geometry; unchanged-face witnesses are reused only with their input hashes bound. No positional tolerance admits crossings.

All other anatomy, all 157 accepted body poses, and every value in the 1,000-row, 60-column coupled trace match the preceding native run exactly. Physical routes, contact, masses and physiology are unchanged. The continuous native framebuffer recording contains 252 frames and the seven anatomical inspection layers.

## Reproduce

    /usr/bin/python3 tools/evidence/native-skin-clearance-1292/reproduce.py

The pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify all launch inputs without execution. The included derivation and verification scripts retain absolute source/evidence bindings; they are not a portable asset installer.

Neck/back integration, remaining muscle and foot defects, and unresolved internal interfaces remain open. Skin clearance at sampled states does not establish every internal anatomical relationship or every time instant. The older five-minute baseline/intervention evidence uses older anatomy and does not qualify this updated body. Performance work is deferred; achieved timing is recorded without claiming real-time execution.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 assert hashlib.sha256(gzip.decompress(raw) if x["encoding"]=="gzip" else raw).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
