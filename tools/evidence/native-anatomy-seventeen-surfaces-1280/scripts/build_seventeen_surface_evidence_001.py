from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"seventeen-surface-native-composition-002"
repo=Path("/Users/n/numi-human-positive-winding-1279");O=repo/"tools/evidence/native-anatomy-seventeen-surfaces-1280"
checks=json.loads((N/"native-verification-001/report.json").read_text());interfaces=json.loads((N/"native-interface-verification-001/report.json").read_text())
assert checks["complete"] and checks["inputs_unchanged"] and checks["all_native_repaired_surface_checks_pass"] and checks["all_coupled_trace_rows_exact_baseline_prefix"]
assert interfaces["complete"] and interfaces["inputs_unchanged"] and len(interfaces["poses"])==8
assert all(len(p["rows"])==17 and all(r["self_count"]==r["new_triangle_pair_count"]==0 and r["external_targets_scanned"]==859 for r in p["rows"]) for p in interfaces["poses"])
O.mkdir(exist_ok=False);inventory={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(src,relative,compress=False):
 src=Path(src);dest=O/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 raw=src.read_bytes();dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
 inventory[relative]={"bytes":dest.stat().st_size,"encoding":"gzip" if compress else "identity","retained_path":str(src),"retained_sha256":hashlib.sha256(raw).hexdigest(),"sha256":sha(dest)}
for f in ["launch_arm.py","launch-guard.json","composition-result.json","source-resolution-bounds.json","material-change-report.json"]:put(N/f,"native/"+f)
for f in ["run-declaration.json","execution-start.json","execution.json"]:put(N/"baseline"/f,"native/baseline/"+f)
for f in ["invocation.json","run-metadata.json","resting-coupled.csv"]:put(N/"baseline/native-run"/f,"native/baseline/native-run/"+f)
for f in ["frame-skin.png","frame-muscles.png","frame-skeleton.png","frame-organs.png"]:put(N/"baseline/native-run"/f,"viewer/"+f)
movies=list((N/"baseline/native-run").glob("*.mov"));assert len(movies)==1
put(movies[0],"viewer/"+movies[0].name)
put(A/"seventeen-surface-movie-inspection-001.log","viewer/movie-inspection.log")
for path,name in [(N/"native-verification-001/report.json","native-self-and-trace.json"),(N/"native-interface-verification-001/report.json","native-interfaces.json"),(A/"parent-positive-allpose-interface-audit-001/report.json","heldout-parent-union-interfaces.json")]:put(path,"checks/"+name)
put(A/"parent-positive-union-lift-001/report.json","derivation/parent-lift-report.json.gz",True)
put(A/"bounded-halfmm-source-resolution-001/report.json","derivation/bounded-source-resolution.json.gz",True)
for sid in [105,106,124]:
 up=A/f"parent-positive-union-{sid}-001"
 put(up/"result.json",f"derivation/stable-{sid}/union-result.json")
 put(up/f"stable-{sid}-exact-union.json",f"derivation/stable-{sid}/exact-union.json.gz",True)
 put(A/"bounded-halfmm-source-resolution-001"/f"stable-{sid}-reference-resolution-row-patch.npz",f"derivation/stable-{sid}/collapsed-parent.npz")
 put(A/"parent-positive-union-lift-001"/f"stable-{sid}-reference-union-row-patch.npz",f"patches/stable-{sid}.npz")
for f in ["prepare_bounded_halfmm_source_resolution_001.py","prepare_parent_positive_union_row_001.py","prepare_parent_positive_union_lift_001.py","audit_parent_positive_allpose_interfaces_001.py","prepare_seventeen_surface_native_002.py","verify_seventeen_surface_native_001.py","verify_seventeen_surface_native_interfaces_001.py"]:put(A/f,"scripts/"+f)
put(A/"seventeen-surface-native-preparation-001.log","rejected/seventeen-preparation-001.log")
put(A/"prepare_seventeen_surface_native_001.py","rejected/prepare_seventeen_surface_native_001.py")
for src,rel in [(A/"sixteen-surface-native-composition-001/native-interface-verification-001/report.json","sixteen-native-interface-failure.json"),(A/"sixteen-surface-pronator-classification-001.log","sixteen-classification-failure.log"),(A/"sixteen-surface-native-composition-001/baseline/execution.json","sixteen-execution.json"),(A/"sixteen-surface-native-composition-001/baseline/run-declaration.json","sixteen-run-declaration.json")]:put(src,"rejected/"+rel)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=json.loads((N/"baseline/native-run/run-metadata.json").read_text());trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"stable_ids":[35,36,55,56,57,58,105,106,107,108,115,116,124,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro, 24 GB, macOS26.6","source_revisions":{"Numi2/numilab-human":"a84aa0e5dca2761d6cc101680e582d3aa497c07b","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"code_revision_equivalence":{"human_positive_winding":"3493b44bf325a3106fed0d0377a15a43be748180","lab_native_build":"015ff2ebd7756247bb511319c8ef21295ba37273"}}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
previous=repo/"tools/evidence/native-anatomy-fourteen-surfaces-1279/reproduce.py"
(O/"reproduce.py").write_text(previous.read_text().replace("fourteen-surface","seventeen-surface"))
(O/"README.md").write_text("""# Seventeen repaired surfaces in the integrated native human

The short heads of both biceps brachii (105/106) and left palmaris longus (124) now join the [previous fourteen repaired surfaces](../native-anatomy-fourteen-surfaces-1279/README.md) in the existing native resting human. Mechanics, Brain respiratory drive, cardiac pumping, circulation and gas exchange remain coupled in the same Metal runtime on the SSH Mac mini.

**Whole-body anatomy and the final five-minute qualification remain incomplete.** Performance optimization is paused. This is a tested anatomy increment, not a full-deliverable claim.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py

The retained Mini runtime and asset paths are required. --prepare-only verifies all bound inputs and prepares a fresh run. The launch declaration retains exact command, configuration, assets and source bytes. Source revisions and measured timing are in summary.json.

## Reference reconstruction

These three passive reference surfaces use the existing bounded edge-collapse owner followed by the explicit positive-winding boundary reconstruction and parent-face attribute lift. Every collapsed parent is replayed against its original source, and every union vertex binds to that direct parent's coordinates and face ancestry. Parent coordinates are not mislabeled as unchanged source measurements. Physical routes, body binding tables, unrelated surface rows and physical mass are preserved.

There are no pure-weight attachment proxy vertices on these three surfaces; preservation of clinical attachment footprints is not claimed. Algebraic signed surface volume changes by approximately -0.916 ppm for each biceps surface and -39 ppm for left palmaris. These are geometric diagnostics, not tissue mass measurements. Binding inference and exact cut-point rounding are recorded.

## Native checks

At eight native captures, all seventeen repaired surfaces pass exact self-intersection and closed oriented manifold checks. Each changed region adds zero source-lineage crossings against all 859 surrounding targets. All 1,000 rows and 60 coupled physiology/mechanics columns exactly match the unchanged baseline011 prefix. The run accepts 8,000 steps, approximately 16 simulated seconds.

The continuous native framebuffer movie, seven-layer inspection log and four inspection frames are retained. This is framebuffer evidence, not a separate claim of remote desktop visibility. The existing support plane and force/state owners are unchanged.

The excluded sixteen-surface attempt is retained in rejected/: its pronator repair passed self checks but changed additional crossing lineage against flexor pollicis longus during early settling, outside its declared shared-head interface. It was not admitted. The first seventeen-surface preparation failed on a missing script import before execution; the corrected preparation is the executed one.

Original BodyParts3D and MyoSim source rights and mixed-source reference status remain as recorded by the prior scene and asset receipts. No mixed-source anatomy is represented as measurements of one person.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw
 assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
