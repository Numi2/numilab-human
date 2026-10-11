from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"thirty-four-surface-native-composition-002";O=A/"thirty-four-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior30_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==856 and len(p["rows"])==4
 for r in p["rows"]:assert r["self_count"]==r["new_triangle_pair_count"]==0 and r["external_targets_scanned"]==859
O.mkdir(exist_ok=False);inventory={}
def put(src,rel,compress=False):
 src=Path(src);dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes()
 dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
 inventory[rel]={"bytes":dest.stat().st_size,"encoding":"gzip" if compress else "identity","retained_path":str(src),"retained_sha256":hashlib.sha256(raw).hexdigest(),"sha256":sha(dest)}
for f in ["launch_arm.py","launch-guard.json","composition-result.json","source-resolution-bounds.json"]:put(N/f,"native/"+f)
for f in ["run-declaration.json","execution-start.json","execution.json"]:put(N/"baseline"/f,"native/baseline/"+f)
for f in ["invocation.json","run-metadata.json","resting-coupled.csv"]:put(N/"baseline/native-run"/f,"native/baseline/native-run/"+f)
for f in ["frame-skin.png","frame-muscles.png","frame-skeleton.png","frame-organs.png","native-viewer.mov"]:put(N/"baseline/native-run"/f,"viewer/"+f)
put(N/"movie-inspection-002.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
rp=A/"sartorius-tibialis-reference-selection-011/report.json";put(rp,"derivation/selected-reference.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for sid in (49,50,59,60):
 for fn in ("rejected-boundary-witness.json","result.json"):
  put(A/f"existing-body-{sid}-rejected-boundary-004"/fn,f"derivation/{sid}-{fn}.gz",True)
 put(A/f"existing-body-{sid}-pinched-reference-001/report.json",f"derivation/{sid}-junction-report.json.gz",True)
 put(A/f"existing-body-{sid}-pinched-reference-001/parameterized_existing_lift.py",f"scripts/parameterized_lift_{sid}.py")
for name in ("sartorius-small-feature-collapse-005","sartorius-tibialis-posed-local-reference-007","sartorius-tibialis-selected-pose-audit-008","right-sartorius-directed-local-reference-009","right-sartorius-selected-pose-audit-010","tibialis-posterior-local-reference-002","sartorius-tibialis-allpose-interface-audit-012"):
 put(A/name/"report.json","derivation/"+name+".json.gz",True)
for f in ["capture_rejected_winding_boundary_004.py",*[f"prepare_existing_body_{sid}_pinched_reference_001.py" for sid in (49,50,59,60)],"prepare_tibialis_posterior_local_reference_002.py","prepare_sartorius_small_feature_collapse_005.py","try_sartorius_tibialis_posed_local_reference_007.py","audit_sartorius_tibialis_selected_poses_008.py","try_right_sartorius_directed_local_reference_009.py","audit_right_sartorius_selected_poses_010.py","prepare_sartorius_tibialis_reference_selection_011.py","audit_sartorius_tibialis_allpose_interfaces_012.py","prepare_thirty_four_surface_native_002.py","verify_thirty_four_surface_native_002.py"]:
 put(A/f,"scripts/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,75,76,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"thirty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-skin-clearance-1292/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","thirty-four-surface"))
(O/"README.md").write_text("""# Thirty-four repaired surfaces in the native resting human

Bilateral sartorius and tibialis posterior repairs now operate in the existing full-body Metal scene with the [verified skin correction](../native-skin-clearance-1292/README.md). **Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.**

## Reference corrections

Existing exact positive-winding boundary extraction removes folded exterior regions while retaining source-face ancestry for anatomical bindings. Rejected point-junction boundaries and unsuccessful source/posed trials remain retained.

The sartorius correction separates finite junctions, collapses sub-0.1-micrometre source features that fail in Float32 poses, and moves one right-side vertex by 100.002 micrometres along its source Y axis. Bilateral tibialis posterior corrections separate point junctions, resolve residual source crossings with two one-micrometre local edits per side, and move six posed-fold vertices by at most 62.173 micrometres through a bounded local smoothing proposal. These are explicit reference inferences, not measured anatomy or source accuracy bounds. Reported bounds apply to the named local stages, not the complete positive-winding reconstruction.

Original exactly one-hot source-point proxies are retained where present: sartorius has none under this proxy definition; tibialis posterior retains seven per side. That proxy check is not measured attachment fidelity. Binding tables, physical muscle routes, mass, force ownership and physiology remain unchanged.

## Verification

Each candidate is closed, nondegenerate and self-intersection-free at source and all 24 retained early/late poses. Changed regions add no source-lineage crossings against all 859 surrounding native structures. The native 16-second run then checks the actual GPU-deformed surfaces at eight accepted captures, using actual preceding native coordinates for the comparison. All four rows remain closed and self-intersection-free, with no new changed-region crossings. Existing internal interfaces remain unqualified, including right sartorius versus rectus femoris and tibialis posterior versus nearby bone/vessel surfaces.

All 856 other rendered surfaces, including skin, and all 157 accepted body poses match the preceding run exactly. All 1,000 rows and 60 columns of the coupled trace match exactly. This is geometry integration; it does not add physiological fidelity.

The retained native framebuffer recording contains 252 frames, five timing markers, and seven anatomical inspection layers on the SSH Mac mini. Launch inputs, source revisions, exact assets, traces and achieved timing are bound in this package. No remote desktop visibility is asserted.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-thirty-four-surfaces-1293/reproduce.py

The pinned Mini runtime and retained assets are required. --prepare-only verifies launch inputs. Neck/back integration, remaining muscle and foot defects, and unresolved anatomical interfaces remain open. The older five-minute baseline/intervention evidence uses older anatomy and does not qualify this updated body. Provenance remains a mixed-source reference adult.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
