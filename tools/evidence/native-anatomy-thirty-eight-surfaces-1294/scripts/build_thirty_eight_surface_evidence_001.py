from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"thirty-eight-surface-native-composition-001";O=A/"thirty-eight-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior34_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
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
put(N/"movie-inspection.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
for stem in ("chest-shoulder-reference-selection-003","pectoral-reference-selection-010"):
 rp=A/stem/"report.json";put(rp,"derivation/"+stem+".json")
 for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for sid in (73,77,78,85):
 for fn in ("rejected-boundary-witness.json","result.json"):
  put(A/f"existing-body-{sid}-rejected-boundary-004"/fn,f"derivation/{sid}-{fn}.gz",True)
 put(A/f"existing-body-{sid}-pinched-reference-001/report.json",f"derivation/{sid}-junction-report.json.gz",True)
for name in ("pectoral-73-fixed-source-junction-reference-002","chest-shoulder-selected-pose-audit-002","chest-shoulder-interface-audit-005","pectoral-source-point-restoration-007","pectoral-source-point-restored-poses-008","pectoral-interface-audit-011"):
 put(A/name/"report.json","derivation/"+name+".json.gz",True)
for f in ["capture_rejected_winding_boundary_004.py",*[f"prepare_existing_body_{sid}_pinched_reference_001.py" for sid in (73,77,78,85)],"prepare_pectoral_73_fixed_source_junction_reference_002.py","audit_chest_shoulder_selected_poses_002.py","prepare_chest_shoulder_reference_selection_003.py","audit_chest_shoulder_interfaces_005.py","restore_pectoral_source_points_007.py","audit_pectoral_source_point_restored_poses_008.py","prepare_pectoral_reference_selection_010.py","audit_pectoral_interfaces_011.py","prepare_thirty_eight_surface_native_001.py","verify_thirty_eight_surface_native_001.py"]:
 put(A/f,"scripts/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,75,76,77,78,85,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"thirty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-thirty-four-surfaces-1293/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","thirty-eight-surface"))
(O/"README.md").write_text("""# Thirty-eight repaired surfaces in the native resting human

The right sternocostal pectoralis major, both clavicular deltoids and right subscapularis now join the [preceding 34 repairs](../native-anatomy-thirty-four-surfaces-1293/README.md) and corrected skin in the full-body native Metal scene. **Whole-body anatomy and final five-minute qualification remain incomplete; performance work is deferred.**

The existing positive-winding owner extracts a reference boundary from inverted folds. Original anatomical identities, source-face ancestry and binding tables remain bound. Junction fans are separated by one micrometre. For the right pectoral surface, original one-hot junction points stay fixed while their copied fan vertices move; two original points removed by boundary extraction are restored onto the nearby boundary (4.356 and 160.952 micrometres from the intermediate boundary). The final candidate retains every original one-hot source-point proxy and its map. This proxy check is not measured attachment fidelity.

All four source surfaces and eight retained early poses are closed, nondegenerate and self-intersection-free, with no new changed-region crossings against 859 surrounding structures. Older late captures omit thoracic body25, so this increment makes no 24-pose claim.

Eight new native captures verify the actual GPU-deformed surfaces with exact predicates. All four remain closed and self-intersection-free. No new changed-region crossings appear, including full comparisons between all six pairs of simultaneously replaced surfaces against their actual preceding native geometry. Existing internal interfaces, including subscapularis/bone and thoracic structures, remain unqualified.

All 856 other rendered surfaces, the 157 accepted body poses and every value in the 1,000-row, 60-column coupled trace remain identical to the preceding run. Physical routes, mass, contact and physiological equations are unchanged. Inferred passive surface corrections are not new physiological fidelity or measured-person anatomy.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-thirty-eight-surfaces-1294/reproduce.py

The pinned Mini runtime and retained source assets are required; --prepare-only checks launch inputs. The package binds revisions, assets, exact launch, native measurements and continuous framebuffer recording. The recording has 252 frames and seven inspection layers on the SSH Mac mini; no remote desktop visibility is asserted.

Remaining neck/back, muscle, foot and unexplained-interface work is still open. The old five-minute paired physiology run uses older anatomy and does not qualify this updated body.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
