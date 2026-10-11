from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"fifty-seven-surface-native-composition-001";O=A/"fifty-seven-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior56_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==859 and len(p["rows"])==1
 for r in p["rows"]:assert r["self_count"]==r["new_triangle_pair_count"]==0 and r["external_targets_scanned"]==859
ex=read(N/"baseline/execution.json")
assert ex["returncode"]==0 and not ex["changed_inputs"] and ex["native_argv_matches_prepared_cli_preview"] and ex["native_environment_matches_prepared_cli_preview"]
movie=(N/"movie-inspection.log").read_text();assert "frames=252 timing_markers=5" in movie and all("frame="+x+" " in movie for x in ["skin","muscles","skeleton","organs","lungs","heart","vessels"])
O.mkdir(exist_ok=False);inventory={}
def put(src,rel,compress=False):
 src=Path(src);dest=O/rel;dest.parent.mkdir(parents=True,exist_ok=True);raw=src.read_bytes()
 if not compress and src.suffix==".py" and raw.endswith(b"\n\n"):
  compress=True;rel+=".gz";dest=O/rel
 dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
 inventory[rel]={"bytes":dest.stat().st_size,"encoding":"gzip" if compress else "identity","retained_path":str(src),"retained_sha256":hashlib.sha256(raw).hexdigest(),"sha256":sha(dest)}
for f in ["launch_arm.py","launch-guard.json","composition-result.json","source-resolution-bounds.json"]:put(N/f,"native/"+f)
for f in ["run-declaration.json","execution-start.json","execution.json"]:put(N/"baseline"/f,"native/baseline/"+f)
for f in ["invocation.json","run-metadata.json","resting-coupled.csv"]:put(N/"baseline/native-run"/f,"native/baseline/native-run/"+f)
for f in ["frame-skin.png","frame-muscles.png","frame-skeleton.png","frame-organs.png","native-viewer.mov"]:put(N/"baseline/native-run"/f,"viewer/"+f)
put(N/"movie-inspection.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
put(A/"native57-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native57_terminal_muscle_screen_001.py","scripts/inherit_native57_terminal_muscle_screen_001.py")
rp=A/"remaining-original-boundary-batch-001/stable-137/current56-selection-013/report.json";put(rp,"derivation/current56-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for sid in (137,):
 stem=f"remaining-original-boundary-batch-001/stable-{sid}"
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_fifty_seven_surface_native_001.py","verify_fifty_seven_surface_native_001.py","write_stable137_current56_selection_013.py","translate_extensor_source_fold_002.py","correct_extensor_retained_pose_folds_005.py","correct_extensor_source_after_posed_008.py","correct_extensor_protected_bone_interface_010.py","correct_extensor_source_after_protected_bone_011.py","refine_remaining_positive_boundary_001.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
put(A/"completed-build-intermediate-archives-1314/restore-verification.json","native/build-archive-restore-verification.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,121,122,123,124,125,126,127,128,133,134,135,136,137,138,141,143,144,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"fifty-six-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current56-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-fifty-six-surfaces-1324/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","fifty-seven-surface"))
(O/"README.md").write_text('# Fifty-seven repaired surfaces in the native resting human\n\nThe right extensor digitorum joins the previous 56 repairs and corrected skin in one native full-body scene. Whole-body anatomy and the final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe existing positive-winding boundary and canonical attribute owners preserve source identity, face ancestry and MyoSim route bindings. Bounded local corrections resolve source and posed intersections while holding four wrist-bone interface parent faces fixed. Maximum displacement from the rational source boundary is 0.48703 mm; enclosed geometric volume changes by 0.01733%. These are declared engineering reconstruction bounds, not anatomical measurement accuracy. There are no exact one-hot attachment proxies in this row, so proxy preservation cannot certify attachment footprints. Physical mass and force ownership are unchanged.\n\nThe candidate is closed, nondegenerate and self-clear at source and in 25 retained poses. Every pose scans all 859 surrounding structures with no added source-parent intersection pairs. Eight contexts use native56, 16 use older long baseline/intervention runs, and one uses the retained K1 foot-stiffness sensitivity. Older contexts retain their actual older target anatomy. Rejected local corrections, including the earlier migration of ulna intersection pairs, remain in the derivation evidence.\n\nEight actual native57 captures independently pass exact self/topology and all 859 target checks. All 859 other rendered surfaces, all 157 accepted body poses and all 1000 rows/60 columns of coupled physiology match native56 exactly. The continuous native framebuffer recording includes all seven inspection layers. The surface audit remained enabled through all 8000 steps. This is a 16-second increment, not final endurance qualification.\n\nThe terminal inventory has 140 closed self-clear muscle/tendon surfaces, two separately declared open Achilles attachment surfaces, and eight remaining defective rows: 21, 22, 23, 24, 129, 130, 131 and 132. This inventory does not certify every tissue interface or every pose. Head, neck and back coverage and interface work also remain.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-fifty-seven-surfaces-1325/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Source revisions, asset identities, configuration, device, invocation, timing, compact trace and the continuous viewer recording are retained. No new interface exemptions were introduced. The older five-minute physiology evidence uses older anatomy.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
