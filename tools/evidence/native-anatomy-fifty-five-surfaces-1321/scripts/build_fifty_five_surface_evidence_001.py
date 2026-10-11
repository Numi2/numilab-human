from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"fifty-five-surface-native-composition-001";O=A/"fifty-five-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior50_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==855 and len(p["rows"])==5
 for r in p["rows"]:assert r["self_count"]==r["new_triangle_pair_count"]==0 and r["external_targets_scanned"]==859
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
put(A/"native55-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native55_terminal_muscle_screen_001.py","scripts/inherit_native55_terminal_muscle_screen_001.py")
rp=A/"stable-121-122-135-136-144-current50-selection-001/report.json";put(rp,"derivation/current50-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for row in read(rp)["rows"]:put(row["per_vertex_bound_path"],f"derivation/stable-{row['stable_id']}-source-bounds.npz")
for sid in (121,122,135,136,144):
 stem=f"remaining-original-boundary-batch-001/stable-{sid}"
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_fifty_five_surface_native_001.py","verify_fifty_five_surface_native_001.py","select_five_forearms_current50_001.py","resolve_forearm_submicrometre_edges_001.py","prepare_remaining_original_boundaries_001.py","lift_remaining_original_boundary_001.py","audit_remaining_original_boundary_001.py"):
 put(A/name,"scripts/"+name)
F=A/"fifty-four-surface-native-composition-001"
for f in ["baseline/execution.json","baseline/run-declaration.json","baseline/native-run/run-metadata.json","baseline/native-run/surface-audit-failure-step-63.json"]:
 put(F/f,"failed-native54/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
put(A/"completed-build-intermediate-archives-1314/restore-verification.json","native/build-archive-restore-verification.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,121,122,123,124,125,126,127,128,133,134,135,136,141,143,144,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"fifty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current50-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-fifty-surfaces-1319/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","fifty-five-surface"))
(O/"README.md").write_text('# Fifty-five repaired surfaces in the native resting human\n\nBoth ulnar heads of flexor carpi ulnaris, both flexor pollicis longus surfaces, and the left extensor pollicis longus join the prior 50 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe original source coordinates and binding maps feed the existing exact positive-winding boundary owner. Isolated point contacts receive an explicitly inferred one-micrometre branch separation. Both thumb flexors also need three topology-preserving edge contractions: submicrometre features in their exact boundary can collapse during Float32 posing. The existing edge-repair owner enforces the manifold link condition, orientation, component/Euler preservation and signed-volume bounds. Retained endpoints keep their canonical maps; removed faces and vertex ancestry are recorded. Maximum additional contraction displacement is 0.2634 micrometres. The combined per-vertex source reconstruction bound stays below 1.025 micrometres, and signed-volume change below 0.00343%. These are engineering reconstruction bounds, not anatomical measurement accuracy. Physical mass and force ownership are unchanged.\n\nThe first four-surface native attempt stopped at accepted step63 (0.126 seconds) when two left thumb-flexor triangles became zero-area. That failure is retained. It was not admitted or overwritten, and the continuous surface audit remains enabled. The later five-surface composition uses the repaired submicrometre geometry.\n\nEach final candidate passes source and 25 retained pose checks, with no added source-parent intersection pairs against 859 actual surrounding targets in every context. All ten simultaneous candidate pairs are checked in all 25 contexts. Eight contexts use native50; 16 use the older long baseline/intervention and one uses the K1 foot-stiffness sensitivity. Those older contexts retain their actual older target anatomy.\n\nEight actual native55 captures independently pass exact self/topology, all859 target checks and all simultaneous repaired-pair checks. All855 other rendered surfaces, all157 accepted body poses and all1000 rows/60 columns of coupled physiology match native50 exactly. The native framebuffer recording includes all seven inspection layers. This is a bounded16-second increment. The terminal inventory has138 closed self-clear muscle/tendon surfaces, two declared open Achilles attachment surfaces and10 remaining defective rows. This inventory does not certify all tissue interfaces or all poses.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-fifty-five-surfaces-1321/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact revisions, asset identities, configuration, device, invocation, timing, compact trace and continuous viewer recording are retained.\n\nRemaining work includes forearm and foot defects, head/neck/back coverage and interfaces, and the final matched five-minute recording with completed anatomy. The older five-minute physiology qualification uses older anatomy.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
