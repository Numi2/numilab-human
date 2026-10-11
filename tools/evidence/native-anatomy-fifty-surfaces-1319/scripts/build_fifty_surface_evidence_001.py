from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"fifty-surface-native-composition-001";O=A/"fifty-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior48_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==858 and len(p["rows"])==2
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
put(A/"native50-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native50_terminal_muscle_screen_001.py","scripts/inherit_native50_terminal_muscle_screen_001.py")
rp=A/"stable-127-143-current48-selection-015/report.json";put(rp,"derivation/current48-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for stem in ("stable-127-original-quotient-parent-004","stable-127-original-point-separation-008","stable-127-original-split-lift-009","stable-127-current48-independent-010","stable-143-parent-gap-correction-012","stable-143-current48-independent-013","stable-127-143-current48-selection-015","epl-geometric-point-separation-003","epl-geometric-point-lift-003-050um","epl-local-self-separation-008","epl-active-external-clearance-005"):
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_fifty_surface_native_001.py","verify_fifty_surface_native_001.py","separate_stable127_original_pinches_008.py","lift_stable127_original_split_009.py","audit_stable127_original_split_010.py","select_stable127_143_current48_015.py","correct_stable143_parent_gaps_012.py","audit_stable143_corrected_013.py","separate_epl_geometric_pinches_003.py","lift_epl_geometric_pinches_003.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
put(A/"completed-build-intermediate-archives-1314/restore-verification.json","native/build-archive-restore-verification.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,123,124,125,126,127,128,133,134,141,143,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-eight-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current48-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-eight-surfaces-1317/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","fifty-surface"))
(O/"README.md").write_text("# Fifty repaired surfaces in the native resting human\n\nThe right ulnar pronator head and right extensor pollicis longus join the preceding 48 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nFor the pronator head, the original source edges and binding maps are retained before the existing positive-winding boundary construction. Two incident branches at an isolated point contact receive a one-micrometre geometric separation. The maximum per-vertex source reconstruction bound is 0.000986 mm. For the thumb extensor, the final source-derived boundary includes two edge repairs, point-branch separation, and local fold/interface correction; its bound is 0.260233 mm. Both remain below the declared 0.5 mm engineering bound. These are inferred reference surfaces, not measured anatomical accuracy. Exact source members, laterality, face ancestry and binding provenance are retained.\n\nThe per-vertex bounds use each exact partition point's incident parent support and its actual final displacement, accounting for the reversal of earlier trial motion. The right thumb extensor's earlier coarse sum of maximum displacements was excessively conservative. The retained bound calculation is explicit and does not reinterpret intermediate failed proposals as admitted geometry. Final signed volume changes are -0.001567% and +0.023796%, within the declared 0.1% reference-surface sensitivity. Physical mass and force ownership are unchanged. No one-hot attachment proxies are present in either row, so attachment-footprint accuracy is not claimed.\n\nAt source and all 25 retained contexts, both candidates are closed and self-clear, with no added source-parent intersection pairs against all 859 actual surrounding targets. Their simultaneous interface is also audited. Eight contexts use native48; 16 are the older long baseline/intervention and one is the K1 foot-stiffness sensitivity. These older contexts retain their actual older target anatomy.\n\nEight actual native50 captures independently pass self/topology, all859 target checks and the simultaneous repaired-pair check. All858 other rendered surfaces, all157 accepted body poses and all1000 rows/60 columns of coupled physiology match native48 exactly. The continuous native framebuffer recording includes all seven inspection layers. This is a bounded16-second increment. The terminal inventory has133 closed self-clear muscle/tendon surfaces, two declared open Achilles attachment surfaces and15 remaining defective rows; it does not certify all tissue interfaces or all poses.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-fifty-surfaces-1319/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact revisions, asset hashes, configuration, device, invocation, timing, compact trace and continuous viewer recording are retained.\n\nRemaining work includes forearm and foot defects, head/neck/back coverage and interfaces, and the final matched five-minute recording with completed anatomy. The older five-minute physiology qualification uses older anatomy.\n")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
