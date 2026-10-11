from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-seven-surface-native-composition-001";O=A/"forty-seven-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior46_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==859 and len(p["rows"])==1
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
put(A/"native47-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native47_terminal_muscle_screen_001.py","scripts/inherit_native47_terminal_muscle_screen_001.py")
rp=A/"stable-125-current46-selection-017/report.json";put(rp,"derivation/current46-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for stem in ("stable-125-restored-edges-parent-012","stable-125-restored-edges-union-013","stable-125-restored-edges-lift-014","stable-125-restored-edges-independent-015","stable-125-current46-selection-017"):
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_forty_seven_surface_native_001.py","verify_forty_seven_surface_native_001.py","restore_stable125_source_edges_012.py","prepare_parent_positive_union_row_001.py","lift_stable125_restored_edges_014.py","audit_stable125_restored_edges_015.py","select_restored_pronator_rows_017.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
put(A/"completed-build-intermediate-archives-1314/restore-verification.json","native/build-archive-restore-verification.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,123,124,125,128,133,134,141,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current46-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-six-surfaces-1313/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-seven-surface"))
(O/"README.md").write_text("# Forty-seven repaired surfaces in the native resting human\n\nThe humeral head of the right pronator teres joins the preceding 46 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe repair restores three original source edges before applying the existing exact positive-winding boundary construction and source-face binding lift. Two independent source-edge corrections remain. Their maximum ancestry displacement plus a conservative Float32 rounding reserve is 0.174597 mm, below the declared 0.5 mm engineering limit. This is an inferred reference reconstruction, not measured anatomical accuracy. Mechanical routes and physical mass remain unchanged. The row has no exact one-hot attachment proxies; attachment-footprint accuracy is not claimed.\n\nSource signed volume changes from 1.214360319e-5 to 1.214318396e-5 cubic metres, a reduction of approximately 0.419 cubic millimetres. The candidate is closed and self-intersection-free at source and all 25 retained poses. Its changed face region adds no source-parent intersection pairs against all 859 targets at each capture. Eight are current native46 captures; 16 are older baseline/intervention captures and one is the K1 foot-stiffness sensitivity capture. Older targets are actual captured anatomy. Failed and superseded local proposals remain retained on the Mini; the independent report's source_union_candidate_path identifies the earlier reference, while candidate_path identifies the final restored-edge candidate.\n\nEight actual native47 captures verify self/topology and all 859 surrounding targets for the repaired region. All 859 other rendered surfaces, all 157 accepted body poses and the 1,000-row, 60-column coupled physiology trace match native46 exactly. This is a bounded 16-second anatomy increment. The terminal muscle inventory has 130 closed self-clear surfaces, two declared open Achilles attachment surfaces, and 18 remaining defective rows; this inventory does not certify all tissue interfaces or every pose.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-seven-surfaces-1315/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact revisions, hashes, configuration, device, invocation, timing, trace and continuous native framebuffer recording are retained. The recording includes all seven anatomical layers; remote-desktop visibility is not asserted.\n\nForearm, neck/back and foot defects, head-muscle coverage, unexplained tissue interfaces, and the final matched five-minute recording with completed anatomy remain open. The older five-minute physiology evidence uses older anatomy.\n")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
