from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-eight-surface-native-composition-001";O=A/"forty-eight-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior47_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"native48-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native48_terminal_muscle_screen_001.py","scripts/inherit_native48_terminal_muscle_screen_001.py")
rp=A/"stable-126-current47-selection-022/report.json";put(rp,"derivation/current47-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for stem in ("stable-126-restored-edges-parent-012","stable-126-restored-edges-union-013","stable-126-restored-edges-lift-014","stable-126-restored-edges-independent-015","stable-126-current47-selection-022","stable-126-parent-gap-correction-020","stable-126-parent-gap-independent-021"):
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_forty_eight_surface_native_001.py","verify_forty_eight_surface_native_001.py","restore_stable126_source_edges_012.py","prepare_parent_positive_union_row_001.py","lift_stable126_restored_edges_014.py","audit_stable126_restored_edges_015.py","select_corrected_left_pronator_022.py","correct_stable126_parent_gaps_020.py","audit_stable126_parent_gap_021.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
put(A/"completed-build-intermediate-archives-1314/restore-verification.json","native/build-archive-restore-verification.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,123,124,125,126,128,133,134,141,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-seven-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current47-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-seven-surfaces-1315/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-eight-surface"))
(O/"README.md").write_text("# Forty-eight repaired surfaces in the native resting human\n\nThe humeral head of the left pronator teres joins the preceding 47 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThree original source edges are restored before the existing exact positive-winding boundary and source-face binding lift. Two bounded parent corrections remain. A subsequent correction moves 11 vertices by at most 0.004948 mm to preserve positive separating gaps from actual accepted parent triangles against the other left pronator head and flexor pollicis longus. Face topology, ancestry and binding arrays remain identical through that correction. The conservative cumulative source bound is 0.179545 mm, below the declared 0.5 mm engineering limit. This is inferred reference geometry, not measured anatomical accuracy. Mechanical routes and physical mass remain unchanged; no exact one-hot attachment proxies are present and no footprint accuracy is claimed.\n\nThe new positions keep each parent face's normal direction, minimum area ratio 0.99336 and minimum normal cosine 0.99660. Signed volume differs from the original by -3.582410e-10 cubic metres (approximately -0.358 cubic millimetres). The candidate is closed and self-intersection-free at source and all 25 retained poses, with no new source-parent intersection pairs in the changed region against all 859 actual targets at every capture. Eight captures are native47; 16 are the older long baseline/intervention and one is the K1 foot-stiffness sensitivity. Older targets are actual captured anatomy. No shared-insertion exemption was needed.\n\nEight actual native48 captures independently pass self/topology and all 859 surrounding-target checks. All 859 other rendered surfaces, all 157 accepted body poses, and all 1,000 rows and 60 columns of coupled physiology match native47 exactly. This is a bounded 16-second increment, not whole-body clearance. The terminal inventory now has 131 closed self-clear muscle/tendon surfaces, two declared open Achilles attachment surfaces and 17 remaining defective rows. It does not certify all tissue interfaces or all poses.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-eight-surfaces-1317/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact revisions, hashes, configuration, device, invocation, timing, compact trace and continuous native framebuffer recording are retained. All seven anatomical layers are present; remote-desktop visibility is not asserted.\n\nRemaining work includes forearm and foot defects, head/neck/back coverage and interfaces, and the final matched five-minute recording with completed anatomy. The older five-minute physiology qualification uses older anatomy.\n")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
