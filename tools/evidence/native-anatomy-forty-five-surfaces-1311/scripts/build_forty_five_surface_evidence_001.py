from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-five-surface-native-composition-001";O=A/"forty-five-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior44_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"native45-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native45_terminal_muscle_screen_001.py","scripts/inherit_native45_terminal_muscle_screen_001.py")
rp=A/"stable-133-neighbor-correction-012/current44-selection-001/report.json";put(rp,"derivation/current44-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for stem in ("stable-133-reference-continuation-006","stable-133-reference-continuation-007","stable-133-neighbor-correction-011","stable-133-neighbor-correction-012"):
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_forty_five_surface_native_001.py","verify_forty_five_surface_native_001.py","try_stable133_reference_continuation_006.py","try_stable133_reference_continuation_007.py","audit_stable133_independent_current44_008.py","audit_stable133_independent_current44_010.py","correct_stable133_neighbor_011.py","correct_stable133_neighbor_012.py","audit_stable133_independent_current44_013.py","write_stable133_current44_selection_014.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
for p in sorted((N/"prelaunch-scope-wording-amendment").iterdir()):
 if p.is_file():put(p,"derivation/prelaunch-scope-wording-amendment/"+p.name)
put(A/"native45-storage-preflight.json","native/storage-preflight.json")
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,124,128,133,134,141,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current44-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-four-surfaces-1299/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-five-surface"))
(O/"README.md").write_text('# Forty-five repaired surfaces in the native resting human\n\nThe right flexor digitorum profundus joins the preceding 44 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe repair reuses the source-positive-winding boundary and ancestry-based binding lift. Ten local source vertices move by at most 0.400007 mm relative to that reference. These are declared inferred corrections, not measurements of one person. Source positions, face ancestry, area and volume differences, binding maps and original source identities are retained. No exact one-hot attachment proxies exist for this row; no attachment-footprint accuracy is claimed. Mechanical routes and physical mass are unchanged.\n\nThe source and 25 retained poses pass exact self and closed-topology checks. This includes eight current44 native poses, 16 older baseline/intervention poses and one K1 foot-stiffness sensitivity pose. All 859 surrounding surfaces are checked at each of the eight current44 poses with no new source-parent intersection pairs. An earlier candidate failed this gate at the neighbouring superficial flexor; restoring one unnecessary source-vertex correction resolves that failure. Failed attempts remain in the derivation evidence.\n\nEight actual native45 captures then verify exact self/topology and all 859 surrounding targets for the repaired region. All 859 other rendered surfaces, all 157 accepted body poses and the 1,000-row, 60-column coupled physiology trace match native44 exactly. This is a bounded 16-second anatomy increment, not whole-body clearance or final five-minute qualification.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-five-surfaces-1311/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact revisions, hashes, configuration, device, invocation, timing, trace and continuous native framebuffer recording are retained. The recording contains a seven-layer inspection tour; no remote-desktop visibility is asserted.\n\nRemaining work includes forearm, neck/back and foot repairs, major-muscle coverage, unexplained tissue interfaces, and a final matched five-minute baseline/intervention recording with completed anatomy. The older five-minute physiology evidence uses older anatomy.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
