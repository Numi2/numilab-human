from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-four-surface-native-composition-002";O=A/"forty-four-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior43_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"native44-terminal-muscle-inherited-screen-001/report.json","checks/terminal-muscle-inventory.json.gz",True)
put(A/"inherit_native44_terminal_muscle_screen_001.py","scripts/inherit_native44_terminal_muscle_screen_001.py")
put(A/"forty-four-surface-native-composition-001/preparation-failure.json","derivation/preparation-attempt-001-failure.json")
put(A/"prepare_forty_four_surface_native_001.py","derivation/prepare_forty_four_surface_native_001.py")
rp=A/"stable-141-positive-union-lift-001/current43-selection-001/report.json";put(rp,"derivation/current43-selection.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for stem in ("stable-141-positive-union-lift-001",):
 for src in sorted((A/stem).rglob("*")):
  if not src.is_file() or "__pycache__" in src.parts or src.suffix not in (".json",".jsonl",".py"):continue
  rel="derivation/"+str(src.relative_to(A));compress=src.suffix in (".json",".jsonl")
  put(src,rel+(".gz" if compress else ""),compress)
for name in ("prepare_forty_four_surface_native_002.py","verify_forty_four_surface_native_001.py"):
 put(A/name,"scripts/"+name)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,26,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,124,128,134,141,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-three-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_owner_identity":"Exact owner source-file hashes and source asset revisions are pinned by derivation/current43-selection.json","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-three-surfaces-1298/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-four-surface"))
(O/"README.md").write_text('# Forty-four repaired surfaces in the native resting human\n\nThe right extensor indicis joins the preceding 43 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe repair reuses the source-positive-winding boundary and ancestry-based binding lift. One bounded correction to reference vertex131 resolves the retained pose-dependent self-intersections. This is inferred reference geometry, not a measured-person anatomical feature. Binding maps and mechanical routes remain unchanged; source resolution, source area and volume changes, and attachment-proxy checks are retained in native/source-resolution-bounds.json. Attachment proxies do not establish footprint accuracy.\n\nThe source and 32 named retained poses pass exact self checks. Eight current43 captures separately pass the current exact predicate and 859 surrounding-target comparisons. Failed derivation attempts are retained with their actual outcomes.\n\nEight actual native44 captures pass full exact self/topology checks and 859-target comparisons for the changed region, with no new external source-parent pairs. All 859 other rendered surfaces and all 157 accepted body poses match native43 exactly. The full 1,000-row, 60-column coupled physiology trace also matches exactly. Physical mass, contact and physiology equations are unchanged.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-four-surfaces-1299/reproduce.py\n\nThe pinned Mac mini runtime and retained assets are required. Use --prepare-only to verify launch inputs. Exact source revisions, asset identities, configuration, device, launch command, timing, compact trace and continuous native framebuffer recording are retained. The viewer exposes seven anatomical layers. No remote-desktop visibility is asserted.\n\nRemaining work includes forearm, neck/back and foot surfaces, major-muscle coverage, inherited unexplained tissue interfaces, and a final five-minute matched baseline/intervention recording using the completed anatomy. The older five-minute physiology evidence uses older anatomy.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
