from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-one-surface-native-composition-001";O=A/"forty-one-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior40_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==859 and len(p["rows"])==1
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
for stem in ("foot25-current-reference-selection-013",):
 rp=A/stem/"report.json";put(rp,"derivation/"+stem+".json")
 for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for name in ("foot-25-fixed-source-junction-reference-002","foot-source-reference-posed-audit-004","foot-25-allpose-generated-reference-007","foot25-local-reference-posed-audit-010","foot25-current-interface-audit-012"):
 put(A/name/"report.json","derivation/"+name+".json.gz",True)
for f in ("prepare_foot_25_fixed_source_junction_reference_002.py","audit_foot_source_reference_poses_004.py","try_foot_25_allpose_generated_reference_007.py","audit_foot25_local_reference_poses_010.py","audit_foot25_current_interfaces_012.py","select_foot25_current_reference_013.py","prepare_forty_one_surface_native_001.py","verify_forty_one_surface_native_001.py"):
 put(A/f,"scripts/"+f)
for f in ("rejected-boundary-witness.json","result.json"):
 put(A/"existing-body-25-rejected-boundary-004"/f,"derivation/source-rejected-"+f+".gz",True)
put(A/"foot-25-fixed-source-junction-reference-002/stable-25-exact-union.json","derivation/fan-separated-reference-union.json.gz",True)
put(A/"foot-25-fixed-source-junction-reference-002/parameterized_existing_lift.py","scripts/parameterized_existing_lift.py")
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[25,29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"forty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-forty-surfaces-1295/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-one-surface"))
(O/"README.md").write_text('# Forty-one repaired surfaces in the native resting human\n\nThe right flexor digitorum longus joins the preceding 40 surface repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nThe repair reuses the exact positive-winding boundary and source-face binding lift. Source fan separation and a bounded local correction remove self-intersections while preserving all 629 original one-hot attachment-proxy positions and maps. Maximum final displacement from the fan-separated candidate is 74.373 micrometres. These proxies and reference corrections are explicit inferences, not measured attachment fidelity or measured-person anatomy.\n\nThe source and 24 retained K0 supported-body poses are closed, nondegenerate and self-intersection-free. The alternate K1 joint-stiffness sensitivity pose still has three self-intersections and is explicitly unqualified. Eight actual new native captures pass full self and 859-target changed-region checks, with no new external lineage pairs. Existing unrelated interfaces remain unqualified.\n\nAll 859 other rendered surfaces, 157 accepted body poses and every value in the 1,000-row, 60-column coupled physiology trace are identical to the preceding run. Physical routes, mass, contact and physiology equations are unchanged. Surface volume changes by about 0.0028 percent; it is not the physical body-mass owner.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-one-surfaces-1296/reproduce.py\n\nThe pinned Mini runtime and retained assets are required; --prepare-only checks launch inputs. Exact revisions, assets, launch and measurements accompany the continuous native framebuffer recording. The seven-layer native viewer runs on the SSH Mac mini. No remote-desktop visibility is asserted.\n\nForearm, neck/back, remaining foot and unexplained-interface work is open. The older five-minute paired physiology run uses older anatomy and does not qualify this updated body.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
