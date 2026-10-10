from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"twenty-six-surface-native-composition-001";O=A/"twenty-six-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior24_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8
for p in d["poses"]:
 assert p["unchanged_surface_referenced_triangles_exact_parent"]==858 and len(p["rows"])==2
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
put(A/"twenty-six-surface-movie-inspection-001.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
rp=A/"acromial-deltoid-local-reference-003/report.json";put(rp,"derivation/reference-003.json")
for row in read(rp)["rows"]:
 trial=next(x for x in row["trials"] if x["alpha"]==.1)
 put(trial["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for sid in (75,76):put(A/f"acromial-deltoid-local-interface-audit-{sid}-001/report.json",f"derivation/endpoint-interfaces-{sid}.json")
put(A/"acromial-deltoid-local-clearance-allpose-audit-001/report.json","derivation/all24-retained-poses.json")
for attempt in (1,2):put(A/f"acromial-deltoid-local-reference-{attempt:03}/report.json",f"derivation/unselected-reference-{attempt:03}.json")
for f in ["try_acromial_deltoid_local_reference_001.py","try_acromial_deltoid_local_reference_002.py","try_acromial_deltoid_local_reference_003.py","audit_acromial_deltoid_local_interfaces_75_001.py","audit_acromial_deltoid_local_interfaces_76_001.py","audit_acromial_deltoid_local_clearance_allpose_001.py","prepare_twenty_six_surface_native_001.py","verify_twenty_six_surface_native_001.py"]:
 put(A/f,"scripts/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,55,56,57,58,61,62,67,68,75,76,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"twenty-four-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-twenty-four-surfaces-1286/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","twenty-six-surface"))
(O/"README.md").write_text('# Twenty-six repaired surfaces in the integrated native human\n\nThe left and right acromial deltoids join the [previous 24 repairs](../native-anatomy-twenty-four-surfaces-1286/README.md) in the existing Metal resting human on the SSH Mac mini.\n\n**Whole-body anatomy and final five-minute qualification remain incomplete. Performance optimization is paused.**\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-twenty-six-surfaces-1287/reproduce.py\n\nThe retained Mini runtime and pinned assets are required. Use --prepare-only to verify inputs without launching. This package retains the exact launch, source revisions, device, timing, coupled trace and continuous native framebuffer recording.\n\n## Shoulder source correction\n\nEach original acromial deltoid has four exact source self-intersection pairs. The existing exact-coordinate quotient and bounded one-ring source correction preserve all original face order, lineage and binding weights. Seven vertices on each side move, with maximum original-source displacements of 0.249817 mm right and 0.249831 mm left. Source volume changes by approximately -0.01252%. No exactly one-hot attachment proxy moves; this is not measured attachment fidelity. Physical muscle routes, mass, contact and the coupled physiological state remain unchanged.\n\nThese are declared reference corrections, not measurements of one individual or a source accuracy guarantee. Earlier candidates either retained self-intersections or slightly exceeded the 0.25 mm bound after Float32 projection; their reports are retained. The preparer independently replays the selected correction and verifies its bounds.\n\n## Native checks and limits\n\nThe new pair is self-intersection-free with closed oriented exact-coordinate topology in all eight actual native captures. Changed regions add no triangle-pair crossings against 859 neighbouring anatomical structures. Five existing crossing pairs with the same-side spinal deltoid remain unchanged; they are unresolved and are not admitted as an anatomical interface.\n\nAll 858 other anatomical surfaces have byte-exact referenced triangle coordinates and all 157 accepted body poses match the previously verified 24-surface run. The prior repair checks are thereby preserved, including the separately bounded left-pronator shared-head classification. All 1,000 rows and 60 coupled mechanics/physiology columns match the previous native run exactly. The 8,000 accepted steps span about 16 simulated seconds, settling and complete breaths. An additional offline comparison checks the new pair against 24 retained early and late baseline/intervention poses.\n\nThe native viewer displays the integrated human on its existing flat support plane, with inspectable anatomy layers and coupled measurements. The movie records its framebuffer; remote desktop visibility is not asserted. Source rights and mixed-source provenance remain bound in the retained assets and receipts.\n\nRemaining muscle defects, unresolved interfaces, foot registration and the eight opt-in neck/back surfaces are still open. This increment is not complete-body or final five-minute acceptance.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
