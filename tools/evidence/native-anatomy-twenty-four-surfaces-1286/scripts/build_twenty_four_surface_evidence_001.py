from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"twenty-four-surface-native-composition-001";O=A/"twenty-four-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior22_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"twenty-four-surface-movie-inspection-001.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
for attempt in (3,4):
 rp=A/f"external-oblique-local-reference-{attempt:03}/report.json";put(rp,f"derivation/reference-{attempt:03}.json")
 row=read(rp)["rows"][0];trial=next(x for x in row["trials"] if x["alpha"]==.1)
 put(trial["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for attempt in (1,2):put(A/f"external-oblique-local-interface-audit-{attempt:03}/report.json",f"derivation/endpoint-interfaces-{attempt:03}.json")
put(A/"external-oblique-local-reference-002/report.json","derivation/partial-baseline-source-002.json")
for f in ["try_external_oblique_local_reference_002.py","try_external_oblique_local_reference_003.py","try_external_oblique_local_reference_004.py","audit_external_oblique_local_interfaces_001.py","audit_external_oblique_local_interfaces_002.py","prepare_twenty_four_surface_native_001.py","verify_twenty_four_surface_native_001.py"]:
 put(A/f,"scripts/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,55,56,57,58,61,62,67,68,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"twenty-two-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-twenty-two-surfaces-1284/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","twenty-four-surface"))
(O/"README.md").write_text("""# Twenty-four repaired surfaces in the integrated native human

The left and right external obliques join the [previous 22 repairs](../native-anatomy-twenty-two-surfaces-1284/README.md) in the existing Metal resting human on the SSH Mac mini.

**Whole-body anatomy and final five-minute qualification remain incomplete. Performance optimization is paused.**

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-twenty-four-surfaces-1286/reproduce.py

The retained Mini runtime and pinned assets are required. Use --prepare-only to verify inputs without launching. The package retains exact commands, source revisions, device, timing, compact physiological trace and a continuous native framebuffer recording.

## Abdominal source correction

Each original external-oblique surface contains seven exact self-intersection pairs. A canonical quotient welds identical source positions only after verifying identical dense binding weights. All source triangles retain their order and lineage. Seven vertices on each side receive a declared one-ring source-coordinate correction. The maximum correction is 0.051821 mm right and 0.051894 mm left; reconstructed source volumes change by less than 0.000003%. Binding weights, physical routes, mass and contact remain unchanged. No exactly one-hot attachment proxy moves; this does not establish measured attachment fidelity.

These are explicitly inferred reference repairs, not measurements of one person or an accuracy guarantee for BodyParts3D. A 0.03 smoothing factor on the right remains intersecting; the chosen 0.1 clears the local fold. Larger unselected proposals remain in retained diagnostics. The preparer independently replays the selected correction and verifies the displacement, unchanged attributes, topology and volume bounds before composition.

## Native checks

All eight actual native captures have zero exact self-intersections and closed oriented topology for both external obliques. Their changed regions introduce no new triangle-pair crossings against 859 neighbouring structures. The 858 unchanged anatomical surfaces have exactly the same referenced triangle coordinates as the previously verified 22-surface run, with identical 157-body accepted poses. That preserves the prior qualification, including its separately declared local left-pronator interface classification, without introducing another exception.

All 1,000 rows and 60 coupled mechanics/physiology columns exactly match the previous native run, whose trace exactly matches unchanged baseline 011. The run accepts 8,000 steps, about 16 simulated seconds, spanning settling and complete breaths. The existing flat contact plane remains the mechanical support.

Remaining source-muscle defects, foot assembly crossings and unqualified neck/back surfaces remain open. The eight newly prepared neck/back surfaces are deliberately opt-in and are not included in this run. This short check does not replace final five-minute baseline/intervention qualification. The continuous movie records the native framebuffer; separate remote desktop visibility is not asserted. Source rights and mixed-source reference provenance remain bound in the original assets and receipts.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
