from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"thirty-surface-native-composition-001";O=A/"thirty-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior28_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"thirty-surface-movie-inspection-001.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
rp=A/"fibularis-reference-selection-001/report.json";put(rp,"derivation/selected-reference.json")
for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for sid,stem,witness in [(39,"fibularis","fibularis-rejected-boundary-004"),(40,"fibularis-left","fibularis-left-rejected-boundary-004")]:
 for kind in ("pinched-reference","junction-pose-sensitivity"):
  put(A/f"{stem}-{kind}-001/report.json",f"derivation/{sid}-{kind}.json.gz",True)
 put(A/witness/"rejected-boundary-witness.json",f"derivation/{sid}-rejected-exact-boundary.json.gz",True)
 put(A/witness/"result.json",f"derivation/{sid}-rejected-boundary-result.json.gz",True)
 put(A/f"{stem}-pinched-reference-001/parameterized_existing_lift.py",f"scripts/parameterized_lift_{sid}.py")
put(A/"fibularis-allpose-interface-audit-001/report.json","derivation/all24-retained-poses.json")
for f in ["capture_rejected_winding_boundary_004.py","prepare_fibularis_pinched_reference_001.py","prepare_fibularis_left_pinched_reference_001.py","audit_fibularis_junction_pose_sensitivity_001.py","audit_fibularis_left_junction_pose_sensitivity_001.py","prepare_fibularis_pair_selection_001.py","audit_fibularis_allpose_interfaces_001.py","prepare_thirty_surface_native_001.py","verify_thirty_surface_native_001.py"]:
 put(A/f,"scripts/"+f)
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,37,38,39,40,55,56,57,58,61,62,67,68,75,76,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"twenty-eight-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-twenty-eight-surfaces-1290/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","thirty-surface"))
(O/"README.md").write_text("""# Thirty repaired surfaces in the native resting human

Both fibularis brevis surfaces join the [previous 28 repairs](../native-anatomy-twenty-eight-surfaces-1290/README.md) in the existing full-body Metal scene. **Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.**

## Source repair and attachment limits

Each original fibularis brevis has nine exact self-intersection pairs, including a long narrow fold. Existing exact positive-winding boundary extraction removes inverted exterior folds but leaves two point junctions. Those rejected results are retained. The existing vertex-fan splitter separates their indices; four junction vertices then retreat toward their own fan-neighbor centroids by one micrometre, resolving the point contacts without changing their source-derived binding maps. This is an explicit reference correction, not measured anatomy or an accuracy claim.

The completed repair changes source surface area by approximately -0.199% and signed volume by -0.00223%. These figures include boundary extraction; the one-micrometre bound applies only to the subsequent junction separation. All eight exactly one-hot attachment proxies per side are preserved. This is not measured attachment fidelity. Physical routes, forces, mass and physiological equations are unchanged.

Source Float32 closure and exact self-intersection checks pass. Sensitivity checks at 0.1, 1 and 10 micrometres pass all 24 retained early and late poses on both sides; the right side also passes six additional scales through 100 micrometres. One micrometre was selected for native verification.

## Actual native checks

All eight native captures show closed, self-intersection-free repaired surfaces. Changed regions add no crossings against 859 surrounding anatomical structures. Existing crossings elsewhere are not classified or waived by this test. All 858 other surfaces and all 157 accepted body poses match the previous 28-repair run exactly; all 1,000 rows and 60 columns of coupled mechanics/physiology match exactly. The native run spans 8,000 accepted steps and 16 simulated seconds, including settling and a complete breathing cycle.

The package records the launch, exact inputs and revisions, device, timing, coupled trace and continuous native framebuffer recording on the SSH Mac mini. The viewer retains whole-body layers, the existing flat contact support and synchronized measurements. No remote desktop visibility is asserted.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-thirty-surfaces-1291/reproduce.py

The pinned Mini runtime and assets are required. Use --prepare-only to verify inputs without launching. Final neck/back integration, remaining muscle defects, foot registration and unexplained interfaces remain open. The old five-minute baseline/intervention evidence uses older anatomy and does not qualify this final body. Source rights and mixed-source provenance remain in the bound receipts.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
