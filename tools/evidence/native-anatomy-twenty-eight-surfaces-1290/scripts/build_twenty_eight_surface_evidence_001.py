from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"twenty-eight-surface-native-composition-001";O=A/"twenty-eight-surface-publication-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior26_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
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
put(A/"twenty-eight-surface-movie-inspection-001.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
rp=A/"iliacus-positive-union-lift-002/report.json";put(rp,"derivation/binding-lift.json.gz",True)
for row in read(rp)["rows"]:
 put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
 sid=row["stable_id"]
 for version in ("001","002"):
  p=A/f"existing-body-{sid}-positive-union-{version}/result.json"
  if p.exists():put(p,f"derivation/source-{sid}-{version}.json.gz",True)
 put(A/f"existing-body-{sid}-positive-union-002/stable-{sid}-exact-union.json",f"derivation/exact-union-{sid}.json.gz",True)
put(A/"iliacus-allpose-interface-audit-001/report.json","derivation/all24-retained-poses.json")
for f in ["prepare_existing_body_positive_union_002.py","prepare_iliacus_positive_union_lift_002.py","audit_iliacus_allpose_interfaces_001.py","prepare_twenty_eight_surface_native_001.py","verify_twenty_eight_surface_native_001.py"]:
 put(A/f,"scripts/"+f)
put(A/"iliacus-positive-union-lift-002/parameterized_existing_lift.py","scripts/parameterized_existing_lift.py")
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,37,38,55,56,57,58,61,62,67,68,75,76,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"twenty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-twenty-six-surfaces-1287/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","twenty-eight-surface"))
(O/"README.md").write_text("""# Twenty-eight repaired surfaces in the native resting human

The left and right iliacus repairs are integrated into the existing full-body Metal scene, preserving the [previous26 repairs](../native-anatomy-twenty-six-surfaces-1287/README.md). Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.

## Repair and provenance

Both original iliacus surfaces contain15 exact self-intersection pairs. The existing exact winding owner now handles closed cut loops meeting at a vertex, using visible exact triangulation diagonals. It changes no cut coordinates and retains its area, incidence and orientation checks. The isolated owner fix is published as d4a8250;51 relevant tests and38 subtests passed on the Mini, including the retained iliacus witness.

The source repair selects strictly positive winding material: negative exterior folded regions are excluded as an explicit reference inference. Exact rational face ancestry determines interpolated deformation bindings. All four exactly one-hot attachment proxies on each side are preserved. This is not measured attachment fidelity or a claim about one measured individual. Physical muscle routes, mass and physiological equations are unchanged. The rejected original-owner attempts are retained.

## Actual native verification

The new pair is closed and free of exact Float32 self-intersections at eight accepted native captures spanning initialization, settling and a complete breathing cycle. Changed regions introduce no new crossing pairs against859 other anatomical surfaces. Two existing changed-region femur pairs per side remain; their anatomical classification is unresolved. The removal of one pair per side is not proof that the complete iliacus/femur interface is qualified.

All858 other surfaces have exactly matching referenced triangle coordinates, all157 accepted body poses match, and every value in the1000-row60-column coupled mechanics/physiology trace matches the prior26-repair run. An independent offline check covers24 retained early and late baseline/intervention poses. Only the16-second native run uses this new geometry; the old five-minute evidence does not qualify final anatomy.

The package records exact revisions, inputs, configuration, launch, timing, traces and continuous native framebuffer recording on the SSH Mac mini. The viewer retains whole-body layers, the existing flat contact support and synchronized physiological measurements. Remote desktop visibility is not asserted.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-twenty-eight-surfaces-1290/reproduce.py

The pinned Mini runtime and assets are required. Use --prepare-only to verify inputs without launching. Remaining muscle defects, unresolved interfaces, foot registration and neck/back additions remain open. Source rights and mixed-source provenance remain in the bound source receipts.
""")
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
