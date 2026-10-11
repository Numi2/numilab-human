from pathlib import Path
import json,hashlib,gzip,csv
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
N=A/"forty-surface-native-composition-001";O=A/"forty-surface-publication-002"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
d=read(N/"native-verification-001/report.json")
assert d["complete"] and d["inputs_unchanged"] and d["all_native_pair_checks_pass"] and d["all_prior38_and_skin_geometry_exact"] and d["all_coupled_trace_rows_exact_parent"]
assert len(d["poses"])==8 and d["all_simultaneously_repaired_pairs_pass"]
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
put(N/"movie-inspection.log","viewer/movie-inspection.log")
put(N/"native-verification-001/report.json","checks/native-verification.json")
for stem in ("left-chest-shoulder-reference-selection-023",):
 rp=A/stem/"report.json";put(rp,"derivation/"+stem+".json")
 for row in read(rp)["rows"]:put(row["candidate_path"],f"patches/stable-{row['stable_id']}.npz")
for name in ("subscapularis-left-directed-reference-012","subscapularis-left-directed-poses-015","pectoral-left-f32-separation-014","pectoral-left-junction-reference-016","pectoral-left-source-point-restoration-017","pectoral-left-generated-boundary-repair-018","pectoral-left-restored-poses-019","pectoral-left-generated-direction-sensitivity-020","pectoral-left-allpose-generated-reference-021","pectoral-left-final-poses-022","left-chest-shoulder-interface-audit-024"):
 rp=A/name/"report.json"
 if not rp.exists():rp=A/name/"source-proposal.json"
 put(rp,"derivation/"+name+".json.gz",True)
for f in ("prepare_pectoral_left_f32_separation_014.py","prepare_pectoral_left_junction_reference_016.py","restore_pectoral_left_source_points_017.py","repair_pectoral_left_generated_boundary_018.py","audit_pectoral_left_restored_poses_019.py","try_pectoral_left_generated_direction_sensitivity_020.py","try_pectoral_left_allpose_generated_reference_021.py","audit_pectoral_left_final_poses_022.py","try_subscapularis_left_directed_reference_012.py","audit_subscapularis_left_directed_poses_015.py","prepare_left_chest_shoulder_reference_selection_023.py","audit_left_chest_shoulder_interfaces_024.py","prepare_forty_surface_native_001.py","verify_forty_surface_native_001.py"):
 put(A/f,"scripts/"+f)
for f in ("source-proposal.json","rejected-boundary-witness.json"):
 put(A/"pectoral-left-f32-separation-014"/f,"derivation/pectoral-left-f32-separation-014-"+f+".gz",True)
put(A/"existing-body-86-rejected-boundary-004/rejected-boundary-witness.json","derivation/subscapularis-left-rejected-boundary.json.gz",True)
put(A/"existing-body-86-pinched-reference-001/report.json","derivation/subscapularis-left-junction.json.gz",True)
put(A/"prepare_existing_body_86_pinched_reference_001.py","scripts/prepare_existing_body_86_pinched_reference_001.py")
put(Path(__file__),"scripts/"+Path(__file__).name)
runtime=read(N/"baseline/native-run/run-metadata.json");trace=list(csv.DictReader((N/"baseline/native-run/resting-coupled.csv").open()))
sim=float(trace[-1][next(iter(trace[-1]))]);wall=runtime["wall_seconds"]
summary={"anatomy_complete":False,"five_minute_final_anatomy_qualification":False,"accepted_steps":8000,"simulated_seconds":sim,"owner_wall_seconds":wall,"real_time_factor":sim/wall,"performance_optimization_paused":True,"stable_ids":[29,30,35,36,37,38,39,40,49,50,55,56,57,58,59,60,61,62,67,68,73,74,75,76,77,78,85,86,105,106,107,108,115,116,124,128,147,148,149,150],"candidate_tissue_sha256":sha(N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"parent_tissue_sha256":sha(A/"thirty-eight-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"),"device":"SSH macmini, Apple M4 Pro,24GB,macOS26.6","source_revisions":{"Numi2/numilab-human":"48f20699315f1039df770a4289360ef9419646d3","Numi2/numi-lab":"3a1cfffaeba4502d946cc0d4aeb8860a0075da79","Numi2/numi-brain":"a1cf7218fae5d26f9aed9d845047fc6c472ad596"},"offline_arrangement_source_revision":"d4a8250","additional_interface_exceptions":False,"parent_qualification":"../native-anatomy-thirty-eight-surfaces-1294/README.md"}
(O/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
repro=Path("/Users/n/numi-human-positive-winding-1279/tools/evidence/native-anatomy-seventeen-surfaces-1280/reproduce.py")
(O/"reproduce.py").write_text(repro.read_text().replace("seventeen-surface","forty-surface"))
(O/"README.md").write_text('# Forty repaired surfaces in the native resting human\n\nThe left sternocostal pectoralis major and left subscapularis join the preceding 38 repairs and corrected skin in one native full-body scene. Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.\n\nBoth repairs reuse the exact positive-winding reference boundary and source-face binding lift. Original source attachment-proxy positions and maps are preserved; these proxies are not measured attachment fidelity. In the left pectoral derivation, a temporary one-Float32-step source separation allows the existing exact boundary owner to resolve a point-only contact. Original source points are then restored, including the temporarily shifted point; only generated boundary points are adjusted to remove residual intersections. Failed intermediate candidates are retained. For the left subscapularis, a 0.987-micrometre local correction clears pose-induced junction intersections. These are explicit reference-geometry inferences, not measured-person anatomy.\n\nBoth source surfaces and eight complete-body poses are closed, nondegenerate and self-intersection-free, with no new changed-region crossings against 859 surrounding structures. Eight new native captures verify the actual GPU-deformed geometry, including a full old-to-new pair comparison between the two repaired surfaces. Existing unrelated interfaces remain unqualified.\n\nAll 858 other rendered surfaces, 157 accepted body poses and every value in the 1,000-row, 60-column coupled physiology trace are identical to the preceding run. Physical routes, mass, contact and physiological equations are unchanged.\n\n## Reproduce\n\n    /usr/bin/python3 tools/evidence/native-anatomy-forty-surfaces-1295/reproduce.py\n\nThe pinned Mini runtime and retained assets are required; --prepare-only checks launch inputs. Exact revisions, asset identities, launch, measurements and a continuous 252-frame native framebuffer recording are retained. The seven-layer viewer runs on the SSH Mac mini. No remote-desktop visibility is asserted.\n\nRemaining forearm, neck/back, foot and unexplained-interface work is open. The older five-minute paired physiology run uses older anatomy and does not qualify this updated body.\n')
(O/"retained-files.json").write_text(json.dumps(inventory,indent=2,sort_keys=True)+"\n")
for rel,x in inventory.items():
 raw=(O/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==x["sha256"]
 decoded=gzip.decompress(raw) if x["encoding"]=="gzip" else raw;assert hashlib.sha256(decoded).hexdigest()==x["retained_sha256"]
print(json.dumps({"output":str(O),"retained_files":len(inventory),"summary":summary},indent=2))
