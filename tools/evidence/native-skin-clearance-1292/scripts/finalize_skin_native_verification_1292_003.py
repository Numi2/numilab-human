from pathlib import Path
import json,gzip,hashlib
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");N=A/"skin-native-integration-1292-001";O=N/"native-verification-003";O.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
rp=N/"native-verification-002/report.json";r=json.loads(rp.read_text())
assert all(r[k] for k in ("complete","inputs_unchanged","all_coupled_trace_rows_exact_parent","all_prior30_geometry_exact"))
assert r["trace_rows"]==1000 and r["trace_columns"]==60 and len(r["poses"])==8
pins={**r["pins"],str(rp):sha(rp),str(Path(__file__)):sha(__file__)}
rows=[]
for p in r["poses"]:
 ap=N/"native-verification-002"/f"step-{p['step']}-exact-audits.json.gz";pins[str(ap)]=sha(ap)
 with gzip.open(ap,"rt") as f:a=json.load(f)
 s,t=a["self"],a["targets"];assert s["count"]==0
 assert s["candidate_world_f32_sha256"]==t["candidate_world_f32_sha256"]==p["native_skin_world_f32_sha256"]
 assert s["face_index_sha256"]==t["face_index_sha256"]==p["skin_face_index_sha256"]
 assert t["target_surface_count"]==len(t["target_audits"])==len(t["pair_changes_by_target"])==859
 assert all(x["count"]==0 and not x["triangle_pairs"] for x in t["target_audits"].values())
 assert all(not x["added"] for x in t["pair_changes_by_target"].values())
 assert t["target_geometry_f32_sha256"]==p["target_audit_summary"]["target_geometry_f32_sha256"]
 rows.append({"step":p["step"],"self_pairs":0,"target_pairs":0,"rendered_targets":859,"changed_skin_faces":len(t["changed_skin_face_rows"]) if isinstance(t["changed_skin_face_rows"],list) else t["changed_skin_face_rows"],"native_skin_world_f32_sha256":p["native_skin_world_f32_sha256"],"audit_sha256":pins[str(ap)]})
e=json.loads((N/"baseline/execution.json").read_text());decl=json.loads((N/"baseline/run-declaration.json").read_text())
assert e["returncode"]==0 and not e["changed_inputs"] and e["native_argv_matches_prepared_cli_preview"] and e["native_environment_matches_prepared_cli_preview"]
assert len(decl["immutable_assets"])==133 and all(sha(p)==v for p,v in decl["immutable_assets"].items())
assert all(sha(p)==v for p,v in pins.items())
out={"scope":"Fresh native Float32 skin proof at all eight retained captures: zero exact self and all859 surrounding target crossings. Reuses hash-bound unchanged-face witnesses, rechecks all changed faces. Geometry-only16s increment; not whole-body/five-minute qualification.","complete":True,"all_native_skin_exact_checks_pass":True,"inputs_unchanged":True,"all_coupled_trace_rows_exact_parent":True,"all_prior30_geometry_exact":True,"trace_rows":1000,"trace_columns":60,"pins":pins,"poses":rows}
(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses")}))
