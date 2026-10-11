from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
sid=int(sys.argv[1]);assert sid in (125,126)
O=A/f"stable-{sid}-current46-selection-017";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
ip=A/f"stable-{sid}-restored-edges-independent-015/report.json";lp=A/f"stable-{sid}-restored-edges-lift-014/report.json";up=A/f"stable-{sid}-restored-edges-union-013/result.json";ap=A/f"stable-{sid}-restored-edges-union-013/stable-{sid}-exact-union.json";rp=A/f"stable-{sid}-restored-edges-parent-012/report.json"
audit=json.loads(ip.read_text());lift=json.loads(lp.read_text());union=json.loads(up.read_text());restored=json.loads(rp.read_text())
assert audit["complete"] and audit["inputs_unchanged"] and audit["source_and_all25_sampled_pose_self_clear"] and audit["all25_retained_targets_no_added_pairs"]
assert audit["all_current46_859_target_keys_scanned"] and audit["no_added_parent_pairs_against_current46_targets"] and audit["all_baseline_forward_capture_replays_within_6e-7m"]
assert len(audit["current46_target_audits"])==8 and len(audit["all25_retained_target_audits"])==25
assert lift["complete"] and lift["inputs_unchanged"] and union["source_geometry_clear"] and union["inputs_unchanged"] and restored["inputs_unchanged"]
r=next(x for x in lift["rows"] if x["stable_id"]==sid);cp=Path(r["candidate_path"]);assert sha(cp)==r["candidate_sha256"]==audit["candidate_sha256"]
assert not r["changed_or_missing_pure_attachment_proxies"]
T=A/"forty-six-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";M=T.with_suffix(".manifest.json")
t=h.load_tissue(T);old=h.row_data(t,sid);z=np.load(cp);mrow=next(x for x in json.loads(M.read_text())["source"]["surfaces"] if x["stable_id"]==sid)
def av(p,f):
 tri=p[f].astype(float);return float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2),float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
area,vol=av(old["positions"],old["faces"]);na,nv=av(z["vertices6"][:,:3],z["faces"])
prp=A/"bounded-halfmm-source-resolution-001/report.json";pr=next(x for x in json.loads(prp.read_text())["rows"] if x["stable_id"]==sid)
reversed_ops={x["reversed_original_operation"] for x in restored["operations"]};remaining=[x for x in pr["attempts"] if x.get("accepted_for_next_offline_iteration") and x["iteration"] not in reversed_ops]
assert len(remaining)==2 and reversed_ops=={0,1,4}
endpoint_bound=max(x["cumulative_ancestry_displacement_m"] for x in remaining)
rounding_reserve=float(np.sqrt(3)*np.abs(np.spacing(z["vertices6"][:,:3].astype(np.float32))).max())
bound=endpoint_bound+rounding_reserve;assert bound<=.0005
pins={}
for doc,key in [(audit,"pins"),(lift,"pins"),(union,"input_sha256"),(restored,"pins")]:
 for p,digest in doc[key].items():
  assert sha(p)==digest,("changed",p)
  if p in pins:assert pins[p]==digest
  pins[p]=digest
for p in [Path(__file__),H,ip,lp,up,ap,rp,prp,cp,T,M,A/f"restore_stable{sid}_source_edges_012.py",A/f"lift_stable{sid}_restored_edges_014.py",A/f"audit_stable{sid}_restored_edges_015.py"]:pins[str(p)]=sha(p)
row={"stable_id":sid,"label":mrow["label"],"member_id":mrow["member_id"],"member_sha256":mrow["member_sha256"],"candidate_path":str(cp),"candidate_sha256":sha(cp),"source_and_sampled_pose_self_clear":True,"pure_attachment_proxy_count":r["pure_attachment_proxy_count"],"changed_or_missing_pure_attachment_proxies":[],"pure_attachment_proxy_meaning":"Exact onehot proxies only; zero does not certify anatomical footprint accuracy.","source_area_m2":area,"candidate_area_m2":na,"source_signed_volume_m3":vol,"candidate_signed_volume_m3":nv,"maximum_displacement_from_union_reference_m":0.0,"restored_original_edges":restored["operations"],"remaining_parent_operations":[x["iteration"] for x in remaining],"current46_capture_pose_audits":audit["current46_target_audits"],"source_union_result_path":str(up),"source_union_ancestry_path":str(ap),"physical_route_force_mass_state_unchanged":True}
report={"schema":"numi.human.reference-surface-selection.v1","scope":"Offline reference-surface selection. Three original source edges/endpoint bindings restored before exact positive-winding union. Two bounded parent collapses remain. Source and25retained contexts self/topology clear and all859actual external target keys scanned each context. Current46eight captures; older contexts retain actual older targets. No native or whole-body qualification.","complete":True,"inputs_unchanged":all(sha(p)==v for p,v in pins.items()),"source_and_sampled_pose_self_clear":True,"all859_changed_star_no_new_parent_pairs":True,"all25_actual_retained_target_contexts_no_new_pairs":True,"candidate_admitted":False,"native_admitted":False,"cumulative_source_displacement_bound_m":bound,"remaining_endpoint_ancestry_bound_m":endpoint_bound,"one_ulp_per_coordinate_union_rounding_reserve_m":rounding_reserve,"bound_interpretation":"Maximum retained independent endpoint ancestry displacement for the two remaining edge collapses plus one Float32 ULP per coordinate Euclidean reserve for generated union points. Exact positive-winding subdivision inherits convex parent-face support; this is a declared engineering reconstruction bound, not source measurement accuracy.","pins":pins,"rows":[row]}
assert report["inputs_unchanged"];(O/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"sid":sid,"candidate_sha256":sha(cp),"report_sha256":sha(O/"report.json"),"source_bound_m":bound,"area_delta":na-area,"volume_delta_m3":nv-vol,"pins":len(pins)}))
