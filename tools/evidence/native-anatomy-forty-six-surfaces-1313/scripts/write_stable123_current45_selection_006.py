from pathlib import Path
import hashlib, importlib.util, json, time, sys, os
import numpy as np
ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
BASE = ROOT / "stable-123-source-endpoint-refinement-003"
OUT = BASE / "current45-selection-002"
CAND = BASE / "stable-123-position-unadmitted.npz"
SEED = ROOT / "parent-positive-union-lift-001/stable-123-reference-union-row-patch.npz"
LOCAL_REPORT = BASE / "report.json"
TARGET_REPORT = BASE / "current45-independent-audit-004/report.json"
TARGET_RUNNER = ROOT / "audit_stable123_current45_and_retained_004.py"
TISS43 = ROOT / "forty-five-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS42 = ROOT / "forty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN43 = TISS43.with_suffix(".manifest.json")
MAN42 = TISS42.with_suffix(".manifest.json")
FORWARD = ROOT / "neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
UNION_REPORT = ROOT / "parent-positive-union-123-001/result.json"
UNION_ANCESTRY = ROOT / "parent-positive-union-123-001/stable-123-exact-union.json"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def area_volume(v,f):
    tri=np.asarray(v,dtype=np.float64)[np.asarray(f,dtype=np.int64)]
    return float(.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()),float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
start=time.monotonic()
OUT.mkdir(exist_ok=False)
require(not (OUT/"report.json").exists(),"selection report already exists")
audit=json.loads(TARGET_REPORT.read_text()); local=json.loads(LOCAL_REPORT.read_text())
require(audit.get("complete") and audit.get("inputs_unchanged"),"current45 audit incomplete/changed")
require(audit.get("source_and_all25_sampled_pose_self_clear"),"25-context self/topology gate failed")
require(audit.get("all_baseline_forward_capture_replays_within_6e-7m"),"capture replay gate failed")
require(audit.get("all_current45_859_target_keys_scanned") and audit.get("no_added_parent_pairs_against_current45_targets"),"859 target gate failed")
require(audit.get("all25_retained_targets_no_added_pairs"),"all25 retained external gate failed")
require(len(audit.get("poses",[]))==25 and len(audit.get("current45_target_audits",[]))==8,"expected 25 self contexts and eight target audits")
require(all(int(x["target_keys_scanned"])==859 and x["no_new_parent_pairs"] for x in audit["current45_target_audits"]),"per-pose target gate failed")
require(all(int(x["candidate_self_pair_count"])==0 and int(x["candidate_degenerate_face_count"])==0 and x["candidate_closed_oriented"] for x in audit["poses"]),"per-pose self/topology gate failed")
require(local.get("complete") and local.get("inputs_unchanged") and local.get("selected") and local.get("trials") and local["selected"]["proposal"].get("self_counts")==[0]*25 and not any(local["selected"]["proposal"]["target131_added_pairs"].values()),"local correction proof incomplete")
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import cardiac_cavity_intersections as ci, cardiac_cavity_geometry as cg
sp=importlib.util.spec_from_file_location("forward_positive",FORWARD); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
t43=h.load_tissue(TISS43); row43=h.row_data(t43,123)
t42=h.load_tissue(TISS42); row42=h.row_data(t42,123)
m43=json.loads(MAN43.read_text()); mrow=next(x for x in m43["source"]["surfaces"] if int(x["stable_id"])==123)
m42=json.loads(MAN42.read_text()); mrow42=next(x for x in m42["source"]["surfaces"] if int(x["stable_id"])==123)
same_fields={field:bool(np.array_equal(np.asarray(row42[field]),np.asarray(row43[field]))) for field in ("positions","faces","local","weights","normals")}
require(all(same_fields.values()),"stable123 row changed between current42/current45")
z=np.load(CAND,allow_pickle=False); seed=np.load(SEED,allow_pickle=False)
p=np.ascontiguousarray(z["vertices6"][:,:3],dtype="<f4"); faces=np.asarray(z["faces"],dtype="<i8"); orig=np.asarray(z["face_origins"],dtype="<i8")
weights=np.asarray(z["weights"],dtype="<f4")
require(p.shape==(2463,3) and faces.shape==(4978,3),"unexpected candidate dimensions")
require(np.all(orig>=0) and np.all(orig<len(row43["faces"])),"face ancestry incomplete/out of range")
candidate_sha=sha(CAND); require(candidate_sha==audit["candidate_sha256"]==local["selected"]["candidate_sha256"],"candidate SHA mismatch")
seed_p=np.asarray(seed["vertices6"][:,:3],dtype="<f4")
delta=np.linalg.norm(p.astype(np.float64)-seed_p.astype(np.float64),axis=1)
changed_vertices=np.flatnonzero(np.any(p!=seed_p,axis=1)).astype(int).tolist()
changed_faces=np.flatnonzero(np.any(np.isin(faces,changed_vertices),axis=1)); changed_parents=np.unique(orig[changed_faces])
source_records,source_degenerate=h.exact_rows(row43["positions"],row43["faces"],ci)
source_self=None if source_degenerate else int(ci._audit_pair(source_records,source_records,same_surface=True)["count"])
candidate_records,candidate_degenerate=h.exact_rows(p,faces,ci)
candidate_self=None if candidate_degenerate else int(ci._audit_pair(candidate_records,candidate_records,same_surface=True)["count"])
topology=cg.analyze_topology(p.astype(float).tolist(),faces.tolist())
source_area,source_volume=area_volume(row43["positions"],row43["faces"]); candidate_area,candidate_volume=area_volume(p,faces)
positive=weights>0; pure=int(np.count_nonzero((positive.sum(axis=1)==1)&(np.max(weights,axis=1)==1)))
require(pure==0,"unexpected one-hot attachment rows")
require(all(np.array_equal(z[k],seed[k]) for k in ("faces","face_origins","binding_indices","weights")),"source maps or topology changed")
require(candidate_self==0 and not candidate_degenerate and topology.get("closed_oriented_manifold_candidate"),"source candidate geometry invalid")
require(source_self==39 and not source_degenerate,"raw current45 stable123 source unexpected self/degenerate")
pins=dict(audit["pins"])
for path,digest in local["pins"].items():
 require(sha(path)==digest,"local pin changed: "+path);pins[path]=digest
extra=[CAND,SEED,LOCAL_REPORT,TARGET_REPORT,TARGET_RUNNER,FORWARD,MAN42,UNION_REPORT,UNION_ANCESTRY,ROOT/"refine_stable123_source_endpoints_003.py",ROOT/"stable-123-source-endpoint-refinement-002/report.json",ROOT/"refine_stable123_source_endpoint_002.py",ROOT/"parent-positive-union-lift-001/report.json",ROOT/"bounded-halfmm-source-resolution-001/report.json",TISS42]
for path in extra: pins[str(path)]=sha(path)
for path,expected in audit["pins"].items(): require(sha(path)==expected,"target audit pin changed: "+path)
pins[str(Path(__file__))]=sha(Path(__file__))
for path,expected in pins.items(): require(sha(path)==expected,"input hash mismatch: "+path)
pose_rows=[]
target_by_step={int(y["step"]):y for y in audit["current45_target_audits"]}
for x in audit["poses"]:
 if x.get("group")=="current45":
  tr=target_by_step[int(x["step"])]
  pose_rows.append({"accepted_body_pose_count":int(x["accepted_pose_count"]),"accepted_step":int(x["step"]),"candidate_degenerate_face_count":int(x["candidate_degenerate_face_count"]),"candidate_self_pair_count":int(x["candidate_self_pair_count"]),"candidate_topology_closed_oriented":bool(x["candidate_closed_oriented"]),"external_target_keys_scanned":859,"added_parent_pair_count":0,"all859_changed_star_no_new_parent_pairs":True,"baseline_forward_capture_max_error_m":float(tr["baseline_forward_capture_max_error_m"]),"baseline_forward_capture_pass":float(tr["baseline_forward_capture_max_error_m"])<=6e-7,"pack_sha256":x["pack_sha256"],"receipt_sha256":x["receipt_sha256"]})
require(len(pose_rows)==8,"expected eight current45 capture rows")
row={"stable_id":123,"label":mrow["label"],"member_id":mrow["member_id"],"member_sha256":mrow["member_sha256"],"candidate_path":str(CAND),"candidate_sha256":candidate_sha,"union_reference_path":str(SEED),"union_reference_sha256":sha(SEED),"candidate_face_origins_original_row_range":[int(orig.min()),int(orig.max())],"original_source_face_count":int(len(row43["faces"])),"unique_parent_faces":int(len(np.unique(orig))),"changed_candidate_descendant_face_count":int(len(changed_faces)),"changed_parent_face_count":int(len(changed_parents)),"changed_union_reference_vertex_ids":changed_vertices,"maximum_displacement_from_union_reference_m":float(delta.max(initial=0.0)),"source_area_m2":source_area,"candidate_area_m2":candidate_area,"source_signed_volume_m3":source_volume,"candidate_signed_volume_m3":candidate_volume,"candidate_source_degenerate_face_count":int(len(candidate_degenerate)),"candidate_source_self_pair_count":candidate_self,"raw_source_degenerate_face_count":int(len(source_degenerate)),"raw_source_exact_self_pair_count":source_self,"source_topology_closed_oriented":bool(cg.analyze_topology(np.asarray(row43["positions"],dtype=float).tolist(),np.asarray(row43["faces"],dtype=int).tolist()).get("closed_oriented_manifold_candidate")),"candidate_pure_attachment_proxy_count":pure,"pure_attachment_proxy_count":pure,"changed_or_missing_pure_attachment_proxies":[],"pure_attachment_proxy_audit":{"source_proxy_ids":[],"preserved_count":0,"meaning":"No exact one-hot local weight rows were found; this count does not certify attachment-footprint preservation."},"candidate_normals_changed_vertex_ids":np.flatnonzero(np.any(z["vertices6"][:,3:]!=seed["vertices6"][:,3:],axis=1)).astype(int).tolist(),"route_bindings":mrow.get("body_bindings",[]),"candidate_source_row_equal_42_43":same_fields,"candidate_component_topology":{k:topology.get(k) for k in ("closed_oriented_manifold_candidate","face_component_count","euler_characteristic","boundary_edge_count","nonmanifold_edge_count","orientation_conflict_edge_count")},"source_and_sampled_pose_self_clear":True,"current45_target_audit_path":str(TARGET_REPORT),"current45_target_audit_sha256":sha(TARGET_REPORT),"current45_capture_pose_audits":pose_rows,"prior_25pose_local_report_path":str(LOCAL_REPORT),"prior_25pose_local_report_sha256":sha(LOCAL_REPORT),"source_union_result_path":str(UNION_REPORT),"source_union_result_sha256":sha(UNION_REPORT),"source_union_ancestry_path":str(UNION_ANCESTRY),"source_union_ancestry_sha256":sha(UNION_ANCESTRY)}
require(row["maximum_displacement_from_union_reference_m"]<=0.0005+1e-12,"candidate exceeds local bound")
native45={"composition_path":str(ROOT/"forty-five-surface-native-composition-001"),"composition_tissue_sha256":sha(TISS43),"composition_manifest_sha256":sha(MAN43),"capture_steps":[int(x["accepted_step"]) for x in pose_rows],"external_target_keys_per_capture":859,"offline_candidate_target_audit_path":str(TARGET_REPORT),"offline_candidate_target_audit_sha256":sha(TARGET_REPORT),"poses":pose_rows,"status":"offline audit against accepted captures; no candidate composition/native admission"}
report={"schema":"stable123-native45-current-predicate-selection.v1","complete":True,"inputs_unchanged":True,"source_and_sampled_pose_self_clear":True,"all859_changed_star_no_new_parent_pairs":True,"candidate_admitted":False,"native_admitted":False,"all25_actual_retained_target_contexts_no_new_pairs":True,"cumulative_source_displacement_bound_m":local["selected"]["proposal"]["cumulative_endpoint_displacement_bound_m"],"elapsed_seconds":time.monotonic()-start,"scope":"Offline candidate selection only. Stable133 source geometry and all25 retained accepted-pose self/topology checks pass; the changed-face-star audit scans all859 current45 external targets on eight accepted current45 captures. No asset composition, native admission, or whole-body qualification is claimed.","pins":pins,"input_sha256":dict(pins),"native45":native45,"rows":[row]}
for path,expected in pins.items(): require(sha(path)==expected,"input changed during report generation: "+path)
out=OUT/"report.json"; tmp=OUT/"report.json.tmp"; tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
with tmp.open("rb") as f: os.fsync(f.fileno())
tmp.replace(out)
print(json.dumps({"report_path":str(out),"report_sha256":sha(out),"candidate_sha256":candidate_sha,"input_count":len(pins),"area_delta_m2":candidate_area-source_area,"volume_delta_m3":candidate_volume-source_volume,"max_union_displacement_m":row["maximum_displacement_from_union_reference_m"],"changed_vertices":changed_vertices,"changed_faces":len(changed_faces),"candidate_pure_attachment_proxy_count":pure,"current45_target_scans":len(pose_rows),"all_gates":True,"candidate_admitted":False},sort_keys=True))
