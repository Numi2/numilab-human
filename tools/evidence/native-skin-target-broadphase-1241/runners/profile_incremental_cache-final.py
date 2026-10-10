#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
sys.dont_write_bytecode=True
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
E=Path("/Users/n/numi-human-resting-evidence-20261005")
ROOT=Path("/Users/n/numi-human-target-broadphase-1241")
B=R/"skin-resting-multipose-clearance-1218"
ADAPTER=B/"fit-attempt-003-adapter-revision-008/fit_17_pose_source_clearance_incremental.py"
SIDE=B/"fit-attempt-003-incremental-008/target-audits/iteration-01-backtrack-01-pose-00.json"
TRIAL=B/"fit-attempt-003-incremental-008/trial-starts/iteration-01-backtrack-01-candidate-source.npy"
PAIR_DOC=B/"local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json"
BASE_COMPACT=B/"bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
OUT=Path(__file__).resolve().parent
REPORT=OUT/"profile-report-002.json"

def sha_path(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()

def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError("module-load-failed:"+str(path))
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module); return module

def deep_size(root):
    seen=set(); stack=[root]; total=0
    while stack:
        obj=stack.pop(); oid=id(obj)
        if oid in seen: continue
        seen.add(oid); total+=sys.getsizeof(obj)
        if isinstance(obj,(tuple,list,set,frozenset)): stack.extend(obj)
        elif isinstance(obj,dict): stack.extend(obj.keys()); stack.extend(obj.values())
    return total

def f32sha(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype="<f4").tobytes(order="C")).hexdigest()

if REPORT.exists(): raise RuntimeError("refusing existing report")
sys.path.insert(0,str(ROOT/"src"))
from numilab_human import common_atlas_skin_clearance as worktree_c
ad=loadmod(ADAPTER,"profile_adapter_008"); c,ci=ad.c,ad.ci
if c is not worktree_c: raise RuntimeError("adapter did not retain the current broadphase worktree module")
base_raw,base_skin=ad.load_skin(ad.OLD_SKIN); current_raw,current_skin=ad.load_skin(ad.SKIN)
ad.BASE_SKIN=base_skin; ad.CURRENT_SKIN=current_skin
allkeys,nonocular,expected_counts,populations=c._load_target_inventory(ad.INV)
ad.TARGET_KEY_TEXT={f"{a}:{b}" for a,b in allkeys}
ref_ids=np.unique(current_skin["faces"]); ad.REFERENCE_IDS=ref_ids
helper=ad.load_module(ad.VALID,"profile_pack_helper")
tv=ad.load_module(ad.TARGET_VALIDATOR,"profile_target_validator")
fwmod=ad.load_module(ad.FORWARD,"profile_forward_915")
refined=tv.row(ad.TISS,64); old=tv.row(ad.BASE_TISS,64)
poses=[ad.read_pose(entry,set(allkeys),helper,tv,refined,old) for entry in ad.PRIOR_POSES[:3]]
map_receipt=poses[0]["receipt"]
base_captures=np.asarray([p["skin_capture"] for p in poses],dtype="<f4")
state_rows=[{"step":p["step"],"body_poses":p["receipt"]["accepted_registered_body_poses"],
             "respiratory_motion":p["receipt"]["accepted_respiratory_motion"]} for p in poses]
tb=time.perf_counter()
fw,map_report,selector=fwmod.build_forward(
    skin=base_skin,source_positions=base_skin["pos"],referenced_ids=ref_ids,
    captured_by_pose=base_captures,state_receipts=state_rows,
    initial_body_poses=map_receipt["initial_anatomical_registration"]["body_poses"],
    map_receipt=map_receipt,
    anatomy_parameters_path=map_receipt["skin_source_mapping"]["anatomy_parameters"]["path"],
    respiration_source=ad.RESP,clearance_module=c)
forward_build_s=time.perf_counter()-tb
base_replay=fw(base_skin["pos"].astype("<f4"))
if base_replay["diagnostics"].get("admissible") is not True: raise RuntimeError("base forward rejected")
base_world_captures=np.asarray(base_replay["world_positions_by_pose"],dtype="<f4")
if not np.array_equal(base_world_captures,base_captures): raise RuntimeError("native capture replay not bit exact")
base_compact=np.load(BASE_COMPACT,allow_pickle=False).astype("<f4")
trial_compact=np.load(TRIAL,allow_pickle=False).astype("<f4")
if base_compact.shape!=(len(ref_ids),3) or trial_compact.shape!=base_compact.shape:
    raise RuntimeError(f"compact source shape mismatch {base_compact.shape} {trial_compact.shape}")
base_full=current_skin["pos"].astype("<f4").copy(); base_full[ref_ids]=base_compact
trial_full=current_skin["pos"].astype("<f4").copy(); trial_full[ref_ids]=trial_compact
t=time.perf_counter(); base_result=fw(base_full); base_forward_s=time.perf_counter()-t
t=time.perf_counter(); trial_result=fw(trial_full); trial_forward_s=time.perf_counter()-t
if base_result["diagnostics"].get("admissible") is not True or trial_result["diagnostics"].get("admissible") is not True:
    raise RuntimeError("forward rejected baseline or trial source")
baseline_world=np.asarray(base_result["world_positions_by_pose"],dtype="<f4")[0]
candidate_world=np.asarray(trial_result["world_positions_by_pose"],dtype="<f4")[0]
side=json.loads(SIDE.read_text()); side_summary=side["incremental_audit_summary"]
if f32sha(baseline_world)!=side_summary["baseline_world_f32_sha256"]: raise RuntimeError("baseline world hash mismatch")
if f32sha(candidate_world)!=side_summary["candidate_world_f32_sha256"]: raise RuntimeError("candidate world hash mismatch")

pack=poses[0]["pack"]
t=time.perf_counter(); target_positions,surfaces,pack_counts=c._pack_surfaces(pack,set(allkeys)); pack_load_s=time.perf_counter()-t
target_faces={tuple(map(int,k)):np.asarray(surfaces[k]["faces"],dtype=np.int64) for k in allkeys}
target_sha=c._target_geometry_f32_sha256(target_positions,target_faces)
if target_sha!=side_summary["target_geometry_f32_sha256"]: raise RuntimeError("target geometry hash mismatch")
compact_faces=np.searchsorted(ref_ids,current_skin["faces"])
face_sha=c._face_index_sha256(compact_faces)
pair_doc=json.loads(PAIR_DOC.read_text())
if pair_doc.get("schema")!="numi.human.skin-incremental-exact-target-audit.v1" or len(pair_doc.get("poses",[]))!=17:
    raise RuntimeError("baseline table schema/coverage mismatch")
baseline_table=pair_doc["poses"][0]
baseline_pair_sha=c._baseline_target_pair_table_sha256(baseline_table)
if baseline_pair_sha!=side_summary["baseline_target_audits_sha256"]: raise RuntimeError("baseline pair table hash mismatch")

ordered_keys=sorted(target_faces); face_counts={k:len(target_faces[k]) for k in ordered_keys}
by_count=sorted(ordered_keys,key=lambda k:(face_counts[k],k))
sample_keys=[by_count[round(q*(len(by_count)-1))] for q in (0,.25,.5,.75,1)]
sample_ord={ordered_keys.index(k):f"{k[0]}:{k[1]}" for k in sample_keys}
meter={"active":False,"target_record_calls":0,"prep_s":[],"prep_faces":[],"query_s":[],
       "aabb":[],"exact_pairs":[],"tri_s":0.0,"tri_calls":0,"sample_bytes":{},
       "skin_index_s":0.0,"target_loop_s":0.0}
orig_exact=c._exact_surface_records; orig_target=c._target_intersection_audit
orig_prepare=c._prepare_surface_aabb; orig_query=c._audit_pair_prepared_first
orig_tri=ci.triangle_intersection_points

def exact_wrap(vertices,faces):
    if not meter["active"]: return orig_exact(vertices,faces)
    idx=meter["target_record_calls"]; meter["target_record_calls"]+=1
    t=time.perf_counter(); records=orig_exact(vertices,faces); dt=time.perf_counter()-t
    meter["prep_s"].append(dt); meter["prep_faces"].append(int(len(faces)))
    if idx in sample_ord:
        meter["sample_bytes"][sample_ord[idx]]={"faces":int(len(faces)),"recursive_deep_bytes":deep_size(records)}
    return records

def target_wrap(skin_records,target_faces,positions,**kwargs):
    meter["active"]=True; t=time.perf_counter()
    try: return orig_target(skin_records,target_faces,positions,**kwargs)
    finally:
        meter["target_loop_s"]=time.perf_counter()-t; meter["active"]=False

def prepare_wrap(records):
    t=time.perf_counter(); out=orig_prepare(records); dt=time.perf_counter()-t
    if meter["active"]: meter["skin_index_s"]+=dt
    return out

def query_wrap(first,second,*,same_surface=False):
    if not meter["active"]: return orig_query(first,second,same_surface=same_surface)
    t=time.perf_counter(); out=orig_query(first,second,same_surface=same_surface); dt=time.perf_counter()-t
    meter["query_s"].append(dt); meter["aabb"].append(int(out["aabb_candidate_pairs"]))
    meter["exact_pairs"].append(int(out["count"]))
    return out

def tri_wrap(a,b):
    if not meter["active"]: return orig_tri(a,b)
    t=time.perf_counter(); out=orig_tri(a,b); meter["tri_s"]+=time.perf_counter()-t
    meter["tri_calls"]+=1; return out

c._exact_surface_records=exact_wrap; c._target_intersection_audit=target_wrap
c._prepare_surface_aabb=prepare_wrap; c._audit_pair_prepared_first=query_wrap
ci.triangle_intersection_points=tri_wrap

input_paths=[ADAPTER,SIDE,TRIAL,PAIR_DOC,BASE_COMPACT,ad.CLEARANCE,ROOT/"src/numilab_human/common_atlas_skin_clearance.py",ad.PREDICATE,ad.MODEL,ad.RESP_FIELD,
             ad.FORWARD,ad.VALID,ad.TARGET_VALIDATOR,ad.INV,ad.NHA,ad.TISS,ad.MAN,ad.OLD_SKIN,ad.SKIN,
             pack,poses[0]["receipt_path"],poses[1]["pack"],poses[1]["receipt_path"],poses[2]["pack"],poses[2]["receipt_path"],
             map_receipt["skin_source_mapping"]["vertex_map"]["path"],
             map_receipt["skin_source_mapping"]["anatomy_parameters"]["path"]]
input_paths=sorted({str(Path(p).resolve()) for p in input_paths})
before={p:sha_path(p) for p in input_paths}
def reset_meter():
    meter.update({"active":False,"target_record_calls":0,"prep_s":[],"prep_faces":[],"query_s":[],
                  "aabb":[],"exact_pairs":[],"tri_s":0.0,"tri_calls":0,"sample_bytes":{},
                  "skin_index_s":0.0,"target_loop_s":0.0})

def audit_once(validated_cache):
    reset_meter()
    t=time.perf_counter()
    out=c.audit_incremental_skin_target_intersections(
        baseline_world_positions=baseline_world,candidate_world_positions=candidate_world,
        baseline_faces=compact_faces,candidate_faces=compact_faces,baseline_target_audits=baseline_table,
        target_faces_by_key=target_faces,target_positions=target_positions,
        expected_baseline_world_f32_sha256=side_summary["baseline_world_f32_sha256"],
        expected_face_index_sha256=face_sha,expected_target_geometry_f32_sha256=target_sha,
        expected_baseline_target_audits_sha256=side_summary["baseline_target_audits_sha256"],
        _validated_target_geometry_cache=validated_cache)
    elapsed=time.perf_counter()-t
    profile={"owner_wall_s":elapsed,"target_exact_record_prep_s":sum(meter["prep_s"]),
             "pair_query_s":sum(meter["query_s"]),"triangle_predicate_s":meter["tri_s"],
             "target_audit_loop_s":meter["target_loop_s"],"skin_aabb_index_s":meter["skin_index_s"],
             "target_record_calls":meter["target_record_calls"],"target_record_faces":sum(meter["prep_faces"]),
             "aabb_candidate_pairs":sum(meter["aabb"]),"exact_pair_count":sum(meter["exact_pairs"]),
             "predicate_calls":meter["tri_calls"],"recursive_sample_bytes":meter["sample_bytes"]}
    return out,profile

uncached,uncached_profile=audit_once(None)
validated_cache=set()
cache_miss,cache_miss_profile=audit_once(validated_cache)
cache_hit,cache_hit_profile=audit_once(validated_cache)
if not (uncached==cache_miss==cache_hit):
    raise RuntimeError("incremental uncached/cache-miss/cache-hit outputs differ")
if len(validated_cache)!=1:
    raise RuntimeError("fit-owned target cache did not retain exactly the validated target geometry digest")
result=cache_hit
cache_comparison={"uncached_vs_cache_miss_vs_cache_hit_exact_output_equal":True,
                  "cache_entries_after_miss":1,"cache_entries_after_hit":1,
                  "runs":{"uncached":uncached_profile,"cache_miss_full_validation":cache_miss_profile,
                          "cache_hit_changed_face_broadphase":cache_hit_profile}}
owner_s=cache_hit_profile["owner_wall_s"]
prep_total=cache_hit_profile["target_exact_record_prep_s"]
query_total=cache_hit_profile["pair_query_s"]
pred_total=cache_hit_profile["triangle_predicate_s"]
after={p:sha_path(p) for p in input_paths}
if before!=after: raise RuntimeError("an input changed during profiling")
observed=result["target_audits"]; expected=side["target_audits"]
if set(observed)!=set(expected) or len(observed)!=859: raise RuntimeError("target coverage differs")
diffs=[]
for key in sorted(expected):
    for field in ("triangle_pairs","count","aabb_candidate_pairs"):
        if observed[key].get(field)!=expected[key].get(field):
            diffs.append({"key":key,"field":field,"expected":expected[key].get(field),"observed":observed[key].get(field)})
if diffs: raise RuntimeError("retained sidecar mismatch "+json.dumps(diffs[:5]))

face_total=sum(face_counts.values()); prep_total=sum(meter["prep_s"])
query_total=sum(meter["query_s"]); pred_total=meter["tri_s"]
samples=[x for x in meter["sample_bytes"].values() if x["faces"] > 0]
weighted_bpf=(sum(x["recursive_deep_bytes"] for x in samples)/sum(x["faces"] for x in samples)) if samples else None
projected_one=int(weighted_bpf*face_total) if weighted_bpf else None
pair_total=sum(int(v["count"]) for v in observed.values())
aabb_total=sum(int(v["aabb_candidate_pairs"]) for v in observed.values())
rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
report={
 "schema":"numi.human.incremental-target-audit-single-pose-profile.v1",
 "status":"complete_exact_pair_identity_match",
 "scope":"one offline exact incremental target scan, pose 0 at accepted step 0; no native simulation",
 "trial_key":side["trial_key"],"accepted_step":side["accepted_step"],
 "profile_script":{"path":str(Path(__file__).resolve()),"sha256":sha_path(__file__),"bytes":Path(__file__).stat().st_size},
 "input_pins":{p:{"sha256":before[p],"bytes":Path(p).stat().st_size} for p in input_paths},
 "inputs_unchanged_before_after":True,
 "reconstruction":{
   "forward_model_build_s":forward_build_s,"base_forward_s":base_forward_s,"trial_forward_s":trial_forward_s,
   "target_pack_load_s":pack_load_s,"base_native_capture_replay_bit_exact":True,
   "baseline_world_f32_sha256":f32sha(baseline_world),"expected_baseline_world_f32_sha256":side_summary["baseline_world_f32_sha256"],
   "candidate_world_f32_sha256":f32sha(candidate_world),"expected_candidate_world_f32_sha256":side_summary["candidate_world_f32_sha256"],
   "target_geometry_f32_sha256":target_sha,"expected_target_geometry_f32_sha256":side_summary["target_geometry_f32_sha256"],
   "baseline_pair_table_sha256":baseline_pair_sha,"expected_baseline_pair_table_sha256":side_summary["baseline_target_audits_sha256"],
   "forward_model_report":map_report},
 "incremental_cache_comparison":cache_comparison,
 "exact_replay":{
   "target_rows":len(observed),"pair_identity_match_to_retained_sidecar":True,
   "per_target_pair_lists_counts_and_aabb_candidates_equal":True,
   "pair_total":pair_total,"sidecar_pair_total":sum(int(v["count"]) for v in expected.values()),
   "aabb_candidate_total":aabb_total,"sidecar_aabb_candidate_total":sum(int(v["aabb_candidate_pairs"]) for v in expected.values()),
   "owner_summary":{k:v for k,v in result.items() if k not in ("target_audits","pair_changes_by_target")}},
 "timing_s":{
   "exact_incremental_owner_wall":owner_s,"target_exact_record_prep":prep_total,
   "pair_query_including_predicates":query_total,"triangle_predicate_only":pred_total,
   "query_minus_predicate_approx_aabb_and_python_overhead":max(0.0,query_total-pred_total),
   "skin_aabb_index_build":meter["skin_index_s"],"target_audit_loop":meter["target_loop_s"],
   "predicate_calls":meter["tri_calls"],"target_record_calls":meter["target_record_calls"],
   "prep_fraction":prep_total/owner_s if owner_s else None,
   "query_fraction":query_total/owner_s if owner_s else None,
   "note":"Timers wrap frozen owner functions; predicate-only time is measured around triangle_intersection_points. Python instrumentation and shared-host contention affect wall time."},
 "memory_estimate":{
   "target_surface_count":len(observed),"target_triangle_records_per_pose":face_total,
   "samples_method":"recursive sys.getsizeof with object-identity de-duplication on five target record lists selected by face-count quantile",
   "samples":meter["sample_bytes"],"sample_weighted_bytes_per_record":weighted_bpf,
   "estimated_cached_exact_records_bytes_one_pose":projected_one,
   "estimated_cached_exact_records_bytes_17_poses":17*projected_one if projected_one else None,
   "maxrss_raw_platform_units":rss,"host_ram_gib":24,"active_fitter_estimated_gib":2.5,
   "limits":"Projection estimates only returned exact-record Python object graphs; not allocator/RSS, caches, arrays, or AABB trees. Across-pose exact records cannot be reused unless target geometry hashes match."},
 "limits":["Single pose only; replay ran concurrently with the existing fitter on the shared Mac mini.",
           "No source, owner, active fit, or native input was modified.",
           "Exact equality is against the frozen pose-0 retained sidecar, not an anatomical qualification."]}
REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(REPORT),"status":report["status"],"owner_s":owner_s,"prep_s":prep_total,
 "query_s":query_total,"predicate_s":pred_total,"face_records":face_total,"pairs":pair_total,
 "aabb_candidates":aabb_total,"one_pose_cache_estimate":projected_one,
 "17_pose_cache_estimate":17*projected_one if projected_one else None,"maxrss":rss},sort_keys=True))

