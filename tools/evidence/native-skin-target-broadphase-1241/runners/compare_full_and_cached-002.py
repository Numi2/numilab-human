#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path("/Users/n/numi-human-target-broadphase-1241")
RUN = Path(__file__).resolve().parent
REPORT = RUN / "comparison.json"
ADAPTER = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/fit-attempt-003-adapter-revision-008/fit_17_pose_source_clearance_incremental.py")
PAIR_DOC = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json")
CANDIDATE_VERIFY = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-verification.json")
CANDIDATE_COMPACT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy")

sys.path.insert(0, str(ROOT / "src"))
from numilab_human import common_atlas_skin_clearance as c

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

if REPORT.exists():
    raise RuntimeError("refusing existing output")

ad = loadmod(ADAPTER, "adapter_008_for_broadphase_fixture")
need(ad.c is c, "adapter did not use isolated broadphase worktree module")
base_raw, base_skin = ad.load_skin(ad.OLD_SKIN)
current_raw, current_skin = ad.load_skin(ad.SKIN)
ad.CURRENT_SKIN = current_skin
ad.BASE_SKIN = base_skin
allkeys, nonocular, expected_counts, populations = c._load_target_inventory(ad.INV)
ad.TARGET_KEY_TEXT = {f"{a}:{b}" for a, b in allkeys}
ref_ids = np.unique(current_skin["faces"])
ad.REFERENCE_IDS = ref_ids
helper = ad.load_module(ad.VALID, "broadphase_pack_helper")
tv = ad.load_module(ad.TARGET_VALIDATOR, "broadphase_target_validator")
forward_model = ad.load_module(ad.FORWARD, "broadphase_forward_915")
refined_row, old_row = tv.row(ad.TISS, 64), tv.row(ad.BASE_TISS, 64)
entries = [(i, entry) for i, entry in enumerate(ad.PRIOR_POSES[:3])] + [(9 + j, ("1218_current_run", ad.CURRENT_RUN, step)) for j, step in enumerate(ad.CURRENT_STEPS)]
global_pose_indices = [global_i for global_i, _ in entries]
poses = [ad.read_pose(entry, set(allkeys), helper, tv, refined_row, old_row) for _, entry in entries]
need(len(poses) == 11, "expected only 3 prior and 8 current accepted pose receipts")
pose_by_global = dict(zip(global_pose_indices, poses))
groups = {
    "1217_native_refinement": [i for i,p in enumerate(poses) if p["cohort"] == "1217_native_refinement"],
    "1218_current_run": [i for i,p in enumerate(poses) if p["cohort"] == "1218_current_run"],
}
need([len(groups[k]) for k in groups] == [3,8], "bounded pose cohort partition mismatch")

pair_doc = json.loads(PAIR_DOC.read_text())
need(pair_doc.get("schema") == "numi.human.skin-incremental-exact-target-audit.v1" and len(pair_doc.get("poses", [])) == 17,
     "full-gate fixture pair table schema/pose coverage mismatch")
compact = np.load(CANDIDATE_COMPACT, allow_pickle=False)
need(compact.shape == (len(ref_ids), 3) and compact.dtype.kind == "f", "candidate compact source shape/type mismatch")
candidate_source = current_skin["pos"].astype("<f4").copy()
candidate_source[ref_ids] = np.asarray(compact, dtype="<f4")

selected_worlds = {}
forward_details = {}
receipt_by_group = {}
for group, idxs in groups.items():
    group_poses = [poses[i] for i in idxs]
    map_receipt = group_poses[0]["receipt"]
    receipt_by_group[group] = map_receipt
    for p in group_poses[1:]:
        for field in ("vertex_map", "anatomy_parameters"):
            need(p["receipt"]["skin_source_mapping"][field]["sha256"] == map_receipt["skin_source_mapping"][field]["sha256"],
                 f"{group} map dependency changed within cohort")
    group_base = base_skin if group != "1218_current_run" else current_skin
    captures = np.asarray([p["skin_capture"] for p in group_poses], dtype="<f4").astype(float)
    states = [{"step": p["step"], "body_poses": p["receipt"]["accepted_registered_body_poses"],
               "respiratory_motion": p["receipt"]["accepted_respiratory_motion"]} for p in group_poses]
    fw, fmap, selector = forward_model.build_forward(
        skin=group_base, source_positions=group_base["pos"], referenced_ids=ref_ids,
        captured_by_pose=captures, state_receipts=states,
        initial_body_poses=map_receipt["initial_anatomical_registration"]["body_poses"],
        map_receipt=map_receipt,
        anatomy_parameters_path=map_receipt["skin_source_mapping"]["anatomy_parameters"]["path"],
        respiration_source=ad.RESP, clearance_module=c)
    base_replay = fw(group_base["pos"].astype("<f4"))
    need(base_replay.get("diagnostics", {}).get("admissible") is True, f"{group} captured base forward rejected")
    need(np.array_equal(np.asarray(base_replay["world_positions_by_pose"], dtype="<f4"), captures.astype("<f4")),
         f"{group} captured base source replay not bit exact")
    replay = fw(candidate_source)
    need(replay.get("diagnostics", {}).get("admissible") is True, f"{group} candidate forward rejected")
    world = np.asarray(replay["world_positions_by_pose"], dtype="<f4")
    need(world.shape == (len(idxs), len(ref_ids), 3), f"{group} candidate forward shape mismatch")
    for local, pose_row_index in enumerate(idxs):
        global_idx = global_pose_indices[pose_row_index]
        selected_worlds[global_idx] = world[local].copy()
    forward_details[group] = {"pose_indices": idxs, "base_replay_bit_exact": True,
                              "candidate_world_hashes": [hashlib.sha256(x.tobytes()).hexdigest() for x in world],
                              "forward_recipe_summary": fmap}

need(set(selected_worlds) >= {0,16}, "selected fixture poses missing")
need(hashlib.sha256(candidate_source.tobytes()).hexdigest() == hashlib.sha256(candidate_source.astype("<f4").tobytes()).hexdigest(),
     "candidate source packing is not canonical F32")

skin_faces = np.asarray(current_skin["faces"], dtype=np.int64)
compact_skin_faces = np.searchsorted(ref_ids, skin_faces)
skin_face_sha = c._face_index_sha256(compact_skin_faces)
expected_pair_tables = {i: pair_doc["poses"][i] for i in (0,16)}
verification = json.loads(CANDIDATE_VERIFY.read_text())
expected_world_hashes = {i: verification["candidate_non_target_gate_checks"][i]["candidate_world_f32_sha256"] for i in (0,16)}
step_names = {0: 0, 16: 155000}
results=[]
input_files = [ADAPTER, PAIR_DOC, CANDIDATE_VERIFY, CANDIDATE_COMPACT, Path(ad.OLD_SKIN), Path(ad.SKIN), Path(ad.TISS), Path(ad.MAN),
               Path(ad.NHA), Path(ad.INV), Path(ad.FORWARD), Path(ad.VALID), Path(ad.TARGET_VALIDATOR),
               Path(ad.RESP_FIELD), Path(__file__), ROOT/"src/numilab_human/common_atlas_skin_clearance.py",
               ROOT/"src/numilab_human/cardiac_cavity_intersections.py"]
for i in (0,16):
    pose=pose_by_global[i]
    input_files.extend([pose["pack"], pose["receipt_path"]])
input_files = sorted({Path(p).resolve() for p in input_files})
before={str(p):sha(p) for p in input_files}
for pose_index in (0,16):
    pose=pose_by_global[pose_index]
    target_positions, surfaces, counts = c._pack_surfaces(pose["pack"], set(allkeys))
    target_faces={tuple(map(int,k)):np.asarray(surfaces[k]["faces"],dtype=np.int64) for k in allkeys}
    need(len(target_faces)==859, f"pose {pose_index} target coverage is not859")
    expected=expected_pair_tables[pose_index]
    need(set(expected)=={f"{a}:{b}" for a,b in allkeys}, f"pose {pose_index} expected table incomplete")
    target_sha=c._target_geometry_f32_sha256(target_positions,target_faces)
    world_hash=hashlib.sha256(np.ascontiguousarray(selected_worlds[pose_index],dtype="<f4").tobytes()).hexdigest()
    need(world_hash==expected_world_hashes[pose_index], f"pose {pose_index} candidate world hash differs from retained full-gate verification")
    skin_records=c._exact_surface_records(selected_worlds[pose_index],compact_skin_faces)
    cache=set()
    t=time.perf_counter(); full=c._target_intersection_audit(skin_records,target_faces,target_positions); full_s=time.perf_counter()-t
    need(len(cache)==0, "default full audit unexpectedly received a cache")
    # Independent expected pair identities are retained with the full-gate fixture.
    for key,row in expected.items():
        actual=full[key]
        exp_pairs=[list(map(int,p)) for p in row["triangle_pairs"]]
        if actual["triangle_pairs"] != exp_pairs:
            raise RuntimeError(f"pose {pose_index} retained pair mismatch {key}: {len(actual['triangle_pairs'])} vs {len(exp_pairs)}")
        need(actual["count"] == int(row["count"]), f"pose {pose_index} retained count mismatch {key}")
    cache=set()
    t=time.perf_counter(); validated=c._target_intersection_audit(skin_records,target_faces,target_positions,
        _validated_target_geometry_cache=cache); validate_s=time.perf_counter()-t
    need(len(cache)==1 and target_sha in cache, "full cache-miss validation receipt missing")
    t=time.perf_counter(); fast=c._target_intersection_audit(skin_records,target_faces,target_positions,
        _validated_target_geometry_cache=cache); fast_s=time.perf_counter()-t
    need(full==validated==fast, f"pose {pose_index} full/validated/cached audit output differs")
    pairs=sum(v["count"] for v in fast.values()); aabb=sum(v["aabb_candidate_pairs"] for v in fast.values())
    results.append({"pose_index":pose_index,"accepted_step":step_names[pose_index],
                    "cohort":pose["cohort"],"pack_sha256":sha(pose["pack"]),"receipt_sha256":sha(pose["receipt_path"]),
                    "candidate_world_f32_sha256":hashlib.sha256(selected_worlds[pose_index].tobytes()).hexdigest(),
                    "target_geometry_f32_sha256":target_sha,"target_surface_count":len(fast),
                    "exact_pair_total":pairs,"exact_aabb_candidate_pair_total":aabb,
                    "retained_full_pair_lists_match":True,"full_scan_s":full_s,
                    "first_cache_miss_full_validation_s":validate_s,"cache_hit_broadphase_s":fast_s,
                    "cache_digest_count":len(cache),"same_pair_count_table_full_vs_cache":True})
    del target_positions,surfaces,counts,target_faces,skin_records,full,validated,fast,cache
    gc.collect()
after={str(p):sha(p) for p in input_files}
need(before==after,"one or more bound inputs changed during comparison")
report={"schema":"numi.human.target-audit-broadphase-exact-comparison.v1",
        "status":"pose0_and_late_fixture_exact_parity",
        "scope":"offline exact target audit only; no fit, native launch, or anatomical qualification",
        "worktree":{"path":str(ROOT),"head":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
                    "working_diff_sha256":sha(ROOT/"src/numilab_human/common_atlas_skin_clearance.py")},
        "fixtures":{"pair_table_path":str(PAIR_DOC),"pair_table_sha256":sha(PAIR_DOC),
                    "candidate_compact_source_path":str(CANDIDATE_COMPACT),"candidate_compact_source_sha256":sha(CANDIDATE_COMPACT),
                    "candidate_verification_path":str(CANDIDATE_VERIFY),"candidate_verification_sha256":sha(CANDIDATE_VERIFY),
                    "skin_face_count":len(skin_faces),"skin_face_index_sha256":skin_face_sha,
                    "pose_results":results},
        "forward_reconstruction":{"adapter_path":str(ADAPTER),"adapter_sha256":sha(ADAPTER),
                                  "groups":forward_details},
        "input_hashes_before_after":{p:{"sha256":before[p],"unchanged":before[p]==after[p]} for p in before},
        "inputs_unchanged":before==after,
        "method":"full exact validation on cache miss; subsequent same target geometry digest uses closed Float32 AABB candidate selection, converts selected target rows to exact records, then invokes the same exact integer AABB/intersection predicates. Every output still contains all859 keys.",
        "limitations":["Cache identity is in-process and private to the multipose fit invocation; default callers retain legacy full validation.",
                        "Cache retains geometry digests only, not target records or pair tables.",
                        "No native/physics behavior or fitted candidate qualification is established."]}
REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(REPORT),"status":report["status"],"pose_results":results,"maxrss_raw":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},sort_keys=True))
