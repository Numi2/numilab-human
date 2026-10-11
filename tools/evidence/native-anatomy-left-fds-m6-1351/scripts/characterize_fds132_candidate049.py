from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np

ROOT = Path("/Users/n/numi-human-fds132-m6-1350/attempt-049")
OUT = ROOT / "source-characterization/result-005"
OUT.mkdir(parents=True, exist_ok=False)
A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
sys.path.insert(0, "/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import cardiac_cavity_geometry as cg

H = A / "neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
C = ROOT / "result-001/stable-132-candidate049-current-pq-retained-root-union-candidate.npz"
B = A / "remaining-original-boundary-batch-001/stable-132/source-lift-001/stable-132-geometric-reference-row-patch.npz"
N = A / "right-pq-native-mtp-two-001"
T = N / "assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CURRENT_AUDIT = ROOT / "audit-current-pq/result-001/report.json"
ALL25_AUDIT = ROOT / "audit-all25/result-002/report.json"
SOLVE = ROOT / "result-001/proposal-report.json"
PREFLIGHT = ROOT / "result-001/all-twenty-two-active-set-preflight.json"
SOLVER = ROOT / "solve_candidate049_all25_root_union.py"
WRAPPER = ROOT / "captured_root_diagnostic_wrapper_049.py"

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
candidate_sha = "7d072ccbadbb13fdcad255b10c6ab1bf692c442b47fb12b917a8e68e80d7d0cf"
assert sha(C) == candidate_sha
current_audit = json.loads(CURRENT_AUDIT.read_text())
all25_audit = json.loads(ALL25_AUDIT.read_text())
solve_report = json.loads(SOLVE.read_text())
preflight = json.loads(PREFLIGHT.read_text())
assert current_audit.get("complete") and current_audit.get("inputs_unchanged")
assert current_audit.get("candidate_sha256") == candidate_sha
assert current_audit.get("source_and_current_pq8_sampled_pose_self_clear")
assert current_audit.get("current_pq_all_external_target_keys_scanned")
assert current_audit.get("no_added_parent_pairs_against_current_pq_targets")
assert all25_audit.get("complete") and all25_audit.get("inputs_unchanged")
assert all25_audit.get("candidate_sha256") == candidate_sha
assert all25_audit.get("source_and_all25_sampled_pose_self_clear")
assert all25_audit.get("all_current_pq_865_target_keys_scanned")
assert all25_audit.get("all25_retained_targets_no_added_pairs")
assert solve_report.get("candidate_sha256") == candidate_sha
assert solve_report.get("candidate_admitted") is False
assert solve_report.get("inputs_unchanged") is True
assert solve_report.get("active_external_row_total") == 514
assert preflight.get("candidate_sha256") == "772afcbd2c8492780d76a69bec214b04cad32707011967883b1c6183e5a2001e"
assert preflight.get("external_child_pair_rows_total") == 514
assert preflight.get("frozen_owner_pose_count_bound") == 17
assert preflight.get("in_memory_pose_count_bound") == 22
assert preflight.get("pose_count_resource_guard_extended_in_memory_only") is True
assert preflight.get("geometric_guards_changed") is False
assert preflight.get("frozen_helper_file_unchanged") is True

spec = importlib.util.spec_from_file_location("positive_forward_132", H)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.TISS = T
h.MAN = T.with_suffix(".manifest.json")
tissue = h.load_tissue(T)
row = h.row_data(tissue, 132)
candidate = dict(np.load(C, allow_pickle=False))
reference = dict(np.load(B, allow_pickle=False))
p = np.ascontiguousarray(candidate["vertices6"][:, :3], dtype="<f4")
basep = np.ascontiguousarray(reference["vertices6"][:, :3], dtype="<f4")
faces = np.asarray(candidate["faces"], dtype=np.int64)
base_faces = np.asarray(reference["faces"], dtype=np.int64)
weights = np.asarray(candidate["weights"], dtype="<f4")
base_weights = np.asarray(reference["weights"], dtype="<f4")
local = np.asarray(candidate["binding_indices"], dtype=np.uint32)
base_local = np.asarray(reference["binding_indices"], dtype=np.uint32)
origins = np.asarray(candidate["face_origins"], dtype=np.int64)
base_origins = np.asarray(reference["face_origins"], dtype=np.int64)
assert np.array_equal(faces, base_faces)
assert np.array_equal(origins, base_origins)
assert np.array_equal(local, base_local)
assert np.array_equal(weights, base_weights)
assert np.all(origins >= 0) and np.all(origins < len(row["faces"]))
assert np.all(local[weights > 0] < int(row["bc"]))
assert np.all(local[weights <= 0] == np.uint32(0xffffffff))
assert np.isfinite(p).all() and np.isfinite(basep).all()
assert np.all(weights >= 0)
assert np.max(np.abs(weights.sum(axis=1, dtype=np.float64) - 1.0)) < 2e-7
assert np.all(weights[weights > 0] > 0)
onehot = np.flatnonzero(weights.max(axis=1) >= 0.999999)
assert np.array_equal(p[onehot], basep[onehot])
assert np.array_equal(weights[onehot], base_weights[onehot])

def metrics(v):
    top = cg.analyze_topology(v.astype(float).tolist(), faces.tolist())
    volumes = []
    for comp in top["face_components"]:
        tri = v[faces[comp]].astype(np.float64)
        tri -= v[np.unique(faces[comp])].mean(axis=0, dtype=np.float64)
        volumes.append(float(np.einsum("ij,ij->i", tri[:, 0],
                       np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0))
    tri = v[faces].astype(np.float64)
    area = float(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0],
                                         tri[:, 2]-tri[:, 0]), axis=1).sum()/2.0)
    return {"closed_oriented": bool(top["closed_oriented_manifold_candidate"]),
            "component_count": len(volumes), "component_volumes_m3": volumes,
            "volume_m3": float(sum(volumes)), "area_m2": area}

before = metrics(basep)
after = metrics(p)
delta = p.astype(np.float64) - basep.astype(np.float64)
support = np.unique(local[weights > 0]).astype(np.int64)
pins = {str(x): sha(x) for x in [Path(__file__), H, C, B, T, T.with_suffix(".manifest.json"),
    Path(cg.__file__), CURRENT_AUDIT, ALL25_AUDIT, SOLVE, PREFLIGHT, SOLVER, WRAPPER]}
poses = []
capture_dir = N / "baseline/native-run/accepted-geometry"
for step in (0, 4767, 5023, 5599, 6207, 6815, 7423, 8000):
    pack = capture_dir / f"step-{step}.mrvpack"
    receipt = capture_dir / f"step-{step}.receipt.json"
    rc = json.loads(receipt.read_text())
    assert (rc.get("pack_file_sha256") or rc.get("accepted_pack_file_sha256")) == sha(pack)
    pins[str(pack)] = sha(pack)
    pins[str(receipt)] = sha(receipt)
    bodyposes = rc.get("accepted_body_poses") or rc.get("accepted_registered_body_poses")
    assert isinstance(bodyposes, list) and len(bodyposes) == 157
    pm = {int(x["body_index"]):
          (np.asarray(x["position_m"], dtype=np.float32),
           np.asarray(x["quaternion_xyzw"], dtype=np.float32)) for x in bodyposes}
    candidate_world = h.forward(p, local, weights, row, tissue, pm)
    reference_world = h.forward(basep, base_local, base_weights, row, tissue, pm)
    d = candidate_world.astype(np.float64) - reference_world.astype(np.float64)
    world_metrics = metrics(np.ascontiguousarray(candidate_world, dtype="<f4"))
    ref_world_metrics = metrics(np.ascontiguousarray(reference_world, dtype="<f4"))
    poses.append({"step": step, "accepted_pose_count": len(bodyposes),
        "max_candidate_vs_source_posed_vertex_delta_m": float(np.linalg.norm(d, axis=1).max()),
        "rms_candidate_vs_source_posed_vertex_delta_m": float(np.sqrt(np.mean(np.sum(d*d, axis=1)))),
        "candidate_posed_geometry": world_metrics,
        "source_posed_geometry": ref_world_metrics,
        "posed_area_relative_change": world_metrics["area_m2"]/ref_world_metrics["area_m2"]-1.0,
        "posed_volume_relative_change": world_metrics["volume_m3"]/ref_world_metrics["volume_m3"]-1.0})

out = {
    "schema": "stable132-candidate049-source-characterization.v1",
    "scope": "Offline inferred-geometry and display-binding characterization against the retained stable132 source-lift patch. This is not measured anatomy, active forearm function, or native composition qualification.",
    "candidate_path": str(C), "candidate_sha256": sha(C),
    "reference_patch_path": str(B), "reference_patch_sha256": sha(B),
    "current_pq_audit_path": str(CURRENT_AUDIT), "current_pq_audit_sha256": sha(CURRENT_AUDIT),
    "all25_audit_path": str(ALL25_AUDIT), "all25_audit_sha256": sha(ALL25_AUDIT),
    "source_reference_geometry": before, "candidate_source_geometry": after,
    "faces_exact_source_reference": np.array_equal(faces, base_faces),
    "face_origins_exact_source_reference": np.array_equal(origins, base_origins),
    "lineage_parent_face_range_valid": bool(np.all((origins >= 0) & (origins < len(row["faces"])))),
    "candidate_face_count": int(len(faces)),
    "source_parent_face_count": int(len(row["faces"])),
    "binding_indices_exact_source_reference": np.array_equal(local, base_local),
    "weights_exact_source_reference": np.array_equal(weights, base_weights),
    "max_abs_weight_change_from_source_reference": float(np.max(np.abs(weights.astype(np.float64)-base_weights.astype(np.float64)))),
    "changed_weight_vertex_count": int(np.count_nonzero(np.any(weights != base_weights, axis=1))),
    "binding_support_exact_source_reference": np.array_equal(np.unique(local[weights > 0]), np.unique(base_local[base_weights > 0])),
    "active_binding_lane_count": int(np.count_nonzero(weights > 0)),
    "active_binding_vertex_count": int(np.count_nonzero(np.any(weights > 0, axis=1))),
    "positive_weight_min": float(weights[weights > 0].min()),
    "positive_weight_max": float(weights[weights > 0].max()),
    "max_row_weight_sum_error": float(np.max(np.abs(weights.sum(axis=1, dtype=np.float64)-1.0))),
    "pure_onehot_proxy_count": int(len(onehot)),
    "pure_onehot_proxy_coordinates_and_weights_exact_source_reference": True,
    "onehot_authority_note": "Stable132 source-lift has no pure one-hot vertices; the proxy count is zero and no attachment proxy authority is inferred.",
    "max_vertex_displacement_from_source_mm": float(np.linalg.norm(delta, axis=1).max()*1000.0),
    "rms_vertex_displacement_from_source_mm": float(np.sqrt(np.mean(np.sum(delta*delta, axis=1)))*1000.0),
    "max_axis_coordinate_displacement_from_source_mm": float(np.abs(delta).max()*1000.0),
    "source_relative_area_change": float(after["area_m2"]/before["area_m2"]-1.0),
    "source_relative_volume_change": float(after["volume_m3"]/before["volume_m3"]-1.0),
    "binding_changes": [],
    "poses": poses,
    "all_components_positive": all(v > 0 for m in [before, after] +
        [x["candidate_posed_geometry"] for x in poses] for v in m["component_volumes_m3"]),
    "complete": True,
    "status": "complete_unadmitted_characterization",
    "candidate_still_unadmitted": True,
    "inputs_unchanged": all(sha(x) == old for x, old in ((Path(k), v) for k, v in pins.items())),
    "pins": pins
}
assert out["all_components_positive"]
assert out["inputs_unchanged"]
(OUT / "report.json").write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps({k:v for k,v in out.items() if k not in ("pins","poses","source_reference_geometry","candidate_source_geometry")}, sort_keys=True), flush=True)
