#!/usr/bin/env python3
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}

def oriented_key(face):
    a, b, c = map(int, face)
    return min((a, b, c), (b, c, a), (c, a, b))

def face_counter(faces):
    return collections.Counter(oriented_key(f) for f in np.asarray(faces, dtype=np.int64))

def f32keys(points):
    a = np.ascontiguousarray(np.asarray(points, dtype=np.float32)).view(np.uint32).reshape(-1, 3)
    return {tuple(map(int, row)) for row in a}

def verify_surface_in_volume(surface_path, volume_points, role):
    with np.load(surface_path, allow_pickle=False) as z:
        faces = np.asarray(z["faces"], dtype=np.int64)
        points = np.asarray(z["points_world_m_f32"], dtype=np.float32)
        view = np.asarray(z["points_world_m"])
        if not np.array_equal(view.astype(np.float32).view(np.uint32), points.view(np.uint32)):
            raise SystemExit(f"{role} surface coordinate views differ")
        active = np.unique(faces.reshape(-1))
        if active.size == 0 or active.min() < 0 or active.max() >= len(points):
            raise SystemExit(f"{role} active point index out of range")
        active_keys = f32keys(points[active])
        if len(active_keys) != len(active):
            raise SystemExit(f"{role} active surface rows are not coordinate-unique")
    volume_keys = f32keys(volume_points)
    missing = active_keys - volume_keys
    if missing:
        raise SystemExit(f"{role} active surface coordinates absent from volume: {len(missing)}")
    return {"active_surface_point_rows": int(active.size),
            "active_coordinate_keys_found_in_volume": int(len(active_keys)),
            "missing_active_coordinate_keys": 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--baseline-state", type=Path, required=True)
    ap.add_argument("--proposal-state", type=Path, required=True)
    ap.add_argument("--rollback-receipt", type=Path, required=True)
    ap.add_argument("--audit-report", type=Path, required=True)
    ap.add_argument("--audit-config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()

    state = json.loads(a.state.read_text())
    baseline_state = json.loads(a.baseline_state.read_text())
    proposal_state = json.loads(a.proposal_state.read_text())
    prep = json.loads(a.rollback_receipt.read_text())
    audit = json.loads(a.audit_report.read_text())
    cfg = json.loads(a.audit_config.read_text())

    expected_status = "PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
    if audit.get("status") != expected_status:
        raise SystemExit("exact surface audit did not pass")
    if audit.get("physical_admission") is not False or audit.get("native_steps") != 0:
        raise SystemExit("audit admission fields are not source-only")
    if prep.get("status") != "PASS_WITNESS_COMPONENT_ROLLBACK_PREPARED_FOR_EXACT_AUDIT":
        raise SystemExit("rollback preparation is not the expected version")
    if prep.get("physical_admission") is not False or prep.get("native_steps") != 0:
        raise SystemExit("rollback preparation admission fields are not source-only")
    if prep["output_state"]["sha256"] != sha(a.state):
        raise SystemExit("rollback state hash mismatch")
    if prep["inputs"]["state"]["sha256"] != sha(a.proposal_state):
        raise SystemExit("665 proposal state hash mismatch")
    if prep["inputs"]["source_baseline_state"]["sha256"] != sha(a.baseline_state):
        raise SystemExit("baseline state hash mismatch")
    if audit["config"]["sha256"] != sha(a.audit_config):
        raise SystemExit("audit report/config hash mismatch")
    for role in ("D", "O", "I"):
        if cfg["candidate_surfaces"][role]["sha256"] != sha(state["source_surfaces"][role]["path"]):
            raise SystemExit(f"{role} candidate surface hash mismatch")
        if cfg["candidate_surfaces"][role]["sha256"] != state["source_surfaces"][role]["sha256"]:
            raise SystemExit(f"{role} state surface pin mismatch")

    boundary_results = {}
    active_maps = {}
    volume_paths = {}
    for volume_name, surface_roles in (("Skin", ("O", "I")), ("Fat", ("D", "I"))):
        meta = state[volume_name + "_current"]
        vp = Path(meta["path"])
        volume_paths[volume_name] = pin(vp)
        with np.load(vp, allow_pickle=False) as z:
            points = np.asarray(z["points_world_m_f32"], dtype=np.float32)
            point_view = np.asarray(z["points_world_m"])
            if not np.array_equal(point_view.astype(np.float32).view(np.uint32), points.view(np.uint32)):
                raise SystemExit(f"{volume_name} coordinate views differ")
            actual = np.asarray(z["boundary_faces"], dtype=np.int64)
            desired = np.asarray(z["desired_boundary_faces"], dtype=np.int64)
            ac, dc = face_counter(actual), face_counter(desired)
            missing = sum((dc - ac).values())
            extra = sum((ac - dc).values())
            if missing or extra or ac != dc:
                raise SystemExit(f"{volume_name} oriented actual/desired boundary mismatch: missing={missing} extra={extra}")
            if actual.size and (actual.min() < 0 or actual.max() >= len(points) or desired.min() < 0 or desired.max() >= len(points)):
                raise SystemExit(f"{volume_name} boundary point index out of range")
            boundary_results[volume_name] = {
                "actual_boundary_faces": int(len(actual)),
                "desired_boundary_faces": int(len(desired)),
                "missing": int(missing),
                "extra": int(extra),
                "oriented_boundary_equal": True
            }
            for role in surface_roles:
                active_maps.setdefault(role, {})[volume_name] = verify_surface_in_volume(
                    state["source_surfaces"][role]["path"], points, role
                )

    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    final_state = dict(state)
    final_state["surface_crossings"] = 0
    final_state["exact_context"] = pin(a.audit_report)
    final_state["source_audit_status"] = "PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY"
    for volume_name in ("Skin", "Fat"):
        v = dict(final_state[volume_name + "_current"])
        v["missing_target_faces"] = 0
        v["extra_actual_faces"] = 0
        v["oriented_volume_target_boundary_equal"] = True
        final_state[volume_name + "_current"] = v
    final_state["physical_admission"] = False
    final_state["native_steps"] = 0

    state_path = out / "state.json"
    state_path.write_text(json.dumps(final_state, indent=2, sort_keys=True) + "\n")

    base_skin = int(baseline_state["Skin_current"]["native_quality_failures"])
    base_fat = int(baseline_state["Fat_current"]["native_quality_failures"])
    trial_skin = int(proposal_state["Skin_current"]["native_quality_failures"])
    trial_fat = int(proposal_state["Fat_current"]["native_quality_failures"])
    q = prep["quality"]
    final_skin = int(q["Skin"]["final_native_failures"])
    final_fat = int(q["Fat"]["final_native_failures"])
    if int(q["Skin"]["baseline_native_good_regressions"]) or int(q["Fat"]["baseline_native_good_regressions"]):
        raise SystemExit("rollback branch reports baseline-good regressions")
    if not q["Skin"]["positive_F32_all_cells"] or not q["Fat"]["positive_F32_all_cells"]:
        raise SystemExit("rollback branch is not all-positive")

    receipt = {
        "schema": "h706.bounded-coupled-point-rollback-source-audit.v1",
        "status": "PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",
        "input_verified_state": pin(a.baseline_state),
        "665_proposal_state": pin(a.proposal_state),
        "rollback_preparation_receipt": pin(a.rollback_receipt),
        "rollback_state": pin(a.state),
        "final_state": pin(state_path),
        "audit_config": pin(a.audit_config),
        "exact_805_audit": pin(a.audit_report),
        "finalizer": pin(Path(__file__)),
        "quality_diagnostics": pin(Path(proposal_state["coupled_point_repair"]["diagnostics"]["path"])),
        "candidate_surfaces": {r: pin(state["source_surfaces"][r]["path"]) for r in ("D", "O", "I")},
        "candidate_volumes": volume_paths,
        "changed_vs_complete_retained_OID_both_walls_all185": audit.get("changed_vs_complete_retained_OID_both_walls_all185"),
        "changed_OID_joint_self_pairs": audit.get("changed_OID_joint_self_pairs"),
        "volume_target_boundaries": boundary_results,
        "active_surface_point_crosswalks": active_maps,
        "native_quality": {
            "thresholds_unchanged": True,
            "Skin": {"input_baseline_failures": base_skin, "665_proposal_failures": trial_skin,
                     "after_witness_rollback_failures": final_skin, "baseline_good_regressions": 0,
                     "positive_F32_all_cells": True},
            "Fat": {"input_baseline_failures": base_fat, "665_proposal_failures": trial_fat,
                    "after_witness_rollback_failures": final_fat, "baseline_good_regressions": 0,
                    "positive_F32_all_cells": True}
        },
        "rollback": {
            "witness_pairs": int(prep["witness_count"]),
            "surface_point_rows": prep["surface_rollback_point_counts"],
            "volume_point_rows": prep["volume_rollback_point_counts"]
        },
        "physical_admission": False,
        "native_steps": 0,
        "limitations": ["Source-surface exact context and volume-boundary identity passed; native quality defects remain.",
                        "No mass rebinding or physical/native admission was performed."]
    }
    receipt_path = out / "source-audit-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "state": str(state_path),
                      "state_sha256": sha(state_path), "receipt": str(receipt_path),
                      "receipt_sha256": sha(receipt_path),
                      "native_quality": receipt["native_quality"],
                      "volume_target_boundaries": boundary_results}, sort_keys=True))

if __name__ == "__main__":
    main()

