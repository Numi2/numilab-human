#!/usr/bin/env python3
"""Two-round all-role continuation with full LP row equilibration, v15 witness remapping, 630 audit, and exact rollback.

This is a source-only workflow. It preserves 705's shared coupled point step,
runs the frozen 630 exact audit, and on any exact witness rolls back the entire
moved surface/tet component through both volume layers before re-auditing.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BUILDER = H / "repair_coupled_quality_points_722_witness_v17_rowscaled.py"
AUDITOR = H / "combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py"
PREPARER = H / "bounded625-local-cavity-repair/prepare_bounded_rollback_625.py"
FINALIZER = H / "bounded625-local-cavity-repair/finalize_v7_rollback.py"
ROLLBACK_HELPER = H / "bounded625-local-cavity-repair/bounded_point_component_rollback_625.py"
BASE_DRIVER = H / "fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py"
QUALITY_LP = H / "629-quality-lp/quality_lp_629_v15_rowscaled_witness.py"
TRANSFER_OWNER = H / "transfer_surface_contractions_cross_tissue_470_root.py"
QUALITY_OWNER = H / "fixed_boundary_volume_flip_198_root.py"
ROLE_ORDER = ("D", "O", "I")
VOLUME_ROLES = {"Skin": ("O", "I"), "Fat": ("I", "D")}
EXPECTED_SHA = {
    "builder": "1ade1a680f93a7fc0b5592e68c50aa44d76d648e2c31f7abd568d38e6feca94a",
    "auditor": "82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85",
    "preparer": "7833ee7b08f1956f331ca6481bb87a081f062026bb0ef6c46e0ef654be471f7f",
    "rollback_helper": "6a731dba8635d0129ab08602ba8706aa466924dbfa6ac3a8e60239c14388327f",
    "finalizer": "d6e67fae205c52277324a714cf6f80b264e5bdde21adc50d0a018b13085abd7b",
    "base_driver": "d4fcc4a2693201f7ad3934b62b277a65e6f021a6662182f4ab67e85db4d96c15",
    "quality_lp": "81f85d8a263cf6f1ffd9c23508d06c431f858cdca8749e8153a031ea3410990a",
    "transfer_owner": "e0c0b5ff0f70966431b920f2cb3e469c0bf1d54518068d7ce6e561e5ba16b2e3",
    "quality_owner": "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0",
}
ZERO_AUDIT_STATUS = "PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked_pin(label: str, path: Path) -> dict:
    got = sha(path)
    expected = EXPECTED_SHA[label]
    if got != expected:
        raise ValueError(f"{label} implementation pin mismatch: {got} != {expected}")
    return {"path": str(path.resolve()), "sha256": got}


def load_state(path: Path) -> dict:
    obj = json.loads(path.read_text())
    obj = obj.get("current_verified_state_20261031", obj)
    for key in ("source_surfaces", "Skin_current", "Fat_current"):
        if key not in obj:
            raise ValueError(f"state missing required key: {key}")
    return obj


def load_selected(path: Path, fields) -> dict:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in fields if k not in z]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        return {k: np.array(z[k], copy=True) for k in fields}


def f32points(archive: dict, label: str) -> np.ndarray:
    p = np.asarray(archive["points_world_m_f32"])
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(label + ": malformed Float32 points")
    view = np.asarray(archive["points_world_m"])
    if view.shape != p.shape or not np.isfinite(view).all() or not np.array_equal(view.astype(np.float32).view(np.uint32), p.view(np.uint32)):
        raise ValueError(label + ": Float64/Float32 coordinate views disagree")
    return p


def point_keys(points: np.ndarray) -> list[bytes]:
    p = np.ascontiguousarray(np.asarray(points, dtype="<f4"))
    return [p[i].tobytes() for i in range(len(p))]


def unique_point_map(points: np.ndarray, label: str) -> dict[bytes, int]:
    out = {}
    for i, key in enumerate(point_keys(points)):
        if key in out:
            raise ValueError(f"{label}: duplicate Float32 coordinate rows {out[key]} and {i}")
        out[key] = i
    return out


def canonical_oriented_faces(faces: np.ndarray, label: str) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 1] == f[:, 2]) or np.any(f[:, 2] == f[:, 0]):
        raise ValueError(label + ": malformed/degenerate faces")
    ix = np.argmin(f, axis=1)
    rows = np.arange(len(f))
    c = np.stack((f[rows, ix], f[rows, (ix + 1) % 3], f[rows, (ix + 2) % 3]), axis=1)
    if len(c) > 1:
        order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
        c = c[order]
    if len(c) > 1 and np.any(np.all(c[1:] == c[:-1], axis=1)):
        raise ValueError(label + ": duplicate oriented triangles")
    return c


def canonical_unoriented_faces(faces: np.ndarray, label: str) -> np.ndarray:
    f = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    if len(f) > 1:
        order = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
        f = f[order]
    if len(f) > 1 and np.any(np.all(f[1:] == f[:-1], axis=1)):
        raise ValueError(label + ": duplicate unoriented triangles")
    return f


def run_command(label: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (log_dir / (label + ".stdout.log")).write_text(proc.stdout)
    (log_dir / (label + ".stderr.log")).write_text(proc.stderr)
    (log_dir / (label + ".command.json")).write_text(json.dumps({
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_s": None,
    }, indent=2) + "\n")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        raise RuntimeError(f"{label} exited {proc.returncode}: {tail}")


def run_audit(config: Path, out: Path, log_dir: Path) -> Path:
    run_command("audit775", [sys.executable, str(AUDITOR), "--config", str(config), "--out", str(out)], log_dir)
    report = out / "report.json"
    if not report.is_file():
        raise RuntimeError("630 auditor returned without a report")
    return report


def audit_counts(report_path: Path) -> tuple[dict, bool]:
    report = json.loads(report_path.read_text())
    against = report.get("changed_vs_complete_retained_OID_both_walls_all185", {})
    self_pairs = report.get("changed_OID_joint_self_pairs", {})
    n_against = int(against.get("count", -1))
    n_self = int(self_pairs.get("count", -1))
    if n_against < 0 or n_self < 0:
        raise ValueError("630 exact pair counts are missing")
    zero = report.get("status") == ZERO_AUDIT_STATUS and n_against == 0 and n_self == 0
    return {"status": report.get("status"), "context_pairs": n_against, "joint_self_pairs": n_self}, zero


def signed_volume_by_material(points: np.ndarray, tets: np.ndarray, materials: np.ndarray) -> dict[int, float]:
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(tets, dtype=np.int64)
    m = np.asarray(materials, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 4 or len(m) != len(t) or (len(t) and (t.min() < 0 or t.max() >= len(p))):
        raise ValueError("malformed volume cell/material arrays")
    result = {int(x): 0.0 for x in np.unique(m)}
    for start in range(0, len(t), 150_000):
        stop = min(len(t), start + 150_000)
        q = p[t[start:stop]]
        a = q[:, 1] - q[:, 0]
        b = q[:, 2] - q[:, 0]
        c = q[:, 3] - q[:, 0]
        vol = np.einsum("ij,ij->i", a, np.cross(b, c), optimize=True) / 6.0
        mm = m[start:stop]
        for material in result:
            mask = mm == material
            if np.any(mask):
                result[material] += float(vol[mask].sum(dtype=np.float64))
    return result


def independent_replay(
    candidate_state_path: Path,
    parent_state_path: Path,
    output_report: Path,
    *,
    proposal_state_path: Path | None,
    rollback_plan_path: Path | None,
) -> dict:
    """Replay typed O/I/D to volume F32 maps, boundaries, quality and mass deltas."""
    state = load_state(candidate_state_path)
    parent = load_state(parent_state_path)
    proposal = load_state(proposal_state_path) if proposal_state_path else None
    owner_spec = importlib.util.spec_from_file_location("quality_owner_708", QUALITY_OWNER)
    if owner_spec is None or owner_spec.loader is None:
        raise RuntimeError("cannot import native quality owner")
    owner = importlib.util.module_from_spec(owner_spec)
    sys.modules["quality_owner_708"] = owner
    owner_spec.loader.exec_module(owner)

    surface_cache = {}
    source_receipts = {}
    trust_cap = float(state.get("coupled_point_repair", {}).get("maximum_displacement_m", 2e-5))
    for role in ROLE_ORDER:
        parent_entry = parent["source_surfaces"][role]
        candidate_entry = state["source_surfaces"][role]
        if sha(Path(parent_entry["path"])) != parent_entry["sha256"] or sha(Path(candidate_entry["path"])) != candidate_entry["sha256"]:
            raise ValueError(role + ": source NPZ hash mismatch")
        base = load_selected(Path(parent_entry["path"]), ("points_world_m_f32", "points_world_m", "faces"))
        cand = load_selected(Path(candidate_entry["path"]), ("points_world_m_f32", "points_world_m", "faces"))
        bp, cp = f32points(base, role + " parent"), f32points(cand, role + " candidate")
        bf, cf = np.asarray(base["faces"], np.int64), np.asarray(cand["faces"], np.int64)
        if bp.shape != cp.shape or not np.array_equal(bf, cf):
            raise ValueError(role + ": source point domain/topology changed")
        if bf.size and (bf.min() < 0 or bf.max() >= len(cp)):
            raise ValueError(role + ": source face index outside point domain")
        canonical_unoriented_faces(bf, role + " parent surface")
        canonical_unoriented_faces(cf, role + " candidate surface")
        active = np.unique(cf.reshape(-1))
        unique_point_map(cp, role + " source point domain")
        active_map = unique_point_map(cp[active], role + " active source surface")
        moved = np.flatnonzero(np.any(bp.view(np.uint32) != cp.view(np.uint32), axis=1))
        l1 = np.abs(cp[moved].astype(np.float64) - bp[moved].astype(np.float64)).sum(axis=1)
        max_l1 = float(l1.max(initial=0.0))
        if max_l1 > trust_cap:
            raise ValueError(role + ": source point step exceeds the recorded L1 trust cap")
        if len(moved):
            with np.load(candidate_entry["path"], allow_pickle=False) as z:
                if "typed_parent_candidate_path" not in z or "typed_parent_candidate_sha256" not in z:
                    raise ValueError(role + ": missing typed source parent pin")
                if str(z["typed_parent_candidate_path"][0]) != parent_entry["path"] or str(z["typed_parent_candidate_sha256"][0]) != parent_entry["sha256"]:
                    raise ValueError(role + ": typed source parent pin mismatch")
                if not np.array_equal(z["source_input_face_rows"], np.arange(len(cf), dtype=np.int64)):
                    raise ValueError(role + ": typed source face rows are not the exact parent identity")
                if not np.array_equal(z["source_input_vertex_quotient"], np.arange(len(cp), dtype=np.int64)):
                    raise ValueError(role + ": source vertex quotient is not the exact parent identity")
                for key in z.files:
                    if key.startswith("source_") and key.endswith("_operation_rows"):
                        rows = np.asarray(z[key])
                        if rows.shape != (len(cf),) or np.any(rows != -1):
                            raise ValueError(role + ": point-only step contains stale source operation rows")
        surface_cache[role] = (base, cand, bp, cp, bf, moved, active)
        source_receipts[role] = {
            "parent": parent_entry, "candidate": candidate_entry,
            "active_point_count": int(len(active)), "moved_point_count": int(len(moved)),
            "maximum_L1_displacement_m": max_l1,
            "source_topology_unchanged": True, "active_source_coordinates_unique": True,
        }

    global_active = {}
    for role in ROLE_ORDER:
        _, _, _, cp, _, _, active = surface_cache[role]
        for row in active:
            key = np.ascontiguousarray(cp[int(row)], dtype="<f4").tobytes()
            if key in global_active:
                raise ValueError(f"active source cross-role coordinate alias: {global_active[key]} and {role}/{int(row)}")
            global_active[key] = f"{role}/{int(row)}"

    rollback_receipt = None
    if rollback_plan_path:
        if proposal is None:
            raise ValueError("rollback replay requires the original 705 proposal state")
        plan = json.loads(rollback_plan_path.read_text())
        surface_rollback = plan["rollback_point_rows_by_role"]
        surface_safe = plan["retained_safe_point_rows_by_role"]
        for role in ROLE_ORDER:
            base_entry = parent["source_surfaces"][role]
            prop_entry = proposal["source_surfaces"][role]
            final_entry = state["source_surfaces"][role]
            base = load_selected(Path(base_entry["path"]), ("points_world_m_f32",))
            prop = load_selected(Path(prop_entry["path"]), ("points_world_m_f32",))
            final = load_selected(Path(final_entry["path"]), ("points_world_m_f32",))
            b, p, f = (np.asarray(x["points_world_m_f32"]) for x in (base, prop, final))
            r = np.asarray(surface_rollback[role], np.int64)
            s = np.asarray(surface_safe[role], np.int64)
            if len(r) and not np.array_equal(f[r].view(np.uint32), b[r].view(np.uint32)):
                raise ValueError(role + ": witness component was not fully restored to parent coordinates")
            if len(s) and not np.array_equal(f[s].view(np.uint32), p[s].view(np.uint32)):
                raise ValueError(role + ": safe coupled source coordinates were not preserved")
        rollback_receipt = {
            "rollback_plan": pin(rollback_plan_path),
            "all_role_witness_component_rows_restored": True,
            "all_safe_source_rows_preserved": True,
        }

    replay_volumes = {}
    mass_pending = {}
    for vrole, source_roles in VOLUME_ROLES.items():
        base_entry = parent[vrole + "_current"]
        candidate_entry = state[vrole + "_current"]
        if sha(Path(base_entry["path"])) != base_entry["sha256"] or sha(Path(candidate_entry["path"])) != candidate_entry["sha256"]:
            raise ValueError(vrole + ": volume NPZ hash mismatch")
        fields = ("points_world_m_f32", "points_world_m", "tetrahedra", "material_indices", "boundary_faces", "desired_boundary_faces")
        base = load_selected(Path(base_entry["path"]), fields)
        cand = load_selected(Path(candidate_entry["path"]), fields)
        bp, cp = f32points(base, vrole + " parent volume"), f32points(cand, vrole + " candidate volume")
        bt, ct = np.asarray(base["tetrahedra"], np.int64), np.asarray(cand["tetrahedra"], np.int64)
        bm, cm = np.asarray(base["material_indices"]), np.asarray(cand["material_indices"])
        if bp.shape != cp.shape or not np.array_equal(bt, ct) or not np.array_equal(bm, cm):
            raise ValueError(vrole + ": volume point/tet/material domains changed")
        for key in ("boundary_faces", "desired_boundary_faces"):
            if not np.array_equal(base[key], cand[key]):
                raise ValueError(vrole + ": volume boundary face identity changed")
        # The candidate all-role point step is explicitly identity-mapped.
        with np.load(candidate_entry["path"], allow_pickle=False) as z:
            for key, size in (("output_point_to_input_point_row", len(cp)), ("output_cell_to_input_cell", len(ct)), ("input_cell_to_output_cell", len(ct)), ("retained_input_cell_rows", len(ct)), ("input_node_to_output_node", len(cp)), ("output_node_to_input_node", len(cp))):
                if key not in z or not np.array_equal(np.asarray(z[key], np.int64), np.arange(size, dtype=np.int64)):
                    raise ValueError(vrole + ": volume typed identity map failed: " + key)
            for key in ("current_geometry_parent_path", "current_geometry_parent_sha256"):
                if key not in z:
                    raise ValueError(vrole + ": missing typed volume parent pin: " + key)
            if str(z["current_geometry_parent_path"][0]) != base_entry["path"] or str(z["current_geometry_parent_sha256"][0]) != base_entry["sha256"]:
                raise ValueError(vrole + ": typed volume parent pin mismatch")
            for key in ("output_point_parent_offsets", "output_point_parent_input_cell_rows", "replacement_parent_offsets", "replacement_parent_cell_rows", "replacement_tet_offsets", "replacement_tetrahedra"):
                if key not in z:
                    raise ValueError(vrole + ": missing volume parent-union field: " + key)
            if not np.array_equal(np.asarray(z["output_point_parent_offsets"], np.int64), np.zeros(len(cp) + 1, np.int64)):
                raise ValueError(vrole + ": point parent-union unexpectedly changes")
            if np.asarray(z["output_point_parent_input_cell_rows"]).size or np.asarray(z["replacement_parent_cell_rows"]).size or np.asarray(z["replacement_tetrahedra"]).size:
                raise ValueError(vrole + ": point-only update contains replacement parent cells")
        actual = canonical_oriented_faces(cand["boundary_faces"], vrole + " actual boundary")
        desired = canonical_oriented_faces(cand["desired_boundary_faces"], vrole + " desired boundary")
        canonical_unoriented_faces(cand["boundary_faces"], vrole + " actual boundary")
        canonical_unoriented_faces(cand["desired_boundary_faces"], vrole + " desired boundary")
        if not np.array_equal(actual, desired):
            raise ValueError(vrole + ": oriented actual and desired volume boundaries differ")
        actual_f = np.asarray(cand["boundary_faces"], np.int64)
        if actual_f.size and (actual_f.min() < 0 or actual_f.max() >= len(cp)):
            raise ValueError(vrole + ": boundary face index outside point domain")
        volume_map = unique_point_map(cp, vrole + " volume")
        for role in source_roles:
            _, _, _, sp, sf, _, active = surface_cache[role]
            keys = point_keys(sp[active])
            rows = [volume_map.get(k) for k in keys]
            if any(r is None for r in rows):
                raise ValueError(role + "/" + vrole + ": active source F32 point missing from candidate volume")
            if len(rows) != len(set(rows)):
                raise ValueError(role + "/" + vrole + ": active source point map aliases a volume row")
        parent_volume_map = unique_point_map(bp, vrole + " parent volume")
        baseline_vol = signed_volume_by_material(bp, bt, bm)
        candidate_vol = signed_volume_by_material(cp, ct, cm)
        d, q = owner.measurements(cp, ct)
        passed = np.asarray(owner.passes(d, q), bool)
        if np.any(np.asarray(d) <= 0):
            raise ValueError(vrole + ": nonpositive Float32 determinant")
        bd, bq = owner.measurements(bp, bt)
        baseline_passed = np.asarray(owner.passes(bd, bq), bool)
        regressions = np.flatnonzero(baseline_passed & ~passed)
        if len(regressions):
            raise ValueError(vrole + ": candidate regresses baseline-native-good cells")
        per_material = {}
        for material in np.unique(cm):
            mask = cm == material
            before = baseline_vol[int(material)]
            after = candidate_vol[int(material)]
            delta = after - before
            per_material[str(int(material))] = {
                "cells": int(np.count_nonzero(mask)),
                "baseline_signed_volume_m3": before,
                "candidate_signed_volume_m3": after,
                "delta_signed_volume_m3": delta,
                "relative_delta": (delta / before if before != 0 else None),
                "mass_status": "PENDING_VOLUME_CHANGE_RECONCILIATION",
                "mass_kg_claim": None,
            }
        replay_volumes[vrole] = {
            "cells": int(len(ct)), "points": int(len(cp)),
            "boundary_faces": int(len(actual_f)), "oriented_actual_equals_desired": True,
            "unique_volume_F32_points": True,
            "baseline_native_failures": int(np.count_nonzero(~baseline_passed)),
            "candidate_native_failures": int(np.count_nonzero(~passed)),
            "baseline_good_regressions": 0, "all_F32_cell_determinants_positive": True,
            "per_material": per_material,
        }
        if rollback_plan_path:
            plan = json.loads(rollback_plan_path.read_text())
            rb_rows = np.asarray(plan.get("rollback_volume_point_rows_by_role", {}).get(vrole, []), np.int64)
            proposal_entry = proposal[vrole + "_current"]
            prop = load_selected(Path(proposal_entry["path"]), ("points_world_m_f32",))
            pp = np.asarray(prop["points_world_m_f32"])
            if len(rb_rows) and not np.array_equal(cp[rb_rows].view(np.uint32), bp[rb_rows].view(np.uint32)):
                raise ValueError(vrole + ": witness volume component was not restored")
            safe_rows = np.setdiff1d(np.arange(len(cp), dtype=np.int64), rb_rows, assume_unique=True)
            if len(safe_rows) and not np.array_equal(cp[safe_rows].view(np.uint32), pp[safe_rows].view(np.uint32)):
                raise ValueError(vrole + ": safe volume coordinates were not preserved")
            rollback_receipt[vrole + "_rollback_volume_points"] = int(len(rb_rows))

    if proposal is not None and rollback_plan_path:
        # Replay the rollback volume coordinate rows against the saved 705 trial.
        for vrole in ("Skin", "Fat"):
            plan = json.loads(rollback_plan_path.read_text())
            rb = np.asarray(plan.get("rollback_volume_point_rows_by_role", {}).get(vrole, []), np.int64)
            pentry = proposal[vrole + "_current"]
            fentry = state[vrole + "_current"]
            prop = load_selected(Path(pentry["path"]), ("points_world_m_f32",))
            final = load_selected(Path(fentry["path"]), ("points_world_m_f32",))
            if len(rb) and not np.array_equal(np.asarray(final["points_world_m_f32"])[rb].view(np.uint32), np.asarray(load_selected(Path(parent[vrole + "_current"]["path"]), ("points_world_m_f32",))["points_world_m_f32"])[rb].view(np.uint32)):
                raise ValueError(vrole + ": full coupled volume rollback failed")
        rollback_receipt["both_volume_layers_restored_as_one_coupled_component"] = True

    receipt = {
        "schema": "h706.coupled-quality-708-independent-point-replay.v1",
        "status": "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY",
        "candidate_state": pin(candidate_state_path), "parent_state": pin(parent_state_path),
        "quality_owner": checked_pin("quality_owner", QUALITY_OWNER),
        "source_roles": source_receipts, "volumes": replay_volumes,
        "witness_component_rollback": rollback_receipt,
        "mass_reconciliation": {
            "status": "PENDING_VOLUME_CHANGE_RECONCILIATION",
            "per_material_volume_changes_reported": True,
            "mass_conservation_claimed": False,
            "note": "Per-material geometric volume changes are measured; material mass reconciliation remains pending.",
        },
        "physical_admission": False, "native_steps": 0,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def finalize_verified_state(source_state_path: Path, exact_report: Path, replay_report: Path, output: Path) -> dict:
    state = load_state(source_state_path)
    report = json.loads(exact_report.read_text())
    replay = json.loads(replay_report.read_text())
    if report.get("status") != ZERO_AUDIT_STATUS:
        raise ValueError("cannot finalize a nonzero exact 630 report")
    if replay.get("status") != "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise ValueError("independent replay did not pass")
    final = dict(state)
    final.update({
        "surface_crossings": 0,
        "exact_context": pin(exact_report),
        "source_audit_status": "PASS_SOURCE_ONLY_EXACT_SURFACE_AND_VOLUME_BOUNDARY",
        "independent_point_replay": pin(replay_report),
        "mass_reconciliation": replay["mass_reconciliation"],
        "physical_admission": False,
        "native_steps": 0,
    })
    for vrole in ("Skin", "Fat"):
        v = dict(final[vrole + "_current"])
        v.update({"missing_target_faces": 0, "extra_actual_faces": 0, "oriented_volume_target_boundary_equal": True})
        final[vrole + "_current"] = v
    output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final



def validate_report_config_pin(report_path: Path, config_path: Path) -> dict:
    report_path = Path(report_path).resolve()
    config_path = Path(config_path).resolve()
    if not report_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("resume audit report or config is missing")
    report = json.loads(report_path.read_text())
    config_pin = report.get("config")
    if not isinstance(config_pin, dict):
        raise ValueError("resume audit report has no config pin")
    if Path(config_pin.get("path", "")).resolve() != config_path:
        raise ValueError("resume audit report config path does not match the proposal config")
    config_sha = sha(config_path)
    if config_pin.get("sha256") != config_sha:
        raise ValueError("resume audit report config SHA does not match the proposal config")
    return report


def validate_proposal_parent_pin(proposal_state: dict, baseline_state_path: Path) -> dict:
    expected = pin(Path(baseline_state_path).resolve())
    actual = proposal_state.get("source_state")
    if not isinstance(actual, dict):
        raise ValueError("saved proposal has no source_state parent pin")
    if Path(actual.get("path", "")).resolve() != Path(expected["path"]):
        raise ValueError("saved proposal parent path does not match the requested baseline state")
    if actual.get("sha256") != expected["sha256"]:
        raise ValueError("saved proposal parent SHA does not match the requested baseline state")
    return expected


def resume_saved_proposal(
    baseline_state_path: Path,
    proposal_state_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    supplied_audit_report: Path | None,
) -> dict:
    """Finish a saved proposal without rerunning its SLP producer."""
    baseline_state_path = Path(baseline_state_path).resolve()
    proposal_state_path = Path(proposal_state_path).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not all(p.is_file() for p in (baseline_state_path, proposal_state_path, config_path)):
        raise FileNotFoundError("resume baseline, saved proposal, or context config is missing")

    implementation_pins = {}
    for label, path in (("builder", BUILDER), ("auditor", AUDITOR), ("preparer", PREPARER),
                        ("rollback_helper", ROLLBACK_HELPER), ("finalizer", FINALIZER),
                        ("base_driver", BASE_DRIVER), ("quality_lp", QUALITY_LP),
                        ("transfer_owner", TRANSFER_OWNER), ("quality_owner", QUALITY_OWNER)):
        implementation_pins[label] = checked_pin(label, path)

    baseline = load_state(baseline_state_path)
    proposal = load_state(proposal_state_path)
    parent_pin = validate_proposal_parent_pin(proposal, baseline_state_path)
    config_doc = json.loads(config_path.read_text())
    if config_doc.get("schema") != "h706.combined-current-exact-surface-diff-config.v1":
        raise ValueError("saved proposal config schema mismatch")
    for role in ROLE_ORDER:
        b = config_doc.get("baseline_candidates", {}).get(role, {})
        c = config_doc.get("candidate_surfaces", {}).get(role, {})
        if Path(b.get("path", "")).resolve() != Path(baseline["source_surfaces"][role]["path"]).resolve() or b.get("sha256") != baseline["source_surfaces"][role]["sha256"]:
            raise ValueError(role + ": resume config baseline source pin differs from requested baseline")
        if Path(c.get("path", "")).resolve() != Path(proposal["source_surfaces"][role]["path"]).resolve() or c.get("sha256") != proposal["source_surfaces"][role]["sha256"]:
            raise ValueError(role + ": resume config candidate pin differs from saved proposal")
    recorded_cap = float(proposal.get("coupled_point_repair", {}).get("maximum_displacement_m", 0.0))
    if not 0.0 < recorded_cap <= 0.0005:
        raise ValueError("saved proposal recorded trust cap is outside the authenticated 0.5 mm bound")
    for role in ROLE_ORDER:
        cfg_cap = float(config_doc.get("bounded_point_repair", {}).get(role, {}).get("maximum_displacement_m", -1.0))
        if cfg_cap != recorded_cap:
            raise ValueError(role + ": saved config and proposal trust caps differ")

    output_dir.mkdir(parents=True)
    if supplied_audit_report is None:
        proposal_audit_dir = output_dir / "proposal-exact-775"
        proposal_report = run_audit(config_path, proposal_audit_dir, output_dir / "proposal-audit-logs")
        audit_origin = "executed_775_against_saved_774_config"
    else:
        proposal_report = Path(supplied_audit_report).resolve()
        validate_report_config_pin(proposal_report, config_path)
        audit_origin = "reused_pinned_775_report_for_saved_774_config"
    proposal_summary, proposal_clean = audit_counts(proposal_report)

    rollback_dir = None
    rollback_plan = None
    final_state_input = proposal_state_path
    final_exact_report = proposal_report
    rollback_summary = None
    if not proposal_clean:
        rollback_dir = output_dir / "witness-component-rollback"
        run_command("rollback625", [sys.executable, str(PREPARER), "--state", str(proposal_state_path),
                    "--config", str(config_path), "--audit", str(proposal_report),
                    "--output", str(rollback_dir)], output_dir / "rollback-logs")
        rollback_state = rollback_dir / "state.json"
        rollback_config = rollback_dir / "context-config.json"
        rollback_plan = rollback_dir / "rollback-plan.json"
        prep_receipt = rollback_dir / "rollback-preparation-receipt.json"
        if not all(p.is_file() for p in (rollback_state, rollback_config, rollback_plan, prep_receipt)):
            raise RuntimeError("625 did not produce complete whole-component rollback outputs")
        rollback_audit_dir = output_dir / "rollback-exact-775"
        final_exact_report = run_audit(rollback_config, rollback_audit_dir, output_dir / "rollback-audit-logs")
        rollback_summary, rollback_clean = audit_counts(final_exact_report)
        validate_report_config_pin(final_exact_report, rollback_config)
        if not rollback_clean:
            raise RuntimeError("775 still reports exact witnesses after 625 rollback: " + json.dumps(rollback_summary, sort_keys=True))
        finalized_dir = output_dir / "finalized"
        run_command("finalize625", [sys.executable, str(FINALIZER), "--state", str(rollback_state),
                    "--baseline-state", str(baseline_state_path), "--proposal-state", str(proposal_state_path),
                    "--rollback-receipt", str(prep_receipt), "--audit-report", str(final_exact_report),
                    "--audit-config", str(rollback_config), "--output-dir", str(finalized_dir)],
                    output_dir / "finalize-logs")
        final_state_input = finalized_dir / "state.json"
        if not final_state_input.is_file():
            raise RuntimeError("625 finalizer did not produce a complete rollback state")

    replay_report = output_dir / "independent-point-replay.json"
    replay = independent_replay(final_state_input, baseline_state_path, replay_report,
                                proposal_state_path=proposal_state_path if rollback_plan else None,
                                rollback_plan_path=rollback_plan)
    if replay.get("status") != "PASS_INDEPENDENT_POINT_F32_MAP_BOUNDARY_DUPLICATE_NATIVE_QUALITY_REPLAY":
        raise RuntimeError("saved proposal independent point replay did not pass")
    verified_state_path = output_dir / "verified-state.json"
    finalize_verified_state(final_state_input, final_exact_report, replay_report, verified_state_path)

    receipt = {
        "schema": "h706.saved-774-proposal-resume-775-v1",
        "status": "PASS_775_EXACT_CONTEXT_AND_INDEPENDENT_REPLAY",
        "baseline_state": parent_pin,
        "saved_proposal_state": pin(proposal_state_path),
        "saved_proposal_config": pin(config_path),
        "proposal_trust_cap_m": recorded_cap,
        "proposal_775_report": pin(proposal_report),
        "proposal_775_origin": audit_origin,
        "proposal_775_summary": proposal_summary,
        "proposal_775_clean": proposal_clean,
        "rollback_state": pin(rollback_dir / "state.json") if rollback_dir else None,
        "rollback_plan": pin(rollback_plan) if rollback_plan else None,
        "rollback_775_summary": rollback_summary,
        "final_exact_775_report": pin(final_exact_report),
        "independent_point_replay": pin(replay_report),
        "verified_state": pin(verified_state_path),
        "implementation_pins": implementation_pins,
        "physical_admission": False,
        "native_steps": 0,
        "mass_reconciliation": replay["mass_reconciliation"],
        "limitations": [
            "The saved SLP proposal was reused; no optimization or geometry producer was rerun.",
            "Physical admission and native qualification were not performed."
        ],
    }
    receipt_path = output_dir / "resume-775-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--trust-cap-m", type=float, default=2e-5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--witness-report", type=Path, action="append", default=[])
    ap.add_argument("--no-auto-witness-reports", action="store_true",
                    help="start without a state-adjacent report when its face topology may be stale")
    ap.add_argument("--resume-proposal-state", type=Path)
    ap.add_argument("--resume-config", type=Path)
    ap.add_argument("--resume-audit-report", type=Path)
    args = ap.parse_args()
    if args.resume_proposal_state is not None or args.resume_config is not None or args.resume_audit_report is not None:
        if args.resume_proposal_state is None or args.resume_config is None:
            raise ValueError("resume mode requires --resume-proposal-state and --resume-config")
        resume_saved_proposal(args.state, args.resume_proposal_state, args.resume_config, args.output,
                              supplied_audit_report=args.resume_audit_report)
        return
    if not 1 <= args.rounds <= 2:
        raise ValueError("708 is bounded to one or two coupled rounds")
    if not 0 < args.trust_cap_m <= 5e-4 or not 0 < args.seconds <= 30:
        raise ValueError("trust cap must be <=0.5 mm and solver limit <=30 seconds")
    source_state = args.state.resolve()
    witness_reports = []
    for report_path in args.witness_report:
        resolved = report_path.resolve()
        if resolved not in witness_reports:
            witness_reports.append(resolved)
    adjacent_report = source_state.parent / "proposal-exact-630" / "report.json"
    if (not args.no_auto_witness_reports and adjacent_report.is_file()
            and adjacent_report.resolve() not in witness_reports):
        witness_reports.append(adjacent_report.resolve())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    implementation_pins = {}
    for label, path in (("builder", BUILDER), ("auditor", AUDITOR), ("preparer", PREPARER),
                        ("rollback_helper", ROLLBACK_HELPER), ("finalizer", FINALIZER),
                        ("base_driver", BASE_DRIVER), ("quality_lp", QUALITY_LP),
                        ("transfer_owner", TRANSFER_OWNER), ("quality_owner", QUALITY_OWNER)):
        implementation_pins[label] = checked_pin(label, path)

    source = load_state(source_state)
    out.mkdir(parents=True)
    current_state_path = source_state
    source_doc = json.loads(source_state.read_text())
    if "current_verified_state_20261031" in source_doc:
        current_state_path = out / "initial-flat-current-verified-state.json"
        current_state_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\\n")
    round_receipts = []
    start_all = time.monotonic()
    for index in range(1, args.rounds + 1):
        round_dir = out / f"round-{index:02d}"
        round_dir.mkdir()
        proposal_dir = round_dir / "proposal"
        witnesses_used = list(witness_reports)
        builder_cmd = [sys.executable, str(BUILDER), "--state", str(current_state_path),
                       "--output", str(proposal_dir), "--trust-cap-m", str(args.trust_cap_m),
                       "--seconds", str(args.seconds), "--allow-external-boundary"]
        for report_path in witnesses_used:
            builder_cmd.extend(["--witness-report", str(report_path)])
        run_command("builder722", builder_cmd, round_dir / "builder-logs")
        proposal_state = proposal_dir / "state.json"
        proposal_config = proposal_dir / "context-config.json"
        if not proposal_state.is_file() or not proposal_config.is_file():
            raise RuntimeError("705 produced no complete coupled point proposal")
        proposal_audit_dir = round_dir / "proposal-exact-630"
        proposal_report = run_audit(proposal_config, proposal_audit_dir, round_dir / "proposal-audit-logs")
        proposal_summary, proposal_clean = audit_counts(proposal_report)
        new_witness_report = None
        if not proposal_clean and proposal_report.resolve() not in witness_reports:
            witness_reports.append(proposal_report.resolve())
            new_witness_report = proposal_report.resolve()
        rollback_dir = None
        rollback_plan = None
        final_state_input = proposal_state
        final_exact_report = proposal_report
        if not proposal_clean:
            rollback_dir = round_dir / "witness-component-rollback"
            run_command("rollback625", [sys.executable, str(PREPARER), "--state", str(proposal_state),
                        "--config", str(proposal_config), "--audit", str(proposal_report),
                        "--output", str(rollback_dir)], round_dir / "rollback-logs")
            rollback_state = rollback_dir / "state.json"
            rollback_config = rollback_dir / "context-config.json"
            rollback_plan = rollback_dir / "rollback-plan.json"
            prep_receipt = rollback_dir / "rollback-preparation-receipt.json"
            if not all(p.is_file() for p in (rollback_state, rollback_config, rollback_plan, prep_receipt)):
                raise RuntimeError("625 did not produce complete whole-component rollback outputs")
            rollback_audit_dir = round_dir / "rollback-exact-630"
            final_exact_report = run_audit(rollback_config, rollback_audit_dir, round_dir / "rollback-audit-logs")
            rollback_summary, rollback_clean = audit_counts(final_exact_report)
            if not rollback_clean:
                raise RuntimeError(f"630 still has exact witnesses after whole-component rollback: {rollback_summary}")
            finalized_dir = round_dir / "finalized"
            run_command("finalize625", [sys.executable, str(FINALIZER), "--state", str(rollback_state),
                        "--baseline-state", str(current_state_path), "--proposal-state", str(proposal_state),
                        "--rollback-receipt", str(prep_receipt), "--audit-report", str(final_exact_report),
                        "--audit-config", str(rollback_config), "--output-dir", str(finalized_dir)],
                        round_dir / "finalize-logs")
            final_state_input = finalized_dir / "state.json"
            if not final_state_input.is_file():
                raise RuntimeError("source-only finalizer produced no state")
        else:
            rollback_summary = None

        replay_report = round_dir / "independent-point-replay.json"
        replay = independent_replay(final_state_input, current_state_path, replay_report,
                                    proposal_state_path=proposal_state if rollback_plan else None,
                                    rollback_plan_path=rollback_plan)
        verified_state_path = round_dir / "verified-state.json"
        finalize_verified_state(final_state_input, final_exact_report, replay_report, verified_state_path)
        summary = {
            "round": index,
            "source_state": pin(current_state_path),
            "proposal_state": pin(proposal_state),
            "proposal_630": proposal_summary,
            "proposal_630_clean": proposal_clean,
            "witness_reports_used": [pin(path) for path in witnesses_used],
            "new_witness_report_retained": pin(new_witness_report) if new_witness_report else None,
            "rollback_state": pin(rollback_dir / "state.json") if rollback_dir else None,
            "rollback_plan": pin(rollback_plan) if rollback_plan else None,
            "rollback_630": rollback_summary,
            "independent_point_replay": pin(replay_report),
            "verified_state": pin(verified_state_path),
            "mass_reconciliation": replay["mass_reconciliation"],
        }
        (round_dir / "round-receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        round_receipts.append(summary)
        current_state_path = verified_state_path

    final = {
        "schema": "h706.coupled-quality-722-v16-one-ring-775-audit-source-only-continuation.v1",
        "status": "PASS_SOURCE_ONLY_EXACT_CONTEXT_AND_INDEPENDENT_REPLAY",
        "rounds_requested": args.rounds,
        "rounds_completed": len(round_receipts),
        "implementation_pins": implementation_pins,
        "original_input_state": pin(source_state),
        "initial_flat_parent_state": pin(current_state_path) if current_state_path != source_state else None,
        "rounds": round_receipts,
        "retained_exact_witness_reports": [pin(path) for path in witness_reports],
        "final_state": pin(current_state_path),
        "elapsed_s": time.monotonic() - start_all,
        "physical_admission": False,
        "native_steps": 0,
        "mass_reconciliation": "PENDING_PER_MATERIAL_VOLUME_CHANGE",
        "limitations": [
            "This is source/geometry evidence only; no physical or native admission was performed.",
            "Per-material geometric volume changes are reported; material mass reconciliation remains pending."
        ],
    }
    (out / "722-receipt.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in final.items() if k != "rounds"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

