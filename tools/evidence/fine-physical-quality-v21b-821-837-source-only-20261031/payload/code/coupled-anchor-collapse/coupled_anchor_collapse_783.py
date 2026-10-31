#!/usr/bin/env python3
"""783 state runner: defer endpoint orientation to the fitted-anchor proof.

This imports the frozen 782 state wrapper and applies one byte-pinned source
rewrite to the 781 prepare_operation guard. It removes only the early test of
replacement tet orientation at the old keep coordinate. Tet row order remains
unchanged; all final F32 native, exact-positive, paired-volume, source, 775,
697, and physical-admission gates remain owned by the pinned implementation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "coupled_anchor_collapse_782.py"
WRAPPER_SHA256 = "f23abf417b9fd1ce017837c32912e7d17a3a485cf941f0230d403faf67524b23"
ENGINE = HERE / "coupled_anchor_collapse_781.py"
ENGINE_SHA256 = "739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb"
OLD_GUARD = '        if np.any(measurements(v["p"],ordered)[0]<=0):return {"status":"REJECT_ORIENTATION","role":role}\n'
CHANGE_ID = "defer_only_pre_fit_endpoint_orientation_rejection"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def import_frozen_wrapper():
    if sha(WRAPPER) != WRAPPER_SHA256:
        raise RuntimeError("frozen 782 wrapper SHA mismatch")
    if sha(ENGINE) != ENGINE_SHA256:
        raise RuntimeError("frozen 781 operation engine SHA mismatch")
    spec = importlib.util.spec_from_file_location("coupled_anchor_collapse_782_for_783", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen 782 wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = import_frozen_wrapper()
M = G.M
ORIGINAL_PREPARE_OPERATION = M.prepare_operation
_BASE_RUN_FROM_CONFIG = M.run_from_config


def install_operation_derivative():
    """Compile a pinned, exact one-line derivative in the 781 module namespace."""
    source = inspect.getsource(ORIGINAL_PREPARE_OPERATION)
    n = source.count(OLD_GUARD)
    if n != 1:
        raise RuntimeError(f"expected exactly one endpoint orientation guard, found {n}")
    derivative = source.replace(
        OLD_GUARD,
        "        # 783: preserve this signed tet order; judge orientation only after anchor fit.\n",
        1,
    )
    code = compile(derivative, str(Path(__file__).resolve()), "exec")
    exec(code, M.__dict__)
    if M.prepare_operation is ORIGINAL_PREPARE_OPERATION:
        raise RuntimeError("783 operation derivative was not installed")
    return {"path": str(Path(__file__).resolve()), "sha256": sha(__file__),
            "frozen_engine_path": str(ENGINE), "frozen_engine_sha256": ENGINE_SHA256,
            "frozen_wrapper_path": str(WRAPPER), "frozen_wrapper_sha256": WRAPPER_SHA256,
            "source_transform": CHANGE_ID,
            "removed_source_line": OLD_GUARD.strip(),
            "source_guard_occurrences": n,
            "tet_order_policy": "unchanged signed order; no swaps or absolute determinants",
            "final_gates": ["F32 positive", "native q and minimum volume", "exact positive",
                            "exact paired volume", "source normal/apex", "complete boundary",
                            "775 all185", "actual 697/217"],
            "physical_admission": False, "native_steps": 0}


OVERRIDE = install_operation_derivative()


def _run_from_config_783(config_path, out_dir, edges_override=None, prune_receipt=None):
    report = _BASE_RUN_FROM_CONFIG(config_path, out_dir, edges_override=edges_override,
                                   prune_receipt=prune_receipt)
    state_path = Path(out_dir).resolve() / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        state["producer"] = {"path": OVERRIDE["path"], "sha256": OVERRIDE["sha256"]}
        state["operation_prepare_override"] = OVERRIDE
        state["base_operation_engine"] = {"path": str(ENGINE), "sha256": ENGINE_SHA256}
        M._write_json(state_path, state)
        report.update({"producer": state["producer"],
                       "operation_prepare_override": OVERRIDE,
                       "base_operation_engine": state["base_operation_engine"]})
    return report


M.run_from_config = _run_from_config_783


def orientation_rescue_fixture():
    """Complete two-tissue edge-collapse regression for the removed early guard."""
    side = 2.0 ** -12
    height = side / 65536.0
    points = np.asarray([
        [0.0, -side, -height / 2.0],  # old keep; new surviving tet is inverted here
        [0.0, 0.0, height],          # drop
        [0.0, 0.0, 0.0],
        [side, 0.0, 0.0],
        [0.0, side, 0.0],
    ], dtype=np.float32)
    raw_tets = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    tets = M.M.orient(points, raw_tets)
    faces = M.boundary(tets)
    surface = {"points_world_m_f32": points.copy(), "faces": faces.copy()}
    volumes = {
        role: {"points_world_m": points.copy(), "tetrahedra": tets.copy(),
               "material_indices": np.full(len(tets), 16, dtype=np.int32),
               "desired_boundary_faces": faces.copy()}
        for role in ("Skin", "Fat")
    }
    mapped = np.where(tets == 1, 0, tets)
    deleted = np.any(np.diff(np.sort(mapped, axis=1), axis=1) == 0, axis=1)
    endpoint_replacements = mapped[~deleted]
    endpoint_det = M.measurements(points, endpoint_replacements)[0]
    if not np.any(endpoint_det <= 0):
        raise AssertionError("fixture does not contain an inverted endpoint replacement tet")

    volume_cache = {role: M.cache_volume(volumes[role]) for role in ("Skin", "Fat")}
    old_result = ORIGINAL_PREPARE_OPERATION(surface, volume_cache, 0, 1, move_cap_m=0.0005)
    if old_result.get("status") != "REJECT_ORIENTATION":
        raise AssertionError(f"frozen guard control changed: {old_result.get('status')}")
    result = M.prepare_operation(surface, volume_cache, 0, 1, move_cap_m=0.0005)
    if result.get("status") != "PASS":
        raise AssertionError(f"783 movable-anchor fixture failed: {result.get('status')}: {result}")

    final_by_role = {}
    for role in ("Skin", "Fat"):
        op = result["volumes"][role]
        det, q = M.measurements(op["points"], op["new_tets"])
        if np.any(det <= 0) or not M.M.passes(det, q) or not M.exact_positive(op["points"], op["new_tets"]):
            raise AssertionError(f"{role} final replacement tet failed a gate")
        final_by_role[role] = {"determinants": det.tolist(), "native_q": q.tolist(),
                               "exact_delta_integer": str(op["exact_delta"]),
                               "old_bad_cells": op["old_bad"]}
    if sum(int(result["volumes"][r]["exact_delta"]) for r in ("Skin", "Fat")) != 0:
        raise AssertionError("fixture paired exact volume delta is not zero")
    if result["fit"]["anchor_displacement_m"] > 0.0005:
        raise AssertionError("fixture exceeded unchanged movement cap")
    if result["fit"]["minimum_new_native_q"] < M.NATIVE_Q:
        raise AssertionError("fixture did not meet unchanged native q gate")
    # Full local topology is retained by the operation. Verify its replacement
    # tetra boundary exactly equals the quotient of the old local volume boundary.
    for role in ("Skin", "Fat"):
        st = result["volumes"][role]
        actual = M.boundary(st["new_tets"])
        old = volume_cache[role]["t"]
        oq = np.where(old == 1, 0, old)
        od = np.any(np.diff(np.sort(oq, axis=1), axis=1) == 0, axis=1)
        expected = M.boundary(oq[~od])
        if not np.array_equal(M.T.boundary_identity(actual), M.T.boundary_identity(expected)):
            raise AssertionError(f"{role} fixture replacement boundary mismatch")
    return {
        "status": "PASS_ENDPOINT_INVERSION_RESCUED_BY_FITTED_ANCHOR",
        "frozen_782_control_status": old_result["status"],
        "old_keep_endpoint_replacement_determinants": endpoint_det.tolist(),
        "optimized_anchor_f32": result["anchor_f32"].tolist(),
        "anchor_displacement_m": result["fit"]["anchor_displacement_m"],
        "move_cap_m": 0.0005,
        "minimum_new_native_q": result["fit"]["minimum_new_native_q"],
        "native_q_gate_unchanged": M.NATIVE_Q,
        "minimum_cell_volume_gate_m3_unchanged": M.MIN_CELL_VOLUME_M3,
        "final_by_role": final_by_role,
        "exact_paired_delta": result["exact_paired_delta"],
        "local_boundary_chain": "PASS_EXACT_ORIENTED_BOUNDARY_IDENTITY",
        "all_scene_775_697_gates": "NOT_RUN_FIXTURE_ONLY",
        "physical_admission": False,
        "native_steps": 0,
    }


def _write_json(path, obj):
    M._write_json(path, obj)


def main():
    ap = argparse.ArgumentParser(description="783 coupled-anchor state runner with fitted-anchor orientation gate")
    ap.add_argument("--mode", choices=("fixture", "prepare-state", "run"), default="fixture")
    ap.add_argument("--state", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--move-cap-m", type=float, default=0.0005)
    ap.add_argument("--max-operations", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--max-edge-length-m", type=float, default=0.0005)
    ap.add_argument("--max-edges", type=int, default=100000)
    a = ap.parse_args()
    if a.mode == "fixture":
        result = {"operation_override": OVERRIDE, "fixture": orientation_rescue_fixture()}
        if a.output:
            if a.output.exists():
                raise FileExistsError(a.output)
            a.output.mkdir(parents=True)
            report = a.output / "fixture-report.json"
            _write_json(report, result)
            result["fixture_report"] = {"path": str(report), "sha256": sha(report)}
        print(json.dumps(result, indent=2, default=str))
        return
    if a.state is None or a.output is None:
        ap.error("--mode prepare-state/run requires --state and --output")
    if a.mode == "prepare-state":
        out = a.output.resolve()
        out.mkdir(parents=True, exist_ok=False)
        cfgpath, prep, *_ = G.state_config(a.state, out, move_cap_m=a.move_cap_m,
            max_operations=a.max_operations, time_limit_s=a.seconds,
            max_edge_length_m=a.max_edge_length_m, max_edges=a.max_edges)
        prep["operation_prepare_override"] = OVERRIDE
        prep_path = out / "inputs" / "state-preparation.json"
        _write_json(prep_path, prep)
        print(json.dumps({"status": "PASS_STATE_PREPARATION_ONLY",
            "preparation": {"path": str(prep_path), "sha256": sha(prep_path)},
            "config": {"path": str(cfgpath), "sha256": sha(cfgpath)},
            "candidate_run": "NOT_STARTED", "physical_admission": False, "native_steps": 0},
            indent=2, default=str))
        return
    result = G.run_verified_state(a.state, a.output, move_cap_m=a.move_cap_m,
        max_operations=a.max_operations, time_limit_s=a.seconds,
        max_edge_length_m=a.max_edge_length_m, max_edges=a.max_edges)
    wrapper_path = a.output.resolve() / "state-wrapper.json"
    if wrapper_path.is_file():
        wrapper = json.loads(wrapper_path.read_text())
        wrapper["operation_prepare_override"] = OVERRIDE
        wrapper["producer"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
        _write_json(wrapper_path, wrapper)
        result = wrapper
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
