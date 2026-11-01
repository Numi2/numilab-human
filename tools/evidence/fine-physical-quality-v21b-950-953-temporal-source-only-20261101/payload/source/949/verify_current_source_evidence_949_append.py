#!/usr/bin/env python3
"""949 append-aware source-evidence owner for authenticated 944 edge bisection.

This is a separate, fail-closed branch.  It validates the complete serialized
944 chain and the current all185 report while interpreting source point identity
as an exact immutable prefix plus journal-authenticated appended midpoint rows.
Legacy 808/775/805/806 paths are imported and left unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "quality-long-edge-refinement-949-root" else HERE
LEGACY_808 = ROOT / "verify_current_source_evidence_808_validation_v26a.py"
LEGACY_808_SHA256 = "6daa77251509436219b9ad4cb710ccaf4e4cf527a55358ad400c0b71e8881ef0"
REFINEMENT_DIR = ROOT / "quality-long-edge-refinement-944-root"
REFINER = REFINEMENT_DIR / "long_edge_refinement_944.py"
REFINER_SHA256 = "eb15f8c4f44b98c84bec835404c36859f61d048b14cb90a4b1ad41d8163aa2ec"
REFINEMENT_VERIFIER = REFINEMENT_DIR / "verify_long_edge_refinement_944.py"
REFINEMENT_VERIFIER_SHA256 = "4e6e748fc234cad59393158be183853f0be3dda65a2a173feafaab0fd6261901"
BATCH948_DIR = ROOT / "quality-long-edge-refinement-948-root"
BATCH948_DRIVER = BATCH948_DIR / "refine_long_edge_batch_948.py"
BATCH948_CORE = BATCH948_DIR / "long_edge_refinement_948_batch_core.py"
BATCH948_HELPER = BATCH948_DIR / "quality_edge_refinement_948_batch_helper.py"
BATCH948_CORE_SHA256 = "503c882af6989ca9cd35f95b7976debca0c35761a80e2f9b7ab207006e7532b7"
BATCH948_DRIVER_SHA256 = "4e92a4d968e6419fc4414e3f2289b7589d8aa4674d8f9241866c90bca46da548"
BATCH948_HELPER_SHA256 = "f6d93351ea91a7407416124fb20c14aa46522b20e28e2a4386823cb7ac33bbe4"
# Filled only after the independent 948 replay owner is frozen and read back.
BATCH948_REPLAYER = BATCH948_DIR / "verify_long_edge_refinement_948.py"
BATCH948_REPLAYER_SHA256 = "a407d459deadfe20ea54b41eec3ae82d10764137fa70c52ccc4884213d2efaa6"
BATCH_DRIVER = REFINEMENT_DIR / "refine_long_edge_batch_944.py"
BATCH_DRIVER_SHA256 = "d0d89dc09d7687c7e182ff66f566ae5ade642c0cdfc739b73be356fae4ea2b3f"
ALL185_SCHEMA = "h706.combined-current-exact-surface-diff-audit.v1"
ALL185_STATUS = "PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
CONFIG_SCHEMA = "h706.combined-current-exact-surface-diff-config.v1"
ROLE_NAMES = ("D", "I", "O")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def import_pinned(name, path, expected_sha):
    path = Path(path)
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError(f"pinned {name} source missing or changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned {name} source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V808 = import_pinned("verify_current_source_evidence_808_v26a_949", LEGACY_808, LEGACY_808_SHA256)
V944 = import_pinned("verify_long_edge_refinement_944_949", REFINEMENT_VERIFIER,
                     REFINEMENT_VERIFIER_SHA256)


def pin(path, expected, label):
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(label + " missing: " + str(p))
    actual = sha(p)
    if actual != str(expected):
        raise ValueError(label + " SHA mismatch: " + str(p))
    return {"path": str(p), "sha256": actual}


def resolve_asset(rec, state_path):
    p = Path(str(rec["path"]))
    return p.resolve() if p.is_absolute() else (Path(state_path).resolve().parent / p).resolve()


def source_pins(state, state_path):
    out = {}
    surfaces = state.get("source_surfaces")
    if not isinstance(surfaces, dict) or set(surfaces) != set(ROLE_NAMES):
        raise ValueError("source surface inventory is not exactly D/I/O")
    for role in ROLE_NAMES:
        rec = surfaces[role]
        p = resolve_asset(rec, state_path)
        out[role] = pin(p, rec.get("sha256", ""), "current " + role + " source")
    return out


def assert_parent_pin(actual, expected, label="parent"):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError(label + " pin must be an object")
    ap = Path(str(actual.get("path", ""))).resolve()
    ep = Path(str(expected.get("path", ""))).resolve()
    if ap != ep or str(actual.get("sha256", "")) != str(expected.get("sha256", "")):
        raise ValueError(label + " path/SHA mismatch")


def assert_append_prefix(parent_f64, parent_f32, candidate_f64, candidate_f32,
                         expected_new_f32, label="source"):
    """Check immutable full/F32 prefix and exact recorded appended-point bytes."""
    p64, q64 = np.asarray(parent_f64), np.asarray(candidate_f64)
    p32, q32 = np.asarray(parent_f32), np.asarray(candidate_f32)
    expected = np.asarray(expected_new_f32, dtype=np.float32).reshape(-1, 3)
    if p64.ndim != 2 or q64.ndim != 2 or p64.shape[1:] != (3,) or q64.shape[1:] != (3,):
        raise ValueError(label + " full-precision coordinate shape is invalid")
    if p64.dtype != q64.dtype or q64.shape != (len(p64) + len(expected), 3):
        raise ValueError(label + " append count/dtype mismatch")
    if q32.shape != q64.shape or q32.dtype != np.float32 or p32.shape != p64.shape or p32.dtype != np.float32:
        raise ValueError(label + " F32 coordinate shape/dtype mismatch")
    if not np.array_equal(p64.view(np.uint8), q64[:len(p64)].view(np.uint8)):
        raise ValueError(label + " pre-existing full-precision rows changed")
    if not np.array_equal(p32.view(np.uint32), q32[:len(p32)].view(np.uint32)):
        raise ValueError(label + " pre-existing F32 rows changed")
    if not np.array_equal(q32[len(p32):].view(np.uint32), expected.view(np.uint32)):
        raise ValueError(label + " appended Float32 row differs from recorded midpoint")
    if not np.array_equal(q64[len(p64):], q32[len(p32):].astype(q64.dtype)):
        raise ValueError(label + " appended full-precision rows differ from canonical F32")


def assert_outer_source_unchanged(parent_pins, current_pins):
    assert_parent_pin(current_pins.get("O"), parent_pins.get("O"), "outer O source")


def _check_all185_context(candidate_state_path, candidate_state, parent_state_path, parent_state,
                          context_report_path=None):
    """Verify current exact report/config pins without calling 808's point-bijection gate.

    The 944 journal independently proves the append-only point namespace.  This
    branch reuses 808's frozen all185 receipt/owner checks and binds its report to
    the exact final source bytes and original source baseline.
    """
    rec = candidate_state.get("exact_context")
    if context_report_path is not None:
        report_path = Path(context_report_path).resolve()
        report_sha = sha(report_path)
        # An explicit current report is required when the batch state inherited
        # a stale parent pointer. If the state already names this report, the
        # two exact pins must agree.
        if isinstance(rec, dict):
            rec_path, rec_sha = V808.V._verify_pin(rec, label="state exact-context pointer")
            if rec_path == report_path and rec_sha != report_sha:
                raise ValueError("state and explicit exact-context report hashes disagree")
    else:
        if not isinstance(rec, dict):
            raise ValueError("append-aware state has no current all185 context; pass --context-report")
        report_path, report_sha = V808.V._verify_pin(rec, label="949 exact-context report")
    report = V808._load(report_path)
    if report.get("schema") != V808.EXPECTED_CONTEXT_SCHEMA or report.get("status") != V808.EXPECTED_CONTEXT_STATUS:
        raise ValueError("append-aware state lacks exact current all185 PASS")
    if report.get("physical_admission") is not False or int(report.get("native_steps", -1)) != 0:
        raise ValueError("exact all185 report contains physical/runtime admission metadata")
    config_path, config_sha = V808.V._verify_pin(report.get("config"), label="949 exact-context config")
    config = V808._load(config_path)
    if config.get("schema") != V808.EXPECTED_CONFIG_SCHEMA:
        raise ValueError("append-aware exact-context config schema mismatch")
    if context_report_path is None and not V808._pin_equal(rec, {"path": str(report_path), "sha256": report_sha}):
        raise ValueError("state does not pin this exact all185 report")

    current = source_pins(candidate_state, candidate_state_path)
    parent = source_pins(parent_state, parent_state_path)
    cand_cfg = config.get("candidate_surfaces")
    base_cfg = config.get("baseline_candidates")
    if not isinstance(cand_cfg, dict) or set(cand_cfg) != set(ROLE_NAMES):
        raise ValueError("all185 config candidate inventory is not D/I/O")
    if not isinstance(base_cfg, dict) or set(base_cfg) != set(ROLE_NAMES):
        raise ValueError("all185 config baseline inventory is not D/I/O")
    # Exact-context audits compare against the long-lived baseline pinned by the
    # parent's context, which may predate the immediate bisection parent.
    parent_rec = parent_state.get("exact_context")
    if not isinstance(parent_rec, dict):
        raise ValueError("pre-bisection source state lacks its authenticated exact-context baseline")
    parent_report_path, _ = V808.V._verify_pin(parent_rec, label="parent exact-context report")
    parent_report = V808._load(parent_report_path)
    parent_config_path, _ = V808.V._verify_pin(parent_report.get("config"), label="parent exact-context config")
    parent_config = V808._load(parent_config_path)
    parent_base_cfg = parent_config.get("baseline_candidates")
    if not isinstance(parent_base_cfg, dict) or set(parent_base_cfg) != set(ROLE_NAMES):
        raise ValueError("parent exact-context baseline inventory is not D/I/O")
    for role in ROLE_NAMES:
        p, h = V808.V._verify_pin(cand_cfg[role], label="all185 candidate " + role)
        if not V808._pin_equal({"path": str(p), "sha256": h}, current[role]):
            raise ValueError("all185 config candidate differs from append-aware current state: " + role)
        p, h = V808.V._verify_pin(base_cfg[role], label="all185 baseline " + role)
        pp, ph = V808.V._verify_pin(parent_base_cfg[role], label="parent all185 baseline " + role)
        if not V808._pin_equal({"path": str(p), "sha256": h}, {"path": str(pp), "sha256": ph}):
            raise ValueError("all185 baseline differs from authenticated parent context: " + role)
        role_report = report.get("roles", {}).get(role)
        if not isinstance(role_report, dict):
            raise ValueError("all185 report lacks role " + role)
        p, h = V808.V._verify_pin(role_report.get("candidate"), label="all185 reported candidate " + role)
        if not V808._pin_equal({"path": str(p), "sha256": h}, current[role]):
            raise ValueError("all185 report candidate differs from current state: " + role)
        p, h = V808.V._verify_pin(role_report.get("baseline"), label="all185 reported baseline " + role)
        pp, ph = V808.V._verify_pin(parent_base_cfg[role], label="parent all185 baseline " + role)
        if not V808._pin_equal({"path": str(p), "sha256": h}, {"path": str(pp), "sha256": str(ph)}):
            raise ValueError("all185 report baseline differs from authenticated parent context: " + role)
        points = role_report.get("points", {})
        if not isinstance(points, dict):
            raise ValueError("all185 point-domain report missing: " + role)
        cdata = load_npz(current[role]["path"])
        ccount = len(V808._points_full(cdata, role))
        if int(points.get("candidate_point_count", -1)) != ccount:
            raise ValueError("all185 candidate point count does not match current typed source: " + role)

    # Source point identity is handled only by the independently replayed 944
    # append chain. The all185 owner still proves zero changed-v-complete and
    # changed-self intersections, both walls, and the pinned record domain.
    all185 = V808._validate_all185(report)
    # Appending midpoint rows does not move any pre-existing point. Verify that
    # directly against the original parent; 944 independently verifies each step.
    for role in ROLE_NAMES:
        b = load_npz(parent[role]["path"])
        c = load_npz(current[role]["path"])
        bp = V808._points_full(b, role)
        cp = V808._points_full(c, role)
        bf = V808._points_f32(b, role)
        cf = V808._points_f32(c, role)
        if len(cp) < len(bp) or bp.dtype != cp.dtype:
            raise ValueError("append-only source point domain shrank or changed dtype: " + role)
        if not np.array_equal(bp.view(np.uint8), cp[:len(bp)].view(np.uint8)):
            raise ValueError("pre-existing full-precision source coordinates changed: " + role)
        if not np.array_equal(bf.view(np.uint32), cf[:len(bf)].view(np.uint32)):
            raise ValueError("pre-existing F32 source coordinates changed: " + role)
        if np.any(~np.isfinite(cp[len(bp):])) or np.any(~np.isfinite(cf[len(bf):])):
            raise ValueError("appended source coordinates are nonfinite: " + role)

    # 944 bisection is I/D only. O file is the exact pinned original and must not
    # be reserialized, reindexed, or rewritten by this branch.
    if not V808._pin_equal(current["O"], parent["O"]):
        raise ValueError("outer O source changed in append-only I/D refinement")
    moved = False
    audit_owner = V808._validate_audit_owner(report, moved, report_path, report_sha, config_path, config)
    return {
        "status": "PASS_CURRENT_EXACT_ALL185_APPEND_NAMESPACE",
        "report": {"path": str(report_path), "sha256": report_sha},
        "config": {"path": str(config_path), "sha256": config_sha},
        "all185": all185,
        "audit_implementation": audit_owner,
        "baseline_surfaces": parent,
        "candidate_surfaces": current,
    }


def _array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.asarray(arrays[key])
        if a.dtype.hasobject:
            raise ValueError("object arrays are forbidden: " + key)
        h.update(key.encode()); h.update(b"\0")
        h.update(a.dtype.str.encode()); h.update(b"\0")
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode()); h.update(b"\0")
        h.update(np.ascontiguousarray(a).tobytes())
    return "array-sha256:" + h.hexdigest()


def _midpoint_sequence(parent_state_path, candidate_state_path, parent_state, candidate_state, ops):
    """Recompute every appended point from exact current rows before accepting journal lineage."""
    expected_by_asset = {"volumes": {}, "surfaces": {}}
    parent_vol = {}; parent_surf = {}
    cur_vol = {}; cur_surf = {};cur_maps={}
    role_maps={
        "Skin_current":{"O":"skin_current_target_rebind_O_point_to_volume_rows",
                        "I":"skin_current_target_rebind_I_point_to_volume_rows"},
        "Fat_current":{"I":"fat_current_target_rebind_I_point_to_volume_rows",
                       "D":"fat_current_target_rebind_D_point_to_volume_rows"},
    }
    for role in ("Skin_current", "Fat_current"):
        asset = resolve_asset(parent_state[role], parent_state_path)
        parent_vol[role] = load_npz(asset)
        cur_vol[role] = np.asarray(V808._points_f32(parent_vol[role], role), np.float32).copy()
        cur_maps[role]={sname:np.asarray(parent_vol[role][key],np.int64).copy()
                        for sname,key in role_maps[role].items()}
        expected_by_asset["volumes"][role] = []
    for sname in ("O", "I", "D"):
        asset = resolve_asset(parent_state["source_surfaces"][sname], parent_state_path)
        parent_surf[sname] = load_npz(asset)
        cur_surf[sname] = np.asarray(V808._points_f32(parent_surf[sname], sname), np.float32).copy()
        expected_by_asset["surfaces"][sname] = []
    for idx, op in enumerate(ops):
        if "O" in op.get("surface_edge_rows", {}):
            raise ValueError("944 journal is not permitted to edit O")
        f_by_role = op.get("fraction_by_role", {})
        f_by_surface = op.get("fraction_by_surface", {})
        common_bits = None
        common_point = None
        new_volume_rows={}
        for role, edge0 in op.get("edge_volume_rows_by_role", {}).items():
            if role not in cur_vol:
                raise ValueError("journal names unknown volume role: " + str(role))
            edge = tuple(map(int, edge0)); f = float(f_by_role[role]); p = cur_vol[role]
            if len(edge) != 2 or min(edge) < 0 or max(edge) >= len(p) or not (0.0 < f < 1.0):
                raise ValueError("journal volume edge/fraction out of range")
            x = ((1.0-f)*p[edge[0]].astype(np.float64)+f*p[edge[1]].astype(np.float64)).astype(np.float32)
            expected_by_asset["volumes"][role].append(x.copy())
            new_volume_rows[role]=len(p)
            cur_vol[role] = np.vstack((p, x))
            if common_point is None:
                common_point = x.copy()
            elif not np.array_equal(common_point.view(np.uint32), x.view(np.uint32)):
                raise ValueError("journal operation midpoint differs across affected volume roles")
            if "I" in op.get("surface_edge_rows", {}):
                if common_bits is not None and not np.array_equal(common_bits, x.view(np.uint32)):
                    raise ValueError("paired I midpoint differs across Skin/Fat Float32 volume rows")
                common_bits = x.view(np.uint32).copy()
        for sname, edge0 in op.get("surface_edge_rows", {}).items():
            if sname not in ("I", "D"):
                raise ValueError("journal names non-I/D source surface")
            edge = tuple(map(int, edge0)); f = float(f_by_surface[sname]); p = cur_surf[sname]
            if len(edge) != 2 or min(edge) < 0 or max(edge) >= len(p) or not (0.0 < f < 1.0):
                raise ValueError("journal source edge/fraction out of range")
            connected={"I":{"Skin_current","Fat_current"},"D":{"Fat_current"}}.get(sname,set())
            if set(new_volume_rows)!=connected:
                raise ValueError("source edge lacks the complete expected volume-role set: "+sname)
            for role in sorted(connected):
                vmap=cur_maps[role][sname]
                mapped=np.asarray(vmap[np.asarray(edge,np.int64)],np.int64)
                vedge=np.asarray(op.get("edge_volume_rows_by_role",{}).get(role,()),np.int64)
                if vedge.shape!=(2,) or set(map(int,mapped))!=set(map(int,vedge)):
                    raise ValueError("journal source edge does not map to declared volume edge: "+sname+"/"+role)
                vp=cur_vol[role]
                if not np.array_equal(vp[mapped].view(np.uint32),p[np.asarray(edge)].view(np.uint32)):
                    raise ValueError("source/volume edge endpoint coordinate bits disagree: "+sname+"/"+role)
            x = ((1.0-f)*p[edge[0]].astype(np.float64)+f*p[edge[1]].astype(np.float64)).astype(np.float32)
            if "I" == sname and common_bits is not None and not np.array_equal(common_bits, x.view(np.uint32)):
                raise ValueError("I source midpoint differs bitwise from paired volume midpoint")
            expected_by_asset["surfaces"][sname].append(x.copy())
            cur_surf[sname] = np.vstack((p, x))
            if common_point is not None and not np.array_equal(common_point.view(np.uint32), x.view(np.uint32)):
                raise ValueError("journal source midpoint differs from affected volume role")
            if common_point is None:
                common_point = x.copy()
            for role in sorted(connected):
                cur_maps[role][sname]=np.r_[cur_maps[role][sname],new_volume_rows[role]].astype(np.int64)
        if common_point is None:
            raise ValueError("journal operation has no affected source/volume rows")
        if not np.array_equal(np.asarray(op.get("new_point_world_m_f32"), np.float32).view(np.uint32),
                              common_point.view(np.uint32)):
            raise ValueError("journal operation midpoint differs from exact recomputed F32 point")
        bits = [hex(int(v)) for v in common_point.view(np.uint32)]
        if op.get("new_point_bits") != bits:
            raise ValueError("journal midpoint bit string does not match exact F32 point")
    for role in ("Skin_current", "Fat_current"):
        candidate = load_npz(resolve_asset(candidate_state[role], candidate_state_path))
        full0 = np.asarray(parent_vol[role]["points_world_m"])
        f320 = V808._points_f32(parent_vol[role], role)
        full1 = np.asarray(candidate["points_world_m"])
        f321 = V808._points_f32(candidate, role)
        expected = np.asarray(expected_by_asset["volumes"][role], np.float32).reshape(-1, 3)
        assert_append_prefix(full0, f320, full1, f321, expected, role)
    for sname in ("O", "I", "D"):
        candidate_pin = candidate_state["source_surfaces"][sname]
        candidate = load_npz(resolve_asset(candidate_pin, candidate_state_path))
        full0 = np.asarray(parent_surf[sname]["points_world_m"])
        f320 = V808._points_f32(parent_surf[sname], sname)
        full1 = np.asarray(candidate["points_world_m"])
        f321 = V808._points_f32(candidate, sname)
        expected = np.asarray(expected_by_asset["surfaces"][sname], np.float32).reshape(-1, 3)
        assert_append_prefix(full0, f320, full1, f321, expected, sname)
    return {k: {n: len(v) for n, v in d.items()} for k, d in expected_by_asset.items()}


def _remap_csr(offsets, values, parent_rows):
    offsets = np.asarray(offsets, np.int64); values = np.asarray(values)
    out_offsets = np.zeros(len(parent_rows) + 1, np.int64); pieces=[]; total=0
    for i, row in enumerate(np.asarray(parent_rows, np.int64)):
        lo, hi = map(int, offsets[int(row):int(row)+2]); piece=values[lo:hi]
        pieces.append(piece); total += len(piece); out_offsets[i+1]=total
    out_values = np.concatenate(pieces) if total else np.asarray([], dtype=values.dtype)
    return out_offsets, out_values


def _derived_lane(values, a, b, fraction):
    arr = np.asarray(values); va, vb = arr[int(a)], arr[int(b)]
    if arr.dtype.kind == "f":
        return np.asarray((1.0-float(fraction))*va.astype(np.float64)+float(fraction)*vb.astype(np.float64), dtype=arr.dtype)
    if arr.dtype.kind in "iu":
        return np.asarray(va if np.array_equal(va,vb) else np.full_like(va,-1), dtype=arr.dtype)
    if arr.dtype.kind == "b":
        return np.asarray(va if np.array_equal(va,vb) else np.zeros_like(va), dtype=arr.dtype)
    if arr.dtype.kind in "US":
        return np.asarray(va if np.array_equal(va,vb) else np.full_like(va,""), dtype=arr.dtype)
    raise ValueError("unsupported typed point lane: " + str(arr.dtype))


_OUTPUT_TO_INPUT_POINT_LANES = {
    "output_point_to_input_point_row", "output_point_to_input_point_rows",
    "output_node_to_input_node", "output_node_to_input_node_rows",
    "quality_milp_output_node_to_input_node",
    "source_interior_vertex_quotient", "source_input_vertex_quotient",
    "source_I_vertex_quotient",
}
_INPUT_TO_OUTPUT_POINT_LANES = {
    "input_point_to_output_point_row", "input_point_to_output_point_rows",
    "input_node_to_output_node", "input_node_to_output_node_rows",
    "quality_milp_input_node_to_output_node", "source_input_node_to_output_node",
    "endpoint_quotient_input_node_map", "endpoint_quotient_batch_input_node_to_output_rows",
}


def _verify_volume_point_metadata(parent_npz, final_npz, role, ops):
    """Verify typed point-domain maps without treating row maps as interpolated data."""
    base=parent_npz; final=final_npz
    n0=len(V808._points_f32(base,role)); n1=len(V808._points_f32(final,role))
    role_ops=[op for op in ops if role in op.get("edge_volume_rows_by_role",{})]
    nnew=len(role_ops)
    if n1 != n0+nnew:
        raise ValueError("volume point-domain append count mismatch: "+role)
    for key in _OUTPUT_TO_INPUT_POINT_LANES:
        if key in base:
            old=np.asarray(base[key])
            if old.ndim==0 or old.shape[0] != n0:
                raise ValueError("output-to-input point map has unexpected parent domain: "+role+"/"+key)
            expected=np.concatenate((old,np.full((nnew,)+old.shape[1:],-1,dtype=old.dtype)),axis=0)
            if key not in final or not np.array_equal(final[key],expected):
                raise ValueError("output-to-input point map append mismatch: "+role+"/"+key)
    for key in _INPUT_TO_OUTPUT_POINT_LANES:
        if key in base:
            old=np.asarray(base[key])
            if key not in final or not np.array_equal(final[key],old):
                raise ValueError("input-to-output map input-domain prefix changed: "+role+"/"+key)
    # Exact directly authenticated construction maps.
    direct_expected={
        "output_point_to_input_point_row": np.r_[np.arange(n0,dtype=np.int64),np.full(nnew,-1,np.int64)],
        "output_node_to_input_node": np.r_[np.arange(n0,dtype=np.int64),np.full(nnew,-1,np.int64)],
        "input_node_to_output_node": np.arange(n0,dtype=np.int64),
        "output_node_to_input_node_rows": np.r_[np.arange(n0,dtype=np.int64),np.full(nnew,-1,np.int64)],
        "output_point_to_input_point_rows": np.r_[np.arange(n0,dtype=np.int64),np.full(nnew,-1,np.int64)],
    }
    for key,expected in direct_expected.items():
        if key in base and (key not in final or not np.array_equal(final[key],expected)):
            raise ValueError("direct point identity lane mismatch: "+role+"/"+key)
    # Existing quotient/ancestry records are inherited exactly; the new point
    # has no one-to-one row in those historical domains.
    for key in ("source_interior_vertex_quotient","source_input_vertex_quotient","source_I_vertex_quotient"):
        if key in base:
            old=np.asarray(base[key]); expected=np.concatenate((old,np.full((nnew,)+old.shape[1:],-1,dtype=old.dtype)),axis=0)
            if key not in final or not np.array_equal(final[key],expected):
                raise ValueError("historical point quotient prefix/suffix mismatch: "+role+"/"+key)
    # Check remaining semantic point lanes with the same typed interpolation
    # rule used by the frozen 948 core, operation by operation.
    skip=set(_OUTPUT_TO_INPUT_POINT_LANES)|set(_INPUT_TO_OUTPUT_POINT_LANES)|{
        "points_world_m","points_world_m_f32","output_point_to_input_point_row",
        "output_point_to_input_point_rows","input_node_to_output_node","output_node_to_input_node",
        "input_node_to_output_node_rows","output_node_to_input_node_rows",
        "source_interior_vertex_quotient","source_input_vertex_quotient","source_I_vertex_quotient",
        "input_point_to_output_point_row","input_point_to_output_point_rows",
    }
    for key,value in base.items():
        arr=np.asarray(value)
        if key in skip or not arr.ndim or arr.shape[0] != n0 or not ("point" in key.lower() or "vertex" in key.lower()):
            continue
        expected=arr.copy()
        for op in role_ops:
            edge=tuple(map(int,op["edge_volume_rows_by_role"][role])); frac=float(op["fraction_by_role"][role])
            expected=np.concatenate((expected,np.asarray(_derived_lane(expected,*edge,frac)).reshape((1,)+expected.shape[1:])),axis=0)
        if key not in final or not np.array_equal(final[key],expected):
            raise ValueError("typed volume point lane mismatch: "+role+"/"+key)
    return {"role":role,"parent_points":n0,"appended_points":nnew,
            "checked_output_to_input_maps":sorted(k for k in _OUTPUT_TO_INPUT_POINT_LANES if k in base),
            "checked_input_to_output_maps":sorted(k for k in _INPUT_TO_OUTPUT_POINT_LANES if k in base)}


def _verify_volume_source_maps(parent_npz, final_npz, role, ops, candidate_state,
                               candidate_state_path):
    maps={
        "Skin_current":{
            "O":"skin_current_target_rebind_O_point_to_volume_rows",
            "I":"skin_current_target_rebind_I_point_to_volume_rows",
        },
        "Fat_current":{
            "I":"fat_current_target_rebind_I_point_to_volume_rows",
            "D":"fat_current_target_rebind_D_point_to_volume_rows",
        },
    }
    oldn=len(V808._points_f32(parent_npz,role));opn=0;expected={}
    for sname,key in maps[role].items():
        if key not in parent_npz:
            raise ValueError("parent volume lacks typed source point map: "+role+"/"+key)
        expected[sname]=np.asarray(parent_npz[key],np.int64).copy()
    for op in ops:
        touched_surfaces=set(op.get("surface_edge_rows",{}))
        for sname in touched_surfaces:
            if sname not in expected or role not in op.get("volume_parent_cell_rows",{}):
                continue
            expected[sname]=np.r_[expected[sname],oldn+opn].astype(np.int64)
        if role in op.get("volume_parent_cell_rows",{}): opn+=1
    for sname,key in maps[role].items():
        if key not in final_npz or not np.array_equal(final_npz[key],expected[sname]):
            raise ValueError("typed source-to-volume map did not append exact new row: "+role+"/"+sname)
        src_rec=candidate_state["source_surfaces"][sname]
        srcpath=resolve_asset(src_rec,candidate_state_path)
        pathkey="typed_"+sname+"_candidate_path"; hashkey="typed_"+sname+"_candidate_sha256"
        if pathkey not in final_npz or hashkey not in final_npz:
            raise ValueError("volume typed source pin fields are missing: "+role+"/"+sname)
        if str(np.asarray(final_npz[pathkey]).reshape(-1)[0])!=str(srcpath):
            raise ValueError("volume typed source path pin is stale: "+role+"/"+sname)
        if str(np.asarray(final_npz[hashkey]).reshape(-1)[0])!=str(src_rec["sha256"]):
            raise ValueError("volume typed source SHA pin is stale: "+role+"/"+sname)
    return {"role":role,"maps":sorted(maps[role])}


def _verify_surface_metadata_lineage(parent_npz, final_npz, surface_name, ops, parent_pin,
                                     journal_pin):
    """Replay source face CSR/lanes and appended point metadata from the base arrays."""
    base = parent_npz; final = final_npz
    base_faces = np.asarray(base["faces"], np.int64)
    face_root = np.arange(len(base_faces), dtype=np.int64)
    current_faces = base_faces.copy()
    base_points = V808._points_f32(base, surface_name)
    current_points = base_points.copy()
    point_lanes = {}
    for key, value in base.items():
        a = np.asarray(value)
        if a.ndim and a.shape[0] == len(base_points) and ("point" in key.lower() or "vertex" in key.lower()) and key not in ("points_world_m", "points_world_m_f32", "quality_star_point_output_to_input_row", "source_input_vertex_quotient"):
            point_lanes[key] = a.copy()
    surface_ops=[]
    for op in ops:
        if surface_name not in op.get("surface_edge_rows", {}): continue
        edge = tuple(map(int, op["surface_edge_rows"][surface_name])); frac=float(op["fraction_by_surface"][surface_name])
        hits=np.flatnonzero(np.isin(current_faces, edge).sum(axis=1)==2)
        declared=np.asarray(op["surface_parent_face_rows"][surface_name],np.int64)
        if not len(hits) or not np.array_equal(hits, declared):
            raise ValueError("journal source parent-face rows are not complete/current: " + surface_name)
        root_parent_faces=np.unique(face_root[hits])
        parents=[]; new_faces=[]
        a,b=edge
        for row, face in enumerate(current_faces):
            f=list(map(int,face))
            if a in f and b in f:
                w=next(v for v in f if v not in edge)
                for j in range(3):
                    u,v=f[j],f[(j+1)%3]
                    if (u,v)==(a,b) or (u,v)==(b,a):
                        new_faces.extend(((u,len(current_points),w),(len(current_points),v,w)))
                        parents.extend((row,row));break
            else:
                new_faces.append(tuple(f));parents.append(row)
        parents=np.asarray(parents,np.int64);face_root=face_root[parents]
        x=((1-frac)*current_points[a].astype(np.float64)+frac*current_points[b].astype(np.float64)).astype(np.float32)
        for key, lane in list(point_lanes.items()):
            point_lanes[key]=np.concatenate((lane, np.asarray(_derived_lane(lane,a,b,frac)).reshape((1,)+lane.shape[1:])),axis=0)
        current_points=np.vstack((current_points,x));current_faces=np.asarray(new_faces,np.int64)
        surface_ops.append((edge,frac,root_parent_faces))
    # Exact face identity and all existing ancestry CSR families.
    for key, expected in (("source_input_face_rows",face_root),
                          ("edge_bisection_output_face_to_immediate_parent_row",face_root)):
        if key not in final or not np.array_equal(final[key],expected):
            raise ValueError("composed source face identity mismatch: " + surface_name + "/" + key)
    face_known={"faces","source_input_face_rows","edge_bisection_output_face_to_immediate_parent_row"}
    csr_companions=set()
    for prefix in ("quality_star_patch_union", "face_ancestry"):
        offkey=prefix+"_offsets"
        parent_keys=[k for k in base if k.startswith(prefix+"_")]
        if offkey not in base:
            if any(k.startswith(prefix+"_") for k in final):
                raise ValueError("unparented source face CSR family: " + surface_name + "/"+prefix)
            continue
        expected_offsets=None
        for key in parent_keys:
            if key==offkey: continue
            no,nv=_remap_csr(base[offkey],base[key],face_root)
            if expected_offsets is None: expected_offsets=no
            elif not np.array_equal(expected_offsets,no): raise ValueError("parent face CSR companions disagree")
            if key not in final or not np.array_equal(final[key],nv):
                raise ValueError("composed source face CSR values mismatch: "+surface_name+"/"+key)
            csr_companions.add(key);face_known.add(key)
        if expected_offsets is None: expected_offsets=np.zeros(len(face_root)+1,np.int64)
        if offkey not in final or not np.array_equal(final[offkey],expected_offsets):
            raise ValueError("composed source face CSR offsets mismatch: "+surface_name+"/"+offkey)
        face_known.add(offkey)
    if "quality_star_face_output_to_input_row" in base:
        expected=np.asarray(base["quality_star_face_output_to_input_row"],np.int64)[face_root]
        if "quality_star_face_output_to_input_row" not in final or not np.array_equal(final["quality_star_face_output_to_input_row"],expected):
            raise ValueError("absolute source face identity mismatch: "+surface_name)
        face_known.add("quality_star_face_output_to_input_row")
    if "quality_star_point_output_to_input_row" in base:
        expected=np.r_[np.asarray(base["quality_star_point_output_to_input_row"],np.int64),np.full(len(surface_ops),-1,np.int64)]
        if "quality_star_point_output_to_input_row" not in final or not np.array_equal(final["quality_star_point_output_to_input_row"],expected):
            raise ValueError("appended source point legacy identity mismatch: "+surface_name)
        face_known.add("quality_star_point_output_to_input_row")
    if "source_input_vertex_quotient" in base:
        expected=np.asarray(base["source_input_vertex_quotient"])
        if "source_input_vertex_quotient" not in final or not np.array_equal(final["source_input_vertex_quotient"],expected):
            raise ValueError("source input-vertex quotient parent-domain bytes changed: "+surface_name)
        face_known.add("source_input_vertex_quotient")
    # This historical map is output-row -> one current-parent row. A derived
    # midpoint has no one-to-one parent row, so preserve the old prefix and use
    # the declared -1 sentinel for each appended point.
    if "endpoint_quotient_output_to_current_parent_rows" in base:
        old=np.asarray(base["endpoint_quotient_output_to_current_parent_rows"])
        expected=np.concatenate((old, np.full(len(surface_ops),-1,dtype=old.dtype)))
        key="endpoint_quotient_output_to_current_parent_rows"
        if key not in final or not np.array_equal(final[key],expected):
            raise ValueError("endpoint-quotient point-parent lane mismatch: "+surface_name)
        face_known.add(key)
    for key, expected in point_lanes.items():
        if key not in final or not np.array_equal(final[key],expected):
            raise ValueError("derived source point lane mismatch: "+surface_name+"/"+key)
        face_known.add(key)
    # Every face-aligned typed lane is selected by the exact face parent map.
    for key, value in base.items():
        a=np.asarray(value)
        if key in face_known or key.endswith("_offsets") or key.startswith("edge_bisection_") or key.startswith("typed_parent_candidate_"):
            continue
        if key.startswith("source_") and key.endswith("_operation_rows"):
            if key not in final or len(final[key]) != 0:
                raise ValueError("source operation history was not cleared after derived batch: "+surface_name+"/"+key)
            face_known.add(key);continue
        if a.ndim and a.shape[0]==len(base_faces) and "point" not in key.lower() and "vertex" not in key.lower():
            if key not in final or not np.array_equal(final[key],a[face_root]):
                raise ValueError("typed face lane ancestry mismatch: "+surface_name+"/"+key)
            face_known.add(key)
        elif key not in point_lanes and key not in ("points_world_m","points_world_m_f32"):
            if key in final and not np.array_equal(final[key],a):
                raise ValueError("unrelated source metadata changed: "+surface_name+"/"+key)
    # Verify appended-point CSR references the original parent faces exactly.
    expected_edges=np.asarray([x[0] for x in surface_ops],np.int64).reshape(-1,2)
    expected_fracs=np.asarray([x[1] for x in surface_ops],np.float64)
    off=[0];vals=[]
    for _,_,parents in surface_ops: vals.extend(map(int,parents));off.append(len(vals))
    exact={"edge_bisection_new_point_parent_edge_rows":expected_edges,
           "edge_bisection_new_point_parent_edge_fraction":expected_fracs,
           "edge_bisection_new_point_parent_edge_support_residual_m":np.zeros(len(surface_ops),np.float64),
           "edge_bisection_new_point_parent_surface_name":np.asarray([surface_name]*len(surface_ops),dtype=np.asarray(final.get("edge_bisection_new_point_parent_surface_name",[surface_name])).dtype),
           "edge_bisection_new_point_parent_immediate_face_offsets":np.asarray(off,np.int64),
           "edge_bisection_new_point_parent_immediate_face_rows":np.asarray(vals,np.int64)}
    for key,expected in exact.items():
        if key not in final or not np.array_equal(final[key],expected):
            raise ValueError("appended source point provenance mismatch: "+surface_name+"/"+key)
    pathkey="typed_parent_candidate_path";hashkey="typed_parent_candidate_sha256"
    if pathkey not in final or hashkey not in final:
        raise ValueError("typed parent pin fields are missing: "+surface_name)
    if str(np.asarray(final[pathkey]).reshape(-1)[0]) != str(parent_pin["path"]):
        raise ValueError("typed source parent path mismatch: "+surface_name)
    if str(np.asarray(final[hashkey]).reshape(-1)[0]) != str(parent_pin["sha256"]):
        raise ValueError("typed source parent hash mismatch: "+surface_name)
    journalkey="edge_bisection_batch_journal_path"
    if journalkey not in final or str(np.asarray(final[journalkey]).reshape(-1)[0]) != str(journal_pin["path"]):
        raise ValueError("source batch journal path mismatch: "+surface_name)
    return {"surface":surface_name,"operations":len(surface_ops),"faces":len(current_faces),
            "face_CSR_families":[p for p in ("quality_star_patch_union","face_ancestry") if p+"_offsets" in base]}


def verify_compact_batch_948(parent_state_path, batch_report_path, candidate_state_path,
                             context_report_path=None):
    """Validate final 948 arrays using its separately pinned in-memory replay owner."""
    if BATCH948_REPLAYER_SHA256 == "PENDING_FREEZE":
        raise ValueError("948 independent replay owner pin has not been frozen")
    parent_state_path = Path(parent_state_path).resolve()
    batch_report_path = Path(batch_report_path).resolve()
    candidate_state_path = Path(candidate_state_path).resolve()
    parent_state = load_json(parent_state_path); candidate_state = load_json(candidate_state_path)
    report = load_json(batch_report_path)
    if parent_state.get("physical_admission") is not False or int(parent_state.get("native_steps",-1))!=0:
        raise ValueError("949 parent is already marked physical/native admitted")
    if parent_state.get("mass_transport") not in (None,"NOT_RUN"):
        raise ValueError("949 parent has an unsupported mass transport claim")
    if not str(parent_state.get("source_audit_status","")).startswith("PASS_"):
        raise ValueError("949 parent lacks a passing source-only audit status")
    if report.get("schema") != "numi.h706.long-edge-refinement-batch-948.v1":
        raise ValueError("unexpected compact 948 batch report schema")
    if report.get("status") != "PASS_ZERO_NATIVE_FAILURES":
        raise ValueError("948 batch did not meet its final zero-native-failure batch gate")
    if report.get("physical_admission") is not False or int(report.get("native_steps", -1)) != 0 or report.get("mass_transport") != "NOT_RUN":
        raise ValueError("948 batch contains unsupported physical/native/mass claims")
    if report.get("promotion_ready") is not False:
        raise ValueError("948 producer may not mark its output promotion-ready")
    parent_pin = pin(parent_state_path, sha(parent_state_path), "949 parent state")
    if report.get("input_state") != parent_pin:
        raise ValueError("948 report does not bind the exact parent state")
    journal_rec = report.get("journal")
    journal_pin = pin(journal_rec.get("path"), journal_rec.get("sha256"), "948 batch journal")
    journal = load_json(journal_pin["path"])
    if journal.get("schema") != report["schema"] or journal.get("input_state") != parent_pin:
        raise ValueError("948 journal schema or parent pin mismatch")
    expected_input_pins = {}
    for role in ("Skin_current", "Fat_current"):
        expected_input_pins[role] = {"path": str(resolve_asset(parent_state[role], parent_state_path)),
                                     "sha256": str(parent_state[role]["sha256"])}
    for sname in ("O", "I", "D"):
        expected_input_pins[sname] = {"path": str(resolve_asset(parent_state["source_surfaces"][sname], parent_state_path)),
                                      "sha256": str(parent_state["source_surfaces"][sname]["sha256"])}
    journal_input_pins = journal.get("input_pins")
    if not isinstance(journal_input_pins, dict) or set(journal_input_pins) != set(expected_input_pins):
        raise ValueError("948 journal input asset inventory mismatch")
    for name, expected_pin in expected_input_pins.items():
        rec = journal_input_pins[name]
        if not V808._pin_equal(rec, expected_pin) or sha(expected_pin["path"]) != expected_pin["sha256"]:
            raise ValueError("948 journal input asset pin mismatch: " + name)
    ops = journal.get("operations")
    if not isinstance(ops, list) or not ops or len(ops) != int(report.get("operation_count", -1)):
        raise ValueError("948 journal operation count mismatch")
    if report.get("operations") != ops:
        raise ValueError("948 report operations differ from the pinned journal")
    if journal.get("operation_count") != len(ops) or journal.get("physical_admission") is not False or journal.get("mass_transport") != "NOT_RUN":
        raise ValueError("948 journal operation count/admission metadata mismatch")
    if (BATCH948_CORE_SHA256 == "PENDING_FREEZE" or BATCH948_DRIVER_SHA256 == "PENDING_FREEZE"
            or BATCH948_HELPER_SHA256 == "PENDING_FREEZE"):
        raise ValueError("948 producer/core owner hashes are not frozen")
    core_pin = pin(BATCH948_CORE, BATCH948_CORE_SHA256, "948 core")
    producer_pin = pin(BATCH948_DRIVER, BATCH948_DRIVER_SHA256, "948 producer")
    helper_pin = pin(BATCH948_HELPER, BATCH948_HELPER_SHA256, "948 helper")
    replay_pin = pin(BATCH948_REPLAYER, BATCH948_REPLAYER_SHA256, "948 independent replay owner")
    if report.get("core", {}).get("sha256") != core_pin["sha256"] or report.get("producer", {}).get("sha256") != producer_pin["sha256"]:
        raise ValueError("948 batch report source hashes differ from frozen owner pins")
    if report.get("core", {}).get("path") != core_pin["path"] or report.get("producer", {}).get("path") != producer_pin["path"]:
        raise ValueError("948 producer/core path pin mismatch")
    final_pin = pin(report.get("final_state", {}).get("path"), report.get("final_state", {}).get("sha256"), "948 final state")
    if final_pin["sha256"] != sha(candidate_state_path):
        raise ValueError("948 final state differs from requested candidate")
    if candidate_state.get("edge_bisection_batch_journal", {}).get("sha256") != journal_pin["sha256"]:
        raise ValueError("candidate state does not reference the exact 948 journal")
    state_journal = candidate_state.get("edge_bisection_batch_journal", {})
    if Path(state_journal.get("path", "")).resolve() != Path(journal_pin["path"]).resolve():
        raise ValueError("candidate state journal path differs from report")
    marker = candidate_state.get("edge_bisection", {})
    if marker.get("schema") != "numi.h706.coupled-source-volume-edge-bisection.batch.v1" or marker.get("journal", {}).get("sha256") != journal_pin["sha256"]:
        raise ValueError("candidate state lacks the batch bisection marker")
    if len(report.get("steps", [])) != len(ops) or any(x.get("status") != "PASS_VIRTUAL_STEP" for x in report.get("steps", [])):
        raise ValueError("948 virtual step inventory is incomplete")
    # Independently close every virtual array-digest transition from the pinned
    # original inputs to the actual final NPZ bytes; no intermediate archives are needed.
    current_digests = {}
    for role in ("Skin_current", "Fat_current"):
        arr = load_npz(resolve_asset(parent_state[role], parent_state_path))
        current_digests[role] = _array_digest(arr)
    for sname in ("O", "I", "D"):
        arr = load_npz(resolve_asset(parent_state["source_surfaces"][sname], parent_state_path))
        current_digests[sname] = _array_digest(arr)
    steps = report["steps"]
    for i, (op, step) in enumerate(zip(ops, steps)):
        if int(op.get("operation_index", -1)) != i:
            raise ValueError("948 operation index sequence is not contiguous")
        in_digests = op.get("input_array_digests")
        out_digests = op.get("output_array_digests")
        if not isinstance(in_digests, dict) or in_digests != current_digests:
            raise ValueError("948 virtual input array digests do not match preceding state at operation " + str(i))
        if not isinstance(out_digests, dict) or not out_digests:
            raise ValueError("948 operation lacks virtual output array digests")
        rawop = {k:v for k,v in op.items() if k not in ("operation_index", "input_array_digests", "output_array_digests")}
        if step.get("operation_index") != i or step.get("operation") != rawop:
            raise ValueError("948 step journal operation differs from virtual step")
        if step.get("virtual_outputs") != out_digests:
            raise ValueError("948 step output digests differ from journal")
        for name, ref in out_digests.items():
            if name not in current_digests or not isinstance(ref, dict) or not str(ref.get("array_digest", "")).startswith("array-sha256:"):
                raise ValueError("948 operation output digest record malformed: " + str(name))
            current_digests[name] = ref["array_digest"]
    # Bind the final NPZ content digests to actual serialized arrays.
    final_assets = report.get("final_assets", {})
    if set(final_assets) != set(report.get("changed_roles", [])) | set(report.get("changed_surfaces", [])):
        raise ValueError("948 final asset inventory does not close")
    for name, rec in final_assets.items():
        p = Path(rec["path"]).resolve()
        if sha(p) != rec.get("sha256"):
            raise ValueError("948 final asset archive SHA mismatch: " + name)
        arrays = load_npz(p)
        if _array_digest(arrays) != rec.get("array_digest"):
            raise ValueError("948 final asset array digest mismatch: " + name)
        if name in ("Skin_current", "Fat_current"):
            strec = candidate_state[name]
        else:
            strec = candidate_state["source_surfaces"][name]
        if not V808._pin_equal({"path": str(p), "sha256": rec["sha256"]},
                               {"path": str(resolve_asset(strec, candidate_state_path)), "sha256": strec["sha256"]}):
            raise ValueError("948 final asset does not match candidate state: " + name)
        if current_digests.get(name) != rec.get("array_digest"):
            raise ValueError("948 final array digest does not close to virtual journal: " + name)
    changed_roles = set(report["changed_roles"]); changed_surfs = set(report["changed_surfaces"])
    if not changed_surfs.issubset({"I", "D"}) or "O" in changed_surfs:
        raise ValueError("948 batch changed an unauthorized source surface")
    expected_roles = set()
    expected_surfs = set()
    for op in ops:
        if "O" in op.get("surface_edge_rows", {}):
            raise ValueError("948 journal edits outer O")
        touched = set(op.get("surface_edge_rows", {}))
        if not touched or not touched.issubset({"I", "D"}):
            raise ValueError("948 operation source surface role mismatch")
        expected_surfs |= touched
        expected_roles |= ({"Skin_current", "Fat_current"} if "I" in touched else {"Fat_current"})
    if changed_surfs != expected_surfs or changed_roles != expected_roles:
        raise ValueError("948 changed role/surface inventory differs from journal")
    if set(report.get("final_metrics", {})) != changed_roles or any(
            int(rec.get("bad", -1)) != 0 or int(rec.get("cells", -1)) < 1 or int(rec.get("nodes", -1)) < 4
            for rec in report["final_metrics"].values()):
        raise ValueError("948 final native quality metrics do not show the complete zero-failure gate")
    # Fixed O/back/wall data stays byte-identical in every changed volume asset.
    for role in changed_roles:
        outvol = load_npz(final_assets[role]["path"])
        inpath = resolve_asset(parent_state[role], parent_state_path)
        invol = load_npz(inpath)
        if role == "Skin_current":
            key = "skin_current_target_rebind_O_point_to_volume_rows"
            if key in invol and not np.array_equal(outvol[key], invol[key]):
                raise ValueError("948 batch changed the protected O-to-Skin point map")
        for key, value in invol.items():
            if key.endswith("_wall_faces_volume_rows") or key.endswith("_wall_faces_collar_point_rows"):
                if key not in outvol or not np.array_equal(outvol[key], value):
                    raise ValueError("948 batch changed protected wall data: " + role + "/" + key)
    # The independent virtual replay owner reconstructs each operation, array
    # digest, exact split, material, cell ancestry, face ancestry and point CSR.
    replay_module = import_pinned("verify_long_edge_refinement_batch_948_949", BATCH948_REPLAYER,
                                  BATCH948_REPLAYER_SHA256)
    replay = replay_module.replay_batch_948(parent_state_path, journal_pin["path"], candidate_state_path)
    if replay.get("status") != "PASS_INDEPENDENT_BATCH_REPLAY":
        raise ValueError("948 independent compact replay did not pass")
    volume_point_lineage={}
    volume_source_maps={}
    for role in sorted(changed_roles):
        parent_vol=load_npz(resolve_asset(parent_state[role],parent_state_path))
        final_vol=load_npz(final_assets[role]["path"])
        volume_point_lineage[role]=_verify_volume_point_metadata(parent_vol,final_vol,role,ops)
        volume_source_maps[role]=_verify_volume_source_maps(
            parent_vol,final_vol,role,ops,candidate_state,candidate_state_path)
    surface_lineage = {}
    for sname in ("O", "I", "D"):
        parent_pin_s = source_pins(parent_state, parent_state_path)[sname]
        base_s = load_npz(parent_pin_s["path"])
        final_s_pin = pin(resolve_asset(candidate_state["source_surfaces"][sname], candidate_state_path),
                          candidate_state["source_surfaces"][sname].get("sha256", ""),
                          "final source " + sname)
        final_s = load_npz(final_s_pin["path"])
        if sname == "O" or not any(sname in op.get("surface_edge_rows", {}) for op in ops):
            if not V808._pin_equal(final_s_pin,parent_pin_s) or _array_digest(final_s) != _array_digest(base_s):
                raise ValueError("unmodified source asset is not byte-identical to parent: "+sname)
            surface_lineage[sname] = {"surface":sname,"operations":0,"unchanged":True}
        else:
            surface_lineage[sname] = _verify_surface_metadata_lineage(
                base_s, final_s, sname, ops, parent_pin_s, journal_pin)
    append_counts = _midpoint_sequence(parent_state_path, candidate_state_path, parent_state,
                                       candidate_state, ops)
    context = _check_all185_context(candidate_state_path, candidate_state, parent_state_path,
                                    parent_state, context_report_path)
    if candidate_state.get("physical_admission") is not False or int(candidate_state.get("native_steps",-1))!=0:
        raise ValueError("948 candidate claims physical/native admission")
    if candidate_state.get("mass_transport")!="NOT_RUN":
        raise ValueError("948 candidate mass transport must remain NOT_RUN")
    if candidate_state.get("boundary_source_binding_status")!="BATCH_EDGE_SUBDIVISION_ONLY_GLOBAL_185_AUDIT_PENDING":
        raise ValueError("948 candidate lacks explicit batch source-binding status")
    if candidate_state.get("source_crossing_reaudit_required") is not True:
        raise ValueError("948 candidate does not require current source-crossing audit")
    marker=candidate_state.get("edge_bisection",{})
    if marker.get("promotion_ready") is not False or marker.get("all185_surface_audit")!="NOT_RUN":
        raise ValueError("948 candidate bisection marker overstates promotion/audit status")
    assert_outer_source_unchanged(source_pins(parent_state, parent_state_path),
                                  source_pins(candidate_state, candidate_state_path))
    return {
        "schema": "numi.h706.append-aware-current-source-evidence-949.v1",
        "status": "PASS_CURRENT_SOURCE_APPEND_AND_ALL185_EVIDENCE_949",
        "physical_admission": False, "native_steps": 0, "mass_transport": "NOT_RUN",
        "parent_state": parent_pin,"parent_source_audit_status":parent_state["source_audit_status"],
        "batch_report": {"path": str(batch_report_path), "sha256": sha(batch_report_path)},
        "batch_journal": journal_pin, "producer": producer_pin, "core": core_pin,
        "independent_replay_owner": replay_pin, "independent_replay": replay,
        "producer_helper":helper_pin,
        "candidate_state": final_pin, "append_counts": append_counts,
        "volume_point_lineage":volume_point_lineage,"volume_source_maps":volume_source_maps,
        "surface_lineage": surface_lineage,
        "current_exact_context": context, "promotion_ready": False,
    }


def verify_append_batch(parent_state_path, batch_report_path, candidate_state_path,
                        context_report_path=None):
    parent_state_path = Path(parent_state_path).resolve()
    batch_report_path = Path(batch_report_path).resolve()
    candidate_state_path = Path(candidate_state_path).resolve()
    parent_hash = sha(parent_state_path)
    batch = load_json(batch_report_path)
    parent_state = load_json(parent_state_path)
    candidate_state = load_json(candidate_state_path)
    if batch.get("schema") != "numi.h706.long-edge-refinement-batch-944.v1":
        raise ValueError("unexpected 944 batch schema")
    if batch.get("physical_admission") is not False or int(batch.get("native_steps", -1)) != 0 or batch.get("mass_transport") != "NOT_RUN":
        raise ValueError("944 batch carries an unsupported physical/native/mass claim")
    if batch.get("promotion_ready") is not False:
        raise ValueError("944 batch producer may not mark itself promotion-ready")
    if batch.get("input_state") != {"path": str(parent_state_path), "sha256": parent_hash}:
        raise ValueError("944 batch does not bind the exact original parent state")
    producer_pin = pin(REFINER, REFINER_SHA256, "944 producer")
    verifier_pin = pin(REFINEMENT_VERIFIER, REFINEMENT_VERIFIER_SHA256, "944 independent verifier")
    driver_pin = pin(BATCH_DRIVER, BATCH_DRIVER_SHA256, "944 batch driver")
    if batch.get("producer") != producer_pin or batch.get("independent_verifier") != verifier_pin:
        raise ValueError("944 batch's producer/verifier pins differ from frozen owners")
    if sha(BATCH_DRIVER) != driver_pin["sha256"]:
        raise ValueError("944 batch driver pin mismatch")

    steps = batch.get("steps")
    if not isinstance(steps, list) or int(batch.get("operation_count", -1)) != sum(
            1 for x in steps if x.get("status") == "PASS_STEP_WITH_FINAL_BATCH_GATE_PENDING"):
        raise ValueError("944 step inventory or operation count is malformed")
    current_path = parent_state_path
    current_hash = parent_hash
    replay_rows = []
    for ix, step in enumerate(steps, 1):
        if step.get("step") != ix:
            raise ValueError("944 step order/index mismatch")
        if step.get("status") != "PASS_STEP_WITH_FINAL_BATCH_GATE_PENDING":
            raise ValueError("944 batch contains a failed/unverified step")
        if step.get("input_state") != {"path": str(current_path), "sha256": current_hash}:
            raise ValueError("944 step input is not the preceding serialized output")
        if step.get("producer_source_sha256") != REFINER_SHA256:
            raise ValueError("944 step producer source pin mismatch")
        rr = step.get("report")
        report_pin = pin(rr.get("path"), rr.get("sha256"), f"944 step {ix} report")
        outstate_rec = step.get("step_state")
        outstate_pin = pin(outstate_rec.get("path"), outstate_rec.get("sha256"), f"944 step {ix} state")
        readback = step.get("independent_readback")
        rb_pin = pin(readback.get("path"), readback.get("sha256"), f"944 step {ix} readback")
        rb_json = load_json(rb_pin["path"])
        if rb_json.get("status") != "PASS_INDEPENDENT_LONG_EDGE_READBACK" or int(rb_json.get("native_steps", -1)) != 0:
            raise ValueError("944 step's stored independent readback is not PASS")
        # Recompute from frozen 944 verifier against actual serialized bytes.
        replay = V944.verify(current_path, report_pin["path"], outstate_pin["path"])
        if replay.get("status") != "PASS_INDEPENDENT_LONG_EDGE_READBACK":
            raise ValueError("944 independent verifier did not pass step " + str(ix))
        step_report = load_json(report_pin["path"])
        if step_report.get("outputs", {}).get("state", {}).get("sha256") != outstate_pin["sha256"]:
            raise ValueError("944 step report does not pin the serialized output state")
        replay_rows.append({
            "step": ix,
            "report": report_pin,
            "state": outstate_pin,
            "stored_readback": rb_pin,
            "recomputed_status": replay["status"],
            "operation": step_report.get("operation", {}),
        })
        current_path = Path(outstate_pin["path"])
        current_hash = outstate_pin["sha256"]

    if int(batch.get("operation_count", -1)) < 1:
        raise ValueError("append-aware verification requires at least one serialized operation")
    # Batch output may either refer to the last step itself or an exact copied
    # zero-failure state. In both cases the full bytes must match.
    final_rec = batch.get("final_state") or batch.get("last_candidate_state")
    if not isinstance(final_rec, dict):
        raise ValueError("944 batch lacks a terminal candidate-state pin")
    final_pin = pin(final_rec.get("path"), final_rec.get("sha256"), "944 terminal state")
    candidate_pin = pin(candidate_state_path, sha(candidate_state_path), "requested candidate state")
    if final_pin["sha256"] != current_hash or candidate_pin["sha256"] != current_hash:
        raise ValueError("requested candidate bytes do not equal the fully replayed final step")
    if batch.get("status") == "PASS_ZERO_NATIVE_FAILURES":
        if final_pin["path"] != str(candidate_state_path):
            # Batch driver copies the final JSON, so compare raw bytes; SHA match
            # above is exact, no path rewriting can alter this criterion.
            pass
        if int(batch.get("final_metrics", {}).get("bad_cells", -1)) != 0:
            raise ValueError("zero-quality batch status does not have zero final bad cells")
    elif batch.get("status") != "NO_ALL_PASS_COMPLETION":
        raise ValueError("944 batch has an unknown terminal status")

    # Authenticate the original parent with the untouched 808 validator before
    # permitting append-aware interpretation. A parent that is itself append-
    # derived must first have its own 949 receipt recorded and chained by caller.
    if isinstance(parent_state.get("edge_bisection"), dict):
        raise ValueError("parent already contains edge-bisection points; recursively authenticate its prior 949 receipt first")
    base_proof = V808.verify_current_source_evidence(parent_state_path)
    if not base_proof.get("status", "").startswith("PASS_"):
        raise ValueError("pinned pre-bisection parent failed frozen 808 source validation")
    context = _check_all185_context(candidate_state_path, candidate_state, parent_state_path,
                                    parent_state, context_report_path)
    if candidate_state.get("physical_admission") is not False or int(candidate_state.get("native_steps", -1)) != 0:
        raise ValueError("candidate claims physical/native admission")
    if candidate_state.get("mass_transport") != "NOT_RUN":
        raise ValueError("candidate mass transport must remain NOT_RUN")
    if candidate_state.get("boundary_source_binding_status") != "LOCAL_EDGE_SPLIT_ONLY_GLOBAL_185_AUDIT_PENDING":
        # The producer state may be wrapped by the final source-bound state but
        # must retain its explicit append lineage marker.
        raise ValueError("candidate lacks the explicit local-bisection source binding status")
    if candidate_state.get("source_crossing_reaudit_required") is not True:
        raise ValueError("candidate does not require current source-crossing audit")

    return {
        "schema": "numi.h706.append-aware-current-source-evidence-949.v1",
        "status": "PASS_CURRENT_SOURCE_APPEND_AND_ALL185_EVIDENCE_949",
        "physical_admission": False,
        "native_steps": 0,
        "mass_transport": "NOT_RUN",
        "parent_state": {"path": str(parent_state_path), "sha256": parent_hash,
                          "808_status": base_proof.get("status")},
        "batch_report": {"path": str(batch_report_path), "sha256": sha(batch_report_path),
                          "schema": batch["schema"], "status": batch["status"],
                          "operations": len(replay_rows), "steps": replay_rows,
                          "producer": producer_pin, "independent_verifier": verifier_pin,
                          "batch_driver": driver_pin},
        "candidate_state": candidate_pin,
        "current_exact_context": context,
        "append_semantics": "exact immutable pre-existing F32/full-precision prefix; appended rows authenticated by sequential 944 edge/fraction/midpoint/face/volume/CSR replay; O source exact same bytes",
        "promotion_ready": False,
    }


def verify_state(state_path, parent_state=None, batch_report=None, context_report=None):
    state_path = Path(state_path).resolve()
    st = load_json(state_path)
    marker = st.get("edge_bisection")
    if not isinstance(marker, dict):
        # New owner remains a compatibility bridge; all legacy states stay on
        # the frozen 808 implementation unchanged.
        return V808.verify_current_source_evidence(state_path)
    if parent_state is None or batch_report is None:
        raise ValueError("edge-bisection state requires explicit parent-state and batch-report pins")
    batch_path = Path(batch_report).resolve()
    batch_schema = load_json(batch_path).get("schema")
    if batch_schema == "numi.h706.long-edge-refinement-batch-948.v1":
        return verify_compact_batch_948(parent_state, batch_path, state_path, context_report)
    return verify_append_batch(parent_state, batch_path, state_path, context_report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--parent-state", type=Path)
    ap.add_argument("--batch-report", type=Path)
    ap.add_argument("--context-report", type=Path,
                    help="fresh all185 report; needed if producer state inherited a stale exact_context pointer")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    result = verify_state(a.state, a.parent_state, a.batch_report, a.context_report)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
