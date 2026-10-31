#!/usr/bin/env python3
"""791 validator for current source-boundary evidence, retaining the 781 checks.

A current exact-context report is authoritative only when it is pinned to the
state's exact D/I/O candidates. An older source-state pointer is considered
only when there is no current report and every inherited surface pin is
unchanged. Bounded coordinate movement is accepted only through the audited
775 report/config contract; this does not grant physical admission.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
LEGACY = HERE / "quality-partial-cavity-684-root" / "independent_promoter_761.py"
LEGACY_SHA256 = "e5b433292ec5a033d5b04635c96207b4bfb0ce5980c4dcacb7e8e00918573e69"
AUDIT775 = HERE / "combined-surface-diff-audit-270" / "audit_combined_bounded_point_surface_diff_775_root.py"
AUDIT775_SHA256 = "82627919f81528c47e8269b37dee25099f6c2d29f3d2d8858000c50a14956a85"
AUDIT805 = HERE / "combined-surface-diff-audit-270" / "audit_combined_bounded_inner_layer_surface_diff_805_root.py"
AUDIT805_SHA256 = "0f41e4114eeb56032d4bb6a6c39484d2e8dd0de52ff977c7211a800273101234"
BACK_SCOPE = HERE / "current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz"
BACK_SCOPE_SHA256 = "1abbf3c5887cfa507c303e845dafbb78d247e6b7a4893722a5d81ed202fd2676"
ROUTES = HERE / "verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
ROUTES_SHA256 = "ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71"
EXPECTED_EXACT_OWNER_SHA256 = "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
EXPECTED_FILTERED_OWNER_SHA256 = "c3f427432096719b9fe3b91cbd2fc1a6f779228c03456cef85dc8bbbae9da16d"
EXPECTED_ALL185_SCHEMA = "existing-bone-record-remesh-barrier.v1"
EXPECTED_ALL185_COUNT = 938272
EXPECTED_ALL185_NATIVE_SHA256 = "e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5"
EXPECTED_ALL185_MANIFEST_SHA256 = "0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c"
EXPECTED_CONTEXT_SCHEMA = "h706.combined-current-exact-surface-diff-audit.v1"
EXPECTED_CONTEXT_STATUS = "PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
EXPECTED_CONFIG_SCHEMA = "h706.combined-current-exact-surface-diff-config.v1"
ROLE_NAMES = ("D", "I", "O")
MAX_AUDITED_REPAIR_M = 0.0005
MAX_INNER_BACK_REPAIR_M = 5.e-5

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _import_legacy():
    if not LEGACY.is_file() or sha(LEGACY) != LEGACY_SHA256:
        raise ValueError("pinned 761 source-proof helper is missing or changed")
    spec = importlib.util.spec_from_file_location("current_source_evidence_761", LEGACY)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned 761 helper")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

V = _import_legacy()

def _load(path):
    with Path(path).open("r") as f:
        return json.load(f)

def _pin_equal(a, b):
    return (Path(str(a.get("path", ""))).resolve() == Path(str(b.get("path", ""))).resolve()
            and str(a.get("sha256", "")) == str(b.get("sha256", "")))

def _points_f32(npz, role):
    if "points_world_m_f32" in npz:
        p = np.asarray(npz["points_world_m_f32"])
        if p.dtype != np.float32:
            raise ValueError(role + " F32 point lane has wrong dtype")
    elif "points_world_m" in npz:
        p = np.asarray(npz["points_world_m"], dtype=np.float32)
    else:
        raise ValueError(role + " has no point coordinate lane")
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(role + " F32 point lane is malformed")
    return np.ascontiguousarray(p)

def _points_full(npz, role):
    if "points_world_m" not in npz:
        raise ValueError(role + " lacks the full-precision coordinate lane")
    p = np.asarray(npz["points_world_m"])
    if p.ndim != 2 or p.shape[1] != 3 or p.dtype.kind != "f" or not np.isfinite(p).all():
        raise ValueError(role + " full-precision coordinate lane is malformed")
    return p

def _validate_all185(report):
    a = report.get("all185_receipt")
    if not isinstance(a, dict):
        raise ValueError("current exact report lacks all185 receipt")
    if (a.get("schema") != EXPECTED_ALL185_SCHEMA or
        int(a.get("record_count", -1)) != EXPECTED_ALL185_COUNT or
        a.get("native_sha256") != EXPECTED_ALL185_NATIVE_SHA256 or
        a.get("pinned_manifest_sha256") != EXPECTED_ALL185_MANIFEST_SHA256):
        raise ValueError("all185 exact-context receipt pin mismatch")
    ranges = a.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 370:
        raise ValueError("all185 two-domain record range census is incomplete")
    domains = {}
    expected = {
        "source_order_native_f32_replay": (0, 469136),
        "pinned_f64_exact": (469136, EXPECTED_ALL185_COUNT),
    }
    for domain, (start, end) in expected.items():
        selected = [row for row in ranges if row.get("domain") == domain]
        if len(selected) != 185:
            raise ValueError("all185 record domain is incomplete: " + domain)
        cursor = start
        for row in selected:
            lo = int(row.get("negative_face_index_start", -1))
            hi = int(row.get("negative_face_index_end", -1))
            if lo != cursor or hi < lo:
                raise ValueError("all185 record ranges have a gap, overlap, or invalid span")
            cursor = hi
        if cursor != end:
            raise ValueError("all185 record ranges do not cover the pinned record domain")
        domains[domain] = {"records": len(selected), "start": start, "end": end}
    against = report.get("changed_vs_complete_retained_OID_both_walls_all185")
    self_pairs = report.get("changed_OID_joint_self_pairs")
    if not isinstance(against, dict) or int(against.get("count", -1)) != 0:
        raise ValueError("changed surfaces do not have an exact zero all185 crossing result")
    if not isinstance(self_pairs, dict) or int(self_pairs.get("count", -1)) != 0:
        raise ValueError("changed O/I/D surfaces have a nonzero exact self-intersection result")
    wall_receipts = report.get("wall_receipts")
    if not isinstance(wall_receipts, list) or len(wall_receipts) != 2:
        raise ValueError("current all185 report lacks both wall receipts")
    walls = {}
    for row in wall_receipts:
        role = str(row.get("role", ""))
        if role not in ("Skin", "Fat") or role in walls or int(row.get("face_count", -1)) != 90:
            raise ValueError("current report has a malformed wall receipt")
        p, h = V._verify_pin(row, label=role + " wall receipt")
        walls[role] = {"path": str(p.resolve()), "sha256": h, "face_count": 90}
    if set(walls) != {"Skin", "Fat"}:
        raise ValueError("current report does not pin both typed walls")
    return {
        "schema": a["schema"], "record_count": EXPECTED_ALL185_COUNT,
        "native_sha256": a["native_sha256"],
        "pinned_manifest_sha256": a["pinned_manifest_sha256"],
        "ranges_by_domain": domains, "changed_vs_all185_count": 0, "changed_self_count": 0,
        "wall_receipts": walls,
    }

def _has_true_inner_policy_flag(config):
    bounded = config.get("bounded_point_repair", {})
    return isinstance(bounded, dict) and any(
        isinstance(entry, dict) and entry.get("inner_back_layer_repair") is True
        for entry in bounded.values())


def _declared_inner_policy_roles(config):
    bounded = config.get("bounded_point_repair", {})
    if not isinstance(bounded, dict):
        raise ValueError("bounded point repair config is malformed")
    if set(bounded) - set(ROLE_NAMES):
        raise ValueError("bounded point repair contains an unknown role")
    roles = []
    for role, entry in bounded.items():
        if not isinstance(entry, dict):
            raise ValueError("bounded point repair role entry is malformed: " + str(role))
        flag = entry.get("inner_back_layer_repair", False)
        if not isinstance(flag, bool):
            raise ValueError("inner back layer flag must be boolean")
        if not flag:
            continue
        if role not in ("I", "D"):
            raise ValueError("inner back layer repair is I/D only")
        maximum = float(entry.get("maximum_displacement_m", float("nan")))
        if not np.isfinite(maximum) or maximum <= 0. or maximum > MAX_INNER_BACK_REPAIR_M:
            raise ValueError("inner back repair role cap exceeds 50 micrometres")
        roles.append(role)
    return roles


def _validate_805_policy_summary(report, config, inner_roles):
    actual_roles = _declared_inner_policy_roles(config)
    if (not inner_roles or len(inner_roles) != len(set(inner_roles)) or
        set(actual_roles) != set(inner_roles)):
        raise ValueError("805 audit requires the config's explicitly declared I/D inner back layer roles")
    policy = report.get("inner_back_layer_repair_policy")
    if not isinstance(policy, dict):
        raise ValueError("805 report lacks the declared inner back layer policy")
    declared = policy.get("declared_roles")
    if (not isinstance(declared, list) or len(declared) != len(set(declared)) or
        set(declared) != set(inner_roles)):
        raise ValueError("805 report inner-policy roles do not match the config declaration")
    if (float(policy.get("maximum_inner_displacement_m", float("nan"))) != MAX_INNER_BACK_REPAIR_M or
        policy.get("outer_back_scope_points_fixed") is not True or
        policy.get("source_rims_and_all149_routes_fixed") is not True or
        policy.get("classification") != "explicit inner layer reconstruction, not unchanged inner-back geometry"):
        raise ValueError("805 report does not carry the pinned inner-layer policy limits")
    # Every declared inner role is independently restricted to the same 50 um cap.
    bounded = config.get("bounded_point_repair", {})
    for role in inner_roles:
        entry = bounded.get(role, {})
        if entry.get("inner_back_layer_repair") is not True:
            raise ValueError("805 config lacks an explicit inner policy for " + role)
        maximum = float(entry.get("maximum_displacement_m", float("nan")))
        if not np.isfinite(maximum) or maximum <= 0. or maximum > MAX_INNER_BACK_REPAIR_M:
            raise ValueError("805 config exceeds the 50 micrometre role cap for " + role)
    return policy


def _load_805_fixed_rows(point_counts):
    if not BACK_SCOPE.is_file() or sha(BACK_SCOPE) != BACK_SCOPE_SHA256:
        raise ValueError("pinned 597 outer-back scope is missing or changed")
    if not ROUTES.is_file() or sha(ROUTES) != ROUTES_SHA256:
        raise ValueError("pinned 149 source routes are missing or changed")
    scope = {}
    with np.load(BACK_SCOPE, allow_pickle=False) as z:
        for role in ROLE_NAMES:
            key = role + "_vertex_rows"
            if key not in z:
                raise ValueError("805 back-scope archive lacks " + key)
            rows = np.asarray(z[key], dtype=np.int64)
            if rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= point_counts[role]):
                raise ValueError("805 back-scope rows are malformed for " + role)
            scope[role] = set(map(int, rows))
    routes_obj = _load(ROUTES)
    routes = {role: set() for role in ROLE_NAMES}
    records = routes_obj.get("surface_routes")
    if not isinstance(records, list) or len(records) != 149:
        raise ValueError("pinned route inventory must contain exactly 149 routes")
    seen_routes = set()
    for route in records:
        role = route.get("role")
        if role not in routes:
            raise ValueError("pinned route has an unknown source role")
        target = np.asarray(route.get("target_world_m_f32"), dtype=np.float32)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("pinned source route point is malformed")
        key = (role, target.tobytes())
        if key in seen_routes:
            raise ValueError("pinned route inventory contains a duplicate role/point")
        seen_routes.add(key)
        routes[role].add(target.tobytes())
    return scope, routes


def _validate_805_role_movement(role, moved, baseline_points, candidate_points,
                                baseline_full, candidate_full, baseline_faces,
                                inner_roles, fixed_rows):
    scope, route_bits = fixed_rows
    bp = np.asarray(baseline_points, dtype=np.float32)
    cp = np.asarray(candidate_points, dtype=np.float32)
    faces = np.asarray(baseline_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces.size and (faces.min() < 0 or faces.max() >= len(bp))):
        raise ValueError("805 baseline source faces are malformed for " + role)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1).reshape((-1, 2)), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_rows = set(map(int, unique_edges[counts == 1].ravel()))
    route_rows = set()
    for bits in route_bits[role]:
        target = np.frombuffer(bits, dtype=np.float32)
        hit = np.flatnonzero(np.all(bp == target[None, :], axis=1))
        if len(hit) != 1:
            raise ValueError("805 protected route point lacks unique baseline row for " + role)
        route_rows.add(int(hit[0]))
    moved_set = set(map(int, np.asarray(moved, dtype=np.int64)))
    inner = role in inner_roles
    # An explicit I/D inner-back declaration relaxes only the otherwise
    # protected core rows; the signed role cap still applies to all I/D moves.
    protected = set(boundary_rows) | set(route_rows)
    if not inner:
        protected |= scope[role]
    if moved_set & protected:
        if role == "O" and moved_set & scope[role]:
            raise ValueError("805 outer O back-scope point bytes changed")
        if moved_set & boundary_rows:
            raise ValueError("805 source-rim boundary point moved")
        if moved_set & route_rows:
            raise ValueError("805 protected149 route point moved")
        raise ValueError("805 protected back-scope point moved: " + role)
    # Verify the protected rows byte-for-byte in both coordinate lanes, independently
    # of the auditor's JSON summary. Base validation checks the whole role as well.
    if protected:
        rows = np.asarray(sorted(protected), dtype=np.int64)
        if not np.array_equal(bp[rows].view(np.uint32), cp[rows].view(np.uint32)):
            raise ValueError("805 fixed back/rim/route Float32 bytes changed: " + role)
        if np.ascontiguousarray(baseline_full[rows]).tobytes() != np.ascontiguousarray(candidate_full[rows]).tobytes():
            raise ValueError("805 fixed back/rim/route full-precision bytes changed: " + role)


def _audit805_logs_dir_for_report(report_path):
    """Resolve the one corresponding audit-log directory from this report's output class."""
    output_dir = Path(report_path).resolve().parent
    name = output_dir.name.lower()
    has_rollback = "rollback" in name
    has_proposal = "proposal" in name
    if has_rollback == has_proposal:
        raise ValueError("805 report output directory must identify exactly one of rollback or proposal")
    log_name = "rollback-audit-logs" if has_rollback else "proposal-audit-logs"
    return output_dir.parent / log_name


def _validate_audit805_owner(report, report_path, report_sha, config_path):
    if not AUDIT805.is_file() or sha(AUDIT805) != AUDIT805_SHA256:
        raise ValueError("pinned bounded inner-layer 805 auditor is missing or changed")
    log_dir = _audit805_logs_dir_for_report(report_path)
    command_path = log_dir / "audit805.command.json"
    if not command_path.is_file():
        raise ValueError("inner-layer movement lacks its recorded 805 audit invocation")
    command = _load(command_path)
    argv = command.get("command")
    if (int(command.get("returncode", -1)) != 0 or not isinstance(argv, list) or
        len(argv) < 6 or Path(argv[1]).resolve() != AUDIT805.resolve()):
        raise ValueError("recorded 805 audit invocation did not pass or used another tool")
    expected_pairs = {"--config": str(Path(config_path).resolve()),
                      "--out": str(Path(report_path).parent.resolve())}
    for key, expected_value in expected_pairs.items():
        try:
            actual_value = argv[argv.index(key) + 1]
        except (ValueError, IndexError):
            raise ValueError("recorded 805 invocation lacks " + key)
        if str(Path(actual_value).resolve()) != expected_value:
            raise ValueError("recorded 805 invocation input/output differs: " + key)
    stdout_path = command_path.with_name("audit805.stdout.log")
    stderr_path = command_path.with_name("audit805.stderr.log")
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("recorded 805 stdout/stderr logs are missing")
    lines = [line.strip() for line in stdout_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError("recorded 805 audit stdout is empty")
    tail = lines[-1].split()
    if (len(tail) != 6 or tail[0] != "against_count" or tail[1] != "0" or
        tail[2] != "joint_self_count" or tail[3] != "0" or
        tail[4] != "report_sha256" or tail[5] != report_sha):
        raise ValueError("recorded 805 audit stdout does not bind this exact zero-crossing report")
    owner = report.get("exact_owner")
    if not isinstance(owner, dict):
        raise ValueError("805 movement report lacks exact-owner pins")
    owner_path, owner_sha = V._verify_pin(owner, label="exact intersection owner")
    if owner_sha != EXPECTED_EXACT_OWNER_SHA256 or owner.get("expected_sha256") != EXPECTED_EXACT_OWNER_SHA256:
        raise ValueError("805 movement report uses an unexpected exact intersection owner")
    filter_entry = {"path": owner.get("filtered_owner_path"), "sha256": owner.get("filtered_owner_sha256")}
    filter_path, filter_sha = V._verify_pin(filter_entry, label="filtered exact owner")
    if filter_sha != EXPECTED_FILTERED_OWNER_SHA256:
        raise ValueError("805 movement report uses an unexpected filtered exact owner")
    return {
        "mode": "AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_805",
        "audit805": {"path": str(AUDIT805), "sha256": AUDIT805_SHA256},
        "exact_owner": {"path": str(owner_path.resolve()), "sha256": owner_sha},
        "filtered_owner": {"path": str(filter_path.resolve()), "sha256": filter_sha},
        "audit_invocation": {"path": str(command_path.resolve()), "sha256": sha(command_path),
            "stdout": {"path": str(stdout_path.resolve()), "sha256": sha(stdout_path)},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": sha(stderr_path)}},
    }


def _validate_audit_owner(report, moved, report_path, report_sha, config_path, config):
    has_805_declaration = ("inner_back_layer_repair_policy" in report or _has_true_inner_policy_flag(config))
    inner_roles = _declared_inner_policy_roles(config) if has_805_declaration else []
    uses_805 = has_805_declaration
    if uses_805:
        _validate_805_policy_summary(report, config, inner_roles)
        inner_moved = sum(int(report.get("roles", {}).get(role, {}).get("points", {}).get("bounded_moved_point_count", -1))
                          for role in inner_roles)
        if inner_moved <= 0:
            raise ValueError("805 declared inner policy has no actual moved I/D point rows")
        return _validate_audit805_owner(report, report_path, report_sha, config_path)
    if not moved:
        return {"mode": "UNCHANGED_COORDINATES", "audit775": None}
    if not AUDIT775.is_file() or sha(AUDIT775) != AUDIT775_SHA256:
        raise ValueError("pinned bounded-point 775 auditor is missing or changed")
    command_path = Path(report_path).parent.parent / "proposal-audit-logs" / "audit775.command.json"
    if not command_path.is_file():
        raise ValueError("bounded movement lacks its recorded 775 audit invocation")
    command = _load(command_path)
    argv = command.get("command")
    if (int(command.get("returncode", -1)) != 0 or not isinstance(argv, list) or
        len(argv) < 6 or Path(argv[1]).resolve() != AUDIT775.resolve()):
        raise ValueError("recorded 775 audit invocation did not pass or used another tool")
    expected_pairs = {"--config": str(Path(config_path).resolve()),
                      "--out": str(Path(report_path).parent.resolve())}
    for key, expected_value in expected_pairs.items():
        try:
            actual_value = argv[argv.index(key) + 1]
        except (ValueError, IndexError):
            raise ValueError("recorded 775 invocation lacks " + key)
        if str(Path(actual_value).resolve()) != expected_value:
            raise ValueError("recorded 775 invocation input/output differs: " + key)
    stdout_path = command_path.with_name("audit775.stdout.log")
    stderr_path = command_path.with_name("audit775.stderr.log")
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise ValueError("recorded 775 stdout/stderr logs are missing")
    stdout_sha = sha(stdout_path)
    stderr_sha = sha(stderr_path)
    lines = [line.strip() for line in stdout_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError("recorded 775 audit stdout is empty")
    tail = lines[-1].split()
    if (len(tail) != 6 or tail[0] != "against_count" or tail[1] != "0" or
        tail[2] != "joint_self_count" or tail[3] != "0" or
        tail[4] != "report_sha256" or tail[5] != report_sha):
        raise ValueError("recorded 775 audit stdout does not bind this exact zero-crossing report")
    owner = report.get("exact_owner")
    if not isinstance(owner, dict):
        raise ValueError("bounded movement report lacks exact-owner pins")
    owner_path, owner_sha = V._verify_pin(owner, label="exact intersection owner")
    if (owner_sha != EXPECTED_EXACT_OWNER_SHA256 or
        owner.get("expected_sha256") != EXPECTED_EXACT_OWNER_SHA256):
        raise ValueError("bounded movement report uses an unexpected exact intersection owner")
    filter_entry = {"path": owner.get("filtered_owner_path"),
                    "sha256": owner.get("filtered_owner_sha256")}
    filter_path, filter_sha = V._verify_pin(filter_entry, label="filtered exact owner")
    if filter_sha != EXPECTED_FILTERED_OWNER_SHA256:
        raise ValueError("bounded movement report uses an unexpected filtered exact owner")
    return {
        "mode": "AUDITED_BOUNDED_POINT_MOVEMENT_630_775",
        "audit775": {"path": str(AUDIT775), "sha256": AUDIT775_SHA256},
        "exact_owner": {"path": str(owner_path.resolve()), "sha256": owner_sha},
        "filtered_owner": {"path": str(filter_path.resolve()), "sha256": filter_sha},
        "audit_invocation": {
            "path": str(command_path.resolve()), "sha256": sha(command_path),
            "stdout": {"path": str(stdout_path.resolve()), "sha256": stdout_sha},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": stderr_sha},
        },
    }

def _reported_moved_point_count(points, measured_moved, role):
    """Check report count against exact-array movement, allowing legacy zero only."""
    if "bounded_moved_point_count" in points:
        reported = int(points["bounded_moved_point_count"])
    elif points.get("f32_coordinate_bytes_unchanged") is True:
        # This is provisional only: caller compares exact F32 and full-precision
        # arrays below. Any measured movement makes this zero disagree and fail.
        reported = 0
    else:
        reported = -1
    if reported != int(measured_moved):
        raise ValueError("reported moved-point count differs from exact F32 arrays: " + role)
    return reported

def _validate_current_context(state_path, state, source_pins):
    """Validate the state-bound exact report, never following an older pointer."""
    rec = state.get("exact_context")
    if not isinstance(rec, dict):
        raise ValueError("current state has no exact-context report")
    report_path, report_sha = V._verify_pin(rec, label="current exact-context report")
    V._check_not_747(report_path, "current exact-context report")
    report = _load(report_path)
    if (report.get("schema") != EXPECTED_CONTEXT_SCHEMA or
        report.get("status") != EXPECTED_CONTEXT_STATUS):
        raise ValueError("current exact-context report is not the all185 PASS")
    if report.get("physical_admission") is not False or int(report.get("native_steps", -1)) != 0:
        raise ValueError("exact-context report has physical/runtime admission metadata")
    config_path, config_sha = V._verify_pin(report.get("config"), label="current exact-context config")
    config = _load(config_path)
    if config.get("schema") != EXPECTED_CONFIG_SCHEMA:
        raise ValueError("exact-context config schema mismatch")
    if set(config.get("candidate_surfaces", {})) != set(ROLE_NAMES):
        raise ValueError("exact-context config candidate surface inventory mismatch")
    if set(config.get("baseline_candidates", {})) != set(ROLE_NAMES):
        raise ValueError("exact-context config baseline surface inventory mismatch")
    if not _pin_equal(rec, {"path": str(report_path), "sha256": report_sha}):
        raise ValueError("current state exact-context report pin mismatch")

    bounded = config.get("bounded_point_repair", {})
    if not isinstance(bounded, dict):
        raise ValueError("bounded point repair config is malformed")
    has_805_declaration = ("inner_back_layer_repair_policy" in report or _has_true_inner_policy_flag(config))
    inner_roles = _declared_inner_policy_roles(config) if has_805_declaration else []
    uses_805 = has_805_declaration
    if uses_805:
        _validate_805_policy_summary(report, config, inner_roles)
        point_counts = {}
        for role in ROLE_NAMES:
            with np.load(source_pins[role]["path"], allow_pickle=False) as source_npz:
                lane = source_npz["points_world_m_f32"] if "points_world_m_f32" in source_npz else source_npz["points_world_m"]
                point_counts[role] = len(lane)
        fixed_rows = _load_805_fixed_rows(point_counts)
    else:
        fixed_rows = None
    movement = {}
    has_moved = False
    for role in ROLE_NAMES:
        cand_cfg = config["candidate_surfaces"][role]
        cand_path, cand_sha = V._verify_pin(cand_cfg, label="config candidate " + role)
        if not _pin_equal({"path": str(cand_path), "sha256": cand_sha}, source_pins[role]):
            raise ValueError("exact-context config candidate differs from current state: " + role)
        role_report = report.get("roles", {}).get(role)
        if not isinstance(role_report, dict):
            raise ValueError("exact-context report lacks role " + role)
        cand_report_path, cand_report_sha = V._verify_pin(role_report.get("candidate"),
                                                          label="report candidate " + role)
        if not _pin_equal({"path": str(cand_report_path), "sha256": cand_report_sha}, source_pins[role]):
            raise ValueError("exact-context report candidate differs from current state: " + role)
        base_cfg = config["baseline_candidates"][role]
        base_path, base_sha = V._verify_pin(base_cfg, label="config baseline " + role)
        base_report_path, base_report_sha = V._verify_pin(role_report.get("baseline"),
                                                          label="report baseline " + role)
        if not _pin_equal({"path": str(base_report_path), "sha256": base_report_sha},
                          {"path": str(base_path), "sha256": base_sha}):
            raise ValueError("report baseline pin differs from config: " + role)

        points = role_report.get("points", {})
        if not isinstance(points, dict):
            raise ValueError("exact-context point proof missing: " + role)
        if (points.get("candidate_to_baseline_rows_bijective") is not True or
            points.get("candidate_rows_injective") is not True or
            points.get("full_precision_array_checked") is not True or
            int(points.get("retired_baseline_point_count", -1)) != 0 or
            points.get("retirement_authorization") is not None):
            raise ValueError("exact-context point identity/full-precision proof failed: " + role)
        baseline_npz = np.load(base_path, allow_pickle=False)
        candidate_npz = np.load(cand_path, allow_pickle=False)
        try:
            bp = _points_f32(baseline_npz, role)
            cp = _points_f32(candidate_npz, role)
            bfull = _points_full(baseline_npz, role)
            cfull = _points_full(candidate_npz, role)
            if bp.shape != cp.shape or bfull.shape != cfull.shape or bfull.dtype != cfull.dtype:
                raise ValueError(role + " candidate point row domain changed")
            moved = np.flatnonzero(np.any(bp.view(np.uint32) != cp.view(np.uint32), axis=1))
            if uses_805:
                _validate_805_role_movement(role, moved, bp, cp, bfull, cfull,
                    np.asarray(baseline_npz["faces"], dtype=np.int64), inner_roles, fixed_rows)
            reported_moved = _reported_moved_point_count(points, len(moved), role)
            if int(points.get("baseline_point_count", -1)) != len(bp) or int(points.get("candidate_point_count", -1)) != len(cp):
                raise ValueError("reported candidate/baseline point counts mismatch: " + role)
            unchanged = len(moved) == 0
            if points.get("f32_coordinate_bytes_unchanged") is not unchanged:
                raise ValueError("reported F32 coordinate identity differs from point arrays: " + role)
            if unchanged:
                if not (np.array_equal(bp.view(np.uint32), cp.view(np.uint32)) and
                        bfull.tobytes(order="C") == cfull.tobytes(order="C")):
                    raise ValueError("unchanged-coordinate report does not match full arrays: " + role)
                movement[role] = {"moved_points": 0, "maximum_displacement_m": 0.0}
            else:
                has_moved = True
                auth = bounded.get(role)
                if not isinstance(auth, dict):
                    raise ValueError("coordinate movement lacks explicit bounded 775 authorization: " + role)
                maximum = float(auth.get("maximum_displacement_m", float("nan")))
                if not np.isfinite(maximum) or maximum <= 0 or maximum > MAX_AUDITED_REPAIR_M:
                    raise ValueError("bounded point repair trust cap is invalid: " + role)
                unchanged_rows = np.ones(len(bp), dtype=bool)
                unchanged_rows[moved] = False
                if (not np.array_equal(bfull[unchanged_rows], cfull[unchanged_rows]) or
                    not np.array_equal(cfull[moved].astype(np.float32).view(np.uint32),
                                       cp[moved].view(np.uint32))):
                    raise ValueError("full-precision coordinates are inconsistent with moved F32 rows: " + role)
                d32 = np.linalg.norm(cp[moved].astype(np.float64) - bp[moved].astype(np.float64), axis=1)
                d64 = np.linalg.norm(cfull[moved].astype(np.float64) - bfull[moved].astype(np.float64), axis=1)
                max_move = float(max(np.max(d32), np.max(d64)))
                if max_move > maximum:
                    raise ValueError("measured point movement exceeds pinned cap: " + role)
                movement[role] = {
                    "moved_points": int(len(moved)),
                    "maximum_displacement_m": max_move,
                    "authorized_maximum_displacement_m": maximum,
                }
        finally:
            baseline_npz.close()
            candidate_npz.close()

    if has_moved and report.get("point_namespace", {}).get("mapping") != (
        "explicit bounded point repair retains authenticated row IDs; unchanged roles map by exact F32 bytes"
    ):
        raise ValueError("bounded point movement report lacks the explicit audited row-identity policy")
    if report.get("paired_D_retirement") is not None:
        raise ValueError("current source proof unexpectedly includes D point retirement")
    retired_faces = list(map(int, report.get("retired_baseline_face_ids", [])))
    if (len(retired_faces) != len(set(retired_faces)) or
        len(retired_faces) != int(report.get("removed_baseline_face_total", -1))):
        raise ValueError("retired face inventory is malformed")
    counts = report.get("candidate_changed_record_count_by_role", {})
    if set(counts) != set(ROLE_NAMES) or sum(map(int, counts.values())) != int(report.get("changed_face_total", -1)):
        raise ValueError("exact-context changed-face record counts do not close")
    if sum(int(report.get("roles", {}).get(r, {}).get("removed_face_count", -1)) for r in ROLE_NAMES) != int(report.get("removed_baseline_face_total", -1)):
        raise ValueError("exact-context removed-face counts do not close")
    all185 = _validate_all185(report)
    audit = _validate_audit_owner(report, has_moved, report_path, report_sha, config_path, config)
    return {
        "status": "PASS_CURRENT_EXACT_SOURCE_EVIDENCE",
        "mode": audit["mode"],
        "state": {"path": str(Path(state_path).resolve()), "sha256": sha(state_path)},
        "report": {"path": str(report_path.resolve()), "sha256": report_sha},
        "config": {"path": str(config_path.resolve()), "sha256": config_sha},
        "source_surfaces": source_pins,
        "movement_by_role": movement,
        "all185": all185,
        "audit_implementation": audit,
        "inherited_stale_pointer": state.get("source_crossing_evidence_inherited_from_unchanged_surfaces"),
        "physical_admission": False,
        "native_steps": 0,
    }

def verify_current_source_evidence(state_path, state_override=None):
    """Return authenticated current source evidence or raise ValueError."""
    state_path = Path(state_path).resolve()
    raw = _load(state_path) if state_override is None else state_override
    state, _wrapped = V._unwrap_state(raw)
    if not isinstance(state, dict):
        raise ValueError("source verified-state is not an object")
    source_pins = V._source_pins(state)
    if "surface_crossings" in state and int(state["surface_crossings"]) != 0:
        raise ValueError("current state does not record exact zero surface crossings")
    if ("source_metadata_current_step_only" in state and
        state["source_metadata_current_step_only"] is not True):
        raise ValueError("current source metadata is not current-step-only")

    if isinstance(state.get("exact_context"), dict):
        # A current report is authoritative. Any mismatch is fatal; never fall
        # back to an ancestor with stale O/I/D coordinates.
        return _validate_current_context(state_path, state, source_pins)

    # Delegate legacy unchanged-coordinate ancestry to the pinned 761
    # _authenticated_source_proof, which in turn calls its _context_anchor.
    # It is reached only when the current state has no exact_context at all.
    legacy = V._authenticated_source_proof(state_path, state)
    return {
        "status": "PASS_INHERITED_UNCHANGED_SOURCE_EVIDENCE",
        "mode": "UNCHANGED_COORDINATE_354_STYLE_INHERITANCE",
        "state": {"path": str(state_path), "sha256": sha(state_path)},
        "source_surfaces": source_pins,
        "anchor": legacy["anchor"],
        "chain": legacy["chain"],
        "physical_admission": False,
        "native_steps": 0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    result = verify_current_source_evidence(a.state)
    print(json.dumps(result, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()

