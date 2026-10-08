#!/usr/bin/env python3
"""Bounded exact 8-pose native lung interface audit using existing Human owners."""
from __future__ import annotations
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
HERE = E / "native-lung-transformed-pose-audit-runner-1096"
BASE = E / "native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1042_pinned.py"
BASE_SHA = "791e2acdfdf917cd8554d83887f78682121a181216a07ff96c7ca2b985214345"
FEATURE = E / "native-lung-exact-classifier-index-1084/native_feature_index.py"
FEATURE_SHA = "2d9eab88a802a0223cad34f8b9d42066e55214d53dab5b7e0fc0f9b6aac9cd5e"
D_ADAPTER = E / "native-lung-d-lobe-boundary-parity-1086/native_feature_index_1085.py"
D_ADAPTER_SHA = "a982d97188fc8c207380afc2d3f8fe0e5934228cb816c2bea827115c84050928"
SOURCE_RULE = Path("/Users/n/numi-human-lung-contact-classifier-001/src/numilab_human/lung_contact_classification.py")
SOURCE_RULE_SHA = "fd95fa6cca92706faa9c7dcf88fbe2243d1a7d865b391d2f632cc0230cf42710"
WRAPPER_1079 = E / "native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1079.py"
WRAPPER_1079_SHA = "5578bc17bfefcbc589e33e6612242de745495702bbd12b82716e715d5ef7f88e"
STEP0_REGRESSION_REPORT = E / "native-lung-transformed-pose-audit-1079/native-1078-step0-corrected-cross-003/report.json"
STEP0_REGRESSION_REPORT_SHA = "dfc11b658c6266691a759936c2adf717240ea8ea3f06251d180e54165822c495"
STEP0_REGRESSION_RUN = E / "final-native-scene-preflight-936/skin-927-lung-1078-viewer-018-v015-attempt1/native-run"
LINEAGE_V2 = HERE / "lobe_lineage_v2.py"
LINEAGE_V2_SHA = "8845a723cfbecdf480359fa79c23bddb840db91ed1320a0016727f974de58ba9"
V8_DMAP_ADAPTER = HERE / "v8_dmap_adapter.py"
V8_DMAP_ADAPTER_SHA = "9a55338d874ecc4695e1545829c21266b9b5734d8ca8d64acab817906091c7bd"
MEMORY_PROBE_REPORT = E / "native-lung-transformed-pose-audit-runner-1090/attempt-002-step0-1078/report.json"
MEMORY_PROBE_REPORT_SHA = "e59aecf383d85c27bb2dfa63a9e2aa8234c8e690551a0a9fcbe1997e5f0e7f1e"
MEMORY_PROBE_MANIFEST = E / "native-lung-transformed-pose-audit-runner-1090/attempt-002-step0-1078/manifest.json"
MEMORY_PROBE_MANIFEST_SHA = "29f992287b908e60f727029ada20eaf25638a93c0d1e66b6f715ac44c0abff5a"
EXPECTED_ROWS = (305, 306, 307, 308, 309, 311)
MAX_WORKERS = 3
MEMORY_BUDGET_GIB = 12.0
GIB = 1024 ** 3

def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def load_module(path: Path | str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def require_hash(path: Path | str, expected: str):
    path = Path(path).resolve()
    if not path.is_file() or sha(path) != expected:
        raise ValueError("pinned input mismatch: " + str(path))
    return path

def option_many(argv, key, count):
    if argv.count(key) != 1 or argv.index(key) + count >= len(argv):
        raise ValueError("missing/duplicate argument: " + key)
    i = argv.index(key)
    return argv[i + 1:i + 1 + count]

def validate_terminal_capture(steps, requested_roots):
    steps = [int(x) for x in steps]
    requested_roots = int(requested_roots)
    if len(steps) != 8 or len(set(steps)) != 8:
        raise ValueError("exactly eight unique accepted capture steps are required")
    if requested_roots not in steps:
        raise ValueError("the requested terminal accepted step is not captured")
    if steps[-1] != requested_roots:
        raise ValueError("terminal accepted state must be the final capture")
    return {"accepted_steps": steps, "requested_roots": requested_roots,
            "terminal_accepted_step": requested_roots}


def exact_pair_coverage(self_rows, cross_rows, row_ids):
    expected_self = list(row_ids)
    expected_cross = [list(pair) for pair in itertools.combinations(row_ids, 2)]
    return {
        "self": [x.get("owner") for x in self_rows] == expected_self,
        "cross": [x.get("owners") for x in cross_rows] == expected_cross,
        "expected_self": expected_self,
        "expected_cross": expected_cross,
    }


def validate_memory_probe(probe, *, probe_path, reference_triangle_count,
                          worker_count, candidate_triangle_count):
    probe_path = Path(probe_path).resolve()
    if probe_path != MEMORY_PROBE_REPORT.resolve() or sha(probe_path) != MEMORY_PROBE_REPORT_SHA:
        raise ValueError("parallel memory gate requires the pinned completed 1078 step-0 probe")
    if sha(MEMORY_PROBE_MANIFEST) != MEMORY_PROBE_MANIFEST_SHA:
        raise ValueError("parallel memory probe manifest changed")
    if (probe.get("status") != "reader_regression_only" or probe.get("accepted_step") != 0
            or probe.get("source_adapter", {}).get("base_sha256") != BASE_SHA
            or probe.get("source_adapter", {}).get("feature_1084_sha256") != FEATURE_SHA
            or probe.get("source_adapter", {}).get("d_adapter_sha256") != D_ADAPTER_SHA
            or probe.get("source_adapter", {}).get("source_rule_sha256") != SOURCE_RULE_SHA):
        raise ValueError("memory gate needs the pinned compatible step-0 reader regression")
    peak = int(probe.get("max_rss_bytes", 0))
    if peak <= 0 or reference_triangle_count <= 0 or candidate_triangle_count <= 0:
        raise ValueError("memory probe resource fields must be positive")
    scale = max(1.0, candidate_triangle_count / reference_triangle_count)
    per_worker = peak * scale * 1.75
    total = worker_count * per_worker
    if worker_count < 1 or worker_count > MAX_WORKERS or total > MEMORY_BUDGET_GIB * GIB:
        raise ValueError("requested concurrency exceeds 12 GiB conservative RSS estimate")
    return {"probe_report": str(probe_path), "probe_report_sha256": MEMORY_PROBE_REPORT_SHA,
            "probe_manifest": str(MEMORY_PROBE_MANIFEST), "probe_manifest_sha256": MEMORY_PROBE_MANIFEST_SHA,
            "reference_run_path": probe.get("run_path"), "reference_nha_sha256": probe.get("NHA", {}).get("sha256"),
            "probe_rss_bytes": peak, "reference_triangle_count": reference_triangle_count,
            "candidate_triangle_count": candidate_triangle_count, "geometry_scale": scale,
            "per_worker_estimate_bytes_with_75pct_reserve": int(per_worker),
            "workers": worker_count, "total_estimated_bytes": int(total),
            "budget_gib": MEMORY_BUDGET_GIB}


def load_native_adapters(base):
    feature = load_module(require_hash(FEATURE, FEATURE_SHA), "native_feature_1084")
    adapter = load_module(require_hash(D_ADAPTER, D_ADAPTER_SHA), "native_d_lobe_adapter_1085")
    require_hash(SOURCE_RULE, SOURCE_RULE_SHA)
    install = getattr(adapter, "install", None)
    if not callable(install):
        raise ValueError("D/lobe adapter has no install(base) entry point")
    install(base, feature_1084=feature)
    if not LINEAGE_V2_SHA or sha(LINEAGE_V2) != LINEAGE_V2_SHA:
        raise ValueError("current-row v2 lineage reader hash mismatch")
    lineage_v2 = load_module(LINEAGE_V2, "native_lobe_lineage_v2")
    v8_dmap = load_module(require_hash(V8_DMAP_ADAPTER, V8_DMAP_ADAPTER_SHA), "native_v8_dmap_adapter")
    return {"feature": feature, "d_adapter": adapter, "lineage_v2": lineage_v2, "v8_dmap": v8_dmap}

def load_owner_context(base, run_path, expected_nha_path=None, expected_nha_sha=None,
                       map_reports=(), lobe_lineage_report=None, regression_1078=False,
                       d_map_composition_report=None, geometry_only_area_mismatch=False):
    run_path = Path(run_path).resolve()
    owner = load_module(base.OWNER, "native_geometry_owner_937")
    owner_module, steps, invocation, metadata = owner.preflight(run_path)
    steps = [int(s) for s in steps]
    verified_steps, requested_roots, dt = owner.validate_documents(
        invocation, metadata, (run_path / "native.log").read_text()
    )
    verified_steps = [int(s) for s in verified_steps]
    if steps != verified_steps:
        raise ValueError("owner preflight and run-metadata capture lists disagree")
    capture_identity = validate_terminal_capture(steps, requested_roots)
    steps = capture_identity["accepted_steps"]
    if not metadata.get("loaded_metal_runtime", {}).get("verified", False):
        raise ValueError("owner preflight does not prove loaded runtime")
    argv = invocation["argv"]
    nha_path = Path(owner.option(argv, "--torso-anatomy-payload")).resolve()
    nha_sha = invocation["asset_sha256"].get(str(nha_path))
    if not nha_sha or sha(nha_path) != nha_sha:
        raise ValueError("current NHA is not invocation-hash bound")
    if expected_nha_path is not None and nha_path != Path(expected_nha_path).resolve():
        raise ValueError("run payload does not equal selected NHA path")
    if expected_nha_sha is not None and nha_sha != expected_nha_sha:
        raise ValueError("run payload hash does not equal selected NHA hash")
    parser = load_module(base.PARSER, "native_anatomy_parser")
    if sha(base.PARSER) != base.PARSER_SHA:
        raise ValueError("pinned native anatomy parser changed")
    nha_rows = parser.parse_payload(nha_path)[1]
    if any(sid not in nha_rows for sid in EXPECTED_ROWS):
        raise ValueError("current NHA lacks one of six audit rows")
    if regression_1078:
        if run_path != STEP0_REGRESSION_RUN.resolve():
            raise ValueError("legacy regression mode accepts only the pinned 1078 run")
        wrapper = load_module(require_hash(WRAPPER_1079, WRAPPER_1079_SHA), "native_context_1079")
        context = wrapper.load_context(base)
        if context["final_path"].resolve() != nha_path or sha(nha_path) != context["map_doc"]["inputs"].get(str(nha_path)):
            raise ValueError("1079 fixture maps are not bound to accepted payload")
        maps_doc = context["map_doc"]
        lineage = context["lineage"]
        map_docs = []
        context_paths = [Path(context["comp_path"]), Path(context["final_path"])]
        context_paths += [Path(p) for p in context["lineage"]["inputs"]]
        context_paths += [Path(p) for p in context["lineage"]["manifest_products"]]
        context_paths += [Path(p) for p in context["map_doc"]["inputs"]]
    else:
        if lobe_lineage_report is None:
            raise ValueError("current source-proven lobe-lobe lineage report is required")
        d_map_doc = None
        if d_map_composition_report is not None:
            if map_reports:
                raise ValueError("use the selected v8 composition report or legacy five maps, not both")
            d_map_adapter = load_module(require_hash(V8_DMAP_ADAPTER, V8_DMAP_ADAPTER_SHA),
                                        "native_v8_dmap_adapter_context")
            d_map_doc = d_map_adapter.load_v8_current_dmap(
                base=base, composition_report_path=d_map_composition_report,
                final_nha_path=nha_path, final_nha_sha=nha_sha, final_rows=nha_rows,
            )
            maps_doc = d_map_doc
            map_docs = []
            context_paths = [Path(d_map_doc["path"]), Path(d_map_doc["map_path"])]
            context_paths.extend(Path(p) for p in d_map_doc["inputs"])
        else:
            if len(map_reports) != 5:
                raise ValueError("legacy full mode requires five explicit D311/lobe map reports")
            map_docs = [base.load_map(p, nha_path, nha_sha, True) for p in map_reports]
            maps = {}
            source_rows = {}
            for doc in map_docs:
                base.verify_map_nha(doc, nha_rows)
                for sid, arrays in doc["source_rows"].items():
                    if sid in source_rows and any(not base.np.array_equal(source_rows[sid][k], arrays[k]) for k in arrays):
                        raise ValueError("D/lobe map reports disagree on source row arrays")
                    source_rows[sid] = arrays
                for sid, pair_map in doc["pairs"].items():
                    if sid in maps:
                        raise ValueError("duplicate D311/lobe map for stable ID " + str(sid))
                    maps[int(sid)] = pair_map
            if set(maps) != set(base.LOBES):
                raise ValueError("maps must explicitly include every lobe pair 305-309")
            maps_doc = {"pairs": maps, "source_rows": source_rows}
            context_paths = []
            for doc in map_docs:
                context_paths.extend([Path(doc["path"]), Path(doc["map_path"])])
                context_paths.extend(Path(p) for p in doc["inputs"])
        if sha(LINEAGE_V2) != LINEAGE_V2_SHA:
            raise ValueError("current-row v2 lineage reader hash mismatch")
        lineage_v2 = load_module(LINEAGE_V2, "native_lobe_lineage_v2_context")
        lineage = lineage_v2.load_v2_bridge(
            base=base, report_path=lobe_lineage_report, final_nha_path=nha_path,
            final_nha_sha=nha_sha, final_rows=nha_rows,
        )
        if set(lineage["pairs"]) != set(base.LOBE_PAIRS):
            raise ValueError("lobe lineage must explicitly declare all ten pairs including zero pairs")
        context_paths += [
            Path(lineage["path"]), Path(lineage["map_path"]), Path(lineage["edge_path"]),
            Path(lineage["checksum_manifest_path"]),
        ]
        context_paths.extend(Path(p) for p in lineage["inputs"])
        context_paths.extend(Path(p) for p in lineage["manifest_products"])
    area = respiration_area_binding(base, invocation, nha_path, nha_sha, nha_rows,
                                     allow_mismatch=geometry_only_area_mismatch)
    tracked = {
        Path(base.OWNER), Path(base.PARSER), Path(base.PRED), Path(base.__file__),
        Path(base.ROOT / "src/numilab_human/common_atlas_skin_clearance.py"),
        FEATURE, D_ADAPTER, SOURCE_RULE, LINEAGE_V2, V8_DMAP_ADAPTER,
        run_path / "invocation.json", run_path / "run-metadata.json", run_path / "native.log",
        nha_path,
    }
    tracked.update(Path(p) for p in invocation.get("asset_sha256", {}))
    for step in steps:
        pack = run_path / ("accepted-geometry/step-%d.mrvpack" % step)
        receipt = pack.with_suffix(".receipt.json")
        tracked.update((pack, receipt))
    tracked.update(context_paths)
    for path in sorted(tracked):
        if not path.is_file():
            raise ValueError("tracked input missing: " + str(path))
    before = {str(p.resolve()): sha(p) for p in sorted(tracked)}
    return {
        "run_path": run_path, "steps": steps, "owner": owner, "owner_module": owner_module,
        "invocation": invocation, "metadata": metadata, "requested_roots": int(requested_roots),
        "dt_seconds": float(dt), "nha_path": nha_path, "nha_sha": nha_sha,
        "nha_rows": nha_rows, "maps_doc": maps_doc, "map_docs": map_docs,
        "d_map_doc": (d_map_doc if not regression_1078 else None), "lineage": lineage,
        "area_binding": area, "tracked": tracked, "input_hashes": before,
    }

def geo_sha(base, row):
    return hashlib.sha256(
        base.np.ascontiguousarray(row["vertices6"], dtype="<f4").tobytes()
        + base.np.ascontiguousarray(row["faces"], dtype="<i8").tobytes()
    ).hexdigest()

def respiration_area_binding(base, invocation, nha_path, nha_sha, nha_rows, *, allow_mismatch=False):
    args = option_many(invocation["argv"], "--resting-scene", 2)
    config_path = Path(args[1]).resolve()
    config_sha = invocation["asset_sha256"].get(str(config_path))
    if not config_sha or sha(config_path) != config_sha:
        raise ValueError("resting respiration config is not invocation-hash bound")
    config = json.loads(config_path.read_text())
    candidate = config.get("source_geometry_candidate")
    if not isinstance(candidate, dict):
        raise ValueError("respiration config lacks geometry-derived area binding")
    deriv_path = Path(candidate.get("derivation_path", "")).resolve()
    deriv_sha = candidate.get("derivation_sha256")
    if not deriv_path.is_file() or not deriv_sha or sha(deriv_path) != deriv_sha:
        raise ValueError("effective-area derivation missing/hash mismatch")
    deriv = json.loads(deriv_path.read_text())
    deriv_payload_sha = deriv.get("payload", {}).get("sha256")
    candidate_payload_sha = candidate.get("payload_sha256")
    per_lobe = {str(sid): geo_sha(base, nha_rows[sid]) for sid in range(305, 310)}
    deriv_lobes = deriv.get("per_lobe_geometry_sha256")
    candidate_lobes = candidate.get("per_lobe_geometry_sha256")
    area = float(config.get("diaphragm_area_m2"))
    derived_area = float(deriv.get("candidate_runtime_area_m2_float32"))
    candidate_area = float(candidate.get("recomputed_area_m2"))
    mismatches = []
    if deriv_payload_sha != nha_sha or candidate_payload_sha != nha_sha:
        mismatches.append("payload_sha256")
    if per_lobe != deriv_lobes or per_lobe != candidate_lobes:
        mismatches.append("per_lobe_geometry_sha256")
    if area != derived_area or candidate_area != area:
        mismatches.append("effective_area_value")
    result = {
        "respiration_config": {"path": str(config_path), "sha256": config_sha},
        "derivation": {"path": str(deriv_path), "sha256": deriv_sha},
        "effective_diaphragm_area_m2": area,
        "derivation_effective_area_m2": derived_area,
        "candidate_recomputed_area_m2": candidate_area,
        "source_payload_sha256": nha_sha,
        "derivation_payload_sha256": deriv_payload_sha,
        "candidate_payload_sha256": candidate_payload_sha,
        "per_lobe_geometry_sha256": per_lobe,
        "derivation_per_lobe_geometry_sha256": deriv_lobes,
        "candidate_per_lobe_geometry_sha256": candidate_lobes,
        "mismatch_fields": mismatches,
        "interpretation": "Effective pressure area is the geometry-derived analytic signed-volume sensitivity; mapped D/lobe surface areas are mesh interface areas and are reported separately.",
        "status": "PASS_exact_geometry_and_config_binding" if not mismatches else "FAIL_geometry_or_payload_binding",
    }
    if mismatches and not allow_mismatch:
        raise ValueError("effective area/config does not bind selected NHA: " + ",".join(mismatches))
    return result


def combine_map_docs(base, docs):
    maps = {}
    source_rows = {}
    for doc in docs:
        base.verify_map_nha(doc, base._RUN_NHA_ROWS)
        for sid, arrays in doc["source_rows"].items():
            if sid in source_rows and any(not base.np.array_equal(source_rows[sid][k], arrays[k]) for k in arrays):
                raise ValueError("map reports disagree on source row arrays")
            source_rows[sid] = arrays
        for sid, pair_map in doc["pairs"].items():
            if sid in maps:
                raise ValueError("duplicate D311/lobe map")
            maps[int(sid)] = pair_map
    if set(maps) != set(base.LOBES):
        raise ValueError("maps must explicitly declare stable IDs 305-309")
    return {"pairs": maps, "source_rows": source_rows}

def mapped_patch_surface_area(base, rows, maps_doc):
    result = {}
    for sid, pair_map in sorted(maps_doc["pairs"].items()):
        dfaces = sorted(int(x) for x in pair_map["d2l"])
        lfaces = sorted(int(x) for x in pair_map["l2d"])
        def area(row, faces):
            v = row["vertices6"][:, :3].astype(base.np.float64, copy=False)
            f = row["faces"]
            total = 0.0
            for fi in faces:
                tri = v[f[fi]]
                total += 0.5 * float(base.np.linalg.norm(base.np.cross(tri[1] - tri[0], tri[2] - tri[0])))
            return total
        da = area(rows[311], dfaces)
        la = area(rows[sid], lfaces)
        if abs(da - la) > max(1e-15, 1e-12 * max(da, la)):
            raise ValueError("reciprocal mapped patch area differs between D and lobe")
        result[str(sid)] = {"mapped_face_count": len(dfaces), "diaphragm_side_surface_area_m2": da,
                            "lobe_side_surface_area_m2": la, "surfaces_equal_within_float64_roundoff": True}
    return result

def prepare_worker_context(cfg):
    base = load_module(require_hash(cfg["base"], BASE_SHA), "native_lung_base_worker")
    if sha(base.PRED) != base.PRED_SHA:
        raise ValueError("exact intersection predicate hash mismatch")
    adapters = load_native_adapters(base)
    owner_ctx = load_owner_context(
        base, cfg["run"], cfg["nha_path"], cfg["nha_sha"],
        cfg["map_reports"], cfg["lobe_lineage_report"], regression_1078=False,
        d_map_composition_report=cfg.get("d_map_composition_report"),
        geometry_only_area_mismatch=cfg.get("geometry_only_area_mismatch", False),
    )
    return base, adapters, owner_ctx

def worker_pose(cfg, step):
    started = time.monotonic()
    try:
        base, adapters, ctx = prepare_worker_context(cfg)
        owner = ctx["owner"]
        pose, pack = base.row_pose(ctx["run_path"], int(step), ctx["nha_rows"])
        maps_doc = ctx["maps_doc"]
        native_d = {sid: base.native_map(sid, maps_doc, pose) for sid in base.LOBES}
        native_l = base.native_lobe_maps(ctx["lineage"], pose)
        topology = None
        topology_error = None
        try:
            topology = owner.topology(ctx["owner_module"], ctx["run_path"], int(step))
        except Exception as exc:
            topology_error = type(exc).__name__ + ": " + str(exc)
        out = Path(cfg["out"])
        self_path = out / ("step-%06d-self.jsonl.gz" % int(step))
        cross_path = out / ("step-%06d-cross.jsonl.gz" % int(step))
        if self_path.exists() or cross_path.exists():
            raise FileExistsError("pose output already exists")
        self_rows = []
        cross_rows = []
        errors = []
        with gzip.open(self_path, "xt", compresslevel=1, encoding="utf-8") as self_sink:
            for sid in base.ROWS:
                try:
                    self_rows.append(base.scan_self(int(step), sid, pose[sid], self_sink))
                except Exception as exc:
                    errors.append({"kind": "self", "owner": sid, "error": type(exc).__name__ + ": " + str(exc)})
        with gzip.open(cross_path, "xt", compresslevel=1, encoding="utf-8") as cross_sink:
            for a, b in itertools.combinations(base.ROWS, 2):
                try:
                    cross_rows.append(base.scan_cross(
                        int(step), a, b, pose[a], pose[b], pose, native_d, native_l, cross_sink
                    ))
                except Exception as exc:
                    errors.append({"kind": "cross", "owners": [a, b], "error": type(exc).__name__ + ": " + str(exc)})
        def map_summary(m):
            return {str(k): {x: y for x, y in v.items() if x not in ("d2l", "l2d", "db", "lb")}
                    for k, v in m.items()}
        coverage = exact_pair_coverage(self_rows, cross_rows, base.ROWS)
        topology_complete = bool(topology and topology.get("all_524_nha_targets_match_current_source") is True)
        result = {
            "accepted_step": int(step),
            "pack": {"path": str(pack), "sha256": sha(pack)},
            "pack_receipt": {"path": str(pack.with_suffix(".receipt.json")), "sha256": sha(pack.with_suffix(".receipt.json"))},
            "topology": topology,
            "topology_error": topology_error,
            "topology_complete": topology_complete,
            "native_D_maps": map_summary(native_d),
            "native_lobe_maps": {
                "%d-%d" % pair: {k: v for k, v in m.items() if k not in ("face_pairs", "shared_vertices", "shared_edges")}
                for pair, m in native_l.items()
            },
            "self": self_rows,
            "cross": cross_rows,
            "unit_errors": errors,
            "self_unallowed": sum(x["unallowed_self_pairs"] for x in self_rows),
            "cross_unclassified": sum(x["unclassified_cross_pairs"] for x in cross_rows),
            "degenerate_faces": sum(len(x["degenerate_face_rows"]) for x in self_rows)
                + sum(sum(len(v) for v in x["degenerate_face_rows"].values()) for x in cross_rows),
            "self_coverage": coverage["self"],
            "cross_coverage": coverage["cross"],
            "expected_self_owners": coverage["expected_self"],
            "expected_cross_pairs": coverage["expected_cross"],
            "witness_files": {
                "self": {"path": str(self_path), "bytes": self_path.stat().st_size, "sha256": sha(self_path)},
                "cross": {"path": str(cross_path), "bytes": cross_path.stat().st_size, "sha256": sha(cross_path)},
            },
            "worker_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "elapsed_s": time.monotonic() - started,
        }
        return {"status": "complete", "result": result}
    except Exception as exc:
        return {"status": "failed", "step": int(step), "error": type(exc).__name__ + ": " + str(exc),
                "elapsed_s": time.monotonic() - started}

def build_context_for_single_regression(cfg):
    base = load_module(require_hash(BASE, BASE_SHA), "native_lung_base_regression")
    adapters = load_native_adapters(base)
    owner = load_module(base.OWNER, "native_geometry_owner_regression")
    owner_module, steps, invocation, metadata = owner.preflight(Path(cfg["run"]).resolve())
    if Path(cfg["run"]).resolve() != STEP0_REGRESSION_RUN.resolve():
        raise ValueError("step-0 regression is pinned to retained1078 fixture")
    if 0 not in steps:
        raise ValueError("pinned 1078 fixture has no accepted step 0")
    wrapper = load_module(require_hash(WRAPPER_1079, WRAPPER_1079_SHA), "native_wrapper_1079")
    context = wrapper.load_context(base)
    nha_path = context["final_path"].resolve()
    if nha_path != Path(owner.option(invocation["argv"], "--torso-anatomy-payload")).resolve():
        raise ValueError("1079 fixture payload differs from accepted run")
    parser = load_module(base.PARSER, "anatomy_parser_regression")
    rows = parser.parse_payload(nha_path)[1]
    # Existing 1079 context returns source-proven full-union map and lobe map.
    return base, adapters, owner, owner_module, invocation, metadata, context, rows, steps

def run_step0_regression(cfg):
    expected_path = require_hash(STEP0_REGRESSION_REPORT, STEP0_REGRESSION_REPORT_SHA)
    expected = json.loads(expected_path.read_text())
    base, adapters, owner, owner_module, invocation, metadata, context, rows, steps = build_context_for_single_regression(cfg)
    out = Path(cfg["out"]).resolve()
    if out.exists() or E not in out.parents:
        raise ValueError("fresh regression output under evidence root required")
    out.mkdir(parents=True)
    pose, pack = base.row_pose(Path(cfg["run"]).resolve(), 0, rows)
    maps_doc = context["map_doc"]
    native_d = {sid: base.native_map(sid, maps_doc, pose) for sid in base.LOBES}
    native_l = base.native_lobe_maps(context["lineage"], pose)
    current_cross = []
    self_rows = []
    with gzip.open(out / "step-000000-self.jsonl.gz", "xt", compresslevel=1, encoding="utf-8") as ss:
        for sid in base.ROWS:
            self_rows.append(base.scan_self(0, sid, pose[sid], ss))
    with gzip.open(out / "step-000000-cross.jsonl.gz", "xt", compresslevel=1, encoding="utf-8") as cs:
        for a, b in itertools.combinations(base.ROWS, 2):
            current_cross.append(base.scan_cross(0, a, b, pose[a], pose[b], pose, native_d, native_l, cs))
    expected_by_pair = {tuple(x["owners"]): x for x in expected["cross"]}
    if set(expected_by_pair) != set(itertools.combinations(base.ROWS, 2)):
        raise ValueError("pinned step-0 comparison is not a complete fifteen-pair report")
    # D/lobe fallback affects D rows only; all ten L-L exact pair predicates/classes stay 1084-identical.
    lobe_regression = []
    for item in current_cross:
        pair = tuple(item["owners"])
        if 311 in pair:
            continue
        old = expected_by_pair.get(pair)
        if old is None or item["aabb_candidates"] != old["aabb_candidates"] or item["exact_hit_pairs"] != old["exact_hit_pairs"] or item["class_counts"] != old["class_counts"]:
            raise ValueError("step-0 L-L regression mismatch for " + str(pair))
        lobe_regression.append({"pair": list(pair), "aabb_candidates": item["aabb_candidates"],
                                "exact_hit_pairs": item["exact_hit_pairs"], "class_counts": item["class_counts"],
                                "unclassified": item["unclassified_cross_pairs"]})
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "schema": "numi.human.native-lung-step0-reader-regression.v1",
        "status": "reader_regression_only",
        "run_path": str(Path(cfg["run"]).resolve()),
        "accepted_step": 0,
        "NHA": {"path": str(context["final_path"]), "sha256": sha(context["final_path"])},
        "expected_reference": {"path": str(expected_path), "sha256": sha(expected_path)},
        "source_adapter": {"path": str(D_ADAPTER), "sha256": sha(D_ADAPTER)},
        "lobe_helper": {"path": str(FEATURE), "sha256": sha(FEATURE)},
        "L_L_pair_regression": lobe_regression,
        "D_lobe_diagnostic_counts": [x for x in current_cross if 311 in x["owners"]],
        "self_summary": self_rows,
        "witness_files": {
            "self": {"path": str(out / "step-000000-self.jsonl.gz"), "bytes": (out / "step-000000-self.jsonl.gz").stat().st_size, "sha256": sha(out / "step-000000-self.jsonl.gz")},
            "cross": {"path": str(out / "step-000000-cross.jsonl.gz"), "bytes": (out / "step-000000-cross.jsonl.gz").stat().st_size, "sha256": sha(out / "step-000000-cross.jsonl.gz")},
        },
        "max_rss_bytes": rss,
        "max_rss_gib": rss / GIB,
        "face_count_reference": sum(len(rows[sid]["faces"]) for sid in base.ROWS),
        "source_adapter": {"base_sha256": sha(BASE), "feature_1084_sha256": sha(FEATURE),
                           "d_adapter_sha256": sha(D_ADAPTER), "source_rule_sha256": sha(SOURCE_RULE)},
        "elapsed_s": time.monotonic() - cfg["started"],
        "qualification": "One-pose reader/classifier/resource regression only; not final-candidate or eight-pose geometry acceptance.",
    }
    write_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(out / "report.json"), "max_rss_gib": report["max_rss_gib"],
                      "unallowed_self": sum(x["unallowed_self_pairs"] for x in self_rows),
                      "cross_unclassified": sum(x["unclassified_cross_pairs"] for x in current_cross)}, sort_keys=True))
    return report

def prepare_final(cfg):
    if cfg["workers"] < 1 or cfg["workers"] > MAX_WORKERS:
        raise ValueError("worker count must be between one and three")
    base = load_module(require_hash(BASE, BASE_SHA), "native_lung_base_parent")
    adapters = load_native_adapters(base)
    ctx = load_owner_context(base, cfg["run"], cfg["nha_path"], cfg["nha_sha"],
                             cfg["map_reports"], cfg["lobe_lineage_report"], regression_1078=False,
                             d_map_composition_report=cfg.get("d_map_composition_report"),
                             geometry_only_area_mismatch=cfg.get("geometry_only_area_mismatch", False))
    # Force construction-time source map validation before any output directory is created.
    areas = mapped_patch_surface_area(base, ctx["nha_rows"], ctx["maps_doc"])
    ctx["area_binding"]["surface_patch_area_m2_by_lobe"] = areas
    if cfg.get("geometry_only_area_mismatch") and ctx["area_binding"]["status"] == "PASS_exact_geometry_and_config_binding":
        raise ValueError("geometry-only area-mismatch mode requires a real failed area-binding comparison")
    if cfg["workers"] > 1:
        if cfg["probe_report"] is None:
            raise ValueError("parallel full scan requires a measured one-pose RSS report")
        probe = json.loads(Path(cfg["probe_report"]).read_text())
        if (probe.get("status") != "reader_regression_only" or probe.get("accepted_step") != 0
                or probe.get("run_path") != str(STEP0_REGRESSION_RUN.resolve())
                or probe.get("NHA", {}).get("sha256") != "7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92"
                or probe.get("source_adapter", {}).get("base_sha256") != BASE_SHA
                or probe.get("source_adapter", {}).get("feature_1084_sha256") != FEATURE_SHA
                or probe.get("source_adapter", {}).get("d_adapter_sha256") != D_ADAPTER_SHA
                or probe.get("source_adapter", {}).get("source_rule_sha256") != SOURCE_RULE_SHA):
            raise ValueError("memory gate needs the pinned 1078 step-0 resource probe with compatible source adapters")
        final_triangles = sum(len(ctx["nha_rows"][sid]["faces"]) for sid in EXPECTED_ROWS)
        memory_plan = validate_memory_probe(
            probe, probe_path=cfg["probe_report"],
            reference_triangle_count=int(probe.get("face_count_reference", 0)),
            worker_count=cfg["workers"], candidate_triangle_count=final_triangles,
        )
    else:
        memory_plan = {"workers": 1, "budget_gib": MEMORY_BUDGET_GIB,
                       "per_worker_estimate_bytes": None,
                       "reason": "single worker is inherently within the configured concurrency ceiling; actual RSS is recorded"}
    out = Path(cfg["out"]).resolve()
    if E not in out.parents or out.exists():
        raise ValueError("fresh output directory under evidence root required")
    before = dict(ctx["input_hashes"])
    if cfg["probe_report"] is not None:
        probe_path = Path(cfg["probe_report"]).resolve()
        if not probe_path.is_file():
            raise ValueError("memory probe report is missing")
        before[str(probe_path)] = sha(probe_path)
        before[str(MEMORY_PROBE_MANIFEST.resolve())] = sha(MEMORY_PROBE_MANIFEST)
    conf = {
        "run": str(ctx["run_path"]), "nha_path": str(ctx["nha_path"]), "nha_sha": ctx["nha_sha"],
        "map_reports": [str(Path(p).resolve()) for p in cfg["map_reports"]],
        "d_map_composition_report": (str(Path(cfg["d_map_composition_report"]).resolve())
                                      if cfg.get("d_map_composition_report") else None),
        "geometry_only_area_mismatch": bool(cfg.get("geometry_only_area_mismatch", False)),
        "lobe_lineage_report": str(Path(cfg["lobe_lineage_report"]).resolve()),
        "base": str(BASE), "feature_adapter": str(FEATURE), "d_adapter": str(D_ADAPTER),
        "lineage_v2_reader": str(LINEAGE_V2),
        "out": str(out), "steps": ctx["steps"],
    }
    out.mkdir(parents=True)
    adapter_identity = {
        "base_auditor": {"path": str(BASE), "sha256": sha(BASE)},
        "native_feature_index_1084": {"path": str(FEATURE), "sha256": sha(FEATURE)},
        "current_row_lobe_lineage_v2": {"path": str(LINEAGE_V2), "sha256": sha(LINEAGE_V2)},
        "v8_current_D_map_adapter": {"path": str(V8_DMAP_ADAPTER), "sha256": sha(V8_DMAP_ADAPTER)},
        "D_lobe_adapter": {"path": str(D_ADAPTER), "sha256": sha(D_ADAPTER),
                           "source_rule_revision": getattr(adapters["d_adapter"], "SOURCE_RULE_REVISION", None),
                           "source_rule": {"path": str(SOURCE_RULE), "sha256": sha(SOURCE_RULE)}},
    }
    write_json(out / "declaration.json", {
        "schema": "numi.human.native-transformed-lung-cycle-audit.declaration.v1",
        "run_path": str(ctx["run_path"]), "accepted_steps": ctx["steps"],
        "requested_roots": ctx["requested_roots"], "dt_seconds": ctx["dt_seconds"],
        "terminal_step_included": ctx["requested_roots"] in ctx["steps"],
        "selected_NHA": {"path": str(ctx["nha_path"]), "sha256": ctx["nha_sha"]},
        "D_lobe_map": ({"composition_report": ctx["d_map_doc"]["path"],
                        "composition_report_sha256": ctx["d_map_doc"]["sha256"],
                        "map": ctx["d_map_doc"]["map_path"],
                        "map_sha256": ctx["d_map_doc"]["map_sha256"],
                        "pairs": ctx["d_map_doc"]["declared_pairs"],
                        "counts_by_lobe": ctx["d_map_doc"]["counts_by_lobe"],
                        "inputs": ctx["d_map_doc"]["inputs"]}
                       if ctx.get("d_map_doc") is not None else None),
        "D_lobe_map_reports": ([{"path": d["path"], "sha256": d["sha256"], "map": d["map_path"], "map_sha256": d["map_sha256"], "pairs": d["declared_pairs"]} for d in ctx["map_docs"]] if ctx.get("d_map_doc") is None else []),
        "lobe_lineage": {"report": ctx["lineage"]["path"], "sha256": ctx["lineage"]["sha256"],
                         "face_map": ctx["lineage"]["map_path"], "face_map_sha256": ctx["lineage"]["map_sha256"],
                         "edge_map": ctx["lineage"]["edge_path"], "edge_map_sha256": ctx["lineage"]["edge_sha256"],
                         "checksum_manifest_sha256": ctx["lineage"]["checksum_manifest_sha256"],
                         "pairs": ctx["lineage"]["declared_pairs"], "source_lineage_validation": ctx["lineage"]["source_simplex_validation"]},
        "loaded_source_adapters": adapter_identity,
        "physiology_area_binding": ctx["area_binding"],
        "mesh_patch_area_by_lobe": areas,
        "rows": list(EXPECTED_ROWS), "self_pairs_per_pose": len(EXPECTED_ROWS),
        "cross_pairs_per_pose": len(EXPECTED_ROWS) * (len(EXPECTED_ROWS) - 1) // 2,
        "gzip_level": 1, "worker_limit": MAX_WORKERS, "selected_workers": cfg["workers"],
        "memory_plan": memory_plan,
        "source_hashes_at_preflight": before,
        "classification_rule": "Exact current reciprocal faces and documented shared-face/edge/vertex adjacency can be classified; all other exact intersections remain retained as unclassified. No distance tolerance.",
        "scope": "Eight discrete accepted native poses; no continuous-time or functional-anatomy qualification.",
        "geometry_only_area_mismatch_mode": bool(cfg.get("geometry_only_area_mismatch", False)),
    })
    work_cfg = conf
    pose_results = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(worker_pose, work_cfg, step): step for step in ctx["steps"]}
        for future in concurrent.futures.as_completed(futures):
            step = futures[future]
            value = future.result()
            if value.get("status") != "complete":
                failures.append(value)
            else:
                pose_results.append(value["result"])
            with (out / "progress.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "worker_status": value.get("status"),
                                    "elapsed_s": value.get("elapsed_s"),
                                    "error": value.get("error")}, sort_keys=True) + "\n")
                f.flush()
    after = {path: sha(path) for path in before}
    unchanged = before == after
    pose_results.sort(key=lambda x: x["accepted_step"])
    expected_steps = [int(s) for s in ctx["steps"]]
    full_coverage = (len(pose_results) == 8 and not failures
                     and [p["accepted_step"] for p in pose_results] == sorted(expected_steps)
                     and all(p["self_coverage"] and p["cross_coverage"] and p["topology_complete"] for p in pose_results))
    degenerate = any(p["degenerate_faces"] for p in pose_results)
    map_bad = any(any(not m.get("valid") for m in p["native_D_maps"].values()) for p in pose_results) or any(
        any(not m.get("valid") for m in p["native_lobe_maps"].values()) for p in pose_results)
    self_bad = sum(p["self_unallowed"] for p in pose_results)
    cross_bad = sum(p["cross_unclassified"] for p in pose_results)
    geometry_ok = full_coverage and unchanged and not (degenerate or map_bad or self_bad or cross_bad)
    area_failed = ctx["area_binding"]["status"] != "PASS_exact_geometry_and_config_binding"
    status = ("complete_geometry_scan_area_binding_failed" if full_coverage and unchanged and area_failed and geometry_ok
              else ("complete_geometry_scan_with_geometry_failures" if full_coverage and unchanged and area_failed
                    else ("complete_scan_with_failures" if full_coverage and (degenerate or map_bad or self_bad or cross_bad or not unchanged)
                          else ("complete_scan_no_unclassified_hits" if full_coverage and unchanged else "incomplete_scan"))))
    report = {
        "schema": "numi.human.native-transformed-lung-cycle-audit.summary.v1",
        "status": status,
        "geometry_scan_status": "PASS_exact_discrete_predicates" if geometry_ok else ("FAIL_or_incomplete" if full_coverage else "INCOMPLETE"),
        "physiology_integration_status": "FAIL_area_binding_mismatch" if area_failed else "PASS_area_binding",
        "run_path": str(ctx["run_path"]), "accepted_steps": ctx["steps"],
        "requested_roots": ctx["requested_roots"], "dt_seconds": ctx["dt_seconds"],
        "terminal_step_included": ctx["requested_roots"] in ctx["steps"],
        "nha": {"path": str(ctx["nha_path"]), "sha256": ctx["nha_sha"]},
        "input_hashes_unchanged": unchanged, "input_hashes_before": before, "input_hashes_after": after,
        "pose_results": pose_results, "worker_failures": failures,
        "complete_pair_coverage": full_coverage, "topology_verified_each_pose": full_coverage, "degenerate_face_seen": degenerate,
        "native_map_failure": map_bad, "unallowed_self_pair_total": self_bad,
        "unclassified_cross_pair_total": cross_bad,
        "all_6_self_and_15_cross_per_pose": full_coverage and all(
            len(p["self"]) == 6 and len(p["cross"]) == 15 for p in pose_results),
        "raw_witnesses_gzip_level1": True, "memory_plan": memory_plan,
        "qualification": ("Geometry-only exact accepted-pose scan; selected run respiration derivation does not bind the current NHA, so this is not integrated anatomy/physiology acceptance. No automatic waiver of contacts or continuous-time qualification." if area_failed else "Discrete accepted-pose exact scan only; no automatic waiver of contacts and no continuous-time/functional-anatomy acceptance."),
    }
    write_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "out": str(out), "poses": len(pose_results),
                      "complete_pair_coverage": full_coverage, "self_unallowed": self_bad,
                      "cross_unclassified": cross_bad, "input_hashes_unchanged": unchanged}, sort_keys=True))
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--nha", type=Path)
    ap.add_argument("--nha-sha256")
    ap.add_argument("--map-report", action="append", type=Path, default=[])
    ap.add_argument("--d-map-composition-report", type=Path)
    ap.add_argument("--geometry-only-area-mismatch", action="store_true",
                    help="continue exact geometry census while recording a failed respiration-area binding; result is not integrated acceptance")
    ap.add_argument("--lobe-lineage-report", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--probe-report", type=Path)
    ap.add_argument("--step0-regression", action="store_true")
    args = ap.parse_args()
    if sha(BASE) != BASE_SHA:
        raise ValueError("pinned base auditor changed")
    started = time.monotonic()
    if args.step0_regression:
        if args.workers != 1 or args.nha or args.nha_sha256 or args.map_report or args.lobe_lineage_report or args.d_map_composition_report or args.geometry_only_area_mismatch:
            raise ValueError("step-0 fixture mode uses its pinned 1078 owner/source adapters only")
        cfg = {"run": args.run, "out": args.out, "started": started}
        run_step0_regression(cfg)
        return 0
    dmap_mode = args.d_map_composition_report is not None
    if (not args.nha or not args.nha_sha256 or not args.lobe_lineage_report
            or (dmap_mode and args.map_report) or (not dmap_mode and len(args.map_report) != 5)):
        ap.error("full mode requires --nha/--nha-sha256, --lobe-lineage-report, and either --d-map-composition-report or five --map-report arguments")
    if args.workers > 1 and args.probe_report is None:
        ap.error("parallel mode requires --probe-report from the one-pose RSS regression")
    if args.geometry_only_area_mismatch and (not dmap_mode or not args.lobe_lineage_report):
        ap.error("geometry-only mismatch continuation requires the current v8 D-map mode and full lineage")
    cfg = {"run": args.run, "out": args.out, "nha_path": args.nha, "nha_sha": args.nha_sha256,
           "map_reports": args.map_report, "d_map_composition_report": args.d_map_composition_report,
           "lobe_lineage_report": args.lobe_lineage_report,
           "geometry_only_area_mismatch": args.geometry_only_area_mismatch,
           "workers": args.workers, "probe_report": args.probe_report}
    if args.workers == 1:
        # A single worker is useful for selected candidates when no old-candidate
        # memory probe is transferable; the scan itself still requires all eight poses.
        cfg["probe_report"] = None
    prepare_final(cfg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

