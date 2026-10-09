#!/usr/bin/env python3
"""Exact full-skin target/self audit for eight sealed native poses in a registered study arm.

The registered receipt adapter is the authority for trial success, runtime, accepted steps,
and source asset bindings. No scene run-metadata file is synthesized or required.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import mmap
import resource
import struct
import sys
import time
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
HERE = E / "native-lung-late-skin-audit-runner-1172"
REGISTERED_ADAPTER = E / "native-lung-late-pose-audit-runner-1171/registered_receipt_adapter_1171.py"
REGISTERED_ADAPTER_SHA = "ba0b88416dbf9cab892f9daa6f954e931003f9db5075db6b758c209370efc1e8"
AUDIT_CORE = E / "native-common-skin-combined-cycle-audit-908/audit_cycle.py"
AUDIT_CORE_SHA = "2eba147ca37ea80a3ed12c8dd725986d77bcd60c194077fae6ddfc5961d9dcde"
SKIN_DIR = E / "native-common-skin-multipose-clearance-candidate-927/asset-candidate-001"
SKIN = SKIN_DIR / "bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_SHA = "bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1"
SKIN_REPORT = SKIN_DIR / "candidate-asset-composition-report.json"
SKIN_REPORT_SHA = "9f1cf73718ddc1e3e922e1097cf485c74ea87f394da3000dc4cd19e9f34990dd"
SKIN_REGISTRATION = SKIN_DIR / "common-atlas-skin-geometry-registration.manifest.json"
SKIN_REGISTRATION_SHA = "120b78369a8b74a4719eccc9ef416a44086574dc5ddcddbc55b721e2d06ccf99"
SKIN_MODULE = Path("/Users/n/numi-human-lung-source-publication-902/src/numilab_human/common_atlas_skin_geometry_registration.py")
SKIN_MODULE_SHA = "a0bd2271cdad2e13472b1770fab33469589bba1193a579e321de20f89752eb52"
INV = E / "native-complete-skin-containment-audit-890/pair-summary-v3.csv"
INV_SHA = "a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
HELPER = E / "cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py"
HELPER_SHA = "eb9e5c762cdbab1e9b3f98c63a1580a21acba54637b105c28e2bc0a39187ba17"
ROOT = Path("/Users/n/numi-human-resting-final-integration-001")
PRED = ROOT / "src/numilab_human/cardiac_cavity_intersections.py"
PRED_SHA = "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb"
CLEARANCE = ROOT / "src/numilab_human/common_atlas_skin_clearance.py"
CLEARANCE_SHA = "1321c31c22e1b1dbf0062947c6767c2cfcae9aae35d29e9ea775862feed5ddd1"
LATTICE_DENOMINATOR = 1 << 149
EXPECTED_SKIN_KEY = (51007, 1)
OCULAR = {(51010, stable_id) for stable_id in range(381, 398)}
EXPECTED_TARGETS = 859
EXPECTED_STEPS = 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_hash(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            "pinned input mismatch: " + str(path))
    return path


def validate_steps(steps, terminal):
    values = [int(x) for x in steps]
    terminal = int(terminal)
    require(len(values) == EXPECTED_STEPS and len(set(values)) == EXPECTED_STEPS,
            "registered capture schedule must contain exactly eight unique steps")
    require(values[-1] == terminal and terminal == 155000,
            "registered terminal must be the final accepted step 155000")
    return values


def validate_skin_asset():
    require_hash(SKIN, SKIN_SHA)
    require_hash(SKIN_REPORT, SKIN_REPORT_SHA)
    require_hash(SKIN_REGISTRATION, SKIN_REGISTRATION_SHA)
    report = json.loads(SKIN_REPORT.read_text())
    registration = json.loads(SKIN_REGISTRATION.read_text())
    payload = registration.get("output_payload", {})
    report_payload = report.get("candidate_payload", {})
    require(payload.get("path") == str(SKIN) and payload.get("sha256") == SKIN_SHA,
            "927 registration does not bind the exact skin payload")
    require(report_payload.get("path") == str(SKIN) and report_payload.get("sha256") == SKIN_SHA,
            "927 composition report does not bind the exact skin payload")
    require(report.get("candidate_registration_manifest", {}).get("sha256") == SKIN_REGISTRATION_SHA,
            "927 report does not bind its exact registration manifest")
    composition = report.get("candidate_receipt_composition", {})
    require(composition.get("candidate_skin_sha256") == SKIN_SHA
            and composition.get("candidate_manifest_sha256") == SKIN_REGISTRATION_SHA,
            "927 candidate receipt composition is not bound to the payload and manifest")
    module = registration.get("code", {}).get("module")
    module_sha = registration.get("code", {}).get("module_sha256")
    require(module == str(SKIN_MODULE) and module_sha == SKIN_MODULE_SHA,
            "927 registration source module identity changed")
    require_hash(SKIN_MODULE, sha(SKIN_MODULE))
    require_hash(INV, INV_SHA)
    core = load_module(require_hash(AUDIT_CORE, AUDIT_CORE_SHA), "skin_audit_core_908_for_registration")
    historical_commit = core.composition_module_history_commit(SKIN_MODULE_SHA)
    return {"payload_sha256": SKIN_SHA, "report_sha256": SKIN_REPORT_SHA,
            "registration_sha256": SKIN_REGISTRATION_SHA,
            "historical_composition_module_sha256": SKIN_MODULE_SHA,
            "composition_module_current_sha256": sha(SKIN_MODULE),
            "historical_composition_module_git_commit": historical_commit}


def validate_invocation(context, expected_nha, expected_nha_sha):
    invocation = context["invocation"]
    argv = invocation.get("argv")
    assets = invocation.get("asset_sha256")
    require(isinstance(argv, list) and isinstance(assets, dict),
            "registered native invocation lacks argv or asset hashes")
    try:
        skin_path = Path(argv[argv.index("--skin-payload") + 1]).resolve()
        nha_path = Path(argv[argv.index("--torso-anatomy-payload") + 1]).resolve()
        receipt_path = Path(argv[argv.index("--resting-anatomy-receipt") + 1]).resolve()
    except (ValueError, IndexError):
        raise ValueError("registered invocation is missing skin/NHA/receipt arguments")
    require(skin_path == SKIN.resolve() and assets.get(str(skin_path)) == SKIN_SHA,
            "registered arm does not bind the exact 927 NHSKIN payload")
    nha_path = Path(expected_nha).resolve()
    require(Path(argv[argv.index("--torso-anatomy-payload") + 1]).resolve() == nha_path,
            "registered arm NHA path differs from requested current payload")
    require(sha(nha_path) == expected_nha_sha and assets.get(str(nha_path)) == expected_nha_sha,
            "registered arm does not bind the requested exact NHA payload")
    require(receipt_path.is_file() and assets.get(str(receipt_path)) == sha(receipt_path),
            "registered arm's anatomy receipt is missing or not asset-bound")
    dt = argv[argv.index("--muscle-step-seconds") + 1]
    roots = int(argv[argv.index("--muscle-step-count") + 1])
    require(float(dt) == 0.002 and roots == 155000,
            "registered native invocation is not the 2 ms, 155000-root physical run")
    return {"invocation": invocation, "skin_path": skin_path, "nha_path": nha_path,
            "nha_sha256": expected_nha_sha, "receipt_path": receipt_path}


def load_predicates():
    require_hash(AUDIT_CORE, AUDIT_CORE_SHA)
    require_hash(HELPER, HELPER_SHA)
    require_hash(PRED, PRED_SHA)
    require_hash(CLEARANCE, CLEARANCE_SHA)
    sys.path.insert(0, str(ROOT / "src"))
    from numilab_human import cardiac_cavity_intersections as ci
    from numilab_human import common_atlas_skin_clearance as clearance
    return ci, clearance, load_module(HELPER, "accepted_mrvpack_audit_1172"), load_module(AUDIT_CORE, "skin_audit_core_908")


def inventory():
    with INV.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(row["second_semantic"]), int(row["second_stable_id"])) for row in rows]
    require(len(keys) == EXPECTED_TARGETS and len(set(keys)) == EXPECTED_TARGETS,
            "pinned 890 inventory must contain exactly 859 unique targets")
    require(OCULAR.issubset(set(keys)), "pinned target inventory omits an ocular monitor")
    return keys, {(int(row["second_semantic"]), int(row["second_stable_id"])): row for row in rows}


def verify_pack(step, pack, receipt_path, validator, nha_sha):
    doc = json.loads(receipt_path.read_text())
    with pack.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            vertex_offset = next(section[2] for section in validator._pack_sections(mapped) if section[0] == 2)
            accepted = validator.validate_accepted_receipt(pack, receipt_path, step, mapped, vertex_offset, {})
    require(accepted.get("accepted_step") == step and doc.get("accepted_pack_path") == str(pack)
            and doc.get("physical_endpoint") == "accepted" and doc.get("surface_audit_endpoint") == "passed",
            "accepted-geometry receipt does not prove the requested accepted step")
    return accepted


def run_pose(step, scene, out, ci, clearance, validator, core, keys, expected_by_key, expected_nha_sha):
    start = time.monotonic()
    pack = scene / "accepted-geometry" / ("step-%d.mrvpack" % step)
    receipt_path = pack.with_suffix(".receipt.json")
    accepted = verify_pack(step, pack, receipt_path, validator, expected_nha_sha)
    positions, surfaces, pack_counts = clearance._pack_surfaces(pack, set(keys))
    require(set(surfaces) == set(keys), "captured pack does not contain every pinned target surface")
    skin_faces = surfaces[EXPECTED_SKIN_KEY]["faces"]
    raw = SKIN.read_bytes()
    magic, abi, binding_count, source_vertex_count, source_index_count, registration_fp, archive_sha = struct.unpack_from("<8s5I32s", raw)
    require(magic == b"NHSKIN1\0" and abi == 5, "registered skin payload ABI is not NHSKIN 5")
    source_offset = 60 + 36 * binding_count + 56 * source_vertex_count
    source_faces = __import__("numpy").frombuffer(raw, dtype="<u4", count=source_index_count, offset=source_offset).reshape(-1, 3)
    skin_base = int(skin_faces.min())
    require(len(skin_faces) == len(source_faces) and int(skin_faces.max()) < skin_base + source_vertex_count
            and __import__("numpy").array_equal(skin_faces - skin_base, source_faces),
            "captured skin triangle rows/topology differ from exact NHSKIN source")
    del source_faces, raw
    np = __import__("numpy")
    skin_ids = np.unique(skin_faces)
    skin_local_faces = np.searchsorted(skin_ids, skin_faces)
    skin_records, skin_record_rows, skin_degenerate = core.exact_records(positions[skin_ids], skin_local_faces, ci)
    targets_path = out / ("step-%d.targets.jsonl" % step)
    cross_path = out / ("step-%d.crossing-witnesses.jsonl" % step)
    self_path = out / ("step-%d.self-witnesses.jsonl" % step)
    invalid_path = out / ("step-%d.invalid-triangles.jsonl" % step)
    totals = {"all_pairs": 0, "ocular_pairs": 0, "nonocular_pairs": 0,
              "target_count": 0, "invalid_target_surfaces": 0, "invalid_target_triangles": 0}
    hit_rows = []
    with targets_path.open("x") as target_stream, cross_path.open("x") as cross_stream, self_path.open("x") as self_stream, invalid_path.open("x") as invalid_stream:
        for row in skin_degenerate:
            ids = [int(v) for v in skin_faces[row]]
            invalid_stream.write(json.dumps({"role": "skin_degenerate_face", "skin_source_face_row": int(row),
                "pack_vertex_ids": ids, "triangle_xyz_f32_m": positions[ids].astype(float).tolist()},
                separators=(",", ":"), allow_nan=False) + "\n")
        for ordinal, key in enumerate(keys, 1):
            target_faces = surfaces[key]["faces"]
            expected = expected_by_key[key]
            target_ids = np.unique(target_faces)
            target_local = np.searchsorted(target_ids, target_faces)
            target_records, target_record_rows, target_degenerate = core.exact_records(positions[target_ids], target_local, ci)
            audit = ci._audit_pair(skin_records, target_records, same_surface=False)
            count = int(audit["count"])
            totals["target_count"] += 1
            totals["all_pairs"] += count
            if key in OCULAR:
                totals["ocular_pairs"] += count
            else:
                totals["nonocular_pairs"] += count
            if target_degenerate:
                totals["invalid_target_surfaces"] += 1
                totals["invalid_target_triangles"] += len(target_degenerate)
                for face_row in target_degenerate:
                    ids = [int(v) for v in target_faces[face_row]]
                    invalid_stream.write(json.dumps({"role": "target_degenerate_face", "target_surface": list(key),
                        "target_surface_face_row": int(face_row), "pack_vertex_ids": ids,
                        "triangle_xyz_f32_m": positions[ids].astype(float).tolist()},
                        separators=(",", ":"), allow_nan=False) + "\n")
            target_stream.write(json.dumps({
                "surface": [key[0], key[1]], "source_owner_or_label": expected["source_owner_or_label"],
                "face_count": int(len(target_faces)), "pinned_890_face_count": int(expected["face_count_other"]),
                "face_count_delta_from_890": int(len(target_faces)) - int(expected["face_count_other"]),
                "aabb_candidate_pairs": int(audit["aabb_candidate_pairs"]),
                "intersecting_triangle_pairs": count,
                "degenerate_face_rows": [int(x) for x in target_degenerate],
                "pair_coverage_complete": not bool(target_degenerate or skin_degenerate),
                "ocular_monitor": key in OCULAR}, separators=(",", ":"), allow_nan=False) + "\n")
            if count:
                hit_rows.append({"surface": list(key), "pairs": count})
                for skin_i, target_i in audit["triangle_pairs"]:
                    core.crossing_witness(cross_stream, semantic=key[0], stable_id=key[1],
                        skin_pair_index=skin_i, target_pair_index=target_i,
                        skin_record=skin_records[skin_i], target_record=target_records[target_i],
                        skin_row=skin_record_rows[skin_i], target_row=target_record_rows[target_i],
                        skin_faces=skin_faces, target_faces=target_faces, positions=positions, ci=ci)
            if ordinal % 100 == 0:
                print(json.dumps({"step": step, "completed_targets": ordinal,
                    "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        self_audit = ci._audit_pair(skin_records, skin_records, same_surface=True)
        for skin_i, other_i in self_audit["triangle_pairs"]:
            core.crossing_witness(self_stream, semantic=EXPECTED_SKIN_KEY[0], stable_id=EXPECTED_SKIN_KEY[1],
                skin_pair_index=skin_i, target_pair_index=other_i,
                skin_record=skin_records[skin_i], target_record=skin_records[other_i],
                skin_row=skin_record_rows[skin_i], target_row=skin_record_rows[other_i],
                skin_faces=skin_faces, target_faces=skin_faces, positions=positions, ci=ci,
                role="skin_self_intersection")
    complete = totals["target_count"] == EXPECTED_TARGETS and not skin_degenerate and totals["invalid_target_triangles"] == 0
    result = {"status": "complete_pair_coverage" if complete else "incomplete_degenerate_input_fail_closed",
        "accepted_step": step, "accepted_receipt": accepted, "scene_path": str(scene),
        "NHA_sha256": expected_nha_sha, "skin_sha256": SKIN_SHA, "surface_target_count": totals["target_count"],
        "all_skin_crossing_pair_count": totals["all_pairs"], "nonocular_crossing_pair_count": totals["nonocular_pairs"],
        "ocular_crossing_pair_count": totals["ocular_pairs"], "ocular_target_count": len(OCULAR),
        "skin_self_crossing_pair_count": int(self_audit["count"]),
        "skin_degenerate_face_rows": [int(x) for x in skin_degenerate],
        "invalid_target_surface_count": totals["invalid_target_surfaces"],
        "invalid_target_triangle_count": totals["invalid_target_triangles"],
        "pair_coverage_complete": complete,
        "targets_sha256": sha(targets_path), "crossing_witnesses_sha256": sha(cross_path),
        "self_witnesses_sha256": sha(self_path), "invalid_triangles_sha256": sha(invalid_path),
        "hit_surfaces": hit_rows, "elapsed_s": time.monotonic() - start,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "predicate": "exact_float32_lattice_triangle_intersection", "contact_exemptions": []}
    writej(out / ("step-%d.result.json" % step), result)
    print(json.dumps({k: result[k] for k in ("accepted_step", "surface_target_count", "all_skin_crossing_pair_count",
        "nonocular_crossing_pair_count", "ocular_crossing_pair_count", "skin_self_crossing_pair_count",
        "invalid_target_surface_count", "invalid_target_triangle_count", "pair_coverage_complete", "elapsed_s", "rss_peak_bytes")}), flush=True)
    return result


def preflight(args):
    adapter = load_module(require_hash(REGISTERED_ADAPTER, REGISTERED_ADAPTER_SHA), "registered_receipt_adapter_1171")
    context = adapter.load_registered_context(args.trial, args.arm)
    steps = validate_steps(context["steps"], context["requested_roots"])
    skin_identity = validate_skin_asset()
    inv = validate_invocation(context, args.nha, args.nha_sha256)
    keys, expected = inventory()
    ci, clearance, validator, core = load_predicates()
    scene = Path(context["scene_path"]).resolve()
    require(scene.is_dir() and not scene.is_symlink(), "registered native scene is missing or symlinked")
    for step in steps:
        pack = scene / "accepted-geometry" / ("step-%d.mrvpack" % step)
        receipt = pack.with_suffix(".receipt.json")
        require(pack.is_file() and receipt.is_file(), "registered accepted geometry pack/receipt is missing: %s" % step)
    tracked = dict(context["input_hashes"])
    for path in (REGISTERED_ADAPTER, AUDIT_CORE, SKIN, SKIN_REPORT, SKIN_REGISTRATION, SKIN_MODULE, INV, HELPER, PRED, CLEARANCE, Path(__file__).resolve()):
        tracked[str(path.resolve())] = sha(path)
    return {"adapter": adapter, "context": context, "steps": steps, "scene": scene,
        "skin_identity": skin_identity, "invocation": inv, "keys": keys, "expected": expected,
        "ci": ci, "clearance": clearance, "validator": validator, "core": core, "tracked": tracked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True, help="registered science-v2 arm trial directory")
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--nha", type=Path, required=True)
    parser.add_argument("--nha-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory under the evidence root")
    parser.add_argument("--validate-only", action="store_true", help="validate registered inputs and hashes without creating output")
    args = parser.parse_args()
    expected_nha_sha = args.nha_sha256.lower()
    require(len(expected_nha_sha) == 64 and all(c in "0123456789abcdef" for c in expected_nha_sha), "NHA SHA must be 64 lowercase hex characters")
    ctx = preflight(args)
    out = args.out.expanduser().resolve()
    require(E in out.parents and not out.exists(), "output must be a fresh directory under the evidence root")
    if args.validate_only:
        print(json.dumps({"status": "registered_skin_audit_inputs_validated_no_scan", "arm": args.arm,
            "trial": str(ctx["context"]["trial_path"]), "scene": str(ctx["scene"]),
            "accepted_steps": ctx["steps"], "terminal": ctx["context"]["requested_roots"],
            "NHA": {"path": str(ctx["invocation"]["nha_path"]), "sha256": expected_nha_sha},
            "skin_sha256": SKIN_SHA, "target_count": len(ctx["keys"]),
            "registered_receipt_chain": ctx["context"]["receipt_chain"]}, sort_keys=True))
        return 0
    before = dict(ctx["tracked"])
    out.mkdir(parents=True)
    writej(out / "declaration.json", {
        "schema": "numi.human.registered-native-full-skin-cycle-audit.declaration.v1",
        "registered_arm": args.arm, "registered_trial_path": str(ctx["context"]["trial_path"]),
        "scene_path": str(ctx["scene"]), "accepted_steps": ctx["steps"],
        "requested_roots": ctx["context"]["requested_roots"], "dt_seconds": ctx["context"]["dt_seconds"],
        "registered_receipt_chain": ctx["context"]["receipt_chain"],
        "NHA": {"path": str(ctx["invocation"]["nha_path"]), "sha256": expected_nha_sha},
        "NHSKIN": {"path": str(SKIN), "sha256": SKIN_SHA, "report_sha256": SKIN_REPORT_SHA,
                   "registration_manifest_sha256": SKIN_REGISTRATION_SHA},
        "target_inventory": {"path": str(INV), "sha256": INV_SHA, "surface_count": EXPECTED_TARGETS,
                             "ocular_monitor_surfaces": [[51010, x] for x in range(381, 398)]},
        "loaded_exact_predicate": {"path": str(PRED), "sha256": PRED_SHA,
                                    "audit_core": str(AUDIT_CORE), "audit_core_sha256": AUDIT_CORE_SHA},
        "source_hashes_at_preflight": before,
        "scope": "Eight discrete accepted native poses; exact captured Float32-lattice skin-target and skin-self predicates; no distance tolerance or contact exemptions; no continuous-time or clinical claim.",
        "scene_run_metadata_required": False})
    results = []
    for step in ctx["steps"]:
        results.append(run_pose(step, ctx["scene"], out, ctx["ci"], ctx["clearance"], ctx["validator"],
                                ctx["core"], ctx["keys"], ctx["expected"], expected_nha_sha))
        gc.collect()
        with (out / "progress.jsonl").open("a") as progress:
            progress.write(json.dumps({"step": step, "status": results[-1]["status"],
                "elapsed_s": results[-1]["elapsed_s"], "rss_peak_bytes": results[-1]["rss_peak_bytes"]}, sort_keys=True) + "\n")
            progress.flush()
    after = {path: sha(path) for path in before}
    unchanged = before == after
    complete = len(results) == EXPECTED_STEPS and [r["accepted_step"] for r in results] == ctx["steps"] and all(r["pair_coverage_complete"] for r in results)
    intersection_free = complete and all(r["all_skin_crossing_pair_count"] == 0 and r["skin_self_crossing_pair_count"] == 0 for r in results)
    summary = {"schema": "numi.human.registered-native-full-skin-cycle-audit.summary.v1",
        "status": "complete_exact_intersection_free" if intersection_free and unchanged else ("complete_with_intersections_or_invalid_geometry" if complete and unchanged else "incomplete_or_input_changed"),
        "registered_arm": args.arm, "trial_path": str(ctx["context"]["trial_path"]),
        "scene_path": str(ctx["scene"]), "accepted_steps": ctx["steps"],
        "requested_roots": ctx["context"]["requested_roots"], "dt_seconds": ctx["context"]["dt_seconds"],
        "NHA": {"path": str(ctx["invocation"]["nha_path"]), "sha256": expected_nha_sha},
        "NHSKIN_sha256": SKIN_SHA, "target_surface_count": EXPECTED_TARGETS,
        "all_eight_steps_complete": complete, "all_target_and_skin_self_pair_coverage_complete": complete,
        "intersection_free": intersection_free, "input_hashes_unchanged": unchanged,
        "input_hashes_before": before, "input_hashes_after": after, "pose_results": results,
        "qualification": "Exact geometry evidence for eight accepted frames only; intersections are retained without waivers and any degenerate input fails coverage. No continuous-time or physiological qualification."}
    writej(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "out": str(out), "accepted_steps": ctx["steps"],
        "complete": complete, "intersection_free": intersection_free, "input_hashes_unchanged": unchanged}, sort_keys=True))
    return 0 if complete and unchanged else 2

if __name__ == "__main__":
    raise SystemExit(main())
