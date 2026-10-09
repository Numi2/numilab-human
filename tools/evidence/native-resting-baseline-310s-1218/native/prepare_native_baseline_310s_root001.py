#!/usr/bin/env python3
"""Prepare the 310 s skin-004/006 native declaration after exact gates pass.

This is preparation only. It never invokes the native runner. By default it
writes a gate report and does not write run-declaration.json. Declaration
creation requires both completed source gates and an explicit root-review flag.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

R = Path("/Users/n/numi-human-retained-delivery-20261009")
PKG = R / "skin-resting-multipose-clearance-1218"
PREP = PKG / "native-baseline-310s-preparation"
PLAN = PREP / "preparation-plan.json"
PLAN_SHA_FILE = PREP / "preparation-plan.sha256"
PARENT_DECL = R / "muscle-conforming-refinement-1216/native-refinement-1217-attempt2/run-declaration.json"
PARENT_EXEC = PARENT_DECL.parent / "execution.json"
PARENT_NATIVE = PARENT_DECL.parent / "native-run/run-metadata.json"
PARENT_LAUNCHER = PARENT_DECL.parent / "run.py"
LAUNCHER = PREP / "run.py"
COMPOSER = PKG / "package-preparation-002/compose_candidate_004_root003b.py"
AUDIT_DIR = PKG / "local-self-reduction-audit-001/scan-001"
AUDIT_SUMMARY = AUDIT_DIR / "summary.json"
AUDIT_DECL = AUDIT_DIR / "declaration.json"
AUDIT_SCRIPT = PKG / "local-self-reduction-audit-001/audit_local_reduction_1218_root003.py"
DEFAULT_COMPOSITION_REVIEW = PKG / "package-preparation-002/composed-candidate/composition-review.json"
RUN_DECL = PREP / "run-declaration.json"
RUN_DIR = PREP / "native-run"
EXPECTED_PARENT_DECL_SHA = "a89a7fdfef2751b00db90d1b8f0332176ef4aea4981bcaa936b10ad71531c12b"
EXPECTED_PARENT_EXEC_SHA = "4c5665166e3084546863b6aa6c48902f483c34d40a9f5e3982d3b0c25961ea69"
EXPECTED_LAUNCHER_SHA = "fd11ec624fad091ad77f8a8c95e0467ef989cd215a01fe6220fb9158692b933e"
EXPECTED_CADENCE_SHA = "a68d865d487ca19840322271db885aa2e693ebf545d008c820707cb90d396e13"
CADENCE_HEADER = Path("/Users/n/numi-human-retired-alias-visibility-018/apps/NumiHumanAcceptedGeometryCadence.hpp")
CAPTURES = [0, 9983, 19999, 47519, 152191, 153183, 154143, 155000]
DURATION_S = 310
DT_S = 0.002
INSPECTION_S = 8.0
TARGETS = 859


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    p = Path(path).resolve(strict=True)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"expected regular non-symlink file: {p}")
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def load_json(path: Path) -> dict:
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise FileNotFoundError(str(p))
    v = json.loads(p.read_text())
    if not isinstance(v, dict):
        raise ValueError(f"expected JSON object at {p}")
    return v


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def flag_value(argv: list[str], flag: str) -> str:
    hits = [i for i, x in enumerate(argv) if x == flag]
    require(len(hits) == 1 and hits[0] + 1 < len(argv), f"expected one {flag} value")
    return argv[hits[0] + 1]


def set_flag(argv: list[str], flag: str, value: str) -> dict:
    old = flag_value(argv, flag)
    i = argv.index(flag) + 1
    argv[i] = value
    return {"setting": flag, "old": old, "new": value, "argv_index": i}


def env_index(argv: list[str], key: str) -> int:
    prefix = key + "="
    hits = [i for i, x in enumerate(argv) if x.startswith(prefix)]
    require(len(hits) == 1, f"expected exactly one environment setting {key}")
    return hits[0]


def set_env(argv: list[str], key: str, value: str) -> dict:
    i = env_index(argv, key)
    old = argv[i].split("=", 1)[1]
    argv[i] = key + "=" + value
    return {"setting": key, "old": old, "new": value, "argv_index": i}


def check_plan(plan: dict) -> dict:
    require(plan.get("schema") == "numi.human.resting.baseline-preparation.1218.v1", "preparation plan schema changed")
    require(sha(PLAN) == PLAN_SHA_FILE.read_text().split()[0], "preparation plan hash does not match retained sha256 file")
    require(plan.get("parent_declaration_path") == str(PARENT_DECL), "parent declaration path differs")
    require(plan.get("parent_declaration_sha256") == EXPECTED_PARENT_DECL_SHA, "parent declaration pin differs")
    require(plan.get("parent_execution_sha256") == EXPECTED_PARENT_EXEC_SHA, "parent execution pin differs")
    require(plan.get("prepared_launcher_path") == str(LAUNCHER), "prepared launcher path differs")
    require(plan.get("prepared_launcher_sha256") == EXPECTED_LAUNCHER_SHA, "prepared launcher pin differs")
    require(plan.get("cadence_header_path") == str(CADENCE_HEADER), "cadence header path differs")
    require(plan.get("cadence_header_sha256") == EXPECTED_CADENCE_SHA, "cadence header pin differs")
    require(plan.get("duration_s") == DURATION_S and plan.get("dt_s") == DT_S, "requested horizon/dt changed")
    require(plan.get("total_accepted_steps") == 155000, "requested accepted step count changed")
    require(plan.get("inspection_period_seconds") == INSPECTION_S, "inspection period changed")
    require(plan.get("capture_count") == len(CAPTURES), "capture count differs")
    require([x["step"] for x in plan.get("capture_steps", [])] == CAPTURES, "capture schedule differs")
    return pin(PLAN)


def validate_parent(plan: dict, check_assets: bool) -> tuple[dict, dict, list[str]]:
    errors = []
    parent = load_json(PARENT_DECL)
    execution = load_json(PARENT_EXEC)
    if sha(PARENT_DECL) != EXPECTED_PARENT_DECL_SHA:
        errors.append("1217 parent declaration hash changed")
    if sha(PARENT_EXEC) != EXPECTED_PARENT_EXEC_SHA:
        errors.append("1217 parent execution hash changed")
    if sha(PARENT_LAUNCHER) != EXPECTED_LAUNCHER_SHA or sha(LAUNCHER) != EXPECTED_LAUNCHER_SHA:
        errors.append("parent/preparation launcher hash changed")
    if not PARENT_NATIVE.is_file():
        errors.append("1217 native run metadata missing")
    else:
        metadata = load_json(PARENT_NATIVE)
        # The plan predates the completed native receipt and has no metadata
        # digest field. Validate it against the retained declaration/execution
        # and current assets, then bind its observed hash in the new output.
        if metadata.get("exit_code") != 0:
            errors.append("1217 native receipt exit_code is not zero")
        if metadata.get("source_files_changed_during_run") != []:
            errors.append("1217 native receipt reports changed source files")
        loaded = metadata.get("loaded_metal_runtime", {})
        if not isinstance(loaded, dict) or loaded.get("verified") is not True:
            errors.append("1217 loaded runtime verification is not true")
        else:
            runtime_path = loaded.get("expected_path")
            runtime_sha = loaded.get("expected_sha256")
            images = loaded.get("observed_images", [])
            if not runtime_path or not runtime_sha:
                errors.append("1217 loaded runtime receipt lacks path/hash")
            elif not Path(runtime_path).is_file() or sha(Path(runtime_path)) != runtime_sha:
                errors.append("1217 loaded runtime file no longer matches receipt hash")
            if not any(isinstance(x, dict) and x.get("path") == runtime_path for x in images):
                errors.append("1217 observed loaded image does not match expected runtime path")
        assets = parent.get("immutable_assets", {})
        observed_assets = metadata.get("asset_sha256", {})
        if not isinstance(assets, dict) or not isinstance(observed_assets, dict) or not observed_assets:
            errors.append("1217 native receipt has no valid asset hash map")
        else:
            for raw, digest in observed_assets.items():
                if assets.get(raw) != digest:
                    errors.append(f"1217 native receipt asset is not bound by parent declaration: {raw}")
                    continue
                path = Path(raw)
                if not path.is_file() or path.is_symlink() or sha(path) != digest:
                    errors.append(f"1217 native receipt asset changed/missing: {path}")
        declared_env = {}
        for token in parent.get("argv", [])[2:]:
            if token.startswith("/"):
                break
            if "=" in token:
                key, value = token.split("=", 1)
                declared_env[key] = value
        recorded_env = metadata.get("environment", {})
        if not isinstance(recorded_env, dict):
            errors.append("1217 native receipt environment is not an object")
            recorded_env = {}
        for key in ("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
                    "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"):
            if key not in declared_env or recorded_env.get(key) != declared_env.get(key):
                errors.append(f"1217 native receipt environment differs from parent declaration: {key}")
        # The owner derives these two runtime environment entries from the
        # declaration's presentation-only CLI flags.
        expected_tour = "1" if "--inspection-tour" in parent.get("argv", []) else "0"
        if recorded_env.get("NUMI_HUMAN_RESTING_INSPECTION_TOUR") != expected_tour:
            errors.append("1217 native inspection-tour environment differs from declared CLI")
        if recorded_env.get("NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS") != flag_value(parent["argv"], "--inspection-period-seconds"):
            errors.append("1217 native inspection-period environment differs from declared CLI")
        native_argv = metadata.get("argv", [])
        if not isinstance(native_argv, list) or len(native_argv) < 5:
            errors.append("1217 native receipt argv is missing or malformed")
        else:
            expected_bin = str(Path(declared_env.get("NUMI_BUILD_DIR", "")) / "bin/numi-human-native")
            if native_argv[0] != expected_bin:
                errors.append("1217 native executable path differs from declared build")
            try:
                expected_scene, expected_receipt = parent_source_paths(parent)
                source_scene = load_json(expected_scene)
                # Native argv has three positional model inputs before its
                # output directory.
                if native_argv[4] != flag_value(parent["argv"], "--output"):
                    errors.append("1217 native output path differs from parent declaration")
                if flag_value(native_argv, "--resting-anatomy-receipt") != str(expected_receipt):
                    errors.append("1217 native anatomy receipt path differs from parent declaration")
                if flag_value(native_argv, "--skin-payload") != source_scene["source"]["skin"]["path"]:
                    errors.append("1217 native skin payload differs from parent scene")
                if flag_value(native_argv, "--support-contact-payload") != source_scene["outputs"]["support_contact"]["path"]:
                    errors.append("1217 native support payload differs from parent scene")
                if flag_value(native_argv, "--tendon-payload") != flag_value(parent["argv"], "--tendon"):
                    errors.append("1217 native tendon path differs from parent declaration")
                scene_flag = native_argv.index("--resting-scene")
                expected_inputs = [flag_value(parent["argv"], "--circulation"),
                                   flag_value(parent["argv"], "--respiration")]
                if native_argv[scene_flag + 1:scene_flag + 3] != expected_inputs:
                    errors.append("1217 native circulation/respiration paths differ from parent declaration")
            except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
                errors.append(f"1217 native receipt argv/input binding could not be checked: {exc}")
    if execution.get("returncode") != 0 or execution.get("changed_inputs") != {}:
        errors.append("1217 parent execution did not close cleanly")
    argv = parent.get("argv", [])
    if argv[0:2] != ["/usr/bin/env", "-i"]:
        errors.append("1217 argv no longer uses the frozen clean environment wrapper")
    if parent.get("seconds") != 40 or parent.get("dt") != DT_S or parent.get("contact_iterations") != 64:
        errors.append("1217 source horizon/dt/contact settings changed")
    if flag_value(argv, "--dt") != "0.002" or flag_value(argv, "--contact-iterations") != "64":
        errors.append("1217 CLI dt/contact iterations differ")
    if "--release-initialization" not in argv or "--rigid-hands" not in argv:
        errors.append("1217 release initialization or rigid hands flag missing")
    if flag_value(argv, "--inspection-period-seconds") != "2.5":
        errors.append("1217 inspection period differs from expected parent")
    if "--inspection-tour" not in argv:
        errors.append("1217 inspection tour flag missing")
    if any(x in argv for x in ("--resting-hip-capsule-reference", "--resting-bed-surface", "--resting-initial-posture")):
        errors.append("unexpected anatomy/posture override in 1217 argv")
    assets = parent.get("immutable_assets", {})
    if not isinstance(assets, dict) or len(assets) != 54:
        errors.append("1217 immutable asset inventory size changed")
    if check_assets and isinstance(assets, dict):
        for raw, digest in assets.items():
            path = Path(raw)
            if not path.is_file() or path.is_symlink() or sha(path) != digest:
                errors.append(f"1217 immutable input changed/missing: {path}")
    return parent, execution, errors


def capture_schedule() -> list[dict]:
    rows = []
    for step in CAPTURES:
        if step == 0:
            cls = "initial"
        elif step == 155000:
            cls = "terminal"
        else:
            require((step + 1) % 16 == 0, f"step {step} is not an accepted submission endpoint")
            cls = "submission_endpoint"
        rows.append({"class": cls, "step": step, "requested_time_s": step * DT_S})
    return rows


def parent_source_paths(parent: dict) -> tuple[Path, Path]:
    argv = parent["argv"]
    return Path(flag_value(argv, "--body-scene")), Path(flag_value(argv, "--anatomy-receipt"))


def validate_composition(review_path: Path, parent: dict, audit: dict) -> dict:
    review = load_json(review_path)
    require(review.get("schema") == "numi.human.skin-resting-local-reduction-004-composition-review.v1", "composition review schema mismatch")
    require(review.get("status") == "source_candidate_composed_native_pending", "composition review is not a completed source composition")
    require(review.get("native_run_performed") is False and review.get("candidate_admitted") is False, "composition review claims native admission/run")
    required_preservation = {
        "full_86_bindings_byte_identical": True,
        "indices_and_face_order_byte_identical": True,
        "NHA_payload_and_functional_bindings_unchanged": True,
        "current_refined_NHTISS_unchanged": True,
        "fixed_flat_bed_and_support_witnesses_unchanged": True,
        "normals_recomputed_by_existing_owner": True,
    }
    for key, expected in required_preservation.items():
        require(review.get("preservation", {}).get(key) is expected, f"composition preservation gate failed: {key}")
    require(review.get("nine_pose_audit", {}).get("aggregate_geometry_gate") is True, "composition review lacks passing aggregate exact audit")
    require(review["nine_pose_audit"].get("summary", {}).get("sha256") == audit["summary"]["sha256"], "composition review audit summary differs from independently verified root003")
    require(review["nine_pose_audit"].get("declaration", {}).get("sha256") == audit["declaration"]["sha256"], "composition review audit declaration differs from root003")
    for field in ("candidate_skin", "candidate_manifest", "composed_receipt", "candidate_scene"):
        obj = review.get(field)
        require(isinstance(obj, dict) and isinstance(obj.get("path"), str), f"composition review missing {field} pin")
        current = pin(Path(obj["path"]))
        require(current["sha256"] == obj.get("sha256") and current["bytes"] == obj.get("bytes"), f"composition {field} changed")
        review[field] = current
    scene_path = Path(review["candidate_scene"]["path"])
    skin_path = Path(review["candidate_skin"]["path"])
    manifest_path = Path(review["candidate_manifest"]["path"])
    receipt_path = Path(review["composed_receipt"]["path"])
    scene = load_json(scene_path)
    skin_manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    parent_scene_path, parent_receipt_path = parent_source_paths(parent)
    parent_scene = load_json(parent_scene_path)
    parent_receipt = load_json(parent_receipt_path)
    expected_scene = copy.deepcopy(parent_scene)
    expected_scene["source"]["skin"].update(path=str(skin_path), sha256=review["candidate_skin"]["sha256"])
    require(scene == expected_scene, "candidate scene differs from parent outside candidate skin path/hash")
    require(scene.get("bed") == parent_scene.get("bed"), "flat bed/support scene changed")
    require(skin_manifest.get("output_payload", {}).get("path") == str(skin_path), "candidate manifest output path differs")
    require(skin_manifest.get("output_payload", {}).get("sha256") == review["candidate_skin"]["sha256"], "candidate manifest output hash differs")
    require(receipt.get("payload") == parent_receipt.get("payload"), "anatomy payload changed")
    require(receipt.get("functional_bindings") == parent_receipt.get("functional_bindings"), "functional anatomy bindings changed")
    require(receipt.get("provenance", {}).get("native_muscle_surfaces") == parent_receipt.get("provenance", {}).get("native_muscle_surfaces"), "refined NHTISS lineage changed")
    require(receipt.get("thorax_source_volume_m3") == parent_receipt.get("thorax_source_volume_m3"), "thorax source volume changed")
    native_tissue = parent_receipt.get("provenance", {}).get("native_muscle_surfaces")
    require(isinstance(native_tissue, dict), "parent receipt lacks NHTISS identity")
    manifest_tissue = skin_manifest.get("current_physics_payloads", {})
    require(manifest_tissue.get("nhtiss_payload", {}).get("sha256") == native_tissue.get("sha256"), "candidate manifest NHTISS payload differs")
    require(manifest_tissue.get("nhtiss_manifest", {}).get("sha256") == native_tissue.get("manifest_sha256"), "candidate manifest NHTISS manifest differs")
    require(manifest_tissue.get("source_receipt", {}).get("sha256") == sha(parent_receipt_path), "candidate manifest source receipt pin differs")
    return {
        "review": pin(review_path),
        "candidate_skin": review["candidate_skin"],
        "candidate_manifest": review["candidate_manifest"],
        "candidate_scene": review["candidate_scene"],
        "composed_receipt": review["composed_receipt"],
        "bed_unchanged": True,
        "nhtiss_unchanged": True,
        "receipt_payload_and_functional_bindings_unchanged": True,
    }


def import_composer() -> object:
    require(sha(COMPOSER) == "5ff5cc9709b0565e00b11addc36d43f79b5903c6e987afcb1672b08f49b66cff", "reviewed composer changed")
    spec = importlib.util.spec_from_file_location("_root003b_composer", COMPOSER)
    require(spec is not None and spec.loader is not None, "cannot import frozen composition verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_argv(parent_argv: list[str], scene_path: Path, receipt_path: Path) -> tuple[list[str], list[dict]]:
    argv = list(parent_argv)
    changes = []
    changes.append(set_env(argv, "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT", str(RUN_DIR / "common-field-failure.json")))
    changes.append(set_env(argv, "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", ",".join(map(str, CAPTURES))))
    changes.append(set_flag(argv, "--body-scene", str(scene_path)))
    changes.append(set_flag(argv, "--anatomy-receipt", str(receipt_path)))
    changes.append(set_flag(argv, "--output", str(RUN_DIR)))
    changes.append(set_flag(argv, "--seconds", str(DURATION_S)))
    changes.append(set_flag(argv, "--inspection-period-seconds", str(INSPECTION_S)))
    expected_settings = {"NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT", "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
        "--body-scene", "--anatomy-receipt", "--output", "--seconds", "--inspection-period-seconds"}
    require({x["setting"] for x in changes} == expected_settings and len(changes) == len(expected_settings), "unexpected argument mutation set")
    require(flag_value(argv, "--dt") == "0.002", "dt changed")
    require(flag_value(argv, "--contact-iterations") == "64", "contact iterations changed")
    require(flag_value(argv, "--postural-activation-cap") == "0.01", "activation cap changed")
    require("--release-initialization" in argv and "--rigid-hands" in argv and "--inspection-tour" in argv, "required preserved CLI flags missing")
    require(flag_value(argv, "--tendon") == flag_value(parent_argv, "--tendon"), "tendon input changed")
    require(flag_value(argv, "--circulation") == flag_value(parent_argv, "--circulation"), "circulation input changed")
    require(flag_value(argv, "--respiration") == flag_value(parent_argv, "--respiration"), "respiration input changed")
    require(not any(x in argv for x in ("--resting-hip-capsule-reference", "--resting-bed-surface", "--resting-initial-posture")), "unapproved posture/bed override")
    return argv, changes


def make_declaration(plan: dict, parent: dict, parent_exec: dict, review_ctx: dict,
                     audit_ctx: dict, plan_pin: dict) -> dict:
    argv, argv_changes = build_argv(parent["argv"], Path(review_ctx["candidate_scene"]["path"]), Path(review_ctx["composed_receipt"]["path"]))
    assets = dict(parent["immutable_assets"])
    additional = [
        PREP / "run.py", Path(__file__), PLAN, PARENT_DECL, PARENT_EXEC, PARENT_NATIVE,
        COMPOSER, AUDIT_SCRIPT, AUDIT_SUMMARY, AUDIT_DECL,
        Path(review_ctx["review"]["path"]), Path(review_ctx["candidate_skin"]["path"]),
        Path(review_ctx["candidate_manifest"]["path"]), Path(review_ctx["candidate_scene"]["path"]),
        Path(review_ctx["composed_receipt"]["path"]),
    ]
    for path in additional:
        assets[str(Path(path).resolve())] = sha(path)
    # At declaration time, verify every inherited runtime input and every added
    # candidate/provenance input before freezing the exact hashes into argv.
    for raw, digest in assets.items():
        p = Path(raw)
        require(p.is_file() and not p.is_symlink() and sha(p) == digest, f"immutable asset changed/missing: {p}")
    cap_rows = capture_schedule()
    return {
        "schema": "numi.human.resting.native-run.declaration.v1",
        "status": "prepared_unlaunched",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": argv,
        "immutable_assets": assets,
        "body_scene": review_ctx["candidate_scene"]["path"],
        "anatomy_receipt": review_ctx["composed_receipt"]["path"],
        "seconds": DURATION_S,
        "dt": DT_S,
        "accepted_steps": 155000,
        "capture_steps": CAPTURES,
        "capture_schedule": cap_rows,
        "contact_iterations": 64,
        "postural_activation_cap": 0.01,
        "release_initialization": True,
        "rigid_hands_enabled": True,
        "flat_support_unchanged": True,
        "hip_reference_enabled": False,
        "inspection_period_seconds": INSPECTION_S,
        "allowed_differences_from_parent": [
            "004/006 composed skin scene and matching composed anatomy receipt; receipt payload, functional bindings, and refined NHTISS are byte/record identical",
            "310 s / 155000 accepted steps",
            "fresh native output and common-field failure-receipt paths",
            "accepted geometry captures at 0, 9983, 19999, 47519, 152191, 153183, 154143, and 155000",
            "inspection period 8.0 s; presentation-only",
        ],
        "parent_declaration": pin(PARENT_DECL),
        "parent_execution": pin(PARENT_EXEC),
        "parent_native_metadata": pin(PARENT_NATIVE),
        "parent_1217_physical_assets_preserved": True,
        "source_candidate": {
            "composition_review": review_ctx["review"],
            "candidate_skin": review_ctx["candidate_skin"],
            "candidate_manifest": review_ctx["candidate_manifest"],
            "candidate_scene": review_ctx["candidate_scene"],
            "composed_receipt": review_ctx["composed_receipt"],
            "root003_audit": audit_ctx,
            "preparation_plan": plan_pin,
        },
        "physical_scope": {
            "same_refined_1216_NHTISS": True,
            "same_flatbed_and_support_witnesses": True,
            "same_physical_respiration_circulation_tendon_and_brain_inputs": True,
            "same_runtime_native_identity_and_contact_law": True,
            "same_dt_s": DT_S,
            "same_contact_iterations": 64,
            "release_initialization": True,
            "rigid_hands": True,
            "argv_changes": argv_changes,
        },
        "qualification_boundary": "Prepared 310 s native run only. No run is performed by this builder; all physiology, geometry, runtime, and long-horizon checks remain pending.",
        "launch_status": "not launched; declaration only",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--composition-review", type=Path, default=DEFAULT_COMPOSITION_REVIEW)
    ap.add_argument("--report", type=Path, default=PREP / "declaration-builder-report-001.json")
    ap.add_argument("--write-declaration", action="store_true")
    ap.add_argument("--root-reviewed", action="store_true")
    args = ap.parse_args()
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        raise SystemExit(f"refuse to overwrite report: {report_path}")
    report = {
        "schema": "numi.human.resting.baseline-310s-declaration-builder-report.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder": pin(Path(__file__)),
        "status": "preflight_pending",
        "declaration_written": False,
        "declaration_path": str(RUN_DECL),
        "native_run_performed": False,
        "gate_checks": {},
        "errors": [],
    }
    try:
        plan = load_json(PLAN)
        plan_pin = check_plan(plan)
        parent, parent_exec, parent_errors = validate_parent(plan, check_assets=False)
        report["gate_checks"]["plan"] = {"passed": True, "pin": plan_pin}
        report["gate_checks"]["parent"] = {
            "passed": not parent_errors,
            "declaration": pin(PARENT_DECL),
            "execution": pin(PARENT_EXEC),
            "native_metadata": pin(PARENT_NATIVE) if PARENT_NATIVE.is_file() else None,
            "errors": parent_errors,
        }
        report["gate_checks"]["output_paths"] = {
            "native_run_directory_absent": not RUN_DIR.exists(),
            "run_declaration_absent": not RUN_DECL.exists(),
        }
        require(not RUN_DIR.exists(), "native output directory already exists")
        require(not RUN_DECL.exists(), "run-declaration.json already exists; refuse overwrite")
        root_missing = [str(p) for p in (AUDIT_SUMMARY, AUDIT_DECL, AUDIT_SCRIPT) if not p.is_file()]
        composition_missing = [] if args.composition_review.is_file() else [str(args.composition_review)]
        report["gate_checks"]["root003_audit_files"] = {
            "summary": str(AUDIT_SUMMARY), "declaration": str(AUDIT_DECL), "source": str(AUDIT_SCRIPT),
            "missing": root_missing,
        }
        report["gate_checks"]["composition_review"] = {
            "path": str(args.composition_review.resolve()), "missing": composition_missing,
        }
        if parent_errors:
            raise ValueError("; ".join(parent_errors))
        if root_missing or composition_missing:
            report["status"] = "waiting_for_passing_root003_audit_and_composition_review"
            report["pending"] = root_missing + composition_missing
        else:
            composer = import_composer()
            static, pre_report, proposal_pins, candidate = composer.verified_context()
            audit_ctx = composer.verify_audit(candidate, proposal_pins)
            review_ctx = validate_composition(args.composition_review, parent, audit_ctx)
            report["gate_checks"]["root003_audit"] = audit_ctx
            report["gate_checks"]["composition_review_valid"] = review_ctx
            if not args.root_reviewed:
                report["status"] = "all_artifact_gates_pass_waiting_for_root_review"
            elif not args.write_declaration:
                report["status"] = "all_gates_pass_declaration_not_requested"
            else:
                parent2, parent_exec2, asset_errors = validate_parent(plan, check_assets=True)
                require(not asset_errors, "; ".join(asset_errors))
                declaration = make_declaration(plan, parent2, parent_exec2, review_ctx, audit_ctx, plan_pin)
                # Exclusive creation prevents replacing any existing prepared run.
                with RUN_DECL.open("x") as f:
                    json.dump(declaration, f, indent=2, sort_keys=True, allow_nan=False)
                    f.write("\n")
                report["declaration_written"] = True
                report["declaration"] = pin(RUN_DECL)
                report["status"] = "prepared_unlaunched"
                report["argv_change_count"] = len(declaration["physical_scope"]["argv_changes"])
                report["immutable_asset_count"] = len(declaration["immutable_assets"])
    except Exception as exc:
        report["status"] = "preparation_refused"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "declaration_written": report["declaration_written"],
                      "native_run_performed": False}, sort_keys=True))
    return 0 if report["status"] != "preparation_refused" else 2


if __name__ == "__main__":
    raise SystemExit(main())

