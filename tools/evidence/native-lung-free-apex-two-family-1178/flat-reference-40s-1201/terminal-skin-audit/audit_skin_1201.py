#!/usr/bin/env python3
"""Run-specific exact terminal skin audit for flat-control capture 1201.

Without --scan this performs only source/invocation/receipt preflight. It reads
an existing native run; it never launches or modifies the native simulation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

R = Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201")
PARENT_DECLARATION = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/run-declaration.json")
PARENT_RUN = PARENT_DECLARATION.parent / "native-run"
PARENT_DECLARATION_SHA = "e9f7ada8ea4eb52760a790edb3fb3a3c05953c5ad06fac3f605b943b6405f05a"
ALLOWED_ENV_DELTA = {"NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT", "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"}
FORBIDDEN_OPTIONS = {"--resting-hip-capsule-reference", "--resting-bed-surface", "--resting-initial-posture"}
E = Path("/Users/n/numi-human-resting-evidence-20261005")
RUN_DEFAULT = R / "native-run"
OUT_DEFAULT = R / "skin-audit-1201-terminal"
STEPS = (20000,)
DT = 0.002
ROOTS = 20000
SKIN = E / "native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
SKIN_SHA = "b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b"
SKIN_REPORT = E / "native-skin-epl143-clearance-1187/attempt-006/report.json"
SKIN_REPORT_SHA = "9df248cf4cdcad92dc28a7ad5b0336c365e93ecf712e5811b8d392a11b708e69"
NHA = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"
NHA_SHA = "1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
NHA_COMPOSITION = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/composition-report.json"
NHA_COMPOSITION_SHA = "ddbfae12a5acd5c35521cb913121f08e49fb8c336c89b7dd60df3272ec0a7fbc"
NHA_MANIFEST = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-anatomy-manifest.json"
NHA_MANIFEST_SHA = "de298b7cc413e0cf079b39cb5d4792ebb537642f23a52cb4747f2dc37fac917f"
TISS = E / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS_SHA = "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd"
TISS_MANIFEST = E / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
TISS_MANIFEST_SHA = "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde"
RUNNER = E / "native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py"
RUNNER_SHA = "b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb"
INV = E / "native-complete-skin-containment-audit-890/pair-summary-v3.csv"
INV_SHA = "a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
PINNED_STATIC = {
    SKIN: SKIN_SHA, SKIN_REPORT: SKIN_REPORT_SHA,
    NHA: NHA_SHA, NHA_COMPOSITION: NHA_COMPOSITION_SHA, NHA_MANIFEST: NHA_MANIFEST_SHA,
    TISS: TISS_SHA, TISS_MANIFEST: TISS_MANIFEST_SHA,
    RUNNER: RUNNER_SHA, INV: INV_SHA,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_hash(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            "pinned file missing, symlinked, or changed: " + str(path))


def option(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        raise ValueError("native invocation lacks " + name)


def load_runner():
    check_hash(RUNNER, RUNNER_SHA)
    spec = importlib.util.spec_from_file_location("_skin_exact_audit_1172_for_1193_attempt002", RUNNER)
    require(spec is not None and spec.loader is not None, "cannot load pinned 1172 audit owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse only the pinned owner functions; the new NHSKIN is explicitly source-pinned here.
    module.SKIN = SKIN
    module.SKIN_SHA = SKIN_SHA
    return module


def _env_and_cli(argv):
    require(isinstance(argv, list) and len(argv) >= 4 and argv[:2] == ["/usr/bin/env", "-i"],
            "owner launch argv is not the expected clean env -i form")
    executable = "/Users/n/numi-human-performance-source-014/tools/numi"
    require(executable in argv, "owner launch argv lacks the pinned numi CLI")
    executable_index = argv.index(executable)
    env = {}
    for item in argv[2:executable_index]:
        require("=" in item, "malformed clean environment entry")
        key, value = item.split("=", 1)
        require(key not in env, "duplicate clean environment entry: " + key)
        env[key] = value
    return env, argv[executable_index:]


def _normalized_cli(cli):
    require(not (FORBIDDEN_OPTIONS & set(cli)), "flat-control launch includes bed, hip capsule, or initial-posture option")
    out = []
    i = 0
    while i < len(cli):
        item = cli[i]
        if item in ("--output", "--seconds"):
            require(i + 1 < len(cli), "owner CLI option lacks value: " + item)
            out.extend((item, "<" + item + ">"))
            i += 2
        else:
            out.append(item)
            i += 1
    return out


def _option_value(cli, name):
    return cli[cli.index(name) + 1] if name in cli and cli.index(name) + 1 < len(cli) else None


def validate_declared_flat_delta():
    declaration_path = R / "run-declaration.json"
    check_hash(PARENT_DECLARATION, PARENT_DECLARATION_SHA)
    child = json.loads(declaration_path.read_text())
    parent = json.loads(PARENT_DECLARATION.read_text())
    require(child.get("parent_declaration") == str(PARENT_DECLARATION)
            and child.get("parent_declaration_sha256") == PARENT_DECLARATION_SHA,
            "1201 does not bind the exact 1191 parent declaration")
    require(child.get("seconds") == 40 and child.get("dt") == DT
            and child.get("capture_steps") == [0, 20000]
            and child.get("contact_iterations") == 64
            and child.get("hip_capsule_enabled") is False,
            "1201 declaration differs from the authorized flat 40 s control settings")
    parent_assets = parent.get("immutable_assets", {})
    child_assets = child.get("immutable_assets", {})
    require(isinstance(parent_assets, dict) and len(parent_assets) == 25
            and isinstance(child_assets, dict), "parent immutable-input inventory is malformed")
    require(all(child_assets.get(path) == digest for path, digest in parent_assets.items()),
            "1201 changes or omits an immutable 1191 input")
    allowed_lineage = {
        str(PARENT_DECLARATION): PARENT_DECLARATION_SHA,
        str(PARENT_DECLARATION.parent / "run.py"): "fd11ec624fad091ad77f8a8c95e0467ef989cd215a01fe6220fb9158692b933e",
    }
    require({path: digest for path, digest in child_assets.items() if path not in parent_assets}
            == allowed_lineage, "1201 adds an unexpected immutable asset")
    for path, digest in child_assets.items():
        check_hash(path, digest)
    parent_env, parent_cli = _env_and_cli(parent["argv"])
    child_env, child_cli = _env_and_cli(child["argv"])
    require({k:v for k,v in parent_env.items() if k not in ALLOWED_ENV_DELTA}
            == {k:v for k,v in child_env.items() if k not in ALLOWED_ENV_DELTA},
            "1201 changed a non-output owner environment value")
    require(parent_env.get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS") == "0,4991,5375,5759,6111,6495,7743,10000"
            and child_env.get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS") == "0,20000",
            "1201 capture export schedule is not the authorized endpoint-only schedule")
    require(parent_env.get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT") !=
            child_env.get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT")
            and child_env.get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT", "").startswith(str(R)),
            "1201 common-failure receipt path was not relocated into its own run directory")
    require(_normalized_cli(parent_cli) == _normalized_cli(child_cli),
            "1201 owner CLI changed outside --seconds/--output")
    require(_option_value(parent_cli, "--seconds") == "20"
            and _option_value(child_cli, "--seconds") == "40"
            and _option_value(parent_cli, "--dt") == _option_value(child_cli, "--dt") == "0.002"
            and _option_value(child_cli, "--contact-iterations") == "64"
            and _option_value(child_cli, "--output") == str(R / "native-run"),
            "1201 owner CLI settings do not match the declared duration/output-only change")
    return {"parent_declaration_sha256": PARENT_DECLARATION_SHA,
            "parent_asset_count": len(parent_assets), "child_asset_count": len(child_assets),
            "authorized_owner_arg_changes": {"--seconds": ["20", "40"],
                "--output": "run-directory relocation",
                "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS": ["0,4991,5375,5759,6111,6495,7743,10000", "0,20000"],
                "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": "run-directory relocation"}}


def _runtime_invocation_delta():
    child_path = RUN_DEFAULT / "invocation.json"
    parent_path = PARENT_RUN / "invocation.json"
    require(child_path.is_file() and parent_path.is_file(),
            "actual native invocation is not yet available for source/input comparison")
    child = json.loads(child_path.read_text())
    parent = json.loads(parent_path.read_text())
    require(isinstance(child.get("asset_sha256"), dict)
            and child["asset_sha256"] == parent.get("asset_sha256"),
            "actual 1201 native asset map differs from the 1191 native input set")
    for path, digest in child["asset_sha256"].items():
        check_hash(path, digest)

    def normalized_args(argv, run_root):
        require(isinstance(argv, list), "native invocation argv is malformed")
        result = []
        i = 0
        while i < len(argv):
            item = argv[i]
            if item == "--muscle-step-count":
                require(i + 1 < len(argv), "native step-count option is missing its value")
                result.extend((item, "<STEP_COUNT>"))
                i += 2
                continue
            try:
                path = Path(item)
                relative = path.resolve().relative_to(run_root.resolve()) if path.is_absolute() else None
            except (OSError, ValueError):
                relative = None
            result.append("<RUN>/" + str(relative) if relative is not None else item)
            i += 1
        return result

    def normalized_environment(environment, run_root):
        require(isinstance(environment, dict), "native invocation environment is malformed")
        result = {}
        for key, value in environment.items():
            if key == "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS":
                require(value == ("0,20000" if run_root == RUN_DEFAULT else
                                  "0,4991,5375,5759,6111,6495,7743,10000"),
                        "native capture export schedule differs from the declared endpoint-only change")
                result[key] = "<CAPTURE_STEPS>"
                continue
            if key == "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT":
                expected = run_root / "common-field-failure.json"
                require(Path(value).resolve() == expected.resolve(),
                        "native common-failure receipt path is not scoped to the correct run")
                result[key] = "<RUN>/common-field-failure.json"
                continue
            try:
                path = Path(value)
                relative = path.resolve().relative_to(run_root.resolve()) if path.is_absolute() else None
            except (OSError, ValueError):
                relative = None
            result[key] = "<RUN>/" + str(relative) if relative is not None else value
        return result

    parent_argv, child_argv = parent.get("argv"), child.get("argv")
    require(not (FORBIDDEN_OPTIONS & (set(parent_argv) | set(child_argv))),
            "native invocation contains a bed, hip-capsule, or initial-posture option")
    require(normalized_args(parent_argv, PARENT_RUN) == normalized_args(child_argv, RUN_DEFAULT),
            "actual native argv differs from 1191 outside step-count/output-path relocation")
    require(_option_value(parent_argv, "--muscle-step-count") == "10000"
            and _option_value(child_argv, "--muscle-step-count") == "20000"
            and _option_value(parent_argv, "--muscle-step-seconds")
                == _option_value(child_argv, "--muscle-step-seconds") == "0.002"
            and _option_value(parent_argv, "--stand-contact-iterations")
                == _option_value(child_argv, "--stand-contact-iterations") == "64",
            "actual native timestep, duration, or contact iteration setting is unexpected")
    require(normalized_environment(parent.get("environment"), PARENT_RUN)
            == normalized_environment(child.get("environment"), RUN_DEFAULT),
            "actual native environment changed outside endpoint capture/failure-output settings")
    parent_support = _option_value(parent_argv, "--support-contact-payload")
    child_support = _option_value(child_argv, "--support-contact-payload")
    require(parent_support == child_support and parent["asset_sha256"].get(parent_support),
            "actual native contact payload differs from the 1191 32-contact input")
    return {"native_asset_count": len(child["asset_sha256"]),
            "native_asset_map_equal_to_1191": True,
            "native_argv_equal_except_step_count_and_output_paths": True,
            "native_environment_equal_except_capture_schedule_and_failure_receipt": True,
            "native_support_payload": parent_support,
            "native_support_payload_sha256": child["asset_sha256"][child_support],
            "native_step_count": 20000, "native_dt_seconds": 0.002,
            "contact_iterations": 64}


def _receipt_without_run_paths(document, run):
    def visit(value):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if key in ("path", "accepted_pack_path", "source_pack_path") and isinstance(child, str):
                    try:
                        result[key] = "<RUN>/" + str(Path(child).resolve().relative_to(run.resolve()))
                    except ValueError:
                        result[key] = child
                else:
                    result[key] = visit(child)
            return result
        if isinstance(value, list):
            return [visit(child) for child in value]
        return value
    return visit(document)


def verify_initial_capture_identity():
    left = RUN_DEFAULT / "accepted-geometry" / "step-0.mrvpack"
    right = PARENT_RUN / "accepted-geometry" / "step-0.mrvpack"
    left_receipt, right_receipt = left.with_suffix(".receipt.json"), right.with_suffix(".receipt.json")
    require(all(p.is_file() and not p.is_symlink() for p in (left, right, left_receipt, right_receipt)),
            "initial capture/receipt missing for 1201-to-1191 identity check")
    same_pack = sha(left) == sha(right)
    same_receipt = _receipt_without_run_paths(json.loads(left_receipt.read_text()), RUN_DEFAULT) == \
                   _receipt_without_run_paths(json.loads(right_receipt.read_text()), PARENT_RUN)
    require(same_pack and same_receipt,
            "1201 step-0 accepted pack/receipt differs from 1191 beyond output-root paths")
    return {"accepted_step": 0, "pack_sha256": sha(left),
            "pack_byte_identical_to_1191": same_pack,
            "receipt_equal_except_run_root_paths": same_receipt}


def validate_static_sources():
    for path, expected in PINNED_STATIC.items():
        check_hash(path, expected)
    skin_report = json.loads(SKIN_REPORT.read_text())
    assets = skin_report["candidate_assets"]
    require(Path(assets["composed_candidate_path"]).resolve() == SKIN.resolve()
            and assets["composed_candidate_sha256"] == SKIN_SHA,
            "1187 report does not bind the selected b2d NHSKIN")
    require(assets["triangle_indices_and_order_identical"] is True
            and assets["all_86_bindings_identical"] is True,
            "1187 candidate topology or 86-binding identity is not preserved")
    anatomy = json.loads(NHA_MANIFEST.read_text())
    require(anatomy["payload"]["path"] == str(NHA)
            and anatomy["payload"]["sha256"] == NHA_SHA,
            "1178 manifest does not bind the selected NHA")
    composition = json.loads(NHA_COMPOSITION.read_text())
    final_nha = composition.get("outputs", {}).get("final_nha", {})
    require(final_nha.get("path") == str(NHA) and final_nha.get("sha256") == NHA_SHA,
            "1178 composition report does not bind the selected NHA")
    tissue = json.loads(TISS_MANIFEST.read_text())
    tissue_payload = tissue.get("payload", {})
    require(tissue_payload.get("file") == TISS.name
            and tissue_payload.get("sha256") == TISS_SHA,
            "039 manifest does not bind the selected NHTISS")
    runner = load_runner()
    owner_pins = {
        runner.AUDIT_CORE: runner.AUDIT_CORE_SHA,
        runner.HELPER: runner.HELPER_SHA,
        runner.PRED: runner.PRED_SHA,
        runner.CLEARANCE: runner.CLEARANCE_SHA,
        **runner.PREDICATE_SOURCE_PINS,
    }
    for path, expected in owner_pins.items():
        check_hash(path, expected)
        PINNED_STATIC[path] = expected
    return {str(Path(path).resolve()): digest for path, digest in PINNED_STATIC.items()}


def preflight(run, out, scan):
    run = Path(run).expanduser().resolve()
    require(run == RUN_DEFAULT.resolve(), "run must be the designated 1201 flat-control native-run directory")
    require(run.is_dir() and not run.is_symlink(), "1201 flat-control native-run is not available as a real directory")
    fixed = validate_static_sources()
    delta = validate_declared_flat_delta()
    runtime_delta = _runtime_invocation_delta()
    invocation_path = run / "invocation.json"
    metadata_path = run / "run-metadata.json"
    log_path = run / "native.log"
    require(invocation_path.is_file() and metadata_path.is_file() and log_path.is_file(),
            "native invocation, run metadata, or native log is missing")
    invocation = json.loads(invocation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    argv, assets = invocation.get("argv"), invocation.get("asset_sha256")
    require(isinstance(argv, list) and isinstance(assets, dict), "invocation lacks argv or asset map")
    require(metadata.get("exit_code") == 0 and metadata.get("argv") == argv
            and metadata.get("asset_sha256") == assets,
            "native execution metadata does not prove the same successful invocation")
    parent_invocation_path = PARENT_RUN / "invocation.json"
    require(parent_invocation_path.is_file(), "1191 parent native invocation is missing")
    parent_invocation = json.loads(parent_invocation_path.read_text())
    require(isinstance(parent_invocation.get("asset_sha256"), dict)
            and assets == parent_invocation["asset_sha256"],
            "1201 actual native invocation asset map differs from 1191")
    require(not (FORBIDDEN_OPTIONS & set(argv)),
            "1201 native invocation includes an excluded bed, hip capsule, or posture option")
    runtime = metadata.get("loaded_metal_runtime", {})
    require(runtime.get("verified") is True, "loaded native Metal runtime is not verified")
    changed_sources = metadata.get("source_files_changed_during_run")
    if isinstance(changed_sources, list):
        require(not changed_sources, "native source files changed during the run")
    elif changed_sources is not None:
        require(changed_sources is False, "native source-change state is not a clean false value")
    require(Path(option(argv, "--skin-payload")).resolve() == SKIN.resolve()
            and assets.get(str(SKIN)) == SKIN_SHA,
            "actual native invocation did not load the exact 1187 b2d NHSKIN")
    require(Path(option(argv, "--torso-anatomy-payload")).resolve() == NHA.resolve()
            and assets.get(str(NHA)) == NHA_SHA,
            "actual native invocation did not load the exact 1178 NHA")
    require(Path(option(argv, "--soft-tissue-payload")).resolve() == TISS.resolve()
            and assets.get(str(TISS)) == TISS_SHA,
            "actual native invocation did not load the exact 039 NHTISS")
    dt, roots = float(option(argv, "--muscle-step-seconds")), int(option(argv, "--muscle-step-count"))
    require(dt == DT and roots == ROOTS, "native run is not the declared 40 s, 2 ms run")
    for step in STEPS:
        pack = run / "accepted-geometry" / ("step-%d.mrvpack" % step)
        receipt = pack.with_suffix(".receipt.json")
        require(pack.is_file() and not pack.is_symlink() and receipt.is_file() and not receipt.is_symlink(),
                "required accepted capture is missing or symlinked: step %d" % step)
        receipt_doc = json.loads(receipt.read_text())
        require(int(receipt_doc.get("accepted_step", -1)) == step,
                "accepted receipt identifies a different step: %d" % step)
        expected_time = step * DT
        require(abs(float(receipt_doc.get("accepted_time_s", float("nan"))) - expected_time) < 2e-6,
                "accepted receipt time does not match the requested capture: %d" % step)
    out = Path(out).expanduser().resolve()
    require(out == OUT_DEFAULT.resolve() and not out.exists(),
            "audit output must be the fresh designated skin-audit-1201-terminal directory")
    candidate_input_paths = [invocation_path, metadata_path, log_path]
    for step in STEPS:
        candidate_input_paths.extend((run / "accepted-geometry" / ("step-%d.mrvpack" % step),
                                      run / "accepted-geometry" / ("step-%d.receipt.json" % step)))
    fixed.update({str(p.resolve()): sha(p) for p in candidate_input_paths})
    # Recheck every asset hash declared by the actual invocation, including binary,
    # shader, NHCNT, and source payloads. This is a read-only identity check.
    for asset_path, expected_sha in assets.items():
        asset = Path(asset_path).resolve()
        check_hash(asset, expected_sha)
        fixed[str(asset)] = expected_sha
    return run, out, invocation, metadata, fixed, delta, runtime_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_DEFAULT,
                        help="read-only native-run directory; fixed to the 1201 flat-control run path")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT,
                        help="fresh audit output path; fixed to the 1201 flat-control sibling directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-check", action="store_true",
                      help="verify pinned static assets/owner sources only; no native-run required")
    mode.add_argument("--scan", action="store_true",
                      help="after strict preflight, run the CPU-only exact terminal geometry audit")
    args = parser.parse_args()
    if args.prepare_check:
        tracked = validate_static_sources()
        delta = validate_declared_flat_delta()
        runtime_delta = _runtime_invocation_delta()
        initial_identity = verify_initial_capture_identity()
        runner = load_runner()
        keys, _ = runner.inventory()
        require(len(keys) == 859, "expected all 859 target surfaces")
        runner.load_predicates()
        print(json.dumps({"status": "static_preparation_pins_valid_no_scan",
                          "targets": len(keys), "steps": list(STEPS),
                          "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
                          "NHTISS_sha256": TISS_SHA, "pinned_static_files": len(tracked),
                          "flat_control_parent_delta": delta,
                          "runtime_invocation_delta": runtime_delta,
                          "initial_capture_identity": initial_identity}, sort_keys=True))
        return 0
    run, out, invocation, metadata, tracked_before, delta, runtime_delta = preflight(args.run, args.out, args.scan)
    runner = load_runner()
    keys, expected = runner.inventory()
    require(len(keys) == 859, "expected all 859 target surfaces")
    ci, clearance, validator, core = runner.load_predicates()
    results = []
    declaration = {
        "schema": "numi.human.flat-reference-1201-terminal-skin-audit.declaration.v1",
        "status": "prepared_only_not_scanned" if not args.scan else "scan_started",
        "native_run": str(run),
        "native_invocation_sha256": sha(run / "invocation.json"),
        "native_run_metadata_sha256": sha(run / "run-metadata.json"),
        "native_log_sha256": sha(run / "native.log"),
        "dt_seconds": DT, "root_steps": ROOTS, "captures": list(STEPS),
        "NHSKIN": {"path": str(SKIN), "sha256": SKIN_SHA,
                   "source_report": str(SKIN_REPORT), "source_report_sha256": SKIN_REPORT_SHA},
        "NHA": {"path": str(NHA), "sha256": NHA_SHA,
                "composition_report_sha256": NHA_COMPOSITION_SHA,
                "manifest_sha256": NHA_MANIFEST_SHA},
        "NHTISS": {"path": str(TISS), "sha256": TISS_SHA,
                   "manifest_sha256": TISS_MANIFEST_SHA},
        "target_inventory": {"path": str(runner.INV), "sha256": INV_SHA,
                             "surfaces": 859, "ocular_surfaces": [[51010, n] for n in range(381, 398)]},
        "predicates": {"runner": str(RUNNER), "runner_sha256": RUNNER_SHA,
                       "audit_core_sha256": runner.AUDIT_CORE_SHA,
                       "accepted_pack_validator_sha256": runner.HELPER_SHA,
                       "exact_intersection_sha256": runner.PRED_SHA,
                       "skin_pack_reader_sha256": runner.CLEARANCE_SHA},
        "scope": "One actual accepted native terminal capture at step 20000 only; exact captured Float32-lattice NHSKIN-versus-all-859-target triangle pairs, whole-skin self intersections, and degenerate triangles. Initial step 0 is separately identity-matched to 1191. No contact exemptions, displacement reconstruction, continuous-time claim, or physiology qualification.",
        "inputs_before_scan": tracked_before,
        "flat_control_parent_delta": delta,
        "runtime_invocation_delta": runtime_delta,
        "initial_capture_identity": verify_initial_capture_identity(),
        "qualified_input_scope": "exact1191 native asset map; same32-contact NHCNT and64 iterations; only40s horizon plus endpoint export/output locations changed",
    }
    if not args.scan:
        print(json.dumps({"status": "preflight_passed_no_scan", "run": str(run),
                          "steps": list(STEPS), "targets": len(keys),
                          "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
                          "NHTISS_sha256": TISS_SHA}, sort_keys=True))
        return 0
    out.mkdir()
    declaration["status"] = "running_exact_terminal_capture_audit"
    (out / "declaration.json").write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    for step in STEPS:
        results.append(runner.run_pose(step, run, out, ci, clearance, validator, core,
                                       keys, expected, NHA_SHA))
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "status": results[-1]["status"],
                                     "elapsed_s": results[-1]["elapsed_s"]}, sort_keys=True) + "\n")
    tracked_after = {path: sha(path) for path in tracked_before}
    unchanged = tracked_before == tracked_after
    complete = len(results) == len(STEPS) and [r["accepted_step"] for r in results] == list(STEPS)                and all(r["pair_coverage_complete"] for r in results)
    free = complete and all(r["all_skin_crossing_pair_count"] == 0
                            and r["skin_self_crossing_pair_count"] == 0 for r in results)
    summary = {"schema": "numi.human.flat-reference-1201-terminal-skin-audit.summary.v1",
               "status": "complete_exact_intersection_free" if free and unchanged else
                         ("complete_with_intersections_or_invalid_geometry" if complete and unchanged else "incomplete_or_input_changed"),
               "native_run": str(run), "steps": list(STEPS), "dt_seconds": DT,
               "root_steps": ROOTS, "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
               "NHTISS_sha256": TISS_SHA, "target_surface_count": 859,
               "all_steps_complete": complete, "all_target_and_skin_self_pair_coverage_complete": complete,
               "intersection_free": free, "input_hashes_unchanged": unchanged,
               "inputs_before": tracked_before, "inputs_after": tracked_after,
               "pose_results": results,
               "qualification": "Exact geometry evidence for terminal step 20000 only; initial step 0 is transferred by complete pack/receipt identity against 1191. Raw crossings and degeneracies are retained. No interpolation between frames and no physical/physiological qualification."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "steps", "all_steps_complete", "intersection_free", "input_hashes_unchanged")}, sort_keys=True))
    return 0 if complete and unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
