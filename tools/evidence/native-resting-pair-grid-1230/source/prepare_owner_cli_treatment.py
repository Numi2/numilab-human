#!/usr/bin/env python3
"""Prepare the existing Human owner-CLI native-run declaration; never launches."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

R = Path("/Users/n/numi-human-retained-delivery-20261009")
PKG = R / "skin-resting-multipose-clearance-1218"
PREP = PKG / "native-baseline-310s-preparation"
BASE_RUN_DECL = PREP / "run-declaration.json"
BASE_RUNNER = PREP / "run.py"
BASE_NATIVE = PREP / "native-run"
DIRECT_DIR = PKG / "direct-pair-preparation-001"
DIRECT_BUILDER = DIRECT_DIR / "prepare_treatment_declaration.py"
DIRECT_DECL = DIRECT_DIR / "treatment-declaration.json"
RUN_ROOT_DEFAULT = PKG / "native-treatment-310s-drive60-100-scale050"
RUN_OUTPUT_NAME = "native-run"
PREFLIGHT_NAME = "native-command-preflight.json"
OWNER = Path("/Users/n/numi-human-terminal-trace-capture-fix-1162/matter/tools/resting_intervention_study.py")
OWNER_SHA = "ef87b09a17b96c985d6f584932e806ba9e7c6a53785da32a37a56ac846a3ac5f"
DIRECT_BUILDER_SHA = "0114ff8260e50ccbcab49b98741e098370ec0dcfb2af336821f9c4db42ef9ea1"
BASE_DECL_SHA = "5319d47c6c2fd9fa0a581edbb02b07e6e6f18301407b060e6b52d13a5169f90f"
BASE_RUNNER_SHA = "fd11ec624fad091ad77f8a8c95e0467ef989cd215a01fe6220fb9158692b933e"
NUMI_TOOL_SHA = "2e971e7c3680ec809c7e9ac71355b89502f710f835bc3c69725a91dbbdb01074"
HUMAN_COMMAND_SHA = "77693c89dfb3bdc368b15d192d7e86f44ede696c516fa5a80e796f8ec8c3ce81"
RESTING_RUN_SHA = "f6bc12635fcf41227a79b4d67e058b36f392a4147d083ce24366f8c3ef41c412"
CAPTURES = [0, 49999, 55007, 155000]
STEPS, DT = 155000, 0.002
INTERVENTION = ["60", "100", "0.5"]
FAILURE_KEY = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
CAPTURE_KEY = "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def check_pin(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f"pinned source changed: {path}")


def parse_owner_cli(command: list[str]) -> tuple[dict[str, str], int, int]:
    if len(command) < 6 or command[:2] != ["/usr/bin/env", "-i"]:
        raise ValueError("expected the existing sanitized /usr/bin/env -i Numi owner CLI")
    i = 2
    environment: dict[str, str] = {}
    while i < len(command) and "=" in command[i] and not command[i].startswith("--"):
        key, value = command[i].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in environment:
            raise ValueError("malformed or duplicate sanitized environment entry")
        environment[key] = value
        i += 1
    if i + 1 >= len(command) or command[i + 1] != "human-resting":
        raise ValueError("owner argv does not dispatch the existing human-resting capability")
    return environment, i, i + 1


def load_resting_owner(environment: dict[str, str]):
    human_root = Path(environment["NUMI_HUMAN_ROOT"])
    lab_root = Path(environment["NUMI_LAB_ROOT"])
    build_dir = Path(environment["NUMI_BUILD_DIR"])
    tool = lab_root / "tools" / "numi"
    capability = lab_root / "numi" / "commands" / "human-resting"
    module_path = human_root / "src" / "numilab_human" / "resting_run.py"
    check_pin(tool, NUMI_TOOL_SHA)
    check_pin(capability, HUMAN_COMMAND_SHA)
    check_pin(module_path, RESTING_RUN_SHA)
    sys.path.insert(0, str(human_root / "src"))
    module = importlib.import_module("numilab_human.resting_run")
    return module, lab_root, build_dir, {
        "dispatcher": {"path": str(tool), "sha256": sha(tool)},
        "capability": {"path": str(capability), "sha256": sha(capability)},
        "resting_run_module": {"path": str(module_path), "sha256": sha(module_path)},
    }


def native_from_cli(command: list[str]) -> tuple[list[str], dict[str, str], dict[str, str], dict]:
    environment, tool_index, command_index = parse_owner_cli(command)
    module, lab_root, build_dir, source_pins = load_resting_owner(environment)
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    args = parser.parse_args(["--lab", str(lab_root), "--build", str(build_dir)] + command[command_index + 1:])
    native_argv, assets = module.command(args)
    if not isinstance(native_argv, list) or not isinstance(assets, dict):
        raise ValueError("Human owner command() returned malformed native argv/assets")
    return native_argv, assets, environment, source_pins


def modify_cli(baseline_argv: list[str], output: Path) -> tuple[list[str], list[dict]]:
    command = list(baseline_argv)
    env, tool_index, _ = parse_owner_cli(command)
    changes = []
    for key, value in (
        (FAILURE_KEY, str(output / "common-field-failure.json")),
        (CAPTURE_KEY, ",".join(map(str, CAPTURES))),
    ):
        prefix = key + "="
        hits = [i for i in range(2, tool_index) if command[i].startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"expected exactly one existing {key} setting")
        index = hits[0]
        old = command[index].split("=", 1)[1]
        command[index] = prefix + value
        changes.append({"setting": key, "old": old, "new": value, "argv_index": index})
    output_hits = [i for i, item in enumerate(command) if item == "--output"]
    if len(output_hits) != 1 or output_hits[0] + 1 >= len(command):
        raise ValueError("expected one owner CLI --output path")
    oi = output_hits[0] + 1
    old_output = command[oi]
    command[oi] = str(output)
    changes.append({"setting": "--output", "old": old_output, "new": str(output), "argv_index": oi})
    if "--drive-intervention" in command:
        raise ValueError("baseline owner CLI already declares a drive intervention")
    command.extend(["--drive-intervention"] + INTERVENTION)
    changes.append({"setting": "--drive-intervention", "old": None,
                    "new": INTERVENTION, "argv_index": len(command) - 4})
    # Check the command was only changed at the four authorized dimensions.
    normalized = list(command[:-4])
    for change in changes[:-1]:
        index = change["argv_index"]
        if change["setting"].startswith("--"):
            normalized[index] = change["old"]
        else:
            normalized[index] = change["setting"] + "=" + change["old"]
    base = list(baseline_argv)
    if normalized != base:
        raise ValueError("treatment owner CLI differs from baseline outside output/capture/intervention settings")
    return command, changes


def load_direct_builder():
    check_pin(DIRECT_BUILDER, DIRECT_BUILDER_SHA)
    spec = importlib.util.spec_from_file_location("direct_treatment_declaration_builder", DIRECT_BUILDER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import the pinned direct declaration builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_intervention_owner():
    check_pin(OWNER, OWNER_SHA)
    spec = importlib.util.spec_from_file_location("resting_intervention_study_1162", OWNER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import the pinned 1162 native command helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derive_direct_native(invocation: dict, output: Path) -> list[str]:
    owner = load_intervention_owner()
    args = argparse.Namespace(steps=STEPS, dt=DT, arm="treatment",
                              start_s=60.0, end_s=100.0, scale=0.5)
    return owner.native_scene_command(invocation, output, args)


def command_only() -> dict:
    check_pin(BASE_RUN_DECL, BASE_DECL_SHA)
    check_pin(BASE_RUNNER, BASE_RUNNER_SHA)
    base_decl = read_json(BASE_RUN_DECL)
    invocation = read_json(BASE_NATIVE / "invocation.json")
    baseline_cli_native, baseline_assets, _, source_pins = native_from_cli(base_decl["argv"])
    if baseline_cli_native != invocation.get("argv") or baseline_assets != invocation.get("asset_sha256"):
        raise ValueError("baseline actual invocation is not reproduced by its pinned owner CLI")
    output = RUN_ROOT_DEFAULT / RUN_OUTPUT_NAME
    treatment_cli, _ = modify_cli(base_decl["argv"], output)
    treatment_native, treatment_assets, _, treatment_sources = native_from_cli(treatment_cli)
    direct_native = derive_direct_native(invocation, output)
    if treatment_native != direct_native:
        raise ValueError("treatment owner CLI native command differs from the direct declaration derivation")
    if treatment_assets != invocation.get("asset_sha256"):
        raise ValueError("treatment owner CLI changed native asset bindings")
    return {"status": "read_only_command_parity_pass", "native_argv_count": len(treatment_native),
            "native_asset_count": len(treatment_assets), "expected_native_argv_sha256": canonical_sha(treatment_native),
            "baseline_owner_source_pins": source_pins, "treatment_owner_source_pins": treatment_sources,
            "native_launch_performed": False}


def build(run_root: Path) -> dict:
    check_pin(BASE_RUN_DECL, BASE_DECL_SHA)
    check_pin(BASE_RUNNER, BASE_RUNNER_SHA)
    check_pin(DIRECT_BUILDER, DIRECT_BUILDER_SHA)
    if not run_root.is_absolute() or run_root.exists() or run_root.is_symlink():
        raise ValueError("treatment declaration root must be a fresh absolute directory")
    run_output = run_root / RUN_OUTPUT_NAME
    if run_output.exists() or run_output.is_symlink():
        raise ValueError("treatment native output path already exists")
    baseline_decl = read_json(BASE_RUN_DECL)
    baseline_execution_path = PREP / "execution.json"
    baseline_execution = read_json(baseline_execution_path)
    baseline_invocation_path = BASE_NATIVE / "invocation.json"
    baseline_metadata_path = BASE_NATIVE / "run-metadata.json"
    baseline_log_path = BASE_NATIVE / "native.log"
    baseline_trace_path = BASE_NATIVE / "resting-coupled.csv"
    if not baseline_metadata_path.is_file():
        raise ValueError("direct baseline is still open; no treatment declaration written")
    baseline_invocation = read_json(baseline_invocation_path)
    baseline_metadata = read_json(baseline_metadata_path)
    if (baseline_execution.get("returncode") != 0 or baseline_execution.get("changed_inputs") != {} or
            baseline_execution.get("declaration_sha256") != sha(BASE_RUN_DECL)):
        raise ValueError("baseline owner execution is not a closed successful run")
    if (baseline_metadata.get("exit_code") != 0 or
            baseline_metadata.get("source_files_changed_during_run") != [] or
            baseline_metadata.get("argv") != baseline_invocation.get("argv") or
            baseline_metadata.get("asset_sha256") != baseline_invocation.get("asset_sha256") or
            baseline_metadata.get("environment") != baseline_invocation.get("environment") or
            baseline_metadata.get("loaded_metal_runtime", {}).get("verified") is not True):
        raise ValueError("baseline native invocation/run-metadata/runtime did not close cleanly")
    if not baseline_log_path.is_file() or not baseline_trace_path.is_file():
        raise ValueError("closed baseline is missing native log or physiology trace")
    native_cli, assets, env, source_pins = native_from_cli(baseline_decl["argv"])
    if native_cli != baseline_invocation["argv"] or assets != baseline_invocation["asset_sha256"]:
        raise ValueError("baseline owner CLI no longer reproduces the actual native invocation")

    direct_builder = load_direct_builder()
    direct = direct_builder.build_template(BASE_NATIVE, run_output)
    direct_bytes = json_bytes(direct)
    direct_sha = hashlib.sha256(direct_bytes).hexdigest()
    if DIRECT_DECL.exists() or DIRECT_DECL.is_symlink():
        raise ValueError("direct treatment declaration path already exists; preserve it and use a fresh attempt")
    treatment_cli, argv_changes = modify_cli(baseline_decl["argv"], run_output)
    treatment_native, treatment_assets, _, treatment_source_pins = native_from_cli(treatment_cli)
    direct_native = direct["treatment"]["argv"]
    if treatment_native != direct_native:
        raise ValueError("owner CLI treatment argv differs from the pinned direct treatment declaration")
    if treatment_assets != direct["treatment"]["immutable_assets"]:
        raise ValueError("owner CLI treatment native assets differ from the direct declaration")

    capture_schedule = [
        {"class": "initial", "requested_time_s": 0.0, "step": 0},
        {"class": "submission_endpoint", "requested_time_s": 99.998, "step": 49999},
        {"class": "submission_endpoint", "requested_time_s": 110.014, "step": 55007},
        {"class": "terminal", "requested_time_s": 310.0, "step": 155000},
    ]
    preflight_path = run_root / PREFLIGHT_NAME
    preflight = {
        "schema": "numi.human.owner-cli-native-command-preflight.v1",
        "status": "prepared_not_launched",
        "baseline_run_declaration": {"path": str(BASE_RUN_DECL), "sha256": sha(BASE_RUN_DECL)},
        "baseline_invocation": {"path": str(baseline_invocation_path), "sha256": sha(baseline_invocation_path)},
        "direct_treatment_declaration": {"path": str(DIRECT_DECL), "sha256": direct_sha},
        "top_level_treatment_argv_sha256": canonical_sha(treatment_cli),
        "direct_expected_native_argv_sha256": canonical_sha(direct_native),
        "owner_cli_native_argv_sha256": canonical_sha(treatment_native),
        "owner_cli_native_argv": treatment_native,
        "native_asset_sha256": treatment_assets,
        "argv_equal_direct_declaration": treatment_native == direct_native,
        "native_assets_equal_direct_declaration": treatment_assets == direct["treatment"]["immutable_assets"],
        "source_pins": treatment_source_pins,
        "native_launch_performed": False,
    }
    preflight_bytes = json_bytes(preflight)
    preflight_sha = hashlib.sha256(preflight_bytes).hexdigest()
    runner_bytes = BASE_RUNNER.read_bytes()
    runner_sha = hashlib.sha256(runner_bytes).hexdigest()
    if runner_sha != BASE_RUNNER_SHA:
        raise ValueError("native run.py changed during declaration preparation")

    run_decl = copy.deepcopy(baseline_decl)
    run_decl.update(status="prepared_unlaunched", launch_status="not launched; declaration only",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    accepted_steps=STEPS, seconds=310, dt=DT, capture_steps=CAPTURES,
                    capture_schedule=capture_schedule, argv=treatment_cli)
    run_decl["allowed_differences_from_parent"] = list(baseline_decl["allowed_differences_from_parent"]) + [
        "Direct historical 310 s baseline is reused as control; no control rerun.",
        "Fresh treatment output and common-field failure-receipt paths.",
        "Treatment-only accepted captures at 0, 49999, 55007, and 155000; full physiology/surface streams unchanged.",
        "Existing owner CLI intervention --drive-intervention 60 100 0.5; all other physical settings match the direct baseline.",
    ]
    scope = run_decl.setdefault("physical_scope", {})
    scope["matched_historical_control"] = {"path": str(BASE_NATIVE),
        "invocation_sha256": sha(baseline_invocation_path), "run_metadata_sha256": sha(baseline_metadata_path)}
    scope["treatment_intervention"] = {"flag": "--drive-intervention", "start_s": 60.0, "end_s": 100.0,
        "scale": 0.5, "semantics": "delivered diaphragm/intercostal excitation scaled on [60,100); Brain chemoreflex remains active"}
    scope["argv_changes_from_direct_control"] = argv_changes
    scope["same_physical_inputs_runtime_timestep_horizon_contact_iterations"] = True
    scope["native_command_parity_verified"] = True
    run_decl["qualification_boundary"] = (
        "Prepared matched native treatment only. No treatment has been launched; physiological response, "
        "long-horizon replay, geometry, and anatomical qualification remain pending."
    )
    run_decl["direct_control_reference"] = {
        "run_path": str(BASE_NATIVE), "invocation_path": str(baseline_invocation_path),
        "invocation_sha256": sha(baseline_invocation_path), "run_metadata_path": str(baseline_metadata_path),
        "run_metadata_sha256": sha(baseline_metadata_path), "run_declaration_path": str(BASE_RUN_DECL),
        "run_declaration_sha256": sha(BASE_RUN_DECL), "execution_path": str(baseline_execution_path),
        "execution_sha256": sha(baseline_execution_path), "trace_sha256": sha(baseline_trace_path),
        "native_log_sha256": sha(baseline_log_path),
    }
    run_decl["direct_treatment_declaration"] = {"path": str(DIRECT_DECL), "sha256": direct_sha}
    run_decl["owner_cli_preflight"] = {"path": str(preflight_path), "sha256": preflight_sha}
    run_decl["immutable_assets"] = dict(run_decl["immutable_assets"])
    external_pins = {
        str(DIRECT_DECL): direct_sha,
        str(preflight_path): preflight_sha,
        str(run_root / "run.py"): runner_sha,
        str(Path(env["NUMI_LAB_ROOT"]) / "tools/numi"): treatment_source_pins["dispatcher"]["sha256"],
        str(Path(env["NUMI_LAB_ROOT"]) / "numi/commands/human-resting"): treatment_source_pins["capability"]["sha256"],
        str(Path(env["NUMI_HUMAN_ROOT"]) / "src/numilab_human/resting_run.py"): treatment_source_pins["resting_run_module"]["sha256"],
        str(DIRECT_BUILDER): DIRECT_BUILDER_SHA,
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str(BASE_RUN_DECL): sha(BASE_RUN_DECL),
        str(BASE_RUNNER): sha(BASE_RUNNER),
        str(PREP / "execution.json"): sha(PREP / "execution.json"),
        str(baseline_invocation_path): sha(baseline_invocation_path),
        str(baseline_metadata_path): sha(baseline_metadata_path),
        str(baseline_log_path): sha(baseline_log_path),
        str(baseline_trace_path): sha(baseline_trace_path),
    }
    for path, digest in external_pins.items():
        previous = run_decl["immutable_assets"].get(path)
        if previous is not None and previous != digest:
            raise ValueError(f"conflicting asset pin for {path}")
        run_decl["immutable_assets"][path] = digest
    run_decl["source_candidate"]["direct_treatment_declaration"] = {
        "path": str(DIRECT_DECL), "sha256": direct_sha}
    run_decl["source_candidate"]["owner_cli_preflight"] = {
        "path": str(preflight_path), "sha256": preflight_sha}
    run_decl["source_candidate"]["owner_cli_source_pins"] = treatment_source_pins

    # Materialize only after all owner/source/command checks passed. All writes
    # are exclusive; a partial failed attempt remains visible for review.
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "run.py").write_bytes(runner_bytes)
    (preflight_path).write_bytes(preflight_bytes)
    with DIRECT_DECL.open("xb") as stream:
        stream.write(direct_bytes)
    with (run_root / "run-declaration.json").open("xb") as stream:
        stream.write(json_bytes(run_decl))
    return {"status": "prepared_unlaunched", "run_root": str(run_root),
            "run_declaration_path": str(run_root / "run-declaration.json"),
            "run_declaration_sha256": sha(run_root / "run-declaration.json"),
            "run_py_path": str(run_root / "run.py"), "run_py_sha256": sha(run_root / "run.py"),
            "direct_treatment_declaration_path": str(DIRECT_DECL), "direct_treatment_declaration_sha256": sha(DIRECT_DECL),
            "owner_cli_preflight_path": str(preflight_path), "owner_cli_preflight_sha256": sha(preflight_path),
            "expected_native_argv_sha256": canonical_sha(treatment_native),
            "native_launch_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--command-only", action="store_true",
                        help="read-only baseline/treatment command mapping check; does not require baseline closure or write files")
    args = parser.parse_args()
    result = command_only() if args.command_only else build(args.package_root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, ImportError) as exc:
        raise SystemExit(f"owner-CLI treatment preparation refused: {exc}")
