"""Source-bound offline stance execution; native C++ owns all mechanics."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from .model import ImportError
from .qualification import fingerprint


def assess(text: str, exit_code: int | None, contact_count: int = 10) -> tuple[dict, list[str]]:
    lines = [s for s in text.splitlines()
             if s.startswith("numi_human_whole_body_support_wrench=")]
    if len(lines) != 1:
        return {}, ["missing_or_duplicate_wrench_summary"]
    metrics = {}
    try:
        for token in shlex.split(lines[0]):
            key, value = token.split("=", 1)
            if key in metrics:
                return metrics, ["duplicate_metric:" + key]
            metrics[key] = value
    except ValueError:
        return {}, ["malformed_wrench_summary"]
    failures = []
    if exit_code != 0:
        failures.append("native_process_failed_or_timed_out")
    geometry = [s for s in text.splitlines()
                if s.startswith("compiled_support_geometry=")]
    if len(geometry) != 2:
        failures.append("missing_or_duplicate_geometry_replay")
    for line in geometry:
        try:
            pairs = [token.split("=", 1) for token in shlex.split(line)]
            values = dict(pairs)
            gap = float(values["compiled_support_min_gap_m"])
            tolerance = float(values["compiled_support_gap_tolerance_m"])
            separated = float(values["compiled_support_max_separated_force_n"])
            if (len(values) != len(pairs) or values["compiled_support_geometry"] != "admissible"
                    or not all(math.isfinite(x) for x in (gap, tolerance, separated))
                    or not 0 < tolerance <= 1e-6 or gap < -tolerance or separated != 0):
                raise ValueError()
        except (KeyError, ValueError):
            failures.append("inadmissible_support_geometry")
    for key, expected in {"numi_human_whole_body_support_wrench": "ok",
                          "replay": "bitwise", "support_contacts": str(contact_count),
                          "passive_joint_tissue": "none"}.items():
        if metrics.get(key) != expected:
            failures.append("missing_or_unexpected:" + key)
    for key, value in metrics.items():
        try:
            numeric = float(value)
        except ValueError:
            continue
        if not math.isfinite(numeric):
            failures.append("nonfinite:" + key)
    try:
        weight = float(metrics["expected_weight_n"])
        forces = [float(metrics[f"contact_{i}_normal_force_n"]) for i in range(contact_count)]
        if (weight <= 0 or any(f < 0 for f in forces)
                or abs(sum(forces) - weight) / weight > 1e-5
                or abs(float(metrics["total_support_force_n"]) - sum(forces)) > 1e-5
                or not 0 <= float(metrics["max_root_force_residual"]) <= 1e-3):
            failures.append("unbalanced_or_inadmissible_wrench")
        if metrics["internal_balanced"] not in ("true", "false"):
            raise ValueError()
    except (KeyError, ValueError, ZeroDivisionError):
        failures.append("missing_or_invalid_wrench_metrics")
    return metrics, failures


def add_arguments(parser):
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[2] / "config/myosim-support-stance.v1.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runtime-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.set_defaults(handler=run)


def run(args):
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ImportError("stance timeout must be finite and positive")
    profile_path = args.profile.resolve()
    profile = json.loads(profile_path.read_text())
    contact_count = profile.get("contact_count", 10)
    if type(contact_count) is not int or not 1 <= contact_count <= 32:
        raise ImportError("invalid support stance contact count")
    inputs, build, output = args.input.resolve(), args.runtime_build.resolve(), args.output.resolve()
    files = {key: inputs / value["file"] for key, value in profile["payloads"].items()}
    for key, path in files.items():
        actual = fingerprint(path)
        if any(actual[field] != profile["payloads"][key][field] for field in ("sha256", "bytes")):
            raise ImportError(f"stance source payload drift: {key}")
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    files.update(binary=binary, library=build / "lib/libmetalrobo.dylib")
    cache = build / "CMakeCache.txt"
    files.update(profile=profile_path, qualifier=Path(__file__).resolve(), build_cache=cache)
    homes = [line.split("=", 1)[1] for line in cache.read_text().splitlines()
             if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    if len(homes) != 1:
        raise ImportError("stance build has no unique native source owner")
    root = Path(homes[0]).resolve()
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments])
    command = [str(binary), str(files["rigid"]), str(files["muscles"]),
               str(files["bones"]), str(output / "native"),
               "--whole-body-support-certificate", "--muscle-step-seconds", "0.0001",
               "--support-contact-payload", str(files["support_contact"]),
               "--joint-equality-payload", str(files["joint_equalities"])]
    for coordinate in profile["coordinates"]:
        command += ["--support-stance-dof", str(coordinate["dof_index"]),
                    str(coordinate["maximum_displacement"])]
    for contact in profile["active_contacts"]:
        command += ["--support-stance-contact", str(contact["witness_index"])]
    receipt = {"schema": "numi.human.support-stance-receipt.v1", "status": "running",
               "runtime_revision": git("rev-parse", "HEAD").decode().strip(),
               "runtime_worktree_status": git("status", "--porcelain").decode(),
               "artifacts": {key: fingerprint(path) for key, path in files.items()},
               "build_cache": fingerprint(cache), "profile": profile,
               "profile_artifact": fingerprint(profile_path),
               "command": command, "standing_qualified": False, "walking_qualified": False,
               "boundary": profile["boundary"]}
    output.mkdir(parents=True, exist_ok=False)
    patch = output / "runtime.patch"
    patch.write_bytes(git("diff", "HEAD", "--binary"))
    receipt["runtime_patch"] = fingerprint(patch)
    snapshot = output / "qualifier.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    receipt["qualifier"] = fingerprint(snapshot)
    def publish():
        temporary = output / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "receipt.json")
    publish()
    stdout, stderr = output / "stdout.log", output / "stderr.log"
    started = time.monotonic()
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = None
    metrics, failures = assess(stdout.read_text(), code, contact_count)
    for key, path in files.items():
        if fingerprint(path)["sha256"] != receipt["artifacts"][key]["sha256"]:
            failures.append("artifact_changed_during_run:" + key)
    if git("rev-parse", "HEAD").decode().strip() != receipt["runtime_revision"]:
        failures.append("runtime_revision_changed_during_run")
    if hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest() != receipt["runtime_patch"]["sha256"]:
        failures.append("runtime_source_changed_during_run")
    receipt.update(status="failed" if failures else "wrench_only_passed", exit_code=code,
                   wall_seconds=time.monotonic() - started, metrics=metrics, failures=failures,
                   stdout=fingerprint(stdout), stderr=fingerprint(stderr))
    publish()
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json"),
                      "standing_qualified": False, "walking_qualified": False}))
    return 1 if failures else 0
