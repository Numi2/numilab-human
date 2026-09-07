"""Reproducible native Human myofascia horizon qualification.

This is an execution receipt, not anatomical/material or standing validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shlex
import subprocess
import time
from pathlib import Path

from .model import ImportError

PAYLOADS = {
    "rigid": "myosim-fullbody-core-reference.nhrigid",
    "muscle": "myosim-fullbody-muscle-reference.nhmyo",
    "bone": "bodyparts3d-myosim-major-bones.nhbones",
    "soft-tissue": "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue",
    "tendon": "numi-human-tendon-attachments.nhtendon",
    "joint-equality": "myosim-fullbody-joint-equalities.nheq",
    "support-contact": "myosim-fullbody-support-contact.nhcnt",
    "pectoralis-fascia": "bodyparts3d-pectoralis-fascia.nhfascia",
}


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def assess(stdout: str, steps: int, exit_code: int | None) -> tuple[dict, list[str]]:
    """Require one native summary and explicit transaction evidence; fail closed."""
    summaries = [line for line in stdout.splitlines()
                 if line.startswith("myosim_articulated_bodyparts_bone_visual=")]
    if len(summaries) != 1:
        return {}, ["native_summary_missing_or_duplicated"]
    metrics = {}
    try:
        tokens = shlex.split(summaries[0])
    except ValueError:
        return {}, ["malformed_native_summary"]
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in metrics:
            return metrics, ["duplicate_metric:" + key]
        metrics[key] = value
    failures = []
    if exit_code != 0:
        failures.append("native_process_failed")
    expected = {
        "myosim_articulated_bodyparts_bone_visual": "ok",
        "persistent_completed_steps": str(steps),
        "pectoralis_fascia_steps": str(steps),
        "pectoralis_fascia_anchor_reaction_audited_steps": str(steps),
        "pectoralis_fascia_replay": "bitwise",
        "pectoralis_fascia_rollback": "verified",
        "stand_deterministic_replay": "bitwise",
        "persistent_root_assistance": "none",
        "tendon_step_transfers": str(832 * steps),
        "tendon_step_transaction": "NHTENDON3",
        "tendon_borrowed_consumer": "same_command_buffer_exact_snapshot",
        "tendon_rigid_state_effect": "source_JT_share_replaced_by_same_command_continuum_anchor_reaction",
        "pectoralis_fascia_directional_regions": "26",
        "pectoralis_fascia_directional_material_objects": "12",
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            failures.append("unexpected_or_missing:" + key)
    for key in ("pectoralis_fascia_minimum_J", "pectoralis_fascia_applied_force_n",
                "pectoralis_fascia_anchor_reaction_min_l1_n",
                "pectoralis_fascia_max_displacement_m",
                "pectoralis_fascia_coupled_transaction_elapsed_ms"):
        try:
            value = float(metrics[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError()
        except (KeyError, ValueError):
            failures.append("nonpositive_nonfinite_or_missing:" + key)
    # Every numeric metric must be finite, including metrics outside the gates.
    for key, value in metrics.items():
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isfinite(number):
            failures.append("nonfinite:" + key)
    device = metrics.get("pectoralis_fascia_device", "")
    if not device.startswith("Apple ") or metrics.get("muscle_force_metal_device") != device:
        failures.append("missing_or_mismatched_apple_devices")
    return metrics, failures


def qualify(arguments) -> int:
    if platform.system() != "Darwin":
        raise ImportError("native Human qualification requires macOS and Apple Metal")
    steps = arguments.steps
    if not steps or steps != sorted(set(steps)) or any(n < 1 or n > 64 for n in steps):
        raise ImportError("qualification steps must be increasing unique integers in [1, 64]")
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0:
        raise ImportError("qualification timeout must be finite and positive")
    root = arguments.runtime_root.resolve()
    build = arguments.runtime_build.resolve()
    inputs = arguments.input.resolve()
    output = arguments.output.resolve()
    binary = build / "bin/metalrobo_numilab_human_myosim_visual_probe"
    metallib = build / "matter/shaders/NumiMatter.metallib"
    files = {key: inputs / name for key, name in PAYLOADS.items()}
    files.update(binary=binary, matter_metallib=metallib,
                 articulated_metallib=build / "shaders/MetalRobo.metallib")
    for library in sorted((build / "lib").glob("*.dylib")):
        files["library:" + library.name] = library
    for path in files.values():
        if not path.is_file():
            raise ImportError(f"qualification input is missing: {path}")
    # Bind the supplied runtime root to the actual CMake build, not a label.
    cache = (build / "CMakeCache.txt").read_text()
    homes = [line.split("=", 1)[1] for line in cache.splitlines()
             if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    if len(homes) != 1 or Path(homes[0]).resolve() != root:
        raise ImportError("qualification runtime root does not own the CMake build")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    receipt = {
        "schema": "numi.human.myofascia-horizon-qualification.v1",
        "runtime_revision": git("rev-parse", "HEAD"),
        "runtime_worktree_status": git("status", "--porcelain"),
        "runtime_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary").encode()).hexdigest(),
        "build_cache": fingerprint(build / "CMakeCache.txt"),
        "artifacts": {key: fingerprint(path) for key, path in files.items()},
        "qualifier": fingerprint(Path(__file__).resolve()),
        "requested_steps": steps,
        "step_seconds": 1e-5,
        "activation_increment": 0.02,
        "selected_source_muscles": [186, 198],
        "runs": [],
        "boundary": "Bounded coupled transaction and replay evidence only; not sustained trunk loading, stable standing, anatomical/material calibration, or GPU kernel profiling.",
    }
    # Never overwrite evidence from an earlier run, including an interrupted run.
    output.mkdir(parents=True, exist_ok=False)
    patch = output / "runtime.patch"
    patch.write_bytes(subprocess.check_output(
        ["git", "-C", str(root), "diff", "HEAD", "--binary"]))
    receipt["runtime_patch"] = fingerprint(patch)
    receipt["runtime_diff_sha256"] = receipt["runtime_patch"]["sha256"]
    receipt["untracked_source"] = {}
    for name in subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]
    ).decode().split("\0"):
        if not name:
            continue
        source = root / name
        if source.is_file():
            snapshot = output / "untracked-source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
            receipt["untracked_source"][name] = fingerprint(snapshot)
    qualifier_snapshot = output / "qualifier.py"
    qualifier_snapshot.write_bytes(Path(__file__).read_bytes())
    receipt["qualifier"] = fingerprint(qualifier_snapshot)
    receipt_path = output / "receipt.json"
    def publish():
        temporary = output / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(receipt_path)
    receipt["status"] = "running"
    publish()
    for count in steps:
        prefix = output / f"steps-{count}"
        command = [str(binary), str(files["rigid"]), str(files["muscle"]),
                   str(files["bone"]), str(prefix)]
        for key in PAYLOADS:
            if key not in ("rigid", "muscle", "bone"):
                command += [f"--{key}-payload", str(files[key])]
        command += ["--muscle-step-seconds", "0.00001", "--muscle-step-count", str(count),
                    "--pectoralis-fascia-step-count", str(count), "--muscle-activation", "0.02",
                    "--selected-tendon-control",
                    "--activated-source-muscle-index", "186",
                    "--activated-source-muscle-index", "198", "--dimension", "512", "--camera-index", "0"]
        stdout_path = prefix.with_suffix(".stdout.txt")
        stderr_path = prefix.with_suffix(".stderr.txt")
        run = {"steps": count, "simulated_seconds": count * 1e-5,
               "command": ["/usr/bin/time", "-l", *command], "status": "running"}
        receipt["runs"].append(run)
        publish()
        started = time.monotonic()
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(run["command"], stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                exit_code = process.wait(timeout=arguments.timeout_seconds)
            except subprocess.TimeoutExpired:
                import os
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                exit_code = None
        run["wall_seconds"] = time.monotonic() - started
        run["exit_code"] = exit_code
        metrics, failures = assess(stdout_path.read_text(), count, exit_code)
        if exit_code is None:
            failures.append("timeout")
        stderr_text = stderr_path.read_text()
        run["joint_failure"] = [line for line in stderr_text.splitlines()
                                if line.startswith("human_joint_failure ")]
        run["contact_failure"] = [line for line in stderr_text.splitlines()
                                  if line.startswith(("deformable_contact_failure ",
                                                      "contact_primitive ", "contact_vertex "))]
        for key, path in files.items():
            if fingerprint(path)["sha256"] != receipt["artifacts"][key]["sha256"]:
                failures.append("artifact_changed_during_run:" + key)
        native_errors = [line for line in stderr_text.splitlines()
                         if line.startswith("myosim_articulated_visual=failed ")]
        run["native_errors"] = native_errors
        statuses = re.findall(r"(?:accepted_status|matter_status)=(\d+)",
                              "\n".join(native_errors + run["joint_failure"]))
        run["native_matter_status"] = int(statuses[-1]) if statuses else None
        run.update(metrics=metrics, failures=failures,
                   status="passed" if not failures else "failed",
                   stdout=fingerprint(stdout_path), stderr=fingerprint(stderr_path),
                   frames=[fingerprint(p) for p in sorted(prefix.glob("*.png"))],
                   peak_resident_bytes=None, retained_native_bytes=None)
        for line in stderr_text.splitlines():
            if line.strip().endswith("maximum resident set size"):
                run["peak_resident_bytes"] = int(line.split()[0])
        if not failures:
            elapsed_ms = float(metrics["pectoralis_fascia_coupled_transaction_elapsed_ms"])
            run["coupled_steps_per_second"] = count * 1000 / elapsed_ms
        publish()
        print(f"steps={count} status={run['status']} receipt={receipt_path}", flush=True)
        if failures:
            receipt["status"] = "failed"
            publish()
            return 1
    receipt["status"] = "passed"
    publish()
    return 0
