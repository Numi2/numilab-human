#!/usr/bin/env python3
"""Launch the verified lung-and-thumb Human resting scene through the existing native owner CLI."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import uuid

EVIDENCE = Path("/Users/n/numi-human-retained-delivery-20261009/launchers")
SOURCE_LAUNCH = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/launch-command.sh")
SOURCE_LAUNCH_SHA256 = "03c5d50d210a743a3f2b61e30dab6cbe8628a498d8b91608bbf2572f550e1970"
NUMI = Path("/Users/n/numi-human-performance-source-014/tools/numi")
NUMI_SHA256 = "2e971e7c3680ec809c7e9ac71355b89502f710f835bc3c69725a91dbbdb01074"
CAPABILITY = Path("/Users/n/numi-human-performance-source-014/numi/commands/human-resting")
CAPABILITY_SHA256 = "77693c89dfb3bdc368b15d192d7e86f44ede696c516fa5a80e796f8ec8c3ce81"
OWNER_MODULE = Path("/Users/n/numi-human-resting-final-integration-001/src/numilab_human/resting_run.py")
OWNER_MODULE_SHA256 = "f6bc12635fcf41227a79b4d67e058b36f392a4147d083ce24366f8c3ef41c412"
COMMON_FAILURE_KEY = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("manual launcher: " + message)


def source_command() -> list[str]:
    pins = (
        (SOURCE_LAUNCH, SOURCE_LAUNCH_SHA256, "S1191 launch command"),
        (NUMI, NUMI_SHA256, "Numi dispatcher"),
        (CAPABILITY, CAPABILITY_SHA256, "human-resting capability"),
        (OWNER_MODULE, OWNER_MODULE_SHA256, "resting run owner"),
    )
    for path, digest, label in pins:
        require(path.is_file() and sha256(path) == digest, label + " source pin changed: " + str(path))

    lines = [line for line in SOURCE_LAUNCH.read_text(encoding="utf-8").splitlines()
             if line.startswith("exec /usr/bin/env -i ")]
    require(len(lines) == 1, "expected one pinned env/owner command in S1191 launch source")
    tokens = shlex.split(lines[0])
    require(tokens[:3] == ["exec", "/usr/bin/env", "-i"], "unexpected S1191 launcher structure")
    tokens = tokens[1:]  # exec is a shell builtin, not part of the process argv.
    try:
        command_at = tokens.index(str(NUMI))
    except ValueError:
        require(False, "pinned Numi executable is absent")
        raise AssertionError("unreachable")
    require(tokens[command_at + 1:command_at + 2] == ["human-resting"],
            "S1191 does not invoke the pinned human-resting owner capability")
    require(all(re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", item)
                for item in tokens[2:command_at]), "unexpected token before owner command")
    return tokens


def single_flag(argv: list[str], flag: str) -> int:
    rows = [i for i, item in enumerate(argv) if item == flag]
    require(len(rows) == 1 and rows[0] + 1 < len(argv), "expected one " + flag)
    return rows[0]


def transform(output: Path, seconds: int) -> list[str]:
    tokens = source_command()
    command_at = tokens.index(str(NUMI))
    env = tokens[2:command_at]
    command = tokens[command_at:]

    sec = single_flag(command, "--seconds")
    require(command[sec + 1] == "20", "S1191 source duration changed; review before adapting")
    command[sec + 1] = str(seconds)

    out = single_flag(command, "--output")
    command[out + 1] = str(output)

    tour = [i for i, item in enumerate(command) if item == "--inspection-tour"]
    period = [i for i, item in enumerate(command) if item == "--inspection-period-seconds"]
    require(len(tour) == 1 and len(period) == 1, "expected exactly one inspection-tour pair")
    require(command[period[0] + 1] == "2.5", "S1191 inspection period changed")
    for i in sorted((tour[0], period[0], period[0] + 1), reverse=True):
        command.pop(i)

    capture_assignments = [i for i, item in enumerate(env)
                           if item.startswith("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=")]
    require(len(capture_assignments) == 1, "expected one accepted geometry capture schedule")
    # Keep the tested complete-breath samples; the terminal step follows the
    # selected duration, since an interior capture must be on the native cadence.
    terminal_step = seconds * 500
    capture_steps = [0, 4991, 5375, 5759, 6111, 6495, 7743, terminal_step]
    env[capture_assignments[0]] = "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=" + ",".join(map(str, capture_steps))

    failure_assignments = [i for i, item in enumerate(env)
                           if item.startswith(COMMON_FAILURE_KEY + "=")]
    require(len(failure_assignments) == 1, "expected one common failure receipt environment binding")
    env[failure_assignments[0]] = COMMON_FAILURE_KEY + "=" + str(output / "common-field-failure.json")
    require(not any(item.startswith("NUMI_HUMAN_RESTING_INSPECTION_TOUR=") or
                    item.startswith("NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS=") for item in env),
            "inspection tour must not be enabled through the environment")

    result = tokens[:2] + env + command
    result_command_at = result.index(str(NUMI))
    app = result[result_command_at:]
    require(app[app.index("--seconds") + 1] == str(seconds),
            "manual duration differs from the requested bounded preset")
    require("--inspection-tour" not in app and "--inspection-period-seconds" not in app,
            "manual viewer must not enable the automatic inspection tour")
    require("--dt" in app and app[app.index("--dt") + 1] == "0.002" and
            "--dimension" in app and app[app.index("--dimension") + 1] == "512",
            "step size or rendering dimension changed")
    return result


def resolve_run_dir(requested: str | None, dry_run: bool) -> Path:
    if requested is None:
        if dry_run:
            return EVIDENCE / ("preview-" + uuid.uuid4().hex)
        return Path(tempfile.mkdtemp(prefix="manual-resting-", dir=str(EVIDENCE)))
    path = Path(requested).expanduser()
    require(path.is_absolute(), "--output-dir must be an absolute path")
    run_dir = path.resolve()
    try:
        run_dir.relative_to(EVIDENCE.resolve())
    except ValueError:
        require(False, "--output-dir must stay under " + str(EVIDENCE))
    require(not os.path.lexists(str(run_dir)), "--output-dir already exists; refusing to reuse it")
    require(run_dir.parent.is_dir(), "the parent of --output-dir must already exist")
    if not dry_run:
        run_dir.mkdir()
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="verify source pins and print the resolved owner command without launching")
    parser.add_argument("--seconds", type=int, choices=(20, 310), default=20,
                        help="default 20-second verified anatomy scene or a 310-second engineering session; late skin clearance remains unresolved")
    parser.add_argument("--output-dir", help="optional fresh absolute run directory under this evidence directory")
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.output_dir, args.dry_run)
    output = run_dir / "native-run"
    argv = transform(output, args.seconds)
    print("Scene qualification: 20-second native lung/thumb geometry and skin audit; late resting skin clearance remains unresolved.")
    print("Pinned S1191 launch SHA-256: " + SOURCE_LAUNCH_SHA256)
    print("resting_run.py SHA-256: " + OWNER_MODULE_SHA256)
    print("Run directory: " + str(run_dir))
    print("Owner output (creates its own directory): " + str(output))
    print("Resolved command: " + shlex.join(argv))
    if args.dry_run:
        print("Dry run only: no output directory was created and no native process was launched.")
        return 0
    result = subprocess.run(argv, cwd=str(run_dir), check=False)
    print("Owner command exit code: " + str(result.returncode))
    print("The existing resting_run owner writes invocation.json and run-metadata.json in native-run.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
