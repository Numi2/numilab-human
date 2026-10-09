#!/usr/bin/env python3
"""Safely replay the pinned P17 control scene through its existing owner CLI."""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005/native-resting-launcher-1159")
PLAN = Path("/Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-917/final-lung1159-plan-attempt2/draft/plan.json")
PLAN_SHA256 = "93881c3f869e3141e64f70676f1d255bb96e270f6758ec11548a57530e035467"
SOURCE_HASHES = Path("/Users/n/numi-human-resting-evidence-20261005/final-integrated-study-readiness-917/final-lung1159-plan-attempt2/source-hashes-final.json")
SOURCE_HASHES_SHA256 = "0da08ff5e1a729c96037f1eda55df7f1ee0cfd04d4c0b5db622943e88e2924a6"
OWNER = Path("/Users/n/numi-human-area-audit-fma-fix-001/matter/tools/resting_intervention_study.py")
OWNER_SHA256 = "81b910a3ef745aa04bd47d1d9c291a1865cccb56f7234021fe10c6335252fe11"
TRIAL_ID = "resting-baseline"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(message):
    raise SystemExit("launcher: " + message)


def arg_value(argv, flag):
    positions = [i for i, item in enumerate(argv) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        fail("pinned trial must contain exactly one " + flag)
    return argv[positions[0] + 1]


def load_pinned_trial():
    if sha256(PLAN) != PLAN_SHA256:
        fail("P17 plan hash changed")
    if sha256(SOURCE_HASHES) != SOURCE_HASHES_SHA256:
        fail("P17 source-hash manifest changed")
    source_hashes = json.loads(SOURCE_HASHES.read_text(encoding="utf-8"))
    if source_hashes.get(str(OWNER)) != OWNER_SHA256 or sha256(OWNER) != OWNER_SHA256:
        fail("owner CLI differs from the P17 source pin")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    trials = [trial for trial in plan.get("trials", []) if trial.get("id") == TRIAL_ID]
    if len(trials) != 1:
        fail("pinned P17 control trial is missing or duplicated")
    trial = trials[0]
    if trial.get("arm") != "control":
        fail("pinned trial is not the control arm")
    argv = list(trial.get("argv", []))
    if len(argv) < 5 or Path(argv[1]).resolve() != OWNER.resolve() or argv[2] != "run-native":
        fail("trial does not call the pinned run-native owner CLI")
    if arg_value(argv, "--arm") != "control":
        fail("pinned trial arm changed")
    expected = {
        "--unit-id": "human-resting-reference-v1-15796c9c2b102cd97c32",
        "--world-fingerprint": "14697719457569737910",
        "--device": "Apple M4 Pro",
        "--steps": "155000",
        "--dt": "0.002",
        "--start-s": "60.0",
        "--end-s": "100.0",
        "--scale": "0.5",
        "--window-s": "30.0",
        "--program-fingerprint": "13368793716876897955",
    }
    for flag, value in expected.items():
        if arg_value(argv, flag) != value:
            fail("pinned control parameter changed: " + flag)
    if arg_value(argv, "--output") != "{run}/scene":
        fail("pinned output placeholder changed")
    invocation_path = Path(arg_value(argv, "--invocation"))
    identity_path = Path(arg_value(argv, "--native-build-identity"))
    identity_sha256 = arg_value(argv, "--native-build-identity-sha256")
    if not invocation_path.is_file() or not identity_path.is_file():
        fail("pinned invocation or native-build identity is missing")
    if sha256(identity_path) != identity_sha256:
        fail("native-build identity differs from the P17 trial pin")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    invocation = identity.get("native_invocation", {})
    if invocation.get("path") != str(invocation_path) or invocation.get("sha256") != sha256(invocation_path):
        fail("native-build identity does not bind the pinned launch template")
    return argv


def resolve_run_dir(output_dir, dry_run):
    if output_dir is None:
        if dry_run:
            # Preview only; never create this path during a dry run.
            return EVIDENCE / ("preview-control-" + uuid.uuid4().hex)
        return Path(tempfile.mkdtemp(prefix="native-resting-control-1159-", dir=str(EVIDENCE)))
    requested = Path(output_dir).expanduser()
    if not requested.is_absolute():
        fail("--output-dir must be an absolute path")
    run_dir = requested.resolve()
    try:
        run_dir.relative_to(EVIDENCE.resolve())
    except ValueError:
        fail("--output-dir must stay under the launcher evidence directory")
    if os.path.lexists(str(run_dir)):
        fail("--output-dir already exists; refusing to reuse it")
    if not run_dir.parent.is_dir():
        fail("the parent of --output-dir must already exist")
    if not dry_run:
        run_dir.mkdir()
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run the exact P17 control scene through the existing Human run-native CLI.")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify pins and print argv/cwd without creating directories or launching")
    parser.add_argument("--output-dir", help="optional fresh absolute run directory under this evidence directory")
    args = parser.parse_args()
    argv = load_pinned_trial()
    run_dir = resolve_run_dir(args.output_dir, args.dry_run)
    argv[argv.index("--output") + 1] = str(run_dir / "scene")
    print("P17 plan: %s sha256=%s" % (PLAN, PLAN_SHA256))
    print("Owner CLI: %s sha256=%s" % (OWNER, OWNER_SHA256))
    print("Run directory: %s" % run_dir)
    print("Scene output: %s" % (run_dir / "scene"))
    print("Working directory: %s" % run_dir)
    print("Resolved argv: %s" % shlex.join(argv))
    if args.dry_run:
        print("Dry run only; no directory was created and the owner was not launched.")
        return 0
    completed = subprocess.run(argv, cwd=str(run_dir), check=False)
    print("Owner process exit code: %d" % completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
