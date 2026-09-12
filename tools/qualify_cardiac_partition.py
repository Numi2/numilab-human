#!/usr/bin/env python3
"""Capture a reproducible offline cardiac partition qualification on macmini.

The physical host runs exact geometry authoring and regression tests here, not
hydraulic or mechanical stepping. Every consumed owner/source and emitted log
is hashed. Existing evidence directories are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["test_cardiac_face_arrangement", "test_cardiac_cavity_partition",
           "test_cardiac_partition_certificate", "test_cardiac_cavity_geometry", "test_cardiac_cavity_intersections"]
OWNERS = ["src/numilab_human/"+name+".py" for name in (
    "__init__", "model", "physiology", "cardiac_cavity_geometry", "cardiac_cavity_intersections",
    "cardiac_face_arrangement", "cardiac_cavity_partition", "cardiac_partition_certificate")]
OWNERS += ["tests/"+name+".py" for name in MODULES]
OWNERS += ["sources.lock.json", ".numi/commands/human-circulation-partition", "tools/qualify_cardiac_partition.py",
           "tools/verify_cardiac_partition.py"]
SOURCES = ["Sources/"+name for name in ("partof_BP3D_4.0_obj_99.zip", "partof_element_parts.txt", "isa_element_parts.txt")]
SOURCES += ["Docs/media/cardiac-cavities-20260912/independent/intersections.json"]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def inventory():
    return {name: digest(ROOT / name) for name in OWNERS+SOURCES}


def execute(output):
    output.mkdir(parents=True, exist_ok=False)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("qualification requires the physical Apple silicon Mac")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if chip != "Apple M4 Pro":
        raise ValueError("unexpected qualification processor")
    before = inventory()
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")+os.pathsep+str(ROOT / "tests"),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    record = {"schema": "HumanPack.cardiac-partition-execution.v1", "status": "running",
        "scope": "offline_exact_geometry_and_regressions", "physical_stepping": False,
        "platform": {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "chip": chip, "python": sys.version,
            "python_executable": sys.executable},
        "inputs_before": before, "runs": []}
    commands = [("compile", [str(ROOT / ".numi/commands/human-circulation-partition"), "--output", str(output / "partition.json")]),
                ("tests", [sys.executable, "-m", "unittest", "-v", *MODULES])]
    environment["NUMI_HUMAN_PYTHON"] = sys.executable
    for name, command in commands:
        print("running "+name, flush=True)
        start = time.monotonic()
        with (output / (name+".log")).open("xb") as stream:
            run = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record["runs"].append({"name": name, "command": command, "exit_code": run.returncode,
            "elapsed_seconds": time.monotonic()-start, "log": name+".log",
            "log_sha256": digest(output / (name+".log"))})
        (output / "execution.partial.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n")
    record["inputs_after"] = inventory()
    record["outputs"] = {name: digest(output / name) for name in ("partition.json", "compile.log", "tests.log")
                         if (output / name).is_file()}
    record["status"] = "pass" if before == record["inputs_after"] and all(run["exit_code"] == 0 for run in record["runs"]) else "fail"
    (output / "execution.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"status": record["status"], "output": str(output), "physical_stepping": False}))
    return 0 if record["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return execute(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
