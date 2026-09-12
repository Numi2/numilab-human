#!/usr/bin/env python3
"""Build and invoke the native anatomy-equivalence probe; no Python stepping."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-native-commit", required=True)
    args = parser.parse_args()
    r, b, inputs, out = [p.resolve() for p in (args.runtime_root, args.build_root, args.inputs, args.output)]
    out.mkdir(parents=True, exist_ok=False)
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(r), *arguments], text=True).strip()
    record = {"schema": "HumanPack.cardiac-cavity-native-execution.v1",
              "native_commit": git("rev-parse", "HEAD"), "native_status_before": git("status", "--porcelain"),
              "runner_sha256": sha(Path(__file__)), "runs": []}
    try:
        if record["native_commit"] != args.expected_native_commit or record["native_status_before"]:
            raise RuntimeError("native source identity or worktree cleanliness differs")
        source = Path(__file__).resolve().with_name("cardiac_cavity_native_check.mm")
        executable = out / "cardiac-cavity-native-check"
        metal = b / "matter/shaders/NumiMatter.metallib"
        compiler, runtime = [b / ("matter/libnumi_matter_" + name + ".a") for name in ("compiler", "runtime")]
        build_inputs = [source, compiler, runtime, metal]
        record["build_inputs_before"] = {str(p): sha(p) for p in build_inputs}
        command = ["xcrun", "clang++", "-O3", "-DNDEBUG", "-std=c++23", "-arch", "arm64", "-fobjc-arc",
                   "-Wall", "-Wextra", "-Werror", '-DNUMI_MATTER_METALLIB="' + str(metal) + '"',
                   "-I" + str(r / "include"), "-I" + str(r / "matter/include"), str(source), str(runtime),
                   str(compiler), "-framework", "Foundation", "-framework", "Metal", "-o", str(executable)]
        record["build_command"] = command
        with (out / "build.log").open("xb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["build_exit_code"] = result.returncode
        record["build_inputs_after"] = {str(p): sha(p) for p in build_inputs}
        if result.returncode or record["build_inputs_before"] != record["build_inputs_after"]:
            raise RuntimeError("native build failed or its inputs changed")
        record["executable_sha256"] = sha(executable)
        record["dynamic_dependencies"] = subprocess.check_output(["otool", "-L", str(executable)], text=True)
        env = dict(os.environ); env["MTL_DEBUG_LAYER"] = "1"
        record["environment"] = {"MTL_DEBUG_LAYER": "1"}

        def run(baseline, associated, log_name):
            paths = [executable, metal, baseline, associated]
            row = {"command": [str(executable), str(baseline), str(associated)], "log": log_name,
                   "inputs_before": {str(p): sha(p) for p in paths}}
            start = time.monotonic()
            with (out / log_name).open("xb") as log:
                result = subprocess.run(row["command"], stdout=log, stderr=subprocess.STDOUT, env=env)
            row.update(exit_code=result.returncode, wall_seconds=time.monotonic() - start,
                       inputs_after={str(p): sha(p) for p in paths})
            return row

        for variant in ("upstream_equation", "heldt_table_aligned"):
            row = run(inputs / (variant + ".baseline.json"), inputs / (variant + ".native.json"), variant + ".native.log")
            row["volume_coordinates"] = variant
            record["runs"].append(row)
            print(json.dumps(row), flush=True)
            if row["exit_code"] or row["inputs_before"] != row["inputs_after"]:
                raise RuntimeError("native pair failed or its inputs changed")
        mutation = json.loads((inputs / "upstream_equation.native.json").read_text())
        mutation["compartments"][15]["initial_volume_m3"] *= 1.01
        mutation_path = out / "physical-mutation.json"
        mutation_path.write_text(json.dumps(mutation, sort_keys=True, separators=(",", ":")) + "\n")
        negative = run(inputs / "upstream_equation.baseline.json", mutation_path, "physical-mutation.log")
        record["negative_control"] = negative
        if (negative["exit_code"] != 1 or negative["inputs_before"] != negative["inputs_after"]
            or "changed physical parameters or unrelated fields" not in (out / negative["log"]).read_text()):
            raise RuntimeError("physical-mutation negative control did not reject for the expected reason")
        record["native_status_after"] = git("status", "--porcelain")
        record["native_commit_after"] = git("rev-parse", "HEAD")
        if record["native_status_after"] or record["native_commit_after"] != record["native_commit"]:
            raise RuntimeError("native source changed during qualification")
        record["status"] = "pass"
        return 0
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        print(json.dumps({"status": "failed", "error": str(error)}), flush=True)
        return 1
    finally:
        (out / "execution.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
