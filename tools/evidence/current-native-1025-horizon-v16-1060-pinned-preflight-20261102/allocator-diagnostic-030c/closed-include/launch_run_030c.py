#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def pin(path):
    p = Path(path)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}

def main():
    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experimental_status") != "READY_DIAGNOSTIC_FULL_ALLOCATOR_RUN_032D_NO_ADMISSION_CLAIM":
        raise SystemExit("run manifest is not diagnostic-ready")
    if manifest.get("physical_admission") is not False or manifest.get("native_steps") != 0:
        raise SystemExit("run manifest carries an invalid admission/native-step status")
    argv = manifest["argv"]
    if not argv or argv[0] != "/usr/bin/env":
        raise SystemExit("manifest argv must use /usr/bin/env")
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("NUMI_H706_FRACTIONAL_"):
            del env[name]
    command = []
    for token in argv[1:]:
        if "=" in token and not command:
            key, value = token.split("=", 1)
            env[key] = value
        else:
            command.append(token)
    if not command:
        raise SystemExit("manifest has no CLI command")
    source = manifest["source"]
    binary = manifest["binary"]
    for item in (source, binary, manifest["source_control_manifest"], manifest["compile_control"]):
        p = Path(item["path"])
        if not p.is_file() or p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise SystemExit(f"source/binary/control pin mismatch: {p}")
    for key in ("carrier_skin_packet", "carrier_fat_packet", "carrier_witness_skin", "carrier_witness_fat", "adaptive_packet_batch", "adaptive_packet_execution"):
        item = manifest["named_input_pins"][key]
        p = Path(item["path"])
        if not p.is_file() or p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]:
            raise SystemExit(f"032d packet input pin mismatch: {p}")
    output_dir = Path(env["NUMI_H706_FRACTIONAL_OUTPUT_DIR"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("refusing nonempty allocator output directory")
    stdout_path = Path(manifest["stdout_log"])
    stderr_path = Path(manifest["stderr_log"])
    execution_path = Path(manifest["execution_json"])
    for path in (stdout_path, stderr_path, execution_path):
        if path.exists():
            raise SystemExit(f"refusing preexisting execution output: {path}")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    exit_code = None
    timed_out = False
    try:
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            result = subprocess.run(
                command,
                cwd=manifest["cwd"],
                env=env,
                stdout=out,
                stderr=err,
                timeout=int(manifest["maximum_wall_seconds"]),
                check=False,
            )
            exit_code = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = 124
    elapsed = time.monotonic() - started
    output_files = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                output_files.append(pin(path))
    execution = {
        "schema": "numi.h706.current-fractional-mass-cli-execution.030c.v1",
        "status": "DIAGNOSTIC_RUN_TIMEOUT" if timed_out else ("DIAGNOSTIC_RUN_EXIT_0" if exit_code == 0 else "DIAGNOSTIC_RUN_EXIT_NONZERO"),
        "command": argv,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_wall_seconds": elapsed,
        "run_manifest": pin(manifest_path),
        "source": source,
        "binary": binary,
        "source_control_manifest": manifest["source_control_manifest"],
        "compile_control": manifest["compile_control"],
        "packet_batch": manifest["adaptive_joint_proposal"]["packet_batch"],
        "proposal": manifest["adaptive_joint_proposal"]["proposal_npz"],
        "stdout_log": pin(stdout_path),
        "stderr_log": pin(stderr_path),
        "output_dir": str(output_dir),
        "output_files": output_files,
        "allocator_run": True,
        "full_allocator_run": True,
        "physical_admission": False,
        "native_steps": 0,
        "capture_provenance": {
            "stdout_and_stderr": "direct subprocess file descriptors to separate run files",
            "terminal_reconstruction": False,
            "truncated": False
        }
    }
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "execution": pin(execution_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_wall_seconds": elapsed,
        "stdout": execution["stdout_log"],
        "stderr": execution["stderr_log"],
        "output_files": output_files
    }, sort_keys=True))
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())

