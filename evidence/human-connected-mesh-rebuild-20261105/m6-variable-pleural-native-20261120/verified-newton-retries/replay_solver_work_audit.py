from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import sys

here = Path.cwd()
base = here.parent/"resting-clearance-and-atrium-013/glide-refresh-ccd-gap-trial/launch.json"
mode = sys.argv[1]
horizon = int(sys.argv[3]) if len(sys.argv) > 3 else None
tag = f"{mode}-{sys.argv[2]}" if mode.startswith("budget-") or mode == "certified-budget" else mode
run = here/(f"{tag}-steps{horizon}" if horizon is not None else tag)
run.mkdir()
launch = json.loads(base.read_text())
for file, digest in launch["pins"].items():
    with open(file,"rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != digest:
            raise RuntimeError(f"pin changed {file}")
launch["argv"][4] = str(run/"native")
if horizon is not None:
    if mode not in ("budget-horizon","certified-budget") or not 8 <= horizon <= 64:
        raise ValueError("extended matching trials require budget-horizon and 8 to 64 steps")
    launch["argv"][launch["argv"].index("--muscle-step-count")+1] = str(horizon)
configuration = launch["environment"]
for key in ("NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT", "NM_FEM_DEFORMABLE_LIMIT_TRACE_MICROTICK", "NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA"):
    configuration.pop(key,None)
if mode == "active-work":
    configuration["NUMI_MATTER_WORK_COUNTERS"] = "1"
elif mode == "certified-budget":
    audit_retry = sys.argv[2] in ("edge-compensated-audit", "edge-compensated-complete-audit")
    binaries = {
        "edge-compensated-audit": "certified-budget-audit-build",
        "edge-compensated-complete-audit": "certified-budget-complete-audit-build",
    }
    binary_directory = binaries.get(sys.argv[2], "certified-budget-build")
    binary = here/binary_directory/"numi-human-native"
    with binary.open("rb") as stream:
        launch["pins"][str(binary)] = hashlib.file_digest(stream,"sha256").hexdigest()
    launch["argv"][0] = str(binary)
    configuration["NUMI_HUMAN_CERTIFIED_NEWTON_BUDGET_TRIAL"] = "1"
    if sys.argv[2] in ("late-witness","late-barrier"):
        configuration.update(
            NM_FEM_NEWTON_TRACE_ROOT="15",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="15",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_MICROTICK="183",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA="1",
        )
    if sys.argv[2] == "late-barrier":
        configuration.pop("NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA")
        configuration.update(
            NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT="614",
            NM_FEM_CONTACT_TRACE_ITERATION="0",
        )
    edge_variants = {
        "edge-derivative": "edge-barrier-derivative",
        "edge-rounded-chord": "edge-barrier-rounded-chord",
        "edge-compensated-parameters": "edge-barrier-compensated-parameters",
        "edge-compensated-audit": "edge-barrier-compensated-parameters",
        "edge-compensated-complete-audit": "edge-barrier-compensated-parameters",
    }
    if sys.argv[2] in edge_variants:
        build_name = edge_variants[sys.argv[2]]
        library = here/build_name/"NumiMatter.metallib"
        with library.open("rb") as stream:
            launch["pins"][str(library)] = hashlib.file_digest(stream,"sha256").hexdigest()
        configuration["NUMI_MATTER_METALLIB_OVERRIDE"] = str(library)
    if sys.argv[2] in edge_variants and not audit_retry:
        configuration.update(
            NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="15",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_MICROTICK="183",
            NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA="1",
        )
elif mode in ("budget-sample","budget-horizon"):
    configuration["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = sys.argv[2]
else:
    raise ValueError(mode)
configuration["NUMI_HUMAN_COUPLED_TRACE_CSV"] = str(run/"coupled.csv")
configuration["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(run/"failure.json")
(run/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
environment = {k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
environment.update(configuration)
time_budget_seconds = 600 if mode == "certified-budget" and audit_retry else 390
start = time.monotonic()
stop = "native_exit"
with (run/"stdout.log").open("wb") as out, (run/"stderr.log").open("wb") as err:
    with subprocess.Popen(launch["argv"],cwd=launch["cwd"],env=environment,stdout=out,stderr=err) as process:
        try:
            while process.poll() is None:
                if mode=="budget-sample" and "resting_native_profile step=2 command_ms=" in (run/"stdout.log").read_text():
                    stop = "two_timing_samples_verified"
                    break
                if time.monotonic()-start >= time_budget_seconds:
                    stop = "time_budget_expired"
                    break
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        code = process.returncode
summary = {"exit":code,"stop":stop,"wall_seconds":time.monotonic()-start}
(run/"execution.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
