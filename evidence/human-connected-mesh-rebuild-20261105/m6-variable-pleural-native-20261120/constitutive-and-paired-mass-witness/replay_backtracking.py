"""Locate the 44 ms correction limit before the deformable-contact stage."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

here = Path.cwd()
prior = here.parent / "resting-clearance-and-atrium-015/motion-extension"
run = here / "motion-backtracking"
run.mkdir()
launch = json.loads((prior / "launch.json").read_text())
for name, expected in launch["pins"].items():
    with open(name, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
argv = launch["argv"]
argv[4] = str(run / "native")
argv[argv.index("--muscle-step-count") + 1] = "22"
launch["environment"].update(
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run / "coupled.csv"),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run / "failure.json"),
    NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="21",
    NM_FEM_DEFORMABLE_LIMIT_TRACE_MICROTICK="191",
    NM_FEM_BACKTRACKING_TRACE_MICROTICK="191",
    NM_FEM_BACKTRACKING_TRACE_ROOT="21",
    NM_FEM_BACKTRACKING_TRACE_OBJECT="all",
    NM_FEM_BACKTRACKING_TRACE_NEWTON="all",
    NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA="1",
)
runtime = here / "backtracking-witness-build/libmetalrobo.dylib"
with runtime.open("rb") as stream:
    launch["pins"][str(runtime)] = hashlib.file_digest(stream, "sha256").hexdigest()
launch["environment"]["DYLD_LIBRARY_PATH"] = str(runtime.parent)
(run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("NM_", "NUMI_"))}
environment.update(launch["environment"])
start = time.monotonic()
stop = "native_exit"
with (run / "stdout.log").open("wb") as stdout, (run / "stderr.log").open("wb") as stderr:
    with subprocess.Popen(argv, cwd=launch["cwd"], env=environment, stdout=stdout, stderr=stderr) as process:
        try:
            process.wait(timeout=540)
        except subprocess.TimeoutExpired:
            stop = "time_budget_expired"
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = {"exit": process.returncode, "stop": stop,
                  "wall_seconds": time.monotonic() - start}
(run / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
