"""Run the same coupled native case with the staged solver work budget."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

here = Path.cwd()
baseline = here.parent / "resting-clearance-and-atrium-014/certified-budget-edge-compensated-complete-audit-steps18"
cases = {"strict-equilibrium-footprint": (1, 90)}
variant = sys.argv[1]
steps, deadline_seconds = cases[variant]
run = here / variant
run.mkdir()
launch = json.loads((baseline/"launch.json").read_text())
for name, expected in launch["pins"].items():
    with open(name, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
binary = here/"staged-solver-accuracy-build/numi-human-native"
with binary.open("rb") as stream:
    launch["pins"][str(binary)] = hashlib.file_digest(stream, "sha256").hexdigest()
launch["argv"][0] = str(binary)
launch["argv"][4] = str(run/"native")
launch["argv"][launch["argv"].index("--muscle-step-count") + 1] = str(steps)
launch["environment"].update(
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run/"coupled.csv"),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run/"failure.json"),
)
# Run the owning equilibrium fit at full gravity. First-touch validation
# applies to the released pose and is replaced here by strict stance admission.
launch["environment"].pop("NUMI_HUMAN_RESTING_REGISTERED_FIRST_TOUCH")
launch["environment"]["NUMI_HUMAN_SUPPORTED_INITIALIZATION"] = "1"
launch["argv"].remove("--resting-release-initialization")
ramp = launch["argv"].index("--resting-gravity-ramp-steps")
del launch["argv"][ramp:ramp + 2]
# Match the lower-hull stance enclosing the projected body COM.
# Fit only root height, roll, and pitch.
launch["argv"].extend([
    "--support-stance-dof", "2", "0.015",
    "--support-stance-dof", "3", "0.020",
    "--support-stance-dof", "4", "0.020",
    "--support-stance-contact", "1",
    "--support-stance-contact", "4",
    "--support-stance-contact", "5",
])
(run/"launch.json").write_text(json.dumps(launch, indent=2)+"\n")
environment = {key:value for key,value in os.environ.items()
               if not key.startswith(("NM_", "NUMI_"))}
environment.update(launch["environment"])
start = time.monotonic()
reason = "native_exit"
with (run/"stdout.log").open("wb") as stdout, (run/"stderr.log").open("wb") as stderr:
    with subprocess.Popen(launch["argv"], cwd=launch["cwd"], env=environment,
                          stdout=stdout, stderr=stderr) as process:
        try:
            process.wait(timeout=deadline_seconds)
        except subprocess.TimeoutExpired:
            reason = "time_budget_expired"
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = {"exit":process.returncode, "stop":reason,
                  "wall_seconds":time.monotonic()-start}
(run/"execution.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))
