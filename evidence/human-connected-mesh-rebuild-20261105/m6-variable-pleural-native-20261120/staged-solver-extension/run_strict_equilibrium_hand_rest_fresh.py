"""Run the same coupled native case with the staged solver work budget."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import struct
import sys
import time

here = Path.cwd()
baseline = here.parent / "resting-clearance-and-atrium-014/certified-budget-edge-compensated-complete-audit-steps18"
cases = {"strict-equilibrium-hand-rest-fresh": (1, 90)}
variant = sys.argv[1]
steps, deadline_seconds = cases[variant]
run = here / variant
run.mkdir()
launch = json.loads((baseline/"launch.json").read_text())
for name, expected in launch["pins"].items():
    with open(name, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
binary = here/"equilibrium-witness-build/numi-human-native"
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
# Use one full correction of the existing hand passive-stiffness block.
# This changes only the initial pose. The native fit recaptures all support,
# recomputes muscle recruitment, and applies the original acceptance gates.
equilibrium = json.loads((here/"strict-equilibrium-witness/residual-owners.json").read_text())
force = {row["velocity_index"]: row["force"]
         for row in equilibrium["residual_rows_by_absolute_acceleration"]}
rigid = Path(launch["argv"][1]).read_bytes()
assert hashlib.sha256(rigid).hexdigest() == equilibrium["rigid_sha256"]
_, _, _, body_count, joint_count, _, _ = struct.unpack_from("<7I", rigid, 8)
dof_start = 224 + 160 * body_count + 144 * joint_count
stiffness_flexion, stiffness_deviation, coupling = 1.28, 1.74, -0.18
determinant = stiffness_flexion * stiffness_deviation - coupling ** 2
pose_changes = []
for start in (40, 78):
    fr_dev, fr_flex = force[start], force[start + 1]
    delta = [
        (stiffness_flexion * fr_dev - coupling * fr_flex) / determinant,
        (stiffness_deviation * fr_flex - coupling * fr_dev) / determinant,
        *(force[start + i + 2] / stiffness
          for i, stiffness in enumerate((0.15, 0.15, 0.10, 0.05))),
    ]
    for i, displacement in enumerate(delta):
        v = start + i
        dof = struct.unpack_from("<4I12f", rigid, dof_start + 64 * v)
        q, native_v = dof[2:4]
        assert q == v + 1 and native_v == v
        lower, upper = dof[8:10]
        owner = next(index + 2 for index in range(len(launch["argv"]) - 2)
                     if launch["argv"][index : index + 2] == ["--pose-q", str(q)])
        previous = float(launch["argv"][owner])
        candidate = max(lower, min(upper, previous + displacement))
        assert abs(candidate - previous) < 0.25
        launch["argv"][owner] = str(candidate)
        pose_changes.append({"velocity_index": v, "coordinate_index": q,
                             "old_rad": previous, "new_rad": candidate,
                             "raw_correction_rad": displacement})
(run/"hand-rest-pose-changes.json").write_text(json.dumps(pose_changes, indent=2) + "\n")
# A modified initial pose needs its own authenticated visual/contact pack.
# Keep the content verification active and rebuild from the same sources.
import shutil
assert shutil.disk_usage(here).free >= 600 * 1024 * 1024
launch["environment"].pop("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE")
launch["environment"].pop("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256")
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
