"""Replay the corrected hand pose with a screened, hash-verified footprint."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

here = Path.cwd()
before = here / "strict-equilibrium-hand-rest-fresh"
run = here / "strict-equilibrium-hand-rest-rescreen"
run.mkdir()
launch = json.loads((before / "launch.json").read_text())
pack = before / "native/resting-human.mrvpack"
with pack.open("rb") as stream:
    pack_hash = hashlib.file_digest(stream, "sha256").hexdigest()
launch["pins"][str(pack)] = pack_hash
for name, expected in launch["pins"].items():
    with open(name, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, name
screen = json.loads((here / "hand-rest-support-stance-screen.json").read_text())
pick = screen["qualified_lower_hull_stances"][0]
assert pick["rows"] == [1, 4, 8] and min(pick["center_of_mass_barycentric"]) > 0
argv = launch["argv"]
argv[4] = str(run / "native")
seen = 0
for index in range(len(argv)-1):
    if argv[index] == "--support-stance-contact":
        argv[index+1] = str(pick["rows"][seen])
        seen += 1
assert seen == 3
launch["environment"].update(
    NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(pack),
    NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=pack_hash,
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run / "coupled.csv"),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run / "failure.json"),
)
(run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("NM_", "NUMI_"))}
environment.update(launch["environment"])
start = time.monotonic()
reason = "native_exit"
with (run / "stdout.log").open("wb") as stdout, (run / "stderr.log").open("wb") as stderr:
    with subprocess.Popen(argv, cwd=launch["cwd"], env=environment, stdout=stdout, stderr=stderr) as process:
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            reason = "time_budget_expired"
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = {"exit": process.returncode, "stop": reason,
                  "wall_seconds": time.monotonic() - start}
(run / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
