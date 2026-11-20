"""Capture native reference assembly before measuring or changing its tetrahedra."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--paired-offset", action="store_true")
parser.add_argument("--neighbors", action="store_true")
args = parser.parse_args()
prefix = ("fascia-neighbors-audit" if args.neighbors else
          "paired-offset-fascia-audit" if args.paired_offset else "registered-fascia-audit")
reference = root.parent / "resting-clearance-and-atrium-015/motion-extension/launch.json"
launch = json.loads(reference.read_text())
local = root / (prefix + "-run")
local.mkdir()
for name, digest in launch["pins"].items():
    with open(name, "rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
native = root / (prefix + "-build") / "numi-human-native"
with native.open("rb") as stream:
    launch["pins"][str(native)] = hashlib.file_digest(stream, "sha256").hexdigest()
argv = launch["argv"]
argv[0] = str(native)
argv[4] = str(local / "native")
argv[argv.index("--muscle-step-count") + 1] = "8"
capture = local / "fascia-rest.csv"
launch["environment"].update(
    NUMI_HUMAN_REGISTERED_FASCIA_AUDIT_CSV=str(capture),
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(local / "coupled.csv"),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(local / "failure.json"),
)
if args.neighbors:
    launch["environment"]["NUMI_HUMAN_REGISTERED_FASCIA_NEIGHBORS"] = str(local/"surrounding.bin")
(local / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
environment = {k:v for k,v in os.environ.items() if not k.startswith(("NM_","NUMI_"))}
environment.update(launch["environment"])
start = time.monotonic()
complete = False
with (local / "stdout.log").open("wb") as stdout,(local / "stderr.log").open("wb") as stderr:
    with subprocess.Popen(argv, cwd=launch["cwd"], env=environment, stdout=stdout, stderr=stderr) as process:
        while process.poll() is None and time.monotonic() - start < 60:
            if capture.exists():
                with capture.open() as stream:
                    complete = sum(1 for _ in stream) > 1200
                if args.neighbors:
                    complete = (complete and (local/"surrounding.bin").exists() and
                                "resting_fascia_fem=admitted" in (local/"stdout.log").read_text())
                if complete:
                    break
            time.sleep(0.05)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        result = {"exit":process.returncode,"completed_registration_capture":complete,
                  "wall_seconds":time.monotonic()-start}
(local / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
assert complete
