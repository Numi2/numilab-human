"""Run one immutable native request and retain failed as well as passing evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("request", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
request = json.loads(args.request.read_text())
args.output.mkdir(exist_ok=False)

def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

def sources():
    return {p: identity(p) for p in request["sources"]}

def artifacts():
    paths = set(request.get("binaries", []))
    paths.update(v for v in request["environment"].values() if Path(v).is_file())
    return {p: identity(p) for p in sorted(paths)}

def git_state():
    result = {}
    for name, root in request["repositories"].items():
        def git(*argv):
            return subprocess.check_output(["git", "-C", root, *argv], text=True).strip()
        result[name] = {"revision": git("rev-parse", "HEAD"), "status": git("status", "--short")}
    return result

record = {"schema": "numilab-human.native-check-execution.v1", "request": request,
          "request_identity": identity(args.request), "runner_identity": identity(__file__),
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    record.update(source_before=sources(), artifacts_before=artifacts(), git_before=git_state())
except (OSError, subprocess.CalledProcessError) as error:
    record.update(status="not_run", preflight_error=str(error), physical_process_started=False)
    (args.output / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(str(error))
inventory = subprocess.check_output(["ps", "-axo", "pid,ppid,pcpu,etime,comm"], text=True)
record["workload_process_inventory"] = [line for line in inventory.splitlines()
    if any(name in line.lower() for name in ("numi", "metalrobo", "ctest", "swift", "clang", "ninja"))]
environment = dict(os.environ, **request["environment"])
environment["PATH"] = "/opt/homebrew/bin:" + environment.get("PATH", "")
start = time.monotonic()
with (args.output / "run.log").open("wb") as log:
    process = subprocess.Popen(request["command"], cwd=request["cwd"], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=request["timeout_seconds"])
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
record.update(returncode=code, elapsed_seconds=time.monotonic() - start,
              source_after=sources(), artifacts_after=artifacts(), git_after=git_state(),
              log_identity=identity(args.output / "run.log"))
record["source_unchanged"] = record["source_before"] == record["source_after"]
record["artifacts_unchanged"] = record["artifacts_before"] == record["artifacts_after"]
record["status"] = "pass" if code == 0 and record["source_unchanged"] and record["artifacts_unchanged"] else "fail"
(args.output / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in ("status", "returncode", "elapsed_seconds", "source_unchanged", "artifacts_unchanged")}))
raise SystemExit(0 if record["status"] == "pass" else 1)
