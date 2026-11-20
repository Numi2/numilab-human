"""Replay an exact runtime receipt with one tested Metal library replacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
for key in ("source", "metal", "run"):
    parser.add_argument(key, type=Path)
parser.add_argument("--roots", type=int)
parser.add_argument("--capture-step", type=int)
parser.add_argument("--budget-wall-seconds", type=int, default=620)
args=parser.parse_args()
source,metal,run = (p.resolve() for p in (args.source,args.metal,args.run))
config = json.loads((source/"launch.json").read_text())
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
for file, sha in config["pins"].items():
    if digest(file)!=sha:
        raise ValueError(f"Pinned input changed: {file}")
def relocate(value):
    return value.replace(str(source),str(run))
argv = [relocate(value) for value in config["argv"]]
overrides = {key:relocate(value) for key,value in config["environment"].items()}
old = overrides["NUMI_MATTER_METALLIB_OVERRIDE"]
overrides["NUMI_MATTER_METALLIB_OVERRIDE"] = str(metal)
pins = {key:value for key,value in config["pins"].items() if key!=old}
pins[str(metal)] = digest(metal)
root_count = argv.index("--muscle-step-count")+1
if args.roots is not None:
    assert args.roots > 0
    argv[root_count] = str(args.roots)
pack_count=1
if args.capture_step is not None:
    assert 0 < args.capture_step <= int(argv[root_count])
    argv.remove("--mechanics-only")
    overrides["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = str(args.capture_step)
    pack_count+=1
storage = os.statvfs(run.parent)
required = 2147483648 + pack_count*(321299964+33554432)
assert storage.f_bavail*storage.f_frsize >= required, (
    f"Below this scene's {required} byte native pack reserve")
run.mkdir()
(run/"launch.json").write_text(json.dumps(dict(argv=argv,cwd=config["cwd"],
    environment=overrides,pins=pins),indent=2)+"\n")
environment = {key:value for key,value in os.environ.items()
               if not key.startswith(("NM_","NUMI_"))}
environment.update(overrides)
start=time.monotonic()
with (run/"stdout.log").open("wb") as out,(run/"stderr.log").open("wb") as err:
    with subprocess.Popen(argv,cwd=config["cwd"],env=environment,stdout=out,stderr=err) as proc:
        try:
            proc.wait(timeout=args.budget_wall_seconds)
            stop="native_exit"
        except subprocess.TimeoutExpired:
            stop="bounded_wall_budget_expired"
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
result=dict(exit=proc.returncode,stop=stop,wall_seconds=time.monotonic()-start)
(run/"execution.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
