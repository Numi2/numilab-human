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
parser.add_argument("--pose-fit", type=Path)
parser.add_argument("--capture-step", type=int)
parser.add_argument("--headless", action="store_true")
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
if args.pose_fit:
    receipt = args.pose_fit.resolve()
    fit = json.loads(receipt.read_text())
    first = argv.index("--root-pose")+1
    assert list(map(float,argv[first:first+7])) == fit["original_root_pose"]
    trial = fit["trial_root_pose"]
    assert len(trial)==7 and fit["tilt_deg"]<1.0
    argv[first:first+7] = [format(value,".17g") for value in trial]
    overrides["NUMI_EXPERIMENT_INITIAL_ROOT_POSE"] = ",".join(argv[first:first+7])
    pins[str(receipt)] = digest(receipt)
root_count = argv.index("--muscle-step-count")+1
if args.roots is not None:
    assert args.roots > 0
    argv[root_count] = str(args.roots)
pack_count=1
if args.headless:
    assert args.capture_step is None
    if "--mechanics-only" not in argv:
        argv.append("--mechanics-only")
    overrides.pop("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", None)
# Keep optional diagnostics separate from the pinned control directory.
for variable in ("NUMI_HUMAN_SKIN_REFERENCE_DUMP", "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"):
    if variable in overrides:
        overrides[variable] = str(run / Path(overrides[variable]).name)
if args.capture_step is not None:
    assert 0 < args.capture_step <= int(argv[root_count])
    if "--mechanics-only" in argv:
        argv.remove("--mechanics-only")
    overrides["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = str(args.capture_step)
    pack_count+=1
# Archived carrier allocations can be absent even though the small launch
# pins all match. Fail before starting Metal or creating a misleading trace.
for role in ("SKIN", "FAT"):
    prefix = f"NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_{role}_"
    receipt = Path(overrides[prefix+"RECEIPT_JSON"])
    if digest(receipt) != overrides[prefix+"RECEIPT_SHA256"]:
        raise ValueError(f"Recorded allocation receipt changed: {receipt}")
    expected = json.loads(receipt.read_text())["flow_csv"]
    files = {Path(value).name: Path(value) for key,value in overrides.items()
             if key.startswith(prefix) and key.endswith("_CSV")}
    if files.keys() != expected.keys():
        raise ValueError(f"Recorded allocation paths disagree with receipt: {role}")
    for name,path in files.items():
        row = expected[name]
        if not path.is_file():
            raise FileNotFoundError(f"Restore the pinned recorded allocation from archive: {path}")
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Recorded allocation changed: {path}")
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
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                v = os.statvfs(run.parent)
                if elapsed >= args.budget_wall_seconds:
                    raise subprocess.TimeoutExpired(argv, elapsed)
                system_disk = os.statvfs(Path.home())
                if (v.f_bavail * v.f_frsize < 536870912 or
                    system_disk.f_bavail * system_disk.f_frsize < 536870912):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
                try:
                    proc.wait(timeout=min(3, args.budget_wall_seconds - elapsed))
                except subprocess.TimeoutExpired:
                    continue
            stop="native_exit" if proc.returncode >= 0 else "disk_reserve_or_signal"
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
