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
parser.add_argument("--host-library", type=Path)
parser.add_argument("--roots", type=int)
parser.add_argument("--bed-placement", type=Path)
parser.add_argument("--capture-step", type=int)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--budget-wall-seconds", type=int, default=620)
args = parser.parse_args()
source, metal, run = (p.resolve() for p in (args.source, args.metal, args.run))
if args.capture_step is not None and args.headless:
    parser.error("Geometry capture requires the native visual path")
# Match the native full-volume visual owner before reading the large input
# pins. The native owner still measures the exact serialized pack size.
pack_count = 1 + int(args.capture_step is not None)
disk_floor = 512 * 1024**2
storage = os.statvfs(run.parent)
available = storage.f_bavail * storage.f_frsize
required = 2 * 1024**3 + pack_count * (321299964 + 33554432)
if available < required:
    parser.error(f"Native capture reserve requires {required} bytes; {available} available")
system_disk = os.statvfs(Path.home())
if system_disk.f_bavail * system_disk.f_frsize < disk_floor:
    parser.error("System disk is below the native watchdog floor")
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
if args.host_library:
    previous = Path(overrides["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib"
    replacement = args.host_library.resolve(strict=True)
    if replacement.name != previous.name or str(previous) not in pins:
        raise ValueError("Host overlay requires the same pinned dynamic library name")
    overrides["DYLD_LIBRARY_PATH"] = str(replacement.parent)
    overrides["DYLD_PRINT_LIBRARIES"] = "1"
    del pins[str(previous)]
    pins[str(replacement)] = digest(replacement)
if args.bed_placement:
    receipt = args.bed_placement.resolve(strict=True)
    placement = json.loads(receipt.read_text())
    first = argv.index("--root-pose")+1
    assert list(map(float,argv[first:first+7])) == placement["argv_root_pose"]
    assert "NUMI_HUMAN_FEM_TRIAL_BED_PLANE" not in overrides
    assert "NUMI_HUMAN_FEM_SUPPORT_BARRIER_RELEASE" not in overrides
    plane = [*placement["normal"], placement["offset_m"]]
    overrides["NUMI_HUMAN_FEM_TRIAL_BED_PLANE"] = ",".join(format(value,".17g") for value in plane)
    pins[str(receipt)] = digest(receipt)
root_count = argv.index("--muscle-step-count")+1
if args.roots is not None:
    assert args.roots > 0
    argv[root_count] = str(args.roots)
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
run.mkdir()
(run/"launch.json").write_text(json.dumps(dict(argv=argv,cwd=config["cwd"],
    environment=overrides,pins=pins),indent=2)+"\n")
environment = {key:value for key,value in os.environ.items()
               if not key.startswith(("NM_","NUMI_"))}
environment.update(overrides)
start=time.monotonic()
with (run/"stdout.log").open("wb") as out,(run/"stderr.log").open("wb") as err:
    with subprocess.Popen(argv,cwd=config["cwd"],env=environment,stdout=out,stderr=err) as proc:
        stop = "native_exit"
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                v = os.statvfs(run.parent)
                if elapsed >= args.budget_wall_seconds:
                    raise subprocess.TimeoutExpired(argv, elapsed)
                system_disk = os.statvfs(Path.home())
                if (v.f_bavail * v.f_frsize < disk_floor or
                    system_disk.f_bavail * system_disk.f_frsize < disk_floor):
                    stop = "disk_reserve"
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
            if stop == "native_exit" and proc.returncode < 0:
                stop = "external_signal"
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
