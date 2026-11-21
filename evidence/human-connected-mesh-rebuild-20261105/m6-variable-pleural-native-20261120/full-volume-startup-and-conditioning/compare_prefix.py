"""Compare accepted states and retain whole-stage GPU timing for one instrumented root."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

control, candidate = map(Path, sys.argv[1:])

def capture(root):
    out = (root / "stdout.log").read_text()
    err = (root / "stderr.log").read_text()
    receipt = json.loads((root / "execution.json").read_text())
    with (root / "resting-coupled.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = defaultdict(list)
    for name, step, elapsed in re.findall(
            r"^matter_gpu_stage=(\w+) step=(\d+) elapsed_ns=(\d+)", err, flags=re.M):
        values[(int(step),name)].append(int(elapsed))
    stages = [
        dict(root_zero_based=step, stage=name, parts=len(samples), gpu_ms=sum(samples)/1e6)
        for (step,name),samples in sorted(values.items())
    ]
    timings = [float(ms) for ms in re.findall(r"^resting_native_profile step=\d+ .*? gpu_ms=(\S+)", out, flags=re.M)]
    memory = {phase: dict(resident=int(res), footprint=int(fp), metal=int(metal))
              for phase,res,fp,metal in re.findall(
        r"^human_preparation_memory phase=(\w+) available=1 resident_bytes=(\d+) "
        r"footprint_bytes=(\d+) metal_allocated_bytes=(\d+)", err, flags=re.M)}
    hashes = list(root.glob("registered-volume-fractional-mass-*.json"))
    assert len(hashes) == 1
    with hashes[0].open("rb") as stream:
        mass_metadata_sha256 = hashlib.file_digest(stream,"sha256").hexdigest()
    mass = re.search(r"^whole_body_skin_fat_registered_volume_install=.*",out,flags=re.M)
    assert mass
    return dict(receipt=receipt, rows=rows, stages=stages, timings=timings, memory=memory,
                mass_metadata_sha256=mass_metadata_sha256, installed_mass_record=mass.group(),
                terminal_records=[line for line in out.splitlines() if line.startswith(
                    ("resting_fascia_fem_terminal_audit=", "resting_fem_skin_bed_accepted_root_reaction "))],
                timing_warnings=[line for line in err.splitlines() if line.startswith(
                    ("matter_gpu_timing_invalid_records", "matter_gpu_timing_samples_dropped",
                     "matter_gpu_stage_incomplete", "matter_gpu_timing_record_unavailable"))])
a,b = map(capture,(control,candidate))
ar,br = a.pop("rows"),b.pop("rows")
assert len(ar) >= len(br) == 3 and b["receipt"]["exit"] == 0
assert a["installed_mass_record"] == b["installed_mass_record"]
assert not a["stages"]
print(json.dumps(dict(scope="Three-root numerical parity and single-root GPU instrumentation",
                      active_accepted_roots=len(br), exact_coupled_rows=ar[:len(br)]==br,
                      differing_columns=[name for name in br[0] if
                                         any(x[name] != y[name] for x,y in zip(ar,br))],
                      control=a, profiled=b, last_accepted_coupled_row=br[-1]),indent=2))
