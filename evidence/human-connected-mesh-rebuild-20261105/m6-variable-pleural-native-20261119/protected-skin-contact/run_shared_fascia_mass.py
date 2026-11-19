"""Screen the integrated contact owner with mass-conserving source contributions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_support_dependency_preflight import HERE, read, write, digest

cfg=read(HERE/"shared-fascia-32-fascia-only/launch.json")
skin_cfg=read(HERE/"recomputed-support-preflight/launch.json")
steps=int(sys.argv[1]) if len(sys.argv)>1 else 32
mode=sys.argv[2] if len(sys.argv)>2 else "mass"
assert mode in ("mass","pressure","ligament")
out=HERE/f"shared-fascia-{mode}-{steps}"
out.mkdir()
binary=HERE/f"shared-fascia-{mode}-build/numi-human-native"
args=cfg["argv"]
old_binary=args[0]
args[0]=str(binary)
args[4]=str(out/"native")
args[args.index("--muscle-step-count")+1]=str(steps)
if mode in ("pressure","ligament"):
    # Use the existing granular ligament owner in the shared transaction.
    # The two hip owners are mutually exclusive in the native ABI.
    args[args.index("--resting-hip-capsule-reference")]="--resting-registered-ligaments"
skin_arg=args.index("--skin-payload")+1
old_skin=args[skin_arg]
args[skin_arg]=skin_cfg["argv"][skin_arg]
e=cfg["environment"]
e["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"failure.json")
assert e["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
assert e["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"]=="1"
for old,new in ((old_binary,args[0]),(old_skin,args[skin_arg])):
    cfg["pins"].pop(old)
    cfg["pins"][new]=digest(Path(new))
for path,h in cfg["pins"].items():
    assert digest(Path(path))==h,path
write(out/"launch.json",cfg)
env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
t=time.monotonic()
with (out/"stdout.log").open("wb") as stdout,(out/"stderr.log").open("wb") as stderr:
    try:
        trial=subprocess.run(args,cwd=cfg["cwd"],env=env|e,stdout=stdout,stderr=stderr,
                             timeout=150 if steps<=32 else 400)
        code=trial.returncode
    except subprocess.TimeoutExpired:
        code=None
result=dict(wall_seconds=time.monotonic()-t,exit_code=code,requested_steps=steps)
write(out/"execution.json",result)
print(json.dumps(result),flush=True)
pack=out/"native/resting-human.mrvpack"
if pack.exists():
    sys.path.insert(0,str(HERE.parent/"resting-clearance-and-atrium-004"))
    from archive_bed_preflight import archive
    archive(pack)
sys.exit(1 if code is None else code)
