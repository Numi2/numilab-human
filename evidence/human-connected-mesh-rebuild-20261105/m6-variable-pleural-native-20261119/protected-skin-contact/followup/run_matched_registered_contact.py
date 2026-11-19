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
assert mode in ("mass","pressure","ligament","support","rebase","direct","trace","linked","witness","constitutive","load","tail")
out=HERE/f"shared-fascia-{mode}-registered-pose-{steps}"
out.mkdir()
binary=HERE/f"shared-fascia-{'rebase' if mode in ('direct','trace','linked','witness','constitutive','load','tail') else mode}-build/numi-human-native"
args=cfg["argv"]
old_binary=args[0]
args[0]=str(binary)
args[4]=str(out/"native")
args[args.index("--muscle-step-count")+1]=str(steps)
if mode in ("pressure","ligament","support","rebase","direct","trace","linked","witness","constitutive","load","tail"):
    # Use the existing granular ligament owner in the shared transaction.
    # The two hip owners are mutually exclusive in the native ABI.
    args[args.index("--resting-hip-capsule-reference")]="--resting-registered-ligaments"
skin_arg=args.index("--skin-payload")+1
old_skin=args[skin_arg]
args[skin_arg]=skin_cfg["argv"][skin_arg]
# Reuse the earlier supported posture in which the registered catalog was
# actually cooked. The independent 0.18-rad control remains separate.
for index,arg in enumerate(args):
    if arg=="--pose-q" and args[index+1] in ("36","74"):
        assert args[index+2]=="0.18"
        args[index+2]="0.1"
e=cfg["environment"]
e["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"failure.json")
if mode in ("direct","trace","linked","witness","constitutive","load","tail"):
    # The fascia gliding owner uses registered reference planes. The
    # optional row-query acceleration requires the multi-master triangle ABI.
    for suffix in ("ROW_CACHE","AABB_EXCLUSION","EXACT_MOTION_CACHE","COMPACT_QUERIES"):
        e[f"NUMI_HUMAN_CONTACT_{suffix}"]="0"
    cache=HERE/"shared-fascia-rebase-registered-pose-32/native/resting-human.mrvpack"
    cache_hash=digest(cache)
    e.update(NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(cache),
             NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=cache_hash)
    cfg["pins"][str(cache)]=cache_hash
if mode in ("linked","witness","constitutive","load","tail"):
    lib=HERE/"shared-fascia-rebase-build/lib/libmetalrobo.dylib"
    original=Path(e["DYLD_LIBRARY_PATH"])/lib.name
    cfg["pins"].pop(str(original))
    e["DYLD_LIBRARY_PATH"]=str(lib.parent)
    cfg["pins"][str(lib)]=digest(lib)
if mode=="tail":
    e.update(NM_FEM_NEWTON_TAIL_DISPATCH="1", NM_FEM_NEWTON_TAIL_ICB="1")
if mode=="load":
    assert steps>=6
    args.extend(("--resting-gravity-ramp-steps", "4"))
    e["NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE"]="1"
if mode=="constitutive":
    # The final material certificate already evaluates the strict law.
    # Use that same admissibility during trial selection and keep it tiled.
    e["NM_FEM_BACKTRACKING_CONSTITUTIVE_VALIDITY"]="1"
    e["NM_FEM_PARALLEL_BACKTRACKING"]="1"
    e["NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING"]="1"
    e["NM_FEM_COMPLETED_BACKTRACKING_COLLAPSE"]="0"
if mode=="witness":
    # Capture the existing solver's load and failed contact evidence.
    e["NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE"]="1"
    e["NM_FEM_PRIVATE_CONTACT_FAILURE_CAPTURE"]="1"
if mode=="trace":
    probe=HERE/"fascia_signal_trace.dylib"
    e["DYLD_INSERT_LIBRARIES"]=str(probe)
    cfg["pins"][str(probe)]=digest(probe)
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
if pack.exists() and mode not in ("direct","trace","linked","witness","constitutive","load","tail"):
    sys.path.insert(0,str(HERE.parent/"resting-clearance-and-atrium-004"))
    from archive_bed_preflight import archive
    archive(pack)
sys.exit(1 if code is None else code)
