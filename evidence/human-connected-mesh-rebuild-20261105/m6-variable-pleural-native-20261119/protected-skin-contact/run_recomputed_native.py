"""Replay the protected-patch native scene with its recompiled bed-contact sections."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_support_dependency_preflight import HERE, read, write, digest

source=HERE/"recomputed-support-preflight"
timing_only="--timing-only" in sys.argv
out=HERE/("recomputed-support-timing" if timing_only else "recomputed-support-native")
out.mkdir()
cache=source/"native/resting-human.mrvpack"
cfg=read(source/"launch.json")
args=cfg["argv"]
args[4]=str(out/"native")
args[args.index("--muscle-step-count")+1]="8000"
e=cfg["environment"]
e.pop("NUMI_HUMAN_TEST_BED_SECTION_FOOTPRINT",None)
e.update(NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS="8000",
         NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR=str(source/"native"),
         NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(cache),
         NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=digest(cache),
         NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(out/"cardiac-failure.json"))
if timing_only:
    e.pop("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS")
    e.pop("NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR")
assert e["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
cfg["pins"][str(cache)]=e["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"]
for path,h in cfg["pins"].items():
    assert digest(Path(path))==h,path
write(out/"launch.json",cfg)
env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
t=time.monotonic()
with (out/"stdout.log").open("wb") as stdout,(out/"stderr.log").open("wb") as stderr:
    try:
        p=subprocess.run(args,cwd=cfg["cwd"],env=env|e,
                         stdout=stdout,stderr=stderr,timeout=380)
        code=p.returncode
    except subprocess.TimeoutExpired:
        code=None
result=dict(wall_seconds=time.monotonic()-t,exit_code=code,requested_steps=8000)
write(out/"execution.json",result)
print(json.dumps(result),flush=True)
if timing_only and code==0:
    reference=HERE/"recomputed-support-native"
    traces=("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv",
            "resting-com-momentum-diagnostic.csv")
    same={n:digest(out/"native"/n)==digest(reference/"native"/n) for n in traces}
    result={**result,"motion_export_repeated":False}
    def get_state_and_presentation(run):
        lines=(run/"stdout.log").read_text().splitlines()
        terminal=[json.loads(line.split("=",1)[1]) for line in lines if line.startswith("stand_terminal_state=")]
        timing=[line for line in lines if line.startswith("resting_integrated_throughput ")]
        frames=[line for line in lines if line.startswith("resting_present_profile ")]
        assert len(terminal)==len(timing)==1 and " finished=1 " in frames[-1]
        return terminal[0],timing[0],frames
    base,_,_=get_state_and_presentation(reference)
    state,motion,frames=get_state_and_presentation(out)
    same["terminal_state"]=state==base
    assert all(same.values()),same
    seconds=read(reference/"summary.json")["timing"]["simulated_seconds"]
    result.update(simulated_seconds=seconds,launch_to_exit_rtf=seconds/result["wall_seconds"],
                  identical_state_and_traces=same,native_presentation_callbacks=len(frames),
                  integrated_motion=motion)
    write(out/"timing.json",result)
sys.exit(1 if code is None else code)
