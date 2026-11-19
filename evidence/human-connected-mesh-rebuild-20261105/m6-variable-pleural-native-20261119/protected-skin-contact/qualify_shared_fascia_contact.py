"""Exercise native fascia-to-body contact on the variable-pressure scene."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_support_dependency_preflight import HERE, RUN, read, write, digest


def main():
    retained_plan=HERE.parent/"stable-interior-native-631/install-motion.plan.json"
    plan=read(retained_plan)
    # The stored plan already pins the exact source fascia payload.
    argv=[]
    def scan(value):
        if isinstance(value,list):
            if "--pectoralis-fascia-payload" in value:
                argv.append(value[value.index("--pectoralis-fascia-payload")+1])
            for item in value:
                scan(item)
        elif isinstance(value,dict):
            for item in value.values():
                scan(item)
    scan(plan)
    assert len(set(argv))==1
    fascia=Path(argv[0])
    key=digest(fascia)
    assert key in retained_plan.read_text(),fascia
    cfg=read(RUN/"launch.json")
    call=cfg["argv"]
    case=HERE/("shared-fascia-"+sys.argv[1])
    case.mkdir()
    call[4]=str(case/"native")
    call[call.index("--muscle-step-count")+1]="32"
    assert "--pectoralis-fascia-payload" not in call
    call.extend(("--pectoralis-fascia-payload",str(fascia)))
    e=cfg["environment"]
    for k in ("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS","NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR"):
        e.pop(k,None)
    e.update(NUMI_HUMAN_RESTING_PASSIVE_FASCIA="1",
             NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(case/"failure.json"))
    if sys.argv[1] in {"32-unreduced","32-fascia-only"}:
        # Partition whitening is defined for the registered reduction only.
        e["NUMI_HUMAN_PARTITIONED_MASS_WHITEN"]="0"
    if sys.argv[1]=="32-fascia-only":
        # Matcher for the absent volumetric skin owner must not be selected.
        # The separately bound fascia body-glide samples remain enabled.
        assert e["NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM"]=="0"
        e.update(NUMI_HUMAN_SKIN_BONE_TRIANGLE_CONTACT="0",
                 NUMI_HUMAN_SKIN_BONE_STATIC_BVH="0")
    assert e["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
    cfg["pins"][str(fascia)]=key
    for path,h in cfg["pins"].items():
        assert digest(Path(path))==h,path
    write(case/"launch.json",cfg)
    env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    t=time.monotonic()
    with (case/"stdout.log").open("wb") as stdout,(case/"stderr.log").open("wb") as stderr:
        try:
            p=subprocess.run(call,cwd=cfg["cwd"],env=env|e,stdout=stdout,stderr=stderr,timeout=85)
        except subprocess.TimeoutExpired:
            code=None
        else:
            code=p.returncode
    r=dict(exit=code,elapsed_seconds=time.monotonic()-t,requested_steps=32,
           timed_out=code is None,fascia_source_sha256=key)
    write(case/"execution.json",r)
    print(r,flush=True)
    appearance=case/"native/resting-human.mrvpack"
    if appearance.exists():
        sys.path.insert(0,str(HERE.parent/"resting-clearance-and-atrium-004"))
        from archive_bed_preflight import archive
        archive(appearance)


if __name__=="__main__":
    main()
