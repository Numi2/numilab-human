"""Replay the first divergent support step with the existing selector observer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-004"
sys.path.insert(0,str(PREVIOUS))
from run_segment_control import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=("control","candidate"))
    args=p.parse_args()
    candidate=args.kind=="candidate"
    launch=(HERE/"skin-stable-support-704" if candidate else PREVIOUS/"native-segment32-control")/"launch.json"
    run=json.loads(launch.read_text())
    for path,fingerprint in run["pins"].items(): assert digest(path)==fingerprint
    out=HERE/f"selector-{args.kind}"
    out.mkdir()
    run["argv"][4]=str(out/"native")
    run["argv"][run["argv"].index("--muscle-step-count")+1]="624"
    run["environment"]["NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT"]="1"
    run["environment"]["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"]="1"
    run["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-failure.json")
    for key in ("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS","NUMI_HUMAN_RESTING_CAPTURE_REPLAY_MAPPING"):
        run["environment"].pop(key,None)
    run["mode"].update(steps=624, capture=None, output=out.name)
    run["original_launch_sha256"]=digest(launch)
    (out/"launch.json").write_text(json.dumps(run,indent=2)+"\n")
    env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    start=time.monotonic()
    with (out/"stdout.log").open("wb") as log,(out/"stderr.log").open("wb") as err:
        completed=subprocess.run(run["argv"],cwd=run["cwd"],env=env|run["environment"],stdout=log,stderr=err,timeout=300)
    receipt=dict(exit_code=completed.returncode,wall_seconds=time.monotonic()-start)
    (out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)
    raise SystemExit(completed.returncode)

if __name__=="__main__":
    main()
