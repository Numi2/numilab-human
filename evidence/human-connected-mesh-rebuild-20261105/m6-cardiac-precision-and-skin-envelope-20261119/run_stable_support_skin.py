from argparse import ArgumentParser
import json
import os
import subprocess
import time
from pathlib import Path

import sys
HERE = Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-004"
sys.path.insert(0, str(PREVIOUS))
from run_segment_control import digest

def main():
    parser=ArgumentParser()
    parser.add_argument("trial",choices=("preflight","full"))
    args=parser.parse_args()
    baseline=json.loads((PREVIOUS/"native-segment32-control/launch.json").read_text())
    for p,expected in baseline["pins"].items():
        assert digest(p)==expected,p
    full=args.trial=="full"
    kind=os.environ.get("NUMI_HUMAN_TRIAL_SKIN_SELF","1")
    assert kind in ("1","2")
    stem="skin-stable-support" if kind=="1" else "skin-live-pinned"
    out=HERE/(stem+("-16s" if full else "-704"))
    if full:
        short=HERE/(stem+"-704")
        assert json.loads((short/"execution.json").read_text())["exit_code"]==0
        for file in ("resting-coupled.csv","resting-com-support-impulses.csv",
                      "resting-com-momentum-diagnostic.csv"):
            expected=(PREVIOUS/"native-segment32-control/native"/file).read_bytes()
            observed=(short/"native"/file).read_bytes()
            control=expected.splitlines()
            trial=observed.splitlines()
            # The terminal observer row includes a capture-mode marker.
            assert control[:len(trial)-1]==trial[:-1],file
    work=HERE/(stem+"-build")
    verified=json.loads((work/"build-verification.json").read_text())
    assert digest(verified["binary"])==verified["sha256"]
    baseline["argv"][0]=verified["binary"]
    baseline["argv"][4]=str(out/"native")
    baseline["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-failure.json")
    baseline["pins"][verified["binary"]]=verified["sha256"]
    skin=HERE/(("skin-self-native" if kind=="1" else "skin-live-pinned-native")+".nhskin")
    baseline["argv"][baseline["argv"].index("--skin-payload")+1]=str(skin)
    baseline["pins"][str(skin)]=digest(skin)
    assert baseline["pins"][str(skin)]==json.loads((skin.with_name(skin.stem+"-payload.json")).read_text())["candidate_sha256"]
    if not full:
        baseline["argv"][baseline["argv"].index("--muscle-step-count")+1]="704"
        baseline["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="704"
        baseline["mode"]["steps"]=704
        baseline["mode"]["capture"]="704"
    if kind=="2":
        baseline["environment"].pop("NUMI_HUMAN_RESTING_CAPTURE_REPLAY_MAPPING",None)
    baseline["mode"]["output"]=out.name
    baseline["design"]["experiment"]="Stable skin support region identity with live corrected surface positions"
    out.mkdir()
    (out/"launch.json").write_text(json.dumps(baseline,indent=2)+"\n")
    env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    t=time.monotonic()
    with (out/"stdout.log").open("wb") as stream,(out/"stderr.log").open("wb") as error:
        trial=subprocess.run(baseline["argv"],cwd=baseline["cwd"],env=env|baseline["environment"],
                             stdout=stream,stderr=error,timeout=330)
    receipt={"exit_code":trial.returncode,"seconds_process":time.monotonic()-t,"guard_stop":None}
    (out/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)
    raise SystemExit(trial.returncode)

if __name__=="__main__":
    main()
