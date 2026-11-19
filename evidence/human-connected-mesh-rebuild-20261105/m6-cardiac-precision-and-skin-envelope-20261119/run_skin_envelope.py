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
    parser.add_argument("trial",choices=("preflight","full","footprint"))
    args=parser.parse_args()
    baseline=json.loads((PREVIOUS/"native-segment32-control/launch.json").read_text())
    for p,expected in baseline["pins"].items():
        assert digest(p)==expected,p
    full=args.trial=="full"
    if full:
        preflight=HERE/"skin-envelope-preflight"
        reference=PREVIOUS/"bed-preflight-baseline"
        assert json.loads((preflight/"execution.json").read_text())["exit_code"]==0
        sections="native/resting-local-bed-sections.csv"
        assert digest(preflight/sections)==digest(reference/sections)==(
            digest(PREVIOUS/"native-segment32-control"/sections))
        receipt="native/accepted-geometry/step-1.receipt.json"
        a,b=[json.loads((directory/receipt).read_text()) for directory in (reference,preflight)]
        assert a["accepted_step"]==b["accepted_step"]==1
        assert a["physical_endpoint"]==b["physical_endpoint"]=="accepted"
        for field in ("accepted_body_state_sha256","accepted_respiration_state_sha256"):
            assert a[field]==b[field],field
    out=HERE/("skin-envelope-16s" if full else "skin-envelope-"+args.trial)
    work=HERE/"skin-envelope-native-build"
    verified=json.loads((work/"build-verification.json").read_text())
    assert digest(verified["binary"])==verified["sha256"]
    baseline["argv"][0]=verified["binary"]
    baseline["argv"][4]=str(out/"native")
    baseline["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"common-failure.json")
    baseline["pins"][verified["binary"]]=verified["sha256"]
    skin=HERE/"skin-envelope-native.nhskin"
    baseline["argv"][baseline["argv"].index("--skin-payload")+1]=str(skin)
    baseline["pins"][str(skin)]=digest(skin)
    assert baseline["pins"][str(skin)]==json.loads((skin.with_name(skin.stem+"-payload.json")).read_text())["candidate_sha256"]
    if not full:
        baseline["argv"][baseline["argv"].index("--muscle-step-count")+1]="1"
        baseline["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="1"
        if args.trial=="footprint":
            baseline["environment"]["NUMI_HUMAN_TEST_BED_SECTION_FOOTPRINT"]="1"
        baseline["mode"]["steps"]=1
        baseline["mode"]["capture"]="1"
    baseline["mode"]["output"]=out.name
    baseline["design"]["experiment"]="Connected exterior envelope replay with unchanged bed dependencies"
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
