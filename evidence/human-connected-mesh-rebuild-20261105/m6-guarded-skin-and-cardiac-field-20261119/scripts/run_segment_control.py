"""Run the existing native 16-second control with a different observer batch size."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"resting-clearance-and-atrium-003"
SOURCE=PRIOR/"wider-release-control-16/launch.json"
KEY="NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"
CACHE_KEY="NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"
CACHE_SHA_KEY="NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"

def digest(path):
    with open(path,"rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cap",type=int,choices=[16,32],nargs="?",default=16)
    treatment=parser.add_mutually_exclusive_group()
    treatment.add_argument("--initial-pack-cache",action="store_true")
    treatment.add_argument("--guarded-skin",action="store_true")
    args=parser.parse_args()
    cap=args.cap
    if args.initial_pack_cache and cap!=32:
        parser.error("the cached qualification uses the 32-step observer cap")
    suffix="-cache" if args.initial_pack_cache else "-skin" if args.guarded_skin else ""
    outdir=HERE/f"native-segment{cap}-control{suffix}"
    controls=json.loads(SOURCE.read_text())
    assert controls["mode"]["steps"]==8000 and controls["mode"]["capture"]=="8000"
    assert controls["environment"][KEY]=="8"
    assert controls["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="0"
    for p,expected in controls["pins"].items():
        assert digest(p)==expected,p
    assert shutil.disk_usage(HERE).free>6*2**30
    controls["argv"][4]=str(outdir/"native")
    controls["environment"][KEY]=str(cap)
    controls["environment"]["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(outdir/"common-failure.json")
    if args.initial_pack_cache:
        reference=PRIOR/"wider-release-control-16/native"
        source_pack=reference/"resting-human.mrvpack"
        previous=json.loads((reference/"accepted-geometry/step-8000.receipt.json").read_text())
        pack_sha=previous["base_pack_file_sha256"]
        assert digest(source_pack)==pack_sha
        controls["environment"].update({CACHE_KEY:str(source_pack),CACHE_SHA_KEY:pack_sha})
        built=json.loads((HERE/"pack-cache-build/build-verification.json").read_text())
        assert digest(built["binary"])==built["sha256"]
        controls["argv"][0]=built["binary"]
        controls["pins"].update({built["binary"]:built["sha256"],str(source_pack):pack_sha})
    if args.guarded_skin:
        skin=HERE/"guarded-skin.nhskin"
        authored=json.loads((HERE/"guarded-skin-payload.json").read_text())
        assert digest(skin)==authored["candidate_sha256"]
        assert authored["support_anchors_byte_equal"]==32 and not authored["new_source_pairs"]
        built=json.loads((HERE/"guarded-skin-build/build-verification.json").read_text())
        assert digest(built["binary"])==built["sha256"]
        controls["argv"][0]=built["binary"]
        controls["argv"][controls["argv"].index("--skin-payload")+1]=str(skin)
        controls["pins"][str(skin)]=authored["candidate_sha256"]
        controls["pins"][built["binary"]]=built["sha256"]
    controls["mode"]["output"]=outdir.name
    controls["design"]["experiment"]=f"Observer segment cap {cap}, with the same 2 ms physics and view cadence"
    controls["control_receipt_sha256"]=digest(SOURCE)
    outdir.mkdir()
    (outdir/"launch.json").write_text(json.dumps(controls,indent=2)+"\n")
    env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    start=time.monotonic()
    with (outdir/"stdout.log").open("wb") as out,(outdir/"stderr.log").open("wb") as err:
        proc=subprocess.Popen(controls["argv"],cwd=controls["cwd"],env=env|controls["environment"],
                              stdout=out,stderr=err)
        stop=None
        while True:
            try:
                status=proc.wait(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic()-start>330 or shutil.disk_usage(HERE).free<5*2**30:
                    stop="wall or disk guard"
                    proc.terminate()
                    status=proc.wait(timeout=20)
                    break
    report={"seconds_process":time.monotonic()-start,"exit_code":status,"guard_stop":stop}
    (outdir/"execution.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)
    raise SystemExit(status)

if __name__=="__main__":
    main()
