"""Replay the pinned integrated release deck with explicit candidate assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
THUMB=HERE.parent/"coupled-thumb-reference-001"

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--steps",type=int,default=8000)
    p.add_argument("--capture",default="")
    p.add_argument("--skin",type=Path,required=True)
    p.add_argument("--original-pressure",action="store_true")
    p.add_argument("--rebind-binary",action="store_true")
    p.add_argument("--capture-reuse",type=Path)
    p.add_argument("--anatomy-receipt",type=Path)
    a=p.parse_args()
    out=HERE/a.output
    assert out.name==a.output and out.parent==HERE and a.steps>0
    assert shutil.disk_usage(HERE).free>(5.75 if a.capture else 5.5)*2**30, "Insufficient startup space for native run"
    deck=json.loads((THUMB/"combined-12s-no-export/launch.json").read_text())
    for path,sha in deck["pins"].items():
        assert digest(path)==sha,path
    argv=deck["argv"]
    if a.rebind_binary:
        built=json.loads((HERE/"build-verification.json").read_text())
        argv[0]=built["binary"]
        assert digest(argv[0])==built["sha256"]
        deck["pins"][argv[0]]=built["sha256"]
    argv[4]=str(out/"native")
    argv[argv.index("--muscle-step-count")+1]=str(a.steps)
    for switch,new in (("--skin-payload",a.skin),("--resting-anatomy-receipt",a.anatomy_receipt)):
        if new:
            absolute=str(new.resolve(strict=True))
            argv[argv.index(switch)+1]=absolute
            deck["pins"][absolute]=digest(absolute)
    env=deck["environment"]
    env.update(NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=a.capture,
               NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(out/"common-failure.json"),
               NM_VASCULAR_STEP_EXTERNAL_PRESSURE="0" if a.original_pressure else "1")
    if a.capture_reuse:
        env["NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR"]=str(a.capture_reuse.resolve(strict=True))
    deck["mode"].update(output=a.output, steps=a.steps, pressure=not a.original_pressure, capture=a.capture)
    deck["design"]={"comparison":"Matched original load-balanced 2 ms release",
                    "gate":"Unchanged native adoption, including declared geometry union and material tolerances",
                    "candidate_skin":str(a.skin.resolve(strict=True)),
                    "pressure":env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]}
    for name in ("MetalRobo.metallib","HumanRestingSkin.metallib"):
        path=THUMB/"metalrobo"/name
        deck["pins"][str(path)]=digest(path)
    out.mkdir()
    (out/"launch.json").write_text(json.dumps(deck,indent=2)+"\n")
    inherited={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    with (out/"stdout.log").open("wb") as so,(out/"stderr.log").open("wb") as se:
        start=time.monotonic()
        process=subprocess.Popen(argv,cwd=deck["cwd"],env=inherited|env,stdout=so,stderr=se)
        stop=None
        while process.poll() is None:
            free=shutil.disk_usage(HERE).free
            if time.monotonic()-start>330:
                stop="wall guard"
            if free<5*2**30:
                stop=f"disk guard ({free/2**30:.3f} GiB)"
            if stop:
                process.terminate()
                break
            time.sleep(.5)
        status=process.wait(timeout=20)
    elapsed=time.monotonic()-start
    report={"exit_code":status,"seconds_process":elapsed,"guard_stop":stop,
            "free_disk_gib_after":shutil.disk_usage(HERE).free/2**30}
    (out/"execution.json").write_text(json.dumps(report,indent=2)+"\n")
    pack=out/"native/resting-human.mrvpack"
    anchor=THUMB/"thumb-16s-resource-retry/native/resting-human.mrvpack"
    if pack.is_file() and anchor.is_file() and digest(pack)==digest(anchor) and not pack.samefile(anchor):
        link=pack.with_suffix(".link")
        os.link(anchor,link)
        os.replace(link,pack)
    print(json.dumps(report),flush=True)
    raise SystemExit(status)
if __name__=="__main__":
    main()
