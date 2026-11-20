"""Run matched short native regional FGMRES trials on the pinned M6 scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pairs",type=int,default=3)
parser.add_argument("--steps",type=int,default=4)
parser.add_argument("--out",default="fgmres-pair")
args=parser.parse_args()
local=root/args.out
local.mkdir()
source=root.parent/"resting-clearance-and-atrium-015/motion-extension/launch.json"
input_launch=json.loads(source.read_text())
for path,digest in input_launch["pins"].items():
    with open(path,"rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==digest,path

enable={
    "NM_FGMRES_DISTRIBUTED_CGS2":"1",
    "NM_FGMRES_CGS2_ICB":"1",
    "NM_FEM_DISTRIBUTED_GLOBAL_MEASUREMENTS":"1",
}
bad=set()
for paired_index in range(args.pairs):
    for variant in (("control","distributed") if paired_index%2==0 else ("distributed","control")):
        if variant in bad:
            continue
        folder=local/f"{paired_index}-{variant}"
        folder.mkdir()
        launch=json.loads(json.dumps(input_launch))
        argv=launch["argv"]
        argv[4]=str(folder/"native")
        argv[argv.index("--muscle-step-count")+1]=str(args.steps)
        mod=launch["environment"]
        mod["NUMI_HUMAN_COUPLED_TRACE_CSV"]=str(folder/"coupled.csv")
        mod["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(folder/"failure.json")
        if variant=="distributed":
            assert set(enable).isdisjoint(mod)
            mod.update(enable)
        (folder/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
        environment={k:v for k,v in os.environ.items() if not k.startswith(("NM_","NUMI_"))}
        environment.update(mod)
        start=time.monotonic()
        stop="native_exit"
        with (folder/"stdout.log").open("wb") as stdout,(folder/"stderr.log").open("wb") as stderr:
            with subprocess.Popen(argv,cwd=launch["cwd"],env=environment,stdout=stdout,stderr=stderr,
                                  start_new_session=True) as process:
                try:
                    process.wait(timeout=140)
                except subprocess.TimeoutExpired:
                    stop="time_budget_expired"
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                result={"exit":process.returncode,"status":stop,"seconds":time.monotonic()-start}
        (folder/"execution.json").write_text(json.dumps(result,indent=2)+"\n")
        print(folder.name,result,flush=True)
        if result["exit"]!=0:
            bad.add(variant)
