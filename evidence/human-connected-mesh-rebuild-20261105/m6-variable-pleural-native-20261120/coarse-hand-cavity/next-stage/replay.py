"""Profile a fixed accepted prefix of the current native Human."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
from time import monotonic
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_hand_trial import digest

parser=ArgumentParser(description=__doc__)
parser.add_argument("output",type=Path)
parser.add_argument("--profile-root",type=int)
parser.add_argument("--steps",type=int,default=4)
parser.add_argument("--skip-completed", action="store_true")
parser.add_argument("--newton", type=int, choices=range(2,25))
parser.add_argument("--adaptive-build",type=Path)
parser.add_argument("--endpoint",action="store_true")
args=parser.parse_args()
output=args.output.resolve()
base=Path(__file__).resolve().parents[1]/"terminal-contact-replay/launch.json"
launch=json.loads(base.read_text())
assert all(digest(path)==sha for path,sha in launch["pins"].items())
assert digest(launch["argv"][0])==launch["native_binary_sha256"]
launch["comparison_launch_sha256"]=digest(base)
argv=launch["argv"]
if args.adaptive_build is not None:
    replacement=args.adaptive_build.resolve()/"numi-human-native"
    launch["pins"].pop(argv[0])
    launch["native_binary_sha256"]=digest(replacement)
    launch["pins"][str(replacement)]=launch["native_binary_sha256"]
    argv[0]=str(replacement)
argv[4]=str(output/"native")
argv[argv.index("--muscle-step-count")+1]=str(args.steps)
env=launch["environment"]
for key in ("NUMI_HUMAN_SKIN_REFERENCE_DUMP","NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP",
            "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP","NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV",
            "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV","NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"):
    env.pop(key,None)
if args.adaptive_build is not None:
    env["NUMI_HUMAN_CERTIFIED_NEWTON_BUDGET_TRIAL"]="1"
if args.endpoint:
    env["NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE"]="1"
if args.profile_root is not None:
    assert 0<=args.profile_root<args.steps
    env.update(NUMI_MATTER_GPU_TIMING="1",NUMI_MATTER_GPU_TIMING_STAGES="1",
               NUMI_MATTER_GPU_TIMING_ROOT=str(args.profile_root))
if args.skip_completed:
    env.update(NM_FGMRES_SKIP_COMPLETED_CYCLE_WORK="1",
               NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS="1")
if args.newton is not None:
    env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"]=str(args.newton)
output.mkdir()
(output/"launch.json").write_text(json.dumps(launch,indent=2)+"\n")
run_env={key:value for key,value in os.environ.items() if not key.startswith(("NM_","NUMI_"))}
run_env.update(env)
begin=monotonic()
with (output/"stdout.log").open("x") as stdout,(output/"stderr.log").open("x") as stderr:
    try:
        result=subprocess.run(argv,env=run_env,cwd=launch["cwd"],stdout=stdout,stderr=stderr,timeout=360)
        status={"exit":result.returncode,"stop":"exit"}
    except subprocess.TimeoutExpired:
        status={"exit":124,"stop":"timeout"}
receipt={**status,"wall_seconds":monotonic()-begin,"profile_root":args.profile_root}
(output/"execution.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
