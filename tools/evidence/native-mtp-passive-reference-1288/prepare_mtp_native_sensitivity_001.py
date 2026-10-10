from pathlib import Path
import sys,json,hashlib,argparse,subprocess
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
H=Path("/Users/n/numi-human-local-clearance-preservation-1280")
L=Path("/Users/n/numi-lab-mtp-passive-1281")
BUILD=Path("/Users/n/numi-lab-mtp-passive-build-1281-001")
sys.path.insert(0,str(H/"src"))
from numilab_human import resting_run as rr
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument("--stiffness",type=float,required=True);parser.add_argument("--suffix",required=True)
opt=parser.parse_args()
assert opt.stiffness in (0.,.5,1.,2.)
O=A/("mtp-passive-native-sensitivity-"+opt.suffix);O.mkdir(exist_ok=False)
B=A/"twenty-six-surface-native-composition-001"
bp=B/"baseline/run-declaration.json";base=json.loads(bp.read_text())
idx=base["argv"].index("numilab_human.resting_run")
av=base["argv"][idx+1:]
for flag,value in [("--output",str(O/"baseline/native-run")),("--lab",str(L)),("--build",str(BUILD))]:
 av[av.index(flag)+1]=value
av+=["--mtp-passive-stiffness-nm-per-rad",str(opt.stiffness)]
parser=argparse.ArgumentParser();rr.add_arguments(parser);args=parser.parse_args(av)
native,assets=rr.command(args)
env=dict(x.split("=",1) for x in base["argv"][2:idx-2])
env.update(PYTHONPATH=str(H/"src"),NUMI_HUMAN_ROOT="/Users/n/numi-human-positive-winding-1279",NUMI_LAB_ROOT=str(L),NUMI_BUILD_DIR=str(BUILD),
 DYLD_LIBRARY_PATH=str(BUILD/"lib")+":"+str(BUILD/"matter"),
 NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS="0,8000",
 NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT="1",
 NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP="7001",
 NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP="8000",
 NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(args.output/"common-field-failure.json"))
argv=["/usr/bin/env","-i"]+[k+"="+v for k,v in sorted(env.items())]+["/Users/n/numi-human-prep-venv-20261005/bin/python3.13","-m","numilab_human.resting_run"]+av
launch=O/"launch_arm.py"
# Existing immutable launcher, bounded storage guard adjusted for only two captures.
launch.write_text((B/"launch_arm.py").read_text().replace("6000 * 1024 * 1024","4000 * 1024 * 1024"))
sources=[L/"apps/numilab_human_myosim_visual_probe.mm",L/"include/metalrobo/NumiHumanPassiveJoint.hpp",L/"tests/numi_human_passive_joint_test.cpp",H/"tests/test_resting_run.py",Path(rr.__file__),Path(__file__),bp,launch]
pins={**assets,**{str(p):sha(p) for p in sources}}
revisions={str(r):subprocess.check_output(["git","-C",str(r),"rev-parse","HEAD"],text=True).strip() for r in [H,L,Path(env["NUMI_HUMAN_ROOT"]),Path("/Users/n/numi-human-resting-integration-20261005/numi-brain")]}
for r in (H,L):
 diff=O/(r.name+".patch");diff.write_bytes(subprocess.check_output(["git","-C",str(r),"diff","--binary"]));pins[str(diff)]=sha(diff)
decl={"schema":"numi.human.native-smoke.run-declaration.v1","seconds":16.0,"accepted_steps":8000,"capture_steps":[0,8000],"mtp_passive_stiffness_nm_per_rad":opt.stiffness,
 "source_revisions":revisions,"immutable_assets":pins,"argv":argv,"launch_adapter":{"path":str(launch),"sha256":sha(launch)},
 "owner_cli_preview":{"native_argv":native,"asset_sha256":assets,"recorded_invocation_environment":rr.invocation_environment(env)},
 "qualification":"Bounded anatomical-mechanics sensitivity on qualified26 geometry, original flat contact plane and coupled physiology. Explicit aggregate linear spring at each shared MTP coordinate through existing GPU implicit passive joint owner; neutral zero radians. Values are reference sensitivity parameters, not measured five-ray stiffness. New source patches bound beside revisions. Zero is compatibility control. 16s does not establish final body or five-minute qualification."}
(O/"baseline").mkdir();dp=O/"baseline/run-declaration.json";dp.write_text(json.dumps(decl,indent=2,sort_keys=True)+"\n")
(O/"launch-guard.json").write_text(json.dumps({"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":str(args.output)}}},indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(O),"stiffness":opt.stiffness,"pins":len(pins),"native":native[0]}))

