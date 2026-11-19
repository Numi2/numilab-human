"""Run the complete reference scene with its original variable pleural loading."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from fit_cardiac_volume_grid import hash_file

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"resting-clearance-and-atrium-003/wider-release-variable-16/launch.json"
BUILD=HERE/"cardiac-split-tube-native-build"


def launch(args):
    config=json.loads(SOURCE.read_text())
    out=HERE/f"cardiac-split-native-{args.steps}{args.suffix}"
    out.mkdir()
    for path,expected in config["pins"].items():
        assert hash_file(Path(path))==expected,path
    argv=config["argv"]
    build=BUILD if not args.skin_trial else HERE/(args.skin_trial+"-build")
    if args.skin_trial:
        payload=HERE/(args.skin_trial+".nhskin")
        proof=json.loads((HERE/(args.skin_trial+"-payload.json")).read_text())
        assert hash_file(payload)==proof["candidate_sha256"]
        argv[argv.index("--skin-payload")+1]=str(payload)
        config["pins"][str(payload)]=proof["candidate_sha256"]
    if args.guarded_skin:
        soil=HERE.parent/"resting-clearance-and-atrium-004/guarded-footprint.nhskin"
        expected="07b518d2d49222596060105ea1f7385ab730737af7721a0f6196fd320d6f2862"
        assert hash_file(soil)==expected
        argv[argv.index("--skin-payload")+1]=str(soil)
        config["pins"][str(soil)]=expected
    if args.reuse_verified_exports:
        reference=SOURCE.parent/"native/resting-human.mrvpack"
        baseline=HERE/"cardiac-split-native-8000/native"
        expected="ebed4b45012bb024958d7cfbba9097446215d3a4c4ba48bcb2c77c14f75069f3"
        assert hash_file(reference)==expected
        config["pins"][str(reference)]=expected
        config["environment"].update(
            NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(reference),
            NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=expected,
            NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR=str(baseline))
    env=config["environment"]
    env["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS"]=str(args.segment)
    assert env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
    if args.rollback_probe:
        env["NUMI_HUMAN_RESTING_TRANSACTION_PROBE"]="1"
    argv[0]=str(build/"numi-human-native")
    argv[4]=str(out/"native")
    argv[argv.index("--muscle-step-count")+1]=str(args.steps)
    if args.capture:
        env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]=str(args.capture)
    else:
        env.pop("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",None)
    env.pop("NUMI_HUMAN_RESTING_CAPTURE_REPLAY_MAPPING",None)
    env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(out/"cardiac-failure.json")
    for p in (argv[0],build/"metalrobo/HumanRestingSkin.metallib",build/"tube-contract.json"):
        config["pins"][str(p)]=hash_file(Path(p))
    config["mode"].update(steps=args.steps,capture=args.capture,output=out.name)
    config["derived_from"]=hash_file(SOURCE)
    (out/"launch.json").write_text(json.dumps(config,indent=2)+"\n")
    environment={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    before=time.monotonic()
    with (out/"stdout.log").open("wb") as stdout,(out/"stderr.log").open("wb") as stderr:
        proc=subprocess.run(argv,cwd=config["cwd"],env=environment|env,
                            stdout=stdout,stderr=stderr,timeout=330)
    elapsed=time.monotonic()-before
    summary=dict(exit_code=proc.returncode,wall_seconds=elapsed,requested_steps=args.steps,
                 variable_pleural_pressure=True)
    (out/"execution.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)
    return proc.returncode


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("steps",type=int)
    cli.add_argument("--capture",type=int)
    cli.add_argument("--suffix",default="")
    cli.add_argument("--segment",type=int,choices=(8,16,32),default=32)
    cli.add_argument("--rollback-probe",action="store_true")
    startup=cli.add_mutually_exclusive_group()
    startup.add_argument("--reuse-verified-exports",action="store_true")
    startup.add_argument("--guarded-skin",action="store_true")
    startup.add_argument("--skin-trial",choices=("trimmed-guarded-pressure-skin","trimmed-oriented-pressure-skin"))
    args=cli.parse_args()
    if args.suffix and (not args.suffix.startswith("-") or not args.suffix[1:].replace("-","").isalnum()):
        cli.error("--suffix must be a dash followed by letters, digits or dashes")
    assert 1<=args.steps<=8000 and (args.capture is None or 1<=args.capture<=args.steps)
    raise SystemExit(launch(args))
