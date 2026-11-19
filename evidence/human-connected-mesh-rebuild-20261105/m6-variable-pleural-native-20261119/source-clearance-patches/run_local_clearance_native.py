"""Exercise hash-admitted source patches in the original native simulation."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from register_tissue_bellies import B, HERE, hash_file, install_for_float32
from qualify_skin_connectivity import archive, read, write

BASE=HERE/"connected-skin-build"
BUILD=HERE/"source-clearance-patches-build"
OUT=HERE/"source-clearance-patches-native"
REF=HERE/"connected-skin-native"
SOURCE=HERE/"source-clearance-patches.nhskin"
NATIVE=OUT/"native"
APP="numi-human-native"
RECEIPT="accepted-geometry/step-8000.receipt.json"
POSES=(2015,8000)
assert (POSES[0]+1)%32==0


def build():
    proof=read(HERE/"source-clearance-patches-emission.json")
    assert proof["candidate_sha256"]==hash_file(SOURCE)
    assert not proof["new_source_self_pairs"] and proof["protected_rows"]==9588
    BUILD.mkdir()
    visual="NumiHumanRestingVisual.hpp"
    unit="numilab_human_myosim_visual_probe.mm"
    for f in (*BASE.glob("*.hpp"), BASE/unit):
        if f.name!=visual:
            os.link(f,BUILD/f.name)
    source=(BASE/visual).read_text()
    old=f'retainedSourceSkinSHA256 == "{proof["source_sha256"]}") &&'
    new=f'retainedSourceSkinSHA256 == "{proof["source_sha256"]}" ||\n                     retainedSourceSkinSHA256 == "{proof["candidate_sha256"]}") &&'
    assert source.count(old)==1
    (BUILD/visual).write_text(source.replace(old,new))
    mapping={str(BASE/n):str(BUILD/n) for n in (unit,APP+".o",APP)}
    commands=[[mapping.get(a,a) for a in cmd] for cmd in read(BASE/"build-commands.json")]
    write(BUILD/"build-commands.json",commands)
    cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native"
    for cmd in commands:
        subprocess.run(cmd,cwd=cwd,check=True)
    (BUILD/"metalrobo").mkdir()
    for f in (BASE/"metalrobo").iterdir():
        os.link(f,BUILD/"metalrobo"/f.name)
    write(BUILD/"build-verification.json",
          {str(f.relative_to(BUILD)):hash_file(f) for f in
           (BUILD/visual,BUILD/APP,*sorted((BUILD/"metalrobo").iterdir()))})


def launch():
    cfg=read(REF/"launch.json")
    argv=cfg["argv"]
    slot=argv.index("--skin-payload")+1
    old=(argv[0],argv[slot])
    argv[0],argv[slot],argv[4]=str(BUILD/APP),str(SOURCE),str(NATIVE)
    env=cfg["environment"]
    for key in ("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE","NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"):
        env.pop(key,None)
    env["NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR"]=str(REF/"native")
    env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(OUT/"cardiac-failure.json")
    env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]=",".join(map(str,POSES))
    assert argv[argv.index("--muscle-step-count")+1]=="8000" and env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
    cfg["pins"]={p:s for p,s in cfg["pins"].items()
                 if p not in old and not p.endswith("/resting-human.mrvpack")}
    cfg["pins"].update({str(p):hash_file(p) for p in (BUILD/APP,SOURCE,*sorted((BUILD/"metalrobo").iterdir()))})
    for p,sha in cfg["pins"].items():
        assert hash_file(Path(p))==sha,p
    OUT.mkdir()
    write(OUT/"launch.json",cfg)
    environment={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    started=time.monotonic()
    with (OUT/"stdout.log").open("wb") as out,(OUT/"stderr.log").open("wb") as err:
        p=subprocess.run(argv,cwd=cfg["cwd"],env=environment|env,stdout=out,stderr=err,timeout=380)
    result=dict(exit_code=p.returncode,wall_seconds=time.monotonic()-started,
                variable_pleural_pressure=True,requested_steps=8000)
    write(OUT/"execution.json",result)
    print(result,flush=True)
    p.check_returncode()


def verify():
    for script,args in (
        ("summarize_cardiac_native_run.py",[OUT,OUT/"summary.json"]),
        ("verify_native_cardiac_split.py",[NATIVE/RECEIPT,OUT/"cardiac-verification.json"]),
    ):
        subprocess.run([sys.executable,str(B/script),*map(str,args)],check=True)
    a,b=[read(p/"native"/RECEIPT) for p in (REF,OUT)]
    same={}
    for k in ("accepted_body_state_sha256","accepted_respiration_state_sha256","accepted_time_s","common_field_map_sha256"):
        same[k]=a[k]==b[k]
    for k in ("cardiac_q","cardiac_wall_q","common_coordinates","common_position_low"):
        same[k]=a["accepted_cardiac_geometry_buffers"][k]["sha256"]==b["accepted_cardiac_geometry_buffers"][k]["sha256"]
    for name in ("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv","resting-com-momentum-diagnostic.csv"):
        same[name]=hash_file(REF/"native"/name)==hash_file(NATIVE/name)
    write(OUT/"physical-comparison.json",same)
    assert all(same.values()),same
    install_for_float32()
    audit=runpy.run_path(str(B.parent/"resting-surface-clearance-001/audit.py"))["run"]
    for step in POSES:
        f=NATIVE/"accepted-geometry"/f"step-{step}.mrvpack"
        audit(f,OUT/f"step-{step}-skin-exact.json")
        archive(f)
        archive(f.with_suffix(".common-position-low.bin"))
    for name in ("skin-source-influences.bin","resting-human.mrvpack"):
        archive(NATIVE/name)


if __name__=="__main__":
    build()
    launch()
    verify()
