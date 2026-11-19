"""Qualify a source-connectivity change with the matched native scene."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from register_tissue_bellies import B, HERE, LATE, READER, hash_file, install_for_float32

OUT=HERE/"connected-skin-native"
BASE=LATE/"native"
PATH=OUT/"native"
RECEIPT="accepted-geometry/step-8000.receipt.json"
sys.path.insert(0,str(B.parent/"resting-clearance-and-atrium-004"))
from archive_bed_preflight import archive
from compare_guarded_variable_pressure import pairs


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    with path.open("x") as out:
        json.dump(data,out,indent=2)
        out.write("\n")


def execute():
    config=read(LATE/"launch.json")
    for p,sha in config["pins"].items():
        assert hash_file(Path(p))==sha,p
    proof=read(HERE/"connected-skin-diagonals.json")
    candidate=HERE/"connected-skin-diagonals.nhskin"
    assert hash_file(candidate)==proof["candidate_sha256"]
    argv=config["argv"]
    skin_index=argv.index("--skin-payload")+1
    assert hash_file(Path(argv[skin_index]))==proof["source_sha256"]
    argv[skin_index]=str(candidate)
    config["pins"][str(candidate)]=proof["candidate_sha256"]
    build_receipt=read(HERE/"connected-skin-build/build-verification.json")
    binary=HERE/"connected-skin-build/numi-human-native"
    assert hash_file(binary)==build_receipt[binary.name]
    argv[0]=str(binary)
    config["pins"][str(binary)]=build_receipt[binary.name]
    argv[4]=str(PATH)
    env=config["environment"]
    env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"]=str(OUT/"cardiac-failure.json")
    baseline_binary=Path(read(LATE/"launch.json")["argv"][0])
    for shader in ("HumanRestingSkin.metallib","MetalRobo.metallib"):
        original=baseline_binary.parent/"metalrobo"/shader
        deployed=binary.parent/"metalrobo"/shader
        assert deployed.samefile(original)
        config["pins"][str(deployed)]=hash_file(original)
    startup=HERE/"rejected-missing-sibling-shader/native"
    cached=startup/"resting-human.mrvpack"
    base_sha=read(HERE/"rejected-unregistered-connectivity/native/resting-human.mrvpack.zst.recovery.json")["sha256"]
    assert hash_file(cached)==base_sha
    env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"]=str(cached)
    env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"]=base_sha
    env["NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR"]=str(startup)
    config["pins"][str(cached)]=base_sha
    assert env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]=="1"
    assert env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]=="8000"
    assert argv[argv.index("--muscle-step-count")+1]=="8000"
    OUT.mkdir()
    write(OUT/"launch.json",config)
    environment={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    before=time.monotonic()
    with (OUT/"stdout.log").open("wb") as stdout,(OUT/"stderr.log").open("wb") as stderr:
        p=subprocess.run(argv,cwd=config["cwd"],env=environment|env,
                         stdout=stdout,stderr=stderr,timeout=360)
    result={"exit_code":p.returncode,"wall_seconds":time.monotonic()-before,
            "requested_steps":8000,"variable_pleural_pressure":True}
    write(OUT/"execution.json",result)
    print(result,flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)


def compare_geometry():
    install_for_float32()
    # The run writes its real emitted pack before any compressed retention.
    pack=PATH/"accepted-geometry/step-8000.mrvpack"
    runpy.run_path(str(B.parent/"resting-surface-clearance-001/audit.py"))["run"](
        pack,OUT/"skin-exact.json")
    receipt=[read(p/RECEIPT) for p in (BASE,PATH)]
    physical={}
    for k in ("accepted_body_state_sha256","accepted_respiration_state_sha256",
              "accepted_root_fingerprint_hex","accepted_time_s","common_field_map_sha256"):
        physical[k]=receipt[0][k]==receipt[1][k]
    for k in ("cardiac_q","cardiac_wall_q","common_coordinates","common_position_low"):
        physical[k]=receipt[0]["accepted_cardiac_geometry_buffers"][k]["sha256"]==receipt[1]["accepted_cardiac_geometry_buffers"][k]["sha256"]
    for n in ("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv",
              "resting-com-momentum-diagnostic.csv"):
        physical[n]=hash_file(BASE/n)==hash_file(PATH/n)
    audits=(read(B/(LATE.name+"-skin-exact.json")),read(OUT/"skin-exact.json"))
    assert all(a["accepted_pack_sha256"]==b["pack_file_sha256"] for a,b in zip(audits,receipt))
    a,b=map(pairs,audits)
    assert a.keys()==b.keys()
    sa,sb=({tuple(p) for p in r["self_intersections"]["triangle_pairs"]} for r in audits)
    result=dict(tissue_pairs_before=sum(map(len,a.values())),tissue_pairs_after=sum(map(len,b.values())),
                tissue_pairs_added=sum(len(b[k]-a[k]) for k in a),
                tissue_pairs_removed=sum(len(a[k]-b[k]) for k in a),
                self_pairs_before=len(sa),self_pairs_after=len(sb),self_pairs_added=len(sb-sa),
                physical_byte_parity=physical)
    write(OUT/"comparison.json",result)
    print(result,flush=True)


def main():
    execute()
    for script,args in (
        ("summarize_cardiac_native_run.py",[OUT,OUT/"summary.json"]),
        ("verify_native_cardiac_split.py",[PATH/RECEIPT,OUT/"cardiac-verification.json"]),
    ):
        subprocess.run([sys.executable,str(B/script),*map(str,args)],check=True)
    reference=HERE/"rejected-unregistered-connectivity/native"
    for name in ("skin-source-influences.bin","resting-human.mrvpack"):
        path=PATH/name
        if path.is_file():
            retained=reference/(name+".zst")
            sha=read(reference/(name+".zst.recovery.json"))["sha256"]
            if hash_file(path)==sha:
                os.link(retained,path.with_suffix(path.suffix+".zst"))
            archive(path)
    compare_geometry()
    folder=PATH/"accepted-geometry"
    for name in ("step-8000.mrvpack","step-8000.common-position-low.bin"):
        archive(folder/name)


if __name__=="__main__":
    main()
