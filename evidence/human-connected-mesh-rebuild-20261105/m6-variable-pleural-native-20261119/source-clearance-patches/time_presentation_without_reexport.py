"""Time the verified native scene without regenerating its retained packs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from register_tissue_bellies import HERE, hash_file
from qualify_skin_connectivity import read, write

CONTROL=HERE/"source-clearance-patches-native"
OUT=HERE/"source-clearance-presentation-timing"


def log_value(path,name,separator="="):
    prefix=name+separator
    with path.open() as lines:
        values=[line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    assert len(values)==1,(path,name,len(values))
    return values[0]


def main():
    root=CONTROL/"native"
    OUT.mkdir()
    cfg=read(CONTROL/"launch.json")
    for path,sha in cfg["pins"].items():
        assert hash_file(Path(path))==sha,path
    with tempfile.TemporaryDirectory(prefix="timing-cache-",dir=HERE) as temp:
        start=Path(temp)/"resting-human.mrvpack"
        receipt=read(root/"resting-human.mrvpack.zst.recovery.json")
        with start.open("xb") as output:
            subprocess.run(["/opt/homebrew/bin/zstd","-dc",root/"resting-human.mrvpack.zst"],
                           stdout=output,check=True)
        assert hash_file(start)==receipt["sha256"] and start.stat().st_size==receipt["bytes"]
        cfg["argv"][4]=str(OUT/"native")
        e=cfg["environment"]
        for k in ("NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR","NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"):
            e.pop(k,None)
        e.update(NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(start),
                 NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=receipt["sha256"],
                 NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(OUT/"cardiac-failure.json"))
        cfg["pins"][str(start)]=receipt["sha256"]
        write(OUT/"launch.json",cfg)
        env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
        t=time.monotonic()
        with (OUT/"stdout.log").open("wb") as stdout,(OUT/"stderr.log").open("wb") as stderr:
            p=subprocess.run(cfg["argv"],cwd=cfg["cwd"],env=env|e,
                             stdout=stdout,stderr=stderr,timeout=340)
        elapsed=time.monotonic()-t
        write(OUT/"execution.json",dict(wall_seconds=elapsed,exit_code=p.returncode))
        p.check_returncode()
    verify()


def verify():
    root=CONTROL/"native"
    receipt=read(root/"resting-human.mrvpack.zst.recovery.json")
    execution=read(OUT/"execution.json")
    assert execution["exit_code"]==0
    elapsed=execution["wall_seconds"]
    baseline=CONTROL/"stdout.log"
    sample=OUT/"stdout.log"
    a,b=[json.loads(log_value(p,"stand_terminal_state")) for p in (baseline,sample)]
    same={k:a[k]==b[k] for k in a}
    with sample.open() as f:
        frames=[line.strip() for line in f if line.startswith("resting_present_profile ")]
    terminal=frames[-1]
    assert terminal.startswith("resting_present_profile step=7999 finished=1 ")
    assert all(same.values()),[k for k,v in same.items() if not v]
    for n in ("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv",
              "resting-com-momentum-diagnostic.csv"):
        same[n]=hash_file(OUT/"native"/n)==hash_file(root/n)
    assert all(same.values()),same
    secs=read(CONTROL/"summary.json")["timing"]["simulated_seconds"]
    result=dict(wall_seconds=elapsed,simulated_seconds=secs,launch_to_exit_rtf=secs/elapsed,
                integrated_motion=log_value(sample,"resting_integrated_throughput"," "),
                terminal_presentation=terminal,native_presentation_callbacks=len(frames),
                identical_terminal_state_and_traces=same,
                timed_with_verified_startup_cache=True,
                reproduced_geometry_export=False)
    write(OUT/"timing.json",result)
    copy=OUT/"native/resting-human.mrvpack"
    assert hash_file(copy)==receipt["sha256"]
    # The cache was only staged for this trial. Preserve the exact
    # compressed input and recovery receipt under the trial output.
    compressed=copy.with_suffix(copy.suffix+".zst")
    import hashlib
    compressed_source=root/"resting-human.mrvpack.zst"
    with subprocess.Popen(["/opt/homebrew/bin/zstd","-dc",compressed_source],stdout=subprocess.PIPE) as proc:
        assert hashlib.file_digest(proc.stdout,"sha256").hexdigest()==receipt["sha256"]
        assert proc.wait()==0
    os.link(compressed_source,compressed)
    receipt["restore_command"]=["/opt/homebrew/bin/zstd","-d",str(compressed),"-o",str(copy)]
    write(copy.with_suffix(copy.suffix+".zst.recovery.json"),receipt)
    copy.unlink()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
