"""Run the existing full visible resting scene with a pinned preparation and source mapping."""
import argparse
import json
from pathlib import Path
from run_bed_work import digest,run
from observe_material_skin import observe

def execute(parent,owner,out,steps,collect=False):
    if steps<=0 or out.exists():raise ValueError("Expected an unused native output directory")
    src=out.with_name(out.name+"-source");src.mkdir()
    config=json.loads((parent/"launch.json").read_text());config["parent_launch"]=digest(parent/"launch.json")
    bed=owner.parent/"foundation.json";env=config["environment"]
    config["pins"][str(bed)]=digest(bed)
    env["NUMI_HUMAN_RESOLVED_BED_WITNESS"]=str(src/"witness.json")
    if collect:
        env["NUMI_HUMAN_SKIN_REFERENCE_DUMP"]=str(src/"initial-skin.csv")
        env["NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"]=str(src/"opening-bodies.json")
        env["NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES"]="1"
    (src/"launch.json").write_text(json.dumps(config,indent=2)+"\n")
    run(src,owner,out,steps,enabled=False,foundation=bed,
        observe_endpoint=(lambda receipt: observe(receipt,out)) if collect else None)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","owner","out"):p.add_argument(name,type=lambda s:Path(s).resolve())
    p.add_argument("steps",type=int);p.add_argument("--collect",action="store_true")
    execute(**vars(p.parse_args()))
