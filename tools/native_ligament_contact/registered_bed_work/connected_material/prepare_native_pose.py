"""Use the dense registered joint configuration in the same supported native scene."""
import json
import sys
from pathlib import Path

from run_bed_work import digest


def prepare(control,dense,out):
    read=lambda p:json.loads((p/"launch.json").read_text())
    job=read(control); src=read(dense); argv=job["argv"]
    def joints(a):return {int(a[i+1]):a[i+2] for i,s in enumerate(a) if s=="--pose-q"}
    before=joints(argv); reference=joints(src["argv"])
    if before.keys()!=reference.keys():raise ValueError("Different joint admission sets")
    if digest(Path(argv[1]))!=digest(Path(src["argv"][1])):raise ValueError("Different articulated bodies")
    for i,s in enumerate(argv):
        if s=="--pose-q":argv[i+2]=reference[int(argv[i+1])]
    out.mkdir()
    job["environment"].pop("NUMI_HUMAN_RECRUIT_AT_FOUNDATION_LOAD",None)
    job["environment"]["NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"]=str(out/"opening-bodies.json")
    job["environment"]["NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES"]="1"
    job["environment"]["NUMI_HUMAN_BED_QUADRATURE_CAPTURE"]="1"
    job["parent_launch"]=digest(control/"launch.json")
    (out/"launch.json").write_text(json.dumps(job,indent=2)+"\n")
    (out/"pose-choice.json").write_text(json.dumps(dict(source_launch=digest(dense/"launch.json"),before=before,reference=reference),indent=2)+"\n")


if __name__=="__main__":prepare(*(Path(x).resolve() for x in sys.argv[1:]))
