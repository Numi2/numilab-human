"""Compare the guarded source skin trial with the matched authored-pressure control."""
import argparse
import json
from pathlib import Path

import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-004"
sys.path.insert(0,str(PREVIOUS))
from run_segment_control import PRIOR, digest
from verify_segment_control import profile, RECEIPT

def load(path):
    return json.loads(Path(path).read_text())

def main():
    run=HERE/"skin-envelope-16s"
    stem="skin-envelope"
    control=PREVIOUS/"native-segment32-control"
    a,b=(load(p/RECEIPT) for p in (control,run))
    assert b["accepted_step"]==8000 and b["physical_endpoint"]=="accepted"
    assert a["accepted_time_s"]==b["accepted_time_s"]
    physics={key:a[key]==b[key] for key in ("accepted_body_state_sha256","accepted_respiration_state_sha256")}
    for k in a["accepted_cardiac_geometry_buffers"]:
        physics[f"cardiac_{k}"]=a["accepted_cardiac_geometry_buffers"][k]["sha256"]==(
            b["accepted_cardiac_geometry_buffers"][k]["sha256"])
    physics["bed_sections_byte_equal"]=(control/"native/resting-local-bed-sections.csv").read_bytes()==(
        run/"native/resting-local-bed-sections.csv").read_bytes()
    for name in ("resting-coupled.csv","resting-com-support-impulses.csv",
                 "resting-com-momentum-diagnostic.csv"):
        path="native/"+name
        physics[name+"_byte_equal"]=digest(control/path)==digest(run/path)
    old=load(PRIOR/"wider-release-control-16-exact.json")
    new=load(HERE/"skin-envelope-native-exact.json")
    assert old["accepted_pack_sha256"]==a["pack_file_sha256"]
    assert new["accepted_pack_sha256"]==b["pack_file_sha256"]
    by_key=lambda reports:{(r["semantic"],r["stable_id"]):r for r in reports["surfaces"]}
    first,last=by_key(old),by_key(new)
    assert first.keys()==last.keys()
    def compare(x,y):
        before,after=[set(map(tuple,z["triangle_pairs"])) for z in (x,y)]
        assert x["count"]==len(before) and y["count"]==len(after)
        return dict(before=len(before),after=len(after),introduced=len(after-before),
                    removed=len(before-after),new_witnesses=sorted(after-before))
    by_target=[dict(semantic=k[0],stable_id=k[1],**compare(first[k],last[k])) for k in sorted(first)]
    totals={key:sum(v[key] for v in by_target) for key in ("before","after","introduced","removed")}
    self_pairs=compare(old["self_intersections"],new["self_intersections"])
    result=dict(physiology_contact_parity=physics,profile=profile(run),native_self_pairs=self_pairs,
                native_tissue_pairs=totals,by_target=by_target,
                original_support_source=digest(PRIOR/"wider-release-control-16/native/resting-local-bed-sections.csv"),
                skin_trial_passed_terminal_regression=all(physics.values()) and not totals["introduced"]
                    and not self_pairs["introduced"] and totals["after"]<totals["before"],
                qualified_variable_pleural_pressure=False,default_model_changed=False)
    print({k:v for k,v in result.items() if k!="by_target"},flush=True)
    (HERE/(stem+"-native-comparison.json")).write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
