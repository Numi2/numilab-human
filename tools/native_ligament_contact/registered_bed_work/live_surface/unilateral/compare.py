"""Compare the installed FEM contact branches with the native dense control."""
import csv
import json
from pathlib import Path
import sys
from compare_surface_release import outcome
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def inspect(root):
    cases = ["bed-certificate/one", "unilateral/one", "feasible-bed/three"]
    results = {}
    for key in cases:
        p=root/key
        run=outcome(p)
        launch=read(p/"launch.json")
        name=key.split("/")[0]
        if name!='bed-certificate':
            ids=read(root/name/"owner/identities.json");build=read(root/name/"build/build-pins.json")
            if (ids["binary_sha256"]!=build["binary"] or sha(Path(launch["argv"][0]))!=build["binary"]
                or run["library"]!=build["metallib"] or sha(Path(run["dynamic_library"]))!=build["dylib"]):
                raise ValueError("Wrong loaded solver: "+name)
            if "registered_bed_dual=1 pairs=76153" not in (p/"stderr.log").read_text():
                raise ValueError("Missing compiled contact owner")
        fail=read(p/"failure.json")
        trace=list(csv.DictReader((p/"native-traces/resting-coupled.csv").open()))
        if (not run["pleural_coupling"] or not run["installed_material_area"]
            or not run["common_coordinates_restored"] or len(trace)!=run["accepted_steps"]
            or fail['respiratory_accepted_status']!=fail['respiratory_candidate_status']):
            raise ValueError("Incomplete integrated physical attribution: "+key)
        results[key]=dict(accepted_steps=run["accepted_steps"],last_accepted_s=float(trace[-1]["time_s"]) if trace else 0.,
                          last_accepted_pleural_Pa=float(trace[-1]["pleural_pa"]) if trace else None,
                          failure_root=fail["segment_start"],fresh_residual=run["residual"],
                          failure_code=fail["matter_diagnostics"][3],first_reporting_object=run["matter_object"],
                          root0_history=[dict(iteration=i["i"],residual=i["residual"],line_fraction=i["fractions"]["final_line_search"])
                                         for i in run["iterations"]],
                          fem_force_samples=len(run["accepted_force_history"]),physical_force_history_sha256=__import__("hashlib").sha256(
                              "\n".join(run["accepted_force_history"]).encode()).hexdigest(),
                          wall_s=run["elapsed_s"],variable_pleural=True,common_and_respiratory_rollback=True,launch=run["launch"])
    for d in ('unilateral','feasible-bed'):
        fd=read(root/d/'derivative.json')
        if (fd['cases']!=8192 or fd['finite_difference_cases']!=fd['cases'] or
            fd['max_action_error_per_mass']>=2.e-5 or fd['source']!=sha(root/d/'contact.metalinc')):
            raise ValueError("Unverified integrated derivative")
    if results[cases[-1]]["accepted_steps"]!=2:
        raise ValueError("Reassess accepted horizon")
    out=root/"feasible-bed/comparison.json"
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    inspect(Path(sys.argv[1]).resolve())
