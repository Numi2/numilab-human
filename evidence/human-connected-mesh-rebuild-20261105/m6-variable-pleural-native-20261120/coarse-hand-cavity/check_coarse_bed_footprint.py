"""Check whether the actual registered initial bed rows can balance gravity."""
from argparse import ArgumentParser
from pathlib import Path
import json
import re
import sys

import numpy as np

from build_hand_trial import HERE, digest
from fit_hand_cavity import table

# Reuse the same convex-hull geometry as the verified fine-volume audit.
sibling=HERE.parent/"tissue-registration-native-022"
sys.path.insert(0,str(sibling))
from check_bed_footprint import hull_distance
sys.path.pop(0)


def report(trial):
    initial=table(trial/"native/resting-surface-audit.csv")[0]
    assert int(initial["step"])==0
    mass_rigid=float(initial["represented_body_mass_kg"])
    com_rigid=np.array([float(initial[f"body_com_{a}_m"]) for a in "xyz"])
    fem=[]
    observed={}
    with (trial/"stdout.log").open() as source:
        for line in source:
            if line.startswith("resting_FEM_node,"):
                values=dict(a.split("=",1) for a in line.rstrip().split(",")[1:])
                p=[float(values[f"initial_{a}"]) for a in "xyz"]
                fem.append((int(values["global"]),float(values["initial_mass_kg"]),*p))
            if line.startswith(("resting_fem_skin_bed_compiled=", "resting_fem_skin_bed_source=",
                                "resting_tissue_shared_runtime=bound ")):
                key,_,_=line.partition("=")
                observed[key]=line.strip()
    points=np.array(sorted(fem))
    assert np.array_equal(points[:,0],np.arange(len(points)))
    native=re.search(r"FEM_mass_kg=(\S+)",observed["resting_tissue_shared_runtime"])
    assert native is not None
    mass_fem=points[:,1].sum()
    native_mass=float(native[1])
    assert abs(mass_fem-native_mass)<5.e-6
    mass=mass_rigid+mass_fem
    assert abs(mass-97.13195069110952)<5.e-5
    center=(mass_rigid*com_rigid+points[:,1] @ points[:,2:])/mass

    assert "normal=0,0,1 plane_offset_m=0" in observed["resting_fem_skin_bed_source"]
    all_skin=table(trial/"initial-skin-fem.csv")
    outer=np.array([[float(v[f"{a}_m"]) for a in "xyz"] for v in all_skin if v["layer"]=="0"])
    n=re.search(r"pair_rows=(\d+) initial_outer_min_gap_m=(\S+) initial_outer_max_gap_m=(\S+)",
                observed["resting_fem_skin_bed_compiled"])
    assert len(outer)==int(n[1])
    assert float(n[2])==outer[:,2].min() and float(n[3])==outer[:,2].max()
    skin_id=np.array([int(v["global_node"]) for v in all_skin if v["layer"]=="0"])
    assert np.max(np.linalg.norm(outer-points[skin_id,2:],axis=1))<3.e-8

    order=np.argsort(outer[:,2],kind="stable")
    outer=outer[order]
    coordinate=outer[:,2]
    assert hull_distance(outer[:,:2],center[:2])[0]==0.0
    lower,upper=1,len(outer)
    while upper-lower>1:
        mid=(lower+upper)//2
        if hull_distance(outer[:mid,:2],center[:2])[0]==0.0:
            upper=mid
        else:
            lower=mid
    metrics=[]
    for count in (1,int(np.searchsorted(coordinate,10.e-6,side="right")),upper):
        distance,area=hull_distance(outer[:count,:2],center[:2])
        metrics.append({"outer_nodes":count,"highest_gap_m":float(coordinate[count-1]),
                        "hull_area_m2":area,"com_to_hull_m":distance})
    return {"launch_sha256":digest(trial/"launch.json"),"native_fem_nodes":len(points),
            "total_compiled_mass_kg":float(mass),"body_com_m":center.tolist(),
            "weight_N":float(mass*9.81),"support_geometry":metrics,
            "scope":"Necessary condition for unilateral flat-bed normal support at the actual initial pose. Does not assign loads or change geometry."}


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument("trial",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    result=report(a.trial)
    with a.out.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
