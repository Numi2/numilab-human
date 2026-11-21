"""Solve independent clearance regions while preserving all incident-cell checks."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from envelope_lp import MM,digest,exact_summary,fit,prepare,selected_sites
from repair_rounding import repair


def solve(points,tets,material,locked,targets,radius,preserve_material_volume=False):
    # Components that touch any common tetrahedron are solved together.
    active=np.empty((len(targets),len(tets)),dtype=bool)
    editable=[]
    tree=cKDTree(points.astype(float))
    for i,q in enumerate(targets):
        foot=q["point_m"].copy()
        foot[0]=q["skin_intercept_m"]
        affected=np.zeros(len(points),dtype=bool)
        affected[tree.query_ball_point(foot,radius*MM)]=True
        affected&=~locked
        active[i]=np.any(affected[tets],axis=1)
        editable.append(affected)
    adjacency=np.eye(len(targets),dtype=bool)
    for i in range(len(targets)):
        for j in range(i):
            adjacency[i,j]=adjacency[j,i]=np.any(active[i]&active[j])
    count,groups=connected_components(adjacency,directed=False)
    composite=points.copy()
    observations=[]
    for i in range(count):
        group=np.flatnonzero(groups==i)
        local=np.any(active[group],axis=0)
        local_targets=[targets[ix] for ix in group]
        origin=dict(group=i,site_rows=group.tolist(),tets=int(local.sum()),radius_mm=radius)
        print(json.dumps(origin),file=sys.stderr,flush=True)
        panel,solves=fit(points,tets[local],material[local],locked,local_targets,radius=radius,preserve_material_volume=preserve_material_volume)
        origin["solves"]=solves
        if panel is not None:
            allowed=np.any(np.asarray(editable)[group],axis=0)
            panel,origin['rounded_quality_repair']=repair(panel,tets[local],points,allowed)
            changed=np.any(panel!=points,axis=1)
            assert np.array_equal(points[changed],composite[changed])
            metrics=exact_summary(panel,tets[local],material[local])
            origin["whole_patch_quality"]=metrics
            if metrics["volume_floor_failures"]==metrics["shape_floor_failures"]==0:
                composite[changed]=panel[changed]
                origin["changed_passing_nodes"]=int(changed.sum())
        observations.append(origin)
    return composite,observations

if __name__=="__main__":
    launch,proposal,bed,faces=map(Path,sys.argv[2:6])
    points,tets,material,locked=prepare(launch,proposal,bed)
    targets=selected_sites(faces)
    candidate,result=solve(points,tets,material,locked,targets,float(sys.argv[6]))
    rows=np.flatnonzero(np.any(candidate!=points,axis=1))
    assert np.array_equal(candidate[locked],points[locked])
    whole=exact_summary(candidate,tets,material)
    assert whole["volume_floor_failures"]==whole["shape_floor_failures"]==0
    passed=[]
    for t in targets:
        shift=(candidate[t["nodal_rows"],0].astype(float)-points[t["nodal_rows"],0])/MM
        achieved=t["side"]*(shift@np.array(t["bary"]))
        passed.append(dict(route=t["route"],site=t["site"],world_m=t["point_m"],
                           intended_slack_mm=achieved-t["gap_m"]/MM,
                           estimated_axis_move_mm=achieved))
    report=dict(changed_nodes=len(rows),sites=passed,groups=result,whole_volume=whole)
    if len(rows):
        asset=Path("grouped-envelope.npz")
        np.savez_compressed(asset,node_rows=rows,position_f32=candidate[rows])
        report["proposal_sha256"]=digest(asset)
    print(json.dumps(report,indent=2))
