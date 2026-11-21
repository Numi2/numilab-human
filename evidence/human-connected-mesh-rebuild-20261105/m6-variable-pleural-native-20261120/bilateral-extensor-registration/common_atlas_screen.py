"""Evaluate consistent rest frames for both extrinsic finger extensors."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from screen import (
    banks, bindings, bind_transforms, hits, indices, linear, manifest,
    neighborhood, pair_index, poses, records, registered, shape, skin, skin_faces, vertices,
)
from route_segment_screen import _myosim_surface_route_context, digest, value_weights
from source_topology import self_pairs

def compose_from_reference(x, weights, atlas, neutral, live):
    rest=x @ (atlas[:3,:3]*1000).T + atlas[:3,3]
    points=[]
    for rest_body, live_body in zip(neutral,live,strict=True):
        pos=np.asarray(rest_body["default_com_position_world_m"])
        q=Rotation.from_quat(rest_body["default_inertial_quaternion_world_xyzw"])
        world=Rotation.from_quat(live_body["quaternion_xyzw"])
        points.append((world*q.inv()).apply(rest-pos)+live_body["position_m"])
    return rest,np.einsum("bn,bnd->nd",weights,np.stack(points))

def bank(stable, atlas, bodies, routes, skin_index):
    desc=next(m for m in manifest["source"]["surfaces"] if m["stable_id"]==stable)
    b0,nb,v0,nv,i0,ni,_,_=map(int,records[records[:,6]==stable][0])
    bs=bindings[b0:b0+nb]
    raw=(indices[i0:i0+ni]-v0).reshape(-1,3)
    used=np.unique(raw)
    f=np.searchsorted(used,raw)
    vs=vertices[v0:v0+nv][used]
    xyz=vs["position"].astype(float)
    w=value_weights(vs,nb)
    native,face=banks[51005,stable]
    assert np.array_equal(f,face)
    methods=[]
    ids=bs["body"].tolist()
    for is_registered,live in [(True,registered),(False,poses)]:
        if all(i in live for i in ids):
            p=linear(xyz,w,*bind_transforms(bs,use_registered=is_registered))
            methods.append((float(np.linalg.norm(p-native,axis=1).max()),[live[i] for i in ids]))
    error,live=min(methods,key=lambda m:m[0])
    assert error<2e-6
    neutral=[bodies[x["myosim_body"]] for x in desc["body_bindings"]]
    assert [r["core_body_index"] for r in neutral]==ids
    rest,transformed=compose_from_reference(xyz,w,atlas,neutral,live)
    terminals=np.array([p["world_m"] for m in desc["matched_muscles"] for p in (
        routes[m["name"]]["route_points"][0],routes[m["name"]]["route_points"][-1])])
    protection=np.clip((cKDTree(terminals).query(rest)[0]-.003)/.009,0,1)
    protection=protection**2*(3-2*protection)
    protection[cKDTree(rest).query(terminals)[1]]=0
    protection[w.max(axis=0)>1-2e-6]=0
    cross=hits(skin_index,native,f)
    self_before=self_pairs(native,f)
    results=[]
    for name, scale in [("all_points_counterfactual",np.ones_like(protection)),
                         ("terminal_preserved",protection)]:
        for alpha in ((1.,) if name=="all_points_counterfactual" else (1.,.5,.25,.125)):
            pos=(native.astype(float)+alpha*scale[:,None]*(transformed-native)).astype("<f4")
            delta=np.linalg.norm(pos-native,axis=1)
            assert not np.any(delta[scale==0])
            target=hits(skin_index,pos,f)
            self_after=self_pairs(pos,f)
            row={"trial":name,"fraction":alpha,"skin_pairs":len(target),
                 "skin_new_pairs":len(target-cross),
                 "skin_removed_pairs":len(cross-target),"self_pairs":len(self_after),
                 "self_new_pairs":len(self_after-self_before),"self_removed_pairs":len(self_before-self_after),
                 "max_displacement_mm":float(1000*delta.max()),
                 "pinned_vertices":int((scale==0).sum()),
                 "shape_vs_pose":shape(native,pos,f)}
            if len(target)<len(cross) and len(self_after)<=len(self_before):
                row["neighbor_changes"]=neighborhood(native,pos,f,stable)
            results.append(row)
            print(json.dumps({"stable":stable,**row}),file=sys.stderr,flush=True)
    return dict(stable_id=stable,baseline_skin_pairs=len(cross),baseline_self_pairs=len(self_before),trials=results)

if __name__=="__main__":
    myosim,config=map(Path,sys.argv[2:4])
    pins=manifest["source"]
    assert digest(myosim/pins["myosim_manifest"]["file"])==pins["myosim_manifest"]["sha256"]
    assert digest(config)==pins["registration"]["sha256"]
    body,route,_=_myosim_surface_route_context(myosim,pins["myosim_source_archive_sha256"])
    atlas=np.array(json.loads(config.read_text())["coordinate_system"]["global_source_mm_to_myosim_world_m"])
    skin_index=pair_index(skin,skin_faces)
    results=[bank(i,atlas,body,route,skin_index) for i in (137,138)]
    print(json.dumps(results,indent=2))
