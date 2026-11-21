"""Constrain distal surface influence sets by the actual connected tendon slips."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from anchors import extensor
from atlas_attachment_check import descriptive_winding, distances
from route_segment_screen import _myosim_surface_route_context, digest
from screen import clearance, hits, manifest, neighborhood, pair_index, skin, skin_faces
from source_topology import self_pairs


def coordinate(points, weights):
    axial = np.linalg.svd(points-points.mean(axis=0),full_matrices=False)[2][0]
    s = points@axial
    if s[weights[0]>.3].mean()>s.mean():
        s=-s
    return (s-s.min())/np.ptp(s)


def slips(faces, axial, split=.70):
    distal=np.flatnonzero(axial>=split)
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a,b=np.searchsorted(distal,edges[(axial[edges]>=split).all(axis=1)]).T
    adjacency=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(distal),len(distal)))
    count,component=connected_components(adjacency,directed=False)
    assert count==4, (count,np.bincount(component))
    return [distal[component==i] for i in range(count)]


def smoothstep(t, lower, upper):
    r=np.clip((t-lower)/(upper-lower),0,1)
    return r*r*(3-2*r)


def endpoint_world(route_endpoint, neutral, current):
    point=np.asarray(route_endpoint["world_m"])
    inner=neutral[route_endpoint["body"]]
    final=current[route_endpoint["core_body_index"]]
    r=Rotation.from_quat(inner["default_inertial_quaternion_world_xyzw"])
    q=Rotation.from_quat(final["quaternion_xyzw"])
    return (q*r.inv()).apply(point-inner["default_com_position_world_m"])+final["position_m"]


def extent_and_trials(stable,bodies,routes,skin_index,skin_tris):
    native,faces,w,parts,source,live,desc,ids=extensor(stable)
    body_ids={entry["myosim_body"]:i for i,entry in enumerate(desc["body_bindings"])}
    base=np.einsum("kn,kni->ni",w,parts)
    source_t=coordinate(source,w)
    pieces=slips(faces,source_t)
    names=[r["name"] for r in desc["matched_muscles"]]
    route_rows=[routes[name] for name in names]
    ownership=[[body_ids[name] for name in route["binding_bodies"]] for route in route_rows]
    common=set.intersection(*(set(row) for row in ownership))
    exclusive=[sorted(set(owners)-common) for owners in ownership]
    assert len(common)==3 and all(len(row)==4 for row in exclusive)
    anchors=[endpoint_world(r["route_points"][-1],bodies,live) for r in route_rows]
    reliability=np.asarray([[w[owner][:,nodes].sum()/len(nodes) for owner in exclusive] for nodes in pieces])
    component,assigned_route=linear_sum_assignment(-reliability)
    assert component.tolist()==list(range(4))
    cross=hits(skin_index,native,faces)
    folded=self_pairs(native,faces)
    base_volume=clearance._signed_surface_integral(native,faces)
    orig_distal=[1000*distances(x,native[faces].astype(float)) for x in anchors]
    assignments=[]
    for i,branch in enumerate(pieces):
        r=assigned_route[i]
        assignments.append(dict(route=names[r],component=i,vertices=len(branch),
                                author_owner_fraction=float(reliability[i,r]),
                                runner_up_fraction=float(np.sort(reliability[i])[-2])))
    proposed=np.array(w,dtype=float)
    for branch,r in zip(pieces,assigned_route,strict=True):
        keep=ownership[r]
        restricted=np.zeros((len(w),len(branch)))
        restricted[keep]=w[keep][:,branch]
        totals=restricted.sum(axis=0)
        # If a source vertex has no accepted route-local ownership, use the
        # retained lunate frame rather than inheriting another finger.
        no_totals=totals<=2e-6
        assert body_ids[route_rows[r]["binding_bodies"][2]]==2
        restricted[2,no_totals]=1
        totals[no_totals]=1
        proposed[:,branch]=restricted/totals
    blend=smoothstep(source_t,.70,.82)
    new_w=w*(1-blend)+proposed*blend
    overflow=np.argsort(-new_w,axis=0,kind="stable")[4:]
    np.put_along_axis(new_w,overflow,0,axis=0)
    new_w[:,blend>0]/=new_w[:,blend>0].sum(axis=0)
    assert np.all((new_w>0).sum(axis=0)<=4)
    delta=np.einsum("kn,kni->ni",new_w,parts)-base
    assert not np.any(delta[source_t<=.70])
    results=[]
    for alpha in (1.,.5,.25):
        alt=(native.astype(float)+alpha*delta).astype("<f4")
        h=hits(skin_index,alt,faces)
        f=self_pairs(alt,faces)
        row={"alpha":alpha,"skin_pairs":len(h),"introduced_skin_pairs":len(h-cross),
             "self_pairs":len(f),"introduced_self_pairs":len(f-folded),
             "max_move_mm":float(np.linalg.norm(alt-native,axis=1).max()*1000),
             "relative_signed_volume_change":float(clearance._signed_surface_integral(alt,faces)/base_volume-1),
             "terminal_distances_mm":[1000*distances(x,alt[faces].astype(float)) for x in anchors]}
        if len(h)<len(cross) and len(f)<len(folded):
            row["nearby"]=neighborhood(native,alt,faces,stable)
            row["sampled_winding"]=descriptive_winding(alt,skin_tris)
        results.append(row)
        print(json.dumps({"stable":stable,**row}),file=sys.stderr,flush=True)
    return {"stable":stable,"matching_scores":reliability.tolist(),"assignments":assignments,
            "shared_owners":sorted(common),"source_slip_start":.70,
            "original_skin_pairs":len(cross),"original_self_pairs":len(folded),
            "original_distal_route_distances_mm":orig_distal,"trials":results}

if __name__=="__main__":
    directory=Path(sys.argv[2])
    pins=manifest["source"]
    assert digest(directory/pins["myosim_manifest"]["file"])==pins["myosim_manifest"]["sha256"]
    skeleton,routes,_=_myosim_surface_route_context(directory,pins["myosim_source_archive_sha256"])
    index=pair_index(skin,skin_faces)
    triangles=skin[skin_faces].astype(float)
    print(json.dumps([extent_and_trials(i,skeleton,routes,index,triangles) for i in (137,138)],indent=2))
