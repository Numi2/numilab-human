"""Screen a source-section transport over four authored, ordered digital routes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from anchors import extensor
from atlas_attachment_check import descriptive_winding, distances
from branch_fit import coordinate, endpoint_world, slips, smoothstep
from route_segment_screen import _myosim_surface_route_context, digest
from screen import clearance, hits, manifest, neighborhood, pair_index, skin, skin_faces
from source_topology import self_pairs


def source_sections(source, coordinate, ids):
    t=coordinate[ids]
    samples=np.linspace(t.min(),t.max(),13)
    bandwidth=(samples[-1]-samples[0])/12
    weights=np.exp(-.5*((samples[:,None]-t)/bandwidth)**2)
    means=weights@source[ids]/weights.sum(axis=1)[:,None]
    return PchipInterpolator(samples,means,axis=0)(t)


def route_curve(start, route, body, live):
    nodes=[endpoint_world(n,body,live) for n in route["route_points"] if n["kind"]=="site"]
    assert len(nodes)==9
    # Use the wrist bridge and measured distal sites. The authored wrap
    # centers are not used as tendon waypoints.
    sites=np.array([start,*nodes[3:]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(sites,axis=0),axis=1))]
    assert np.all(np.diff(s)>0)
    return PchipInterpolator(s/s[-1],sites,axis=0)


def src_frame(source, coordinate, branch=None):
    if branch is not None:
        source, coordinate=source[branch],coordinate[branch]
    axis=coordinate-coordinate.mean()
    direction=axis@(source-source.mean(axis=0))
    return direction/np.linalg.norm(direction)


def transport(parts,source,native,source_t,branch,curve,scale):
    t=source_t[branch]
    u=(t-t.min())/np.ptp(t)
    source_center=source_sections(source,source_t,branch)
    axis=src_frame(source,source_t,branch)
    offset=source[branch]-source_center
    offset-=np.outer(offset@axis,axis)
    # Recover the rigid radius transform applied by the authoritative decoder.
    x=source-source.mean(axis=0)
    y=parts[1]-parts[1].mean(axis=0)
    left,_,right=np.linalg.svd(x.T@y)
    matrix=left@right
    assert np.linalg.det(matrix)>0.999
    original_axis=axis@matrix
    goal=curve.derivative()(u)
    goal/=np.linalg.norm(goal,axis=1)[:,None]
    direction=np.cross(original_axis,goal)
    sine=np.linalg.norm(direction,axis=1)
    cosine=goal@original_axis
    assert np.all((sine>1e-9)|(cosine>0))
    axis_angle=direction*np.divide(np.arctan2(sine,cosine),sine,out=np.ones_like(sine),where=sine>1e-9)[:,None]
    local=Rotation.from_rotvec(axis_angle).apply(offset@matrix)
    arc=curve(u)+local*scale
    trust=smoothstep(u,0,.28)
    return native[branch]+trust[:,None]*(arc-native[branch])


def evaluate(stable,bodies,routes,index,triangles):
    native,f,w,parts,source,live,desc,_=extensor(stable)
    t=coordinate(source,w)
    pieces=slips(f,t)
    per_branch=[np.median(np.abs(source[br[t[br]>=np.quantile(t[br],.8)],0])) for br in pieces]
    # The mirrored raw source ranks agree bilaterally and the most radial
    # slip retains a majority of the original index-finger frame influences.
    order=np.argsort(per_branch)[::-1]
    ordered=[pieces[i] for i in order]
    names=[r["name"] for r in desc["matched_muscles"]]
    assert all(name.replace("_l","")==f"EDC{i}" for i,name in enumerate(names,2))
    index_finger_bodies={r for r in routes[names[0]]["binding_bodies"] if "2" in r or "second" in r}
    body_ids={entry["myosim_body"]:i for i,entry in enumerate(desc["body_bindings"])}
    assert w[[body_ids[n] for n in index_finger_bodies]][:,ordered[0]].sum()/len(ordered[0])>.7
    crosses=hits(index,native,f)
    self_before=self_pairs(native,f)
    rows=[]
    native_tri=native[f].astype(float)
    endpoints=[endpoint_world(routes[n]["route_points"][-1],bodies,live) for n in names]
    baseline_gap=[1000*distances(x,native_tri) for x in endpoints]
    baseline_winding=descriptive_winding(native,triangles)
    base_volume=clearance._signed_surface_integral(native,f)
    for scale in (1.,.65):
        proposal=native.astype(float)
        for branch,name in zip(ordered,names,strict=True):
            root=np.quantile(t[branch],.10)
            root_xyz=np.median(native[branch[t[branch]<=root]],axis=0)
            curve=route_curve(root_xyz,routes[name],bodies,live)
            proposal[branch]=transport(parts,source,native,t,branch,curve,scale)
        for alpha in (1.,.5,.25):
            alt=(native+alpha*(proposal-native)).astype("<f4")
            c=hits(index,alt,f)
            self_after=self_pairs(alt,f)
            row={"section_scale":scale,"fit_fraction":alpha,
                 "skin_pairs":len(c),"introduced_skin_pairs":len(c-crosses),
                 "self_pairs":len(self_after),"introduced_self_pairs":len(self_after-self_before),
                 "largest_shift_mm":float(np.linalg.norm(alt-native,axis=1).max()*1000),
                 "volume_relative_change":float(clearance._signed_surface_integral(alt,f)/base_volume-1),
                 "distal_site_gap_mm":[1000*distances(x,alt[f].astype(float)) for x in endpoints]}
            if len(c)<len(crosses) and len(self_after)<len(self_before) and alpha==1:
                row["sampled_skin_winding"]=descriptive_winding(alt,triangles)
                row["nearby_changes"]=neighborhood(native,alt,f,stable)
            rows.append(row)
            print(json.dumps({"stable":stable,**row}),file=sys.stderr,flush=True)
    return {"stable":stable,"routes":names,"source_component_order":order.tolist(),
            "baseline_skin_pairs":len(crosses),"baseline_fold_pairs":len(self_before),
            "baseline_distal_site_gap_mm":baseline_gap,
            "baseline_sampled_skin_winding":baseline_winding,"trials":rows}

if __name__=="__main__":
    myosim=Path(sys.argv[2])
    pins=manifest["source"]
    assert digest(myosim/pins["myosim_manifest"]["file"])==pins["myosim_manifest"]["sha256"]
    bodies,routes,_=_myosim_surface_route_context(myosim,pins["myosim_source_archive_sha256"])
    index=pair_index(skin,skin_faces)
    triangles=skin[skin_faces].astype(float)
    print(json.dumps([evaluate(i,bodies,routes,index,triangles) for i in (137,138)],indent=2))
