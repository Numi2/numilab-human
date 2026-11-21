"""Measure route proximity and enclosure for the zero-pair counterfactual."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from common_atlas_screen import compose_from_reference
from route_segment_screen import _myosim_surface_route_context, digest, value_weights
from screen import (
    banks, bindings, bind_transforms, clearance, indices, linear, manifest,
    poses, records, registered, skin, skin_faces, vertices,
)
from source_topology import topology

def distances(point, triangles):
    a,b,c=triangles.transpose(1,0,2)
    m=np.cross(b-a,c-a)
    m/=np.linalg.norm(m,axis=1)[:,None]
    q=point-a
    height=np.sum(q*m,axis=1)
    foot=point-height[:,None]*m
    sides=[]
    for start,end in ((a,b),(b,c),(c,a)):
        d=end-start
        value=np.sum((foot-start)*d,axis=1)/np.sum(d*d,axis=1)
        direction=np.sum(np.cross(d,foot-start)*m,axis=1)
        vector=start+np.clip(value,0,1)[:,None]*d-point
        sides.append((direction,np.sum(vector*vector,axis=1)))
    inside=(np.stack([s[0] for s in sides])>=0).all(axis=0)
    return float(np.sqrt(np.where(inside,height*height,np.min([s[1] for s in sides],axis=0)).min()))

def descriptive_winding(points, triangles):
    # Exact plane contacts are handled by the pair audit. With a possibly open
    # facial shell this winding sample is a diagnostic rather than containment certification.
    selected=np.linspace(0,len(points)-1,17,dtype=int)
    return [float(clearance._generalized_winding_number(points[i],triangles)) for i in selected]

if __name__=="__main__":
    myosim,config=map(Path,sys.argv[2:4])
    pins=manifest["source"]
    assert digest(myosim/pins["myosim_manifest"]["file"])==pins["myosim_manifest"]["sha256"]
    assert digest(config)==pins["registration"]["sha256"]
    body,route,_=_myosim_surface_route_context(myosim,pins["myosim_source_archive_sha256"])
    atlas=np.array(json.loads(config.read_text())["coordinate_system"]["global_source_mm_to_myosim_world_m"])
    skin_f64=skin[skin_faces].astype(float)
    report=[]
    for stable in (137,138):
        desc=next(r for r in pins["surfaces"] if r["stable_id"]==stable)
        b0,nb,v0,nv,i0,ni,_,_=map(int,records[records[:,6]==stable][0])
        bs=bindings[b0:b0+nb]
        raw=(indices[i0:i0+ni]-v0).reshape(-1,3)
        used=np.unique(raw)
        f=np.searchsorted(used,raw)
        vs=vertices[v0:v0+nv][used]
        xyz=vs["position"].astype(float)
        w=value_weights(vs,nb)
        native,faces=banks[51005,stable]
        assert np.array_equal(f,faces)
        neutral=[body[r["myosim_body"]] for r in desc["body_bindings"]]
        ids=bs["body"].tolist()
        methods=[(float(np.linalg.norm(linear(xyz,w,*bind_transforms(bs,use_registered=flag))-
                    native,axis=1).max()),flag,live)
                 for flag,live in ((True,registered),(False,poses)) if all(i in live for i in ids)]
        error,flag,live=min(methods,key=lambda t:t[0])
        assert error<2e-6
        rest,common=compose_from_reference(xyz,w,atlas,neutral,[live[i] for i in ids])
        ref_tris=native[f].astype(float)
        common_tris=common.astype("<f4")[f].astype(float)
        endpoint=[]
        for m in desc["matched_muscles"]:
            r=route[m["name"]]
            for owner,p in (("origin",r["route_points"][0]),("insertion",r["route_points"][-1])):
                b=body[p["body"]]
                assert p["kind"]=="site"
                rest_pos=np.asarray(b["default_com_position_world_m"])
                rest_q=Rotation.from_quat(b["default_inertial_quaternion_world_xyzw"])
                world=live[p["core_body_index"]]
                live_q=Rotation.from_quat(world["quaternion_xyzw"])
                t=(live_q*rest_q.inv()).apply(np.asarray(p["world_m"])-rest_pos)+world["position_m"]
                endpoint.append({"route":m["name"],"endpoint":owner,"owner_body":p["body"],
                                 "native_mm":1000*distances(t,ref_tris),
                                 "common_mm":1000*distances(t,common_tris),
                                 "atlas_rest_mm":1000*distances(p["world_m"],rest[f])})
        report.append({"stable_id":stable,"endpoints":endpoint,
                       "native_extent_m":[native.min(axis=0).tolist(),native.max(axis=0).tolist()],
                       "common_extent_m":[common.min(axis=0).tolist(),common.max(axis=0).tolist()],
                       "common_sampled_skin_winding":descriptive_winding(common,skin_f64),
                       "native_sampled_skin_winding":descriptive_winding(native,skin_f64),
                       "registered_live_poses":flag})
    print(json.dumps({"outer_skin_index_topology":topology(skin_faces),"surfaces":report},indent=2))
