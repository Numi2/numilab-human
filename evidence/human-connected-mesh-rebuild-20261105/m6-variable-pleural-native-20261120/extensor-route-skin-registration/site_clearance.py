"""Check source tendon sites against the current skin, fat and bone envelopes."""
import json
from pathlib import Path
import sys

import numpy as np

from anchors import extensor
from atlas_attachment_check import distances
from branch_fit import endpoint_world
from route_segment_screen import _myosim_surface_route_context, digest
from screen import banks, clearance, manifest, skin, skin_faces


def winding(point, tri):
    v=clearance._generalized_winding_number(point,tri)
    assert np.isfinite(v)
    return v

def layer(point, tri):
    return {"winding":winding(point,tri),"surface_distance_mm":1000*distances(point,tri)}

if __name__=="__main__":
    model=Path(sys.argv[2])
    proof=manifest["source"]
    assert digest(model/proof["myosim_manifest"]["file"])==proof["myosim_manifest"]["sha256"]
    bodies,routes,_=_myosim_surface_route_context(model,proof["myosim_source_archive_sha256"])
    keys=[(51007,4294966545),(51007,4294966546)]
    tri={"outer_skin":skin[skin_faces].astype(float)}
    for name,key in zip(("inner_skin","deep_fat"),keys,strict=True):
        points,face=banks[key]
        tri[name]=points[face].astype(float)
    report=[]
    for stable in (137,138):
        _,_,_,_,_,live,desc,_=extensor(stable)
        surface={"stable_id":stable,"routes":[]}
        for r in desc["matched_muscles"]:
            route=routes[r["name"]]
            sites=[p for p in route["route_points"] if p["kind"]=="site"]
            marks=[]
            for k,point in enumerate(sites):
                xyz=endpoint_world(point,bodies,live)
                result={name:layer(xyz,triangles) for name,triangles in tri.items()}
                marks.append({"site_order":k,"owner":point["body"],"world_m":xyz.tolist(),
                              "is_distal_attachment":k==len(sites)-1,**result})
            surface["routes"].append({"name":r["name"],"sites":marks})
        report.append(surface)
    print(json.dumps({"scope":"Descriptive winding on fixed native Float32 surfaces and nearest-triangle distances. Open ocular boundaries prevent closed-solid certification.","surfaces":report},indent=2))
