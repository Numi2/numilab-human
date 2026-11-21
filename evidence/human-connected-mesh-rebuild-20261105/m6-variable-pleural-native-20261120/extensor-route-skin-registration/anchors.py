"""Compare stable single-body transports to the captured extensor deformation."""
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from atlas_attachment_check import distances, descriptive_winding
from route_segment_screen import value_weights
from screen import (
    banks, bindings, bind_transforms, hits, indices, linear, manifest,
    pair_index, poses, records, registered, skin, skin_faces, vertices,
)
from source_topology import self_pairs


def extensor(stable):
    b0,nb,v0,nv,i0,ni,_,_=map(int,records[records[:,6]==stable][0])
    body=bindings[b0:b0+nb]
    source_faces=(indices[i0:i0+ni]-v0).reshape(-1,3)
    used=np.unique(source_faces)
    source_vertices=vertices[v0:v0+nv][used]
    faces=np.searchsorted(used,source_faces)
    w=value_weights(source_vertices,nb)
    source=source_vertices["position"].astype(float)
    xyz,native_faces=banks[51005,stable]
    assert np.array_equal(faces,native_faces)
    ids=body["body"].tolist()
    choices=[]
    for use_registered, body_map in ((True,registered),(False,poses)):
        if all(i in body_map for i in ids):
            sc,q,t=bind_transforms(body,use_registered=use_registered)
            positions=np.array([Rotation.from_quat(rot).apply(source)*scale+offset
                                for scale,rot,offset in zip(sc,q,t,strict=True)])
            decoded=np.einsum("bn,bnd->nd",w,positions)
            choices.append((float(np.linalg.norm(decoded-xyz,axis=1).max()), positions, body_map))
    error,parts,body_map=min(choices,key=lambda r:r[0])
    assert error<2e-6
    desc=next(r for r in manifest["source"]["surfaces"] if r["stable_id"]==stable)
    assert ids==[r["core_body_index"] for r in desc["body_bindings"]]
    return xyz,faces,w,parts,source,body_map,desc,used


def inspect(stable,skin_index, skin_triangles):
    xyz,f,w,parts,src,*rest=extensor(stable)
    result=[]
    for i,p in enumerate(parts[:3]):
        proposal=p.astype("<f4")
        row={"owner":i,"body":rest[-2]["body_bindings"][i]["myosim_body"],
             "crossings":len(hits(skin_index,proposal,f)),
             "self_pairs":len(self_pairs(proposal,f)),
             "maximum_shift_mm":float(np.linalg.norm(proposal-xyz,axis=1).max()*1000),
             "skin_winding":descriptive_winding(proposal,skin_triangles),
             "extent_m":[proposal.min(axis=0).tolist(),proposal.max(axis=0).tolist()],
             "median_translation_mm":(np.median(proposal-xyz,axis=0)*1000).tolist()}
        result.append(row)
    q=np.linalg.svd(src-src.mean(axis=0),full_matrices=False)[2][0]
    x=src@q
    if np.mean(x[w[0]>.3])>np.mean(x):
        x=-x
    x=(x-x.min())/np.ptp(x)
    sections=[]
    for s in range(10):
        ids=np.flatnonzero((x>=s/10)&(x<=(s+1)/10))
        if not len(ids):
            continue
        sections.append({"t":s,"count":len(ids),"mean_humerus_radius_lunate_weight":w[:3,ids].mean(axis=1).tolist(),
                         "distal_fraction":float(w[3:,ids].sum(axis=0).mean()),
                         "median_native_xyz_m":np.median(xyz[ids],axis=0).tolist(),
                         "median_H_R_registration_gap_mm":float(np.median(np.linalg.norm(parts[0,ids]-parts[1,ids],axis=1))*1000),
                         "median_R_W_registration_gap_mm":float(np.median(np.linalg.norm(parts[1,ids]-parts[2,ids],axis=1))*1000)})
    return {"stable_id":stable,"anchors":result,"sections":sections}

if __name__=="__main__":
    index=pair_index(skin,skin_faces)
    triangles=skin[skin_faces].astype(float)
    print(json.dumps([inspect(i,index,triangles) for i in (137,138)],indent=2))
