"""Locate flips and collapsed Float32 faces across every sampled atlas overrun."""
import json
from pathlib import Path

import numpy as np

from cardiac_partition_motion import load
from inspect_cardiac import sha

HERE=Path(__file__).resolve().parent

def area(xyz,faces):
    p=xyz[faces].astype(float)
    return np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])

def main():
    path=HERE/"cardiac-range.npz"
    sample=np.load(path)
    data,coeff,field,receipt=load()
    ranges=field["vertex_ranges"]+field["passive_attachment_ranges"]
    source=receipt.parent/field["domain_boxes"]["path"]
    assert sha(source)==field["domain_boxes"]["sha256"]
    original=np.fromfile(source,"<f4").reshape(-1,2,8)[:,:,:7]
    out={"trajectory_sha256":sha(path),"field_sha256":field["map"]["sha256"],
         "anatomy_sha256":field["anatomy_payload_sha256"],"runtime_boxes_sha256":sha(source),
         "limit":"Discrete authoring pose sweep in local coordinates. No collision, containment, wall-interface, or native arithmetic certificate.",
         "samples":[]}
    for j in np.flatnonzero(sample["miss"]>1e-7):
        q=sample["trajectory"][j].astype("<f4").astype(float)
        box=sample["nearest"][j]
        ref=np.clip(q,*original[box])
        rec={"time_s":float(sample["time"][j]),"nearest_box":int(box),"q":q.tolist(),
             "degenerate_faces":0,"nonpositive_dots":0,"first":[],"minimum_area_ratio":1.0}
        for row in ranges:
            key,first,count=(row[k] for k in ("stable_id","first_vertex","vertex_count"))
            s=data[key]
            p=s["vertices6"][:,:3]
            d=coeff[first:first+count]
            pts=(p+np.einsum("vqd,q->vd",d,q,dtype=float)).astype("<f4")
            r0=(p+np.einsum("vqd,q->vd",d,ref,dtype=float)).astype("<f4")
            a,b=(area(x,s["faces"]) for x in (r0,pts))
            deg=np.einsum("ij,ij->i",b,b)==0
            ref2=np.einsum("ij,ij->i",a,a)
            valid=ref2>0
            alignment=np.einsum("ij,ij->i",a,b)
            flips=(alignment<=0) & (valid & ~deg)
            if np.any(valid):
                min_ratio=float(np.min(np.sqrt(np.einsum("ij,ij->i",b,b)[valid]/ref2[valid])))
                rec["minimum_area_ratio"]=min(rec["minimum_area_ratio"],min_ratio)
            rec["degenerate_faces"]+=int(deg.sum())
            rec["nonpositive_dots"]+=int(flips.sum())
            if deg.any() or flips.any():
                rec["first"].append({"id":key,"degenerate":np.flatnonzero(deg)[:8].tolist(),
                                     "nonpositive":np.flatnonzero(flips)[:8].tolist()})
        out["samples"].append(rec)
        print(rec["time_s"],rec["degenerate_faces"],rec["nonpositive_dots"],rec["minimum_area_ratio"],flush=True)
    (HERE/"cardiac-swept-samples.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    main()
