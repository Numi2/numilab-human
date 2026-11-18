"""Bound local face orientation over compact extensions of the retained cardiac atlas.

This is an authoring screen. An orientation bound alone cannot clear collision
or containment and never writes a native domain-box file.
"""
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from cardiac_partition_motion import load
from inspect_cardiac import sha

HERE = Path(__file__).resolve().parent

def missing_ranges(time, trajectory, missing):
    ids=np.flatnonzero(missing>1e-7)
    for _, group in groupby(enumerate(ids), key=lambda kv: kv[1]-kv[0]):
        selected=[k for _,k in group]
        padding=0.0005
        # Include one complete transition sample at each end.
        portion=trajectory[max(0,selected[0]-1):selected[-1]+2]
        low,high=portion.min(axis=0)-padding,portion.max(axis=0)+padding
        yield {"t_s":[float(time[selected[0]]),float(time[selected[-1]])],
               "lower":low.tolist(),"upper":high.tolist(),"missing_samples":len(selected)}

def bound_faces(xyz, field, faces, box):
    lower=np.asarray(box["lower"])
    upper=np.asarray(box["upper"])
    center=(lower+upper)*.5
    radius=(upper-lower)*.5
    x=xyz+np.einsum("vqd,q->vd",field,center)
    triangle=x[faces]
    a=triangle[:,1]-triangle[:,0]
    b=triangle[:,2]-triangle[:,0]
    cross=np.cross(a,b)
    area=np.linalg.norm(cross,axis=1)
    assert np.all(area>0)
    normal=cross/area[:,None]
    k0=field[faces[:,0]]
    da=field[faces[:,1]]-k0
    db=field[faces[:,2]]-k0
    n_da=np.cross(normal[:,None],da)
    derivative=np.einsum("fc,fqc->fq",b,n_da)+np.einsum("fc,fqc->fq",a,np.cross(db,normal[:,None]))
    linear=np.abs(derivative)@radius
    # Bound the bilinear remainder conservatively using all seven coordinates.
    ra=np.einsum("fqc,q->fc",abs(da),radius)
    rb=np.einsum("fqc,q->fc",abs(db),radius)
    quadratic=np.linalg.norm(ra,axis=1)*np.linalg.norm(rb,axis=1)
    # Standard gamma_n bound with ample allowance for a non-fused multiply
    # and add per basis column. This covers local coordinates only.
    unit_roundoff=2**-24
    gamma=20*unit_roundoff/(1-20*unit_roundoff)
    scale=abs(xyz)+np.einsum("vqd,q->vd",abs(field),np.maximum(abs(lower),abs(upper)))
    err=gamma*np.linalg.norm(scale,axis=1)
    ea=err[faces[:,0]]+err[faces[:,1]]
    eb=err[faces[:,0]]+err[faces[:,2]]
    rounding=(np.linalg.norm(a,axis=1)+np.linalg.norm(ra,axis=1))*eb
    rounding+=(np.linalg.norm(b,axis=1)+np.linalg.norm(rb,axis=1))*ea + ea*eb
    # Reserve 64 binary64 ULPs for the evaluation of the bound.
    bound=area*(1-64*np.finfo(float).eps) - (linear+quadratic+rounding)*(1+64*np.finfo(float).eps)
    return bound/area

def main():
    input_file=HERE/"cardiac-range.npz"
    sample=np.load(input_file)
    boxes=list(missing_ranges(sample["time"],sample["trajectory"],sample["miss"]))
    rows, coeff, field, receipt=load()
    for box in boxes:
        box["surfaces"]=[]
    for r in field["vertex_ranges"]:
        sid,first,count=(r[k] for k in ("stable_id","first_vertex","vertex_count"))
        source=rows[sid]
        xyz=source["vertices6"][:,:3].astype(float)
        d=coeff[first:first+count].astype(float)
        assert len(xyz)==count
        for box in boxes:
            faces=source["faces"]
            ratios=np.concatenate([
                bound_faces(xyz,d,faces[start:start+32768],box)
                for start in range(0,len(faces),32768)
            ])
            bad=np.flatnonzero(ratios<=0)
            box["surfaces"].append({"stable_id":sid, "total_faces":len(faces), "unproven_faces":len(bad),
                                    "minimum_projected_area_ratio_bound":float(ratios.min()),
                                    "first_unproven_rows":bad[:16].tolist()})
        print(sid,len(source["faces"]),flush=True)
    summary={"calibration_sha256":sha(input_file), "source_field_sha256":field["map"]["sha256"],
             "anatomy_sha256":field["anatomy_payload_sha256"], "native_receipt_sha256":sha(receipt),
             "maximum_guarded_ulp_scope":"local mapping arithmetic only",
             "samples_outside":sum(b["missing_samples"] for b in boxes),
             "continuous_intersection_and_containment_status":"unverified",
             "atlas_update_created":False,"boxes":boxes}
    (HERE/"cardiac-orientation-envelope.json").write_text(json.dumps(summary,indent=2)+"\n")
    for box in boxes:
        print(box["t_s"],min(r["minimum_projected_area_ratio_bound"] for r in box["surfaces"]),
              sum(r["unproven_faces"] for r in box["surfaces"]),flush=True)

if __name__=="__main__":
    main()
