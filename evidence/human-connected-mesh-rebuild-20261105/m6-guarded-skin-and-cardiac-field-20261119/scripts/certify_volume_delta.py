"""Check closed source-row volume and derivative bounds for the fitted field."""
import ast
from collections import defaultdict
import json
import itertools
import math
from pathlib import Path
import sys

import numpy as np
from correct_cardiac_field_soft import geometry,position

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/"resting-clearance-and-atrium-003"
sys.path.insert(0,str(PRIOR))
from prepare_cardiac_interval import load,prove_closed,sha

OWNER=PRIOR/"retained_audit_cardiac_flip_volume.py"
selected=[n for n in ast.parse(OWNER.read_bytes()).body if isinstance(n,ast.FunctionDef) and
          n.name in ("tetra_volume_delta_polynomial","relative_volume_bounds")]
assert len(selected)==2
exec(compile(ast.Module(body=selected,type_ignores=[]),str(OWNER),"exec"))


def source_faces(surfaces,field):
    regions=[r["stable_id"] for r in field["vertex_ranges"]]
    xyz=np.concatenate([surfaces[k]["vertices6"][:,:3] for k in regions],axis=0)
    unique,reverse=np.unique(xyz,axis=0,return_inverse=True)
    faces={}
    for r in field["vertex_ranges"]:
        k,start,n=(r[x] for x in ("stable_id","first_vertex","vertex_count"))
        faces[k]=reverse[start:start+n][surfaces[k]["faces"]]
    return unique,faces


def main():
    path=Path(sys.argv[1])
    original=PRIOR/"common-cardiac-input.bin"
    _,rows,baseline,_=geometry(original)
    _,candidate,new,windows=geometry(path)
    changed=np.any(candidate!=rows,axis=(1,2))
    assert np.array_equal(candidate[:,0],rows[:,0])
    ancestry=np.load(PRIOR/"common-cardiac-face-owners.npz")
    owners=ancestry["owner"]
    source_row=ancestry["face_row"]
    faces_changed=np.any(baseline!=new,axis=1)
    surfaces,_,field,_=load()
    xyz,per_owner=source_faces(surfaces,field)
    assert np.array_equal(xyz,rows[:,0])
    ref={k:v for k,v in zip(per_owner,field["source_reference_volumes_m3"])}
    delta=[]
    origin=np.zeros((8,3))
    for owner,indices in per_owner.items():
        updated=indices.copy()
        ids=np.flatnonzero(faces_changed & (owners==owner))
        assert np.array_equal(baseline[ids],indices[source_row[ids]])
        updated[source_row[ids]]=new[ids]
        prove_closed(updated)
        differs=np.any(changed[indices],axis=1)|np.any(indices!=updated,axis=1)
        coefficients=defaultdict(list)
        reserve=0.0
        for src,dst in zip(indices[differs],updated[differs]):
            for side,tri,basis in ((-1,src,rows),(1,dst,candidate)):
                c,p=tetra_volume_delta_polynomial(origin,*basis[tri])
                reserve+=p
                for degree,value in c.items():
                    coefficients[degree].append(side*value)
        poly={power:math.fsum(v) for power,v in coefficients.items()}
        bound,derivative,_=relative_volume_bounds(poly,reserve,ref[owner])
        zero=(0,)*7
        assert abs(poly.get(zero,0.))<ref[owner]*2.e-7
        direct_error=[]
        for q in windows[:,0][:3]:
            mon=lambda x: math.prod(float(a)**b for a,b in zip(q,x))
            predicted=math.fsum(value*mon(power) for power,value in poly.items())
            before=position(rows,q)[indices[differs]]*.001
            after=position(candidate,q)[updated[differs]]*.001
            def aggregate(t):
                return math.fsum((np.sum(t[:,0]*np.cross(t[:,1],t[:,2]),axis=1)/6).tolist())
            observed=aggregate(after)-aggregate(before)
            direct_error.append(abs(observed-predicted)/ref[owner])
        report=dict(owner=owner,source_ref_m3=ref[owner],changed_faces=int(differs.sum()),
            coefficient_delta_uniform_rel_bound=bound,
            jacobian_entry_delta_uniform_rel_bound=derivative,
            maximum_fused_sample_delta_error=float(max(direct_error)),
            coefficients=[dict(exponents=k,delta_m3=v) for k,v in sorted(poly.items()) if v])
        delta.append(report)
        print({k:v for k,v in report.items() if k!="coefficients"},flush=True)
    result=dict(candidate_sha256=sha(path),baseline_sha256=sha(original),
                maximum_uniform_relative_volume_change=max(x["coefficient_delta_uniform_rel_bound"] for x in delta),
                domain="[-1,1]^7",zero_vertices_unchanged=True,all_region_faces_closed=True,
                regions=delta)
    (ROOT/(path.stem+"-volume-delta.json")).write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    main()
