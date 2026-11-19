"""Classify swept candidate witnesses at native and exact unrounded source precision."""
import argparse
from collections import Counter
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from fit_cardiac_volume_grid import HERE, view, hash_file
from numilab_human import cardiac_cavity_intersections as exact

DENOM=1<<149

def f32_scalar(value):
    mant,denom=float(value).as_integer_ratio()
    assert DENOM % denom==0
    return mant*(DENOM//denom)

def packed(points,q):
    xyz=points[:,0].copy()
    for i in range(7):
        xyz=(xyz.astype(float)+points[:,i+1].astype(float)*float(q[i])).astype("<f4")
    return xyz

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("geometry",type=Path)
    ap.add_argument("witnesses",type=Path)
    args=ap.parse_args()
    fields,faces,windows=view(args.geometry)
    assert np.array_equal(windows[:,0],windows[:,1])
    owners=np.load(HERE.parent/"resting-clearance-and-atrium-003/common-cardiac-face-owners.npz")["owner"]
    by_window={i:[] for i in range(len(windows))}
    with args.witnesses.open() as file:
        for w in csv.DictReader(file):
            a,b=int(w["first_face"]),int(w["second_face"])
            if b>=0:
                by_window[int(w["window"])].append((a,b))
    counts=[]
    for i,q in enumerate(windows[:,0]):
        pairs=by_window[i]
        ids=np.unique(np.asarray(pairs,dtype=int))
        hit={"native":set(),"exact_affine_source":set()}
        if len(ids):
            v,nodes=np.unique(faces[ids],return_inverse=True)
            local=fields[v]
            native=[exact.float32_point_lattice_key(x) for x in packed(local,q)]
            factor=[f32_scalar(x) for x in q]
            def exact_xyz(row):
                lattice=[exact.float32_point_lattice_key(x) for x in row]
                return tuple(a*DENOM+sum(f*b for f,b in zip(factor,basis)) for a,*basis in zip(*lattice))
            for name,xyz in (("native",native),("exact_affine_source",list(map(exact_xyz,local)))):
                records={}
                for k,indices in enumerate(nodes.reshape(-1,3)):
                    tri=tuple(xyz[x] for x in indices)
                    face_id=int(ids[k])
                    records[face_id]=(tri,tuple(map(min,zip(*tri))),tuple(map(max,zip(*tri))),
                                      face_id,tuple(int(v[x]) for x in indices))
                flag=exact._audit_record_pairs(((records[a],records[b]) for a,b in pairs),same_surface=True)
                hit[name].update(tuple(v) for v in flag["triangle_pairs"])
        a,b=hit.values()
        banks=Counter("/".join(map(str,sorted((owners[j],owners[k])))) for j,k in a)
        row=dict(window=i,potential_pairs=len(pairs),native_exact=len(a),exact_affine_source=len(b),
                 native_only=len(a-b),exact_source_only=len(b-a),native_banks=dict(sorted(banks.items())))
        counts.append(row)
        print(row,flush=True)
    result=dict(geometry_sha256=hash_file(args.geometry),witness_sha256=hash_file(args.witnesses),
                windows=counts,totals={k:sum(r[k] for r in counts) for k in
                                       ("potential_pairs","native_exact","exact_affine_source","native_only","exact_source_only")})
    (HERE/f"{args.geometry.stem}-precision.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result["totals"],flush=True)

if __name__=="__main__":
    main()
