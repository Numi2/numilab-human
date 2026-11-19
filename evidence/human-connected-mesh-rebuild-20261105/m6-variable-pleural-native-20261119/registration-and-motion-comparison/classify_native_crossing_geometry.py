"""Separate transverse intersections from tangencies on an exact frozen capture."""
from collections import Counter
import json
import mmap
from pathlib import Path
import runpy
import sys

import numpy as np

from fast_skin_intersection_audit import signed_distances, owner
from fit_cardiac_volume_grid import HERE, hash_file


SCALE=2.**149
reader=runpy.run_path(str(HERE.parent/"resting-surface-clearance-001/audit.py"))
record=owner.float32_point_lattice_key
audit_integer=owner._triangle_intersection_points_integer


def normal(t):
    return owner._cross(owner._sub(t[1],t[0]),owner._sub(t[2],t[0]))


def layer(positions, triangles, ids):
    faces=triangles[np.asarray(ids,dtype=int)]
    used,inv=np.unique(faces,return_inverse=True)
    keys=[record(positions[v]) for v in used]
    return {int(i):tuple(keys[k] for k in face) for i,face in zip(ids,inv.reshape(-1,3))}


def stats(skin, target, pairs):
    kinds=Counter()
    lengths=[]
    for i,j in pairs:
        a,b=skin[i],target[j]
        sa=signed_distances(a,normal(a),b)
        sb=signed_distances(b,normal(b),a)
        crossings=audit_integer(a,b)
        assert crossings
        if not any(sa):
            kind="coplanar"
        elif min(sa)<0<max(sa) and min(sb)<0<max(sb) and len(set(crossings))>=2:
            kind="transverse_segment"
        else:
            kind="boundary_contact_or_ambiguous"
        kinds[kind]+=1
        p=np.array(crossings,dtype=float)/SCALE
        delta=p[:,None]-p[None,:]
        lengths.append(float(np.sqrt(np.max(np.sum(delta*delta,axis=2)))*1000.))
    return kinds,lengths


def main():
    pack=Path(sys.argv[1]).resolve()
    report=Path(sys.argv[2]).resolve()
    frozen=json.loads(report.read_text())
    receipt=json.loads(pack.with_suffix(".receipt.json").read_text())
    assert hash_file(pack)==frozen["accepted_pack_sha256"]==receipt["pack_file_sha256"]
    bodies={b["body_index"]:b for b in receipt["accepted_body_poses"]}
    tables={(r["semantic"],r["stable_id"]):r["triangle_pairs"] for r in frozen["surfaces"] if r["count"]}
    skin_rows=sorted({p[0] for pairs in tables.values() for p in pairs})
    rows=[]
    counts=Counter()
    quantiles=[0.,.5,.9,.99,1.]
    all_lengths=[]
    with pack.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        s=list(reader["primitives"](mapped,bodies,skin_only=True))
        assert len(s)==1
        skin=layer(s[0][2],s[0][3],skin_rows)
        for key,_,x,faces in reader["primitives"](mapped,bodies):
            if key not in tables:
                continue
            pairs=tables.pop(key)
            inside=layer(x,faces,sorted({p[1] for p in pairs}))
            kinds, length=stats(skin,inside,pairs)
            rows.append(dict(semantic=key[0], stable_id=key[1], count=len(pairs),
                             kinds=dict(kinds), span_mm=np.quantile(length,quantiles).tolist()))
            all_lengths.extend(length)
            counts.update(kinds)
    assert not tables
    output=dict(pack=receipt["pack_file_sha256"],exact_audit=hash_file(report),
                counts=dict(counts),quantiles=quantiles,
                span_mm=np.quantile(all_lengths,quantiles).tolist(),
                per_bank=sorted(rows,key=lambda r:-r["count"]),
                limit="Intersection span measures the exact intersection extent. It is distinct from penetration depth.")
    (HERE/"retained-native-crossing-geometry.json").write_text(json.dumps(output,indent=2)+"\n")
    print({k:v for k,v in output.items() if k!="per_bank"},flush=True)


if __name__=="__main__":
    main()
