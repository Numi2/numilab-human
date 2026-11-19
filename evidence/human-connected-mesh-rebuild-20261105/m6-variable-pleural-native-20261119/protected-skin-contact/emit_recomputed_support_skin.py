"""Emit the exact skin repair after independently checking its source seams."""
import hashlib
import sys

import numpy as np

from run_support_dependency_preflight import HERE, PREVIOUS, read, write, digest
sys.path.insert(0,str(PREVIOUS))
from emit_source_clearance import witness_pairs
from fast_skin_intersection_audit import install_for_float32
from fit_clearance_patches import area, decode, oriented


def main():
    install_for_float32()
    s=PREVIOUS/"source-clearance-patches.nhskin"
    audit=read(HERE/"recooked-clearance-screen.json")
    d=HERE/"recooked-source-delta.npz"
    assert audit["accepted"] and digest(d)==audit["delta_sha256"]
    assert audit["skin_sha256"]==digest(s)
    source=decode(s.read_bytes())
    p=source["verts"][:,:3]
    f=source["inds"]
    with np.load(d,allow_pickle=False) as z:
        delta=z["delta"]
        fixed=z["protected"]
    assert delta.shape==p.shape and np.isfinite(delta).all() and not np.any(delta[fixed])
    v=source["verts"].copy()
    v[:,:3]=(p.astype(float)+delta).astype("<f4")
    moved=np.any(v[:,:3]!=p,axis=1)
    affected=np.flatnonzero(moved[f].any(axis=1))
    assert moved.sum() and oriented(p[f[affected]],v[f[affected],:3])
    prior=witness_pairs(p,f,affected)
    after=witness_pairs(v[:,:3],f,affected)
    assert not after-prior
    a=area(p[f].astype(float))
    b=area(v[f,:3].astype(float))
    a0=np.zeros_like(p,dtype=float)
    shift=np.zeros_like(a0)
    for k in range(3):
        np.add.at(a0,f[:,k],a)
        np.add.at(shift,f[affected,k],(b-a)[affected])
    rows=np.setdiff1d(np.unique(f[affected]),fixed)
    scale=np.linalg.norm(a0[rows],axis=1)
    assert min(scale)>0.
    n=v[rows,3:6]+shift[rows]/scale[:,None]
    n/=np.linalg.norm(n,axis=1)[:,None]
    assert np.isfinite(n).all()
    v[rows,3:6]=n.astype("<f4")
    start,end=source["vertoff"],source["indexoff"]
    payload=source["raw"][:start]+v.tobytes()+source["raw"][end:]
    emitted=decode(payload)
    assert np.array_equal(emitted["inds"],f)
    assert np.array_equal(emitted["verts"][fixed],source["verts"][fixed])
    assert np.array_equal(emitted["verts"][:,6:],source["verts"][:,6:])
    dst=HERE/"recomputed-support-skin.nhskin"
    with dst.open("xb") as out:
        out.write(payload)
    result=dict(source_sha256=digest(s),candidate_sha256=hashlib.sha256(payload).hexdigest(),
                relocated_vertices=int(moved.sum()),faces_checked=len(affected),
                anchors_and_source_thickness_sample_rows=len(fixed),
                rest_pairs_before=len(prior),rest_pairs_after=len(after),
                old_guard_vertices_relocated=audit["changed_legacy_guard_vertices"],
                displacement_sha256=audit["delta_sha256"])
    write(HERE/"recomputed-support-skin-emission.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
