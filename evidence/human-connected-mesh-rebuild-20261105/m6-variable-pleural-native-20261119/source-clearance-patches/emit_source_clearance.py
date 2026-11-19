"""Emit source-authoring repairs through the existing NHSKIN payload owner."""
import hashlib
import json

import numpy as np

from fit_clearance_patches import area, decode, HERE, mesh_edges, clear, owner, hash_file
from fast_skin_intersection_audit import install_for_float32
from trim_skin_witnesses import positioned_records, exact


def witness_pairs(points, faces, altered):
    records=positioned_records(points,faces,np.arange(len(faces)))
    tree=exact._prepare_surface_aabb(records)
    pairs=set()
    for a,b in exact._query_surface_aabb((tree.records[i] for i in altered),tree):
        i,j=sorted((a[3],b[3]))
        if i!=j:
            pairs.add((i,j))
    check=exact._audit_record_pairs(((tree.records[i],tree.records[j]) for i,j in sorted(pairs)),
                                    same_surface=True)
    return set(map(tuple,check["triangle_pairs"]))


def main():
    install_for_float32()
    source=HERE/"connected-skin-diagonals.nhskin"
    proof=json.loads((HERE/"source-clearance-patches.json").read_text())
    assert proof["skin_sha256"]==hash_file(source) and proof["accepted"]
    delta_path=HERE/"source-clearance-patches.npz"
    assert hash_file(delta_path)==proof["displacement_sha256"]
    with np.load(delta_path,allow_pickle=False) as z:
        delta=z["displacement_m"]
    s=decode(source.read_bytes())
    x,f=s["verts"][:,:3],s["inds"]
    assert delta.shape==x.shape and np.isfinite(delta).all()
    e,n=mesh_edges(f)
    fixed=set(np.load(HERE.parent/"resting-clearance-and-atrium-004/bed-section-pins.npz")["vertices"])
    fixed.update(e[n!=2].ravel())
    fixed.update(clear._exact_coordinate_tied_source_vertex_ids(x))
    fixed=sorted(fixed)
    assert len(fixed)==proof["protected_rows"] and not np.any(delta[fixed])
    y=(x.astype(float)+delta).astype("<f4")
    affected=np.flatnonzero(np.any(np.any(y!=x,axis=1)[f],axis=1))
    assert len(affected)
    a,b=area(x[f].astype(float)),area(y[f].astype(float))
    ratio=np.einsum("ij,ij->i",a[affected],b[affected])/np.einsum("ij,ij->i",a[affected],a[affected])
    assert min(ratio)>.675
    old=witness_pairs(x,f,affected)
    new=witness_pairs(y,f,affected)
    report=dict(source_sha256=proof["skin_sha256"],displacement_sha256=proof["displacement_sha256"],
                modified_vertices=int(np.any(y!=x,axis=1).sum()),changed_faces=len(affected),
                protected_rows=len(fixed),minimum_source_area_ratio=float(min(ratio)),
                source_self_before=len(old),source_self_after=len(new),new_source_self_pairs=sorted(new-old))
    print(report,flush=True)
    assert not new-old
    # Preserve supplied normals except where source face geometry changed.
    # Thickness-gated bed rows keep their entire original Float32 record.
    v=s["verts"].copy()
    v[:,:3]=y
    touched=np.setdiff1d(np.unique(f[affected]),fixed)
    n0=np.zeros_like(x,dtype=float)
    change=np.zeros_like(n0)
    for k in range(3):
        np.add.at(n0,f[:,k],a)
        np.add.at(change,f[affected,k],(b-a)[affected])
    scale=np.linalg.norm(n0[touched],axis=1)
    assert np.all(scale>0.)
    n1=v[touched,3:6]+change[touched]/scale[:,None]
    n1/=np.linalg.norm(n1,axis=1)[:,None]
    assert np.isfinite(n1).all()
    v[touched,3:6]=n1.astype("<f4")
    start,end=s["vertoff"],s["indexoff"]
    payload=s["raw"][:start]+v.tobytes()+s["raw"][end:]
    emitted=decode(payload)
    assert np.array_equal(emitted["inds"],f) and np.array_equal(emitted["verts"][fixed],s["verts"][fixed])
    assert np.array_equal(emitted["verts"][:,6:],s["verts"][:,6:])
    dst=HERE/"source-clearance-patches.nhskin"
    with dst.open("xb") as out:
        out.write(payload)
    report["candidate_sha256"]=hashlib.sha256(payload).hexdigest()
    (HERE/"source-clearance-patches-emission.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
