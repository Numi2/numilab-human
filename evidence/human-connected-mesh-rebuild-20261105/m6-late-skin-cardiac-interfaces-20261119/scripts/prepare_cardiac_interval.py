"""Bind the right-heart interval probe to the exact registered native source."""
import json
from pathlib import Path
import struct

import numpy as np

from cardiac_partition_motion import load
from inspect_cardiac import sha

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "resting-clearance-and-atrium-002"

def intervals():
    first, worst = (json.loads((PRIOR/f"cardiac-{name}-geom.json").read_text())
                    for name in ("failed","worst"))
    a=np.asarray(first["authorized_boundary_q"],dtype="<f4")
    b=np.asarray(first["candidate_q"],dtype="<f4")
    c=np.asarray(worst["candidate_q"],dtype="<f4")
    def around(q,r):
        return np.nextafter(q - r,np.float32(-np.inf)),np.nextafter(q + r,np.float32(np.inf))
    return [
        ("old_in_range_boundary", a, a),
        ("first_rejection_exact_coordinates", b, b),
        ("first_rejection_radius_10e-6", *around(b, np.float32(.000010))),
        ("bridge_to_first_rejection", np.minimum(a,b), np.maximum(a,b)),
        ("deepest_sample_exact_coordinates", c,c),
        ("deepest_sample_radius_10e-6", *around(c, np.float32(.000010))),
    ]

def prove_closed(faces):
    edges=faces[:,[0,1,1,2,2,0]].reshape(-1,2)
    undirected=np.sort(edges, axis=1)
    _, inv, cnt=np.unique(undirected,axis=0,return_inverse=True,return_counts=True)
    assert (cnt==2).all() and (edges[:,0]!=edges[:,1]).all()
    winding=np.bincount(inv,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    assert not winding.any()

def main():
    surfaces, coeff, field, receipt = load()
    specs={r["stable_id"]:r for r in field["vertex_ranges"]}
    a,b=[surfaces[k]["vertices6"][:,:3] for k in (318,319)]
    xyz,index=np.unique(np.vstack((a,b)),axis=0,return_inverse=True)
    ra,rv=np.split(index,[len(a)])
    maps=np.empty((len(xyz),7,3),dtype="<f4")
    f=[]
    for k,ids in ((318,ra),(319,rv)):
        row=specs[k]
        start=row["first_vertex"]
        values=coeff[start:start+row["vertex_count"]]
        assert len(np.unique(ids))==len(ids)
        face = ids[surfaces[k]["faces"]].astype("<u4")
        prove_closed(face)
        if k==319:
            weld,index_a,_=np.intersect1d(ids,ra,return_indices=True)
            assert np.array_equal(values[index_a],maps[weld])
        maps[ids]=values
        f.append(face)
    unique_rv={tuple(sorted(t)):t for t in f[1]}
    assert len(unique_rv)==len(f[1])
    shared=[]
    keep=[]
    for i,a in enumerate(f[0]):
        b=unique_rv.get(tuple(sorted(a)))
        if b is None:
            keep.append(i)
        else:
            assert any(np.array_equal(np.roll(b,k),a[::-1]) for k in range(3))
            shared.append(i)
    assert len(shared)==690
    merged=np.vstack((f[0][keep],f[1])).astype("<u4")
    windows=intervals()
    path=HERE/"right-heart-interval-input.bin"
    with path.open("wb") as out:
        out.write(struct.pack("<III",len(xyz),len(merged),len(windows)))
        np.concatenate((xyz[:,None,:],maps),axis=1).astype("<f4").tofile(out)
        merged.tofile(out)
        for _,lo,hi in windows:
            lo.astype("<f4").tofile(out)
            hi.astype("<f4").tofile(out)
    np.savez_compressed(HERE/"right-heart-interval-face-lineage.npz",
        ra_source_face_rows=keep,rv_source_face_rows=np.arange(len(f[1])),shared_ra_faces=shared)
    provenance=dict(source_receipt=str(receipt),anatomy_sha256=field["anatomy_payload_sha256"],
        common_map_sha256=field["map"]["sha256"],geometry_input_sha256=sha(path),
        surfaces=[318,319], welded_vertices=len(weld), shared_source_faces=len(shared),
        each_surface_closed_and_oriented=True, vertices=len(xyz),faces=len(merged),
        windows=[dict(name=name,lower=lo.tolist(),upper=hi.tolist()) for name,lo,hi in windows])
    (HERE/"right-heart-interval-input.json").write_text(json.dumps(provenance,indent=2)+"\n")
if __name__=="__main__":
    main()
