"""Retain exact dominant native skin donors while the owner's mapping lease is alive."""
from pathlib import Path
import json
import numpy as np
from run_bed_work import digest


def observe(receipt,out):
    state=json.loads(receipt.read_text());meta=state["skin_source_mapping"]
    if meta["influence_layout"]!="vertex-major":raise ValueError("Unknown source layout")
    def bind(k,dt):
        row=meta[k];p=Path(row["path"])
        if p.stat().st_size!=row["bytes"] or digest(p)!=row["sha256"]:raise ValueError("Changed native mapping")
        return np.memmap(p,mode="r",dtype=dt)
    maps=bind("vertex_map",np.dtype([("i","<u4",4),("p","<f4",4),("r","<f4",4)]))
    weights=bind("influences",np.dtype([("p","<f4",4),("n","<f4",4),("b","<u4",4)]))
    starts,counts=maps["i"][:,:2].astype("i8").T
    if len(maps)!=meta["source_vertex_count"] or min(counts)<=0:raise ValueError("Unbound source vertex")
    starts=starts-meta["influence_first_index"]
    offsets=np.r_[0,np.cumsum(counts)[:-1]]
    idx=np.arange(sum(counts))+np.repeat(starts-offsets,counts); rec=weights[idx]
    b=rec["b"][:,0].astype("i8");w=rec["p"][:,3].astype(float)
    if max(b)>=state["accepted_body_pose_count"] or np.max(np.abs(np.add.reduceat(w,offsets)-1))>1e-5:raise ValueError("Bad skin weights")
    uid=np.repeat(np.arange(len(maps)),counts)*256+b
    order=np.argsort(uid,kind="stable");keys,first=np.unique(uid[order],return_index=True);values=np.add.reduceat(w[order],first)
    row,bone=keys//256,keys%256
    ix=np.lexsort((bone,-values,row))
    _,fr=np.unique(row[ix],return_index=True);best=ix[fr]
    np.savez_compressed(out/"skin-donors.npz",owner=bone[best].astype("i4"),fraction=values[best],
                        source_rows=row.astype("i4"),source_bodies=bone.astype("i4"),weights=values)
    probe=17764
    if probe>=len(maps):raise ValueError("Missing registered shoulder witness")
    points=weights[np.arange(starts[probe],starts[probe]+counts[probe])]
    np.savez_compressed(out/"shoulder-terms.npz",body=points["b"][:,0],local=points["p"][:,:3],weight=points["p"][:,3])
    (out/"skin-donors.json").write_text(json.dumps(dict(pack=state["base_pack_file_sha256"],map=meta["vertex_map"]["sha256"],
          influences=meta["influences"]["sha256"],donors=digest(out/"skin-donors.npz")),indent=2)+"\n")
