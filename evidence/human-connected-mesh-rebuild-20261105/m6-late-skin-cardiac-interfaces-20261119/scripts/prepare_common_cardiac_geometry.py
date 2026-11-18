"""Check all seven source volume surfaces together, preserving their shared partitions."""
import json
from pathlib import Path
import struct

import numpy as np

from prepare_cardiac_interval import intervals, load, prove_closed, sha

HERE = Path(__file__).resolve().parent

def main():
    surfaces, coeff, field, receipt = load()
    targets = field["vertex_ranges"]
    names = [r["stable_id"] for r in targets]
    xyz,src_ids = np.unique(np.concatenate([surfaces[k]["vertices6"][:, :3] for k in names]),
                            axis=0,return_inverse=True)
    common = np.empty((len(xyz),7,3),dtype="<f4")
    written = np.zeros(len(xyz),dtype=bool)
    faces,region_rows,seams=[],[],{}
    volume_faces = []
    for spec in targets:
        region,start,count=(spec[k] for k in ("stable_id","first_vertex","vertex_count"))
        idx=src_ids[start:start+count]
        deltas=coeff[start:start+count]
        assert len(np.unique(idx))==len(idx)
        prior=written[idx]
        assert np.array_equal(common[idx[prior]],deltas[prior])
        common[idx]=deltas
        written[idx]=True
        local=idx[surfaces[region]["faces"]]
        prove_closed(local)
        volume_faces.append(local)
        faces.append(local)
        region_rows.append(np.full(len(local),region,dtype="<u4"))
    output_faces=np.concatenate(faces).astype("<u4")
    output_regions=np.concatenate(region_rows)
    canonical=np.sort(output_faces,axis=1)
    keys,owner,count=np.unique(canonical,axis=0,return_inverse=True,return_counts=True)
    assert count.max()<=2
    sorted_rows=np.argsort(owner,kind="stable")
    twice=(owner[sorted_rows[:-1]]==owner[sorted_rows[1:]])
    first=sorted_rows[:-1][twice]
    second=sorted_rows[1:][twice]
    assert np.all(output_regions[first]!=output_regions[second])
    opposite=output_faces[second,::-1]
    assert np.logical_or.reduce([np.all(output_faces[first]==np.roll(opposite,i,axis=1),axis=1)
                                 for i in range(3)]).all()
    for a,b in zip(output_regions[first],output_regions[second]):
        key=f"{min(a,b)}/{max(a,b)}"
        seams[key]=seams.get(key,0)+1
    survivor=np.ones(len(output_faces),dtype=bool)
    survivor[second]=False
    combined=output_faces[survivor]
    probes=[item for item in intervals() if np.array_equal(item[1],item[2])]
    path=HERE/"common-cardiac-input.bin"
    with path.open("wb") as out:
        out.write(struct.pack("<III",len(xyz),len(combined),len(probes)))
        np.concatenate((xyz[:,None,:],common),axis=1).astype("<f4").tofile(out)
        combined.tofile(out)
        for _,lo,hi in probes:
            lo.astype("<f4").tofile(out)
            hi.astype("<f4").tofile(out)
    np.savez_compressed(HERE/"common-cardiac-face-owners.npz",
        owner=output_regions[survivor],face_row= np.concatenate(
            [np.arange(len(f),dtype="<u4") for f in volume_faces])[survivor])
    result={"source_receipt":str(receipt),"anatomy_sha256":field["anatomy_payload_sha256"],
        "basis_sha256":field["map"]["sha256"],"input_sha256":sha(path),
        "stable_ids":names,"shared_interfaces":seams,
        "xyz_vertices":len(xyz),"original_faces":len(output_faces),
        "distinct_faces":len(combined),"windows":[name for name,_,_ in probes]}
    (HERE/"common-cardiac-input.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)

if __name__=="__main__":
    main()
