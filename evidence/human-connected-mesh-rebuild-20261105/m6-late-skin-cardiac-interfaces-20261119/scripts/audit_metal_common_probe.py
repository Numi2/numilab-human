"""Audit the registered right-heart union using emitted Metal Float32 positions."""
import csv
import json
from pathlib import Path
import struct

import numpy as np
from numilab_human import cardiac_cavity_intersections as exact

from inspect_cardiac import sha

HERE = Path(__file__).resolve().parent

def main():
    geometry=HERE/"right-heart-interval-input.bin"
    native=HERE/"right-heart-metal-poses.bin"
    dims=np.fromfile(geometry,"<u4",count=3)
    n,nf,windows_n=map(int,dims)
    rows=np.fromfile(geometry,"<f4",offset=12,count=n*24).reshape(n,8,3)
    faces=np.fromfile(geometry,"<u4",offset=12+96*n,count=nf*3).reshape(nf,3)
    windows=np.fromfile(geometry,"<f4",offset=12+96*n+12*nf).reshape(windows_n,2,7)
    witnesses=list(csv.DictReader((HERE/"right-heart-interval-witnesses.csv").open()))
    provenance=json.loads((HERE/"right-heart-interval-input.json").read_text())
    reports=[]
    with native.open("rb") as file:
        while raw:=file.read(4):
            step,=struct.unpack("<I",raw)
            points=np.frombuffer(file.read(n*12),dtype="<f4").reshape(n,3)
            q=windows[step,0]
            assert np.array_equal(q,windows[step,1])
            v1,v2=rows[:,0].copy(),rows[:,0].copy()
            for b,d in zip(q,rows[:,1:].transpose(1,0,2)):
                v1=(v1.astype(float)+b.astype(float)*d.astype(float)).astype("<f4")
                v2+=b*d
            central,_=exact.float32_surface_records(points,faces)
            selected=[r for r in witnesses if int(r["window"])==step and int(r["second_face"])>=0]
            checked=exact._audit_record_pairs(
                ((central[int(r["first_face"])],central[int(r["second_face"])]) for r in selected),
                same_surface=True)
            print("interval witness audit",step,checked["count"],flush=True)
            all_sites=exact._audit_pair_prepared_first(exact._prepare_surface_aabb(central),central,same_surface=True)
            record=dict(index=step,label=provenance["windows"][step]["name"],
                float_components_different_fused=int(np.count_nonzero(points!=v1)),
                float_components_different_unfused=int(np.count_nonzero(points!=v2)),
                compact_witnesses=checked,all_pairs=all_sites)
            reports.append(record)
            print(record["label"],record["float_components_different_fused"],
                record["float_components_different_unfused"],checked["count"],all_sites["count"],flush=True)
    result=dict(device="Apple M6",scope="Right-heart original geometry quotient in its registered local frame. Same seven-term shader expression compiled with the retained flags. No accepted body steps, wall clearance, or atlas admission.",
        input_sha256=sha(geometry),metal_positions_sha256=sha(native),states=reports)
    (HERE/"right-heart-metal-exact.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":
    main()
