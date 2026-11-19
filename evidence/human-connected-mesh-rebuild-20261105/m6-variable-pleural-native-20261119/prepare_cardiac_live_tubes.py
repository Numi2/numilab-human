"""Bound every exterior root and the adjacent motion with matched GPU coordinates."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from fit_cardiac_volume_grid import hash_file


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference",type=Path)
    cli.add_argument("gpu_coordinates",type=Path)
    cli.add_argument("first_native_rejection",type=Path)
    cli.add_argument("destination",type=Path)
    args=cli.parse_args()
    raw=np.fromfile(args.gpu_coordinates,"<f4").reshape(-1,16)
    status=raw.view("<u4")[:,8]
    assert len(raw)==8000 and np.isin(status,(0,5)).all()
    first=np.flatnonzero(status==5)
    assert len(first) and 0<first[0]<first[-1]<len(raw)-1
    rejected=json.loads(args.first_native_rejection.read_text())["candidate_q"]
    q=raw[:,:7].astype(float)
    assert np.array_equal(q[first[0]],rejected)
    groups=np.split(first,np.flatnonzero(np.diff(first)>1)+1)
    windows=[]
    for g in groups:
        i=g[0]-1
        final=g[-1]+1
        while i<final:
            # Adjacent intervals meet at the same Float32 physical sample.
            j=min(i+8,final)
            while True:
                a,b=q[[i,j]]
                d=b-a
                pivot=int(np.argmax(abs(d)))
                assert d[pivot]!=0.
                t=(q[i:j+1,pivot]-a[pivot])/d[pivot]
                residual=q[i:j+1]-a-np.outer(t,d)
                if np.min(t)>=0 and np.max(t)<=1 and np.max(abs(residual))<=1.8e-4:
                    break
                j-=1
                assert j>i
            pad=2.e-6
            lo=np.nextafter((np.minimum(residual.min(0),0)-pad).astype("<f4"),-np.inf)
            hi=np.nextafter((np.maximum(residual.max(0),0)+pad).astype("<f4"),np.inf)
            lo[pivot]=hi[pivot]=0.
            windows.append((int(i+1),int(j+1),a,b,lo,hi,pivot))
            i=j
    with args.reference.open("rb") as source,args.destination.open("xb") as out:
        nv,nf,_=struct.unpack("<III",source.read(12))
        out.write(struct.pack("<III",nv,nf,len(windows)))
        out.write(source.read(nv*96+nf*12))
        assert source.tell()==12+nv*96+nf*12
        for _,_,a,b,lo,hi,pivot in windows:
            np.asarray((a,b,lo,hi),dtype="<f4").tofile(out)
            out.write(struct.pack("<I",pivot))
    report=dict(binary_sha256=hash_file(args.destination),source_sha256=hash_file(args.reference),
                target_q_sha256=hash_file(args.gpu_coordinates),first_exterior_step=int(first[0]+1),
                exact_native_first_rejection=True,missing_steps=(first+1).tolist(),
                windows=[[i,j,0.002*i,0.002*j] for i,j,*_ in windows],native_qualified=False)
    args.destination.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(len(first),"exterior roots covered by",len(windows),"proof intervals.")
    print("Exact native rejection step",first[0]+1,"verified.")


if __name__=="__main__":
    main()
