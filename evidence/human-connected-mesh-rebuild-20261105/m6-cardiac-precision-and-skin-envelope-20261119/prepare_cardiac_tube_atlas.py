"""Construct overlapping local proof intervals for all retained pressure excursions."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("trajectory",type=Path)
    parser.add_argument("first_native_rejection",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--padding",type=float,default=1.2e-5)
    args=parser.parse_args()
    if not 0 < args.padding < 1.e-3:
        parser.error("padding outside the bounded authoring range")
    retained=np.load(args.trajectory)
    coords=retained["trajectory"].astype("<f4").astype(float)
    missing=np.flatnonzero(retained["miss"]>1.e-7)
    assert len(missing) and missing[0]>0 and missing[-1]<len(coords)-1
    actual=np.asarray(json.loads(args.first_native_rejection.read_text())["candidate_q"])
    center=int(np.argmin(np.linalg.norm(coords-actual,axis=1)))
    assert center==missing[0]
    groups=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
    windows=[]
    for group in groups:
        start=group[0]-1
        final=group[-1]+1
        while start<final:
            last=min(final,start+2)
            while True:
                a,b=coords[[start,last]]
                dq=b-a
                axis=int(np.argmax(np.abs(dq)))
                assert dq[axis]!=0
                samples=np.concatenate((coords[start:last+1],
                    actual[None,:] if start<center<last else np.empty((0,7))))
                parameter=(samples[:,axis]-a[axis])/dq[axis]
                residual=samples-a-np.outer(parameter,dq)
                if last==start+1 or (np.all((parameter>=0)&(parameter<=1)) and
                        np.max(abs(residual))<=3.5e-5):
                    break
                last-=1
            # Exact Float32 endpoints are the only channel used for t.
            # Use outward-rounded residual bounds for the other channels.
            low=np.nextafter((residual.min(0)-args.padding).astype("<f4"), -np.inf, dtype="<f4")
            high=np.nextafter((residual.max(0)+args.padding).astype("<f4"), np.inf, dtype="<f4")
            low[axis]=high[axis]=0.
            assert np.all((parameter>=0)&(parameter<=1))
            windows.append((int(start),int(last),a,b,low,high,axis))
            start=last
    with args.reference.open("rb") as source,args.destination.open("xb") as f:
        nv,nf,_=struct.unpack("<III",source.read(12))
        f.write(struct.pack("<III",nv,nf,len(windows)))
        f.write(source.read(nv*96+nf*12))
        assert source.tell()==12+nv*96+nf*12
        for _,_,a,b,low,high,axis in windows:
            np.asarray((a,b,low,high),dtype="<f4").tofile(f)
            f.write(struct.pack("<I",axis))
    start_and_end=[[int(a),int(b),float(retained["time"][a]),float(retained["time"][b])] for a,b,*_ in windows]
    proof=dict(binary_sha256=sha(args.destination),source_sha256=sha(args.reference),
               trajectory_sha256=sha(args.trajectory),padding=args.padding,
               required_samples=len(missing),windows=start_and_end,admitted=False)
    args.destination.with_suffix(".json").write_text(json.dumps(proof,indent=2)+"\n")
    print(f"Prepared {len(windows)} correlated intervals for {len(missing)} exterior calibration samples.")

if __name__=="__main__":
    main()
