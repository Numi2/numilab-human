"""Check the probe's binary32 arithmetic against the owner shader."""
import argparse
from pathlib import Path
import struct

import numpy as np

from correct_cardiac_field_soft import geometry

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("probe",type=Path)
    p.add_argument("positions",type=Path)
    args=p.parse_args()
    _,rows,_,windows=geometry(args.probe)
    with args.positions.open("rb") as native:
        seen=0
        for i,(lo,hi) in enumerate(windows):
            if not np.array_equal(lo,hi):
                continue
            assert struct.unpack("<I",native.read(4))[0]==i
            actual=np.frombuffer(native.read(12*len(rows)),"<f4").reshape(-1,3)
            expected=rows[:,0].copy()
            for q,d in zip(lo,rows[:,1:].transpose(1,0,2)):
                expected=(expected.astype(float)+float(q)*d.astype(float)).astype("<f4")
            differ=int(np.count_nonzero(actual.view("<u4")!=expected.view("<u4")))
            print(f"state={i} mismatching_f32_components={differ}")
            assert differ==0
            seen+=1
        assert seen>0 and not native.read(1)

if __name__=="__main__":
    main()
