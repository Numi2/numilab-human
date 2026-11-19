"""Build discrete held-out probes and a small interval around a known overrun."""
import itertools
import json
from pathlib import Path
import struct

import numpy as np

from correct_cardiac_field_soft import geometry, ROOT

ARCHIVE=ROOT.parent/"resting-clearance-and-atrium-002/cardiac-range.npz"

def main():
    data=np.load(ARCHIVE)
    windows=data['trajectory']
    time=data['time']
    miss=data['miss']
    indices=np.flatnonzero(miss>1.e-7)
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    representative=[int(group[np.argmax(miss[group])]) for group in groups if len(group)]
    for t in (0.,2.,4.,8.,12.,16.):
        representative.append(int(np.argmin(np.abs(time-t))))
    selected=np.asarray(sorted(set(representative)))
    raw,rows,faces,_=geometry(ROOT/"field-trial-3.bin")
    hold=windows[selected].astype("<f4")
    n=len(rows)
    f=len(faces)
    metadata=[{"t_s":float(time[i]),"sample_index":int(i),"old_box_gap":float(miss[i]),"q":q.tolist()}
              for i,q in zip(selected,hold)]
    lower=np.nextafter(hold[0],np.float32(-np.inf))
    upper=np.nextafter(hold[0],np.float32(np.inf))
    q=np.load(ARCHIVE)['trajectory'][int(np.argmax(miss))].astype("<f4")
    small=np.stack([np.nextafter(q-np.float32(1.e-6),np.float32(-np.inf)),
                    np.nextafter(q+np.float32(1.e-6),np.float32(np.inf))])
    for tag,coords in (("held-out",np.repeat(hold[:,None,:],2,axis=1)),
                       ("small-interval",small[None])):
        (ROOT/f"cardiac-{tag}.bin").write_bytes(struct.pack("<III",n,f,len(coords))+
           raw[12:12+96*n+12*f]+coords.astype("<f4").tobytes())
    (ROOT/"cardiac-held-out.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(len(selected),len(groups),flush=True)

if __name__=="__main__":
    main()
