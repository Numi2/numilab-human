"""Compare Metal admission against exact dyadic tube inequalities."""
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from fit_cardiac_volume_grid import hash_file

HERE=Path(__file__).resolve().parent
ATLAS=HERE/"cardiac-live-tube-tight-atlas.bin"
COORDINATES=HERE/"cardiac-coordinates-live.bin"


def read_tubes():
    with ATLAS.open("rb") as f:
        nv,nf,nw=struct.unpack("<III",f.read(12))
        f.seek(nv*96+nf*12,1)
        tubes=np.zeros((nw,4,8),dtype="<f4")
        for i in range(nw):
            tubes[i,:,:7]=np.fromfile(f,"<f4",28).reshape(4,7)
            tubes[i,0,7]=struct.unpack("<I",f.read(4))[0]
        assert not f.read(1)
    return tubes


def exact_inside(point,tube):
    values=[Fraction(float(v)) for v in point]
    low,high,minimum,maximum=([Fraction(float(v)) for v in row[:7]] for row in tube)
    pivot=int(tube[0,7])
    t=(values[pivot]-low[pivot])/(high[pivot]-low[pivot])
    if not 0<=t<=1:
        return False
    for i in range(7):
        residual=values[i]-low[i]-t*(high[i]-low[i])
        if not minimum[i]<=residual<=maximum[i]:
            return False
    return True


def run(states,tubes,stem):
    program=HERE/"cardiac-tube-membership-probe"
    shaders=HERE/"cardiac-tube-membership-probe.metallib"
    source=HERE/(stem+"-states.bin")
    limits=HERE/(stem+"-tubes.bin")
    output=HERE/(stem+"-results.bin")
    states.tofile(source)
    tubes.tofile(limits)
    subprocess.run([str(p) for p in (program,shaders,source,limits,output)],cwd=HERE,check=True)
    data=np.fromfile(output,"<u4")
    assert len(data)==len(states)
    accepted=np.flatnonzero(data!=np.uint32(0xffffffff))
    for i in accepted:
        assert exact_inside(states[i,:7], tubes[data[i]]),(i,data[i])
    return data,accepted


def main():
    tubes=read_tubes()
    states=np.fromfile(COORDINATES,"<f4").reshape(-1,16)
    data,accepted=run(states,tubes,"cardiac-live-membership")
    rejected=np.flatnonzero((data==np.uint32(0xffffffff))&(states.view("<u4")[:,8]==5))
    tests=[]
    # Dedicated two-phase geometry exercises exact boundary and outside points,
    # even when an outside point might otherwise lie inside a neighboring tube.
    for i in (0,10,19,32,52,len(tubes)-1):
        t=tubes[i]
        pivot=int(t[0,7])
        phase=(t[0,:7].astype(float)+t[1,:7].astype(float))*.5
        columns=[r for r in range(7) if r!=pivot]
        cases=[phase]
        for j in columns:
            offsets=[float(t[2,j]),float(t[3,j])]
            for side,offset in zip((-1,1),offsets):
                p=phase.copy()
                p[j]+=offset+side*2.e-6
                cases.append(p)
                p2=phase.copy()
                p2[j]+=offset-side*.5e-6
                cases.append(p2)
        test=np.zeros((len(cases),16),"<f4")
        test[:,:7]=cases
        test.view("<u4")[:,8]=5
        got,inside=run(test,tubes[i:i+1],f"cardiac-probe-tube-{i:02}")
        assert 0 in inside
        assert len(inside)==1,(i,inside)  # Remaining rows are outside or in the roundoff reserve.
        tests.append(dict(tube=i,admitted=len(inside),tested=len(got)))
    summary=dict(atlas_sha256=hash_file(ATLAS),native_coordinate_sha256=hash_file(COORDINATES),
                 samples=len(data),exterior_samples_admitted=len(accepted),
                 exterior_missing_steps=(rejected+1).tolist(),
                 every_gpu_admission_exact=True,fixture_reports=tests,changes_physical_admission=False)
    (HERE/"cardiac-live-membership.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(len(accepted),"exactly contained roots.",len(rejected),"missing.")
    assert not len(rejected)


if __name__=="__main__":
    main()
