"""Verify the actual split-precision Metal field over all original cardiac source rows."""
from pathlib import Path
import json
import struct

import numpy as np

from fit_cardiac_volume_grid import HERE, view, hash_file

def main():
    file=HERE/"original-native-20.bin"
    output=HERE/"original-native-20-m6.bin"
    rows,_,windows=view(file)
    assert np.array_equal(windows[:,0],windows[:,1])
    n=len(rows)
    record_size=4+n*24
    assert output.stat().st_size==record_size*len(windows)
    per_pose=[]
    with output.open("rb") as stream:
        for i,q in enumerate(windows[:,0]):
            assert struct.unpack("<I",stream.read(4))[0]==i
            parts=np.fromfile(stream,"<f4",count=n*6).reshape(n,2,3)
            assert np.isfinite(parts).all()
            expected=rows[:,0].astype(float)
            magnitude=np.abs(expected)
            for j in range(7):
                term=rows[:,j+1].astype(float)*float(q[j])
                expected+=term
                magnitude+=np.abs(term)
            reference_error=magnitude*2.**-48
            allowance=np.maximum(magnitude*2.**-40,2.**-122)
            rounded=parts[:,0].astype(float)+parts[:,1].astype(float)
            # The host summation error is added explicitly before comparing
            # the native coordinates to the declared solver arithmetic reserve.
            upper=np.abs(expected-rounded)+reference_error
            limit=(upper<=allowance)
            assert np.all(limit),(i,int((~limit).sum()),upper.max(),allowance.max())
            row=dict(window=i,all_finite=True,maximum_absolute_error_bound_m=float(upper.max()),
                     maximum_fraction_of_arithmetic_bound=float((upper/allowance).max()),
                     maximum_low_component_m=float(np.abs(parts[:,1]).max()))
            per_pose.append(row)
            print(row,flush=True)
    receipt=dict(source_sha256=hash_file(file),metal_buffer_sha256=hash_file(output),
                 metal_shader_sha256=hash_file(HERE/"cardiac-compensated-field.metallib"),
                 device=(HERE/"gpu-compensation.log").read_text().strip(),
                 all_vertices=len(rows),poses=per_pose,
                 conservative_relative_arithmetic_reserve=2.**-40,
                 changes_to_pressure_law=False)
    (HERE/"gpu-compensation-verification.json").write_text(json.dumps(receipt,indent=2)+"\n")

if __name__=="__main__":
    main()
