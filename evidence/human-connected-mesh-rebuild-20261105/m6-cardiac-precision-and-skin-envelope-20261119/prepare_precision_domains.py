"""Prepare original native cardiac topology and correlated precision probes."""
from pathlib import Path
import json
import struct

import numpy as np

from fit_cardiac_volume_grid import HERE, INPUT, view, hash_file

def emit(tag,rows,faces,windows):
    target=HERE/f"{tag}.bin"
    with target.open("xb") as f:
        f.write(struct.pack("<III",len(rows),len(faces),len(windows)))
        rows.tofile(f)
        faces.tofile(f)
        windows.astype("<f4").tofile(f)
    return hash_file(target)

def main():
    orig=HERE.parent/"resting-clearance-and-atrium-003/common-cardiac-input.bin"
    source,faces,_=view(orig)
    basis,tight,windows=view(INPUT)
    assert np.array_equal(source,basis)
    assert np.array_equal(windows[:,0],windows[:,1])
    q=windows[:,0].copy()
    raw=emit("original-native-20",source,faces,windows)
    off=np.float32(1.e-6)
    padded=np.stack((np.nextafter(q-off,-np.inf,dtype="<f4"),np.nextafter(q+off,np.inf,dtype="<f4")),axis=1)
    box=emit("original-native-20-radius1e-6",source,faces,padded)
    (HERE/"original-native-20-inputs.json").write_text(json.dumps(dict(
        original_source_sha256=hash_file(orig),source_inputs_same_coefficients=True,
        original_face_rows_replaced_in_previous_tight_trial=int(np.any(faces!=tight,axis=1).sum()),
        point_probes_sha256=raw,box_probes_sha256=box,padding_each_coordinate=float(off)),indent=2)+"\n")

if __name__=="__main__":
    main()
