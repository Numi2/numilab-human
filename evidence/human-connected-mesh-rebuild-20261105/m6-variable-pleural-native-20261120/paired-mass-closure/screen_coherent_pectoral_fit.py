"""Screen a coherent rest-sheet fit under fixed atlas shape and source winding."""
import json
from pathlib import Path
import struct

import numpy as np

root = Path(__file__).resolve().parent
source_id = json.loads((root.parent/"resting-clearance-and-atrium-016/fascia-source-cell-witness.json").read_text())
raw = Path(source_id["asset"]).read_bytes()
_, nr, nodes, cells = struct.unpack_from("<4I", raw, 8)
regions = np.frombuffer(raw, offset=104, count=nr*8, dtype="<u4").reshape(-1,8)
base = 104 + nr*32
core = np.ndarray((nodes,3), dtype="<f4", buffer=raw, offset=base, strides=(32,4)).astype(float)
connectivity = np.frombuffer(raw, offset=base+nodes*32, count=cells*5, dtype="<u4").reshape(-1,5)
frame = np.loadtxt(root/"registered-fascia-audit-run/fascia-rest.csv", delimiter=",", skiprows=1)
candidate = np.loadtxt(root/"paired-offset-fascia-audit-run/fascia-rest.csv", delimiter=",", skiprows=1)
assert np.array_equal(frame[:,0], np.arange(nodes))
assert np.array_equal(candidate[:,0], frame[:,0])
report = []

def quality(p, local_tets, reference_sign):
    x = p[local_tets].astype(float)
    dx = x[:,1:] - x[:,:1]
    det = np.linalg.det(dx) * reference_sign
    lengths = np.linalg.norm(x[:,:,None,:]-x[:,None,:,:],axis=-1).max(axis=(1,2))
    return det / lengths**3

for ri in range(6):
    first, length = map(int, regions[ri,3:5])
    rows = slice(first, first+length)
    selected = connectivity[connectivity[:,4]==ri,:4]-first
    source = core[rows]
    x = source[selected]
    reference_sign = np.sign(np.linalg.det(x[:,1:]-x[:,:1]))
    fixed = frame[rows,4] != 0
    for variant, native in (("original",frame),("paired",candidate)):
        y = native[rows,1:4]
        source_relative, target_relative = source-source.mean(axis=0),y-y.mean(axis=0)
        left, _, right = np.linalg.svd(source_relative.T@target_relative)
        sign = np.linalg.det(left@right)
        left[:,-1] *= sign
        rigid = source_relative@left@right + y.mean(axis=0)
        displacement = y-rigid
        keep = []
        for anchored in (False,True):
            solution = None
            for t in np.linspace(1.,0.,1001):
                new = rigid + t*displacement
                if anchored:
                    new[fixed] = y[fixed]
                new = new.astype("<f4")
                q = quality(new,selected,reference_sign)
                if np.min(q) >= 1e-5:
                    error = np.linalg.norm(new-y,axis=1)
                    solution = {"fraction_of_native_deformation":float(t),
                                "minimum_normalized_volume":float(q.min()),
                                "rms_change_m":float(np.sqrt(np.mean(error*error))),
                                "maximum_change_m":float(error.max()),
                                "fixed_maximum_change_m":float(error[fixed].max())}
                    break
            keep.append({"fixed_target_rows":anchored,"best_screened_fit":solution})
        report.append({"region":ri,"native_reference":variant,"fits":keep})

(root/"coherent-pectoral-fit-screen.json").write_text(json.dumps(report,indent=2)+"\n")
print([(r["region"],r["native_reference"],[f["best_screened_fit"] and
                                          f["best_screened_fit"]["maximum_change_m"] for f in r["fits"]]) for r in report])
