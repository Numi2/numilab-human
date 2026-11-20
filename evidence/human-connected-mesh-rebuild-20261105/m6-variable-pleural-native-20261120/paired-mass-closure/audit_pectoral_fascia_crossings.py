"""Run the existing exact surface predicate on baseline and fitted fascia boundaries."""
import ctypes
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

root = Path(__file__).resolve().parent
witness = json.loads((root.parent / "resting-clearance-and-atrium-016/fascia-source-cell-witness.json").read_text())
raw = Path(witness["asset"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == witness["sha256"]
_, nr, nodes, count = struct.unpack_from("<4I", raw, 8)
begin = 104 + nr*32
records = np.frombuffer(raw, dtype="<u4", count=count*5, offset=begin+nodes*32).reshape(-1,5)
tets = records[:,:4]
regions = records[:,4]
owner = root.parent / "connected-fold-repair-62/exact-pair.dylib"
compare = ctypes.CDLL(str(owner)).beyond
index_ptr = np.ctypeslib.ndpointer(dtype=np.uint32,ndim=1,flags="C_CONTIGUOUS")
compare.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags="C_CONTIGUOUS"), index_ptr,index_ptr]
compare.restype = ctypes.c_uint

def evaluate(points):
    x, tet = np.asarray(points, dtype=np.float64).copy(), np.asarray(tets,dtype=np.uint32).copy()
    p = x[tet]
    sign = np.linalg.det(p[:,1:] - p[:,:1]) < 0
    tet[sign,:2] = tet[sign,:2][:,::-1]
    face_bank = tet[:,((1,2,3),(0,3,2),(0,1,3),(0,2,1))].reshape(-1,3)
    face_region = np.repeat(regions,4)
    unique_key = np.column_stack((face_region,np.sort(face_bank,axis=1)))
    _, i, counts = np.unique(unique_key, axis=0,return_index=True,return_counts=True)
    assert counts.max()<=2
    exposed = i[counts==1]
    faces, face_region = face_bank[exposed], face_region[exposed]
    # Deduplicate coincident coordinate identities to handle matching source seams
    # consistently with the reader. Only identical opposite-oriented faces are exempt.
    unique, inverse = np.unique(x,axis=0,return_inverse=True)
    unique = np.ascontiguousarray(unique)
    mapped = np.ascontiguousarray(inverse.ravel()[faces],dtype=np.uint32)
    three = unique[mapped]
    lo, hi = three.min(axis=1),three.max(axis=1)
    crossings = []
    exact_interfaces = 0
    for i in range(len(mapped)-1):
        keep = (np.all(hi[i]>=lo[i+1:],axis=1) & np.all(hi[i+1:]>=lo[i],axis=1))
        if face_region[i]>=6:
            keep &= face_region[i+1:]<6
        for j in np.flatnonzero(keep) + i+1:
            a, b = mapped[i],mapped[j]
            if set(a)==set(b):
                if any(np.array_equal(np.roll(a[::-1], k), b) for k in range(3)):
                    exact_interfaces += 1
                    continue
            if compare(unique,a,b):
                crossings.append([int(face_region[i]),int(face_region[j]),
                                  faces[i].tolist(),faces[j].tolist()])
    return {"faces":len(faces),"exact_shared_opposite_interfaces":exact_interfaces,
            "within_region_crossings":sum(r[0]==r[1] for r in crossings),
            "between_region_crossings":sum(r[0]!=r[1] for r in crossings),
            "witnesses":crossings[:200]}

capture = np.loadtxt(root/"registered-fascia-audit-run/fascia-rest.csv", delimiter=",", skiprows=1)
assert np.array_equal(capture[:,0], np.arange(nodes))
with np.load(root/"pectoral-material-sheet-f32.npz") as asset:
    repaired = asset["positions_world_m"]
report = {"predicate_sha256":hashlib.sha256(owner.read_bytes()).hexdigest(),
          "baseline":evaluate(capture[:,1:4]),"candidate":evaluate(repaired),
          "scope":"All regional fascia boundary intersections incident to a pectoral region at the native reference pose. Surrounding muscle, bone, fat and skin audit separate."}
(root/"pectoral-fascia-crossings.json").write_text(json.dumps(report,indent=2)+"\n")
for name in ("baseline","candidate"):
    print(name,{k:v for k,v in report[name].items() if k!="witnesses"})
