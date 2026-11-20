"""Check both fascia candidates against the actual posed source tissues and skin."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from audit_pectoral_fascia_crossings import compare, records, tets, regions

root = Path(__file__).resolve().parent
asset = root/"fascia-neighbors-audit-run/surrounding.bin"
capture = np.loadtxt(root/"fascia-neighbors-audit-run/fascia-rest.csv",delimiter=",",skiprows=1)
original = np.loadtxt(root/"registered-fascia-audit-run/fascia-rest.csv",delimiter=",",skiprows=1)
assert np.array_equal(capture, original)
mesh = np.fromfile(asset,dtype=[("identity","<u4",(3,)),("triangle","<f4",(3,3))])
assert 0 < len(mesh) and np.all((mesh["identity"][:,0] >= 1) & (mesh["identity"][:,0] <= 3))
assert np.isfinite(mesh["triangle"]).all()
triangles = mesh["triangle"]
low, high = triangles.min(axis=1),triangles.max(axis=1)
face_bank = tets[:,((1,2,3),(0,3,2),(0,1,3),(0,2,1))].reshape(-1,3)
face_region = np.repeat(regions,4)
keys = np.column_stack((face_region,np.sort(face_bank,axis=1)))
_, i, counts = np.unique(keys,axis=0,return_index=True,return_counts=True)
boundary = i[(counts==1) & (face_region[i] < 6)]
ids = face_bank[boundary]
with np.load(root/"pectoral-material-sheet-f32.npz") as candidate:
    positions = (capture[:,1:4], candidate["positions_world_m"])

def audit(points):
    exterior = points[ids]
    ceil, floor = exterior.max(axis=1), exterior.min(axis=1)
    report = Counter()
    witnesses = []
    for f in range(len(exterior)):
        overlap = np.flatnonzero(np.all((ceil[f] >= low) & (floor[f] <= high), axis=1))
        x, inv = np.unique(np.vstack((exterior[f],triangles[overlap].reshape(-1,3))),axis=0,return_inverse=True)
        x = x.astype(np.float64)
        ii = inv.ravel().astype(np.uint32)
        a = ii[:3]
        for j,b in zip(overlap,ii[3:].reshape(-1,3)):
            if compare(x,a,b):
                kind, stable, row = map(int,mesh["identity"][j])
                region = int(face_region[boundary[f]])
                report[(kind,stable,region)] += 1
                if len(witnesses)<1000:
                    witnesses.append([kind,stable,row,region,ids[f].tolist()])
    return {"total":sum(report.values()),"by_pair":[[*row,n] for row,n in sorted(report.items())],
            "witnesses":witnesses}

out = {"surrounding_file_sha256":hashlib.sha256(asset.read_bytes()).hexdigest(),"surrounding_faces":len(mesh),
       "baseline":audit(positions[0]),"candidate":audit(positions[1]),
       "scope":"Triangle surface-pair crossings with nearby posed NHTISSUE, NHBONES and NHSKIN. Gliding interface pairs remain visible in the raw count for review."}
(root/"fascia-surrounding-audit.json").write_text(json.dumps(out,indent=2)+"\n")
print([out[k]["total"] for k in ("baseline","candidate")])
