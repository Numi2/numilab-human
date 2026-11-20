"""Measure registered sheet orientation and thickness before runtime rewinding."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--paired-offset", action="store_true")
args = parser.parse_args()
prefix = "paired-offset-fascia" if args.paired_offset else "registered-fascia"
scope = json.loads((here.parent / "resting-clearance-and-atrium-016/fascia-source-cell-witness.json").read_text())
raw = Path(scope["asset"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == scope["sha256"]
assert raw[:8] == b"NHFASC4\0"
abi, regions, n, count = struct.unpack_from("<4I", raw, 8)
assert abi == 4
reg = np.frombuffer(raw, offset=104, count=8*regions, dtype="<u4").reshape(-1, 8)
records = 104+32*regions
source = np.ndarray((n, 3), dtype="<f4", buffer=raw, offset=records, strides=(32,4)).astype(np.float64)
vertices = np.ndarray(n, dtype="<u4", buffer=raw, offset=records+20, strides=(32,))
tets = np.frombuffer(raw, dtype="<u4", count=count*5, offset=records+n*32).reshape(-1,5)
path = here / (prefix + "-audit-run") / "fascia-rest.csv"
mapped = np.loadtxt(path, delimiter=",", skiprows=1)
assert np.array_equal(mapped[:,0], np.arange(n))
xyz = mapped[:,1:4]
assert np.array_equal(xyz, xyz.astype(np.float32).astype(np.float64))

def shape(points):
    p = points[tets[:,:4]]
    delta = (p[:,1:] - p[:,:1])
    determinant = np.linalg.det(delta)
    scale = np.linalg.norm(p[:,:,None,:] - p[:,None,:,:], axis=-1).max(axis=(1,2))**3
    return determinant, abs(determinant) / scale, np.linalg.cond(delta)

src_d, src_q, src_k = shape(source)
dst_d, dst_q, dst_k = shape(xyz)
flipped = np.signbit(src_d) != np.signbit(dst_d)
report = []
for i in range(regions):
    current = tets[:,4] == i
    groups = np.where(current)[0]
    record = reg[i]
    # Nodes are two contiguous layers with an immutable source vertex
    # correspondence. Check the pairing even when a registration has moved it.
    first, length = map(int, record[3:5])
    upper = np.arange(first, first+length//2)
    lower = upper + length//2
    assert len(upper) > 0 and int(lower[-1]) < n
    local_pairs = vertices[upper] == vertices[lower]
    distances = np.linalg.norm(xyz[upper]-xyz[lower], axis=-1)
    faces = Counter()
    for tet, parity in zip(tets[groups,:4], np.signbit(dst_d[groups])):
        tet = list(map(int,tet))
        if parity:
            tet[0],tet[1] = tet[1],tet[0]
        for a,b,c in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
            x,y,z = tet[a],tet[b],tet[c]
            faces[tuple(sorted((x,y,z)))] += -1 if ((x>y)+(x>z)+(y>z)) % 2 else 1
    report.append({
        "region":i,
        "cells":len(groups),
        "registered_orientation_reversals":int(np.count_nonzero(flipped[current])),
        "same_winding_internal_faces_after_reorientation":sum(abs(k)>1 for k in faces.values()),
        "minimum_raw_normalized_volume":float(src_q[current].min()),
        "minimum_mapped_normalized_volume":float(dst_q[current].min()),
        "median_raw_shape_condition":float(np.median(src_k[current])),
        "median_mapped_shape_condition":float(np.median(dst_k[current])),
        "maximum_mapped_shape_condition":float(dst_k[current].max()),
        "aligned_source_vertex_pairs":int(local_pairs.sum()),
        "layer_pair_count":len(upper),
        "layer_distance_m":np.quantile(distances,[0.,.5,1.]).tolist(),
    })
witness = []
for item in scope["measurements"]:
    row = item["source_tet"]
    assert np.array_equal(tets[row,:4], item["nodes"])
    witness.append({
        "compiled_row":item["compiled_tet"],
        "source_row":row,
        "region":item["source_region"],
        "source_normalized_volume":float(src_q[row]),
        "registered_normalized_volume":float(dst_q[row]),
        "source_to_registered_signed_volume_ratio":float(dst_d[row]/src_d[row]),
        "orientation_reversal":bool(flipped[row]),
        "source_condition":float(src_k[row]),
        "registered_condition":float(dst_k[row]),
    })
output = {
    "native_reference_positions_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    "source_sha256":scope["sha256"],
    "total_registered_orientation_reversals":int(flipped.sum()),
    "regions":report,
    "active_failure_witnesses":witness,
    "scope":"Reference setup at the authored pose, including source repair mapping, before FEM state creation and native motion.",
}
(here / (prefix + "-reference-audit.json")).write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({"first_six_regions":report[:6], "active_failure_witnesses":witness},indent=2))
