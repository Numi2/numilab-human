"""Locate connected, collision-free donor-face patches for the fascia sheet."""
import argparse
from collections import defaultdict
import ctypes
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--region", type=int, nargs="*", choices=range(6), default=list(range(6)))
args = parser.parse_args()
witness = json.loads((root.parent/"resting-clearance-and-atrium-016/fascia-source-cell-witness.json").read_text())
raw = Path(witness["asset"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == witness["sha256"]
_, count, nodes, _ = struct.unpack_from("<4I",raw,8)
thickness = struct.unpack_from("<f",raw,24)[0]
assert 0.00059 < thickness < 0.00061
mesh = np.fromfile(root/"fascia-neighbors-audit-run/surrounding.bin",
                   dtype=[("id","<u4",(3,)),("xyz","<f4",(3,3))])
reference = np.loadtxt(root/"registered-fascia-audit-run/fascia-rest.csv", delimiter=",",skiprows=1)
assert len(reference) == nodes
flags = np.ndarray(nodes,dtype="<u4",buffer=raw,offset=104+32*count+16,strides=(32,))
xyz = mesh["xyz"]
mid = xyz.mean(axis=1)
lower, upper = xyz.min(axis=1), xyz.max(axis=1)
tree = cKDTree(mid)
sphere = np.linalg.norm(xyz-mid[:,None],axis=-1).max(axis=1)
maximum_radius = sphere.max()
pair = ctypes.CDLL(str(root.parent/"connected-fold-repair-62/exact-pair.dylib")).beyond
index_ptr = np.ctypeslib.ndpointer(dtype=np.uint32,ndim=1,flags="C_CONTIGUOUS")
pair.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags="C_CONTIGUOUS"),index_ptr,index_ptr]
pair.restype = ctypes.c_uint
prism_faces = np.array([[0,1,2],[3,4,5],[0,1,3],[1,4,3],[1,2,4],[2,5,4],
                        [2,0,5],[0,3,5]],dtype=np.int32)

def candidate_crossing(prism, donor):
    a,b = prism.min(axis=0),prism.max(axis=0)
    center = .5*(a+b)
    radius = np.linalg.norm(b-center)
    near = np.asarray(tree.query_ball_point(center, radius+maximum_radius),dtype=int)
    near = near[np.all((b >= lower[near]) & (a <= upper[near]),axis=1)]
    near = near[near != donor]
    coords, inverse = np.unique(np.vstack((prism,xyz[near].reshape(-1,3))),axis=0,return_inverse=True)
    local = inverse.ravel().astype(np.uint32)
    # Exempt only the original donor face. Exact edge and endpoint contact
    # with other faces is left to the established shared-feature predicate.
    own = local[:6]
    adjacent = local[6:].reshape(-1,3)
    for f in own[prism_faces]:
        lo, hi = coords[f].min(axis=0), coords[f].max(axis=0)
        for j in np.flatnonzero(np.all((hi >= lower[near]) & (lo <= upper[near]),axis=1)):
            if pair(coords, f, adjacent[j]):
                i = near[j]
                return mesh["id"][i].tolist()
    return None

def components(index, area, accepted):
    banks = defaultdict(list)
    for row in accepted:
        face = index[row]
        for a,b in ((0,1),(1,2),(2,0)):
            banks[tuple(sorted((int(face[a]),int(face[b]))))].append(row)
    connected = defaultdict(list)
    for rows in banks.values():
        if len(rows) == 2:
            a,b = rows
            connected[a].append(b)
            connected[b].append(a)
    seen = set()
    result = []
    for first in accepted:
        if first in seen:
            continue
        seen.add(first)
        stack, chosen = [first], []
        while stack:
            row = stack.pop()
            chosen.append(row)
            for neighbor in connected[row]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result.append(chosen)
    return sorted(result,key=lambda rows:-area[rows].sum())

reports = []
for region in args.region:
    _, _, first, length, _, _, stable = struct.unpack_from("<8s6I",raw,104+32*region)
    h = length//2
    ref = reference[first:first+h,1:4]
    band = flags[first:first+h]
    donor = np.flatnonzero((mesh["id"][:,0] == 1) & (mesh["id"][:,1] == stable))
    p, inv = np.unique(xyz[donor].reshape(-1,3),axis=0,return_inverse=True)
    tr = inv.ravel().reshape(-1,3)
    face_cross = np.cross(xyz[donor,1]-xyz[donor,0],xyz[donor,2]-xyz[donor,0])
    magnitude = np.linalg.norm(face_cross,axis=1)
    area = .5*magnitude
    vertex_normal = np.zeros_like(p,dtype=np.float64)
    np.add.at(vertex_normal,tr.ravel(),np.repeat(face_cross,3,axis=0))
    length_n = np.linalg.norm(vertex_normal,axis=1)
    vertex_normal /= np.maximum(length_n[:,None],1e-12)
    # The scene is supine. Use the anterior donor surface, restricted to
    # the existing registered patch bounds with the existing 30 mm cap.
    candidates = np.flatnonzero((magnitude > 2e-11) &
            (face_cross[:,2] > .25*magnitude) &
            np.all((mid[donor]>=ref.min(axis=0)-.030)&(mid[donor]<=ref.max(axis=0)+.030),axis=1))
    accepted, rejected = [], defaultdict(int)
    for local in candidates:
        face = p[tr[local]].astype(np.float64)
        surface = (face + thickness*vertex_normal[tr[local]]).astype(np.float32).astype(np.float64)
        k = np.vstack((face,surface))
        if np.any(vertex_normal[tr[local]]@face_cross[local] < .25*magnitude[local]):
            rejected["distorted_offset"] += 1
            continue
        crossing = candidate_crossing(k, donor[local])
        if crossing is None:
            accepted.append(int(local))
        else:
            rejected[f"neighbor_{crossing[0]}_{crossing[1]}"] += 1
    selected = components(tr,area,accepted)[:12]
    witnesses = []
    for rows in selected:
        x = p[np.unique(tr[rows])]
        kt = cKDTree(x)
        gap, _ = kt.query(ref)
        witnesses.append({"faces":len(rows),"area_m2":float(area[rows].sum()),
            "closest_fixed_target_m":float(gap[band==1].min()),"worst_fixed_target_m":float(gap[band==1].max()),
            "closest_load_target_m":float(gap[band==2].min()),"worst_load_target_m":float(gap[band==2].max()),
            "source_face_rows":mesh["id"][donor[rows],2].tolist()})
    report = {"region":region,"source_stable_id":stable,"anterior_faces_tested":len(candidates),
              "collision_free_local_prisms":len(accepted),"rejections":dict(rejected),"components":witnesses}
    reports.append(report)
    print(region,len(candidates),len(accepted),[(w["faces"],w["worst_fixed_target_m"],w["worst_load_target_m"]) for w in witnesses[:3]],flush=True)
(root/"pectoral-exterior-availability.json").write_text(json.dumps({
    "source_fascia_sha256":witness["sha256"],"thickness_m":thickness,
    "surrounding_sha256":hashlib.sha256((root/"fascia-neighbors-audit-run/surrounding.bin").read_bytes()).hexdigest(),
    "regions":reports, "native_admitted":False,
    "scope":"Connected component pilot over exact posed donor faces. Independent shell-level self-intersection, volume and complete attachment fit are still required."
},indent=2)+"\n")
