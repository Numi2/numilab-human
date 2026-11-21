"""Transfer admitted skin/fat densities via local, checked tetrahedral overlap."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError

launch_path, proposal_path, owner_path = map(Path, sys.argv[1:])
owner = owner_path.read_text()
first = owner.index("EDGES = ")
last = owner.index("\n\nroot = Path.cwd()")
# Reuse the repository's tetrahedral overlap implementation without loading
# the authoring script's separate paired route experiment.
exec(compile(owner[first:last], str(owner_path), "exec"))

reference_weights = weights

def weights(points, tet, inverse):
    beta = reference_weights(points, tet, inverse)
    shared = np.all(points[:,None] == tet[None,:],axis=2)
    rows, corners = np.nonzero(shared)
    # These are the same Float32 source vertices, so these weights are exact.
    # A tiny negative value from the inverse would wrongly clip the vertex.
    beta[rows] = np.eye(4)[corners]
    return beta

from fractions import Fraction

def exact_overlap(left, right, volume_scale):
    # Clip edges against exact signed-volume barycentric half spaces. Convert
    # vertices to float only for volume integration after exact clipping.
    def det(u,v,w):
        return sum(u[i]*(v[(i+1)%3]*w[(i+2)%3] - v[(i+2)%3]*w[(i+1)%3]) for i in range(3))

    def diff(a,b):
        return tuple(x-y for x,y in zip(a,b))

    def endpoint_weights(point, tet):
        origin = tet[0]
        a,b,c = (diff(v, origin) for v in tet[1:])
        q = diff(point,origin)
        signed = det(a,b,c)
        coords = (det(q,b,c)/signed,det(a,q,c)/signed,det(a,b,q)/signed)
        return (1-sum(coords),*coords)

    first,second = (tuple(tuple(Fraction(float(x)) for x in p) for p in tet)
                    for tet in (left,right))
    corners = set()
    for p,q in ((first,second),(second,first)):
        values = [endpoint_weights(point,q) for point in p]
        for i,j in EDGES:
            lo,hi = Fraction(0),Fraction(1)
            for start,end in zip(values[i],values[j]):
                if start < 0 and end < 0:
                    lo,hi = Fraction(1),Fraction(0)
                    break
                if start < 0:
                    lo = max(lo,-start/(end-start))
                elif end < 0:
                    hi = min(hi,start/(start-end))
            if lo <= hi:
                for t in (lo,hi):
                    corners.add(tuple(x + (y-x)*t for x,y in zip(p[i],p[j])))
    if len(corners) < 4:
        return 0.
    origin,*rest = sorted(corners)
    vertices = np.array([(0.,0.,0.),*(tuple(float(y) for y in diff(x,origin)) for x in rest)])
    if np.linalg.matrix_rank(vertices[1:]) < 3:
        return 0.
    return float(ConvexHull(vertices).volume / volume_scale)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

launch = json.loads(launch_path.read_text())["environment"]
proposal = json.loads(proposal_path.read_text())
volume_key = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
positions, elements = (Path(launch[volume_key+k]) for k in ("NODE","ELEMENTS"))
assert digest(positions) == proposal["input_sha256"]["NODE"] == launch[volume_key+"NODE_SHA256"]
assert digest(elements) == proposal["input_sha256"]["ELEMENTS"] == launch[volume_key+"ELEMENTS_SHA256"]
points = np.loadtxt(positions, skiprows=1, usecols=(1,2,3))
data = np.loadtxt(elements, skiprows=1, dtype=np.int32)
assert np.array_equal(data[:,0],np.arange(1,len(data)+1))
tets, roles = data[:,1:5] - 1, data[:,5]
role_bounds = {15:(900.,1100.),16:(850.,950.)}
mass = np.empty(len(tets))
density = np.empty(len(tets))
native_volume = np.empty(len(tets))
mass_pins = {}
row_for_role = {}
for material, name in ((15,"SKIN"),(16,"FAT")):
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"+name
    path = Path(launch[prefix+"_CHILD_MASS_CSV"])
    observed = digest(path)
    assert observed == launch[prefix+"_CHILD_MASS_CSV_SHA256"]
    mass_pins[name] = observed
    role_rows = np.flatnonzero(roles == material)
    rows = np.loadtxt(path,delimiter=",",skiprows=1)
    assert len(rows)==len(role_rows) and np.array_equal(rows[:,0],np.arange(len(rows)))
    native_volume[role_rows], mass[role_rows], density[role_rows] = rows[:,1:].T
    assert np.all(mass[role_rows] > 0.)
    row_for_role[material] = role_rows

def vol(rows):
    return np.abs(np.linalg.det(rows[:,1:] - rows[:,:1]))/6

validated = []
status = Counter()
volume_err = []
cooked_err = []
relative_mass_error = []
relative_volume_adjustment = []
eligible_mass = 0.
net_mass_diff = {15:0.,16:0.}
gradient_proxy_removed = 0.
candidate_cells = 0
all_densities = {15:[],16:[]}
nodal_delta = np.zeros(len(points))
nodal_baseline = np.bincount(tets.ravel(),weights=np.repeat(mass*.25,4),minlength=len(points))
first_moment_delta = np.zeros(3)
projection = []
# Every accepted matrix partitions both tetrahedralizations. Qualify overlap
# before balancing roundoff against the existing native volume convention.
for identity, entry in enumerate(proposal["replacements"]):
    before = np.asarray(entry["retired"],dtype=np.int32)
    after = np.asarray(entry["new"],dtype=np.int32)
    material = int(entry["material_id"])
    p,q = points[tets[before]],points[after]
    center = p[0][0]
    scale = np.ptp(p.reshape(-1,3),axis=0).max()
    p_scaled,q_scaled = (p-center)/scale,(q-center)/scale
    before_volume, after_volume = vol(p_scaled),vol(q_scaled)
    p_inv = [transform(x) for x in p_scaled]
    q_inv = [transform(x) for x in q_scaled]
    overlap_volume = np.asarray([[overlap(x,y,a,b) for y,b in zip(q_scaled,q_inv)]
                                 for x,a in zip(p_scaled,p_inv)])
    local_volume_error = float(max(abs(overlap_volume.sum(axis=1)/before_volume-1).max(),
                                   abs(overlap_volume.sum(axis=0)/after_volume-1).max()))
    if local_volume_error > 2e-6:
        status["exact_fallback"] += 1
        overlap_volume = np.asarray([[exact_overlap(x,y,scale**3) for y in q] for x in p])
        local_volume_error = float(max(abs(overlap_volume.sum(axis=1)/before_volume-1).max(),
                                       abs(overlap_volume.sum(axis=0)/after_volume-1).max()))
    volume_err.append(local_volume_error)
    if local_volume_error > 2e-6:
        status["overlap_closure"] += 1
        continue
    # Remove only rounding error, preserving overlap support and old per-cell mass.
    overlap_share = overlap_volume / overlap_volume.sum(axis=1,keepdims=True)
    direct_mass = overlap_share.T @ mass[before]
    target_volume = after_volume * scale**3
    rho_low,rho_high = role_bounds[material]
    old_density_error = float(np.max(abs(before_volume*scale**3/native_volume[before]-1)))
    cooked_err.append(old_density_error)
    if np.any(direct_mass / target_volume < rho_low-2e-7) or np.any(direct_mass / target_volume > rho_high+2e-7):
        status["physical_density_bound"] += 1
        continue
    old_moment = mass[before] @ (p.mean(axis=1)-center)
    new_moment = direct_mass @ (q.mean(axis=1)-center)
    fraction_error = abs(direct_mass.sum()/mass[before].sum()-1)
    assert fraction_error <= 2e-15
    first_moment_delta += new_moment-old_moment
    np.add.at(nodal_delta,tets[before].ravel(),np.repeat(-.25*mass[before],4))
    np.add.at(nodal_delta,after.ravel(),np.repeat(.25*direct_mass,4))
    relative_mass_error.append(float(fraction_error))
    relative_volume_adjustment.append(float(np.max(abs(overlap_share.T @ before_volume / after_volume-1))))
    net_mass_diff[material] += direct_mass.sum()-mass[before].sum()
    all_densities[material].extend((direct_mass/target_volume).tolist())
    eligible_mass += mass[before].sum()
    gradient_proxy_removed += entry["old_gradient_energy_proxy"]-entry["new_gradient_energy_proxy"]
    candidate_cells += len(after)-len(before)
    validated.append(identity)
    for old_index, new_index in zip(*np.nonzero(overlap_share)):
        projection.append((identity,int(before[old_index]),int(new_index),
                           float(overlap_share[old_index,new_index])))
    status["eligible"] += 1

maximum_nodal_relative = float(np.max(np.abs(nodal_delta)/np.maximum(nodal_baseline,1e-30)))
total_mass = float(mass.sum())
result = dict(
    scope="Local transport of existing native cell masses, pending original source-carrier proof and native volume replay.",
    proposal_sha256=digest(proposal_path),
    overlap_owner_sha256=digest(owner_path), old_mass_sha256=mass_pins,
    checked_operations=len(proposal["replacements"]), outcome=dict(status),
    eligible_operation_ids=validated, retained_cells=len(tets)+candidate_cells,
    total_mass_kg=total_mass, cavity_mass_transferred_kg=float(eligible_mass),
    total_mass_delta_kg={key:float(value) for key,value in net_mass_diff.items()},
    density_range_kg_m3={k:[min(v),max(v)] for k,v in all_densities.items()},
    maximum_relative_cavity_mass_error=max(relative_mass_error),
    maximum_relative_overlap_volume_error_all=max(volume_err),
    maximum_relative_new_overlap_volume_error_accepted=max(relative_volume_adjustment),
    maximum_relative_old_native_volume_difference=max(cooked_err),
    maximum_nodal_mass_difference_kg=float(abs(nodal_delta).max()),
    maximum_relative_nodal_mass_difference=maximum_nodal_relative,
    nodal_mass_redistribution_kg=float(abs(nodal_delta).sum()/2.),
    whole_body_center_of_mass_shift_m=(first_moment_delta/total_mass).tolist(),
    whole_body_gradient_proxy_fraction_removed=proposal["whole_body_proxy_fraction_removed"] *
        gradient_proxy_removed / sum(e["old_gradient_energy_proxy"]-e["new_gradient_energy_proxy"]
                                    for e in proposal["replacements"]),
)
with Path("cavity-mass-shares.csv").open("x",newline="") as stream:
    writer = csv.writer(stream,lineterminator="\n")
    writer.writerow(("operation_id","old_global_cell","replacement_cell","old_cell_mass_fraction"))
    writer.writerows(projection)
result["shares_sha256"] = digest(Path("cavity-mass-shares.csv"))
print(json.dumps(result,indent=2))
