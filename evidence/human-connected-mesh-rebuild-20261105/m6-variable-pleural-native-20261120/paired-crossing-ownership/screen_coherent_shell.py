"""Screen a shared physical skin/fat displacement against exact crossing witnesses."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, sys.argv[2])
from native_source import banks, clearance, exact, run, skin, skin_faces


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clipped_witness(surface, target, normal):
    # Intersect the projected tissue triangle with the skin face. The maximum
    # of the affine signed height occurs at a vertex of this clipped polygon.
    # Coordinates are two skin-face barycentric weights and signed height.
    edges = surface[1:] - surface[0]
    relative = target - surface[0]
    coordinates = np.linalg.solve(edges @ edges.T, (relative @ edges.T).T).T
    polygon = np.column_stack((coordinates, relative @ normal))
    for coordinate, offset in (((1,0),0),((0,1),0),((-1,-1),1)):
        if not len(polygon):
            return None
        side = polygon[:, :2] @ coordinate + offset
        clipped = []
        for j in range(len(polygon)):
            i = j-1
            if (side[i] >= 0) != (side[j] >= 0):
                clipped.append(polygon[i] + (polygon[j]-polygon[i]) *
                               (side[i]/(side[i]-side[j])))
            if side[j] >= 0:
                clipped.append(polygon[j])
        polygon = np.asarray(clipped)
    if not len(polygon):
        return None
    point = polygon[np.argmax(polygon[:, 2])]
    return surface[0] + point[:2] @ edges, point[2]


def witness_samples(paired, terminal, origin, normals):
    known = {(b["semantic"], b["stable_id"]): b for b in terminal["surfaces"]}
    selected = {}
    for bank in paired["banks"]:
        if "initial" not in bank or not bank["initial"]:
            continue
        key = tuple(bank["key"])
        xyz, faces = banks[key]
        xyz = xyz.astype(float)
        inherited = (set(map(tuple, known[key]["triangle_pairs"])) -
                     set(map(tuple, bank["terminal_only_pairs"])))
        initial = inherited | set(map(tuple, bank["initial_only_pairs"]))
        assert len(initial) == bank["initial"]
        for s, t in initial:
            witness = clipped_witness(origin[s], xyz[faces[t]], normals[s])
            if witness is None:
                continue
            point, height = witness
            if height <= 0 or (s in selected and selected[s][1] >= height):
                continue
            selected[s] = (point, height)
    samples = sorted(selected.items())
    return (np.array([w[0] for _,w in samples]), np.array([w[1] for _,w in samples]),
            normals[[i for i,_ in samples]])


def displacement(points, sites, depth, direction, radius):
    tree = cKDTree(sites)
    field = np.zeros(points.shape)
    for start in range(0,len(points),8192):
        end = min(len(points),start+8192)
        nearby = cKDTree(points[start:end]).sparse_distance_matrix(
            tree, radius, output_type="ndarray")
        if not len(nearby):
            continue
        i, j, distance = (nearby[n] for n in ("i", "j", "v"))
        q = distance/radius
        w = (1-q)**4*(1+4*q)
        volume_weight = np.bincount(i, weights=w, minlength=end-start)
        target = depth[j,None]*direction[j]
        # Normalize overlapping samples. The unit denominator also tapers an
        # isolated sample smoothly to zero at the compact support boundary.
        for axis in range(3):
            field[start:end,axis] = (np.bincount(i, weights=w*target[:,axis],
                                               minlength=end-start) / np.maximum(volume_weight,1.))
    return field


def margin(points, cells):
    # Float64 arithmetic on the stored binary32 coordinate values.
    det = np.empty(len(cells))
    quality = np.empty(len(cells))
    for start in range(0,len(cells),32768):
        end = min(len(cells),start+32768)
        tri = points[cells[start:end]].astype(float)
        d = tri[:, 1:] - tri[:, :1]
        limit = np.maximum.reduce([np.sum((tri[:,i]-tri[:,j])**2,axis=1)
                                   for i in range(4) for j in range(i)])
        det[start:end] = np.einsum("ij,ij->i",d[:,0],np.cross(d[:,1],d[:,2]))
        quality[start:end] = det[start:end]/limit**1.5
    return det, quality


def metrics(points, cells):
    det, quality = margin(points, cells)
    return dict(nonpositive=int((det<=0).sum()), volume_floor=int((det<=6e-18).sum()),
                quality_floor=int((quality<1e-5).sum()), min_quality=float(quality.min()),
                min_volume_m3=float(det.min()/6))


def exact_trial(candidate, baseline):
    # Only changed faces can contribute pairs that differ. Preserve original
    # skin triangle IDs when comparing predicates with the source report.
    moved = np.any(candidate != skin,axis=1)
    changed = np.flatnonzero(moved[skin_faces].any(axis=1))
    changed_mask = np.zeros(len(skin_faces),bool)
    changed_mask[changed] = True
    faces = skin_faces[changed]
    ids = np.unique(faces)
    records = clearance._exact_surface_records(candidate[ids],np.searchsorted(ids,faces))
    index = exact._prepare_surface_aabb(records)
    broad = clearance._float32_skin_aabb_index(records)
    by_bank = []
    for key, geometry in banks.items():
        if key[0] == 51010 and 381 <= key[1] <= 397:
            continue
        xyz, f = geometry
        chosen,count = clearance._candidate_target_rows_for_skin_aabb(f,xyz,broad)
        if not len(chosen) and key not in baseline:
            continue
        used = np.unique(f[chosen])
        recs = clearance._exact_surface_records(xyz[used],np.searchsorted(used,f[chosen]))
        recs = [(a,b,c,int(chosen[d]),e) for a,b,c,d,e in recs]
        audit = exact._audit_pair_prepared_first(index,recs)
        assert audit["aabb_candidate_pairs"] == count
        after = {(int(changed[i]),j) for i,j in audit["triangle_pairs"]}
        old = {pair for pair in baseline.get(key,set()) if changed_mask[pair[0]]}
        removed, added = old-after, after-old
        if removed or added:
            by_bank.append(dict(key=key,removed=len(removed),added=len(added),
                                added_pairs=sorted(added),removed_pairs=sorted(removed)))
    # Check the complete candidate shell with its original face IDs.
    all_records = clearance._exact_surface_records(candidate,skin_faces)
    cross_self = exact._audit_pair_prepared_first(
        exact._prepare_surface_aabb(all_records), all_records, same_surface=True)
    return dict(changed_outer_faces=len(changed),changed_outer_vertices=int(moved.sum()),
                removed_pairs=sum(x["removed"] for x in by_bank),
                new_pairs=sum(x["added"] for x in by_bank), by_bank=by_bank,
                changed_outer_self_pairs=cross_self["count"])


paired_file, topology_file, bed_file, output = map(Path,sys.argv[3:7])
paired = json.loads(paired_file.read_text())
terminal = json.loads((run/"exact-skin-surface.json").read_text())
assert paired["terminal_report_sha256"] == digest(run/"exact-skin-surface.json")
launch = json.loads((run/"launch.json").read_text())["environment"]
prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"


def pinned(name):
    path = Path(launch[prefix+name])
    sha = prefix + name.removesuffix("_CSV") + "_SHA256"
    assert digest(path) == launch[sha]
    return path


node_table = np.loadtxt(pinned("TETGEN_NODE"),skiprows=1,usecols=(1,2,3),dtype=np.float32)
assert np.array_equal(node_table[:len(skin)],skin)
raw_cells = np.loadtxt(pinned("TETGEN_ELEMENTS"),skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1
receipt = json.loads(topology_file.with_name("proposal-receipt.json").read_text())
assert digest(topology_file) == receipt["proposal_sha256"]
assert receipt["source_volume_sha256"] == launch[prefix+"OPERATIONAL_CANDIDATE_VOLUME_SHA256"]
with np.load(topology_file,allow_pickle=False) as p:
    cells = np.concatenate((raw_cells[p["retained_input_cell_rows"]],p["replacement_tetrahedra"]),
                           dtype=np.int32,casting="unsafe")
del raw_cells
initial_quality = metrics(node_table,cells)
assert all(initial_quality[k]==0 for k in ("nonpositive","volume_floor","quality_floor"))

anchor_rows = set()
with pinned("DEEP_ATTACHMENT_REQUESTS_CSV").open() as f:
    for row in csv.DictReader(f):
        assert row["source_domain"] == "D37_coarse_parent"
        anchor_rows.add(int(row["stable_source_vertex"]))
anchors = []
with pinned("COMBINED_NODE_MAP_CSV").open() as f:
    for row in csv.DictReader(f):
        if row["role"] != "D" or row["source_domain"] != "D37_coarse_parent":
            continue
        if int(row["source_vertex_row"]) in anchor_rows:
            anchors.append(int(row["joint_node_id"])-1)
assert len(anchors) == len(anchor_rows) and len(anchors) == len(set(anchors))
bed = json.loads(bed_file.read_text())
bed_normal = np.array(bed["normal"])
bed_distance = node_table.astype(float)@bed_normal-bed["offset_m"]
anchor_distance = cKDTree(node_table[anchors]).query(node_table)[0]


def ramp(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


fixed = ramp((bed_distance-.002)/.012)*ramp(anchor_distance/.015)
assert np.all(fixed[anchors]==0)
shell = skin[skin_faces].astype(float)
normals = np.cross(shell[:,1]-shell[:,0],shell[:,2]-shell[:,0])
area = np.linalg.norm(normals,axis=1)
assert np.all(area>0)
normals *= np.sign(clearance._signed_surface_integral(skin, skin_faces))/area[:,None]
locations, depths, directions_check = witness_samples(paired,terminal,shell,normals)
baseline = {}
for bank in terminal["surfaces"]:
    key = bank["semantic"],bank["stable_id"]
    if not bank["ocular_source_id"] and bank["count"] and key in banks:
        baseline[key] = set(map(tuple,bank["triangle_pairs"]))
for bank in paired["banks"]:
    if "initial" not in bank:
        continue
    key = tuple(bank["key"])
    old = (baseline.get(key,set()) - set(map(tuple,bank["terminal_only_pairs"])))
    baseline[key] = old | set(map(tuple,bank["initial_only_pairs"]))
    assert len(baseline[key]) == bank["initial"]
# Co-located sites would weight dense tissue tessellation more than sparse
# neighborhoods. Keep the largest measured separation in each 2 mm cube.
locations_mm = np.floor(locations/.002).astype(np.int32)
buckets = {}
for i,key in enumerate(map(tuple,locations_mm)):
    if key not in buckets or depths[i]>depths[buckets[key]]:
        buckets[key] = i
chosen = np.array(sorted(buckets.values()))
buffered_depth = np.minimum(.003, depths[chosen]+.000025)
output.mkdir(exist_ok=False)
observations = dict(paired_pairs_sha256=digest(paired_file), topology_sha256=digest(topology_file),
    total_reference_pairs=sum(map(len,baseline.values())), selected_witness_faces=len(depths),
    selected_control_sites=len(chosen), raw_projected_depth_mm_percentiles=dict(zip(
        ("minimum","median","p90","p99","maximum"),
        map(float,np.percentile(depths*1000,[0,50,90,99,100])))),
    maximum_requested_shift_m=.003, anchor_rows=sorted(anchors),
    reference_volume_quality=initial_quality, trials=[])
print(json.dumps(observations),flush=True)

for radius in (.012,.024):
    delta = displacement(node_table,locations[chosen],buffered_depth,directions_check[chosen],radius)
    delta *= fixed[:,None]
    float64_candidate = node_table.astype(float)+delta
    print(json.dumps(dict(radius_m=radius, unrounded_full_shift=metrics(float64_candidate,cells))),flush=True)
    candidate = None
    for fraction in (1,.5,.25,.125,.0625,.03125,.015625):
        point = (node_table.astype(float)+fraction*delta).astype(np.float32)
        assert np.array_equal(point[anchors],node_table[anchors])
        assert np.array_equal(point[bed_distance<=.002],node_table[bed_distance<=.002])
        m = metrics(point,cells)
        row = dict(radius_m=radius,fraction=fraction,
                   max_shift_m=float(np.linalg.norm(point-node_table,axis=1).max()),**m)
        observations["trials"].append(row)
        print(json.dumps(row),flush=True)
        if all(m[k]==0 for k in ("nonpositive","volume_floor","quality_floor")):
            candidate = point
            break
    # Evaluate the full proposed gain if no usable binary32 step was found.
    # It is retained explicitly as a rejected geometry diagnostic.
    row = dict(radius_m=radius,fraction=fraction if candidate is not None else 1.,
               reference_cell_gate=candidate is not None)
    if candidate is None:
        candidate = float64_candidate.astype(np.float32)
    result = exact_trial(candidate[:len(skin)], baseline)
    print(json.dumps(dict(row,removed=result["removed_pairs"],new=result["new_pairs"],
                          self_pairs=result["changed_outer_self_pairs"])),flush=True)
    (output/f"exact-{round(radius*1000)}mm.json").write_text(
        json.dumps(dict(row,**result),indent=2)+"\n")
    if row["reference_cell_gate"] and result["removed_pairs"] and not result["new_pairs"] and not result["changed_outer_self_pairs"]:
        np.savez_compressed(output/f"screened-{round(radius*1000)}mm.npz",
                            points=candidate,nearest_anchor_distance_m=anchor_distance,
                            protected_bed_nodes=bed_distance<=.002)
(output/"screen-summary.json").write_text(json.dumps(observations,indent=2)+"\n")
