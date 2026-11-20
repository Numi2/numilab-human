"""Carry archived source quadrature onto the exact repaired child row domain."""
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import COMPILER, PREVIOUS, load, sha
from export_repaired_tetgen import array_hash

state_path = root/"complete-fine-p1-current-parent/current-paired-volume-state.json"
state = json.loads(state_path.read_text())
receipt = json.loads((root/"archived-compensation-independent-check.json").read_text())
archival = json.loads((COMPILER/"mass-admission/mass-receipt.json").read_text())
base, points, cells, material, orig, changed, _, old_cells, old_materials = load()
with np.load(PREVIOUS/"fine-mass-local-position-delta.npz",allow_pickle=False) as patch:
    points[patch["changed_rows"]] = patch["positions_f32"]
if (array_hash(points,"<f4") != receipt["positions_sha256"] or
    array_hash(cells,"<i4") != receipt["connectivity_sha256"]):
    raise ValueError("Carrier audit must use the independently admitted positions and topology")
source_tets = Path(state["source330_identity"]["source330_tets"]["path"])
source_nodes = Path(state["source330_identity"]["source330_nodes"]["path"])
for k,p in (("source330_tets",source_tets),("source330_nodes",source_nodes)):
    if sha(p) != state["source330_identity"][k]["sha256"]:
        raise ValueError(f"Paired source geometry has changed: {p}")
rest = np.loadtxt(source_nodes,delimiter=",",skiprows=1,usecols=(5,6,7))
parents = np.loadtxt(source_tets,delimiter=",",skiprows=1,usecols=(3,4,5,6),dtype=int)
labels = Path(state["source330_identity"]["source330_labels"]["path"])
if sha(labels) != state["source330_identity"]["source330_labels"]["sha256"]:
    raise ValueError("Source parent labels changed")
source_labels = np.loadtxt(labels,delimiter=",",skiprows=1,dtype=int)
out = root/"paired-carrier-packet-audit"
out.mkdir(exist_ok=True)
reports = []
for layer, role in (("skin",15),("fat",16)):
    pin = archival["input_sha256"][f"carrier_{layer}_packet"]
    original = Path(pin["path"])
    if sha(original) != pin["sha256"]:
        raise ValueError("Archived quadrature packet hash changed")
    with np.load(original,allow_pickle=False) as packet:
        data = {k:packet[k] for k in packet.files}
    roles = np.flatnonzero(material==role)
    prior_roles = np.flatnonzero(old_materials==role)
    base_ids = prior_roles[data["carrier_target_child_rows"]]
    old_to_new = np.full(len(old_materials),-1,dtype=int)
    retained = roles[orig[roles] >= 0]
    if len(np.unique(orig[retained])) != len(retained):
        raise ValueError(f"{layer}: ambiguous mapping from archived child rows")
    old_to_new[orig[retained]] = retained
    targets = old_to_new[base_ids]
    lost = np.flatnonzero(targets<0)
    with_ancestry = targets>=0
    old_t = old_cells[base_ids[with_ancestry]]
    new_t = cells[targets[with_ancestry]]
    same = np.all(old_t==new_t,axis=1)
    exact = np.zeros(len(targets),dtype=bool)
    exact[with_ancestry] = same & np.all(base[old_t]==points[new_t],axis=(1,2))
    srows = np.repeat(data["global_source330_tet_rows"],np.diff(data["parent_offsets"]).astype(int))
    source = np.einsum("ni,nij->nj",data["carrier_barycentric"],rest[parents[srows]])
    valid = np.flatnonzero(with_ancestry)
    center = points[cells[targets[valid]]].mean(axis=1,dtype=np.float64)
    d = np.linalg.norm(source[valid]-center,axis=1)
    report = {
        "role":layer,"carrier_rows":len(targets),"exact_retained_targets":int(exact.sum()),
        "moved_targets":int(with_ancestry.sum()-exact.sum()),"missing_targets":len(lost),
        "max_distance_on_existing_targets_m":float(d.max()),
        "above_existing_transport_cutoff":int(np.count_nonzero(d > receipt["transport_distance_limit_m"])),
        "missing_examples":[[int(i),int(base_ids[i]),int(srows[i])] for i in lost[:24]],
        "archive_packet_sha256":pin["sha256"],
    }
    p1 = json.loads((state_path.parent/f"{layer}-p1-receipt.json").read_text())
    path = Path(p1["p1_one_hop_path"])
    if sha(path) != p1["p1_one_hop_sha256"]:
        raise ValueError("Current one-hop role support changed")
    with np.load(path,allow_pickle=False) as support:
        offsets = support["cell_admissible_region_offsets"]
        region_ids = support["cell_admissible_region_ids"]
    geometry_source = Path(p1["role_geometry_path"])
    if sha(geometry_source) != p1["role_geometry_sha256"]:
        raise ValueError("Role geometry changed")
    with np.load(geometry_source,allow_pickle=False) as geometry:
        if not np.array_equal(geometry["combined_child_rows"],roles):
            raise ValueError("Role-local source row order differs from current child order")
    regions_by_carrier = source_labels[srows,3]
    region_bound = np.diff(offsets).max()
    candidate = np.searchsorted(roles,targets[valid])
    region_rows = offsets[candidate,None] + np.arange(region_bound)[None,:]
    in_row = region_rows < offsets[candidate+1,None]
    inside_region = np.any(in_row & (region_ids[np.where(in_row,region_rows,0)] ==
                                     regions_by_carrier[valid,None]),axis=1)
    lost_region = valid[~inside_region]
    rebind = np.sort(np.r_[lost,lost_region])
    typed_children = np.repeat(np.arange(len(roles)),np.diff(offsets))
    center = points[cells[roles]].mean(axis=1,dtype=np.float64)
    distance = np.zeros(len(targets))
    distance[valid] = d
    search = {}
    locality_fallbacks = 0
    rerouted = []
    final_targets = targets.copy()
    for carrier in rebind:
        region = regions_by_carrier[carrier]
        if region not in search:
            subset = typed_children[region_ids==region]
            search[region] = (subset,cKDTree(center[subset]))
        subset, tree = search[region]
        descendants = []
        if targets[carrier] < 0:
            for group, prior in changed.items():
                if base_ids[carrier] in prior:
                    local = np.flatnonzero(orig[roles] == group)
                    descendants.extend(local[np.isin(local,subset)].tolist())
        if descendants:
            descendants = np.array(sorted(set(descendants)))
            ix = descendants[np.argmin(np.linalg.norm(center[descendants]-source[carrier],axis=1))]
        else:
            _, index = tree.query(source[carrier])
            ix = subset[index]
            locality_fallbacks += 1
        distance[carrier] = np.linalg.norm(center[ix]-source[carrier])
        final_targets[carrier] = roles[ix]
        rerouted.append([int(carrier),int(base_ids[carrier]),int(roles[ix]),int(region),
                         float(distance[carrier]),bool(len(descendants))])
    if np.any(distance > receipt["transport_distance_limit_m"]):
        raise ValueError("Rebound carrier exceeded the existing transport radius")
    data["carrier_target_child_rows"] = np.searchsorted(roles,final_targets).astype("<u4")
    data["carrier_target_distance_m"] = distance
    if layer == "skin":
        old = data["regional_compensation_child_rows"]
        mapped = old_to_new[old]
        keep = mapped >= 0
        before = old_cells[old[keep]]
        after = cells[mapped[keep]]
        identical = np.all(before==after,axis=1) & np.all(base[before]==points[after],axis=(1,2))
        new_scope = np.sort(mapped[keep][identical]).astype("<u4")
        if len(new_scope) == 0 or np.any(material[new_scope]!=15):
            raise ValueError("Compensation surface lacks an unchanged current skin domain")
        witnesses = np.loadtxt(root/"skin-cut-outside-carriers.csv",delimiter=",",skiprows=1,
                               usecols=(0,),dtype=int)
        if not np.all(np.isin(witnesses,new_scope)):
            raise ValueError("Rebound compensation does not preserve previously verified recipients")
        data["regional_compensation_child_rows"] = new_scope
        report["compensation_original_children"] = len(old)
        report["compensation_retained_identical_children"] = len(new_scope)
    packet_path = out/f"{layer}-current-source-carriers.npz"
    np.savez_compressed(packet_path,**data)
    witness_pin = archival["input_sha256"][f"carrier_witness_{layer}"]
    original_witness = Path(witness_pin["path"])
    if sha(original_witness) != witness_pin["sha256"]:
        raise ValueError("Archived quadrature construction witness changed")
    construction = {}
    with gzip.open(original_witness,"rt",newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if row["record_kind"] == "carrier":
                index = int(row["packet_carrier_row"])
                if index in construction or int(row["global_source330_tet_row"]) != srows[index]:
                    raise ValueError("Archived carrier identity disagrees with its packet")
                construction[index] = row["carrier_construction"]
    if len(construction) != len(srows):
        raise ValueError("Missing archived construction provenance")
    weights = data["carrier_weights"]
    bary = data["carrier_barycentric"]
    has_carriers = data["parent_offsets"][1:] > data["parent_offsets"][:-1]
    starts = data["parent_offsets"][:-1][has_carriers]
    weight_error = np.abs(np.add.reduceat(weights,starts)-1.).max()
    bary_error = np.abs(np.add.reduceat(weights[:,None]*bary,starts,axis=0)-0.25).max()
    if weight_error > 2e-12 or bary_error > 2e-10:
        raise ValueError("Source carrier quadrature no longer preserves mass and centroid")
    local_parents = np.repeat(data["source_parent_tet_rows"], np.diff(data["parent_offsets"]).astype(int))
    witness_path = out/f"{layer}-primary-carrier-witnesses.csv.gz"
    with witness_path.open("wb") as f, gzip.GzipFile(fileobj=f,filename="",mode="wb",mtime=0) as compressed:
        with io.TextIOWrapper(compressed,newline="") as stream:
            writer = csv.writer(stream,lineterminator="\n")
            writer.writerow(fields)
            for row in range(len(srows)):
                local = data["carrier_target_child_rows"][row]
                kind = construction[row]
                mode = ("source_tet_canonical_quadrature"
                        if kind=="source_tet_canonical_degree1_quadrature_1046e"
                        else "CurrentCarrierPointProjection")
                writer.writerow(("carrier",layer.title(),kind,row,local_parents[row],srows[row],
                                 local,final_targets[row],*source_labels[srows[row],1:4],mode,-1,-1,
                                 *bary[row],weights[row],*source[row],*center[local],distance[row]))
    report["weight_sum_error_max"] = float(weight_error)
    report["barycentric_centroid_error_max"] = float(bary_error)
    report["primary_witness_sha256"] = sha(witness_path)
    report["primary_witness_path"] = str(witness_path)
    report.update(reassigned_missing=len(lost), reassigned_onehop_failures=len(lost_region),
                  same_region_nearest_fallbacks=locality_fallbacks,reassigned_witnesses=rerouted,
                  maximum_current_carrier_distance_m=float(distance.max()),
                  current_packet_sha256=sha(packet_path),current_packet_path=str(packet_path))
    reports.append(report)
    print(layer,len(targets),"exact",exact.sum(),"moved",with_ancestry.sum()-exact.sum(),"missing",len(lost),flush=True)
    output = out/f"{layer}-target-ancestry.npz"
    np.savez_compressed(output,global_source_tet=srows,archived_target_combined_row=base_ids,
                        rebound_target_combined_row=targets,source_reference_m=source,
                        inherited_target_distance_m=d,inherited_target_carrier_row=valid)
    report["target_ancestry_sha256"] = sha(output)

(out/"receipt.json").write_text(json.dumps({"current_state_sha256":sha(state_path),
    "reports":reports,"source_carrier_packets_admitted":False},indent=2)+"\n")
