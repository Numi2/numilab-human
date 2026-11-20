"""Count exactly which proven current flow edges still need native tuple witnesses."""
import argparse
import csv
import gzip
import io
import json
import shutil
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/"resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import sha

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--emit-witnesses",action="store_true")
args = parser.parse_args()
closure = json.loads((root/"archived-compensation-independent-check.json").read_text())
candidate = root/"paired-density-closed-flow-candidate.npz"
if sha(candidate) != closure["mass_sha256"]:
    raise ValueError("Current verified flow changed")
p1 = root/"complete-fine-p1-current-parent"
state = json.loads((p1/"current-paired-volume-state.json").read_text())
carrier_receipt = json.loads((root/"paired-carrier-packet-audit/receipt.json").read_text())
labels_path = Path(state["source330_identity"]["source330_labels"]["path"])
if sha(labels_path) != state["source330_identity"]["source330_labels"]["sha256"]:
    raise ValueError("Current paired label input changed")
labels = np.loadtxt(labels_path,delimiter=",",skiprows=1,dtype=np.int64)[:,1:4]
groups, actual_group = np.unique(labels,axis=0,return_inverse=True)
actual_group = actual_group.ravel()
group_ids = {tuple(row): i for i,row in enumerate(groups)}
domains = len(groups)

def pack(child, group):
    return child*domains + group

primary_keys, preference_keys, carriers_by_role = [], [], []
for cr in carrier_receipt["reports"]:
    role = cr["role"]
    prereq = json.loads((p1/f"{role}-p1-receipt.json").read_text())
    geometry = Path(prereq["role_geometry_path"])
    support = Path(prereq["p1_one_hop_path"])
    carriers = Path(cr["current_packet_path"])
    for path, check in ((geometry,prereq["role_geometry_sha256"]),(support,prereq["p1_one_hop_sha256"]),
                        (carriers,cr["current_packet_sha256"])):
        if sha(path) != check:
            raise ValueError(f"Current support artifact changed: {path}")
    with np.load(geometry,allow_pickle=False) as z:
        children = z["combined_child_rows"]
        if args.emit_witnesses:
            corners = z["points_world_m_f32"][z["tetrahedra"]].astype(np.float64)
            centroids = corners.sum(axis=1)*.25

    with np.load(support,allow_pickle=False) as z:
        offsets = z["cell_group_offsets"]
        remap = np.array([group_ids[tuple(row)] for row in z["group_tuples"]],dtype=np.int64)
        preference_keys.append(pack(np.repeat(children,np.diff(offsets)),remap[z["cell_group_ids"]]))
    with np.load(carriers,allow_pickle=False) as z:
        parent_rows = np.repeat(z["global_source330_tet_rows"], np.diff(z["parent_offsets"]))
        primary_keys.append(pack(children[z["carrier_target_child_rows"]],actual_group[parent_rows]))
        if args.emit_witnesses:
            source_geometry = Path(state["source330_identity"]["source330_nodes"]["path"])
            source_tets = Path(state["source330_identity"]["source330_tets"]["path"])
            for name,path in (("source330_nodes",source_geometry),("source330_tets",source_tets)):
                if sha(path) != state["source330_identity"][name]["sha256"]:
                    raise ValueError("Current carrier rest frame changed")
            rest = np.loadtxt(source_geometry,delimiter=",",skiprows=1,usecols=(5,6,7))
            tets = np.loadtxt(source_tets,delimiter=",",skiprows=1,usecols=(3,4,5,6),dtype=int)
            beta = z["carrier_barycentric"]
            carriers_by_role.append({
                "children":children,"centroids":centroids,
                "parent_rows":parent_rows,"bary":beta,"weights":z["carrier_weights"],
                "source_position":np.einsum("ni,nij->nj",beta,rest[tets[parent_rows]]),
                "local_parent":np.repeat(z["source_parent_tet_rows"],np.diff(z["parent_offsets"])),
                "primary_children":children[z["carrier_target_child_rows"]],
            })


supported = np.unique(np.concatenate(preference_keys+primary_keys))
with np.load(candidate,allow_pickle=False) as z:
    cell_groups = z["group_for_cell"]
    edge = z["mass_flow_kg"] > 0.
    edge_group = z["group"][edge]
    edge_parent = z["source_parent"][z["carrier_row"][edge]]
    edge_carrier = z["carrier_row"][edge]
    outside = z["archived_compensation"][edge]
n = np.bincount(cell_groups)
offsets = np.r_[0,np.cumsum(n)]
cell_order = np.argsort(cell_groups,kind="stable")
edge_length = n[edge_group]
edge_repeat = np.repeat(np.arange(len(edge_group)),edge_length)
cell_rows = (np.arange(edge_length.sum()) +
    np.repeat(offsets[edge_group] - (np.cumsum(edge_length)-edge_length),edge_length))
child = cell_order[cell_rows]
expanded = pack(child,actual_group[edge_parent[edge_repeat]])
indices = np.searchsorted(supported,expanded)
matches = (indices < len(supported)) & (supported[np.minimum(indices,len(supported)-1)]==expanded)
is_carrier = edge_carrier[edge_repeat] < sum(r["carrier_rows"] for r in carrier_receipt["reports"])
is_compensation = outside[edge_repeat]
missing = ~matches & ~is_compensation
report = {
    "model_installed":False,
    "current_state_sha256":sha(p1/"current-paired-volume-state.json"),
    "verified_flow_sha256":closure["mass_sha256"],
    "expanded_flow_rows":len(expanded),
    "existing_preference_and_primary_tuple_slots":len(supported),
    "matching_existing_support_edges":int(matches.sum()),
    "previous_compensation_edges":int(is_compensation.sum()),
    "remaining_native_witness_edges":int(missing.sum()),
    "remaining_unique_child_tuple_pairs":len(np.unique(expanded[missing])),
    "remaining_carrier_supported_edges":int(np.count_nonzero(missing & is_carrier)),
    "remaining_centroid_fallback_edges":int(np.count_nonzero(missing & ~is_carrier)),
    "remaining_centroid_fallback_unique_child_tuple_pairs":len(np.unique(expanded[missing & ~is_carrier])),
}
if len(expanded) != closure["expanded_route_checks"]:
    raise ValueError("Expanded row count differs from independently verified candidate")
if args.emit_witnesses:
    if report["remaining_centroid_fallback_edges"]:
        raise ValueError("Additional centroid rows require the existing typed centroid proof path")
    out = root/"paired-native-flow-witness"
    out.mkdir()
    report["generated_witnesses"] = []
    offset = 0
    for cr,data in zip(carrier_receipt["reports"],carriers_by_role):
        role = cr["role"]
        count = len(data["parent_rows"])
        start, end = offset, offset+count
        selected = np.flatnonzero(missing)
        c = edge_carrier[edge_repeat[selected]]
        local = (c>=start) & (c<end)
        selected, c = selected[local],c[local]-start
        local_child = np.searchsorted(data["children"],child[selected])
        if not np.array_equal(data["children"][local_child],child[selected]):
            raise ValueError("Expanded carrier goes outside its material")
        dist = np.linalg.norm(data["source_position"][c]-data["centroids"][local_child],axis=1)
        key = expanded[selected]
        order = np.lexsort((c,dist,key))
        unique = np.r_[True,np.diff(key[order]) != 0]
        pick = order[unique]
        if np.any(dist[pick]>closure["transport_distance_limit_m"]):
            raise ValueError("Existing distance cut exceeded by a new exact tuple witness")
        if np.any(child[selected[pick]]==data["primary_children"][c[pick]]):
            raise ValueError("Additional tuple witness duplicates its retained primary")
        previous = Path(cr["primary_witness_path"])
        if sha(previous)!=cr["primary_witness_sha256"]:
            raise ValueError("Current primary witness changed")
        output = out/f"{role}-carrier-witnesses.csv.gz"
        with output.open("wb") as f, gzip.GzipFile(fileobj=f,filename="",mode="wb",mtime=0,
                                                 compresslevel=5) as zipped:
            with io.TextIOWrapper(zipped,newline="") as stream:
                with gzip.open(previous,"rt",newline="") as original:
                    shutil.copyfileobj(original,stream)
                writer = csv.writer(stream,lineterminator="\n")
                for i in pick:
                    carr = c[i]
                    cl = local_child[i]
                    parent = data["parent_rows"][carr]
                    writer.writerow(("carrier_child_slot",role.title(),"current_source_carrier_edge_support",
                        carr,data["local_parent"][carr],parent,cl,child[selected[i]],*labels[parent],
                        "CurrentCarrierPointProjection",-1,-1,*data["bary"][carr],data["weights"][carr],
                        *data["source_position"][carr],*data["centroids"][cl],dist[i]))
        report["generated_witnesses"].append({
            "role":role,"path":str(output),"sha256":sha(output),
            "additional_unique_tuple_rows":len(pick),"primary_witness_rows":count,
            "max_supplemental_distance_m":float(dist[pick].max()),
        })
        offset = end
    assert sum(x["additional_unique_tuple_rows"] for x in report["generated_witnesses"]) == report["remaining_unique_child_tuple_pairs"]
(root/"paired-native-flow-support-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
