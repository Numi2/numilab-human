"""Identify scalar-label choices that can retain prior child ancestry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
previous = root.parent/"resting-clearance-and-atrium-016"
sys.path.insert(0,str(previous))
from screen_paired_mass_routes import COMPILER, load, sha
from export_repaired_tetgen import array_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--emit",action="store_true")
args = parser.parse_args()
p1 = root/"complete-fine-p1-current-parent"
state_path = p1/"current-paired-volume-state.json"
state = json.loads(state_path.read_text())
receipt = json.loads((COMPILER/"mass-admission/mass-receipt.json").read_text())
_, _, cells, material, ancestry, changed, _, old_topology, old_material = load()
if array_hash(cells,"<i4") != state["tetrahedra_sha256"]:
    raise ValueError("Current TetGen ancestry changed")
reports = []
if args.emit:
    source = Path(json.loads((COMPILER/"p1-export.command.json").read_text())["command"][1])
    owner = source.parent.parent/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/p1_reference_anchor_217.py"
    if sha(owner)!="67d6a0f3dd4f345433305e154e906f73b4f24633d9f3805103f496ce164846f8":
        raise ValueError("Native P1 primary tie rule owner changed")
    historical_owner = owner.read_text()
    if "tied score groups use minimum authenticated group ID" not in historical_owner:
        raise ValueError("Missing native tie rule")
    combined = np.full((len(cells),5),-1,dtype=np.int64)
    combined[:,0] = np.arange(len(cells))
for layer, role in (("skin",15),("fat",16)):
    info = json.loads((p1/f"{layer}-p1-receipt.json").read_text())
    p1_path = Path(info["p1_one_hop_path"])
    if sha(p1_path) != info["p1_one_hop_sha256"]:
        raise ValueError("Current support must match its receipt")
    label_pin = receipt["input_sha256"][f"child_{layer}_labels"]
    labels_path = Path(label_pin["path"])
    if sha(labels_path) != label_pin["sha256"]:
        raise ValueError("Historical labels must match the native allocator receipt")
    archived = np.loadtxt(labels_path,delimiter=",",skiprows=1,dtype=np.int32)
    old_rows = np.flatnonzero(old_material==role)
    if len(archived) == len(old_material):
        assert np.array_equal(archived[:,0],np.arange(len(old_material)))
        assert np.array_equal(archived[:,1],old_material)
        archived = archived[old_rows]
    else:
        assert len(archived) == len(old_rows)
        assert np.array_equal(archived[:,0],np.arange(len(old_rows)))
    assert np.all(archived[:,1]==role)
    rows = np.flatnonzero(material==role)
    old_crosswalk = np.full(len(old_material),-1,dtype=np.int32)
    old_crosswalk[old_rows] = np.arange(len(old_rows))
    with np.load(p1_path,allow_pickle=False) as z:
        groups = z["group_tuples"]
        group_rows = np.repeat(np.arange(len(rows)),np.diff(z["cell_group_offsets"]))
        slot = z["cell_group_ids"]
        n = z["cell_group_p1_score_numerators"]
        d = z["cell_group_p1_score_denominators"]
        max_score = np.maximum.reduceat(n,z["cell_group_offsets"][:-1])
        winning = n == max_score[group_rows]
        assert (d == z["cell_p1_score_denominator"][group_rows]).all()
        assert (z["cell_p1_score_denominator"] > 0).all()
        tied = np.bincount(group_rows[winning],minlength=len(rows)) > 1
        same_role = ancestry[rows] >= 0
        old_local = old_crosswalk[np.maximum(ancestry[rows],0)]
        valid = same_role[group_rows] & (old_local[group_rows] >= 0)
        exact = np.all(groups[slot]==archived[np.maximum(old_local[group_rows],0),1:4],axis=1)
        preserved = np.bincount(group_rows[winning & valid & exact],minlength=len(rows))==1
        changes = []
        topology_ties = set(np.flatnonzero(tied & ~same_role))
        for local in sorted(topology_ties):
            child = int(rows[local])
            prior = old_crosswalk[changed[ancestry[child]]]
            assert np.all(prior>=0)
            winner_groups = slot[(group_rows==local)&winning]
            common = set(tuple(g) for g in groups[winner_groups]) & set(tuple(g) for g in archived[prior,1:4])
            changes.append({"child_row":child, "winning_tuples":len(winner_groups),
                            "matching_topology_parent_label_tuples":len(common)})
        report = {

            "role":layer,"cells":len(rows),"tie_cells":int(tied.sum()),
            "unresolved_label_cells_reported_by_p1_owner":len(z["unresolved_child_tet_rows"]),
            "unique_max_tuple_without_accepted_scalar_label":int(np.count_nonzero(
                ~tied[z["unresolved_child_tet_rows"]])),
            "p1_certified_label_rows":len(z["resolved_child_tet_rows"]),
            "tie_cells_with_existing_winning_parent_label":int(np.count_nonzero(tied&preserved)),
            "retained_tie_cells_without_a_winning_parent_label":int(np.count_nonzero(tied&same_role&~preserved)),
            "remeshed_tie_cells":len(topology_ties),
            "remeshed_tie_cells_with_unique_retired_parent_winner":sum(
                c["matching_topology_parent_label_tuples"]==1 for c in changes),
            "remeshed_tie_cells_detail":changes,
            "old_labels_sha256":label_pin["sha256"],"current_p1_sha256":info["p1_one_hop_sha256"],
        }
        if args.emit:
            starts = z["cell_group_offsets"][:-1]
            chosen_slots = np.minimum.reduceat(np.where(winning,np.arange(len(slot)),len(slot)),starts)
            winning_group = slot[chosen_slots]
            resolved = z["resolved_child_tet_rows"]
            if not np.array_equal(winning_group[resolved],z["resolved_group_ids"]):
                raise ValueError("Current scalar selection disagrees with P1 unique maxima")
            face_offsets = z["cell_group_stable_face_offsets"]
            source_faces = z["cell_group_stable_face_rows"]
            face_counts = (face_offsets[chosen_slots+1]-face_offsets[chosen_slots])
            if np.any(face_counts<=0):
                raise ValueError("Primary P1 tuples lack source face ancestry")
            # Follow the P1 owner's existing derived-face convention for a
            # complete multi-face group. This ID is not a raw source face.
            faces = np.where(face_counts==1,source_faces[face_offsets[chosen_slots]],2**31+winning_group.astype(np.int64))
            if not np.array_equal(faces[resolved],z["stable_source_face_id"]):
                raise ValueError("Current scalar face convention disagrees with the P1 owner")
            combined[rows,1:4] = groups[winning_group]
            combined[rows,4] = faces
            report["derived_multiface_label_rows"] = int(np.count_nonzero(face_counts>1))
            report["retained_legacy_tuple_matches_current_scalar"] = int(np.count_nonzero(
                same_role & np.all(combined[rows,1:4]==archived[np.maximum(old_local,0),1:4],axis=1)))
    reports.append(report)
    print(layer,{k:v for k,v in report.items() if isinstance(v,int)},flush=True)
if args.emit:
    if np.any(combined<0) or not np.array_equal(combined[:,1],material):
        raise ValueError("Combined label domain has unresolved rows or altered materials")
    output = root/"paired-current-child.labels.csv"
    np.savetxt(output,combined,delimiter=",",fmt="%d",
              header="child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id",comments="")
    result={"path":str(output),"sha256":sha(output),"rows":len(combined),
            "tie_rule_owner_sha256":sha(owner),
            "scope":"Scalar compatibility labels preserve the complete P1 preference, ancestry and tie arrays. The native scene has not admitted this volume."}
    (root/"paired-current-child-labels-receipt.json").write_text(json.dumps(result,indent=2)+"\n")
(root/"paired-primary-label-audit.json").write_text(json.dumps({
    "current_state_sha256":sha(state_path),"reports":reports,
    "scope":"Scalar label provenance preflight. No new geometric label choices or physical installation.",
},indent=2)+"\n")
