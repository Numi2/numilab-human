"""Build a pinned proposal for the native fixed-cavity owner."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

launch, topology, routes, out = map(Path, sys.argv[1:5])
environment = json.loads(launch.read_text())["environment"]
prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def pinned(field):
    path = Path(environment[prefix + field])
    assert digest(path) == environment[prefix + field + "_SHA256"], path
    return path

label = np.loadtxt(pinned("CHILD_LABELS"), delimiter=",", skiprows=1,
                   usecols=(0, 1, 2, 3), dtype=np.uint32)
source = np.loadtxt(pinned("TETGEN_NODE"), skiprows=1, usecols=(1, 2, 3))
connectivity = np.loadtxt(pinned("TETGEN_ELEMENTS"), skiprows=1, dtype=np.int32)
assert np.array_equal(label[:, 0], np.arange(len(label)))
assert np.array_equal(connectivity[:, 0], np.arange(1, len(label) + 1))
assert np.array_equal(connectivity[:, 5], label[:, 1])
cells = connectivity[:, 1:5] - 1
eligibility = json.loads(routes.read_text())
assert eligibility["proposal_sha256"] == digest(topology)
direct = set(eligibility["direct_overlap_eligible_ids"])
raw = json.loads(topology.read_text())["replacements"]
assert max(direct) < len(raw)
assert 0 not in direct or len(direct) == eligibility["direct_overlap_eligible_operations"]
omit = set()
if len(sys.argv) > 5:
    rejected, prior = (json.loads(Path(p).read_text()) for p in sys.argv[5:7])
    if not rejected["complete"]:
        raise ValueError("Require complete native witness receipt")
    omit.update(prior["original_ids"][index] for index in rejected["cavity_indices"])

def min_quality(t):
    p = source[t]
    edge = p[:, :, None] - p[:, None, :]
    longest_squared = (edge**2).sum(axis=3).max(axis=(1, 2))
    delta = p[:, 1:] - p[:, :1]
    return np.min(np.einsum("ni,ni->n",delta[:, 0],np.cross(delta[:, 1],delta[:, 2]))
                  / longest_squared**1.5)

selected = []
retired = np.zeros(len(cells), dtype=bool)
replacements, patch_offsets, parent_offsets, parent_rows = [], [0], [0], []
rejections = Counter()
for i, patch in enumerate(raw):
    old, new = patch["retired"], patch["new"]
    if i not in direct:
        rejections["overlap_route_preflight"] += 1
        continue
    if i in omit:
        rejections["native_transport_witness"] += 1
        continue
    if np.any(label[old, 1:] != label[old[0], 1:]):
        rejections["native_ownership_domain"] += 1
        continue
    if min_quality(np.asarray(new)) < 1.02 * min_quality(cells[old]):
        rejections["native_minimum_quality_improvement"] += 1
        continue
    assert not retired[old].any()
    old_nodes = set(cells[old].flat)
    assert old_nodes == {v for tet in new for v in tet}
    retired[old] = True
    for tet in new:
        replacements.append(tet)
        parent_rows.extend(old)
        parent_offsets.append(len(parent_rows))
    patch_offsets.append(len(replacements))
    selected.append(i)
assert selected
out.mkdir(exist_ok=False)
path = out / "fixed-boundary-volume-proposal.npz"
volume_sha = environment[prefix + "OPERATIONAL_CANDIDATE_VOLUME_SHA256"]
assert digest(Path(environment[prefix + "OPERATIONAL_CANDIDATE_VOLUME_ASSET"])) == volume_sha
np.savez_compressed(path, retained_input_cell_rows=np.flatnonzero(~retired).astype(np.uint32),
                    replacement_tetrahedra=np.asarray(replacements, dtype=np.uint32),
                    replacement_tet_offsets=np.asarray(patch_offsets, dtype=np.uint32),
                    replacement_parent_offsets=np.asarray(parent_offsets, dtype=np.uint32),
                    replacement_parent_cell_rows=np.asarray(parent_rows, dtype=np.uint32),
                    source_volume_sha256=np.array(volume_sha))
receipt = dict(proposal_sha256=digest(path), source_volume_sha256=volume_sha,
               topology_sha256=digest(topology), transport_screen_sha256=digest(routes),
               selected=len(selected), rejections=dict(rejections), original_ids=selected,
               retired=int(retired.sum()), replacements=len(replacements),
               estimated_proxy_removed=sum(raw[i]["old_gradient_energy_proxy"] -
                                           raw[i]["new_gradient_energy_proxy"] for i in selected))
(out / "proposal-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({key:value for key,value in receipt.items() if key != "original_ids"}))
