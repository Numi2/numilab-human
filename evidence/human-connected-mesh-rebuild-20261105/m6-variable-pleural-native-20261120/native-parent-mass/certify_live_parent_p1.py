"""Certify retained P1 ancestry against a separately captured runtime parent."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
previous = root / "complete-fine-p1-volume-floor"
prior_state = previous / "current-paired-volume-state.json"
state = json.loads(prior_state.read_text())
captured = root / "floor-live-parent-capture-002/compiled-parent-cell-mass.csv"
out = root / "complete-fine-p1-live-parent"
out.mkdir()
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

historical = state["source330_identity"]
paths = {key: Path(pin["path"]) for key, pin in historical.items()}
for key, path in paths.items():
    assert digest(path) == historical[key]["sha256"]
live = {key: Path(str(captured) + suffix) for key, suffix in (
    ("source330_mass", ""), ("source330_labels", ".labels.csv"),
    ("source330_nodes", ".nodes.csv"), ("source330_tets", ".tets.csv"),
    ("source330_contributions", ".source.csv"),
)}
archived = root.parent.parent / "numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
for path in live.values():
    suffix = str(path)[len(str(captured)):]
    assert digest(path) == digest(str(archived) + suffix)
prv_nodes = np.loadtxt(paths["source330_nodes"], delimiter=",", skiprows=1)
new_nodes = np.loadtxt(live["source330_nodes"], delimiter=",", skiprows=1)
prv_tets = np.loadtxt(paths["source330_tets"], delimiter=",", skiprows=1, usecols=range(7), dtype=np.int64)
new_tets = np.loadtxt(live["source330_tets"], delimiter=",", skiprows=1, usecols=range(7), dtype=np.int64)
assert np.array_equal(prv_nodes[:, [0, 1, 9]], new_nodes[:, [0, 1, 9]])
assert np.array_equal(prv_tets, new_tets)
assert digest(live["source330_labels"]) == historical["source330_labels"]["sha256"]
frame_delta = np.linalg.norm(prv_nodes[:, 2:5] - new_nodes[:, 2:5], axis=1)
tet_delta = np.max(frame_delta[new_tets[:, 3:7]], axis=1)
max_range = .02
state.update(source330_identity={key: {"path": str(path), "sha256": digest(path)} for key, path in live.items()},
             historical_preference_source330=historical, projection_frame_unchanged=False,
             p1_preference_policy="Retain typed source ancestry, certify each retained map within the existing distance bound. Recorded distances are historical.",
             p1_current_distance_limit_m=max_range,
             p1_current_parent_maximum_displacement_m=float(frame_delta.max()))
for role in ("skin", "fat"):
    receipt = json.loads((previous / f"{role}-p1-receipt.json").read_text())
    with np.load(receipt["p1_path"]) as data:
        off = data["node_group_offsets"]
        rows = data["node_group_parent_tet_rows"]
        parent_off = data["node_group_parent_tet_offsets"]
        node_of_group = np.repeat(np.arange(len(off) - 1), np.diff(off))
        node_of_row = np.repeat(node_of_group, np.diff(parent_off))
        dist = data["node_parent_registration_distance_m"][node_of_row] + tet_delta[rows]
        limit = float(dist.max())
        if not np.all(np.isfinite(dist) & (dist <= max_range)):
            raise ValueError(f"{role} original ancestry exceeds the retained distance bound")
        state[f"p1_{role}_current_distance_upper_bound_m"] = limit
        print(role, "retained typed parent witnesses", len(dist), "current upper bound m", limit, flush=True)
pinned = out / prior_state.name
pinned.write_text(json.dumps(state, indent=2) + "\n")
for role in ("skin", "fat"):
    receipt = json.loads((previous / f"{role}-p1-receipt.json").read_text())
    department = out / role.title()
    department.mkdir()
    old_base = receipt["p1_sha256"]
    for path_key, hash_key in (("p1_path", "p1_sha256"), ("p1_one_hop_path", "p1_one_hop_sha256")):
        source = Path(receipt[path_key])
        assert digest(source) == receipt[hash_key]
        with np.load(source, allow_pickle=False) as packet:
            data = dict(packet)
        assert str(data["current_state_sha256"]) == digest(prior_state)
        assert str(data["source330_mass_sha256"]) == historical["source330_mass"]["sha256"]
        data.update(current_state_path=np.asarray(str(pinned)), current_state_sha256=np.asarray(digest(pinned)),
                    source330_mass_sha256=np.asarray(digest(captured)))
        if path_key == "p1_one_hop_path":
            assert str(data["admissibility_source_support_sha256"]) == old_base
            data["admissibility_source_support_sha256"] = np.asarray(receipt["p1_sha256"])
        dest = department / source.name
        np.savez_compressed(dest, **data)
        receipt[path_key], receipt[hash_key] = str(dest), digest(dest)
    receipt.update(state_path=str(pinned), state_sha256=digest(pinned), live_parent_revalidated=True)
    (out / f"{role}-p1-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
