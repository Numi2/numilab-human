"""Replay only the six-carrier-parent compensation already certified for this digit."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0, str(PRIOR))
from screen_paired_mass_routes import COMPILER, PREVIOUS, load, sha
from export_repaired_tetgen import array_hash

FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def pinned(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"Archived source pin changed: {path}")
    return path


class ArchivedSkinCompensation:
    def __init__(self, projection, transport):
        run = json.loads((COMPILER / "mass-admission/run-manifest.json").read_text())
        source = json.loads(pinned(run["source_control_manifest"]).read_text())
        recipe = json.loads((COMPILER / "mass-admission/mass-receipt.json").read_text())
        policy_file = pinned(source["policy"])
        policy = json.loads(policy_file.read_text())
        packet = pinned(recipe["input_sha256"]["carrier_skin_packet"])
        with np.load(packet, allow_pickle=False) as p:
            parent = p["regional_compensation_parent_rows"]
            old_receivers = p["regional_compensation_child_rows"]
            self.budget, distance_cap = p["regional_compensation_limits"]
        assert parent.tolist() == policy["parent_rows"]
        assert self.budget == policy["maximum_budget_kg"]
        assert distance_cap == policy["maximum_distance_m"]
        self.parents = frozenset(parent.tolist())
        on_policy = np.isin(transport["source_parent"][transport["carrier_row"]], parent)
        source_capacity = float(transport["mass_flow_kg"][on_policy].sum())
        assert source_capacity <= self.budget
        self.distance_cap = min(distance_cap, projection.limit)
        self.policy_sha256 = sha(policy_file)
        self.packet_sha256 = sha(packet)
        archived = HERE / "skin-cut-outside-carriers.csv"
        trace_summary = json.loads((HERE / "skin-cut-carrier-trace.json").read_text())
        assert trace_summary["archived_carrier_trace_decoded_sha256"] == recipe["output_sha256"]["carrier_trace"]["sha256"]
        assert trace_summary["archived_parent_label_sha256"] == recipe["source330"]["labels"]["sha256"]
        assert trace_summary["outside_witness_sha256"] == sha(archived)

        old, xyz, cells, materials, origins, _, _, old_tets, old_materials = load()
        with np.load(PREVIOUS / "fine-mass-local-position-delta.npz") as patch:
            xyz[patch["changed_rows"]] = patch["positions_f32"]
        assert np.array_equal(xyz, projection.points) and np.array_equal(cells, projection.cells)
        parent_mesh = projection.parent
        skin_rows = np.flatnonzero(parent_mesh["materials"] == 15)
        parent_regions = parent_mesh["regions"]
        faces = np.sort(parent_mesh["tets"][skin_rows][:, FACES], axis=2).reshape(-1, 3)
        _, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        face_order = np.argsort(inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(counts)]
        adjacent = defaultdict(set)
        for face in np.flatnonzero(counts == 2):
            first, second = face_order[offsets[face]:offsets[face+1]] // 4
            a, b = int(parent_regions[skin_rows[first]]), int(parent_regions[skin_rows[second]])
            adjacent[a].add(b)
            adjacent[b].add(a)

        self.rows = defaultdict(set)
        self.distance = {}
        self.route_mass = defaultdict(float)
        self.archived_mass = 0.
        # Old and paired carrier packs use identical parent ordering. The
        # current registration may move a carrier, so measure each route again.
        with archived.open(newline="") as stream:
            for raw in csv.DictReader(stream):
                row = int(raw["new_child_row"])
                original = int(raw["archived_child_row"])
                carrier = int(raw["carrier_row"])
                donor = int(raw["parent_row"])
                assert donor in self.parents and original in old_receivers
                assert origins[row] == original
                assert materials[row] == old_materials[original] == 15
                assert transport["source_parent"][carrier] == donor
                assert projection.tuples[carrier,0] == 15
                assert parent_mesh["regions"][donor] == projection.tuples[carrier,2]
                assert np.array_equal(cells[row], old_tets[original])
                assert np.array_equal(xyz[cells[row]], old[cells[row]])
                region = int(parent_regions[donor])
                assert all(dest == region or dest in adjacent[region] or
                           any(dest in adjacent[j] for j in adjacent[region])
                           for dest in map(int, trace_summary["cut_regions"]))
                distance = float(np.linalg.norm(projection.centers[row] -
                                                projection.supplier_positions[carrier]))
                assert distance <= self.distance_cap
                self.rows[row].add(carrier)
                self.distance[(carrier, row)] = distance
                self.route_mass[(carrier, row)] += float(raw["mass_kg"])
                self.archived_mass += float(raw["mass_kg"])
        assert abs(self.archived_mass - trace_summary["outside_mass_kg"]) < 1e-14
        assert self.archived_mass <= self.budget
