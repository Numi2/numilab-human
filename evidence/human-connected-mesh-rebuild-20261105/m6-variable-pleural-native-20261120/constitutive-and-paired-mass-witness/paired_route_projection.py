"""Build missing carrier edges with the existing exact typed projection owner."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from screen_paired_mass_routes import COMPILER, HERE, PREVIOUS, load, paired_reference_carriers, sha
from export_repaired_tetgen import array_hash

class PairedRouteProjection:
    def __init__(self, transport, audit, cell_domains):
        prior = json.loads((HERE.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())
        pins = prior["parent_inputs"]
        paths = {key: Path(pin["path"]) for key, pin in pins.items()}
        for name, path in paths.items():
            assert sha(path) == pins[name]["sha256"]
        recipe = COMPILER / "mass-admission/mass-receipt.json"
        config = json.loads(recipe.read_text())
        p1 = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
        assert sha(p1) == prior["owner_sha256"]
        p1_owner = self.load_module("pinned_role_p1", p1)
        assert sha(p1_owner.OWNER230) == p1_owner.OWNER230_SHA
        self.p1 = p1_owner
        self.registration = p1_owner._load_230(p1_owner.OWNER230)
        self.owner = self.registration.native
        parent = self.owner.read_parent(paths["source330_mass"])
        self.rest, self.tets, _, _ = paired_reference_carriers(
            COMPILER, paths["source330_nodes"], paths["source330_tets"])
        assert np.array_equal(self.tets, parent["tets"])
        recorded = np.loadtxt(paths["source330_nodes"], delimiter=",", skiprows=1)
        parent["points"] = recorded[:, 2:5] + (self.rest - recorded[:, 5:8])
        projection = (p1.parents[1] /
                      "current-native-1024-root/current-217-1036/register_current_combined_volume_nodes_217_1036.py")
        self.projection = self.load_module("paired_typed_projection_owner", projection)
        self.lookup = self.projection.make_lookup(parent, self.owner)
        self.parent = parent
        self.index = self.owner.build_index(parent)
        self.material_index = self.registration.build_material_indices(parent)
        self.groups = {role: self.p1._source_groups(parent, role) for role in self.p1.ROLE_MATERIAL}
        self.projection_owner_sha256 = sha(projection)
        self.limit = config["transport_distance_limit_m"]
        self.projection_limit = 0.019353
        parents = transport["source_parent"]
        self.tuples = np.column_stack((parent["materials"], parent["donors"], parent["regions"]))[parents]
        self.supplier_positions = self.read_carrier_positions(config["input_sha256"], parents)
        _, points, cells, self.material, _, _, _, _, _ = load()
        with np.load(PREVIOUS / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
            points[patch["changed_rows"]] = patch["positions_f32"]
        assert array_hash(points, "<f4") == audit["geometry_sha256"]
        assert array_hash(cells, "<i4") == audit["tetrahedron_sha256"]
        self.points, self.cells = points, cells
        self.centers = points[cells].mean(axis=1, dtype=np.float64)
        self.order = np.argsort(cell_domains, kind="stable")
        counts = np.bincount(cell_domains)
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.group_centers = np.add.reduceat(self.centers[self.order], self.offsets[:-1], axis=0) / counts[:, None]
        self.cache = {}
        self.direct_cache = {}

    @staticmethod
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def read_carrier_positions(self, pins, expected_parents):
        positions, rows = [], []
        for role in ("skin", "fat"):
            packet = pins[f"carrier_{role}_packet"]
            path = Path(packet["path"])
            assert sha(path) == packet["sha256"]
            with np.load(path, allow_pickle=False) as data:
                indices = np.repeat(data["global_source330_tet_rows"], np.diff(data["parent_offsets"]).astype(int))
                positions.append(np.einsum("na,naj->nj", data["carrier_barycentric"], self.rest[self.tets[indices]]))
                rows.append(indices)
        canonical = np.concatenate(rows)
        fallback = np.setdiff1d(np.arange(len(self.tets)), canonical)
        assert np.array_equal(np.r_[canonical, fallback], expected_parents)
        return np.vstack((*positions, self.rest[self.tets[fallback]].mean(axis=1)))

    def cells_of(self, dest):
        return self.order[self.offsets[dest]:self.offsets[dest+1]]

    def direct_support(self, dest):
        if dest not in self.direct_cache:
            rows = self.cells_of(dest)
            role = "Skin" if self.material[rows[0]] == 15 else "Fat"
            groups, lookup = self.groups[role]
            global_nodes, local_cells = np.unique(self.cells[rows], return_inverse=True)
            support = self.p1.build_node_support(self.points[global_nodes], role, self.parent,
                self.index, self.material_index, groups, lookup, self.registration)
            cell_maps = []
            for tet in local_cells.reshape(-1, 4):
                cell_map = {}
                for corner in tet:
                    for group, parents in support["maps"][corner].items():
                        key = tuple(int(item) for item in groups[group])
                        cell_map.setdefault(key, set()).update(parents)
                cell_maps.append(cell_map)
            self.direct_cache[dest] = cell_maps
        return self.direct_cache[dest]

    def verify(self, carrier, dest):
        rows = self.cells_of(dest)
        distances = np.linalg.norm(self.centers[rows] - self.supplier_positions[carrier], axis=1)
        if np.max(distances) > self.limit:
            return None
        identity = tuple(int(x) for x in self.tuples[carrier])
        key = (dest, identity)
        if key not in self.cache:
            evidence = []
            mesh = self.lookup[identity][0]
            extent = mesh["points"][np.unique(mesh["tets"])]
            lo = extent.min(axis=0) - self.projection_limit
            hi = extent.max(axis=0) + self.projection_limit
            direct = self.direct_support(dest)
            for row, cell_map in zip(rows, direct):
                if identity in cell_map:
                    evidence.append({"child": int(row), "exact_P1_parent_rows": sorted(cell_map[identity])})
                    continue
                point = self.centers[row]
                if np.any(point < lo) or np.any(point > hi):
                    evidence = None
                    break
                try:
                    projection = self.projection.nearest_for_tuple(
                        point, identity, self.lookup, self.owner, self.projection_limit, 1e-8)
                except (ValueError, RuntimeError):
                    evidence = None
                    break
                source = int(projection["tet"])
                bary = projection["beta"]
                distance_rest = float(np.linalg.norm(point - bary @ self.rest[self.tets[source]]))
                if (projection["distance"] > self.projection_limit or distance_rest > self.projection_limit or
                        not np.isfinite(bary).all() or np.min(bary) < 0. or abs(bary.sum()-1.) > 2e-12):
                    evidence = None
                    break
                evidence.append([int(row), source, bary.tolist(), float(projection["distance"]), distance_rest])
            self.cache[key] = evidence
        proof = self.cache[key]
        if proof is None:
            return None
        return {"carrier": int(carrier), "group": int(dest), "tuple": identity,
                "maximum_carrier_distance_m": float(np.max(distances)),
                "rows": proof}
