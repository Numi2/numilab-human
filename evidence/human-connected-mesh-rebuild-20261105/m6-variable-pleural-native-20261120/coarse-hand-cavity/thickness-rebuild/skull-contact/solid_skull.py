"""Reusable exact point queries for individual closed bones in the skull bank."""
from collections import Counter

import numpy as np

from audit_source import owner, records


class ClosedBone:
    def __init__(self, identity, triangles):
        self.identity = int(identity)
        self.lower = triangles.min(axis=(0, 1))
        self.upper = triangles.max(axis=(0, 1))
        by_position = {}
        edge_use = Counter()
        direction_balance = Counter()
        snapshot = []
        for vertices, lo, hi, face, _ in records(triangles):
            indices = tuple(by_position.setdefault(vertex, len(by_position)) for vertex in vertices)
            for a, b in zip(indices, indices[1:] + indices[:1], strict=True):
                edge = tuple(sorted((a, b)))
                edge_use[edge] += 1
                direction_balance[edge] += 1 if a < b else -1
            snapshot.append((vertices, lo, hi, face, indices))
        if any(value != 2 for value in edge_use.values()) or any(direction_balance.values()):
            raise ValueError(f"bone {self.identity} does not have a closed oriented boundary")
        self.predicate = owner.prepare_point_location(snapshot)

    def locations(self, points, nodes):
        selected = nodes[np.all((points[nodes] >= self.lower) & (points[nodes] <= self.upper), axis=1)]
        located = {}
        for node in selected:
            point = owner.float32_point_lattice_key(points[node])
            location = owner.point_location_prepared_indexed(point, self.predicate)["location"]
            if location != "outside":
                located[int(node)] = location
        return located


class SolidSkull:
    def __init__(self, bank, nodes, initial):
        self.bones = [ClosedBone(identity, bank[bank["bone"] == identity]["vertices"])
                      for identity in np.unique(bank["bone"])]
        self.nodes = nodes
        old = self.locations(initial, nodes)
        if any(status == "indeterminate" for status in old.values()):
            raise ValueError("unresolved original skull point predicates")
        self.original = old

    def locations(self, state, nodes):
        return {(bone.identity, node): location for bone in self.bones
                for node, location in bone.locations(state, nodes).items()}

    def newly_unsafe(self, trial, nodes):
        locations = self.locations(trial, nodes)
        unsafe = set()
        for key, status in locations.items():
            before = self.original.get(key)
            if before is None or status == "indeterminate" or (status == "inside" and before != "inside"):
                unsafe.add(key[1])
        return sorted(unsafe)
