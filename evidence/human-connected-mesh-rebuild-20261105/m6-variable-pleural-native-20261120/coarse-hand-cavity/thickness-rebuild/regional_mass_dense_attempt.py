"""Conserve native source mass through bounded, same-owner shell routes."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree

# Fixed density intervals from the native Human .nmatter material definitions.
DENSITY_KG_M3 = {15: (900., 1100.), 16: (850., 950.)}
MATERIAL_MACROS = {
    15: ("NUMI_HUMAN_SKIN_DERMIS_MATERIAL", "c575b204d975bfb77ae16973708e833796e3a4ad71794d69ac7e433f58dbd7ad"),
    16: ("NUMI_HUMAN_SUBCUTANEOUS_ADIPOSE_MATERIAL", "3d3a83e9d1803a60b103a45064c4b4020ebc6bfc8ae518acb6e4d3ec0cc8e29f"),
}
RADIUS_M = .075


@dataclass
class RegionalMass:
    group_for_parent: np.ndarray
    source_kg: np.ndarray
    material: np.ndarray
    donor: np.ndarray
    source_centroid: np.ndarray
    density_limits: np.ndarray

    @classmethod
    def from_original(cls, points, elements, rows, build_commands):
        typed, donor = (np.array([int(row[name]) for row in rows])
                        for name in ("material_index", "donor_body"))
        mass = np.array([float(row["parent_mass_kg"]) for row in rows])
        assert np.all(mass > 0.)
        _, group = np.unique(np.stack((typed, donor), axis=1), axis=0, return_inverse=True)
        compiled_intervals = np.array([DENSITY_KG_M3[int(t)] for t in typed])
        # Verify that the supplied bounds are those of the exact native materials.
        from build_hand_trial import digest
        arguments = json.loads(build_commands.read_text())
        for t, (macro, sha) in MATERIAL_MACROS.items():
            prefix = "-D" + macro + "="
            actual = Path(next(word[len(prefix):].strip('"') for cmd in arguments
                               for word in cmd if word.startswith(prefix)))
            assert digest(actual) == sha
            bounds = re.search(r"density\s*:\s*kg/m\^3\s*=\s*[\d.]+\s+in\s*\[([\d.]+)\s*,\s*([\d.]+)\]", actual.read_text())
            assert bounds and tuple(map(float, bounds.groups())) == DENSITY_KG_M3[t]
        return cls(group, mass, typed, donor, points[elements].mean(axis=1), compiled_intervals)

    def determinant_bounds(self):
        group = self.group_for_parent
        low, high = self.density_limits.T
        return (6. * np.bincount(group, weights=self.source_kg / high),
                6. * np.bincount(group, weights=self.source_kg / low))

    def allocate(self, points, elements, parent, det, active_parent):
        volume = det / 6.
        total = np.bincount(parent, weights=volume, minlength=len(self.source_kg))
        child_center = points[elements].mean(axis=1)
        center_sum = np.stack([np.bincount(parent, weights=volume * axis, minlength=len(total))
                               for axis in child_center.T], axis=1)
        new_center = center_sum / total[:, None]
        offsets = np.linalg.norm(child_center - new_center[parent], axis=1)
        max_offset = np.zeros(len(total))
        np.maximum.at(max_offset, parent, offsets)
        group = self.group_for_parent
        eligible = np.isin(group, np.unique(group[active_parent]))
        old = np.flatnonzero(eligible)
        if len(old) == 0:
            raise ValueError("the selected mass repair has no eligible carriers")

        sources, targets, distance = [], [], []
        for g in np.unique(group[old]):
            indices = np.flatnonzero(group == g)
            tree = cKDTree(new_center[indices])
            neighbors = tree.query_ball_point(self.source_centroid[indices], RADIUS_M)
            for a, nearby in zip(indices, neighbors, strict=True):
                # Keep every eligible route to avoid artificial capacity cuts.
                b = indices[nearby]
                d = np.linalg.norm(new_center[b] - self.source_centroid[a], axis=1)
                inside = d + max_offset[b] <= RADIUS_M
                b, d = b[inside], d[inside]
                sources.extend([a] * len(b))
                targets.extend(b)
                # Prefer retaining a source parent when both routes are valid.
                distance.extend(np.where(b == a, d, RADIUS_M + d))

        sources = np.asarray(sources, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.int32)
        route = np.arange(len(sources))
        compact = np.full(len(total), -1, dtype=np.int32)
        compact[old] = np.arange(len(old))
        dims = (len(old), len(route))
        debit = coo_matrix((np.ones(len(route)), (compact[sources], route)), shape=dims).tocsr()
        credit = coo_matrix((np.ones(len(route)), (compact[targets], route)), shape=dims).tocsr()
        density_low, density_high = self.density_limits.T
        lower, upper = density_low * total, density_high * total
        # Scale to micrograms for HiGHS feasibility on the smallest cells.
        units = 1.e9
        result = linprog(distance, A_eq=debit, b_eq=units * self.source_kg[old],
                         A_ub=vstack((credit, -credit), format="csr"),
                         b_ub=units * np.r_[upper[old], -lower[old]],
                         method="highs", options={"time_limit": 25.})
        summary = {"success": bool(result.success), "status": int(result.status),
                   "message": result.message, "eligible_parents": int(len(old)),
                   "eligible_groups": int(len(np.unique(group[old]))),
                   "eligible_routes": len(route)}
        if not result.success:
            return None, None, summary
        selected = result.x > 0.
        untouched = np.flatnonzero(~eligible)
        a = np.r_[sources[selected], untouched]
        b = np.r_[targets[selected], untouched]
        weights = np.r_[result.x[selected] / units, self.source_kg[untouched]]
        assigned = np.bincount(b, weights=weights, minlength=len(total))
        residual = np.bincount(a, weights=weights, minlength=len(total)) - self.source_kg
        route_distance = np.linalg.norm(self.source_centroid[a] - new_center[b], axis=1)
        density = assigned / total
        # Older untouched parent densities are audited as well. Tolerance here
        # only covers floating-point summation of the linear-program solution.
        max_violation = float(np.maximum(density_low - density, density - density_high).max())
        assert max_violation < 1.e-5, max_violation
        assert np.max(np.abs(residual)) < 1.e-12
        assert np.max(route_distance + max_offset[b]) <= RADIUS_M
        assert np.array_equal(self.material[a], self.material[b])
        assert np.array_equal(self.donor[a], self.donor[b])
        mass = assigned[parent] * volume / total[parent]
        summary.update(positive_routes=len(weights), nonidentity_routes=int((a != b).sum()),
                       rerouted_kg=float(weights[a != b].sum()),
                       maximum_source_residual_kg=float(np.abs(residual).max()),
                       maximum_transport_bound_m=float(np.max(route_distance + max_offset[b])),
                       maximum_density_violation_kg_m3=max(0., max_violation),
                       density_by_material={str(t): [float(density[self.material == t].min()),
                                                    float(density[self.material == t].max())]
                                            for t in DENSITY_KG_M3})
        return mass, (a, b, weights), summary
