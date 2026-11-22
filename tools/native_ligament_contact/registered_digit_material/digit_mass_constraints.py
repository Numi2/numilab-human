"""Constrain the signed P1 volume change using the existing per-group physical mass."""

import json
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from material_context import sha


class FixedCarrierVolumes:
    def __init__(
        self,
        owner,
        pins,
        ids,
        role,
        cell_gradient,
        local_volume,
        make_matrix,
        eligible_source=None,
    ):
        receipt = json.loads(
            (owner / "fine-mass-routed/mass-transfer.json").read_text()
        )
        source = owner / "fine-mass-routed/current-density-closed-flow.npz"
        measure = owner / "native.volumes"
        if (
            sha(source) != receipt["flow_sha256"]
            or sha(measure) != receipt["native_volume_sha256"]
            or sha(owner / "mesh/mesh.node") != pins["reference_sha256"]
        ):
            raise ValueError("Mass owner does not match reference mesh")
        f = np.load(source)
        cell = f["group_for_cell"]
        mass = np.bincount(f["group"], weights=f["mass_flow_kg"])
        ref = np.fromfile(measure, dtype="f8")
        n = len(mass)
        total = np.bincount(cell, weights=ref, minlength=n)
        self.cell_groups = cell[ids]
        _, first = np.unique(cell, return_index=True)
        if not np.array_equal(role[first][cell], role):
            raise ValueError("Material changes inside owner group")
        lo = np.where(role[first] == 15, 900.0, 850.0)
        hi = np.where(role[first] == 15, 1100.0, 950.0)
        mandatory = mass
        possible = mass
        if eligible_source:
            dense = np.load(eligible_source / "current-density-closed-flow.npz")
            cc = f["carrier_row"]
            gg = f["group"]
            flow = f["mass_flow_kg"]
            if not np.array_equal(cc, dense["carrier_row"]) or not np.array_equal(
                gg, dense["group"]
            ):
                raise ValueError("Unmatched carrier domain")
            active = ((flow > 0) | (dense["mass_flow_kg"] > 0)) & ~f[
                "archived_compensation"
            ]
            fixed = np.bincount(gg[~active], weights=flow[~active], minlength=n)
            carriers, groups = np.unique(
                np.stack((cc[active], gg[active]), axis=1), axis=0
            ).T
            degree = np.bincount(carriers)
            quota = np.bincount(cc[active], weights=flow[active])
            one = degree[carriers] == 1
            mandatory = fixed + np.bincount(
                groups[one], weights=quota[carriers[one]], minlength=n
            )
            possible = fixed + np.bincount(groups, weights=quota[carriers], minlength=n)
            if np.any(mandatory > mass + 1e-15) or np.any(possible < mass - 1e-15):
                raise ValueError("Baseline transfer lies outside eligible capacities")
            mandatory = np.minimum(mandatory, mass)
            possible = np.maximum(possible, mass)
        self.lower = mandatory / (hi * total) - 1
        self.upper = possible / (lo * total) - 1
        if np.any(self.lower > 0) or np.any(self.upper < 0):
            raise ValueError("Baseline violates group density")
        agg = coo_matrix(
            (
                local_volume / total[self.cell_groups],
                (self.cell_groups, np.arange(len(ids))),
            ),
            shape=(n, len(ids)),
        ).tocsr()
        self.matrix = (agg @ make_matrix(cell_gradient, np.arange(len(ids)))).tocsr()
        self.matrix.eliminate_zeros()
        self.row_l1 = np.asarray(abs(self.matrix).sum(1)).ravel()
        self.active = set()
        self.frozen = set()

    def constraints(self, x, quantum):
        # Certify the exact displacement that both coordinate files will emit, in millimetres.
        unit = 1000.0 * quantum
        y = self.matrix @ (np.rint(x / unit) * unit)
        bad = np.flatnonzero((y < self.lower) | (y > self.upper))
        rows = []
        rhs = []
        freeze = set()
        for group in bad:
            sign = 1 if y[group] > self.upper[group] else -1
            key = (int(group), sign)
            if key in self.active:
                # Rounded near-bound repeats are resolved by the admitted zero change.
                freeze.update(self.matrix.getrow(group).indices.tolist())
                continue
            self.active.add(key)
            bound = self.upper[group] if sign == 1 else -self.lower[group]
            margin = min(bound * 0.5, 0.501 * unit * self.row_l1[group] + 2e-10)
            rows.append(self.matrix.getrow(group) * sign)
            rhs.append(max(0.0, bound - margin))
        if rows:
            from scipy.sparse import vstack

            A = vstack(rows, format="csr")
        else:
            A = csr_matrix((0, len(x)))
        new = freeze - self.frozen
        self.frozen.update(new)
        return hstack([A, csr_matrix(A.shape)], format="csr"), np.asarray(rhs), new
