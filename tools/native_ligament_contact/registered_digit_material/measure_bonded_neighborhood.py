"""Compare cell-scale and surrounding material deformation on the same accepted GPU step."""

import json
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import installed_topology


def measure(frame, witnesses):
    evidence = json.loads(witnesses.read_text())
    launch = json.loads((frame / "launch.json").read_text())
    X, Y, src, roles, pins = inputs(frame)
    accepted = frame / "physical-accepted.node"
    if (
        pins != evidence["geometry"]
        or sha(accepted) != evidence["accepted_geometry_sha256"]
    ):
        raise ValueError("Different reference or accepted coordinates")
    Y = Y.astype(float)
    X = X.astype(float)
    Z = np.loadtxt(accepted, skiprows=1, usecols=(1, 2, 3))
    cells, kind = installed_topology(src, roles, launch["environment"])

    def edges(v, ids):
        t = v[ids]
        return t[:, 1:] - t[:, :1]

    center = Y[cells].mean(1)
    base = edges(X, cells)
    start = edges(Y, cells)
    end = edges(Z, cells)
    v0 = np.linalg.det(base)
    v1 = np.linalg.det(start)
    v2 = np.linalg.det(end)
    if not np.all((v0 > 0) & (v1 > 0) & (v2 > 0)):
        raise ValueError("Invalid accepted geometry")

    def quantile(x, w, p):
        idx = np.argsort(x)
        cum = np.cumsum(w[idx])
        return np.interp(np.array(p) * cum[-1], cum, x[idx]).tolist()

    out = []
    for w in evidence["before_failure"]:
        k = w["native_cell"] - evidence["global_element_offset"]
        c = cells[k]
        if list(c) != w["nodes"] or kind[k] != w["material"]:
            raise ValueError("Wrong native cell")
        record = {
            "native_cell": w["native_cell"],
            "source_cell": w["source_cell"],
            "J_release": v2[k] / v1[k],
            "J_material": v2[k] / v0[k],
            "radii": [],
        }
        near = np.linalg.norm(center - center[k], axis=1)
        for radius in (0.005, 0.010, 0.020):
            mask = (near < radius) & (kind == kind[k])
            local = np.flatnonzero(mask)
            # Restrict to the face-connected component of this cell. The sphere
            # can contain an opposing limb, disconnected fat, or nearby tissue.
            owners = {}
            for j in local:
                for ii in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
                    face = tuple(sorted(cells[j, list(ii)]))
                    owners.setdefault(face, []).append(j)
            selected = {k}
            pre = [k]
            while pre:
                nxt = set()
                for j in pre:
                    for ii in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)):
                        nxt.update(owners[tuple(sorted(cells[j, list(ii)]))])
                pre = list(nxt - selected)
                selected.update(pre)
            idx = np.asarray(sorted(selected))
            idx = idx[idx != k]
            if not len(idx):
                continue
            verts, inv = np.unique(cells[idx], return_inverse=True)
            wt = np.bincount(inv.ravel(), weights=np.repeat(v1[idx] / 4, 4))
            # The local affine predictor excludes the target element and uses
            # surrounding material volume, independent of the element count.
            o = np.average(Y[verts], axis=0, weights=wt)
            design = np.c_[Y[verts] - o, np.ones(len(verts))]
            ww = np.sqrt(wt / wt.sum())
            mat, _, rank, _ = np.linalg.lstsq(
                design * ww[:, None], (Z[verts] - Y[verts]) * ww[:, None], rcond=None
            )
            if rank != 4:
                continue
            G = np.eye(3) + mat[:3]
            pred = (Y[c] - o) @ mat[:3] + mat[3]
            actual = Z[c] - Y[c]
            nr = np.linalg.norm(actual - pred, axis=1)
            record["radii"].append(
                dict(
                    radius_mm=radius * 1000,
                    adjacent_cells=len(idx),
                    surrounding_affine_J=float(np.linalg.det(G)),
                    surrounding_volume_ratio=float(v2[idx].sum() / v1[idx].sum()),
                    volume_weighted_J_quantiles=quantile(
                        v2[idx] / v1[idx], v1[idx], [0.01, 0.5, 0.99]
                    ),
                    target_nonaffine_corner_max_um=float(nr.max() * 1e6),
                    minimum_reference_altitude_um=w["minimum_material_altitude_um"],
                )
            )
        out.append(record)
    return dict(source=evidence["geometry"], accepted=sha(accepted), witnesses=out)


if __name__ == "__main__":
    print(json.dumps(measure(*(Path(s) for s in sys.argv[1:])), indent=2))
