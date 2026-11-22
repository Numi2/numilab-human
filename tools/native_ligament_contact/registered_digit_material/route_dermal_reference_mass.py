"""Bind fixed source-carrier deliveries to the repaired native tetrahedra."""

import json
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from registered_material_stars import project_mass


def route(control, material, case):
    authorities = json.loads((case / "material-identity.json").read_text())
    if (
        sha(case / "mass-patch.npz") != authorities["mass_patch"]
        or sha(case / "current.node") != authorities["current"]
        or sha(case / "mesh/mesh.node") != authorities["reference"]
    ):
        raise ValueError("Changed material emission")
    d = material / "fine-mass-routed"
    name = "current-density-closed-flow.npz"
    source = d / name
    lineage = json.loads((d / "mass-transfer.json").read_text())
    x, y, t, role, pins = inputs(control)
    if sha(source) != lineage["flow_sha256"] or pins != authorities["source"]:
        raise ValueError("Unknown mass or material parent")
    with np.load(source) as f:
        data = dict(f)
    with np.load(case / "mass-patch.npz") as f:
        rows = f["cells"]
        prior = f["old_mass_kg"]
        new = f["element_mass_kg"]
    m0 = data["element_mass_kg"]
    m = m0.copy()
    cg = data["group_for_cell"]
    g = data["group"]
    supply = np.bincount(g, weights=data["mass_flow_kg"])
    original = np.bincount(cg, weights=m0)
    if not np.array_equal(m0[rows], prior) or not np.all(role[rows] == 15):
        raise ValueError("Wrong source mass cells")
    volume = np.fromfile(case / "native.volumes", dtype="f8")
    if volume.shape != m0.shape or np.any(volume <= 1e-18):
        raise ValueError("Invalid native volume")
    # Use the exact production import volumes and the donor's original supply.
    # Leave room for the final binary64 normalization applied by the role writer.
    m[rows] = new
    lo = np.where(role == 15, 900.0, 850.0)
    hi = np.where(role == 15, 1100.0, 950.0)
    slack = 1.0e-5
    lower = (
        np.maximum(lo[rows], lo[rows] * original[cg[rows]] / supply[cg[rows]]) + slack
    )
    upper = (
        np.minimum(hi[rows], hi[rows] * original[cg[rows]] / supply[cg[rows]]) - slack
    )
    z = np.loadtxt(control / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    X = x.astype(float)
    Y = y.astype(float)
    Z = z.copy()
    ids = np.array(authorities["rows"])
    move = np.array(authorities["moves_m"])
    for points in (X, Y, Z):
        points[ids] += move
    if not np.array_equal(
        X[ids], np.loadtxt(case / "mesh/mesh.node", skiprows=1, usecols=(1, 2, 3))[ids]
    ) or not np.array_equal(
        Y[ids], np.loadtxt(case / "current.node", skiprows=1, usecols=(1, 2, 3))[ids]
    ):
        raise ValueError("Material coordinate mismatch")
    stats = []
    for component in authorities["cases"]:
        # Each fitted cavity was disjoint. Preserve the affine source moment of
        # the complete union and verify each donor separately below.
        if not component["force_and_energy"]:
            raise ValueError("Missing regional mechanics check")
    nodes = np.unique(t[rows])
    tt = np.searchsorted(nodes, t[rows])
    # project_mass computes its reference volumes directly. Correct the input
    # density interval so the resulting element masses meet the native volumes.
    from torso_cell_geometry import geom

    ratio = volume[rows] / geom(X[nodes], tt)[0]
    for inclusion in (slack, 1e-7, 1e-9):
        lim = (lower * ratio - (slack - inclusion), upper * ratio + (slack - inclusion))
        candidate = project_mass(
            [s[nodes].astype(float) for s in (x, y, z)],
            [s[nodes] for s in (X, Y, Z)],
            tt,
            prior,
            cg[rows],
            lim,
        )
        if candidate is not None:
            m[rows] = candidate
            break

    def violations():
        rescale = supply / np.bincount(cg, weights=m)
        raw = m / volume
        physical = m * rescale[cg] / volume
        return np.flatnonzero(
            (raw < lo) | (raw > hi) | (physical < lo) | (physical > hi)
        )

    if len(violations()):
        # Recover bit-identical group sums so untouched boundary-density cells
        # receive exactly their original native normalization.
        for group in np.unique(cg[rows]):
            allids = np.flatnonzero(cg == group)
            subset = rows[cg[rows] == group]
            headroom = np.minimum(
                m[subset] - lo[subset] * volume[subset],
                hi[subset] * volume[subset] - m[subset],
            )
            j = subset[np.argmax(headroom)]
            for _ in range(6):
                current = np.cumsum(m[allids])[-1]
                delta = original[group] - current
                if delta == 0:
                    break
                m[j] += delta
        # Repair residual binary64 allocation deficits with a paired exchange
        # inside the same source group and the same fitted region.
        for _ in range(6):
            bad = violations()
            if not len(bad):
                break
            scales = supply / np.bincount(cg, weights=m)
            for j in bad:
                if j not in rows:
                    raise ValueError(f"Unchanged density witness {j}")
                group = cg[j]
                q = rows[cg[rows] == group]
                q = q[q != j]
                r = scales[group]
                if not len(q):
                    raise ValueError("No same-group capacity")
                lower_j = max(lo[j], lo[j] / r) * volume[j]
                upper_j = min(hi[j], hi[j] / r) * volume[j]
                guard = max(32 * np.spacing(m[j]), volume[j] * 1e-8)
                shift = (
                    lower_j + guard - m[j]
                    if m[j] <= lower_j
                    else upper_j - guard - m[j]
                )
                remaining = (
                    m[q] - np.maximum(lo[q], lo[q] / r) * volume[q]
                    if shift > 0
                    else np.minimum(hi[q], hi[q] / r) * volume[q] - m[q]
                )
                peer = q[np.argmax(remaining)]
                if remaining.max() < 2 * abs(shift):
                    raise ValueError("Insufficient source-group capacity")
                m[j] += shift
                m[peer] -= shift
        bad = violations()
        if len(bad):
            raise ValueError(
                f"Native density bounds failed for {len(bad)} cells, first {bad[:10].tolist()}"
            )
    delta_m = m - m0
    grp = float(np.abs(np.bincount(cg[rows], weights=delta_m[rows])).max())
    if grp > 2e-12:
        raise ValueError("Source-group mass not preserved")
    labpath = case / "mesh/current-child.labels.csv"
    lab = np.loadtxt(labpath, delimiter=",", skiprows=1, usecols=(2,), dtype=int)
    for donor in np.unique(lab[rows]):
        ix = rows[lab[rows] == donor]
        ds = [
            (
                delta_m[ix] @ b[t[ix]].mean(1) + m[ix] @ (a[t[ix]] - b[t[ix]]).mean(1)
            ).tolist()
            for b, a in ((x.astype(float), X), (y.astype(float), Y), (z, Z))
        ]
        if np.max(np.abs(ds)) > 1e-10 or abs(sum(delta_m[ix])) > 1e-10:
            raise ValueError("Donor mass or first moment drift")
        stats.append({"donor": int(donor), "first_moment_change_kg_m": ds})
    data["element_mass_kg"] = m
    # Preserve the exact historical summation outside affected tetrahedra.
    nm = data["nodal_mass_kg"].copy()
    for k in range(4):
        np.add.at(nm, t[rows, k], delta_m[rows] / 4)
    data["nodal_mass_kg"] = nm
    out = case / "fine-mass-routed"
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / name, **data)
    report = {
        "parent_flow_sha256": sha(source),
        "flow_sha256": sha(out / name),
        "geometry_sha256": sha(case / "proposal.npz"),
        "native_volume_sha256": sha(case / "native.volumes"),
        "admitted_route_distance_m": lineage["admitted_route_distance_m"],
        "maximum_carrier_mass_change_kg": 0.0,
        "archived_compensation_unchanged": True,
        "changed_edges": 0,
        "density_outliers": 0,
        "group_max_abs_kg": grp,
        "corrected_cell_count": int(np.sum(m != m0)),
        "regional_mass": stats,
    }
    launch_path = material / "material/native-role-launch.json"
    reader = json.loads(
        (material / "material/native-role-verification.json").read_text()
    )
    launch = json.loads(launch_path.read_text())
    if reader.get("passed") is not True or reader.get("exit_code") != 0:
        raise ValueError(
            "Inherited allocation was not admitted by the production reader"
        )
    if (
        Path(launch["argv"][-3]).resolve() != source.resolve()
        or launch["argv"][-2] != lineage["flow_sha256"]
    ):
        raise ValueError("Parent reader was admitted against a different carrier flow")
    count = int(launch["argv"][-1])
    if count < 0:
        raise ValueError("Invalid inherited allocation count")
    with np.load(source) as parent:
        for key in parent.files:
            if key not in {"element_mass_kg", "nodal_mass_kg"} and not np.array_equal(
                data[key], parent[key]
            ):
                raise ValueError(f"Changed inherited source route {key}")
    report.update(
        solver_output=f"inherited source-carrier allocation augmentations={count}",
        used_new_typed_routes=0,
        inherited_allocation_augmentation_count=count,
        new_carrier_augmentations=0,
        allocation_provenance=sha(launch_path),
    )
    (out / "mass-transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    route(*(Path(v).resolve() for v in sys.argv[1:]))
