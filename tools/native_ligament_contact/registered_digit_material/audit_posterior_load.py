"""Account for posterior tissue, gravity and bed loads using every incident tetrahedron."""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from bind_posterior_equilibrium import coordinates
from fit_dorsal_material import inputs
from fit_native_bed_barrier_load import force
from fit_posterior_equilibrium import materials
from material_context import sha


def account(run, parent, case, bed, split, out):
    ref, previous, tet, role, ids = inputs(run)
    fit = json.loads((case / "fit.json").read_text())
    if ids != fit["inputs"] or sha(case / "current.node") != fit["output_pose"]:
        raise ValueError("Starting geometry hashes differ")
    support = json.loads((bed / "load.json").read_text())
    anchor = np.array(
        [
            i["volume_row"]
            for i in json.loads((bed / "loaded-registration.json").read_text())[
                "attachment_rows"
            ]
        ]
    )
    layers = np.load(split / "physical.npz")
    bed_nodes = np.array(support["nodes"])
    target = (cKDTree(previous[bed_nodes, :2]).query(previous[:, :2])[0] < 0.08) & (
        previous[:, 2] < 0.075
    )
    cells = np.flatnonzero(target[tet].any(axis=1))
    conn = tet[cells]
    ref = ref.astype(float)
    rest = ref[conn[:, 1:]] - ref[conn[:, :1]]
    vol = np.linalg.det(rest) / 6
    G = np.linalg.inv(rest).transpose(0, 2, 1)
    G = np.concatenate([-G.sum(1, keepdims=True), G], axis=1)
    param, pins = materials(parent / "native-owner")
    mu, bulk = param[role[cells] - 15].T
    massfile = parent / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(massfile) != fit["source_mass"]:
        raise ValueError("Unmatched P1 mass")
    mass = np.load(massfile)["nodal_mass_kg"]
    rows = np.flatnonzero(target)
    group = np.full(len(rows), "interior", dtype="<U20")
    for k in "DIO":
        group[np.isin(rows, layers[k + "rows"])] = k
    group[np.isin(rows, bed_nodes)] = "bed"
    group[np.isin(rows, anchor)] = "anchor"
    measurements = {}
    for key, xyz in [
        ("parent", previous.astype(float)),
        ("candidate", coordinates(case / "current.node")),
    ]:
        F = (xyz[conn[:, 1:]] - xyz[conn[:, :1]]).transpose(0, 2, 1) @ G[:, 1:]
        det = np.linalg.det(F)
        P = mu[:, None, None] * F + (bulk * np.log(det) - mu)[
            :, None, None
        ] * np.linalg.inv(F).transpose(0, 2, 1)
        nodal = np.zeros(xyz.shape)
        internal = -vol[:, None, None] * (G @ P.transpose(0, 2, 1))
        np.add.at(nodal, conn, internal)
        nodal[rows, 2] -= mass[rows] * 9.81
        # All 143 original outer gaps are unchanged and support load admission checks their eligibility.
        traction = force(
            xyz[bed_nodes, 2], mass[bed_nodes], support["dt_s"], support["band_m"]
        )
        nodal[bed_nodes, 2] += traction
        result = {}
        for name in sorted(set(group)):
            subset = rows[group == name]
            norm = np.linalg.norm(nodal[subset], axis=1)
            top = int(subset[np.argmax(norm)])
            result[name] = dict(
                count=len(subset),
                total_N=nodal[subset].sum(axis=0).tolist(),
                force_l2_N=float(np.linalg.norm(norm)),
                max_N=float(norm.max()),
                max_row=top,
            )
        measurements[key] = dict(groups=result, stationary_bed_N=float(traction.sum()))
    data = dict(
        source=ids,
        fitted_pose=fit["output_pose"],
        mass=sha(massfile),
        materials=pins,
        measurement_rows=len(rows),
        incident_tetrahedra=len(conn),
        results=measurements,
        scope="Initial skin/fat constitutive stress, P1 gravity and stationary bed tractions. Other anatomy contact, skeleton insertion reactions and inertia are determined by the native owner.",
    )
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        [
            (n, round(v["force_l2_N"], 1))
            for n, v in measurements["candidate"]["groups"].items()
        ]
    )


if __name__ == "__main__":
    account(*(Path(p).resolve() for p in sys.argv[1:]))
