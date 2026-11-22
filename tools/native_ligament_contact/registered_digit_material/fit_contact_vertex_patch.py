"""Test native Float32 repairs for a captured nonadjacent vertex/triangle contact on registered skin."""

import csv, json, runpy, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from material_context import sha
from fit_dorsal_material import inputs, evaluate
from compare_hand_interfaces import query


def fit(root, run, witness, reader, exe, out):
    report = json.loads(witness.read_text())
    c = report["contact"]["contact"]
    if (
        c["identity"][:3] != [15, 15, 0]
        or c["geometry_coordinate_frame"] != "reference_local"
    ):
        raise ValueError("Witness is outside the vertex/triangle skin domain")
    cfg = json.loads((run / "launch.json").read_text())
    rc = json.loads((run / "native/accepted-geometry/step-1.receipt.json").read_text())
    ref, loaded, tet, role, pins = inputs(run)
    face_data = np.load(root / "physical.npz")
    endpoint = np.loadtxt(run / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    if (
        sha(run / "physical-accepted.node")
        != json.loads((run / "matched-digital-interfaces.json").read_text())[
            "repaired"
        ]["accepted_material"]
    ):
        raise ValueError("Incorrect material endpoint")
    v = lambda name: np.array(c[name], dtype="f4").astype(float)
    s = v("potential_feature_start_high") + v("potential_feature_start_low")
    e = v("potential_feature_end_high") + v("potential_feature_end_low")
    # Bind native global indices to local material indices using the same offset at all corners and the actual O face.
    ds, near = cKDTree(endpoint).query(s + v("reference_origin_m"), k=2)
    rows = near[:, 0]
    if (
        len(set(np.array(c["nodes"][:4]) - rows)) != 1
        or np.any(ds[:, 0] > 1e-7)
        or np.any(ds[:, 1] < 2 * ds[:, 0])
    ):
        raise ValueError("Ambiguous solver/material ownership")
    maps = {layer: face_data[layer + "rows"] for layer in "OID"}
    face = {layer: maps[layer][face_data[layer + "f"]] for layer in "OID"}
    target = np.flatnonzero(
        np.all(np.sort(face["O"], axis=1) == np.sort(rows[1:]), axis=1)
    )
    if (
        len(target) != 1
        or not np.isin(rows, maps["O"]).all()
        or len(np.unique(rows)) != 4
    ):
        raise ValueError("Unsupported primitive ownership")

    def gap(x):
        normal = np.cross(x[2] - x[1], x[3] - x[1])
        normal /= np.linalg.norm(normal)
        return float(np.sign(normal @ v("normal")) * (x[0] - x[1]) @ normal)

    # Both endpoints prove that this update approaches a real signed-clearance boundary.
    clearance = c["numerical_clearance_offset_m"]
    width = c["barrier_parameters"][2]
    qenv = cfg["environment"]
    mpath = Path(
        qenv[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if (
        sha(mpath)
        != qenv[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256"
        ]
    ):
        raise ValueError("Registration mismatch")
    provenance = {}
    with mpath.open() as f:
        for rec in csv.DictReader(f):
            i = int(rec["joint_node_id"]) - 1
            if i in rows:
                provenance[i] = {
                    k: rec[k]
                    for k in (
                        "source_domain",
                        "donor_body",
                        "donor_binding",
                        "source_mapping_kind",
                        "registered_surface_layer",
                        "source_vertex_row",
                    )
                }
    reach = endpoint[face["O"][np.isin(face["O"], rows).any(1)]].reshape(-1, 3)
    lo = reach.min(0) - 1e-5
    hi = reach.max(0) + 1e-5
    xx = []
    ff = []
    ids = []
    start = 0

    def add(x, f, node):
        nonlocal start
        if np.any(x.max(0) < lo) or np.any(x.min(0) > hi):
            return
        tri = x[f]
        keep = np.flatnonzero(np.all(tri.min(1) <= hi, 1) & np.all(tri.max(1) >= lo, 1))
        if not len(keep):
            return
        ix = np.unique(f[keep])
        xx.append(x[ix])
        ff.append(start + np.searchsorted(ix, f[keep]))
        ids.append(node[ix] if node is not None else np.full(len(ix), -1))
        start += len(ix)

    for layer in "OID":
        add(endpoint[maps[layer]], face_data[layer + "f"], maps[layer])
    import hashlib

    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-dcq", str(run / "terminal.mrvpack.zst")]
    )
    if hashlib.sha256(raw).hexdigest() != rc["pack_file_sha256"]:
        raise ValueError("Incorrect anatomical endpoint")
    bodies = {r["body_index"]: r for r in rc["accepted_body_poses"]}
    for (sem, bank), _, x, f in runpy.run_path(str(reader))["primitives"](raw, bodies):
        if sem != 51007:
            add(x, f, None)
    X = np.concatenate(xx)
    F = np.concatenate(ff)
    I = np.concatenate(ids)
    tested = []
    for i in rows:
        i = int(i)
        local = np.flatnonzero(I == i)
        touch = np.isin(F, local).any(1)
        cells = tet[np.any(tet == i, 1)]
        old = query(exe, X, F, touch)
        for units in (-1, 1, -2, 2, -4, 4):
            dx = float(abs(np.spacing(ref[i, 0]))) * units
            gaps = []
            invalid = []
            for a in (ref, loaded):
                b = a[rows].astype(float)
                b[rows == i, 0] += dx
                if not np.array_equal(b, b.astype("f4")):
                    raise ValueError("Coordinate change not representable")
                gaps.append(gap(b))
                q = a.copy()
                q[i, 0] += dx
                invalid.append(evaluate(q, cells)["bad"])
            # Restoration must clear the actual native barrier activation and all touched volume gates.
            if min(gaps) <= width or any(invalid):
                continue
            candidate = X.copy()
            candidate[local, 0] += dx
            after = query(exe, candidate, F, touch)
            tested.append(
                dict(
                    node=i,
                    delta_x_m=dx,
                    source_gap_m=gaps[0],
                    loaded_gap_m=gaps[1],
                    new_pairs=len(after - old),
                    cleared_pairs=len(old - after),
                    incident_cells=len(cells),
                    prior_pairs=len(old),
                )
            )
    good = [r for r in tested if not r["new_pairs"]]
    result = dict(
        physical_rows=rows.tolist(),
        outer_triangle=int(target[0]),
        ancestry=provenance,
        reference_gap_m=gap(ref[rows].astype(float)),
        loaded_gap_m=gap(loaded[rows].astype(float)),
        iteration_start_gap_m=gap(s),
        requested_fraction_gap_m=gap(s + c["requested_alpha"] * (e - s)),
        unit_correction_gap_m=gap(e),
        protected_margin_m=clearance,
        activation_m=width,
        object_node_offset=int(c["nodes"][0] - rows[0]),
        source=witness.name,
        witness_sha256=sha(witness),
        trial_count=len(tested),
        trials=tested,
        inputs=pins,
    )
    out.mkdir(exist_ok=True)
    if good:
        selected = min(
            good,
            key=lambda r: (
                abs(r["delta_x_m"]),
                -r["cleared_pairs"],
                -r["source_gap_m"],
            ),
        )
        result["selected"] = selected
        np.savez(
            out / "local-field.npz",
            nodes=[selected["node"]],
            delta_x_m=[selected["delta_x_m"]],
        )
    (out / "pelvic-fit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("selected", {})))


if __name__ == "__main__":
    fit(*(Path(v).resolve() for v in sys.argv[1:]))
