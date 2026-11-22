"""Bind native trial determinant failures to exact source cells, physical rows and donor identities."""

from pathlib import Path
import csv, json, re, sys
import numpy as np
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from material_context import sha


def describe(run, witness, one, out):
    stdout = (run / "stdout.log").read_text()

    def extract(regex, expected):
        m = re.findall(regex, stdout, re.M)
        if len(m) != expected:
            raise ValueError((regex, len(m), expected))
        return np.array(m, dtype=int)

    fs = extract(
        r"^resting_fascia_object=source_prepared object=\d+ .*?nodes=(\d+) tetrahedra=(\d+)",
        12,
    )
    acl = extract(r"^resting_inferred_acl=admitted .*?nodes=(\d+) tetrahedra=(\d+)", 2)
    wall = extract(
        r"^resting_aortic_wall_outer_contact=accepted .*?outer_ring_rigid_query_nodes=(\d+) .*?retained_wall_faces=(\d+)",
        1,
    )[0]
    # The registered annulus has two node rings at each axial station and six tetrahedra per sector,
    # with two exposed triangles per outer quad. Its hydraulic end ports contribute no solid cells.
    offset = fs.sum(axis=0) + acl.sum(axis=0) + np.array([2 * wall[0], 3 * wall[1]])
    w = json.loads(witness.read_text())
    if offset[0] != w["object_node_offset"]:
        raise ValueError("Object prefix differs from witnessed node transfer")
    trace = None
    for line in (run / "stderr.log").open():
        if line.startswith("fem_backtracking_trials="):
            if trace is not None:
                raise ValueError("Multiple candidate selection traces")
            trace = json.loads(line.partition("=")[2])
    if not trace or trace["object"] != 15:
        raise ValueError("Missing skin/fat trace")
    cells = sorted({i for r in trace["trials"] for i in r["tile_witness_tets"]})
    if any(not t["tile_witnesses_complete"] for t in trace["trials"]):
        raise ValueError("Truncated failure trace")
    xx, yy, t, mat, ids = inputs(run)
    use = np.asarray(cells) - offset[1]
    if np.any(use < 0) or np.any(use >= len(t)):
        raise ValueError("Witness escaped material volume")
    if one is not None:
        control = json.loads((one / "native-outcome.json").read_text())
        if (
            control["accepted_steps"] != 1
            or control["exit_code"]
            or inputs(one)[4] != ids
        ):
            raise ValueError("Checkpoint source differs")
        end = np.loadtxt(one / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    else:
        end = None
    select = t[use]
    ref = xx.astype(float)
    loaded = yy.astype(float)
    vol, shape = geom(ref, select)
    loadV, _ = geom(loaded, select)
    endV = None if end is None else geom(end, select)[0]
    D = ref[select[:, 1:]] - ref[select[:, :1]]
    sv = np.linalg.svd(D, compute_uv=False)
    m = json.loads((run / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    node_map = Path(m[prefix + "COMBINED_NODE_MAP_CSV"])
    if sha(node_map) != m[prefix + "COMBINED_NODE_MAP_SHA256"]:
        raise ValueError("Registration changed")
    membership = {}
    wanted = set(select.ravel().tolist())
    with node_map.open() as f:
        for r in csv.DictReader(f):
            n = int(r["joint_node_id"]) - 1
            if n in wanted:
                membership.setdefault(n, set()).add(
                    (r["registered_surface_layer"], r["donor_body"], r["source_domain"])
                )
    info = []
    for i, g in enumerate(cells):
        rr = select[i]
        origin = loaded[rr].mean(0)
        info.append(
            dict(
                global_element=g,
                source_row=int(use[i]),
                material=int(mat[use[i]]),
                physical_nodes=rr.tolist(),
                centroid_m=origin.tolist(),
                source_volume_m3=float(vol[i]),
                source_normalized_determinant=float(shape[i]),
                reference_edge_condition_number=float(sv[i, 0] / sv[i, -1]),
                loaded_J=float(loadV[i] / vol[i]),
                first_accepted_J=None if end is None else float(endV[i] / vol[i]),
                ancestry=[sorted(membership[int(n)]) for n in rr],
            )
        )
    res = dict(
        rows=info,
        selected_alpha=trace["selected_alpha"],
        newton=trace["iteration"],
        witness_trials=trace["trials"],
        source_element_offset=int(offset[1]),
        source_node_offset=int(offset[0]),
        source=ids,
        checkpoint_sha256=None if end is None else sha(one / "physical-accepted.node"),
        scope="First invalid witness per native reduction tile and trial, plus its independently evaluated source and checkpoint Jacobian.",
    )
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(
        [
            (
                r["global_element"],
                r["material"],
                r["source_normalized_determinant"],
                r["loaded_J"],
                r["first_accepted_J"],
            )
            for r in info
        ]
    )


if __name__ == "__main__":
    r, w, endpoint, o = sys.argv[1:]
    describe(Path(r), Path(w), None if endpoint == "-" else Path(endpoint), Path(o))
