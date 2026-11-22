"""Recover an emitted material-quality witness in its pinned source topology."""

import json
import re
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def accepted_node_pin(trial):
    historical = trial / "physical-recovery.json"
    if historical.is_file():
        return json.loads(historical.read_text())["verified_member_sha256"][
            "accepted.node"
        ]
    layers = json.loads((trial / "accepted-layers.json").read_text())
    receipt = json.loads((trial / "terminal.receipt.json").read_text())
    state = json.loads((trial / "admission.json").read_text())
    if (
        receipt["physical_endpoint"] != "accepted"
        or receipt["pack_file_sha256"] != state["terminal_pack_sha256"]
        or receipt["accepted_step"] != state["accepted_steps"]
        or state["exit_code"]
    ):
        raise ValueError("Missing accepted material transaction")
    return layers["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"]


def locate(run, proposal, out):
    launch = json.loads((run / "launch.json").read_text())
    if (
        sha(proposal)
        != launch["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]
    ):
        raise ValueError("Unmatched cavity composition")
    X, Y, t, tag, pins = inputs(run)
    path = run / "physical-accepted.node"
    if sha(path) != accepted_node_pin(run):
        raise ValueError("Different native coordinates")
    end = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))
    rows = []
    pattern = r"^registered_cell_shape_witness local=(\d+) material=(\d+) node0=(\d+) node1=(\d+) node2=(\d+) node3=(\d+) volume_m3=(\S+) shape=(\S+)$"
    with np.load(proposal) as p:
        keep = p["retained_input_cell_rows"]
        for m in re.finditer(pattern, (run / "stdout.log").read_text(), re.MULTILINE):
            offset, material, *nodes = map(int, m.groups()[:6])
            expected = float(m.group(8))
            if offset >= len(keep):
                raise ValueError("Witness lies in a retessellated core")
            orig = int(keep[offset])
            node = t[orig]
            delta = np.asarray(nodes) - node
            if np.any(delta != delta[0]) or tag[orig] != material:
                raise ValueError("Native material node order disagrees")
            # Node and cell offsets are independent in the packed physical object.
            v, q = geom(end, node[None])
            if not np.isclose(q[0], expected, rtol=1e-6, atol=0) or not np.isclose(
                v[0], float(m.group(7)), rtol=1e-6, atol=0
            ):
                raise ValueError("Witness determinant or shape differs")
            rows.append(
                dict(
                    parents=[orig],
                    exactly_retained=True,
                    native_material_row=offset,
                    local_nodes=node.tolist(),
                    native_node_offset=int(delta[0]),
                    material=material,
                    accepted_shape=float(q[0]),
                    reference_shape=float(geom(X.astype(float), node[None])[1][0]),
                    loaded_shape=float(geom(Y.astype(float), node[None])[1][0]),
                )
            )
    if not rows:
        raise ValueError("No emitted quality witnesses")
    out.write_text(
        json.dumps(
            dict(material_witnesses=rows, accepted=sha(path), source=pins), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    locate(*(Path(s).resolve() for s in sys.argv[1:]))
