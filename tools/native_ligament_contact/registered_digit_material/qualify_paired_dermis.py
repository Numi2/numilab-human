"""Check the exact installed solid and its new mass provenance before native release."""

import json
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import check_installed


def run(control, material, case, parent_transfer, out):
    source = json.loads((case / "material-identity.json").read_text())
    role = json.loads((case / "material/native-role-verification.json").read_text())
    config = json.loads((case / "qualified-parent/launch.json").read_text())
    env = config["environment"]
    owner = case / "native-owner"
    ids = json.loads((owner / "identities.json").read_text())
    parent = json.loads(parent_transfer.read_text())
    mass = json.loads((case / "fine-mass-routed/mass-transfer.json").read_text())
    if (
        not role["passed"]
        or config["connected_material_volume"]["role_reader_sha256"]
        != sha(case / "material/native-role-verification.json")
        or ids["binary_sha256"] != sha(owner / "numi-human-native")
    ):
        raise ValueError("Unverified native material")
    if (
        source["reference"] != sha(case / "mesh/mesh.node")
        or source["current"] != sha(case / "current.node")
        or ids["loaded_pose_override"]["sha256"] != source["current"]
    ):
        raise ValueError("Unmatched physical pose")
    _x0, y0, t, tag, pins = inputs(control)
    if (
        pins != source["source"]
        or mass["parent_flow_sha256"] != parent["mass"]
        or mass["new_carrier_augmentations"] != 0
        or mass["changed_edges"] != 0
        or mass["density_outliers"]
    ):
        raise ValueError("Changed measured source")
    x, y = [
        np.loadtxt(case / n, skiprows=1, usecols=(1, 2, 3))
        for n in ("mesh/mesh.node", "current.node")
    ]
    previous = json.loads((control / "launch.json").read_text())["environment"]
    key = "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"
    pin = "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"
    if (
        sha(Path(previous[key])) != parent["topology"]
        or previous[pin] != parent["topology"]
    ):
        raise ValueError("Unmatched native cavity selection")
    rpt = check_installed(
        x, y, t, tag, {**env, key: previous[key], pin: previous[pin]}, owner
    )
    # The set of outer bed nodes and their physical masses are identical at
    # initialization; the 4 relocated material points are interior.
    with (
        np.load(material / "fine-mass-routed/current-density-closed-flow.npz") as a,
        np.load(case / "fine-mass-routed/current-density-closed-flow.npz") as b,
    ):
        active = y0[:, 2] < float.fromhex("0x1.4f8b580000000p-17")
        if not np.array_equal(y0[active], y[active]) or not np.array_equal(
            a["nodal_mass_kg"][active], b["nodal_mass_kg"][active]
        ):
            raise ValueError("Bed preload differs")
    ids["installed_tissue_admission"] = rpt
    (owner / "identities.json").write_text(json.dumps(ids, indent=2) + "\n")
    parent.update(
        current=source["current"],
        mass=mass["flow_sha256"],
        parent_transfer_sha256=sha(parent_transfer),
        matched_bed_rows_and_masses=True,
    )
    out.write_text(json.dumps(parent, indent=2) + "\n")
    (case / "qualification.json").write_text(json.dumps(rpt, indent=2) + "\n")


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))
