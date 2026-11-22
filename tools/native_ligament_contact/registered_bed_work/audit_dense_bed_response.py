"""Replay the installed flat-bed elastic barrier on accepted coordinates to diagnose support imbalance."""

import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
from fit_native_bed_barrier_load import force
from run_bed_work import digest as sha


def measure(case, material, transfer, peak_loads=False):
    job = json.loads((case / "launch.json").read_text())
    receipt = json.loads((case / "terminal.receipt.json").read_text())
    audit = json.loads((case / "accepted-layers.json").read_text())
    provenance = json.loads(transfer.read_text())
    env = job["environment"]
    args = job["argv"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def pinned(key, digest):
        p = Path(env[prefix + key])
        if sha(p) != env[prefix + digest]:
            raise ValueError("Unmatched material input " + key)
        return p

    table = pinned("COMBINED_NODE_MAP_CSV", "COMBINED_NODE_MAP_SHA256")
    with table.open() as source:
        mapped = np.array(
            [
                (int(row["joint_node_id"]) - 1, int(row["donor_body"]))
                for row in csv.DictReader(source)
                if row["registered_surface_layer"] == "O"
            ]
        )
    outer, donors = mapped.T
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if (
        sha(flow) != provenance["mass"]
        or sha(flow)
        != json.loads((flow.parent / "mass-transfer.json").read_text())["flow_sha256"]
    ):
        raise ValueError("Mass flow differs from loaded material")
    replacement = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if (
        sha(replacement) != provenance["topology"]
        or sha(replacement) != env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]
    ):
        raise ValueError("Tetrahedral replacement differs")
    with np.load(flow) as f:
        mass = f["element_mass_kg"]
        nodal = f["nodal_mass_kg"].copy()
    tets = (
        np.loadtxt(
            pinned("TETGEN_ELEMENTS", "TETGEN_ELEMENTS_SHA256"),
            skiprows=1,
            usecols=(1, 2, 3, 4),
            dtype="i4",
        )
        - 1
    )
    with np.load(replacement) as f:
        lost = np.setdiff1d(np.arange(len(tets)), f["retained_input_cell_rows"])
        for corner in range(4):
            np.add.at(nodal, tets[lost, corner], -mass[lost] / 4)
            np.add.at(
                nodal,
                f["replacement_tetrahedra"][:, corner],
                f["replacement_mass_kg"] / 4,
            )
    if (nodal[outer] <= 0).any() or len(np.unique(outer)) != len(outer):
        raise ValueError("Invalid bed mass map")
    log = (case / "stdout.log").read_text()
    if "normal=0,0,1 plane_offset_m=0" not in log or "witness_force=0" not in log:
        raise ValueError(
            "Requires the actual native horizontal plane with observational witnesses"
        )
    terminal = next(
        l
        for l in log.splitlines()
        if l.startswith("resting_fem_skin_bed_accepted_root_reaction ")
    )
    native = {
        k: float(re.search(r"(?:^| )" + k + r"=([\d.eE+-]+)", terminal)[1])
        for k in ("mean_normal_force_N", "root_dt_s", "valid_contact_sample_count")
    }
    dt = float(args[args.index("--muscle-step-seconds") + 1])
    if not np.isclose(dt, native["root_dt_s"], rtol=1e-8):
        raise ValueError("Root timestep mismatch")
    width = float.fromhex("0x1.4f8b580000000p-17")

    def state(name):
        p = case / f"physical-{name}.node"
        if sha(p) != audit["inputs"][f"NATIVE_{name.upper()}_NODE_SHA256"]:
            raise ValueError("Different physical snapshot")
        z = np.loadtxt(p, skiprows=1, usecols=(3,))[outer]
        active = z < width
        if z.min() <= 0:
            raise ValueError("Bed surface has penetrated")
        rows = np.zeros(len(z))
        rows[active] = force(z[active], nodal[outer[active]], dt, width)
        return z, rows

    ref, init = state("reference")
    end, final = state("accepted")
    weight = float(args[args.index("--resting-reference-mass-kg") + 1]) * 9.81
    if (
        abs(init.sum() - provenance["after_N"]) > 0.01
        or int((init > 0).sum()) != provenance["support_rows"]
    ):
        raise ValueError("Replay failed the exact preparation check")
    order = np.argsort(np.abs(final - init))[::-1]
    from numilab_human.capsular_material_registration import (
        read_material_banks,
        material_bank,
    )

    bones = Path(args[3])
    if sha(bones) != job["pins"][str(bones)]:
        raise ValueError("Bone names differ from native trial")
    bank = read_material_banks(bones)
    groups = []
    for donor in np.unique(donors[(init > 0) | (final > 0)]):
        names = {
            material_bank(bank, sid, int(donor))[0]["myosim_body"]
            for sid, rec in bank[2].items()
            if rec[0] == donor
        }
        mask = donors == donor
        groups.append(
            dict(
                donor_body=int(donor),
                names=sorted(names),
                initial_N=float(init[mask].sum()),
                endpoint_N=float(final[mask].sum()),
                initial_rows=int(np.sum(init[mask] > 0)),
                endpoint_rows=int(np.sum(final[mask] > 0)),
            )
        )

    def record(i):
        return dict(
            outer_material_node=int(outer[i]),
            donor_body=int(donors[i]),
            nodal_mass_kg=float(nodal[outer[i]]),
            initial_gap_um=float(ref[i] * 1e6),
            endpoint_gap_um=float(end[i] * 1e6),
            initial_N=float(init[i]),
            endpoint_N=float(final[i]),
            gap_change_um=float((end[i] - ref[i]) * 1e6),
        )

    return dict(
        source_launch=sha(case / "launch.json"),
        receipt=sha(case / "terminal.receipt.json"),
        physical_step=receipt["accepted_step"],
        dt_s=dt,
        width_m=width,
        weight_N=weight,
        native=native,
        initial_rows=int((init > 0).sum()),
        endpoint_rows=int((final > 0).sum()),
        initial_barrier_N=float(init.sum()),
        endpoint_barrier_N=float(final.sum()),
        endpoint_barrier_minus_native_N=float(
            final.sum() - native["mean_normal_force_N"]
        ),
        minimum_initial_gap_um=float(ref.min() * 1e6),
        minimum_endpoint_gap_um=float(end.min() * 1e6),
        left_band=int(np.sum((init > 0) & (final == 0))),
        entered_band=int(np.sum((final > 0) & (init == 0))),
        load_by_donor=groups,
        top_four_endpoint_load_fraction=float(np.sort(final)[-4:].sum() / final.sum()),
        largest_force_changes=[record(i) for i in order[:8]],
        **(
            {
                "largest_endpoint_loads": [
                    record(i) for i in np.lexsort((outer, -final))[:4] if final[i] > 0
                ]
            }
            if peak_loads
            else {}
        ),
        scope="The preparation law uses zero closing speed and the inertial barrier stiffness in world metres. The native plane reaction is the accepted root impulse divided by its duration. Actual closing speed, frozen stiffness and the path within the step can affect that comparison.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("case", "material", "transfer"):
        parser.add_argument(key, type=lambda p: Path(p).resolve())
    parser.add_argument(
        "--peak-loads",
        action="store_true",
        help="Also retain the four largest endpoint forces with deterministic node tie breaking.",
    )
    print(json.dumps(measure(**vars(parser.parse_args())), indent=2))
