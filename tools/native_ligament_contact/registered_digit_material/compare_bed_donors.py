"""Compare support by registered rigid donor on the two existing native load paths."""

import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from bind_posterior_equilibrium import coordinates
from fit_native_bed_barrier_load import force
from material_context import sha


def compare(compact, fine, case, split, bed, out):
    compact_launch = json.loads((compact / "launch.json").read_text())
    fine_launch = json.loads((fine / "launch.json").read_text())
    rigid = [
        (Path(j["argv"][1]), j["pins"][j["argv"][1]])
        for j in (compact_launch, fine_launch)
    ]
    if len({h for p, h in rigid}) != 1 or any(sha(p) != h for p, h in rigid):
        raise ValueError("Articulated donor numbering differs")
    env = fine_launch["environment"]
    pref = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    path = Path(env[pref + "COMBINED_NODE_MAP_CSV"])
    if sha(path) != env[pref + "COMBINED_NODE_MAP_SHA256"]:
        raise ValueError("Material ancestry differs")
    outer = np.load(split / "physical.npz")["Orows"]
    owner = np.full(len(outer), -1, int)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["registered_surface_layer"] != "O":
                continue
            i = int(row["registered_surface_vertex_row"])
            if int(row["joint_node_id"]) != outer[i] + 1 or owner[i] != -1:
                raise ValueError("Duplicate or changed outer material indexing")
            owner[i] = int(row["donor_body"])
    if np.any(owner < 0):
        raise ValueError("Unmapped physical skin")
    with (compact / "resting-com-momentum-diagnostic.csv").open() as f:
        samples = {
            int(r["accepted_step"])
            for r in csv.DictReader(f)
            if float(r["time_s"]) >= 58
        }
    loads = defaultdict(float)
    with (compact / "resting-com-support-impulses.csv").open() as f:
        for row in csv.DictReader(f):
            if (
                row["row_class"] == "external_support_contact"
                and int(row["accepted_step"]) in samples
            ):
                loads[int(row["body_index"])] += float(row["normal_impulse_ns"]) / (
                    0.002 * len(samples)
                )
    req = json.loads((bed / "load.json").read_text())
    reg = json.loads((case / "supported-registration.json").read_text())
    mf = case / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(mf) != reg["source_mass"]:
        raise ValueError("Mass mismatch")
    mass = np.load(mf)["nodal_mass_kg"]
    dense = {}
    for key in ("reference", "accepted"):
        X = coordinates(fine / ("physical-" + key + ".node"))[outer]
        d = X[:, 2]
        active = np.flatnonzero(d < req["band_m"])
        reaction = force(d[active], mass[outer[active]], req["dt_s"], req["band_m"])
        donors = np.unique(owner)
        dense[key] = {
            int(b): dict(
                bed_N=float(reaction[owner[active] == b].sum()),
                minimum_gap_m=float(d[owner == b].min()),
            )
            for b in donors
        }
    result = {}
    for b in sorted(set(loads) | set(dense["reference"])):
        state = {key: row.get(b) for key, row in dense.items()}
        if not loads[b] and all(v is None or not v["bed_N"] for v in state.values()):
            continue
        result[b] = dict(compact_mean_N=loads[b], dense=state)
    late = sum(loads.values())
    missing = sum(
        loads[b] for b in loads if not dense["reference"].get(b, {}).get("bed_N")
    )
    if (
        abs(sum(v["bed_N"] for v in dense["reference"].values()) - req["weight_N"])
        > 0.01
        or abs(late - req["weight_N"]) > 0.1
    ):
        raise ValueError("Retained support does not balance weight")
    out.write_text(
        json.dumps(
            dict(
                rigid_sha256=rigid[0][1],
                dense_map=sha(path),
                compact=sha(compact / "launch.json"),
                fine=sha(fine / "launch.json"),
                compact_58_to_60_s_mean_N=late,
                compact_load_on_donors_without_initial_dense_bed_contact_N=missing,
                comparison_scope="Loads at the two separately recorded time intervals grouped by common rigid-body index. Distinct contact discretizations and postures. Dense figures are the stationary component of contact.",
                donors=result,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        late,
        missing,
        [
            (
                b,
                round(result[b]["compact_mean_N"], 1),
                round((result[b]["dense"]["reference"] or {}).get("bed_N", 0), 1),
            )
            for b in result
        ],
    )


if __name__ == "__main__":
    compare(*(Path(v).resolve() for v in sys.argv[1:]))
