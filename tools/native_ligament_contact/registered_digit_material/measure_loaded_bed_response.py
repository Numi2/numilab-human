"""Separate the geometric stationary barrier contribution from the measured native dynamic reaction."""

from pathlib import Path
import json, sys
import numpy as np
from fit_native_bed_barrier_load import force
from material_context import sha


def run(native, split, case, baseline):
    report = json.loads((native / "native-outcome.json").read_text())
    a = json.loads((native / "matched-digital-interfaces.json").read_text())
    if not report["bed_reaction"] or report["accepted_steps"] != a["repaired"]["step"]:
        raise ValueError("Unmatched dynamic reaction")
    loaded = json.loads((baseline / "load.json").read_text())
    alloc = json.loads((case / "fine-mass-routed/mass-transfer.json").read_text())
    ledger = case / "fine-mass-routed/current-density-closed-flow.npz"
    if sha(ledger) != alloc["flow_sha256"]:
        raise ValueError("Unmatched source mass")
    mass = np.load(ledger)["nodal_mass_kg"]
    outer = np.load(split / "physical.npz")["Orows"]
    w = loaded["band_m"]
    dt = loaded["dt_s"]

    def coordinates(file):
        with file.open() as f:
            n = int(next(f).split()[0])
            return np.loadtxt(f, max_rows=n, usecols=(1, 2, 3))

    state = {}
    points = []
    for label, file in [
        ("initial", native / "physical-reference.node"),
        ("accepted", native / "physical-accepted.node"),
    ]:
        xyz = coordinates(file)
        gap = xyz[outer, 2]
        ids = outer[gap < w]
        active = gap < w
        if label == "initial" and (
            not np.array_equal(np.sort(loaded["nodes"]), np.sort(ids))
        ):
            raise ValueError("Initial footprint differs")
        if label == "accepted" and sha(file) != a["repaired"]["accepted_material"]:
            raise ValueError("Different accepted tissue")
        if min(gap) <= 0:
            raise ValueError("Ground penetration at sampled endpoint")
        traction = force(gap[active], mass[ids], dt, w)
        ordered = np.sort(traction)[::-1]
        state[label] = dict(
            active=len(ids),
            smallest_gap_m=float(min(gap)),
            stationary_barrier_N=float(traction.sum()),
            top_ten_fraction=float(ordered[:10].sum() / ordered.sum()),
        )
        points.append(xyz)
    change = points[1] - points[0]
    p = points[0][outer, 2] < w
    measured = float(report["bed_reaction"][0]["mean_normal_force_N"])
    weight = loaded["weight_N"]
    result = dict(
        states=state,
        weight_N=weight,
        native_dynamic_reaction_N=measured,
        dynamic_weight_ratio=measured / weight,
        maximum_nodal_travel_m=float(np.linalg.norm(change, axis=1).max()),
        peak_initial_contact_gap_change_m=float(np.max(abs(change[outer[p], 2]))),
        initial_contact_node_median_gap_change_m=float(np.median(change[outer[p], 2])),
        source_mass_sha256=sha(ledger),
        interpretation="The stationary term uses the existing mass and squared-distance barrier at the sampled geometry. The dynamic native reaction includes the physical contact response over the accepted step.",
    )
    (native / "supported-load-response.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    run(*(Path(p).resolve() for p in sys.argv[1:]))
