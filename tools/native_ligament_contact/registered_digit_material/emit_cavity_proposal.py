"""Emit a checked feasible mass route into the existing Human fixed-cavity NPZ."""

import json, sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from material_context import sha
from registered_density import density_bounds


def emit(run, fit, ledger, out):
    config = json.loads((run / "launch.json").read_text())
    fitted = json.loads(fit.read_text())
    ref, installed, tags4, tag, pins = inputs(run)
    if fitted["source"] != pins or fitted["ledger_sha256"] != sha(ledger):
        raise ValueError("Changed input")
    solution = fitted["proposal"]
    old = np.array(solution["retired_rows"])
    t = np.array(solution["replacement_tets"])
    mass = np.array(solution["mass_kg"])
    prior = np.load(ledger)["element_mass_kg"][old]
    volume = geom(ref.astype(float), t)[0]
    if (
        len(set(old)) != len(old)
        or len(t) != len(mass)
        or np.any(tag[old] != tag[old[0]])
    ):
        raise ValueError("Invalid ancestry")
    env = config["environment"]
    label_path = Path(
        env["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS"]
    )
    if (
        sha(label_path)
        != env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS_SHA256"
        ]
    ):
        raise ValueError("Changed registered ancestry")
    labels = np.loadtxt(
        label_path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3), dtype=np.int32
    )
    if not np.array_equal(labels[old, 0], old) or not np.all(
        labels[old, 1:] == labels[old[0], 1:]
    ):
        raise ValueError("A proposed cavity crosses a native material, donor or region")
    lo, hi = density_bounds(Path(config["argv"][0]).parent, tag[old[0]])
    if (
        not np.all(np.isfinite(mass))
        or np.min(mass / volume) < lo - 1e-6
        or np.max(mass / volume) > hi + 1e-6
    ):
        raise ValueError("Material density")
    if abs(prior.sum() - mass.sum()) > prior.sum() * 1e-10:
        raise ValueError("Mass closure")
    for points in (ref, installed):
        points = points.astype(float)
        points -= points[np.unique(t)].mean(0)
        moment = mass @ points[t].mean(1) - prior @ points[tags4[old]].mean(1)
        if np.max(abs(moment)) > 1e-10 * prior.sum() * np.max(abs(points[t])):
            raise ValueError("First moment")
    # Preserve the ordering of all retained cells exactly as the loader requires.
    retained = np.flatnonzero(~np.isin(np.arange(len(tag)), old)).astype("<u4")
    order = np.sort(old).astype("<u4")
    np.savez_compressed(
        out,
        retained_input_cell_rows=retained,
        replacement_tetrahedra=t.astype("<u4"),
        replacement_tet_offsets=np.array([0, len(t)], dtype="<u4"),
        replacement_parent_offsets=np.arange(len(t) + 1, dtype="<u4") * len(old),
        replacement_parent_cell_rows=np.tile(order, len(t)),
        replacement_mass_kg=mass.astype("<f8"),
        source_volume_sha256=np.array(
            config["environment"][
                "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CANDIDATE_VOLUME_SHA256"
            ],
            dtype="<U64",
        ),
    )


if __name__ == "__main__":
    emit(*(Path(x).resolve() for x in sys.argv[1:]))
