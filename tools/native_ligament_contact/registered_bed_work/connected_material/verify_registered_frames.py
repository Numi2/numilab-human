"""Certify a registered material edit using the native Float32 source and loaded frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_connected_columns import digest, load_registered_layers
from supported_material_admission import certify, materials, stress_limits


def apply_nodes(canonical, world, ids, edited, rotation, shift, stored):
    if not np.allclose(canonical.astype(float) @ rotation.T + shift, world, rtol=0, atol=1e-7):
        raise ValueError("The source and scene coordinates have different registrations")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(world)) or np.any(ids[:-1]>=ids[1:]):
        raise ValueError("Edit identifiers must be valid, sorted and unique")
    if edited.shape != (len(ids), 3) or edited.dtype != np.float32 or not np.isfinite(edited).all():
        raise ValueError("Invalid emitted scene coordinates")
    result = canonical.copy()
    if stored is not None:
        if stored.dtype != np.float32 or stored.shape != edited.shape or not np.isfinite(stored).all():
            raise ValueError("Invalid canonical coordinates")
        if not np.array_equal((stored.astype(float) @ rotation.T + shift).astype("f4"), edited):
            raise ValueError("Canonical coordinates disagree with the emitted scene")
        result[ids] = stored
    else:
        # Preserve the exact source for unchanged rows across a lossy world-frame round trip.
        changed = np.any(edited != world[ids].astype("f4"), axis=1)
        result[ids[changed]] = (edited[changed].astype(float) - shift) @ rotation
    return result


def verify(registered, dense, patch):
    mesh = load_registered_layers(dense, registered / "pose-one")
    base = registered / "material-source/corrected"
    proof = json.loads((base / "connected-columns.json").read_text())
    if mesh["identities"] != proof["source_identities"] or digest(base / "outer-boundaries.npz") != proof["outer_boundaries"]:
        raise ValueError("The material does not match the recorded bed preparation")
    if not mesh["registration"]["single_rigid_transform"] or not mesh["rigid_loaded"]:
        raise ValueError("The skin has multiple registrations")
    if np.linalg.norm(mesh["reference_rotation"] - mesh["loaded_rotation"]) > 1e-10:
        raise ValueError("A relative material-frame rotation needs an explicit stress transformation")

    launch = dense / "launch.json"
    properties, programs = materials(Path(json.loads(launch.read_text())["argv"][0]).parent)
    limits = stress_limits(programs)
    source, candidate, frames = [], [], {}
    with np.load(patch, allow_pickle=False) as edits, np.load(base / "outer-boundaries.npz", allow_pickle=False) as boundary:
        for name, native, scene in (("reference", "reference", "posed"), ("loaded", "current_source", "loaded")):
            rotation, shift = mesh[name + "_rotation"], mesh[name + "_translation"]
            world = mesh[scene]
            if not np.array_equal(boundary[name + "_xyz"], world[mesh["outer_nodes"]].astype("f4")):
                raise ValueError("The body opening changed")
            original = mesh[native].astype("f4")
            source.append(original)
            candidate.append(apply_nodes(original, world, edits["node"], edits[name], rotation, shift, edits.get("canonical_" + name)))
            frames[name] = dict(rotation=rotation.tolist(), translation_m=shift.tolist())

    results = {}
    for name, states in (("physical_parent", source), ("registered_physical", candidate)):
        try:
            results[name] = certify(*states, mesh["cells"], mesh["material"], properties, limits)
        except (ValueError, np.linalg.LinAlgError) as error:
            results[name] = dict(passed=False, failure=str(error))
    return dict(geometry=digest(patch), source=digest(launch), frames=frames, programs=programs, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("registered", "dense", "patch", "report"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    report = verify(args.registered, args.dense, args.patch)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(s["passed"] for s in report["results"].values()) else 1)
