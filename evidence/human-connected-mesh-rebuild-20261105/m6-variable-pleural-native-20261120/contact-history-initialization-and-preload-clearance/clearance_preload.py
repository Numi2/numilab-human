"""Admit an offline preload endpoint only after exact skin/fat clearance checks."""
from argparse import ArgumentParser
import json
from pathlib import Path
import time

import numpy as np

from adaptive import prepare
from audit import digest
from run_cavity_load import solve
from verify import verify


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("run", "fit", "balance", "build", "warm", "bones", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("load", nargs="+", type=float)
    args = parser.parse_args()
    if max(args.load) > 1. or min(args.load) <= 0.:
        parser.error("load fractions must be in (0, 1]")
    model = prepare(args.run, args.fit, args.balance, args.build)
    with np.load(args.warm) as state:
        accepted = state["coordinates"]
        warm_load = float(state["load_fraction"])
    args.output.mkdir()
    with (args.output / "inputs.json").open("x") as stream:
        paths = (args.warm, args.fit / "posterior-interface-fit.npz",
                 args.fit / "balanced-mass.npz", args.bones, args.balance)
        json.dump({"inputs": {str(p): digest(p) for p in paths},
                   "load": args.load, "warm_fraction": warm_load}, stream, indent=2)
    # The warm state is evaluated only as a numerical predictor. It is never
    # an admitted endpoint until its independently audited geometry passes.
    trace = []
    for index, load in enumerate(args.load):
        started = time.monotonic()
        candidate, iterations = solve(model, accepted.copy(), 65, weights=(load,),
                                      guard=True, regularize=True)
        row = {"load_fraction": load, "iteration_count": len(iterations), **iterations[-1]}
        if row["outcome"] == "converged":
            _, residual, force, jacobian = model.state(candidate, load)
            assert np.max(np.abs(residual)) < 1.e-4
            assert abs(force.sum() - load * model.body_weight) < 1.e-4
            state_path = args.output / f"trial-{index:02d}.npz"
            np.savez_compressed(state_path, coordinates=candidate, load_fraction=load,
                                residual=residual, contact_force_n=force, determinant=jacobian)
            geometry = verify(args.run, args.fit, args.balance, args.build,
                              state_path, args.bones)
            comparison = geometry["skin_fat_bone_pairs"]
            added = comparison["added_witnesses"]
            passed = (not added and not any(geometry["self_crossing_pairs"].values()) and
                      not geometry["native_volume_guard_violations"] and
                      not geometry["shape_bad_cells"])
            row.update(geometry_clear=passed, added_bone_pairs=len(added),
                       removed_bone_pairs=len(comparison["removed_witnesses"]),
                       exhausted_guard_cells=geometry["native_volume_guard_violations"])
            (args.output / f"trial-{index:02d}-clearance.json").write_text(
                json.dumps(geometry, indent=2, allow_nan=False)+"\n")
            if passed:
                accepted = candidate
        row["elapsed_s"] = time.monotonic() - started
        trace.append(row)
        (args.output / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False)+"\n")
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
