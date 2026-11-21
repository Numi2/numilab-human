"""Interpolate a feasible endpoint against the same compiled mass reference."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from audit import rows


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("nodes", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("scale", type=float)
    args = parser.parse_args()
    assert 0. < args.scale < 1.
    before = rows(args.nodes)
    with np.load(args.geometry) as data:
        reference = data["nodes_reference"]
        current = data["nodes_current"]
        cells = data["cells"]
    original = [np.array([[float(r[f"{mode}_{a}_m"]) for a in "xyz"] for r in before])
                for mode in ("current", "rest")]
    adjusted = [values + args.scale * (fit - values)
                for values, fit in zip(original, (current, reference), strict=True)]
    args.output.mkdir()
    np.savez_compressed(args.output / "posterior-interface-fit.npz",
                        nodes_current=adjusted[0].astype(np.float32),
                        nodes_reference=adjusted[1], cells=cells)


if __name__ == "__main__":
    main()
