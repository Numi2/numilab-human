"""Run the native cavity predicates on an emitted TetGen mesh and a registered NPZ proposal."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def check(nodes, elements, proposal, executable):
    coordinates = np.loadtxt(nodes, skiprows=1, usecols=(0, 1, 2, 3))
    if not np.array_equal(coordinates[:, 0], np.arange(1, len(coordinates) + 1)):
        raise ValueError("TetGen node numbering must match the registered volume")
    original = np.loadtxt(elements, dtype=np.int64, skiprows=1, usecols=range(5))
    if not np.array_equal(original[:, 0], np.arange(1, len(original) + 1)):
        raise ValueError("TetGen element numbering must match the registered volume")
    cells = original[:, 1:] - 1
    reports = []
    with np.load(proposal, allow_pickle=False) as z, tempfile.TemporaryDirectory() as tmp:
        kids = z["replacement_tetrahedra"]
        children = z["replacement_tet_offsets"].astype(np.int64)
        offsets = z["replacement_parent_offsets"].astype(np.int64)
        parents = z["replacement_parent_cell_rows"]
        if (len(children) < 2 or children[0] != 0 or children[-1] != len(kids)
                or np.any(np.diff(children) < 2) or np.any(np.diff(children) > 96)
                or len(offsets) != len(kids) + 1 or offsets[0] != 0
                or offsets[-1] != len(parents) or np.any(np.diff(offsets) < 2)
                or np.any(np.diff(offsets) > 96)
                or parents.min() < 0 or parents.max() >= len(cells)):
            raise ValueError("Invalid cavity parent or replacement ranges")
        path = Path(tmp) / "reference.txt"
        for patch, (start, end) in enumerate(zip(children[:-1], children[1:])):
            old_rows = parents[offsets[start]:offsets[start + 1]]
            if len(np.unique(old_rows)) != len(old_rows) or any(
                not np.array_equal(parents[offsets[j]:offsets[j + 1]], old_rows)
                for j in range(start + 1, end)
            ):
                raise ValueError("Replacement cells do not share the declared cavity parents")
            old = cells[old_rows]
            new = kids[start:end]
            local_nodes = np.union1d(old.ravel(), new.ravel())
            if local_nodes[0] < 0 or local_nodes[-1] >= len(coordinates):
                raise ValueError("Cell escapes source node domain")
            with path.open("w") as stream:
                print(len(local_nodes), len(old), len(new), file=stream)
                np.savetxt(stream, coordinates[local_nodes, 1:], fmt="%.17g")
                np.savetxt(stream, np.searchsorted(local_nodes, old), fmt="%d")
                np.savetxt(stream, np.searchsorted(local_nodes, new), fmt="%d")
            run = subprocess.run([str(executable.resolve()), str(path)],
                                 capture_output=True, text=True)
            reports.append(dict(patch=patch, returned=run.returncode,
                                detail=(run.stdout + run.stderr).strip()))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("nodes", "elements", "proposal", "executable"):
        parser.add_argument("--" + arg, required=True, type=Path)
    results = check(**vars(parser.parse_args()))
    print(json.dumps(results, indent=2))
    return int(any(r["returned"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
