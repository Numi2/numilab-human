"""Screen bounded boundary refinement before expensive geometric certification."""
from pathlib import Path
from time import monotonic
import json

from igl.copyleft import tetgen
import numpy as np

from audit import digest
from extension import determinant, quality
from test_volume import boundary_of


def screen(path, flags):
    start = monotonic()
    with np.load(path) as mesh:
        nodes, faces = mesh["source_nodes"], mesh["source_faces"]
    result = {
        "input_sha256": digest(path), "flags": flags,
        "input_nodes": len(nodes), "input_faces": len(faces),
    }
    try:
        x, t, *_ = tetgen.tetrahedralize(nodes.astype(float), faces, flags=flags)
    except RuntimeError as failure:
        result["error"] = str(failure)
    else:
        before = determinant(x, t)
        t[before < 0.] = t[before < 0.][:, [0, 2, 1, 3]]
        x = x.astype(np.float32).astype(float)
        det = determinant(x, t)
        q = quality(x, t, det)
        result.update(cells=len(t), nodes=len(x), nonpositive=int((det <= 0.).sum()),
                      poor_count=int((q < 1e-5).sum()), min_quality=float(q.min()),
                      boundary_faces=len(boundary_of(t)))
    result["wall_s"] = monotonic() - start
    return result


def main():
    previous = Path("../sacral-interface-046")
    reports = []
    for bank, partition in ((29, "10"), (30, "11")):
        path = previous / f"partition-{partition}/gluteal-{bank}.npz"
        for flags in ("pq1.5nS1800Q", "pqnS3600Q"):
            report = {"bank": bank, "partition": partition, **screen(path, flags)}
            print(report, flush=True)
            reports.append(report)
    with open("surface-split-screen.json", "x") as output:
        json.dump(reports, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
