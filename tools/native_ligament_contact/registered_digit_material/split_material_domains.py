"""Separate connected edits at their exact unchanged material boundaries."""

import csv, runpy
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from connected_elbow import make_graph


def domains(s, t, cases):
    ids, raw_faces = np.unique(s["inds"], return_inverse=True)
    f = raw_faces.reshape(-1, 3)
    key, back = np.unique(s["verts"][ids, :3], axis=0, return_inverse=True)
    back = back.ravel()
    changed = np.any(s["weights"][ids] != t["weights"][ids], axis=1)
    active = np.unique(back[changed])
    graph = make_graph(key, back[f])
    n, component = connected_components(graph[active, :][:, active], directed=False)
    tag = np.full(len(key), -1)
    tag[active] = component
    part = tag[back]
    bed = set()
    for case in cases:
        for file, field in [
            ("resting-local-bed-footprint.csv", "vertex"),
            ("resting-support-selected-skin-witness.csv", "selected_skin_vertex_local"),
        ]:
            with (case / file).open() as stream:
                for row in csv.DictReader(stream):
                    bed.add(int(row[field]))
    contact = np.isin(ids, list(bed))
    incident = np.unique(f[contact[f].any(axis=1)])
    return ids, part, changed, contact, incident


if __name__ == "__main__":
    import sys
    import json

    decoder, base, new, *cases = map(Path, sys.argv[1:])
    a = json.loads((base / "launch.json").read_text())["argv"]
    dec = runpy.run_path(str(decoder))["decode"]
    s, t = [dec(p.read_bytes()) for p in (Path(a[a.index("--skin-payload") + 1]), new)]
    idx, d, changed, bed, ring = domains(s, t, cases)
    for i in sorted(set(d) - {-1}):
        mask = (d == i) & changed
        if mask.sum() > 5:
            print(i, mask.sum(), (mask & bed).sum(), mask[ring].sum())
