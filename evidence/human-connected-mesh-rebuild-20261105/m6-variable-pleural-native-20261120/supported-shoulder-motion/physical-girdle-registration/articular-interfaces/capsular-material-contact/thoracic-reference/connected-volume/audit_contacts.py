import json, sys
from pathlib import Path
import numpy as np
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32


def run(capture, points, out):
    z = np.load(capture)
    x = np.load(points)
    install_for_float32()
    rows = []
    for key in z.files:
        if key[0] != "l":
            continue
        layer = int(key[1:])
        f = z[key]
        v = x[f]
        lo = v.min(1)
        hi = v.max(1)
        for sid in sorted(int(n[1:]) for n in z.files if n.startswith("b")):
            y = z[f"b{sid}"]
            g = z[f"f{sid}"]
            mask = np.flatnonzero(np.all((lo <= y.max(0)) & (hi >= y.min(0)), axis=1))
            if not len(mask):
                continue
            # Keep native triangle IDs while avoiding re-indexing an entire
            # distant body surface for each finger.
            uf, inv = np.unique(f[mask], return_inverse=True)
            ans = intersections(y, g, x[uf], inv.reshape(-1, 3))
            if ans:
                rows.append([sid, layer, [[j, int(mask[k])] for j, k in ans]])
    out.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    print(
        {
            layer: sum(len(p) for s, l, p in rows if l == layer)
            for layer in [1, 0xFFFFFD11, 0xFFFFFD12]
        }
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
