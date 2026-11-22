"""Query the physical skin/fat and cortical surfaces around both digital extensor routes."""

import json
from pathlib import Path
import numpy as np
from cortical_distance import CorticalSurface


def complete_remote_loops(vertices, faces, hand_points):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, ids, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True
    )
    if np.max(count) > 2:
        raise ValueError("Nonmanifold tissue boundary")
    rim = edges[ids[count == 1]]
    if not len(rim):
        return vertices, faces, None
    # The authored open edges are head openings. Auxiliary faces only disambiguate
    # sign far from the hand. The physical triangles and the physical asset stay intact.
    interval = vertices[rim.ravel(), 1]
    separation = max(
        hand_points[:, 1].min() - interval.max(),
        interval.min() - hand_points[:, 1].max(),
    )
    if separation < 0.5:
        raise ValueError("Query patch too close to a true material edge")
    following = {int(a): int(b) for a, b in rim}
    if len(following) != len(rim) or set(following) != set(following.values()):
        raise ValueError("Open rim has no unique orientation")
    centers = []
    caps = []
    while following:
        first = next(iter(following))
        v = first
        loop = []
        while True:
            loop.append(v)
            v = following.pop(v)
            if v == first:
                break
        base = len(vertices) + len(centers)
        centers.append(vertices[loop].mean(0))
        caps.extend((b, a, base) for a, b in zip(loop, loop[1:] + loop[:1]))
    return np.vstack((vertices, centers)), np.vstack((faces, caps)), float(separation)


class DigitalClearance:
    def __init__(self, root, lib, field=None):
        self.meta = json.loads((root / "geometry.json").read_text())
        self.mesh = np.load(root / "physical.npz")
        delta = None
        if field is not None:
            patch = np.load(field)
            delta = np.zeros(max(int(self.mesh[r + "rows"].max()) for r in "OID") + 1)
            i = patch["nodes"]
            valid = i < len(delta)
            delta[i[valid]] = patch["delta_x_m"][valid]
        self.surfaces = {}
        self.boxes = {}
        self.head_separation = {}
        all_sites = np.array(
            [
                s["world_m"]
                for r in self.meta["digits"].values()
                for s in r["route"][-6:]
            ]
        )
        for key in ["O", "I", "D", *("b" + i for i in self.meta["bone_names"])]:
            x = self.mesh[key + "v"]
            f = self.mesh[key + "f"]
            if key in ("O", "I", "D"):
                if delta is not None:
                    x = x.copy()
                    x[:, 0] += delta[self.mesh[key + "rows"]]
                x, f, self.head_separation[key] = complete_remote_loops(x, f, all_sites)
            self.surfaces[key] = CorticalSurface(lib, x, f)
            self.boxes[key] = np.array([x.min(0), x.max(0)])

    def query(self, points, bones=None, layers=("O", "I", "D")):
        pts = np.asarray(points).reshape(-1, 3)
        if not len(pts) or not np.isfinite(pts).all():
            raise ValueError("Invalid route coordinates")
        out = {k: -self.surfaces[k].query(pts)[0] for k in layers}
        low = pts.min(0)
        high = pts.max(0)
        distances = np.full(len(pts), np.inf)
        ids = np.full(len(pts), -1)
        for k, box in self.boxes.items():
            if not k.startswith("b") or (bones is not None and int(k[1:]) not in bones):
                continue
            # Distant bones are irrelevant to a sub-centimetre clearance bound.
            if np.any(box[0] > high + 0.025) or np.any(box[1] < low - 0.025):
                continue
            q = self.surfaces[k].query(pts)[0]
            closer = q < distances
            ids[closer] = int(k[1:])
            distances = np.minimum(q, distances)
        out["bone"] = distances
        out["nearest_bone"] = ids
        return out


if __name__ == "__main__":
    import sys

    root, lib = map(Path, sys.argv[1:])
    z = DigitalClearance(root, lib)
    report = {}
    for name, r in z.meta["digits"].items():
        pts = np.array(
            [s["world_m"] for s in r["route"][-6:]] + [r["trial_enthesis"]["world_m"]]
        )
        ds = z.query(pts)
        report[name] = {
            k: (v if k == "nearest_bone" else 1000 * v).tolist() for k, v in ds.items()
        }
        print(
            name,
            "last 2 sites deep-fat clearance mm",
            np.round(report[name]["D"][-3:-1], 3),
            "inferred enthesis",
            round(report[name]["D"][-1], 3),
        )
    (root / "site-clearance.json").write_text(
        json.dumps(
            dict(site_results=report, head_opening_y_separation_m=z.head_separation),
            indent=2,
        )
        + "\n"
    )
