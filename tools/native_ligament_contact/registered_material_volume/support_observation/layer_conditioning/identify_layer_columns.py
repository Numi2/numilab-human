"""Locate paired skin interfaces and source elements for a through-thickness repair of rejected cells."""

from pathlib import Path
import json
import sys
import numpy as np
import igl
from compile_connected_joint import digest


def inspect(case, witness_path, out):
    c = json.loads((case / "launch.json").read_bytes())
    root = Path(
        c["environment"][
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"
        ]
    ).parent
    x = np.loadtxt(root / "mesh.node", skiprows=1, usecols=(1, 2, 3)) * 1000
    labels = np.loadtxt(
        root / "current-child.labels.csv", delimiter=",", skiprows=1, dtype="i8"
    )
    raw = np.loadtxt(root / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype="i4")
    t = raw[:, :4] - 1
    with np.load(root / "material-walls.npz") as z:
        walls = {k: np.ascontiguousarray(z[k], dtype="i8") for k in ["I", "O"]}
    trees = {}
    for k, faces in walls.items():
        b = igl.AABB()
        b.init(x, faces)
        trees[k] = b
    result = []
    # A local broad phase followed by barycentric clipping gives the exact
    # reference line intervals within each tetrahedron.
    lo = x[t[:, 0]].copy()
    hi = lo.copy()
    for j in range(1, 4):
        lo = np.minimum(lo, x[t[:, j]])
        hi = np.maximum(hi, x[t[:, j]])
    for q in json.loads(witness_path.read_bytes()):
        wids = t[q["row"]]
        layer = next(f["walls"][0] for f in q["faces"] if f["walls"])
        other = "I" if layer == "O" else "O"
        # Each old boundary triangle has an opposed skin-side apex in the failed tetrahedron.
        cc = []
        material_ids = set()
        for j in [v for v in q["faces"] if v["walls"]]:
            fa = j["face"]
            p = x[fa].mean(0)
            normal = np.cross(x[fa[1]] - x[fa[0]], x[fa[2]] - x[fa[0]])
            normal /= np.linalg.norm(normal)
            fourth = next(i for i in wids if i not in fa)
            if normal @ (x[fourth] - p) < 0:
                normal = -normal
            b = trees[other]
            tri = walls[other]
            _, _, nearest = b.squared_distance(x, tri, np.ascontiguousarray(p[None]))
            gap = nearest[0] - p
            if np.linalg.norm(gap) < 1e-9:
                raise ValueError("Coincident outer and inner skin")
            n = normal.copy()
            # Search the skin-facing hemisphere. A plausible opposing wall must
            # be the first boundary encountered along the entire chord.
            axis = np.eye(3)[np.abs(n).argmin()]
            u = np.cross(n, axis)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            z = (np.arange(192) + 0.5) / 192
            phi = np.arange(192) * np.pi * (3.0 - np.sqrt(5.0))
            rays = z[:, None] * n + np.sqrt(1.0 - z * z)[:, None] * (
                np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v
            )
            origins = np.ascontiguousarray(np.broadcast_to(p, rays.shape))
            rays = np.ascontiguousarray(rays)
            opp = b.intersect_ray(x, tri, origins, rays)
            same = trees[layer].intersect_ray(x, walls[layer], origins, rays)
            choices = []
            for ri, (end, same_side) in enumerate(zip(opp, same)):
                d = min((h[1] for h in end if h[1] > 1e-5), default=float("inf"))
                back = min(
                    (h[1] for h in same_side if h[1] > 1e-4), default=float("inf")
                )
                if np.isfinite(d) and d < back - 1e-5:
                    choices.append((d, ri))
            probes = [
                ("material_normal", n),
                ("nearest_opposing", gap / np.linalg.norm(gap)),
            ]
            probes.extend(
                (f"visible_opposing_{rank}", rays[idx])
                for rank, (dist, idx) in enumerate(sorted(choices)[:3])
            )
            for mode, normal in probes:
                hits = b.intersect_ray(
                    x,
                    tri,
                    np.ascontiguousarray(p[None]),
                    np.ascontiguousarray(normal[None]),
                )[0]
                hits = sorted(
                    (float(distance), int(face), float(u), float(v))
                    for face, distance, u, v in hits
                    if distance > 0
                )
                if not hits:
                    cc.append(
                        dict(
                            start_face=fa,
                            probe=mode,
                            valid=False,
                            reason="normal misses opposing anatomical wall",
                        )
                    )
                    continue
                L, face, b1, b2 = hits[0]
                delta = L * normal
                stop = p + delta
                mask = (lo <= np.maximum(p, stop) + 1e-7).all(1) & (
                    hi >= np.minimum(p, stop) - 1e-7
                ).all(1)
                ids = np.flatnonzero(mask)
                tet = x[t[ids]]
                M = (tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1)
                y = np.linalg.solve(
                    M,
                    np.broadcast_to(p, (len(tet), 3))[:, :, None] - tet[:, 0, :, None],
                )[:, :, 0]
                dy = np.linalg.solve(
                    M, np.broadcast_to(delta, (len(tet), 3))[:, :, None]
                )[:, :, 0]
                weights = np.column_stack([1.0 - y.sum(1), y])
                slope = np.column_stack([-dy.sum(1), dy])
                lower = np.zeros(len(tet))
                upper = np.ones(len(tet))
                for v, s in zip(weights.T, slope.T):
                    nonzero = np.abs(s) > 1e-12
                    cross = np.divide(-v, s, out=np.zeros_like(v), where=nonzero)
                    lower = np.where(s > 1e-12, np.maximum(lower, cross), lower)
                    upper = np.where(s < -1e-12, np.minimum(upper, cross), upper)
                    upper = np.where(~nonzero & (v < -1e-8), -1.0, upper)
                good = upper - lower > 1e-10
                ordered = sorted(
                    (float(a), float(b), int(i))
                    for a, b, i in zip(lower[good], upper[good], ids[good])
                )
                skin = [(a, b, i) for a, b, i in ordered if raw[i, -1] == 15]
                coverage = sum(b - a for a, b, i in skin)
                # Check holes/overlaps directly, not just cancellation of interval lengths.
                err = (
                    max(
                        [abs(skin[0][0]), abs(skin[-1][1] - 1.0)]
                        + [abs(q1[1] - q2[0]) for q1, q2 in zip(skin, skin[1:])]
                    )
                    if skin
                    else 1.0
                )
                target = np.array([1.0 - b1 - b2, b1, b2]) @ x[tri[face]]
                good = err < 1e-6 and np.linalg.norm(target - stop) < 1e-4
                if good:
                    material_ids.update(i for a, b, i in skin)
                cc.append(
                    dict(
                        start_face=fa,
                        probe=mode,
                        normal_cosine=float(n @ normal),
                        target_face_id=face,
                        target_vertices=tri[face].tolist(),
                        target_barycentric=[1.0 - b1 - b2, b1, b2],
                        target_error_mm=np.linalg.norm(target - stop),
                        thickness_mm=L,
                        skin_coverage=coverage,
                        max_interval_error=err,
                        valid=bool(good),
                        crossed_skin_cells=[i for a, b, i in skin],
                        crossed_regions=sorted({int(labels[i, 3]) for a, b, i in skin}),
                    )
                )
        result.append(
            dict(
                failing_cell=q["row"],
                starting_wall=layer,
                opposing_wall=other,
                probes=cc,
                material_corridor=sorted(material_ids),
            )
        )
    out.write_text(
        json.dumps(
            dict(
                launch=digest(case / "launch.json"),
                inputs={
                    f: digest(root / f)
                    for f in [
                        "mesh.node",
                        "mesh.ele",
                        "material-walls.npz",
                        "current-child.labels.csv",
                    ]
                },
                rows=result,
                scope="Compiler-side material traversal. A valid corridor intersects only contiguous skin from one registered interface to the other. The listed cells and barycentric parents identify a possible joint remesh and preserve the native material labels. No runtime cells have been replaced.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    inspect(*(Path(x).resolve() for x in sys.argv[1:]))
