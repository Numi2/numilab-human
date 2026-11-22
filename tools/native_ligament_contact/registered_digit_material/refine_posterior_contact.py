"""Conforming bisection of the registered skin and all material cells incident on each bed-facing edge."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from fit_native_bed_barrier_load import force
from material_context import sha
from registered_material_stars import response
from torso_cell_geometry import geom


class Stars:
    def __init__(self, cells):
        self.rows = {}
        self.parent = {}
        self.incidence = defaultdict(set)
        self.next = 0
        for i, row in enumerate(cells):
            self.add(tuple(map(int, row)), i)

    def add(self, row, ancestor):
        i = self.next
        self.next += 1
        self.rows[i] = row
        self.parent[i] = ancestor
        for n in row:
            self.incidence[n].add(i)

    def users(self, a, b):
        return sorted(self.incidence[a] & self.incidence[b])

    def split(self, a, b, n, users):
        for i in users:
            row = self.rows.pop(i)
            parent = self.parent.pop(i)
            for v in row:
                self.incidence[v].remove(i)
            self.add(tuple(n if v == b else v for v in row), parent)
            self.add(tuple(n if v == a else v for v in row), parent)

    def arrays(self):
        return np.array(list(self.rows.values()), dtype=int), np.array(
            [self.parent[i] for i in self.rows]
        )


def boundary(t):
    oriented = t[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    _, idx, inv, count = np.unique(
        np.sort(oriented, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
        return_inverse=True,
    )
    if np.max(count) > 2:
        raise ValueError("Nonmanifold material cell face")
    inversions = (
        (oriented[:, 0] > oriented[:, 1]).astype(int)
        + (oriented[:, 0] > oriented[:, 2])
        + (oriented[:, 1] > oriented[:, 2])
    )
    winding = np.bincount(inv.ravel(), weights=1 - 2 * (inversions % 2))
    if np.any(winding[count == 2] != 0):
        raise ValueError("Shared material face has inconsistent orientation")
    return oriented[idx[count == 1]]


def keys(faces):
    return {tuple(sorted(r)) for r in faces.tolist()}


def pullback(vector, pairs):
    v = vector.copy()
    for n in reversed(range(len(pairs))):
        a, b = pairs[n]
        mid = len(v) - len(pairs) + n
        v[a] += v[mid] * 0.5
        v[b] += v[mid] * 0.5
    return v[: len(v) - len(pairs)]


def mechanical(states, t, tags, props):
    x, y = states
    E = 0.0
    F = np.zeros_like(y)
    for role in np.unique(tags):
        e, f = response(x, y, t[tags == role], *props[role - 15])
        E += e
        F += f
    return float(E), F


def metrics(data, coords, t, parent, pairs):
    counts = np.bincount(parent, minlength=len(data["t"]))
    if np.any(counts == 0):
        raise ValueError("Lost physical cell")
    x, y, end = coords
    vol, shape = geom(x, t)
    if np.min(vol) <= 0:
        raise ValueError("Nonpositive material reference")
    by_parent = np.bincount(parent, weights=vol, minlength=len(counts))
    mass = data["cell_mass"][parent] * vol / by_parent[parent]
    source_vol = geom(data["x"], data["t"])[0]
    m = np.r_[data["mass"], np.zeros(len(pairs))]
    for k in range(4):
        np.add.at(m, data["t"][:, k], -data["cell_mass"] / 4)
        np.add.at(m, t[:, k], mass / 4)
    outer = np.r_[data["layer"] == 1, np.ones(len(pairs), bool)]
    if np.min(m[outer]) <= 0:
        raise ValueError("Invalid contact mass")
    recovered = np.bincount(parent, weights=mass, minlength=len(counts))
    source_energies = []
    for phase, original, now in (
        ("initial", data["y"], y),
        ("endpoint", data["end"], end),
    ):
        baseline, oldf = mechanical(
            (data["x"], original), data["t"], data["tags"], data["props"]
        )
        e, newf = mechanical((x, now), t, data["tags"][parent], data["props"])
        v, q = geom(now, t)
        a = outer & (now[:, 2] < data["width"])
        if np.min(now[outer, 2]) <= 0:
            raise ValueError("Penetrated plane")
        normal = np.zeros(len(now))
        normal[a] = force(now[a, 2], m[a], data["dt"], data["width"])
        eq = pullback(normal, pairs)
        result = {
            "phase": phase,
            "elastic_energy_J": e,
            "relative_energy_error": abs(baseline - e) / max(abs(baseline), 1e-25),
            "relative_material_force_error": float(
                np.linalg.norm(pullback(newf, pairs) - oldf)
                / max(np.linalg.norm(oldf), 1e-25)
            ),
            "inverted_cells": int(np.sum(v <= 0)),
            "low_quality_cells": int(np.sum(q < 1e-5)),
            "small_cells": int(np.sum(v <= 1e-18)),
            "minimum_quality": float(np.min(q)),
            "minimum_J": float(np.min(v / vol)),
            "maximum_J": float(np.max(v / vol)),
            "contact_nodes": int(a.sum()),
            "normal_N": float(normal.sum()),
            "normal_first_moment_Nm": (normal @ now[:, :2]).tolist(),
            "peak_N": float(np.max(normal)),
            "peak_source_nodal_generalized_N": float(np.max(eq)),
            "contact_pullback_moment_error_Nm": float(
                np.linalg.norm(eq @ original[:, :2] - normal @ now[:, :2])
            ),
            "mass_first_moment_change_kg_m": (
                m @ now - data["mass"] @ original
            ).tolist(),
        }
        source_energies.append(result)
    return {
        "added_nodes": len(pairs),
        "cells": len(t),
        "changed_source_cells": int(np.sum(counts > 1)),
        "max_parent_mass_error_kg": float(np.max(abs(recovered - data["cell_mass"]))),
        "max_source_lumped_mass_error_kg": float(
            np.max(abs(pullback(m, pairs) - data["mass"]))
        ),
        "max_parent_volume_relative_error": float(
            np.max(abs(by_parent - source_vol) / source_vol)
        ),
        "reference_low_quality_cells": int(np.sum(shape < 1e-5)),
        "reference_small_cells": int(np.sum(vol <= 1e-18)),
        "phases": source_energies,
    }, mass


def trial(prepared, out, levels, guard):
    if levels < 0:
        raise ValueError("Negative refinement depth")
    src = json.loads((prepared / "source.json").read_text())
    if sha(prepared / "patch.npz") != src["topology_patch_sha256"]:
        raise ValueError("Different patch")
    with np.load(prepared / "patch.npz") as a:
        data = dict(a)
    out.mkdir(exist_ok=False)
    cells = Stars(data["t"])
    faces = Stars(data["f"])
    skin = Stars(boundary(data["t"]))
    original_boundary = keys(boundary(data["t"]))
    actual_o = keys(data["f"])
    if not actual_o <= original_boundary:
        raise ValueError("Surface face is not on material boundary")
    source_nodes = len(data["x"])
    coords = [list(data[k]) for k in ("x", "y", "end")]
    pairs = []
    history = []
    for level in range(levels + 1):
        t, parent = cells.arrays()
        f, fp = faces.arrays()
        border, _ = skin.arrays()
        if keys(boundary(t)) != keys(border) or not keys(f) <= keys(border):
            raise ValueError("Refinement broke a material or skin boundary")
        full = [np.array(v) for v in coords]
        report, weights = metrics(data, full, t, parent, pairs)
        report["level"] = level
        report["outer_faces"] = len(f)
        if level == 0:
            for row, real in zip(report["phases"], data["native_wrench"], strict=True):
                if not np.allclose(
                    [row["normal_N"], *row["normal_first_moment_Nm"]],
                    real,
                    rtol=1e-11,
                    atol=1e-10,
                ):
                    raise ValueError("Source native force or moment mismatch")
        history.append(report)
        np.savez_compressed(
            out / f"level-{level}.npz",
            local_source_nodes=data["nodes"],
            midpoint_parents=np.asarray(pairs, dtype=int).reshape(-1, 2),
            tetrahedra=t,
            installed_cell_ancestry=data["cell_ids"][parent],
            child_mass_kg=weights,
            source_material=data["tags"][parent],
            outer_triangles=f,
            registered_face_ancestry=data["face_ids"][fp],
            reference=full[0].astype("f4"),
            loaded=full[1].astype("f4"),
        )
        if level == levels:
            break
        focus = f[data["marked"][fp]]
        edges = {
            tuple(sorted(edge))
            for row in focus
            for edge in combinations(map(int, row), 2)
        }
        ordered = sorted(
            edges, key=lambda e: (-np.linalg.norm(full[0][e[0]] - full[0][e[1]]), e)
        )
        skipped = defaultdict(int)
        for a, b in ordered:
            users = cells.users(a, b)
            exterior = faces.users(a, b)
            exposed = skin.users(a, b)
            if not users or len(exposed) != 2 or len(exterior) != 2:
                raise ValueError("Incomplete registered outer edge star")
            n = source_nodes + len(pairs)
            mid = [((q[a] + q[b]) * 0.5).astype("f4").astype(float) for q in coords]
            if guard:
                patch = np.array([cells.rows[i] for i in users])
                sq1 = patch.copy()
                sq1[sq1 == a] = n
                sq2 = patch.copy()
                sq2[sq2 == b] = n
                kids = np.vstack((sq1, sq2))
                ids, local = np.unique(kids, return_inverse=True)
                child = local.reshape(kids.shape)
                reason = None
                for z, new in zip(coords, mid, strict=True):
                    loc = np.array([z[i] if i != n else new for i in ids])
                    v, quality = geom(loc, child)
                    if np.any(v <= 1e-18) or np.any(quality < 1e-5):
                        reason = "cell_volume_or_shape"
                        break
                if reason:
                    skipped[reason] += 1
                    continue
            for z, new in zip(coords, mid, strict=True):
                z.append(new)
            pairs.append((a, b))
            cells.split(a, b, n, users)
            faces.split(a, b, n, exterior)
            skin.split(a, b, n, exposed)
        report["next_level_eligible_edges"] = len(edges)
        report["next_level_skipped"] = dict(skipped)
    meta = {
        "source": src,
        "guarded": guard,
        "levels": history,
        "scope": "Conforming captured material refinement. Each midpoint is an independent material node in the candidate. Forces here are computed on prescribed matched configurations using the installed stationary plane law. No time step was taken.",
    }
    (out / "measurement.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            [
                (
                    h["added_nodes"],
                    [(p["normal_N"], p["low_quality_cells"]) for p in h["phases"]],
                )
                for h in history
            ]
        )
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("prepared", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--levels", type=int, default=2)
    a.add_argument("--guard", action="store_true")
    trial(**vars(a.parse_args()))
