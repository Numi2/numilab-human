"""Test exact local tetrahedral reconnectivity including previously remeshed neighbors."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import sys, json
import numpy as np
from enumerate_material_pocket import fillings
from screen_layer_spanning_cavities import quality, oriented_faces
from screen_remaining_stars import load_module
from fit_dorsal_material import inputs
from supported_material_admission import installed_topology
from fit_posterior_equilibrium import materials
from audit_cavity_response import elastic
from material_context import sha


def screen(frame, source, lib):
    trace = json.loads(source.read_text())
    cfg = json.loads((frame / "launch.json").read_text())
    e = cfg["environment"]
    X, Y, base, mat, pins = inputs(frame)
    file = frame / "physical-accepted.node"
    if sha(file) != trace["accepted_geometry_sha256"] or pins != trace["geometry"]:
        raise ValueError("Mismatched material witnesses")
    t, roles = installed_topology(base, mat, e)
    X = X.astype(float)
    Y = Y.astype(float)
    Z = np.loadtxt(file, skiprows=1, usecols=(1, 2, 3))
    props, mpin = materials(Path(cfg["argv"][0]).parent)
    pre = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    labels = Path(e[pre + "CHILD_LABELS"])
    composition = Path(e["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if (
        sha(labels) != e[pre + "CHILD_LABELS_SHA256"]
        or sha(composition) != e["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]
    ):
        raise ValueError("Changed lineage")
    lab = np.loadtxt(
        labels, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3), dtype="i4"
    )
    if not np.array_equal(lab[:, 0], np.arange(len(base))) or not np.array_equal(
        lab[:, 1], mat
    ):
        raise ValueError("Bad source roles")
    with np.load(composition) as npz:
        kept = npz["retained_input_cell_rows"]
        po = npz["replacement_parent_offsets"]
        parents = npz["replacement_parent_cell_rows"]

    # Share only proven homogeneous ancestry. Retained and earlier remeshed cells
    # join the same graph when their physical material and donor identities agree.
    def ancestors(k):
        return (
            [int(kept[k])]
            if k < len(kept)
            else parents[po[k - len(kept)] : po[k - len(kept) + 1]]
        )

    exact = load_module(lib / "fixed_boundary_volume_flip_198_root.py")
    reports = []
    for w in trace["before_failure"]:
        k = w["native_cell"] - trace["global_element_offset"]
        c = t[k]
        if list(c) != w["nodes"]:
            raise ValueError("Mismatched topology")
        # Superset covers every incident cell that can participate in an edit or
        # contain a replacement edge from a tested cavity.
        ids = np.unique(t[np.isin(t, c).any(1)])
        rows = np.flatnonzero(np.isin(t, ids).any(1))
        owners = defaultdict(list)
        stars = defaultdict(set)
        for j in rows:
            for f in oriented_faces(t[j]):
                owners[tuple(sorted(f))].append(int(j))
            for v in t[j]:
                stars[v].add(int(j))
        data = lab[ancestors(k)[0], 1:]
        ok = {int(j) for j in rows if np.all(lab[ancestors(j), 1:] == data)}

        def adj(p):
            return {
                n
                for j in p
                for f in oriented_faces(t[j])
                for n in owners[tuple(sorted(f))]
            } - set(p)

        counts = Counter()
        direct = {}
        for j in adj([k]):
            direct[j] = dict(
                same_material=bool(roles[j] == roles[k]),
                same_donor_and_region=j in ok,
                previous_replacement=j >= len(kept),
            )
        packs = [(k,)]
        best = None
        passing = []
        for size in (2, 3, 4, 5):
            packs = sorted({tuple(sorted((*p, j))) for p in packs for j in adj(p) & ok})
            for p in packs:
                counts["patches"] += 1
                old = t[list(p)]
                ids = np.unique(old)
                loc = np.searchsorted(ids, old)
                if len(ids) > 9:
                    continue
                x, y, z = X[ids], Y[ids], Z[ids]
                banned = tuple(sorted(np.searchsorted(ids, c)))

                def eligible(tet):
                    return tuple(sorted(tet)) != banned and quality(
                        x, y, z, np.asarray([tet]), exact
                    )

                fills, _ = fillings(x, loc, eligible)
                for fill in fills:
                    new = np.asarray(fill)
                    nt = ids[new]
                    counts["geometric"] += 1
                    if not exact.certify(x, loc, new)[0]:
                        continue

                    def simplices(cells, n):
                        return {
                            tuple(sorted(e)) for v in cells for e in combinations(v, n)
                        }

                    added = (simplices(nt, 2) - simplices(old, 2)) | (
                        simplices(nt, 3) - simplices(old, 3)
                    )
                    if any(
                        set.intersection(*(stars[v] for v in e)) - set(p) for e in added
                    ):
                        continue
                    counts["valid_topology"] += 1
                    errs = []
                    for pose in (y, z):
                        E, F, _ = elastic(
                            x, pose, loc, np.arange(len(ids)), *props[roles[k] - 15]
                        )
                        e2, f2, _ = elastic(
                            x, pose, new, np.arange(len(ids)), *props[roles[k] - 15]
                        )
                        errs.append(
                            [
                                float(abs(e2 - E) / max(abs(E), 1e-30)),
                                float(
                                    np.linalg.norm(f2 - F)
                                    / max(np.linalg.norm(F), 1e-30)
                                ),
                            ]
                        )
                    score = max(map(max, errs))
                    if best is None or score < best["worst"]:
                        best = dict(
                            worst=score,
                            relative_energy_force=errs,
                            installed_rows=list(p),
                            replacement=nt.tolist(),
                        )
                    if score <= 0.01:
                        passing.append(
                            dict(
                                relative_energy_force=errs,
                                installed_rows=list(p),
                                replacement=nt.tolist(),
                            )
                        )
        reports.append(
            dict(
                source_cell=w["source_cell"],
                counts=counts,
                direct_neighbors=direct,
                best=best,
                passing=passing,
            )
        )
    return dict(
        source=pins,
        accepted=sha(file),
        previous_cavities=sha(composition),
        materials=mpin,
        results=reports,
    )


if __name__ == "__main__":
    print(json.dumps(screen(*(Path(k) for k in sys.argv[1:])), indent=2))
