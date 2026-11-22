"""Seek a registered through-thickness cavity at each failing same-layer tetrahedron."""

import csv, itertools, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from screen_remaining_stars import load_module
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from cavity_material_witness import accepted_node_pin
from audit_cavity_response import elastic
from torso_cell_geometry import geom

FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def oriented_faces(t):
    return [tuple(int(t[i]) for i in p) for p in FACES]


def quality(x, y, z, t, exact):
    if any(not exact.passes(*exact.measurements(p, t)).all() for p in (x, y, z)):
        return False
    X = x[t[:, 1:]] - x[t[:, :1]]
    for cur in (y, z):
        T = np.linalg.solve(X, cur[t[:, 1:]] - cur[t[:, :1]])
        sv = np.linalg.svd(T, compute_uv=False)
        if (
            np.linalg.det(T).min() < 0.4
            or np.linalg.det(T).max() > 2.5
            or sv.min() < 2 / 3
            or sv.max() > 1.5
        ):
            return False
    return True


def screen(lib, run, first, witness, owner, out):
    exact = load_module(lib / "fixed_boundary_volume_flip_198_root.py")
    rx, y, t, tag, pins = inputs(run)
    x = rx.astype(float)
    y = y.astype(float)
    p = first / "physical-accepted.node"
    if sha(p) != accepted_node_pin(first):
        raise ValueError("Unknown deformation")
    z = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    pr, matpin = materials(owner)
    env = json.loads((run / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    m = Path(env[prefix + "COMBINED_NODE_MAP_CSV"])
    if sha(m) != env[prefix + "COMBINED_NODE_MAP_SHA256"]:
        raise ValueError("Unregistered material sheet")
    layers = np.zeros(len(x), dtype=np.uint8)
    with m.open() as f:
        for row in csv.DictReader(f):
            i = int(row["joint_node_id"]) - 1
            k = row["registered_surface_layer"]
            layers[i] |= {"O": 1, "I": 2, "D": 4}.get(k, 0)
    trace = json.loads(witness.read_text())
    w = [q["parents"][0] for q in trace["material_witnesses"] if q["exactly_retained"]]
    by_layer = {}
    for k in (15, 16):
        cells = t[tag == k]
        masks = np.bitwise_or.reduce(layers[cells], axis=1)
        by_layer[k] = {str(v): int(np.sum(masks == v)) for v in np.unique(masks)}
    proposals = []
    excluded = Counter()
    seen = set()
    counts = {}
    for r in w:
        base = t[r]
        mat = tag[r]
        same = layers[base]
        one_sheet = np.all(same == same[0]) and same[0] in (1, 2, 4)
        opposite = (
            {15: {1: 2, 2: 1}, 16: {2: 4, 4: 2}}[int(mat)].get(int(same[0]))
            if one_sheet
            else None
        )
        # A mixed-boundary or interior sliver has no unique opposite sheet.
        # Try existing same-material vertices around its star, keeping every
        # distinct exterior interface and exact oriented boundary.
        if one_sheet and opposite is None:
            continue
        mid = x[base].mean(0)
        rad = 0.025
        local_nodes = np.flatnonzero(np.linalg.norm(x - mid, axis=1) < rad)
        local_nodes = np.union1d(local_nodes, base)
        rows = np.flatnonzero(np.isin(t, local_nodes).any(axis=1))
        face_map = defaultdict(list)
        star = defaultdict(set)
        for i in rows:
            for face in oriented_faces(t[i]):
                face_map[tuple(sorted(face))].append(int(i))
            for node in t[i]:
                star[int(node)].add(int(i))
        incident = set().union(*(star[int(n)] for n in base))
        candidates = set()
        for i in incident:
            if tag[i] == mat:
                candidates.update(
                    int(n) for n in t[i] if opposite is None or layers[n] & opposite
                )
        near = (
            local_nodes
            if opposite is None
            else local_nodes[(layers[local_nodes] & opposite) != 0]
        )
        candidates.update(
            near[np.argsort(np.linalg.norm(x[near] - mid, axis=1))[:30]].tolist()
        )
        candidates.difference_update(base.tolist())
        geom_cache = {}
        nc = 0
        for apex in sorted(candidates):
            keys = {int(r)}
            issue = ""
            final = None
            for iteration in range(16):
                if len(keys) > 24:
                    issue = "grown core exceeds local extent"
                    break
                boundary = {}
                for i in keys:
                    for f in oriented_faces(t[i]):
                        key = tuple(sorted(f))
                        if key in boundary:
                            del boundary[key]
                        else:
                            boundary[key] = f
                bad = []
                out_t = []
                for key, f in boundary.items():
                    if apex in f:
                        continue
                    cell = (apex, *f)
                    h = cell
                    if h not in geom_cache:
                        geom_cache[h] = (
                            quality(x, y, z, np.array([cell]), exact)
                            if geom(x, np.array([cell]))[0][0] > 0
                            else None
                        )
                    # The required side depends on the orientation even for the same geometric face.
                    if not geom_cache[h] or geom(x, np.array([cell]))[0][0] <= 0:
                        bad.append(key)
                    else:
                        out_t.append(cell)
                if bad:
                    more = set()
                    for key in bad:
                        nxt = set(face_map[key]) - keys
                        if len(nxt) != 1 or tag[next(iter(nxt))] != mat:
                            issue = "required growth crosses a material boundary"
                            break
                        more.update(nxt)
                    if issue:
                        break
                    keys.update(more)
                    continue
                if not out_t or apex not in t[list(keys)]:
                    issue = "apex is outside the old patch"
                    break
                oi = np.array(sorted(keys))
                old = t[oi]
                nt = np.array(out_t)
                if set(old.ravel()) != set(nt.ravel()):
                    issue = "would remove a physical node"
                    break
                if {tuple(sorted(c)) for c in old} == {tuple(sorted(c)) for c in nt}:
                    issue = "identity"
                    break
                vs = np.unique(old)
                idx = np.searchsorted(vs, old)
                ni = np.searchsorted(vs, nt)
                ok, why = exact.certify(rx[vs], idx, ni)
                if not ok:
                    issue = why
                    break

                # Any introduced edge/triangle has an endpoint in this local domain. Check it against original incident rows.
                def simplexes(tt, n):
                    return {
                        tuple(sorted(k))
                        for cell in tt
                        for k in itertools.combinations(cell, n)
                    }

                introduced = (simplexes(nt, 2) - simplexes(old, 2)) | (
                    simplexes(nt, 3) - simplexes(old, 3)
                )
                if any(
                    not any(v in local_nodes for v in s)
                    or set.intersection(*(star[int(n)] for n in s)) - keys
                    for s in introduced
                ):
                    issue = "separator conflicts with old outside topology"
                    break
                oldq = float(exact.measurements(rx, old)[1].min())
                newq = float(exact.measurements(rx, nt)[1].min())
                if newq < 1.02 * oldq:
                    issue = "worst conditioning did not improve"
                    break
                sig = (tuple(oi), tuple(sorted(tuple(sorted(c)) for c in nt)))
                if sig in seen:
                    issue = "duplicate"
                    break
                seen.add(sig)
                mu, K = pr[int(mat) - 15]
                scores = []
                for cur in (y, z):
                    E, F, _ = elastic(x, cur, old, vs, mu, K)
                    e, f, J = elastic(x, cur, nt, vs, mu, K)
                    scores.append(
                        dict(
                            energy_relative_error=abs(e - E) / E,
                            nodal_force_l2_change_N=float(np.linalg.norm(f - F)),
                            min_J=float(J.min()),
                            max_J=float(J.max()),
                        )
                    )
                if max(q["energy_relative_error"] for q in scores) > 0.05:
                    issue = "regional energy gate"
                    break
                final = dict(
                    witness=int(r),
                    apex=int(apex),
                    material=int(mat),
                    old=oi.tolist(),
                    new=nt.tolist(),
                    before=oldq,
                    after=newq,
                    matched_pose_response=scores,
                )
                break
            if final:
                proposals.append(final)
                nc += 1
            else:
                excluded[issue or "growth budget"] += 1
        counts[int(r)] = dict(apices=len(candidates), passed=nc)
    out.write_text(
        json.dumps(
            dict(
                source=pins,
                material=matpin,
                registered_node_map_sha256=sha(m),
                source_checkpoint=sha(p),
                role_sheet_mask_codes={"O": 1, "I": 2, "D": 4},
                reference_counts_by_combined_sheet_mask=by_layer,
                candidates=counts,
                excluded=dict(excluded),
                proposals=proposals,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    screen(*(Path(i).resolve() for i in sys.argv[1:]))
