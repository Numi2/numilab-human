"""Screen local, boundary-preserving fat retriangulations against the native witness."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
sys.path.insert(0, str(OLD / "loaded-initialization-023"))
from run_recruitment import digest

PARENT = OLD / "resting-clearance-and-atrium-017/floor-live-parent-capture-002/compiled-parent-cell-mass.csv"
ENDPOINT = OLD / "loaded-initialization-023/same-source-parent-2"
FACES = tuple(combinations(range(4), 3))
EDGES = tuple(combinations(range(4), 2))
ROWS = (26104, 26201)
INVALID = 2**32 - 1


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def determinant(nodes, tets):
    p = nodes[np.asarray(tets, np.intp)]
    return np.linalg.det(p[:, 1:] - p[:, :1])


def typed_faces(tets):
    faces = Counter(tuple(sorted(tet[k] for k in face)) for tet in tets for face in FACES)
    if max(faces.values(), default=0) > 2:
        return None
    return {f for f, frequency in faces.items() if frequency == 1}


def quality(nodes, tets):
    corners = nodes[np.asarray(tets, np.intp)]
    edges = np.stack([corners[:, a] - corners[:, b] for a, b in EDGES], axis=1)
    diameter = np.linalg.norm(edges, axis=-1).max(axis=1)
    return np.abs(determinant(nodes, tets)) / diameter**3


def orient(nodes, tets):
    rows = [list(tet) for tet in tets]
    values = determinant(nodes, rows)
    if np.any(values == 0.):
        return []
    for row, value in zip(rows, values, strict=True):
        if value < 0.:
            row[0], row[1] = row[1], row[0]
    return [tuple(row) for row in rows]


def replacements(parent, owners, support):
    face_owners = defaultdict(list)
    edge_owners = defaultdict(list)
    for row, tet in enumerate(parent):
        for face in FACES:
            face_owners[tuple(sorted(tet[k] for k in face))].append(row)
        for edge in EDGES:
            edge_owners[tuple(sorted(tet[k] for k in edge))].append(row)

    def group_allowed(rows):
        return bool(support.intersection(rows)) and len(set(owners[list(rows)])) == 1

    for face, rows in face_owners.items():
        if len(rows) != 2 or not group_allowed(rows):
            continue
        a, b = (next(node for node in parent[row] if node not in face) for row in rows)
        if tuple(sorted((a, b))) in edge_owners:
            continue
        yield rows, [(a, b, *edge) for edge in combinations(face, 2)]

    for edge, rows in edge_owners.items():
        if len(rows) not in (3, 4) or not group_allowed(rows):
            continue
        link = {tuple(node for node in sorted(parent[row]) if node not in edge) for row in rows}
        vertices = sorted(set().union(*link))
        if len(link) != len(rows) or len(vertices) != len(rows):
            continue
        if len(rows) == 3:
            if tuple(vertices) in face_owners:
                continue
            yield rows, [(*vertices, tip) for tip in edge]
        else:
            for alpha, beta in combinations(vertices, 2):
                if (alpha, beta) in link or (alpha, beta) in edge_owners:
                    continue
                other = [v for v in vertices if v not in (alpha, beta)]
                cycle = [(edge[0], other[0]), (other[0], edge[1]),
                         (edge[1], other[1]), (other[1], edge[0])]
                yield rows, [(alpha, beta, *side) for side in cycle]


def main():
    src_nodes = PARENT.with_name(PARENT.name + ".nodes.csv")
    src_tets = PARENT.with_name(PARENT.name + ".tets.csv")
    endpoint = ENDPOINT / "accepted-parent.csv"
    identity = json.loads((ENDPOINT / "parent-endpoint.json").read_text())
    assert digest(endpoint) == identity["node_table_sha256"]
    nodes, cells, moved = table(src_nodes), table(src_tets), table(endpoint)
    assert all(node["parent_node_index"] == str(i) == current["parent_node_index"]
               and node["compiled_global_node"] == current["compiled_global_node"]
               and node["compiled_lumped_mass_kg"] == current["compiled_lumped_mass_kg"]
               for i, (node, current) in enumerate(zip(nodes, moved, strict=True)))
    assert all(c["parent_tet_row"] == str(i) for i, c in enumerate(cells))
    world = np.array([[float(node[f"current_{axis}_m"]) for axis in "xyz"] for node in nodes])
    reference = np.array([[float(node[f"rest_{axis}_m"]) for axis in "xyz"] for node in nodes])
    accepted = np.array([[float(node[f"current_{axis}_m"]) for axis in "xyz"] for node in moved])
    inherited = [tuple(int(cell[f"parent_node{i}"]) for i in range(4)) for cell in cells]
    owners = np.array([int(c["material_index"]) * (1 << 32) + int(c["donor_body"]) for c in cells])
    material = int(cells[ROWS[0]]["material_index"])
    witness_nodes = set().union(*(inherited[row] for row in ROWS))
    support = {i for i, tet in enumerate(inherited)
               if int(cells[i]["material_index"]) == material and witness_nodes.intersection(tet)}
    retained = {}
    rejected = Counter()
    for old_rows, children in replacements(inherited, owners, support):
        if int(cells[old_rows[0]]["material_index"]) != material:
            continue
        old_tets = [inherited[i] for i in old_rows]
        children = orient(reference, children)
        if not children or typed_faces(children) != typed_faces(old_tets):
            rejected["cavity_boundary"] += 1
            continue
        # Same boundary alone is insufficient for a nonconvex cavity.
        # Positive orientation plus signed-volume closure rejects overlap.
        for p in (reference, world):
            old_volume = np.abs(determinant(p, old_tets)).sum()
            new_det = determinant(p, children)
            if np.min(new_det) <= 0.0 or not np.isclose(new_det.sum(), old_volume, rtol=1.e-6, atol=0.0):
                break
        else:
            old_q, new_q = quality(world, old_tets), quality(world, children)
            old_j = determinant(accepted, old_tets) / determinant(world, old_tets)
            new_j = determinant(accepted, children) / determinant(world, children)
            if new_q.min() <= max(1.e-5, old_q.min()) or np.min(new_j) <= .35 or np.max(new_j) >= 2.3:
                rejected["shape_or_motion"] += 1
                continue
            retained[tuple(old_rows)] = dict(
                rows_removed=old_rows, reversed_orientation=False,
                children=children, original_quality_min=float(old_q.min()),
                proposed_quality_min=float(new_q.min()),
                old_endpoint_determinant_max=float(old_j.max()),
                proposed_endpoint_determinant_max=float(new_j.max()),
                touches_native_failure_witness=bool(set(ROWS).intersection(old_rows)),
                owner_body=int(cells[old_rows[0]]["donor_body"]),
            )
            continue
        rejected["volume_or_orientation"] += 1
    candidates = sorted(retained.values(), key=lambda r: (
        not r["touches_native_failure_witness"], -r["proposed_quality_min"]))
    print(json.dumps(dict(
        parent_tets_sha256=digest(src_tets), parent_nodes_sha256=digest(src_nodes),
        native_endpoint_sha256=digest(endpoint), active_fat_tets=len(support),
        witnesses=list(ROWS), rejections=dict(rejected), candidates=candidates
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
