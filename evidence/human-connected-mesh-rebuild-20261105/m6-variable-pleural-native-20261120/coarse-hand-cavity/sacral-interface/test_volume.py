"""Check TetGen's fixed-boundary volume against the compiled Float32 quality gates."""
from argparse import ArgumentParser
import json
from pathlib import Path

from igl.copyleft import tetgen
import numpy as np

from audit import digest
from extension import determinant, quality
from improve_cavity_quality import improve
from improve_interior_quality import optimize
from partition_gluteal import surface_volume


def boundary_of(cells):
    faces = cells[:, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]].reshape(-1, 3)
    faces, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    assert counts.max() <= 2
    return {tuple(f) for f in faces[counts == 1]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--polish-interior", action="store_true")
    parser.add_argument("--polish-cavities", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    args_mesh = args.mesh
    with np.load(args_mesh) as src:
        points = src["source_nodes"]
        faces = src["source_faces"]
        vertex_source = src["source_vertex_ids"]
        face_source = src["source_face_parents"]

    # The existing gluteal builder preserves authored facet boundaries.
    flags = "pYq1.5nQ" + ("M" if args.no_merge else "")
    print("tetrahedralizing", args_mesh, flags, len(points), len(faces), flush=True)
    nodes, cells, *_ = tetgen.tetrahedralize(points.astype(float), faces, flags=flags)
    assert len(cells) and np.isfinite(nodes).all()
    exact = determinant(nodes, cells)
    flip = exact < 0
    if flip.any():
        cells[flip] = cells[flip][:, [0, 2, 1, 3]]
    nodes = nodes.astype(np.float32)
    node_lookup = {tuple(point): i for i, point in enumerate(nodes)}
    assert len(node_lookup) == len(nodes)
    faces_mapped = np.array([[node_lookup[tuple(points[i])] for i in face] for face in faces])
    actual_boundary = boundary_of(cells)
    expected_boundary = {tuple(sorted(f)) for f in faces_mapped}
    missing = expected_boundary - actual_boundary
    added = actual_boundary - expected_boundary
    boundary_equal = not missing and not added
    fixed = {v for face in actual_boundary | expected_boundary for v in face}
    polish = None
    if args.polish_interior:
        nodes, polish = optimize(nodes, cells, fixed, 1e-5)
    cavities = None
    if args.polish_cavities:
        cells, cavities = improve(nodes, cells, 1e-5)
        assert actual_boundary == boundary_of(cells)
    actual = determinant(nodes.astype(float), cells)
    q = quality(nodes.astype(float), cells, actual)
    volume = surface_volume(points.astype(float), faces)
    relative_error = abs(actual.sum() / (6. * volume) - 1.)
    result = dict(cells=len(cells), nodes=len(nodes), emitted_boundary_matches=boundary_equal,
                  nonpositive=int((actual <= 0.).sum()), low_quality=int((q < 1e-5).sum()),
                  minimum_normalized_determinant=float(q.min()),
                  missing_boundary_faces=len(missing), added_boundary_faces=len(added),
                  minimum_cell_edge_m=float(min(np.linalg.norm(
                      nodes[cells[:, a]].astype(float) - nodes[cells[:, b]], axis=1).min()
                      for a in range(4) for b in range(a+1, 4))),
                  volume_error_relative=float(relative_error),
                  source_sha256=digest(args_mesh), tetgen_flags=flags)
    bad = (actual <= 0.) | (q < 1e-5)
    result["low_quality_all_on_boundary"] = sum(
        all(int(v) in fixed for v in tet) for tet in cells[bad])
    if polish is not None:
        result["interior_optimization"] = polish
    if cavities is not None:
        result["cavity_optimization"] = cavities
    if bad.any():
        np.savez_compressed(args.output / "unqualified-witnesses.npz",
                            indices=np.flatnonzero(bad), corners=nodes[cells[bad]],
                            normalized_determinants=q[bad], determinants=actual[bad])
    if boundary_equal and not result["nonpositive"] and not result["low_quality"] and relative_error <= 1e-6:
        np.savez_compressed(args.output / "volume.npz", nodes=nodes, cells=cells,
                            source_nodes=points, source_faces=faces,
                            source_vertex_ids=vertex_source, source_face_parents=face_source)
        result["qualified_reference_volume_sha256"] = digest(args.output / "volume.npz")
    with (args.output / "assessment.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
