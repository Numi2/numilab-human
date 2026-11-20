"""Emit the hash-pinned repaired Float32 geometry for the existing TetGen importer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "resting-clearance-and-atrium-011"
VERIFICATION = HERE.parent / "resting-clearance-and-atrium-014/fine-semantic-cycle-mass-verification.json"
sys.path.insert(0, str(PREVIOUS))
from candidate import load


def array_hash(array, dtype):
    return hashlib.sha256(array.astype(dtype, copy=False).tobytes()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def oriented_boundary(tets, materials):
    # Patches keep the exterior oriented 2-chain of each material. Only cells
    # entering or leaving the candidate can contribute to a topology change.
    count = Counter()
    for tet, material in zip(tets.tolist(), materials.tolist()):
        for a, b, c in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            tri = tet[a], tet[b], tet[c]
            sign = -1 if ((tri[0] > tri[1]) + (tri[0] > tri[2]) + (tri[1] > tri[2])) % 2 else 1
            count[(material, *sorted(tri))] += sign
    return {face: orientation for face, orientation in count.items() if orientation}


def emit(path, points, tets, materials):
    with (path / "mesh.node").open("w") as stream:
        stream.write(f"{len(points)} 3 0 0\n")
        for start in range(0, len(points), 32768):
            chunk = points[start:start+32768]
            np.savetxt(stream, np.column_stack((np.arange(start+1, start+1+len(chunk)), chunk)),
                       fmt=("%d", "%.17g", "%.17g", "%.17g"))
    with (path / "mesh.ele").open("w") as stream:
        stream.write(f"{len(tets)} 4 1\n")
        for start in range(0, len(tets), 32768):
            chunk = tets[start:start+32768]
            np.savetxt(stream, np.column_stack((np.arange(start+1, start+1+len(chunk)), chunk+1,
                        materials[start:start+len(chunk)])), fmt="%d")


def verify_emission(path, points, tets, materials):
    with (path / "mesh.node").open() as stream:
        assert stream.readline().split() == [str(len(points)), "3", "0", "0"]
        for start in range(0, len(points), 32768):
            chunk = points[start:start+32768]
            serialized = np.loadtxt(stream, max_rows=len(chunk))
            assert np.array_equal(serialized[:, 0], np.arange(start+1, start+1+len(chunk)))
            assert np.array_equal(serialized[:,1:], chunk.astype(np.float64))
        assert stream.read() == ""
    with (path / "mesh.ele").open() as stream:
        assert stream.readline().split() == [str(len(tets)), "4", "1"]
        for start in range(0, len(tets), 32768):
            chunk = tets[start:start+32768]
            serialized = np.loadtxt(stream, max_rows=len(chunk), dtype=np.int32)
            assert np.array_equal(serialized[:, 0], np.arange(start+1, start+1+len(chunk)))
            assert np.array_equal(serialized[:,1:5] - 1, chunk)
            assert np.array_equal(serialized[:,5], materials[start:start+len(chunk)])
        assert stream.read() == ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    expected = json.loads(VERIFICATION.read_text())
    old_points, points, tets, materials, origins, replacements, roles, old_tets, old_materials = load()
    with np.load(PREVIOUS / "fine-mass-local-position-delta.npz", allow_pickle=False) as patch:
        points[patch["changed_rows"]] = patch["positions_f32"]
    assert np.isfinite(points).all()
    assert (tets >= 0).all() and tets.max() < len(points)
    assert array_hash(points, "<f4") == expected["positions_sha256"]
    assert array_hash(tets, "<i4") == expected["connectivity_sha256"]

    retained = origins >= 0
    retired = np.setdiff1d(np.arange(len(old_tets)), origins[retained], assume_unique=True)
    assert np.array_equal(tets[retained], old_tets[origins[retained]])
    assert np.array_equal(materials[retained], old_materials[origins[retained]])
    assert np.array_equal(retired, sorted({row for source in replacements.values() for row in source}))
    old_boundary = oriented_boundary(old_tets[retired], old_materials[retired])
    new_boundary = oriented_boundary(tets[~retained], materials[~retained])
    assert old_boundary == new_boundary
    changed = np.any(old_points != points, axis=1)
    active_nodes = np.flatnonzero(changed)
    changed_cells = ~retained | np.any(changed[tets], axis=1)

    if not args.verify_existing:
        args.output.mkdir()
        emit(args.output, points, tets, materials)
    verify_emission(args.output, points, tets, materials)
    published = {
        "node_count": len(points),
        "cell_count": len(tets),
        "nodes_moved": len(active_nodes),
        "nodes_moved_by_role": {str(role): int(np.count_nonzero(changed & (roles == role)))
                                for role in np.unique(roles)},
        "cells_with_geometry_changes": int(np.count_nonzero(changed_cells)),
        "new_cell_rows": np.flatnonzero(~retained).tolist(),
        "retired_cell_rows": retired.tolist(),
        "material_boundary_chain_equal": True,
        "float32_array_sha256": expected["positions_sha256"],
        "tetrahedron_array_sha256": expected["connectivity_sha256"],
        "mesh_node_sha256": file_hash(args.output / "mesh.node"),
        "mesh_ele_sha256": file_hash(args.output / "mesh.ele"),
        "source_quality_verification_sha256": file_hash(VERIFICATION),
        "scope": "native TetGen node and element geometry only; no label, flow, or runtime admission",
    }
    (args.output / "geometry-receipt.json").write_text(json.dumps(published, indent=2) + "\n")
    print(json.dumps({key: value for key, value in published.items()
                      if key not in ("new_cell_rows", "retired_cell_rows")}, indent=2))


if __name__ == "__main__":
    main()
