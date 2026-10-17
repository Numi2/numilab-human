"""Tests for exact coincident forearm interface-birth topology repair."""
import numpy as np
import pytest

from numilab_human.forearm_topology_patch import (
    check_closed_oriented_vertex_links,
    repair_duplicate_interface_birth_faces,
)


def _joined_tetrahedra_with_duplicate_interface_vertex():
    # Two tetrahedral shells share the ABC interface with opposite winding.
    # C and C_alias have byte-identical positions but distinct material GIDs.
    points = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 0], [0, 0, -1],
    ], dtype="<f4")
    gids = np.asarray([100, 101, 102, 103, 202, 105], dtype="<u8")
    faces = np.asarray([
        [0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0],
        [0, 4, 1], [0, 5, 4], [4, 5, 1], [1, 5, 0],
    ], dtype="<u4")
    return points, gids, faces


def test_duplicate_birth_rewire_preserves_rows_and_cancels_opposite_interface():
    points, gids, faces = _joined_tetrahedra_with_duplicate_interface_vertex()
    point_bytes, gid_bytes = points.tobytes(), gids.tobytes()
    source_weights = np.arange(6 * 21, dtype="<f8").reshape(6, 21)
    weight_bytes = source_weights.tobytes()
    output, report = repair_duplicate_interface_birth_faces(
        faces, points, gids,
        [{"keep_gid": 102, "retire_gid": 202, "face_indices": [0, 4]}],
    )
    assert output.shape == (6, 3)
    assert report["all_source_vertex_arrays_unchanged"]
    assert report["both_lineage_records_retained"]
    assert report["removed_face_indices"] == [0, 4]
    assert report["old_to_new_face_ordinal_map"].tolist() == [-1, 0, 1, 2, -1, 3, 4, 5]
    assert report["indexed_topology"]["closed_oriented_indexed_manifold"]
    assert points.tobytes() == point_bytes
    assert gids.tobytes() == gid_bytes
    assert source_weights.tobytes() == weight_bytes


def test_duplicate_birth_repair_rejects_nonidentical_position_or_same_winding():
    points, gids, faces = _joined_tetrahedra_with_duplicate_interface_vertex()
    changed = points.copy()
    changed[4, 0] = np.nextafter(changed[4, 0], np.float32(2))
    with pytest.raises(ValueError, match="not byte-identical"):
        repair_duplicate_interface_birth_faces(
            faces, changed, gids,
            [{"keep_gid": 102, "retire_gid": 202, "face_indices": [0, 4]}],
        )
    same_winding = faces.copy()
    same_winding[4] = np.asarray([0, 1, 4], dtype="<u4")
    with pytest.raises(ValueError, match="oppositely wound"):
        repair_duplicate_interface_birth_faces(
            same_winding, points, gids,
            [{"keep_gid": 102, "retire_gid": 202, "face_indices": [0, 4]}],
        )


def test_link_checker_rejects_disconnected_oriented_links():
    # Two closed shells sharing only one indexed vertex have a pinched link.
    faces = np.asarray([
        [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
        [0, 4, 5], [0, 5, 6], [5, 4, 6], [4, 0, 6],
    ], dtype="<u4")
    with pytest.raises(ValueError, match="closed oriented"):
        check_closed_oriented_vertex_links(faces, 7)


def test_shared_precheck_ignores_unused_slots_but_prepare_surface_rejects_defects():
    import os
    from pathlib import Path
    import subprocess
    import sys

    repo = Path("/Users/n/numi-human-forearm-parent-bridge-20261017")
    shared = Path("/Users/n/numi-human-aortic-port-proof-consumer-20261016")
    if not (repo / "src/numilab_human/forearm_topology_patch.py").is_file() or not (
            shared / "src/numilab_human/resting_anatomy_verify.py").is_file():
        pytest.skip("Numi shared topology owner is unavailable")

    smoke = r"""
import sys
import numpy as np
from pathlib import Path
repo = Path(sys.argv[1])
shared = Path(sys.argv[2])
sys.path.insert(0, str(shared / "src"))
import numilab_human
numilab_human.__path__.append(str(repo / "src/numilab_human"))
from numilab_human.resting_anatomy_verify import prepare_surface
from numilab_human.forearm_topology_patch import referenced_surface_topology_gate

points = np.asarray([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9],
], dtype="<f4")
tetra = np.asarray([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
], dtype="<u4")
ids = np.arange(len(points), dtype="<u8")
prepared = prepare_surface((51005, 132), points, tetra, source_vertex_ids=ids)
gate = referenced_surface_topology_gate(prepared)
assert prepared.closed is True
assert gate["passed_referenced_surface_topology"] is True
assert gate["unused_allocated_source_slot_count"] == 1
assert prepared.quotient_topology["closed_oriented_manifold_candidate"] is False

def must_reject(points, faces):
    try:
        prepared_defect = prepare_surface(
            (51005, 134), points, faces,
            source_vertex_ids=np.arange(len(points), dtype="<u8"))
    except ValueError:
        return
    assert referenced_surface_topology_gate(prepared_defect)[
        "passed_referenced_surface_topology"] is False

must_reject(points[:4], tetra[:3])
misoriented = tetra.copy()
misoriented[0] = misoriented[0, ::-1]
must_reject(points[:4], misoriented)

pinched_points = np.asarray([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [-1, 0, 0], [0, -1, 0], [0, 0, 1],
], dtype="<f4")
pinched_faces = np.asarray([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
    [0, 5, 4], [0, 4, 6], [4, 5, 6], [5, 0, 6],
], dtype="<u4")
must_reject(pinched_points, pinched_faces)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shared / "src")
    result = subprocess.run(
        [sys.executable, "-c", smoke, str(repo), str(shared)],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

