"""Exact surface quadrature regressions; not clinical/loaded qualification."""
import copy
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from numilab_human import model as h


def assert_paths(surface, candidates, centre, radius):
    source_indices = surface.get("_source_triangle_indices", list(range(len(surface["triangles"]))))
    triangles = {key: [surface["vertices"][index] for index in tri]
                 for key, tri in zip(source_indices, surface["triangles"], strict=True)}
    for candidate in candidates:
        source = candidate["source"]
        expected = np.asarray(source["barycentric"]) @ np.asarray(triangles[source["source_triangle_index"]])
        assert candidate["point"] == pytest.approx(expected, abs=1e-12)
        points = source["path_points_m"]
        indices = source["path_triangle_indices"]
        assert points[0] == centre
        assert points[-1] == candidate["point"]
        assert len(points) == len(indices) + 1
        for index, first, second in zip(indices, points[:-1], points[1:], strict=True):
            for point in (first, second):
                closest, _ = h._tendon_closest_point_on_triangle(point, triangles[index])
                assert math.dist(point, closest) <= 1e-10
        length = sum(math.dist(first, second) for first, second in zip(points[:-1], points[1:], strict=True))
        assert length == pytest.approx(source["path_length_m"], abs=1e-12)
        assert length <= radius + 1e-12
        assert math.dist(centre, candidate["point"]) <= radius + 1e-12


def test_coarse_faces_supply_bounded_surface_points_without_nearby_vertices():
    surface = {"vertices": [[-.03,-.03,0.], [.03,-.03,0.], [.03,.03,0.], [-.03,.03,0.]],
               "triangles": [(0,1,2), (0,2,3)], "_source_triangle_indices": [23,41]}
    original = copy.deepcopy(surface)
    centre = [0.,0.,0.]
    assert all(math.dist(centre, v) > .012 for v in surface["vertices"])
    candidates, visited = h._tendon_connected_face_path_candidates(surface, 0, centre, .012)
    assert visited == 2
    assert {c["source"]["source_triangle_index"] for c in candidates} == {23,41}
    assert_paths(surface, candidates, centre, .012)
    assert surface == original
    assert h._tendon_connected_face_path_candidates(surface, 0, centre, .012) == (candidates, visited)


def test_connected_face_paths_do_not_borrow_a_nearby_disconnected_fragment():
    surface = {"vertices": [[-.03,-.03,0.], [.03,-.03,0.], [.03,.03,0.], [-.03,.03,0.],
                            [-.004,-.004,.001], [.004,-.004,.001], [0.,.004,.001]],
               "triangles": [(0,1,2), (0,2,3), (4,5,6)]}
    candidates, visited = h._tendon_connected_face_path_candidates(surface, 0, [0.,0.,0.], .012)
    assert visited == 2
    assert all(c["source"]["source_triangle_index"] != 2 for c in candidates)
    assert_paths(surface, candidates, [0.,0.,0.], .012)


@pytest.mark.parametrize("muscle,endpoint,member", [
    ("vasint_l", "insertion", "FJ3282"), ("vasmed_l", "origin", "FJ3259"),
])
def test_real_repaired_knee_recovers_surface_without_moving_source_point(muscle, endpoint, member):
    artifact = os.environ.get("NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT")
    bones = os.environ.get("NUMILAB_HUMAN_MOTION_BONES")
    tendon = os.environ.get("NUMILAB_HUMAN_NATIVE_TENDON_PAYLOAD")
    if not all((artifact, bones, tendon)):
        pytest.skip("exact repaired source anatomy inputs were not supplied")
    reference = json.loads((Path(artifact) / "myosim-fullbody-reference.manifest.json").read_text())
    manifest = json.loads((Path(tendon).parent / "numi-human-tendon-attachments.manifest.json").read_text())
    endpoint_record = next(e for e in manifest["endpoints"] if e["muscle"] == muscle and e["endpoint"] == endpoint)
    source = list(endpoint_record["source_local_point_m"])
    surfaces, _, _ = h._numi_human_bone_envelope_surfaces(Path(bones), reference["source"]["archive_sha256"])
    surface = next(s for s in surfaces[endpoint_record["body_index"]] if s["member_id"] == member)
    surface["body_index"] = endpoint_record["body_index"]
    original = copy.deepcopy(surface)
    with patch.object(h, "_tendon_connected_face_path_candidates", return_value=([],0)):
        old, _ = h._numi_human_tendon_surface_envelope(source, surface, .012, .012, 4.)
    assert old is None
    envelope, reason = h._numi_human_tendon_surface_envelope(source, surface, .012, .012, 4.)
    assert reason == "admitted_connected_face_path_exact_surface_patch"
    assert envelope["bone_member_id"] == member
    assert envelope["resolved_local_point_m"] == source
    assert envelope["surface_distance_m"] <= .012
    assert envelope["patch_radius_m"] <= .012 + 1e-12
    assert envelope["sampled_total_force_amplification"] < 2.
    assert source == endpoint_record["source_local_point_m"]
    assert surface["vertices"] == original["vertices"]
    assert surface["triangles"] == original["triangles"]
    candidates = [{"point": point, "source": witness} for point, witness in
                  zip(envelope["node_local_points_m"], envelope["node_surface_sources"], strict=True)]
    assert_paths(surface, candidates, envelope["nearest_local_point_m"], .012)
    nodes = np.asarray(envelope["node_local_points_m"])
    maps = np.asarray(envelope["force_maps"])
    for direction in ([1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [1.,2.,-3.]):
        force = np.asarray(direction); force /= np.linalg.norm(force)
        nodal = maps @ force
        assert np.linalg.norm(nodal.sum(axis=0) - force) <= 2e-6
        assert np.linalg.norm(np.cross(nodes - source, nodal).sum(axis=0)) <= 2e-8
