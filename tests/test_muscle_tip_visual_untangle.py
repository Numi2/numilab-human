"""Pinned bilateral source geometry must untangle without hiding source drift."""

from pathlib import Path
import hashlib

import pytest

from numilab_human import model
from numilab_human.muscle_tip_visual_untangle import untangle


SOURCES = model.REPOSITORY_ROOT/'Sources'


@pytest.mark.parametrize('member', ['FJ1444', 'FJ1444M'])
def test_pinned_source_tip_visual_candidate(member):
    _, path, obj = model._bodyparts_obj_member(SOURCES, 'is_a', member)
    vertices, faces = model._bodyparts_obj_triangles(obj, path)
    before = [point[:] for point in vertices]
    derived, proof = untangle(vertices, faces, member, hashlib.sha256(obj).hexdigest())
    assert vertices == before
    assert len(derived) == len(vertices) and len(faces) == 1728
    assert proof['source_exact_self_intersection_pairs_before'] == 3
    assert proof['compiled_fp32_self_intersection_pairs_after'] == 0
    assert proof['maximum_vertex_displacement_mm'] < .066
    assert len(proof['moved_source_vertex_ids']) == 6


def test_source_hash_and_geometry_drift_refused():
    member = 'FJ1444'
    _, path, obj = model._bodyparts_obj_member(SOURCES, 'is_a', member)
    vertices, faces = model._bodyparts_obj_triangles(obj, path)
    digest = hashlib.sha256(obj).hexdigest()
    with pytest.raises(ValueError, match='pinned member identity'):
        untangle(vertices, faces, member, '0'*64)
    changed = [list(point) for point in vertices]
    changed[553][0] += .1
    with pytest.raises(ValueError, match='source crossing signature changed|coincident source-point groups changed'):
        untangle(changed, faces, member, digest)
