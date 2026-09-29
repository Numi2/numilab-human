import numpy as np
import pytest

from numilab_human import anatomical_laterality as laterality
from numilab_human.model import ImportError as HumanImportError


def test_centroid_cannot_clear_a_disconnected_wrong_side_component():
    # The small contralateral triangle contributes little area, as in FJ1337.
    world = np.array([
        [-.05, 0, 0], [-.05, .02, 0], [-.05, 0, .02],
        [.03, 0, 0], [.03, .0001, 0], [.03, 0, .0001],
    ])
    faces = [[0, 1, 2], [3, 4, 5]]
    kwargs = dict(torso_origin=np.zeros(3), left_axis=np.array([1., 0., 0.]),
                  side_sign=-1)
    measured = laterality.support_metrics(world, faces, source_faces=[71, 99], **kwargs)
    assert measured['centroid_on_expected_side']
    assert not measured['all_selected_vertices_on_expected_side']
    assert measured['wrong_side_selected_face_ids'] == [1]
    assert measured['wrong_side_source_face_ids'] == [99]
    repaired = laterality.support_metrics(world[:3], faces[:1], source_faces=[71], **kwargs)
    assert repaired['all_selected_vertices_on_expected_side']
    assert repaired['wrong_side_triangle_count'] == 0


def test_bilateral_cohort_cannot_silently_drop_a_member():
    metadata = [{'stable_id': i, 'label': 'other'} for i in range(1, 580)]
    cursor = 0
    for family, count in laterality.COUNTS.items():
        if 'lung lobe' in family:
            name = 'Inferior lobe of left lung' if family.startswith('left') else 'Inferior lobe of right lung'
            for _ in range(count):
                metadata[304+cursor]['object_name'] = name
                metadata[304+cursor].pop('label')
                cursor += 1
        else:
            for row in metadata:
                if row['label'] == 'other':
                    row['label'] = family
                    break
            for _ in range(count-1):
                for row in metadata:
                    if row['label'] == 'other':
                        row['label'] = family
                        break
    # The production Z-Anatomy cohort occupies IDs 305-309.
    selected = laterality.cohort(metadata)
    assert len(selected) == 58
    metadata[0]['label'] = 'other'
    with pytest.raises(HumanImportError, match='semantic coverage changed'):
        laterality.cohort(metadata)


def test_native_torso_left_axis_is_explicit():
    orientation = laterality.quaternion_rotation([.5, -.5, -.5, .5])
    assert np.allclose(orientation @ np.array([0., 0., -1.]), [1., 0., 0.])
    with pytest.raises(HumanImportError, match='quaternion norm'):
        laterality.quaternion_rotation([0, 0, 0, 0])
