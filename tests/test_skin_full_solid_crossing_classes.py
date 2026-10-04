from collections import Counter

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.skin_full_solid_crossing_classes import classify_pairs


def test_classify_retained_triangle_pairs_by_outer_subset():
    faces = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    outer = {tuple(sorted(faces[0])), tuple(sorted(faces[2]))}

    classes = classify_pairs([[0, 1], [1, 2], [0, 2]], faces, outer)

    assert classes == Counter({
        "inner_or_connector_to_outer": 2,
        "outer_to_outer": 1,
    })


def test_classification_rejects_invalid_face_id():
    with pytest.raises(HumanImportError, match="triangle id range"):
        classify_pairs([[0, 4]], [[0, 1, 2]], {(0, 1, 2)})
