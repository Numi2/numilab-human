import pytest

from numilab_human.model import (
    ImportError,
    _bodyparts_exact_fp32_degenerate_faces,
    _bodyparts_non_degenerate_fp32_projection,
)


def test_projection_uses_bounded_source_segment_to_avoid_float32_collapse() -> None:
    source = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    target = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    triangles = [(0, 1, 2)]
    result, stored, evidence = _bodyparts_non_degenerate_fp32_projection(
        source, target, triangles, lambda vertices: vertices, "synthetic"
    )
    assert evidence["degenerate_face_ids_before"] == [0]
    assert evidence["degenerate_face_ids_after"] == []
    assert evidence["status"] == "bounded_source_segment_backoff"
    assert 0 < evidence["max_retreat_m"] < 1.0e-5
    assert result == stored
    assert _bodyparts_exact_fp32_degenerate_faces(stored, triangles) == []
    assert source[2] == [0.0, 1.0, 0.0] and target[2] == [0.0, 0.0, 0.0]


def test_projection_refuses_a_source_face_that_cannot_be_recovered() -> None:
    source = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    with pytest.raises(ImportError, match="unresolved compiled Float32 degenerate faces"):
        _bodyparts_non_degenerate_fp32_projection(
            source, source, [(0, 1, 2)], lambda vertices: vertices, "synthetic"
        )
