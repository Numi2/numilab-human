import numpy as np
import pytest
from numilab_human.skin_full_solid_motion import (
    bind_tissue_layer_offsets, transport_tissue_layer,
)

def fixture():
    outer = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    parents = np.array([[0, 1, 2], [0, 1, 2]])
    bary = np.array([[.2, .3, .5], [.6, .1, .3]])
    tissue = bary @ outer + np.array([[0., 0., -.001], [.0002, -.0001, -.002]])
    offsets = bind_tissue_layer_offsets(outer, tissue, parents, bary)
    return outer, parents, bary, tissue, offsets

def test_source_thickness_and_rigid_motion_are_preserved():
    outer, parents, bary, tissue, offsets = fixture()
    np.testing.assert_allclose(transport_tissue_layer(
        outer, outer, parents, bary, offsets), tissue, atol=1e-15)
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    translated = outer @ rotation.T + [4., 5., 6.]
    result = transport_tissue_layer(outer, translated, parents, bary, offsets)
    np.testing.assert_allclose(result, tissue @ rotation.T + [4., 5., 6.], atol=1e-15)
    # Independent world-space offsets would place this hidden sheet on the wrong
    # side after rotation. The material normal and each source thickness rotate.
    normal = rotation @ [0., 0., 1.]
    signed = (result - np.einsum("ni,nij->nj", bary, translated[parents])) @ normal
    np.testing.assert_allclose(signed, [-.001, -.002], atol=1e-15)

def test_stretch_keeps_source_thickness_without_moving_outer_points():
    outer, parents, bary, _, offsets = fixture()
    moved = outer * [2., 3., 1.]
    before = moved.copy()
    result = transport_tissue_layer(outer, moved, parents, bary, offsets)
    np.testing.assert_array_equal(moved, before)
    np.testing.assert_allclose(result[:, 2], [-.001, -.002], atol=1e-15)

@pytest.mark.parametrize("failure", ["collapse", "nan", "negative", "bad_parent", "source_collapse"])
def test_invalid_tissue_bindings_fail(failure):
    outer, parents, bary, _, offsets = fixture()
    moved = outer.copy()
    if failure == "collapse": moved[2] = moved[1]
    if failure == "source_collapse": outer[2] = outer[1]
    if failure == "nan": offsets[0, 0] = np.nan
    if failure == "negative": bary[0] = [-.1, .4, .7]
    if failure == "bad_parent": parents[0, 0] = 99
    with pytest.raises(Exception):
        transport_tissue_layer(outer, moved, parents, bary, offsets)


def test_gpu_records_preserve_material_fields_and_reject_write_races():
    from numilab_human.skin_full_solid_motion import tissue_layer_gpu_records
    _, parents, bary, _, offsets = fixture()
    outputs = np.array([3, 4])
    normals = np.array([[0., 0., -1.], [0., 0., -1.]])
    records = tissue_layer_gpu_records(parents, outputs, bary, offsets, normals,
                                      vertex_count=5)
    assert records.dtype.itemsize == 64
    assert records.nbytes == 128
    np.testing.assert_array_equal(records["parent_and_output"][:, 3], outputs)
    np.testing.assert_array_equal(records["offset"][:, :3], offsets.astype(np.float32))
    for bad in (np.array([3, 3]), np.array([2, 4]), np.array([-1, 4]), np.array([3, 5])):
        with pytest.raises(Exception):
            tissue_layer_gpu_records(parents, bad, bary, offsets, normals,
                                     vertex_count=5)
