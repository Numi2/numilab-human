"""Checks the payload gate rejects edits outside the declared weight transform."""
import struct
import numpy as np
import pytest
from numilab_human.skin_surface_audit import verify_bone_proximity_weight_edit_payload

IDS = [128, 131, 145, 20]
PARAMS = dict(minimum_femur_distance_m=.020, maximum_femur_distance_m=.085,
              pelvis_to_femur_ratio_start=1.5, pelvis_to_femur_ratio_full=2.5,
              femur_weight_start=.02, femur_weight_full=.12,
              pelvis_weight_start=.02, pelvis_weight_full=.10, transfer_fraction=1.)

def payload(row):
    n, b = 4, 4
    raw = bytearray(struct.pack('<8s5I32s', b'NHSKIN1\0', 5, b, n, 6, 7, bytes(32)))
    for body in IDS:
        raw += struct.pack('<I8f', body, 0, 0, 0, 0, 0, 0, 1, 0)
    field = np.tile(np.asarray(row, dtype='<f4'), (n, 1))
    points = [(0,0,0), (0,0,0), (1,0,0), (0,1,0)]
    for p, weights in zip(points, field):
        order = np.argsort(-weights, kind='stable')[:4]
        compact = weights[order].astype(np.float64)
        compact /= compact.sum()
        raw += struct.pack('<6f4I4f', *p, 0,0,1, *order, *compact)
    raw += struct.pack('<6I', 0,2,3, 1,3,2)
    raw += field.tobytes()
    return bytes(raw)

def inputs():
    source = payload([.3,.3,.2,.2])
    candidate = payload([0,.6,.2,.2])
    distances = {128:np.full(4,.3), 131:np.full(4,.01), 145:np.full(4,.15)}
    return source, candidate, distances

def verify(source, candidate, distances, parameters=None):
    return verify_bone_proximity_weight_edit_payload(
        source, candidate, distances, pelvis_body_id=128, femur_body_ids=(131,145),
        parameters=PARAMS if parameters is None else parameters)

def test_replay_preserves_duplicate_seam_and_validates_full_and_compact_fields():
    report = verify(*inputs())
    assert report['passed']
    assert report['changed_vertex_count'] == 4
    assert report['exact_coincident_vertex_group_count'] == 1
    assert report['compact_influence_cache_exact']

@pytest.mark.parametrize('offset,match', [
    (60, 'canonical binding records'),
    (60+36*4, 'source geometry normals and topology'),
    (60+36*4+56*4, 'source geometry normals and topology'),
    (60+36*4+24, 'compact owner cache'),
    (60+36*4+40, 'compact weight cache'),
    (60+36*4+56*4+4*6, 'full field replay'),
])
def test_payload_tampering_is_rejected(offset, match):
    source, candidate, distances = inputs()
    corrupted = bytearray(candidate)
    corrupted[offset] ^= 1
    with pytest.raises(Exception, match=match):
        verify(source, bytes(corrupted), distances)

def test_duplicate_seam_distance_mismatch_is_rejected_before_replay():
    source, candidate, distances = inputs()
    distances[131][1] += .001
    with pytest.raises(Exception, match='seam distances'):
        verify(source, candidate, distances)

@pytest.mark.parametrize('key,value', [('transfer_fraction',float('nan')),
                                      ('maximum_femur_distance_m',float('inf'))])
def test_nonfinite_parameters_rejected(key, value):
    with pytest.raises(Exception, match='explicit finite parameters'):
        verify(*inputs(), parameters={**PARAMS,key:value})
