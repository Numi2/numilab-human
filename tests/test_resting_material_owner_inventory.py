"""Explicit native support inventories preserve structural zeros and exact shared fields."""
import numpy as np
import pytest
from numilab_human import resting_anatomy_verify as av

CORES=(41,43,49,45,52,53,54)
FACES=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.uint32)
POINTS=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)

def surface(sid=109, *, expected=CORES, cores=CORES, weights=None):
    gids=np.array([0,1,2,3] if sid==109 else [0,11,12,13],dtype=np.uint64)
    weights=np.tile(np.r_[np.full(4,.25),np.zeros(len(cores)-4)],(4,1)).astype(np.float64) if weights is None else weights
    lineage={int(g):dict(source_gid=int(g),source_parent_gids=[int(g)],material_mapping_kind=3,material_binding='independent_existing_core_carrier',core_body_indices=list(cores),core_weights=w.tolist()) for g,w in zip(gids,weights)}
    positions=POINTS.copy() if sid==109 else -POINTS
    positions[0]=POINTS[0]
    return av.prepare_surface((51005,sid),positions,FACES if sid==109 else FACES[:,::-1],source_vertex_ids=gids,source_face_lineage=[dict(source_parent_face=i) for i in range(4)],source_vertex_lineage=lineage,ordered_core_body_indices=cores,core_weight_gids=gids,core_weights_float64=weights,expected_ordered_core_body_indices=expected)

def declarations(first,second):
    result=av.audit_surface_pair(first,second)
    return [dict(first_identity=list(first.key),second_identity=list(second.key),first_face_index=a,second_face_index=b,feature_gids=[0]) for a,b in result['unexpected_triangle_pairs']]

def test_explicit_native_inventory_keeps_all_coefficients_and_structural_zeros():
    a,b=surface(),surface(135)
    assert a.declared_core_body_indices==CORES
    assert a.core_weights_by_gid[0].tobytes()==np.array([.25,.25,.25,.25,0,0,0],np.float64).tobytes()
    assert av.audit_surface_pair(a,b,intended_shared_feature_pairs=declarations(a,b))['passed']
    proof=av._validate_sequence_prepared_surface(a.key,a)
    assert proof['declared_material_owner_order']==list(CORES)
    assert a.diagnostics()['declared_material_owner_order']==list(CORES)

def test_legacy_seven_owner_rows_remain_rejected_without_explicit_inventory():
    with pytest.raises(ValueError,match='21 unique'):surface(expected=None)

def test_legacy_twenty_one_owner_zero_support_is_still_rejected():
    with pytest.raises(ValueError,match='strictly positive'):surface(expected=None,cores=tuple(range(21)))

def test_explicit_inventory_is_compared_in_exact_order():
    with pytest.raises(ValueError,match='differs'):surface(expected=tuple(reversed(CORES)))

@pytest.mark.parametrize('inventory',[(),(41,41),(True,),(-1,),(192,),(41.0,),set(CORES),tuple(range(193))])
def test_invalid_declared_inventory_is_rejected(inventory):
    with pytest.raises(ValueError,match='inventory'):surface(expected=inventory)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.001,.3])
def test_full_source_coefficients_must_be_finite_nonnegative_and_normalized(bad):
    weights=np.tile(np.array([.25,.25,.25,.25,0,0,0],np.float64),(4,1));weights[0,0]=bad
    with pytest.raises(ValueError):surface(weights=weights)

def test_shared_field_tampering_rejects_even_when_both_rows_still_normalize():
    a,b=surface(),surface(135);d=declarations(a,b)
    b.core_weights_by_gid[0][0]+=.01;b.core_weights_by_gid[0][1]-=.01
    assert not av.audit_surface_pair(a,b,intended_shared_feature_pairs=d)['passed']

def test_mutated_weight_width_cannot_pass_shared_contact():
    a,b=surface(),surface(135);d=declarations(a,b)
    a.core_weights_by_gid[0]=b.core_weights_by_gid[0]=np.array([1.],np.float64)
    assert not av.audit_surface_pair(a,b,intended_shared_feature_pairs=d)['passed']

def test_sequence_revalidates_declared_owner_order_and_coefficients():
    a=surface();a.core_weights_by_gid[0][0]=-.1
    with pytest.raises(ValueError,match='material weights'):av._validate_sequence_prepared_surface(a.key,a)
    a=surface();a.declared_core_body_indices=tuple(reversed(CORES))
    with pytest.raises(ValueError,match='inventory'):av._validate_sequence_prepared_surface(a.key,a)

def test_explicit_inventory_requires_source_lineage_and_actual_material_rows():
    with pytest.raises(ValueError,match='complete source'):
        av.prepare_surface((51005,109),POINTS,FACES,expected_ordered_core_body_indices=CORES)
