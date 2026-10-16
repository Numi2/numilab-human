import pytest
from numilab_human.passive_attachment_composition import _validate_parent_rest_inventory


def test_parent_rest_legacy_scope_stays_ten():
    _validate_parent_rest_inventory([{}] * 10, [{}] * 8)
    with pytest.raises(Exception, match="ten surfaces"):
        _validate_parent_rest_inventory([{}] * 12, [{}] * 8)


def test_parent_rest_explicit_extended_scope_is_complete_and_order_independent():
    ids = tuple(range(1, 13))
    _validate_parent_rest_inventory([{"stable_id": sid} for sid in reversed(ids)], [{}] * 8, ids)


@pytest.mark.parametrize("ids", [(), (1, 1), (True,), (0,), [1]])
def test_parent_rest_declared_inventory_rejects_invalid_ids(ids):
    with pytest.raises(Exception, match="invalid"):
        _validate_parent_rest_inventory([{"stable_id": 1}], [{}] * 8, ids)


@pytest.mark.parametrize("rows", [[{"stable_id": 1}], [{"stable_id": 1}, {"stable_id": 1}], [{"stable_id": 1}, {"stable_id": 3}], [{"stable_id": True}, {"stable_id": 2}]])
def test_parent_rest_rejects_missing_duplicate_swapped_or_untyped_rows(rows):
    with pytest.raises(Exception):
        _validate_parent_rest_inventory(rows, [{}] * 8, (1, 2))


def test_parent_rest_carrier_count_remains_eight():
    with pytest.raises(Exception, match="eight existing carriers"):
        _validate_parent_rest_inventory([{"stable_id": 1}], [{}] * 9, (1,))
