from fractions import Fraction

import numpy as np
import pytest

from numilab_human import resting_brachiocephalic_geometry_gate as geometry_gate
from numilab_human import resting_brachiocephalic_ulnar_source_junction as ulnar


def _source_fixture():
    source_map = {}
    rows = {}
    for link in ulnar.BRANCH_LINKS:
        for role in ("parent", "child"):
            stable_id = link[f"{role}_stable_id"]
            member = link[f"{role}_member"]
            body = link[f"{role}_body_index"]
            vertex_count, face_count = ulnar.EXPECTED_COUNTS[stable_id]
            source_map[str(stable_id)] = {
                "source_member": member,
                "source_sha256": ulnar.EXPECTED_MEMBER_SHA256[member],
                "name": link[f"{role}_name"],
                "body_index": body,
                "layer": 2,
                "vascular_compartment_id": link[f"{role}_compartment_id"],
                "source_owner_metadata": {
                    "concept_id": link[f"{role}_fma"],
                    "myosim_body": link[f"{role}_myo_body"],
                    "core_body_index": body,
                },
            }
            vertices6 = np.zeros((vertex_count, 6), dtype="<f4")
            faces = np.zeros((face_count, 3), dtype="<u4")
            record = (body, 0, vertex_count, 0, face_count * 3, stable_id, 2, 0)
            rows[stable_id] = {
                "record": record,
                "body_index": body,
                "layer": 2,
                "flags": 0,
                "vertices6": vertices6,
                "faces": faces,
            }
    return source_map, rows


def test_missing_ulnar_links_are_exact_named_vascular_interfaces():
    assert ulnar.BRANCH_LINKS[0]["parent_stable_id"] == 479
    assert ulnar.BRANCH_LINKS[0]["child_stable_id"] == 487
    assert ulnar.BRANCH_LINKS[0]["parent_fma"] == "FMA22692"
    assert ulnar.BRANCH_LINKS[0]["child_fma"] == "FMA22798"
    assert ulnar.BRANCH_LINKS[1]["parent_stable_id"] == 481
    assert ulnar.BRANCH_LINKS[1]["child_stable_id"] == 489
    assert ulnar.BRANCH_LINKS[1]["parent_fma"] == "FMA22936"
    assert ulnar.BRANCH_LINKS[1]["child_fma"] == "FMA22952"
    assert geometry_gate.ULNAR_BRANCH_PORTS == {
        ((51011, 479), (51011, 487)),
        ((51011, 481), (51011, 489)),
    }
    assert ((51011, 479), (51011, 489)) not in geometry_gate.DECLARED_VESSEL_PORTS


def test_source_identity_selection_binds_member_fma_body_layer_and_compartment():
    source_map, rows = _source_fixture()
    selected = ulnar.validate_source_branch_identities(source_map, rows)
    assert set(selected) == {"479", "481", "487", "489"}
    assert selected["487"]["vascular_compartment_id"] == 3
    assert selected["489"]["vascular_compartment_id"] == 4
    assert selected["489"]["core_body_index"] == 92


@pytest.mark.parametrize(
    ("stable_id", "field", "wrong_value"),
    [
        (487, "vascular_compartment_id", 4),
        (481, "source_member", "FJ2260"),
        (489, "body_index", 91),
    ],
)
def test_source_identity_selection_rejects_mislabeled_rows(stable_id, field, wrong_value):
    source_map, rows = _source_fixture()
    source_map[str(stable_id)][field] = wrong_value
    with pytest.raises(ValueError, match="typed owner/member/FMA/compartment"):
        ulnar.validate_source_branch_identities(source_map, rows)


def test_source_identity_selection_rejects_changed_record_topology():
    source_map, rows = _source_fixture()
    row = rows[479]
    record = list(row["record"])
    record[4] += 3
    row["record"] = tuple(record)
    with pytest.raises(ValueError, match="ABI5 counts"):
        ulnar.validate_source_branch_identities(source_map, rows)


def test_exact_contact_component_grouping_uses_shared_rational_points():
    p0 = (Fraction(0), Fraction(0), Fraction(0))
    p1 = (Fraction(1, 3), Fraction(1, 2), Fraction(0))
    p2 = (Fraction(1), Fraction(1), Fraction(0))
    groups = ulnar._contact_components({
        (1, 7): (p0, p1),
        (2, 7): (p1,),
        (9, 11): (p2,),
    })
    assert groups == [[(1, 7), (2, 7)], [(9, 11)]]


def test_exact_contact_components_keep_disjoint_point_sets_separate():
    a = (Fraction(1, 3), Fraction(1, 2), Fraction(0))
    b = (Fraction(3333333333333333, 10**16), Fraction(1, 2), Fraction(0))
    assert a != b
    groups = ulnar._contact_components({(4, 8): (a,), (5, 8): (b,)})
    assert groups == [[(4, 8)], [(5, 8)]]


def test_exact_point_digest_is_order_independent_and_rational():
    first = {
        (2, 7): ((Fraction(1, 3), Fraction(1, 2), Fraction(0)),),
        (1, 4): ((Fraction(0), Fraction(0), Fraction(0)),),
    }
    reversed_order = dict(reversed(list(first.items())))
    assert ulnar._exact_points_digest(first) == ulnar._exact_points_digest(reversed_order)
