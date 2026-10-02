from numilab_human.whole_body_source_overlap_census import (
    eligible_pairs,
    map_compiled_sample_to_source,
    scope_summary,
    source_frame_rows,
)


def _source_semantic(member_id, body_name, body_index):
    return {
        "source_member_id": member_id,
        "source_provider": "bodyparts3d",
        "body_name": body_name,
        "body_index": body_index,
        "source_geometry_sha256": "geometry-" + member_id,
        "source_stable_id": 1,
        "visible_stable_id": 1,
    }


def test_family_and_baseline_maps_resolve_same_owner_pairs():
    member_a = {
        "member_id": "A",
        "source_member": "partof_BP3D_4.0_obj_99/A.obj",
        "source_member_sha256": "a" * 64,
        "source_body_id": 4,
        "core_body_index": 7,
        "myosim_body": "Abdomen",
        "hierarchy": "part_of",
        "vertex_count": 4,
        "triangle_count": 4,
    }
    manifests = [{
        "schema": "numi.human.source-organ-family-composite-payload.v1",
        "surfaces": [member_a],
    }]
    body_map = {
        "schema": "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1",
        "entries": [
            {"member_id": "A", "hierarchy": "part_of", "myosim_body": "Abdomen"},
            {"member_id": "B", "hierarchy": "part_of", "myosim_body": "Abdomen"},
            {"member_id": "C", "hierarchy": "part_of", "myosim_body": "torso"},
        ],
    }
    semantics = {
        "schema": "numi.human.whole-body-source-semantics.v1",
        "rows": [
            _source_semantic("A", "Abdomen", 7),
            _source_semantic("B", "Abdomen", 7),
            _source_semantic("C", "torso", 20),
        ],
    }

    resolved = source_frame_rows(manifests, body_map, semantics)
    pairs = [
        {"status": "surface_crossing", "body_index": 7,
         "first": {"source_member_id": "A", "body_name": "Abdomen", "body_index": 7},
         "second": {"source_member_id": "B", "body_name": "Abdomen", "body_index": 7}},
        {"status": "surface_crossing", "body_index": 7,
         "first": {"source_member_id": "A", "body_name": "Abdomen", "body_index": 7},
         "second": {"source_member_id": "C", "body_name": "torso", "body_index": 20}},
        {"status": "separate_closed_domains", "body_index": 7,
         "first": {"source_member_id": "A", "body_name": "Abdomen", "body_index": 7},
         "second": {"source_member_id": "B", "body_name": "Abdomen", "body_index": 7}},
    ]

    assert resolved["A"]["frame_source"] == "source_family_manifest"
    assert resolved["B"]["source_member"] == "partof_BP3D_4.0_obj_99/B.obj"
    assert resolved["B"]["frame_source"] == "baseline_body_map"
    assert eligible_pairs({"exact_pairs": pairs}, resolved) == [pairs[0]]


def test_scope_separates_tested_pairs_from_crossing_pairs():
    overlap = {"exact_pairs": [
        {"status": "surface_crossing"},
        {"status": "separate_closed_domains"},
        {"status": "surface_crossing"},
    ]}

    assert scope_summary(overlap, [overlap["exact_pairs"][0]], 2) == {
        "compiled_pairs_tested": 3,
        "compiled_crossing_pairs": 2,
        "eligible_same_declared_owner_pairs": 1,
        "source_members_read": 2,
        "excluded_crossing_pairs_without_same_declared_owner_map": 1,
    }


def test_compiled_crossing_sample_maps_back_to_original_source_faces():
    assert map_compiled_sample_to_source(
        [[2, 0], [3, 4]],
        [10, 11, 15, 17],
        [20, 22, 23, 24, 29],
    ) == [[15, 20], [17, 29]]
