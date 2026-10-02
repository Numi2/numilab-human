import gzip
import json

import pytest

from numilab_human.model import ImportError
from numilab_human.whole_body_organ_overlap import _surface_reference
from numilab_human.whole_body_overlap_semantics import audit


def write_gzip_json(path, value):
    path.write_bytes(gzip.compress(
        (json.dumps(value, sort_keys=True) + "\n").encode(), mtime=0
    ))


def receipts(tmp_path):
    first = {
        "source_stable_id": 1,
        "visible_stable_id": 1,
        "source_member_id": "FJ0001",
        "body_index": 7,
        "body_name": "Abdomen",
        "source_family_names": ["small_intestine"],
        "bodyparts3d_ontology_priority_class": "organ_region",
        "source_layer_code": 1,
        "selected_closed_embedded_surface_candidate": True,
        "compiled_geometry_sha256": "geometry-1",
        "triangle_count": 10,
    }
    second = {
        **first,
        "source_stable_id": 2,
        "visible_stable_id": 2,
        "source_member_id": "FJ0002",
        "compiled_geometry_sha256": "geometry-2",
        "triangle_count": 12,
    }
    third = {
        **first,
        "source_stable_id": 3,
        "visible_stable_id": 3,
        "source_member_id": "FJ0003",
        "compiled_geometry_sha256": "geometry-3",
        "triangle_count": 14,
    }
    provenance = {
        "selection_report_sha256": "selection",
        "inspection_selection_sha256": "inspection",
        "candidate_payload_sha256": "payload",
        "candidate_audit_sha256": "audit",
        "indexed_census_summary_sha256": "indexed-summary",
        "indexed_census_identity_sha256": "indexed-identity",
        "quotient_census_summary_sha256": "quotient-summary",
        "quotient_census_identity_sha256": "quotient-identity",
        "native_body_manifest_sha256": "manifest",
        "source_lock_sha256": "lock",
        "source_hierarchy_sha256": {"partof": "partof", "isa": "isa"},
    }
    semantics = {"schema": "numi.human.whole-body-source-semantics.v1",
                 "source_surface_count": 3, "rows": [first, second, third], **provenance,
                 "selected_report_sha256": provenance["selection_report_sha256"]}
    pair = {
        "first": {"source_stable_id": 1, "visible_stable_id": 1,
                  "source_member_id": "FJ0001", "body_index": 7,
                  "body_name": "Abdomen", "compiled_geometry_sha256": "geometry-1",
                  "triangle_count": 10},
        "second": {"source_stable_id": 2, "visible_stable_id": 2,
                   "source_member_id": "FJ0002", "body_index": 7,
                   "body_name": "Abdomen", "compiled_geometry_sha256": "geometry-2",
                   "triangle_count": 12},
        "source_hierarchy_annotation": {
            "shared_direct_concepts": {
                "part_of": [{"concept_id": "FMA20394", "name": "human body"}],
                "is_a": [],
            },
            "direct_concept_ancestry": [],
        },
        "exact_intersecting_triangle_pairs": 3,
        "maximum_intersection_segment_m": 0.004,
        "status": "surface_crossing",
    }
    overlap = {
        "schema": "numi.human.whole-body-organ-overlap-diagnostic.v1",
        "exact_pair_tests": 1,
        "surface_crossing_pair_count": 1,
        "selected_organ_surface_count": 3,
        "selected_organ_surfaces": [first, second, third],
        "aabb_disjoint_pairs": 1,
        "aabb_disjoint_pair_rows": [{
            "body_index": 7,
            "first_stable_id": 1,
            "first_visible_stable_id": 1,
            "first_member_id": "FJ0001",
            "second_stable_id": 3,
            "second_visible_stable_id": 3,
            "second_member_id": "FJ0003",
        }],
        "exact_pairs": [pair],
        "selection_report_sha256": provenance["selection_report_sha256"],
        "inspection_selection_sha256": provenance["inspection_selection_sha256"],
        "source_payload_sha256": provenance["candidate_payload_sha256"],
        "candidate_audit_sha256": provenance["candidate_audit_sha256"],
        "indexed_census_summary_sha256": provenance["indexed_census_summary_sha256"],
        "indexed_census_identity_sha256": provenance["indexed_census_identity_sha256"],
        "quotient_census_summary_sha256": provenance["quotient_census_summary_sha256"],
        "quotient_census_identity_sha256": provenance["quotient_census_identity_sha256"],
        "native_body_manifest_sha256": provenance["native_body_manifest_sha256"],
        "source_lock_sha256": provenance["source_lock_sha256"],
        "source_hierarchy_sha256": provenance["source_hierarchy_sha256"],
    }
    overlap_path = tmp_path / "overlap.json.gz"
    semantics_path = tmp_path / "semantics.json.gz"
    write_gzip_json(overlap_path, overlap)
    write_gzip_json(semantics_path, semantics)
    return overlap_path, semantics_path


def test_audit_joins_family_and_keeps_overlap_intent_unresolved(tmp_path):
    overlap_path, semantics_path = receipts(tmp_path)

    result = audit(overlap_path, semantics_path)

    assert result["relation_class_counts"] == {"shared_declared_source_family": 1}
    assert result["crossings"][0]["shared_source_families"] == ["small_intestine"]
    assert result["crossings"][0]["maximum_intersection_segment_m"] == 0.004
    assert result["overlap_intent_resolved"] is False
    assert result["clinical_anatomy"] is False
    assert result["selected_organ_surfaces_identity_joined"] == 3


def test_audit_rejects_crosswalk_identity_drift(tmp_path):
    overlap_path, semantics_path = receipts(tmp_path)
    semantics = json.loads(gzip.decompress(semantics_path.read_bytes()))
    semantics["rows"][0]["body_index"] = 8
    write_gzip_json(semantics_path, semantics)

    with pytest.raises(ImportError, match="missing or differs in crosswalk"):
        audit(overlap_path, semantics_path)


def test_audit_rejects_incomplete_pair_receipt(tmp_path):
    overlap_path, semantics_path = receipts(tmp_path)
    overlap = json.loads(gzip.decompress(overlap_path.read_bytes()))
    overlap["exact_pair_tests"] = 2
    write_gzip_json(overlap_path, overlap)

    with pytest.raises(ImportError, match="complete exact-pair set"):
        audit(overlap_path, semantics_path)


def test_audit_rejects_receipts_from_different_selection_states(tmp_path):
    overlap_path, semantics_path = receipts(tmp_path)
    overlap = json.loads(gzip.decompress(overlap_path.read_bytes()))
    overlap["inspection_selection_sha256"] = "other-selection"
    write_gzip_json(overlap_path, overlap)

    with pytest.raises(ImportError, match="input provenance differs"):
        audit(overlap_path, semantics_path)


def test_audit_rejects_aabb_visible_id_drift(tmp_path):
    overlap_path, semantics_path = receipts(tmp_path)
    overlap = json.loads(gzip.decompress(overlap_path.read_bytes()))
    overlap["aabb_disjoint_pair_rows"][0]["second_visible_stable_id"] = 99
    write_gzip_json(overlap_path, overlap)

    with pytest.raises(ImportError, match="AABB surface missing or differs"):
        audit(overlap_path, semantics_path)


def test_surface_reference_keeps_source_and_visible_stable_ids_distinct():
    row = {"source_stable_id": 20, "source_member_id": "FJ2409",
           "source_body_index": 7}

    result = _surface_reference(row, ["root"] * 7 + ["Abdomen"],
                                visible_stable_id=581)

    assert result["source_stable_id"] == 20
    assert result["visible_stable_id"] == 581
    assert result["body_name"] == "Abdomen"
