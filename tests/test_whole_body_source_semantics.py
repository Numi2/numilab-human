"""The atlas taxonomy crosswalk stays source-bound and keeps defects separate."""

from numilab_human.model import REPOSITORY_ROOT
from numilab_human.whole_body_source_semantics import audit


def test_crosswalk_covers_surfaces_and_exposes_organ_layer_semantics():
    root = REPOSITORY_ROOT
    media = root / "Docs/media"
    topology = media / "source-topology-repair-20260929"
    whole_body = media / "whole-body-embeddedness-20260929"
    report = audit(
        whole_body / "selected-surface-gate.json",
        whole_body / "indexed-census.tar.gz",
        whole_body / "quotient-census.tar.gz",
        topology / "payload/source-topology-repair-candidates.nhanatomy",
        topology / "source-audit.json",
        topology / "inspection-selection.final.json",
        root / "Sources",
        root / "sources.lock.json",
        root / "Build/myosim-fullbody/myosim-fullbody-reference.manifest.json",
    )

    assert report["source_surface_count"] == 579
    assert report["unique_source_member_count"] == 573
    assert report["source_identity_complete_surface_count"] == 579
    assert report["bodyparts3d_member_id_missing_surface_count"] == 6
    assert report["source_configured_object_count"] == 6
    assert report["source_configured_fma_concept_count"] == 5
    assert report["source_configured_fma_concept_id_unmapped_surface_count"] == 1
    assert report["source_identity_partition_surface_counts"] == {
        "declared_bodyparts3d_family_member": 571,
        "bodyparts3d_baseline_member_outside_families": 2,
        "zanatomy_source_objects": 6,
    }
    assert report["declared_source_family_count"] == 46
    assert report["baseline_source_member_count"] == 381
    assert report["baseline_source_member_in_family_count"] == 379
    outside_family = [row for row in report["rows"] if row[
        "source_family_membership_status"
    ] == "bodyparts3d_member_outside_declared_source_families"]
    assert [(row["source_stable_id"], row["source_member_id"])
            for row in outside_family] == [(12, "FJ1737"), (23, "FJ2428")]
    assert report["bodyparts3d_ontology_priority_class_order"] == [
        "organ", "organ_region", "organ_component", "cardinal_organ_part",
    ]
    organ = report["layer_counts"]["organ"]
    assert organ["surface_count"] == 120
    assert organ["bodyparts3d_ontology_priority_class_counts"] == {
        "cardinal_organ_part": 8,
        "organ": 21,
        "organ_component": 21,
        "organ_region": 70,
    }
    assert sum(
        not row["selected_closed_embedded_surface_candidate"]
        for row in report["rows"] if row["source_layer_code"] == 1
    ) == 8
    assert all(row["bodyparts3d_ontology_priority_class"] != "ontology_unmapped"
               for row in report["rows"] if row["source_layer_code"] == 1)
    missing = [row for row in report["rows"] if row["source_member_id"] is None]
    assert [row["source_stable_id"] for row in missing] == [305, 306, 307, 308, 309, 310]
    assert all(not row["selected_closed_embedded_surface_candidate"] for row in missing)
    expected_concepts = {
        305: "FMA7371", 306: "FMA7337", 307: "FMA7383",
        308: "FMA7370", 309: "FMA7333", 310: None,
    }
    assert {row["source_stable_id"]: row["source_anatomy_concept_id"]
            for row in missing} == expected_concepts
    assert all(row["source_anatomy_concept_mapping_status"]
               == "matched_to_pinned_part_of_parent_edge" for row in missing[:5])
    assert missing[5]["source_anatomy_concept_mapping_status"] == (
        "source_config_has_no_fma_concept_id"
    )
    assert report["clinical_anatomy"] is False
    assert report["physical_volume"] is False
    assert report["organ_mechanics"] is False
