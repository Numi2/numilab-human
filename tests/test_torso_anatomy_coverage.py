"""Real atlas component/space typing and missing-family regressions."""
import copy
from pathlib import Path

import pytest

from numilab_human import model as human
from numilab_human.torso_anatomy_coverage import source_family_coverage, source_family_topology, source_organ_coverage
from numilab_human.torso_anatomy_audit import _native_source_family_coverage


@pytest.fixture(scope="module")
def source_data():
    mapping = human.read_json(human.REPOSITORY_ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json")
    relations = {h: human._bodyparts_source_element_relation_names(human.REPOSITORY_ROOT / "Sources", h)
                 for h in ["is_a", "part_of"]}
    return mapping, relations


def types_for(relations, member):
    return [{"concept_id": concept, "label": label} for concept, label, m in relations["is_a"] if m == member]


def test_real_named_ventricular_wall_is_a_component_not_a_whole_organ(source_data):
    mapping, relations = source_data
    wall = next(s for s in mapping["entries"] if s["member_id"] == "FJ2428")
    result = source_organ_coverage(wall, types_for(relations, wall["member_id"]))
    assert result["source_named_structure_type_matches"]
    assert not result["source_named_organ_type_matches"]
    assert result["source_structure_kind"] == "organ_component"
    assert result["organ_coverage"] == "source_named_organ_component_representation"
    assert not any(m == "FJ2428" for _, _, m in relations["part_of"])


@pytest.mark.parametrize("member", ["FJ2422", "FJ2423", "FJ2424", "FJ2425"])
def test_real_cardiac_cavities_cannot_be_organ_tissue(source_data, member):
    _, relations = source_data
    with pytest.raises(ValueError, match="anatomical space as tissue"):
        source_organ_coverage({"concept_id": "FMA7088", "source_name": "heart"}, types_for(relations, member))


def test_retained_family_is_complete_only_with_all_three_wall_members(source_data):
    mapping, relations = source_data
    full = source_family_coverage(mapping["coverage_requirements"], mapping["entries"], relations)
    assert full["passed"]
    assert full["requirements"][0]["selected_members"] == ["FJ2428", "FJ2438", "FJ2439"]
    old = source_family_coverage(mapping["coverage_requirements"], mapping["entries"][:22], relations)
    assert not old["passed"] and old["requirements"][0]["missing_members"] == ["FJ2428", "FJ2438"]


def test_real_ventricular_topology_defects_survive_complete_membership_and_cached_receipt_mutation():
    _, member, obj = human._bodyparts_obj_member(human.REPOSITORY_ROOT / "Sources", "is_a", "FJ2428")
    result = source_family_topology(obj, member)
    topology = result["exact_coordinate_quotient"]
    assert topology["nonmanifold_edges"] == 7 and topology["degenerate_face_ids"] == 3
    assert topology["duplicate_face_ids"] == 4 and topology["face_component_count"] == 8
    assert not topology["closed_oriented_manifold_candidate"]
    assert not result["physics_admitted"] and not result["repair_applied"] and not result["payload_geometry_modified"]
    topology["closed_oriented_manifold_candidate"] = True
    assert not source_family_topology(obj, member)["exact_coordinate_quotient"]["closed_oriented_manifold_candidate"]


@pytest.mark.parametrize("corruption", ["missing", "wrong_body", "wrong_layer", "duplicate"])
def test_native_family_oracle_checks_rendered_members_and_owner_independently(source_data, corruption):
    mapping, relations = source_data
    measured = [{"member_id": m, "layer": "organ", "core_body_index": 20}
                for m in ["FJ2428", "FJ2438", "FJ2439"]]
    assert _native_source_family_coverage(mapping["coverage_requirements"], measured, relations, {"torso": (20, {})})["passed"]
    if corruption == "missing":measured.pop()
    elif corruption == "wrong_body":measured[0]["core_body_index"] = 7
    elif corruption == "wrong_layer":measured[0]["layer"] = "vessel"
    else:measured.append(dict(measured[0]))
    report = _native_source_family_coverage(mapping["coverage_requirements"], measured, relations, {"torso": (20, {})})
    assert not report["passed"]


@pytest.mark.parametrize("corruption", ["missing_wall", "wrong_body", "extra_cavity", "missing_requirement"])
def test_compiler_rejects_incomplete_or_mistyped_anatomy_before_output(source_data, monkeypatch, tmp_path, corruption):
    mapping, relations = source_data
    mutated = copy.deepcopy(mapping)
    if corruption == "missing_wall":
        mutated["entries"] = [s for s in mutated["entries"] if s["member_id"] != "FJ2438"]
        message = "source family coverage is incomplete"
    elif corruption == "wrong_body":
        mutated["entries"][-1]["myosim_body"] = "Abdomen"
        message = "source family coverage is incomplete"
    elif corruption == "extra_cavity":
        mutated["entries"].append({"concept_id": "FMA7088", "source_name": "heart", "member_id": "FJ2422",
                                   "hierarchy": "part_of", "layer": "organ", "myosim_body": "torso"})
        message = "anatomical space as tissue"
    else:
        mutated.pop("coverage_requirements")
        message = "coverage requirements are missing"
    actual_read = human.read_json
    map_path = human.REPOSITORY_ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json"
    monkeypatch.setattr(human, "read_json", lambda path: mutated if Path(path).resolve() == map_path else actual_read(path))
    sources = human.REPOSITORY_ROOT / "Sources"
    anatomy = human.parse_bodyparts3d(sources, human.REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    registration = human.REPOSITORY_ROOT / "Build/knee-parity-registration-20260929/candidate.v6.registration.json"
    artifact = human.REPOSITORY_ROOT / "Build/myosim-fullbody"
    output = tmp_path / "rejected"
    with pytest.raises(human.ImportError, match=message):
        human.bodyparts_myosim_torso_anatomy_visual_payload(sources, anatomy, registration, artifact, output)
    assert not output.exists()
