from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

import numilab_human.ownership as ownership_module
from numilab_human.model import ImportError as HumanImportError
from numilab_human.ownership import (
    AUTHORING_SCHEMA,
    BODY_COMPOSITION,
    MUSCLE_ROUTES,
    _counts,
    _immutable_write,
    compile_manifest,
    compile_paths,
    validate_ownership,
)
from numilab_human.target_coverage import (
    MANDATORY,
    OBLIGATIONS,
    Inventory,
    canonical_bytes,
    digest,
)
from numilab_human.target_coverage import validate_manifest as validate_coverage


def _coverage(route_names: list[str]) -> dict:
    inventory = Inventory()
    mandatory = inventory.source(
        "mandatory",
        {"authority": "Docs/DEVELOPMENT_ROADMAP.md", "catalog_version": 1,
         "license": "Apache-2.0"},
    )
    for domain, names in MANDATORY.items():
        for name in names:
            inventory.leaf(
                mandatory, f"{domain}/{name}", name.replace("_", " "),
                "mandatory_target", domain=domain,
                declaration={"domain": domain, "name": name},
            )
    myosim = inventory.source(
        "myosim_fullbody",
        {"repository": "https://github.com/MyoHub/myo_sim", "revision": "fixture",
         "license": "Apache-2.0"},
    )
    for index, name in enumerate(route_names):
        inventory.leaf(
            myosim,
            "composed/myofullbody/muscles/" + quote(name, safe=""),
            name,
            "composed:muscles",
            declaration={"id": index, "name": name},
        )
    leaves = sorted(inventory.leaves.values(), key=lambda item: item["leaf_sha256"])
    result = {
        "schema": "HumanPack.target-coverage.v1",
        "compiler": "numilab-human.target-coverage.2",
        "source_lock_sha256": "a" * 64,
        "previous_manifest_sha256": None,
        "obligations": OBLIGATIONS,
        "source_records": sorted(inventory.sources.values(), key=lambda item: item["record_sha256"]),
        "registers": [],
        "register_history": [],
        "leaves": leaves,
        "counts": {
            "leaves": len(leaves),
            "mandatory_leaves": 95,
            "by_kind": dict(sorted(Counter(item["kind"] for item in leaves).items())),
            "unresolved_current_registers": 0,
        },
        "scope_status": "source_union_materialized",
        "integrated_qualification": "unknown",
        "evidence_boundary": "Fixture source inventory only; no physical qualification.",
    }
    result["manifest_sha256"] = digest(result)
    validate_coverage(result)
    return result


def _route_identities(count: int) -> dict:
    return {
        "schema": "HumanPack.muscle-route-volume-join-candidate.v1",
        "status": "partial",
        "counts": {"source_route_count": count},
        "qualification": {
            "source_route_identity_bound": True,
            "unbound_routes_retained": True,
        },
        "route_rows": [
            {"source_actuator_index": index, "name": f"route_{index}"}
            for index in range(count)
        ],
    }


def _routes(count: int, *, with_candidate: bool = True) -> dict:
    rows = []
    for index in range(count):
        candidate = with_candidate and index < 2
        rows.append({
            "source_actuator_index": index,
            "name": f"route_{index}",
            "candidate_volume_m3": (index + 1) * 0.001 if candidate else 0.0,
            "candidate_mass_kg": float(index + 1) if candidate else 0.0,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "volumetric_active_force_owner": False,
        })
    identities = _route_identities(count)
    identity_hash = hashlib.sha256(canonical_bytes(identities) + b"\n").hexdigest()
    allocated_volume = sum(row["candidate_volume_m3"] for row in rows)
    allocated_mass = sum(row["candidate_mass_kg"] for row in rows)
    return {
        "schema": "HumanPack.muscle-route-mass-partition-candidate.v1",
        "status": "partial",
        "source": {
            "route_volume_receipt": "fixture-route-identities.json",
            "route_volume_receipt_sha256": identity_hash,
        },
        "policy": {"name": "equal_incidence", "measured_partition": False},
        "counts": {"source_route_count": count},
        "totals": {
            "source_candidate_volume_m3": allocated_volume,
            "allocated_candidate_volume_m3": allocated_volume,
            "candidate_volume_residual_m3": 0.0,
            "source_candidate_mass_kg": allocated_mass,
            "allocated_candidate_mass_kg": allocated_mass,
            "candidate_mass_residual_kg": 0.0,
            "candidate_partition_is_disjoint": True,
            "candidate_is_mechanical_mass": False,
        },
        "qualification": {
            "source_route_identity_bound": True,
            "source_surface_mass_bound": True,
            "equal_incidence_candidate_partition": True,
            "candidate_mass_and_volume_close": True,
            "unbound_routes_retained": True,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "volumetric_active_force_owner": False,
        },
        "route_budgets": rows,
    }


def _body(route_count: int) -> dict:
    return {
        "schema": "HumanPack.body-composition-release-join.v1",
        "status": "partial",
        "subject": "fixture subject",
        "domain_counts": {
            "muscle_source_routes": route_count,
            "native_recruited_muscles": route_count,
        },
        "ownership": {
            "physical_owner_count": 0,
            "organ_physical_volume_owner": False,
            "mechanical_blood_mass_owner": False,
            "mechanical_tissue_mass_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "fat_mechanical_mass_owner": False,
            "whole_body_dynamic_mass_matrix_owner": False,
        },
        "qualification": {
            "cross_domain_owner_nonduplication": True,
            "muscle_route_volume_identity_bound": True,
            "integrated_human_qualification": False,
        },
    }


def _authoring(*, declarations: list | None = None,
               moment_closures: list | None = None) -> dict:
    return {
        "schema": AUTHORING_SCHEMA,
        "id": "fixture-ownership",
        "declarations": declarations or [],
        "moment_closures": moment_closures or [],
        "boundary": "Fixture authoring declarations do not promote mechanics.",
    }


def _compile(count: int = 2, *, authoring: dict | None = None) -> dict:
    routes = _routes(count)
    return compile_manifest(
        coverage=_coverage([row["name"] for row in routes["route_budgets"]]),
        body_composition=_body(count),
        muscle_routes=routes,
        muscle_route_identities=_route_identities(count),
        authoring=authoring or _authoring(),
    )


def test_current_416_routes_are_retained_without_owner_promotion() -> None:
    routes = json.loads(MUSCLE_ROUTES.read_text(encoding="utf-8"))
    body = json.loads(BODY_COMPOSITION.read_text(encoding="utf-8"))
    route_identities = json.loads(
        Path(routes["source"]["route_volume_receipt"]).read_text(encoding="utf-8")
    )
    names = [row["name"] for row in routes["route_budgets"]]
    result = compile_manifest(
        coverage=_coverage(names),
        body_composition=body,
        muscle_routes=routes,
        muscle_route_identities=route_identities,
        authoring=_authoring(),
    )
    assert result["counts"]["semantic_actions"] == 416
    assert result["counts"]["unmapped_route_records"] == 0
    assert result["counts"]["candidate_moment_records"] == 78
    assert result["qualification"]["all_source_routes_retained"]
    assert not result["qualification"]["production_physical_ownership"]
    route_records = [row for row in result["records"] if row["entity_kind"] == "muscle_route"]
    assert {row["action"]["source_index"] for row in route_records} == set(range(416))
    assert all(row["owners"][role]["status"] == "unresolved"
               for row in route_records for role in row["owners"])


def test_path_compiler_loads_and_hashes_declared_route_identity_receipt(
        tmp_path: Path) -> None:
    routes = json.loads(MUSCLE_ROUTES.read_text(encoding="utf-8"))
    names = [row["name"] for row in routes["route_budgets"]]
    coverage_path = tmp_path / "target-coverage.json"
    coverage_path.write_bytes(canonical_bytes(_coverage(names)) + b"\n")
    result = compile_paths(coverage_path=coverage_path)
    identity_input = result["inputs"]["muscle_route_identities"]
    assert identity_input["file_sha256"] == routes["source"]["route_volume_receipt_sha256"]
    assert result["counts"]["semantic_actions"] == 416


def test_double_owner_for_one_semantic_resource_is_rejected() -> None:
    semantic_id = "mandatory:exterior_and_systemic_anatomy/organs"

    def declaration(owner: str) -> dict:
        return {
            "semantic_id": semantic_id,
            "owners": {"mechanical_mass": {"status": "candidate", "owner_id": owner}},
        }

    with pytest.raises(HumanImportError, match="double-assigns mechanical_mass owner"):
        _compile(authoring=_authoring(declarations=[
            declaration("candidate:owner-a"), declaration("candidate:owner-b"),
        ]))


def test_zero_owner_join_rejects_production_promotion() -> None:
    declaration = {
        "semantic_id": "mandatory:exterior_and_systemic_anatomy/organs",
        "owners": {
            "mechanical_mass": {
                "status": "production", "owner_id": "matter:organ-mass-owner",
            },
        },
    }
    with pytest.raises(HumanImportError, match="owner status is invalid"):
        _compile(authoring=_authoring(declarations=[declaration]))


def test_body_composition_ownership_summary_rejects_schema_incompatible_values() -> None:
    body = _body(2)
    body["ownership"]["invalid_extra"] = "not-an-owner-state"
    with pytest.raises(HumanImportError, match="ownership values are invalid"):
        compile_manifest(
            coverage=_coverage(["route_0", "route_1"]),
            body_composition=body,
            muscle_routes=_routes(2),
            muscle_route_identities=_route_identities(2),
            authoring=_authoring(),
        )


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            {"topology": {"status": "production", "ids": ["runtime:topology"]}},
            "binding status is invalid",
        ),
        (
            {"moments": {
                "status": "production", "frame_id": None, "volume_m3": 1.0,
                "zeroth_mass_kg": 1.0, "first_mass_moment_kg_m": None,
                "second_mass_moment_kg_m2": None,
            }},
            "moment status is invalid",
        ),
        (
            {"force_semantics": {
                "status": "production", "mode": "replacement",
                "replaces_owner_ids": ["runtime:force-owner"],
            }},
            "force-semantics status is invalid",
        ),
        ({"qualification_status": "production"}, "qualification status is invalid"),
    ],
)
def test_v1_rejects_production_authoring_surfaces(
        declaration: dict, message: str) -> None:
    declaration = {
        "semantic_id": "mandatory:exterior_and_systemic_anatomy/organs",
        **declaration,
    }
    with pytest.raises(HumanImportError, match=message):
        _compile(authoring=_authoring(declarations=[declaration]))


def test_v1_validator_rejects_production_action() -> None:
    result = _compile()
    route = next(record for record in result["records"] if record["action"] is not None)
    route["action"]["status"] = "production"
    result["manifest_sha256"] = digest({
        key: value for key, value in result.items() if key != "manifest_sha256"
    })
    with pytest.raises(HumanImportError, match="semantic action is invalid"):
        validate_ownership(result)


def test_v1_rejects_production_moment_closure() -> None:
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_0"
    closure = {
        "id": "forbidden-production-closure",
        "status": "production",
        "member_semantic_ids": [semantic_id],
        "frame_id": None,
        "expected": {
            "volume_m3": 0.001,
            "zeroth_mass_kg": 1.0,
            "first_mass_moment_kg_m": None,
            "second_mass_moment_kg_m2": None,
        },
        "absolute_tolerance": 0.0,
    }
    with pytest.raises(HumanImportError, match="moment closure .* status is invalid"):
        _compile(authoring=_authoring(moment_closures=[closure]))


def test_route_action_indices_must_match_source_identity_receipt() -> None:
    routes = _routes(2)
    routes["route_budgets"][0]["source_actuator_index"] = 1
    routes["route_budgets"][1]["source_actuator_index"] = 0
    with pytest.raises(HumanImportError, match="differs from the source identity receipt"):
        compile_manifest(
            coverage=_coverage(["route_0", "route_1"]),
            body_composition=_body(2),
            muscle_routes=routes,
            muscle_route_identities=_route_identities(2),
            authoring=_authoring(),
        )


@pytest.mark.parametrize(
    ("table", "key", "value", "message"),
    [
        ("totals", "candidate_partition_is_disjoint", False, "partition is not disjoint"),
        ("totals", "candidate_is_mechanical_mass", True, "promotes mechanical mass"),
        (
            "qualification", "equal_incidence_candidate_partition", False,
            "identity or closure is not qualified",
        ),
        ("totals", "candidate_mass_residual_kg", 1.0, "masses do not close"),
    ],
)
def test_route_partition_admission_checks_physical_boundary_fields(
        table: str, key: str, value: object, message: str) -> None:
    routes = _routes(2)
    routes[table][key] = value
    with pytest.raises(HumanImportError, match=message):
        compile_manifest(
            coverage=_coverage(["route_0", "route_1"]),
            body_composition=_body(2),
            muscle_routes=routes,
            muscle_route_identities=_route_identities(2),
            authoring=_authoring(),
        )


def test_source_composition_conflict_is_retained_and_blocks_manifest() -> None:
    routes = _routes(1)
    coverage = _coverage(["route_0"])
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_0"
    original = next(row for row in coverage["leaves"] if row["semantic_id"] == semantic_id)
    conflicting = copy.deepcopy(original)
    conflicting["declaration_sha256"] = "b" * 64
    conflicting["leaf_sha256"] = digest({
        key: value for key, value in conflicting.items() if key != "leaf_sha256"
    })
    coverage["leaves"].append(conflicting)
    coverage["leaves"].sort(key=lambda row: row["leaf_sha256"])
    coverage["counts"]["leaves"] += 1
    coverage["counts"]["by_kind"]["composed:muscles"] += 1
    coverage["manifest_sha256"] = digest({
        key: value for key, value in coverage.items() if key != "manifest_sha256"
    })
    validate_coverage(coverage)
    result = compile_manifest(
        coverage=coverage,
        body_composition=_body(1),
        muscle_routes=routes,
        muscle_route_identities=_route_identities(1),
        authoring=_authoring(),
    )
    record = next(row for row in result["records"] if row["semantic_id"] == semantic_id)
    assert result["status"] == "blocked"
    assert result["counts"]["composition_conflicts"] == 1
    assert record["composition"]["status"] == "conflict"
    assert len(record["composition"]["conflict_ids"]) == 2
    with pytest.raises(HumanImportError, match="downgrades composition"):
        compile_manifest(
            coverage=coverage,
            body_composition=_body(1),
            muscle_routes=routes,
            muscle_route_identities=_route_identities(1),
            authoring=_authoring(declarations=[{
                "semantic_id": semantic_id,
                "composition": {
                    "status": "single_source",
                    "component_semantic_ids": [],
                    "conflict_ids": [],
                },
            }]),
        )


def test_blocked_source_coverage_propagates_to_ownership_status() -> None:
    coverage = _coverage(["route_0"])
    coverage["registers"] = [{
        "source_record_sha256": coverage["source_records"][0]["record_sha256"],
        "path": "missing-source-register.xml",
        "expected_sha256": "a" * 64,
        "actual_sha256": None,
        "status": "missing",
        "declaration_count": 0,
    }]
    coverage["counts"]["unresolved_current_registers"] = 1
    coverage["scope_status"] = "blocked"
    coverage["manifest_sha256"] = digest({
        key: value for key, value in coverage.items() if key != "manifest_sha256"
    })
    validate_coverage(coverage)
    result = compile_manifest(
        coverage=coverage,
        body_composition=_body(1),
        muscle_routes=_routes(1),
        muscle_route_identities=_route_identities(1),
        authoring=_authoring(),
    )
    assert result["status"] == "blocked"
    validate_ownership(result)


def test_validator_rejects_missing_represented_coverage_leaf() -> None:
    result = _compile()
    result["records"] = [
        record for record in result["records"]
        if record["semantic_id"] != "mandatory:exterior_and_systemic_anatomy/organs"
    ]
    result["counts"] = _counts(result["records"], result["moment_closures"])
    result["manifest_sha256"] = digest({
        key: value for key, value in result.items() if key != "manifest_sha256"
    })
    with pytest.raises(HumanImportError, match="represented coverage leaf count differs"):
        validate_ownership(result)


@pytest.mark.parametrize("field", ["topology", "fields"])
def test_declarations_cannot_downgrade_route_source_bindings(field: str) -> None:
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_0"
    with pytest.raises(HumanImportError, match=f"downgrades {field}"):
        _compile(authoring=_authoring(declarations=[{
            "semantic_id": semantic_id,
            field: {"status": "unresolved", "ids": []},
        }]))


def test_declarations_cannot_erase_route_source_binding_ids() -> None:
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_0"
    with pytest.raises(HumanImportError, match="erases fields source IDs"):
        _compile(authoring=_authoring(declarations=[{
            "semantic_id": semantic_id,
            "fields": {
                "status": "candidate",
                "ids": [semantic_id + "/field/activation"],
            },
        }]))


def test_declarations_cannot_downgrade_qualification_facts() -> None:
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_0"
    with pytest.raises(HumanImportError, match="downgrades qualification status"):
        _compile(authoring=_authoring(declarations=[{
            "semantic_id": semantic_id,
            "qualification_status": "source_only",
        }]))


def test_declarations_can_fill_unresolved_slots_monotonically() -> None:
    semantic_id = "mandatory:exterior_and_systemic_anatomy/organs"
    result = _compile(authoring=_authoring(declarations=[{
        "semantic_id": semantic_id,
        "topology": {"status": "candidate", "ids": ["candidate:organ-topology"]},
        "fields": {"status": "source", "ids": ["source:organ-field"]},
        "motor_compartments": {
            "status": "candidate", "ids": ["candidate:organ-motor-compartment"],
        },
        "qualification_status": "candidate",
    }]))
    record = next(row for row in result["records"] if row["semantic_id"] == semantic_id)
    assert record["topology"] == {
        "status": "candidate", "ids": ["candidate:organ-topology"],
    }
    assert record["fields"] == {"status": "source", "ids": ["source:organ-field"]}
    assert record["motor_compartments"] == {
        "status": "candidate", "ids": ["candidate:organ-motor-compartment"],
    }
    assert record["qualification_status"] == "candidate"


def test_unresolved_owner_motor_and_spatial_moments_are_preserved() -> None:
    result = _compile()
    route = next(row for row in result["records"]
                 if row["semantic_id"].endswith("/route_0"))
    assert route["motor_compartments"] == {"status": "unresolved", "ids": []}
    assert route["moments"]["status"] == "candidate"
    assert route["moments"]["zeroth_mass_kg"] == 1.0
    assert route["moments"]["first_mass_moment_kg_m"] is None
    assert route["moments"]["second_mass_moment_kg_m2"] is None
    assert route["force_semantics"]["mode"] == "replacement"
    assert all(value == {"status": "unresolved", "owner_id": None}
               for value in route["owners"].values())


@pytest.mark.parametrize(
    ("moments", "message"),
    [
        (
            {
                "status": "candidate", "frame_id": "human-frame",
                "volume_m3": 0.0, "zeroth_mass_kg": 0.0,
                "first_mass_moment_kg_m": [1.0, 0.0, 0.0],
                "second_mass_moment_kg_m2": None,
            },
            "zero mass carries a nonzero first moment",
        ),
        (
            {
                "status": "candidate", "frame_id": "human-frame",
                "volume_m3": 1.0, "zeroth_mass_kg": 1.0,
                "first_mass_moment_kg_m": None,
                "second_mass_moment_kg_m2": [
                    [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                ],
            },
            "negative principal moment",
        ),
        (
            {
                "status": "candidate", "frame_id": "human-frame",
                "volume_m3": 1.0, "zeroth_mass_kg": 1.0,
                "first_mass_moment_kg_m": [2.0, 0.0, 0.0],
                "second_mass_moment_kg_m2": [
                    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                ],
            },
            "centered second mass moment has a negative principal moment",
        ),
    ],
)
def test_spatial_mass_moments_must_be_physically_feasible(
        moments: dict, message: str) -> None:
    declaration = {
        "semantic_id": "mandatory:exterior_and_systemic_anatomy/organs",
        "moments": moments,
    }
    with pytest.raises(HumanImportError, match=message):
        _compile(authoring=_authoring(declarations=[declaration]))


def test_force_replacement_must_name_an_admitted_owner() -> None:
    semantic_id = "mandatory:exterior_and_systemic_anatomy/organs"
    with pytest.raises(HumanImportError, match="replace an unknown owner"):
        _compile(authoring=_authoring(declarations=[{
            "semantic_id": semantic_id,
            "force_semantics": {
                "status": "candidate", "mode": "replacement",
                "replaces_owner_ids": ["missing:force-owner"],
            },
        }]))


def test_force_replacement_can_name_a_declared_candidate_owner() -> None:
    semantic_id = "mandatory:exterior_and_systemic_anatomy/organs"
    owner_id = "candidate:organ-force-owner"
    result = _compile(authoring=_authoring(declarations=[{
        "semantic_id": semantic_id,
        "owners": {
            "active_force": {"status": "candidate", "owner_id": owner_id},
        },
        "force_semantics": {
            "status": "candidate", "mode": "replacement",
            "replaces_owner_ids": [owner_id],
        },
    }]))
    record = next(row for row in result["records"] if row["semantic_id"] == semantic_id)
    assert record["force_semantics"]["replaces_owner_ids"] == [owner_id]


def test_force_replacement_cannot_name_a_non_force_owner() -> None:
    semantic_id = "mandatory:exterior_and_systemic_anatomy/organs"
    owner_id = "candidate:organ-mass-owner"
    with pytest.raises(HumanImportError, match="replace an unknown owner"):
        _compile(authoring=_authoring(declarations=[{
            "semantic_id": semantic_id,
            "owners": {
                "mechanical_mass": {"status": "candidate", "owner_id": owner_id},
            },
            "force_semantics": {
                "status": "candidate", "mode": "replacement",
                "replaces_owner_ids": [owner_id],
            },
        }]))


def test_manifest_is_deterministic_hash_bound_and_immutable(tmp_path: Path) -> None:
    first = _compile()
    second = _compile()
    assert canonical_bytes(first) == canonical_bytes(second)
    validate_ownership(first)
    path = tmp_path / "ownership.json"
    assert _immutable_write(path, first) == _immutable_write(path, second)
    forged = copy.deepcopy(first)
    forged["records"][0]["qualification_status"] = "production"
    with pytest.raises(HumanImportError, match="hash mismatch"):
        validate_ownership(forged)


@pytest.mark.parametrize("target_exists", [False, True])
def test_cli_rejects_dangling_and_existing_output_symlinks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_exists: bool) -> None:
    result = _compile()
    target = tmp_path / "redirected-target.json"
    if target_exists:
        target.write_bytes(canonical_bytes(result) + b"\n")
    output = tmp_path / "ownership.json"
    output.symlink_to(target.name)
    monkeypatch.setattr(ownership_module, "compile_paths", lambda **_arguments: result)
    arguments = SimpleNamespace(
        coverage=Path("unused-coverage.json"),
        body_composition=Path("unused-body.json"),
        muscle_routes=Path("unused-routes.json"),
        authoring=Path("unused-authoring.json"),
        output=output,
    )
    with pytest.raises(HumanImportError, match="output is redirected"):
        ownership_module.run(arguments)
    assert output.is_symlink()
    assert target.exists() is target_exists


def test_action_surface_has_no_fixed_416_ceiling() -> None:
    result = _compile(417)
    actions = [row["action"] for row in result["records"] if row["action"] is not None]
    assert len(actions) == 417
    assert {row["source_index"] for row in actions} == set(range(417))
    assert next(row for row in actions if row["source_index"] == 416)["semantic_action_id"].endswith(
        "/route_416/action/stimulation"
    )


def test_zeroth_first_and_second_moments_close_in_one_frame() -> None:
    first = "mandatory:exterior_and_systemic_anatomy/organs"
    second = "mandatory:exterior_and_systemic_anatomy/vessels"
    matrix_a = [[2.0, 2.0, 3.0], [2.0, 6.0, 6.0], [3.0, 6.0, 12.0]]
    matrix_b = [[12.0, 10.0, 12.0], [10.0, 17.5, 15.0], [12.0, 15.0, 24.0]]
    declarations = [
        {
            "semantic_id": first,
            "moments": {
                "status": "candidate", "frame_id": "human-frame",
                "volume_m3": 0.01, "zeroth_mass_kg": 1.0,
                "first_mass_moment_kg_m": [1.0, 2.0, 3.0],
                "second_mass_moment_kg_m2": matrix_a,
            },
            "qualification_status": "candidate",
        },
        {
            "semantic_id": second,
            "moments": {
                "status": "candidate", "frame_id": "human-frame",
                "volume_m3": 0.02, "zeroth_mass_kg": 2.0,
                "first_mass_moment_kg_m": [4.0, 5.0, 6.0],
                "second_mass_moment_kg_m2": matrix_b,
            },
            "qualification_status": "candidate",
        },
    ]
    closure = {
        "id": "organ-vessel-candidate-moments",
        "status": "candidate",
        "member_semantic_ids": sorted([first, second]),
        "frame_id": "human-frame",
        "expected": {
            "volume_m3": 0.03,
            "zeroth_mass_kg": 3.0,
            "first_mass_moment_kg_m": [5.0, 7.0, 9.0],
                "second_mass_moment_kg_m2": [
                    [14.0, 12.0, 15.0], [12.0, 23.5, 21.0], [15.0, 21.0, 36.0],
                ],
        },
        "absolute_tolerance": 1.0e-15,
    }
    result = _compile(authoring=_authoring(
        declarations=declarations, moment_closures=[closure],
    ))
    receipt = next(row for row in result["moment_closures"]
                   if row["id"] == closure["id"])
    assert receipt["actual"]["volume_m3"] == pytest.approx(closure["expected"]["volume_m3"])
    assert receipt["actual"]["zeroth_mass_kg"] == closure["expected"]["zeroth_mass_kg"]
    assert receipt["actual"]["first_mass_moment_kg_m"] == closure["expected"]["first_mass_moment_kg_m"]
    for actual, expected in zip(
            receipt["actual"]["second_mass_moment_kg_m2"],
            closure["expected"]["second_mass_moment_kg_m2"], strict=True):
        assert actual == pytest.approx(expected)
    assert all(value == 0.0 for value in receipt["residual"]["first_mass_moment_kg_m"])
    assert all(abs(value) <= closure["absolute_tolerance"]
               for row in receipt["residual"]["second_mass_moment_kg_m2"]
               for value in row)


def test_semantic_ids_remain_stable_when_the_action_surface_extends() -> None:
    before = _compile(2)
    after = _compile(3)
    semantic_id = "myosim_fullbody:composed/myofullbody/muscles/route_1"
    old = next(row for row in before["records"] if row["semantic_id"] == semantic_id)
    new = next(row for row in after["records"] if row["semantic_id"] == semantic_id)
    assert old["semantic_id"] == new["semantic_id"]
    assert old["action"] == new["action"]
    assert old["topology"] == new["topology"]


def test_schema_and_workspace_commands_are_exposed() -> None:
    schema = json.loads(Path("schemas/humanpack-ownership.v1.schema.json").read_text())
    assert schema["properties"]["schema"]["const"] == "HumanPack.ownership.v1"
    standalone = Path(".numi/commands/human-ownership-compile").read_text()
    overlay = Path(".numi/commands/human").read_text()
    assert "numilab_human.ownership" in standalone
    assert '"ownership-compile"' in overlay


def test_schema_rejects_states_rejected_by_authoritative_validator() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("schemas/humanpack-ownership.v1.schema.json").read_text())
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    validator = validator_type(schema)
    result = _compile()
    validator.validate(result)

    unresolved_binding_with_id = copy.deepcopy(result)
    unresolved_binding_with_id["records"][0]["motor_compartments"] = {
        "status": "unresolved", "ids": ["invalid:carried-id"],
    }
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(unresolved_binding_with_id)

    candidate_owner_without_id = copy.deepcopy(result)
    candidate_owner_without_id["records"][0]["owners"]["mechanical_mass"] = {
        "status": "candidate", "owner_id": None,
    }
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(candidate_owner_without_id)

    production_owner = copy.deepcopy(result)
    production_owner["records"][0]["owners"]["mechanical_mass"] = {
        "status": "production", "owner_id": "matter:mass-owner",
    }
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(production_owner)
