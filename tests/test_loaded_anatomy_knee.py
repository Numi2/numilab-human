from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

import numilab_human.loaded_anatomy_knee as knee
from numilab_human.target_coverage import canonical_bytes, digest


def _profile() -> dict:
    return knee._load_frozen_profile()


def _topology(profile: dict) -> dict:
    identity = profile["topology_identity"]
    spans = [
        {
            key: row[key]
            for key in (
                "name",
                "first_node",
                "node_count",
                "first_tetrahedron",
                "tetrahedron_count",
                "runtime_first_node",
                "runtime_node_count",
                "runtime_first_tetrahedron",
                "runtime_tetrahedron_count",
            )
        }
        for row in identity["regions"]
    ]
    return {
        **profile["topology"],
        "identity_sha256": identity["identity_sha256"],
        "executable_fem_topology_sha256": identity["executable_fem_topology_sha256"],
        "executable_fem_topology_encoding": "uint32-le-tetrahedra-profile-region-order-source-element-order-"
        "concatenated-local-node-indices",
        "source_global_node_index_sha256": identity["source_global_node_index_sha256"],
        "source_global_node_index_encoding": "uint32-le-source-global-node-indices-profile-region-order",
        "anchor_ownership_sha256": identity["anchor_ownership_sha256"],
        "anchor_ownership_encoding": "uint32-le-source-global-node,anchor-body,flags;float32-le-anchor-local-xyz",
        "region_spans": spans,
        "ordering": "authoring-profile-region-order-then-source-local-order",
    }


def _region_moments(profile: dict) -> list[dict]:
    result = []
    for index, name in enumerate(knee.REGION_NAMES, start=1):
        mass = 0.01 * index
        result.append(
            {
                "name": name,
                "volume_m3": mass / 1000.0,
                "zeroth_mass_kg": mass,
                "first_mass_moment_kg_m": [0.0, 0.0, 0.0],
                "raw_second_mass_moment_kg_m2": [
                    [0.01 * index, 0.0, 0.0],
                    [0.0, 0.01 * index, 0.0],
                    [0.0, 0.0, 0.01 * index],
                ],
                "minimum_jacobian_determinant": 1.0,
                "maximum_jacobian_determinant": 1.0,
                "raw_f32_node_mass_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
        )
    return result


def _lab_export(
    profile: dict, *, x_ref_sha: str, x_source_sha: str, node_mass_sha: str
) -> dict:
    topology = _topology(profile)
    donors = []
    for donor in profile["donor_policy"]["donors"]:
        donors.append(
            {
                "semantic_id": donor["semantic_id"],
                "source_body_id": donor["source_body_id"],
                "core_body_index": donor["core_body_index"],
                "moments": {
                    "zeroth_mass_kg": knee._float32(donor["source_mass_kg"]),
                    "first_mass_moment_kg_m": [0.0, 0.0, 0.0],
                    "raw_second_mass_moment_kg_m2": [
                        [10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0],
                    ],
                },
            }
        )
    result = {
        "schema": knee.LAB_EXPORT_SCHEMA,
        "status": "candidate",
        "manifest_canonicalization": knee.CANONICALIZATION,
        "manifest_hash_exclusion": knee.HASH_EXCLUSION,
        "manifest_sha256": "",
        "subject_id": profile["subject_id"],
        "side": "left",
        "x_ref": {
            "bytes": 62402 * 12,
            "file_sha256": x_ref_sha,
            "encoding": "float32-le-xyz",
            "node_count": 62402,
            "region_order": list(knee.REGION_NAMES),
            "local_order": "source-local-node-order",
            "region_spans": topology["region_spans"],
            "source_global_node_index_sha256": topology[
                "source_global_node_index_sha256"
            ],
            "source_global_node_index_encoding": topology[
                "source_global_node_index_encoding"
            ],
            "executable_fem_topology_sha256": topology[
                "executable_fem_topology_sha256"
            ],
            "executable_fem_topology_encoding": topology[
                "executable_fem_topology_encoding"
            ],
            "anchor_ownership_sha256": topology["anchor_ownership_sha256"],
            "anchor_ownership_encoding": topology["anchor_ownership_encoding"],
        },
        "source": {
            "nhknee_sha256": profile["source"]["nhknee"]["sha256"],
            "x_source_sha256": x_source_sha,
            "source_rigid_payload_sha256": "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44",
            "equality_payload_sha256": "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a",
            "source_model_fingerprint_sha256": "c" * 64,
        },
        "poses": {
            "source_default": {
                "id": profile["reference_state"]["source_body_pose_id"],
                "identity_sha256": "d" * 64,
            },
            "projected_reference": {
                "id": profile["reference_state"]["reference_body_pose_id"],
                "identity_sha256": "e" * 64,
            },
        },
        "mapping": {
            "id": profile["reference_state"]["source_to_reference_mapping_id"],
            "algorithm": knee.LAB_MAPPING_ALGORITHM,
            "code_identity_sha256": "f" * 64,
            "code_identity_encoding": knee.LAB_MAPPING_CODE_IDENTITY_ENCODING,
            "diagnostics": {
                "finite": True,
                "source_node_count": 62402,
                "output_node_count": 62402,
                "maximum_displacement_m": 0.02,
                "equality_residual_maximum": 0.0,
                "jacobian": {
                    "finite": True,
                    "minimum_determinant": 1.0,
                    "maximum_determinant": 1.0,
                    "orientation_preserving": True,
                },
                "raw_f32_node_mass_sha256": node_mass_sha,
                "raw_f32_node_mass_algorithm": knee.RAW_F32_NODE_MASS_ALGORITHM,
                "regions": [
                    {
                        "name": name,
                        "substep_count": 2 if name == "PTL" else 1,
                        "direct_map_status": (
                            "rejected_inversion" if name == "PTL" else "accepted"
                        ),
                        "direct_failure_tetrahedron": 419 if name == "PTL" else None,
                        "direct_failure_jacobian": (
                            -0.20709127479457548 if name == "PTL" else None
                        ),
                        "minimum_jacobian": 1.0,
                        "maximum_jacobian": 1.0,
                        "maximum_source_anchor_reconstruction_residual_m": 1.0e-8,
                        "maximum_anchor_residual_m": 1.0e-8,
                        "maximum_persisted_f32_anchor_residual_m": 2.0e-8,
                    }
                    for name in knee.REGION_NAMES
                ],
            },
        },
        "donor_moments": {
            "policy_id": profile["donor_policy"]["id"],
            "frame_id": profile["donor_policy"]["frame_id"],
            "source_kind": "cooked-myosim-rigid-body-mass-properties",
            "donors": donors,
        },
        "boundary": knee.LAB_EXPORT_BOUNDARY,
    }
    result["manifest_sha256"] = digest(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )
    return result


def _manifest() -> dict:
    profile = _profile()
    x_ref_sha = "a" * 64
    x_source_sha = "b" * 64
    node_mass_sha = "9" * 64
    region_moments = _region_moments(profile)
    lab_export = _lab_export(
        profile,
        x_ref_sha=x_ref_sha,
        x_source_sha=x_source_sha,
        node_mass_sha=node_mass_sha,
    )
    decoded = {
        "regions": profile["topology_identity"]["regions"],
        "source_global_node_index_sha256": profile["topology_identity"][
            "source_global_node_index_sha256"
        ],
        "executable_fem_topology_sha256": profile["topology_identity"][
            "executable_fem_topology_sha256"
        ],
        "anchor_ownership_sha256": profile["topology_identity"][
            "anchor_ownership_sha256"
        ],
    }
    geometry = {
        "maximum_displacement_m": 0.02,
        "minimum_jacobian_determinant": 1.0,
        "maximum_jacobian_determinant": 1.0,
        "raw_f32_node_mass_sha256": node_mass_sha,
    }
    mass = knee._compile_mass_partition(
        lab_export,
        profile,
        region_moments,
        x_ref_sha,
        x_source_sha,
        decoded,
        geometry,
    )
    topology = _topology(profile)
    mapping_identity = digest(
        {
            "x_source_sha256": x_source_sha,
            "source_frame_id": profile["reference_state"]["source_frame_id"],
            "source_body_pose_id": profile["reference_state"]["source_body_pose_id"],
            "source_registration_id": profile["reference_state"][
                "source_registration_id"
            ],
            "x_ref_sha256": x_ref_sha,
            "construction_id": profile["reference_state"]["construction_id"],
            "reference_body_pose_id": profile["reference_state"][
                "reference_body_pose_id"
            ],
            "source_to_reference_mapping_id": profile["reference_state"][
                "source_to_reference_mapping_id"
            ],
        }
    )
    regions = []
    topology_by_name = {
        row["name"]: row for row in profile["topology_identity"]["regions"]
    }
    moment_by_name = {row["name"]: row for row in region_moments}
    for specification in profile["regions"]:
        name = specification["name"]
        active = (
            {
                "status": "candidate",
                "owner_id": specification["owners"]["active_force_owner_id"],
            }
            if name == "QAT"
            else {"status": "none", "owner_id": None}
        )
        regions.append(
            {
                "name": name,
                "topology_semantic_id": specification["topology_semantic_id"],
                "material_semantic_ids": specification["material_semantic_ids"],
                "payload": topology_by_name[name],
                "material": {
                    **specification["material"],
                    "fiber_world": specification["fiber_world"],
                    "calibration_status": "source_population_prior",
                },
                "owners": {
                    "physical_volume": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["physical_volume_owner_id"],
                    },
                    "mechanical_mass": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["mechanical_mass_owner_id"],
                    },
                    "material": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["material_owner_id"],
                    },
                    "active_force": active,
                    "state": {
                        "status": "candidate",
                        "owner_id": specification["owners"]["state_owner_id"],
                    },
                },
                "donor_body_semantic_id": specification["donor_body_semantic_id"],
                "projected_reference_moments": moment_by_name[name],
            }
        )
    pairs = []
    for pair in profile["articular_contact_pairs"]:
        pairs.append(
            {
                **pair,
                "semantic_id": "open_knee_oks003:Geometry.feb#/febio_spec[1]/Geometry[1]/"
                f"SurfacePair[name={pair['name']}][1]",
                "state_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/contact/{pair['name']}/state",
                },
            }
        )
    replacements = []
    for route in profile["quadriceps_routes"]:
        semantic_id = f"myosim_fullbody:composed/myofullbody/muscles/{route['name']}"
        replacements.append(
            {
                "name": route["name"],
                "source_actuator_index": route["source_actuator_index"],
                "route_semantic_id": semantic_id,
                "semantic_action_id": semantic_id + "/action/stimulation",
                "replaces_owner_id": semantic_id + "/owner/source-jt",
                "replacement_owner_id": "humanpack:loaded-anatomy-knee:left/QAT/active-force",
                "replacement_scale": 1.0,
                "status": "candidate",
                "endpoints": copy.deepcopy(
                    knee.EXPECTED_ROUTE_ENDPOINTS[route["name"]]
                ),
            }
        )
    ownership_identity = "1" * 64
    result = {
        "schema": knee.SCHEMA,
        "compiler": knee.COMPILER,
        "manifest_canonicalization": knee.CANONICALIZATION,
        "manifest_hash_exclusion": knee.HASH_EXCLUSION,
        "manifest_sha256": "",
        "status": "candidate",
        "source_ownership_status": "blocked",
        "subject_id": profile["subject_id"],
        "side": "left",
        "ownership_manifest_sha256": ownership_identity,
        "inputs": {
            "ownership": {
                "schema": "HumanPack.ownership.v1",
                "file_sha256": "2" * 64,
                "identity_sha256": ownership_identity,
            },
            "authoring_profile": {
                "schema": knee.PROFILE_SCHEMA,
                "file_sha256": knee.FROZEN_PROFILE_FILE_SHA256,
                "identity_sha256": knee.FROZEN_PROFILE_IDENTITY_SHA256,
            },
            "open_knee_payload": {
                "schema": "numi.human.open-knee-oks003-payload.v3",
                "file_sha256": profile["source"]["nhknee"]["sha256"],
                "identity_sha256": profile["source"]["nhknee"]["sha256"],
            },
            "tendon_payload": {
                "schema": profile["tendon_payload"]["schema"],
                "file_sha256": profile["tendon_payload"]["sha256"],
                "identity_sha256": profile["tendon_payload"]["sha256"],
            },
            "x_ref": {
                "schema": "numi.human.loaded-anatomy-knee-x-ref-f32le.v1",
                "file_sha256": x_ref_sha,
                "identity_sha256": x_ref_sha,
            },
            "lab_export": {
                "schema": knee.LAB_EXPORT_SCHEMA,
                "file_sha256": hashlib.sha256(
                    canonical_bytes(lab_export) + b"\n"
                ).hexdigest(),
                "identity_sha256": lab_export["manifest_sha256"],
            },
        },
        "semantic_scope": {
            "ownership_manifest_sha256": ownership_identity,
            "coverage_leaf_sha256s": ["3" * 64],
        },
        "lab_authoring_export": lab_export,
        "source": {
            **profile["source"],
            "tendon_payload": profile["tendon_payload"],
            "material_and_density_status": "source_population_prior",
        },
        "topology": topology,
        "coordinates": {
            "x_source": {
                "scope": "six-loaded-regions",
                "encoding": "float32-le-xyz",
                "node_count": 62402,
                "sha256": x_source_sha,
                "frame_id": profile["reference_state"]["source_frame_id"],
                "body_pose_id": profile["reference_state"]["source_body_pose_id"],
                "registration_id": profile["reference_state"]["source_registration_id"],
            },
            "x_ref": {
                "scope": "six-loaded-regions",
                "encoding": "float32-le-xyz",
                "node_count": 62402,
                "sha256": x_ref_sha,
                "mapping_identity_sha256": mapping_identity,
                **profile["reference_state"],
            },
            "x_current": {
                "scope": "six-loaded-regions-in-authoring-profile-order",
                "encoding": "float32-le-xyz",
                "hash_algorithm": "sha256",
                "accepted_sha256": None,
                "acceptance_receipt_sha256": None,
                "authority": "separate-lab-runtime-acceptance-receipt",
            },
        },
        "density_conversion": profile["density_conversion"],
        "regions": regions,
        "mass_partition": mass,
        "articular_contact_pairs": pairs,
        "active_force_replacements": replacements,
        "passive_ligament_owners": [
            {
                "name": name,
                "semantic_id": f"humanpack:loaded-anatomy-knee:left/region/{name}/passive-force",
                "active_force": {"status": "none", "owner_id": None},
                "passive_force_owner": {
                    "status": "candidate",
                    "owner_id": f"humanpack:loaded-anatomy-knee:left/region/{name}/material",
                },
            }
            for name in knee.PASSIVE_NAMES
        ],
        "full_state_authority": {
            **profile["full_state_authority"],
            "identity_sha256": digest(profile["full_state_authority"]),
        },
        "qualification": {
            "candidate_only": True,
            "source_identity_bound": True,
            "ownership_identity_bound": True,
            "topology_identity_bound": True,
            "projected_reference_identity_bound": True,
            "unloaded_reference_qualified": False,
            "prestrain_reference_reset_executed": False,
            "subject_material_calibrated": False,
            "donor_mass_and_raw_moments_subtracted": True,
            "mesh_convergence_qualified": False,
            "specimen_load_validation_qualified": False,
            "clinical_validity_qualified": False,
            "production_physical_ownership": False,
            "production_active_force": False,
            "runtime_x_current_accepted": False,
            "integrated_human_qualification": False,
        },
        "boundary": knee.BOUNDARY,
    }
    result["manifest_sha256"] = digest(
        {key: value for key, value in result.items() if key != "manifest_sha256"}
    )
    return result


def _rehash(value: dict) -> None:
    value["manifest_sha256"] = digest(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )


def _rehash_embedded_lab_export(value: dict) -> None:
    export = value["lab_authoring_export"]
    _rehash(export)
    value["inputs"]["lab_export"] = {
        "schema": knee.LAB_EXPORT_SCHEMA,
        "file_sha256": hashlib.sha256(canonical_bytes(export) + b"\n").hexdigest(),
        "identity_sha256": export["manifest_sha256"],
    }
    value["mass_partition"]["provenance"]["lab_export_manifest_sha256"] = export[
        "manifest_sha256"
    ]
    value["mass_partition"]["identity_sha256"] = digest(
        {
            key: item
            for key, item in value["mass_partition"].items()
            if key != "identity_sha256"
        }
    )
    _rehash(value)


def _sync_export_mapping_and_rehash(value: dict) -> None:
    value["mass_partition"]["provenance"]["mapping"] = copy.deepcopy(
        value["lab_authoring_export"]["mapping"]
    )
    _rehash_embedded_lab_export(value)


def _sync_export_provenance_and_rehash(value: dict) -> None:
    export = value["lab_authoring_export"]
    provenance = value["mass_partition"]["provenance"]
    for key, item in export["source"].items():
        provenance[key] = item
    provenance["poses"] = copy.deepcopy(export["poses"])
    provenance["mapping"] = copy.deepcopy(export["mapping"])
    _rehash_embedded_lab_export(value)


def _source_compliant_nheq2_manifest() -> dict:
    value = _manifest()
    export = value["lab_authoring_export"]
    export["source"]["equality_payload_sha256"] = (
        knee.SOURCE_COMPLIANT_NHEQ2_RUNTIME_EQUALITY_PAYLOAD_SHA256
    )
    export["source"]["equality_payload_role"] = (
        knee.SOURCE_COMPLIANT_EQUALITY_PAYLOAD_ROLE
    )
    export["source"]["equality_projection_applied"] = False
    export["x_ref"]["file_sha256"] = export["source"]["x_source_sha256"]
    export["poses"]["projected_reference"] = copy.deepcopy(
        export["poses"]["source_default"]
    )
    export["mapping"]["id"] = knee.SOURCE_DEFAULT_LAB_MAPPING_ID
    export["mapping"]["algorithm"] = knee.SOURCE_DEFAULT_LAB_MAPPING_ALGORITHM
    diagnostics = export["mapping"]["diagnostics"]
    diagnostics["maximum_displacement_m"] = 0.0
    diagnostics["equality_residual_maximum"] = (
        knee.SOURCE_DEFAULT_NHEQ2_DIAGNOSTIC_RESIDUAL_MAXIMUM
    )
    ptl = next(
        row
        for row in diagnostics["regions"]
        if row["name"] == "PTL"
    )
    ptl["substep_count"] = 1
    ptl["direct_map_status"] = "accepted"
    ptl["direct_failure_tetrahedron"] = None
    ptl["direct_failure_jacobian"] = None
    provenance = value["mass_partition"]["provenance"]
    for key, item in export["source"].items():
        provenance[key] = item
    provenance["x_ref_sha256"] = export["source"]["x_source_sha256"]
    provenance["poses"] = copy.deepcopy(export["poses"])
    provenance["mapping"] = copy.deepcopy(export["mapping"])
    value["inputs"]["authoring_profile"] = {
        "schema": knee.SOURCE_DEFAULT_PROFILE_SCHEMA,
        "file_sha256": knee.FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256,
        "identity_sha256": knee.FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256,
    }
    value["inputs"]["x_ref"]["file_sha256"] = export["source"][
        "x_source_sha256"
    ]
    value["inputs"]["x_ref"]["identity_sha256"] = export["source"][
        "x_source_sha256"
    ]
    value["coordinates"]["x_ref"].update(
        knee._load_frozen_source_default_profile()["reference_state"]
    )
    value["coordinates"]["x_ref"]["sha256"] = export["source"][
        "x_source_sha256"
    ]
    value["coordinates"]["x_ref"]["mapping_identity_sha256"] = digest(
        {
            "x_source_sha256": value["coordinates"]["x_source"]["sha256"],
            "source_frame_id": value["coordinates"]["x_source"]["frame_id"],
            "source_body_pose_id": value["coordinates"]["x_source"][
                "body_pose_id"
            ],
            "source_registration_id": value["coordinates"]["x_source"][
                "registration_id"
            ],
            "x_ref_sha256": value["coordinates"]["x_ref"]["sha256"],
            "construction_id": value["coordinates"]["x_ref"]["construction_id"],
            "reference_body_pose_id": value["coordinates"]["x_ref"][
                "reference_body_pose_id"
            ],
            "source_to_reference_mapping_id": value["coordinates"]["x_ref"][
                "source_to_reference_mapping_id"
            ],
        }
    )
    export["boundary"] = knee.SOURCE_DEFAULT_LAB_EXPORT_BOUNDARY
    value["boundary"] = knee.SOURCE_DEFAULT_BOUNDARY
    _rehash_embedded_lab_export(value)
    return value


def test_checked_in_profile_and_real_source_decoders_when_available() -> None:
    profile = _profile()
    profile_bytes = knee.PROFILE.read_bytes()
    assert profile_bytes == canonical_bytes(profile) + b"\n"
    assert hashlib.sha256(profile_bytes).hexdigest() == knee.FROZEN_PROFILE_FILE_SHA256
    source_default_profile = knee._load_frozen_source_default_profile()
    source_default_profile_bytes = knee.SOURCE_DEFAULT_PROFILE.read_bytes()
    assert source_default_profile_bytes == canonical_bytes(source_default_profile) + b"\n"
    assert hashlib.sha256(source_default_profile_bytes).hexdigest() == (
        knee.FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256
    )
    assert digest(source_default_profile) == (
        knee.FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256
    )
    assert source_default_profile["reference_state"] == {
        **profile["reference_state"],
        "reference_state_class": "source-default-registered-reference",
        "construction_id": knee.SOURCE_DEFAULT_LAB_MAPPING_ID,
        "reference_body_pose_id": profile["reference_state"]["source_body_pose_id"],
        "source_to_reference_mapping_id": knee.SOURCE_DEFAULT_LAB_MAPPING_ID,
    }
    assert source_default_profile["reference_state"][
        "unloaded_reference_qualified"
    ] is False
    expected_pair_order = [
        "TBC-L_To_FMC",
        "TBC-L_To_MNS-L",
        "PTC_To_FMC",
        "MNS-L_To_FMC",
        "MNS-M_To_TBC-M",
        "MNS-M_To_FMC",
        "TBC-M_To_FMC",
    ]
    assert list(knee.PAIR_NAMES) == expected_pair_order
    assert [row["name"] for row in profile["articular_contact_pairs"]] == (
        expected_pair_order
    )
    assert profile["topology_identity"]["executable_fem_topology_sha256"].startswith(
        "56af"
    )
    payload = Path("/tmp/open-knee-oks003-left-v4.nhknee")
    tendon = Path("/tmp/numi-human-tendon-attachments-current.nhtendon")
    if not payload.is_file() or not tendon.is_file():
        pytest.skip("real ABI3 and NHTENDON3 payload copies are not present in /tmp")
    decoded = knee._decode_payload(payload.read_bytes(), profile)
    endpoints, identity = knee._decode_tendon(tendon.read_bytes(), profile)
    assert (
        decoded["executable_fem_topology_sha256"]
        == "56af24e63c9a4c2ecafb2c94bab98fa95bbcbc77085f425c84b9c1a95e3a9040"
    )
    assert [row["name"] for row in endpoints] == list(knee.ROUTE_NAMES)
    assert [row["name"] for row in decoded["pairs"]] == expected_pair_order
    assert [row["index"] for row in decoded["pairs"]] == [2, 3, 8, 14, 15, 16, 18]
    assert identity["sha256"] == profile["tendon_payload"]["sha256"]


def test_articular_pair_manifest_order_is_fail_closed() -> None:
    expected = [
        "TBC-L_To_FMC",
        "TBC-L_To_MNS-L",
        "PTC_To_FMC",
        "MNS-L_To_FMC",
        "MNS-M_To_TBC-M",
        "MNS-M_To_FMC",
        "TBC-M_To_FMC",
    ]
    value = _manifest()
    assert [row["name"] for row in value["articular_contact_pairs"]] == expected

    value["articular_contact_pairs"][2], value["articular_contact_pairs"][3] = (
        value["articular_contact_pairs"][3],
        value["articular_contact_pairs"][2],
    )
    _rehash(value)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="order differs"):
        knee.validate_manifest(value)


def test_decoder_derives_articular_pair_order_from_abi3_table() -> None:
    profile = _profile()
    expected_by_name = {row["name"]: row for row in profile["articular_contact_pairs"]}
    source_name_by_index = dict(zip(knee.PAIR_SOURCE_INDICES, knee.PAIR_NAMES))
    pairs = []
    for index in range(19):
        if index in source_name_by_index:
            pairs.append(
                {
                    "index": index,
                    **copy.deepcopy(expected_by_name[source_name_by_index[index]]),
                }
            )
        else:
            pairs.append(
                {
                    "index": index,
                    "name": f"non-articular-{index}",
                    "master_surface": f"master-{index}",
                    "slave_surface": f"slave-{index}",
                }
            )

    selected = knee._select_articular_pairs(pairs, profile)
    assert [row["name"] for row in selected] == list(knee.PAIR_NAMES)
    assert [row["index"] for row in selected] == list(knee.PAIR_SOURCE_INDICES)

    reordered = copy.deepcopy(pairs)
    reordered[8], reordered[14] = reordered[14], reordered[8]
    reordered[8]["index"] = 8
    reordered[14]["index"] = 14
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="exact ABI3 source order",
    ):
        knee._select_articular_pairs(reordered, profile)


def test_custom_rehashed_profile_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    profile["source"]["dataset_id"] = "attacker-dataset"
    path = tmp_path / "profile.json"
    path.write_bytes(canonical_bytes(profile) + b"\n")
    monkeypatch.setattr(knee, "PROFILE", path)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="profile file identity"):
        knee._load_frozen_profile()


def test_signed_tetrahedron_moments_reject_inversion() -> None:
    source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    inverted = [source[0], source[2], source[1], source[3]]
    with pytest.raises(knee.LoadedAnatomyKneeError, match="inverted"):
        knee._tet_moments(inverted, source, 1000.0)


def test_matter_node_mass_rounds_volume_before_fp64_accumulation() -> None:
    volume = 1.0 / 6.0
    old_double_volume_sum = 0.0
    referenced_matter_sum = 0.0
    for _ in range(7):
        old_double_volume_sum += 1000.0 * volume * 0.25
        referenced_matter_sum += knee._matter_referenced_tet_node_mass(volume, 1000.0)

    assert struct.pack("<f", old_double_volume_sum).hex() == "55d59143"
    assert struct.pack("<f", referenced_matter_sum).hex() == "56d59143"
    assert struct.pack("<f", old_double_volume_sum) != struct.pack(
        "<f", referenced_matter_sum
    )


def test_donor_raw_second_moment_is_subtracted_once_and_closes() -> None:
    manifest = _manifest()
    for donor in manifest["mass_partition"]["donors"]:
        source = donor["source"]["raw_second_mass_moment_kg_m2"]
        subtracted = donor["subtracted"]["raw_second_mass_moment_kg_m2"]
        remaining = donor["remaining"]["raw_second_mass_moment_kg_m2"]
        for row in range(3):
            for column in range(3):
                assert remaining[row][column] == pytest.approx(
                    source[row][column] - subtracted[row][column], abs=1e-15
                )
        assert donor["closure_residual"]["raw_second_mass_moment_kg_m2"] == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]


def test_donor_source_mass_requires_exact_profile_nominal_float32() -> None:
    value = _manifest()
    expected = knee._float32(8.4)
    bits = struct.unpack("<I", struct.pack("<f", expected))[0]
    next_float32 = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    assert value["mass_partition"]["donors"][0]["source"]["zeroth_mass_kg"] == expected
    assert next_float32 != expected

    value["lab_authoring_export"]["donor_moments"]["donors"][0]["moments"][
        "zeroth_mass_kg"
    ] = next_float32
    _rehash_embedded_lab_export(value)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="donor source mass differs"):
        knee.validate_manifest(value)


def test_manifest_rejects_authority_donor_source_and_dataset_mutations() -> None:
    original = _manifest()
    knee.validate_manifest(original)
    mutations = []
    authority = copy.deepcopy(original)
    authority["full_state_authority"]["owner_id"] = "attacker:owns-current"
    mutations.append(authority)
    donor = copy.deepcopy(original)
    donor["mass_partition"]["donors"][0]["source"]["zeroth_mass_kg"] = 999.0
    donor["mass_partition"]["identity_sha256"] = digest(
        {
            key: value
            for key, value in donor["mass_partition"].items()
            if key != "identity_sha256"
        }
    )
    mutations.append(donor)
    dataset = copy.deepcopy(original)
    dataset["source"]["dataset_id"] = "attacker-dataset"
    mutations.append(dataset)
    for value in mutations:
        _rehash(value)
        with pytest.raises(knee.LoadedAnatomyKneeError):
            knee.validate_manifest(value)


def test_manifest_rejects_scope_coordinate_and_lab_export_laundering() -> None:
    original = _manifest()
    coordinate = copy.deepcopy(original)
    coordinate["coordinates"]["x_ref"]["stress_free"] = True
    _rehash(coordinate)
    scope = copy.deepcopy(original)
    scope["semantic_scope"]["coverage_leaf_sha256s"] = []
    _rehash(scope)
    exported_mapping = copy.deepcopy(original)
    exported_mapping["lab_authoring_export"]["mapping"]["algorithm"] = "other-map"
    _rehash_embedded_lab_export(exported_mapping)
    exported_pose = copy.deepcopy(original)
    exported_pose["lab_authoring_export"]["poses"]["projected_reference"][
        "identity_sha256"
    ] = "4" * 64
    _rehash_embedded_lab_export(exported_pose)
    exported_source = copy.deepcopy(original)
    exported_source["lab_authoring_export"]["source"][
        "source_model_fingerprint_sha256"
    ] = "5" * 64
    _rehash_embedded_lab_export(exported_source)
    exported_donor = copy.deepcopy(original)
    exported_donor["lab_authoring_export"]["donor_moments"]["donors"][0]["moments"][
        "first_mass_moment_kg_m"
    ][0] = 0.125
    _rehash_embedded_lab_export(exported_donor)
    for value in (
        coordinate,
        scope,
        exported_mapping,
        exported_pose,
        exported_source,
        exported_donor,
    ):
        with pytest.raises(knee.LoadedAnatomyKneeError):
            knee.validate_manifest(value)


def test_manifest_rejects_fully_resealed_lab_export_boundary_mutation() -> None:
    value = _manifest()
    value["lab_authoring_export"]["boundary"] = (
        "Production and clinical qualification established."
    )
    _rehash_embedded_lab_export(value)

    export = value["lab_authoring_export"]
    assert export["manifest_sha256"] == digest(
        {key: item for key, item in export.items() if key != "manifest_sha256"}
    )
    assert value["inputs"]["lab_export"] == {
        "schema": knee.LAB_EXPORT_SCHEMA,
        "file_sha256": hashlib.sha256(canonical_bytes(export) + b"\n").hexdigest(),
        "identity_sha256": export["manifest_sha256"],
    }
    assert value["mass_partition"]["provenance"][
        "lab_export_manifest_sha256"
    ] == export["manifest_sha256"]
    assert value["mass_partition"]["identity_sha256"] == digest(
        {
            key: item
            for key, item in value["mass_partition"].items()
            if key != "identity_sha256"
        }
    )
    assert value["manifest_sha256"] == digest(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )

    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="Lab authoring export evidence boundary differs",
    ):
        knee.validate_manifest(value)


def test_mapping_continuation_region_evidence_is_strict_and_cross_checked() -> None:
    original = _manifest()
    knee.validate_manifest(original)
    diagnostics = original["lab_authoring_export"]["mapping"]["diagnostics"]
    assert original["lab_authoring_export"]["mapping"]["id"] == knee.LAB_MAPPING_ID
    assert (
        original["lab_authoring_export"]["mapping"]["algorithm"]
        == knee.LAB_MAPPING_ALGORITHM
    )
    assert (
        original["lab_authoring_export"]["mapping"]["code_identity_encoding"]
        == knee.LAB_MAPPING_CODE_IDENTITY_ENCODING
    )
    assert [row["name"] for row in diagnostics["regions"]] == list(knee.REGION_NAMES)
    assert [row["direct_map_status"] for row in diagnostics["regions"]] == [
        "accepted",
        "accepted",
        "accepted",
        "accepted",
        "rejected_inversion",
        "accepted",
    ]
    assert [row["substep_count"] for row in diagnostics["regions"]] == [
        1,
        1,
        1,
        1,
        2,
        1,
    ]

    def mutate_region(value: dict, name: str) -> dict:
        return next(
            row
            for row in value["lab_authoring_export"]["mapping"]["diagnostics"][
                "regions"
            ]
            if row["name"] == name
        )

    mutations: list[dict] = []

    algorithm = copy.deepcopy(original)
    algorithm["lab_authoring_export"]["mapping"]["algorithm"] = "other-map"
    mutations.append(algorithm)

    node_mass_algorithm = copy.deepcopy(original)
    node_mass_algorithm["lab_authoring_export"]["mapping"]["diagnostics"][
        "raw_f32_node_mass_algorithm"
    ] = "double-volume-final-f32"
    mutations.append(node_mass_algorithm)

    code_identity_encoding = copy.deepcopy(original)
    code_identity_encoding["lab_authoring_export"]["mapping"][
        "code_identity_encoding"
    ] = "sha256(core-file-sha256)"
    mutations.append(code_identity_encoding)

    reordered = copy.deepcopy(original)
    rows = reordered["lab_authoring_export"]["mapping"]["diagnostics"]["regions"]
    rows[0], rows[1] = rows[1], rows[0]
    mutations.append(reordered)

    ptl_status = copy.deepcopy(original)
    mutate_region(ptl_status, "PTL")["direct_map_status"] = "accepted"
    mutations.append(ptl_status)

    ptl_failure = copy.deepcopy(original)
    mutate_region(ptl_failure, "PTL")["direct_failure_jacobian"] = 0.01
    mutations.append(ptl_failure)

    ptl_substeps = copy.deepcopy(original)
    mutate_region(ptl_substeps, "PTL")["substep_count"] = 4
    mutations.append(ptl_substeps)

    qat_status = copy.deepcopy(original)
    mutate_region(qat_status, "QAT")["direct_map_status"] = "rejected_inversion"
    mutations.append(qat_status)

    regional_jacobian = copy.deepcopy(original)
    mutate_region(regional_jacobian, "ACL")["minimum_jacobian"] = 0.9
    mutations.append(regional_jacobian)

    aggregate_jacobian = copy.deepcopy(original)
    aggregate_jacobian["lab_authoring_export"]["mapping"]["diagnostics"]["jacobian"][
        "minimum_determinant"
    ] = 0.9
    mutations.append(aggregate_jacobian)

    anchor_residual = copy.deepcopy(original)
    mutate_region(anchor_residual, "PCL")["maximum_persisted_f32_anchor_residual_m"] = (
        2.1e-7
    )
    mutations.append(anchor_residual)

    extra = copy.deepcopy(original)
    mutate_region(extra, "ACL")["unvalidated"] = True
    mutations.append(extra)

    for value in mutations:
        _sync_export_mapping_and_rehash(value)
        with pytest.raises(knee.LoadedAnatomyKneeError):
            knee.validate_manifest(value)


def test_nheq2_runtime_link_uses_source_default_identity_reference() -> None:
    value = _source_compliant_nheq2_manifest()
    knee.validate_manifest(value)
    export = value["lab_authoring_export"]
    provenance = value["mass_partition"]["provenance"]
    assert (
        export["source"]["equality_payload_sha256"]
        == knee.SOURCE_COMPLIANT_NHEQ2_RUNTIME_EQUALITY_PAYLOAD_SHA256
    )
    assert provenance["equality_payload_sha256"] == export["source"][
        "equality_payload_sha256"
    ]
    assert export["poses"] == provenance["poses"]
    assert export["mapping"] == provenance["mapping"]
    assert export["source"]["equality_payload_role"] == (
        knee.SOURCE_COMPLIANT_EQUALITY_PAYLOAD_ROLE
    )
    assert export["source"]["equality_projection_applied"] is False
    assert value["inputs"]["authoring_profile"] == {
        "schema": knee.SOURCE_DEFAULT_PROFILE_SCHEMA,
        "file_sha256": knee.FROZEN_SOURCE_DEFAULT_PROFILE_FILE_SHA256,
        "identity_sha256": knee.FROZEN_SOURCE_DEFAULT_PROFILE_IDENTITY_SHA256,
    }
    assert export["x_ref"]["file_sha256"] == export["source"]["x_source_sha256"]
    assert export["poses"]["projected_reference"] == export["poses"][
        "source_default"
    ]
    assert export["mapping"]["id"] == knee.SOURCE_DEFAULT_LAB_MAPPING_ID
    assert (
        export["mapping"]["algorithm"]
        == knee.SOURCE_DEFAULT_LAB_MAPPING_ALGORITHM
    )
    assert export["mapping"]["diagnostics"]["maximum_displacement_m"] == 0.0
    assert export["mapping"]["diagnostics"]["equality_residual_maximum"] == (
        knee.SOURCE_DEFAULT_NHEQ2_DIAGNOSTIC_RESIDUAL_MAXIMUM
    )
    assert export["mapping"]["diagnostics"]["jacobian"] == {
        "finite": True,
        "minimum_determinant": 1.0,
        "maximum_determinant": 1.0,
        "orientation_preserving": True,
    }
    for row in export["mapping"]["diagnostics"]["regions"]:
        assert row["direct_map_status"] == "accepted"
        assert row["substep_count"] == 1
        assert row["direct_failure_tetrahedron"] is None
        assert row["direct_failure_jacobian"] is None

    two_step = copy.deepcopy(value)
    two_step["lab_authoring_export"]["mapping"]["diagnostics"]["regions"][4][
        "substep_count"
    ] = 2
    _sync_export_mapping_and_rehash(two_step)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="PTL direct-map status or continuation count differs",
    ):
        knee.validate_manifest(two_step)

    pose = copy.deepcopy(value)
    pose["lab_authoring_export"]["poses"]["projected_reference"][
        "identity_sha256"
    ] = "4" * 64
    _sync_export_provenance_and_rehash(pose)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="source-default A-to-B_ref identity evidence differs",
    ):
        knee.validate_manifest(pose)

    displacement = copy.deepcopy(value)
    displacement["lab_authoring_export"]["mapping"]["diagnostics"][
        "maximum_displacement_m"
    ] = 1.0e-12
    _sync_export_provenance_and_rehash(displacement)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="source-default A-to-B_ref identity evidence differs",
    ):
        knee.validate_manifest(displacement)

    residual = copy.deepcopy(value)
    residual["lab_authoring_export"]["mapping"]["diagnostics"][
        "equality_residual_maximum"
    ] = 0.0
    _sync_export_provenance_and_rehash(residual)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="source-default A-to-B_ref identity evidence differs",
    ):
        knee.validate_manifest(residual)

    regional_jacobian = copy.deepcopy(value)
    regional_jacobian["lab_authoring_export"]["mapping"]["diagnostics"]["regions"][
        0
    ]["minimum_jacobian"] = 0.999
    _sync_export_provenance_and_rehash(regional_jacobian)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="ACL source-default identity Jacobians differ",
    ):
        knee.validate_manifest(regional_jacobian)


def test_legacy_projection_and_runtime_link_modes_are_correlated() -> None:
    legacy_with_nheq2_mapping = _manifest()
    legacy_ptl = legacy_with_nheq2_mapping["lab_authoring_export"]["mapping"][
        "diagnostics"
    ]["regions"][4]
    legacy_ptl.update(
        {
            "substep_count": 1,
            "direct_map_status": "accepted",
            "direct_failure_tetrahedron": None,
            "direct_failure_jacobian": None,
        }
    )
    _sync_export_mapping_and_rehash(legacy_with_nheq2_mapping)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="PTL direct-map status or continuation count differs",
    ):
        knee.validate_manifest(legacy_with_nheq2_mapping)

    nheq2_with_legacy_mapping = _source_compliant_nheq2_manifest()
    nheq2_ptl = nheq2_with_legacy_mapping["lab_authoring_export"]["mapping"][
        "diagnostics"
    ]["regions"][4]
    nheq2_ptl.update(
        {
            "substep_count": 2,
            "direct_map_status": "rejected_inversion",
            "direct_failure_tetrahedron": 419,
            "direct_failure_jacobian": -0.20709127479457548,
        }
    )
    _sync_export_mapping_and_rehash(nheq2_with_legacy_mapping)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="PTL direct-map status or continuation count differs",
    ):
        knee.validate_manifest(nheq2_with_legacy_mapping)

    unknown = _source_compliant_nheq2_manifest()
    unknown_sha256 = "0" * 64
    unknown["lab_authoring_export"]["source"][
        "equality_payload_sha256"
    ] = unknown_sha256
    unknown["mass_partition"]["provenance"][
        "equality_payload_sha256"
    ] = unknown_sha256
    _rehash_embedded_lab_export(unknown)
    with pytest.raises(
        knee.LoadedAnatomyKneeError,
        match="Lab equality role or payload identity differs",
    ):
        knee.validate_manifest(unknown)


def test_blocked_source_status_remains_visible_without_production_promotion() -> None:
    value = _manifest()
    knee.validate_manifest(value)
    assert value["status"] == "candidate"
    assert value["source_ownership_status"] == "blocked"
    assert value["qualification"]["production_physical_ownership"] is False
    binding = knee.build_binding(value)
    knee.validate_binding(binding, value)
    assert binding["manifest"]["source_ownership_status"] == "blocked"


def test_binding_rejects_a_different_companion_manifest() -> None:
    manifest = _manifest()
    binding = knee.build_binding(manifest)
    changed = copy.deepcopy(manifest)
    changed["source_ownership_status"] = "partial"
    _rehash(changed)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="differs from its companion"):
        knee.validate_binding(binding, changed)


def test_atomic_bundle_is_idempotent_and_rejects_symlink_outputs(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    binding = knee.build_binding(manifest)
    bundle = tmp_path / "bundle"
    first = knee._publish_bundle(bundle, manifest, binding)
    second = knee._publish_bundle(bundle, manifest, binding)
    assert first == second
    assert {path.name for path in bundle.iterdir()} == {
        knee.MANIFEST_FILENAME,
        knee.BINDING_FILENAME,
    }
    existing_link = tmp_path / "existing-link"
    existing_link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="redirected"):
        knee._publish_bundle(existing_link, manifest, binding)
    dangling_link = tmp_path / "dangling-link"
    dangling_link.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(knee.LoadedAnatomyKneeError, match="redirected"):
        knee._publish_bundle(dangling_link, manifest, binding)


def test_json_schema_behavior_matches_candidate_boundary() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (knee.ROOT / "schemas/humanpack-loaded-anatomy-knee.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    value = _manifest()
    validator.validate(value)
    source_default = _source_compliant_nheq2_manifest()
    validator.validate(source_default)
    for mutation in (
        lambda item: item["inputs"].__setitem__(
            "authoring_profile", value["inputs"]["authoring_profile"]
        ),
        lambda item: item["lab_authoring_export"]["source"].__setitem__(
            "equality_payload_role", "authoring-projection"
        ),
        lambda item: item["lab_authoring_export"]["source"].__setitem__(
            "equality_projection_applied", True
        ),
        lambda item: item["lab_authoring_export"]["mapping"].__setitem__(
            "id", knee.LAB_MAPPING_ID
        ),
        lambda item: item["lab_authoring_export"]["mapping"][
            "diagnostics"
        ].__setitem__("maximum_displacement_m", 1.0e-6),
        lambda item: item["lab_authoring_export"]["mapping"][
            "diagnostics"
        ].__setitem__("equality_residual_maximum", 0.0),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"][
            "jacobian"
        ].__setitem__("minimum_determinant", 0.999),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"][
            "regions"
        ][4].__setitem__("substep_count", 2),
        lambda item: item["coordinates"]["x_ref"].__setitem__(
            "reference_state_class", "projected-rest-candidate"
        ),
    ):
        changed = copy.deepcopy(source_default)
        mutation(changed)
        assert list(validator.iter_errors(changed))
    for mutation in (
        lambda item: item["qualification"].__setitem__(
            "production_physical_ownership", True
        ),
        lambda item: item["coordinates"]["x_current"].__setitem__(
            "accepted_sha256", "a" * 64
        ),
        lambda item: item["regions"][0]["owners"].__setitem__(
            "active_force", {"status": "candidate", "owner_id": "attacker"}
        ),
        lambda item: item["active_force_replacements"][0].__setitem__(
            "replacement_scale", 0.5
        ),
        lambda item: item["source"].__setitem__("dataset_id", "attacker"),
        lambda item: item["full_state_authority"].__setitem__(
            "owner_id", "attacker:state"
        ),
        lambda item: item["regions"][0]["material"].__setitem__("c1_mpa", 99.0),
        lambda item: item["regions"][0]["owners"]["physical_volume"].__setitem__(
            "owner_id", "attacker:volume"
        ),
        lambda item: item["mass_partition"]["donors"][0].__setitem__(
            "semantic_id", "attacker:donor"
        ),
        lambda item: item["mass_partition"]["donors"][0]["source"].__setitem__(
            "zeroth_mass_kg", 999.0
        ),
        lambda item: item["articular_contact_pairs"][0]["state_owner"].__setitem__(
            "owner_id", "attacker:contact"
        ),
        lambda item: item["active_force_replacements"][0].__setitem__(
            "route_semantic_id", "attacker:route"
        ),
        lambda item: item["active_force_replacements"][0]["endpoints"][0].__setitem__(
            "body_index", 999
        ),
        lambda item: item["passive_ligament_owners"][0].__setitem__(
            "semantic_id", "attacker:passive"
        ),
        lambda item: item["passive_ligament_owners"][0][
            "passive_force_owner"
        ].__setitem__("owner_id", "attacker:passive-owner"),
        lambda item: item["topology"]["region_spans"].__setitem__(0, "attacker"),
        lambda item: item["lab_authoring_export"]["donor_moments"][
            "donors"
        ].__setitem__(0, {"attacker": True}),
        lambda item: item["lab_authoring_export"]["mapping"].__setitem__(
            "algorithm", "other-map"
        ),
        lambda item: item["lab_authoring_export"]["mapping"].__setitem__(
            "code_identity_encoding", "sha256(core-file-sha256)"
        ),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"].__setitem__(
            "raw_f32_node_mass_algorithm", "double-volume-final-f32"
        ),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"]["regions"][
            4
        ].__setitem__("direct_map_status", "accepted"),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"]["regions"][
            4
        ].__setitem__("substep_count", 4),
        lambda item: item["lab_authoring_export"]["mapping"]["diagnostics"]["regions"][
            0
        ].__setitem__("maximum_anchor_residual_m", 2.1e-7),
        lambda item: item["lab_authoring_export"].__setitem__(
            "boundary", "Production and clinical qualification established."
        ),
    ):
        changed = copy.deepcopy(value)
        mutation(changed)
        assert list(validator.iter_errors(changed))

    referencing = pytest.importorskip("referencing")
    binding_schema = json.loads(
        (
            knee.ROOT / "schemas/humanpack-loaded-anatomy-knee-binding.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator.check_schema(binding_schema)
    registry = referencing.Registry().with_resource(
        schema["$id"], referencing.Resource.from_contents(schema)
    )
    binding_validator = jsonschema.Draft202012Validator(
        binding_schema,
        registry=registry,
    )
    binding = knee.build_binding(value)
    binding_validator.validate(binding)
    binding["manifest"]["source"]["dataset_id"] = "attacker"
    assert list(binding_validator.iter_errors(binding))
