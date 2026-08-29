"""Resolve ambiguous abdominal endpoints from pinned thorax mesh topology.

This stage does not infer attachment sites from the BodyParts3D geometry.  It
joins the exact source-audit triangle back to the connected component of the
pinned MyoSim thorax mesh, then reuses the already-gated rib-component to
BodyParts3D member correspondence.  Non-rib and source-non-bone endpoints stay
point-owned for future cartilage/fascia mechanics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_bone_proximity import (
    AUDIT_SCHEMA,
    WORKLIST_SCHEMA,
    _compiled_meshes_by_body,
)
from .myosim_export import export_fullbody
from .rib_registration import _rib_components_by_side_level
from .upper_limb_registration import _body_frame_to_core


SCHEMA = "numi.human.myosim-abdominal-source-component-enthesis.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}
TARGET_DISPOSITION = "semantic_bone_member_resolution_needed"
RIB_DISPOSITION = "source_topology_resolved_rib_member"
NON_RIB_DISPOSITION = "source_thorax_non_rib_component_endpoint"
NON_BONE_DISPOSITION = "source_model_non_bone_endpoint"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_signature(component: list[int]) -> str:
    encoded = ",".join(str(index) for index in sorted(component)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def propose_abdominal_enthesis_registration(
    *, sources: Path, registration_path: Path, source_audit_path: Path,
    worklist_path: Path, tendon_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - pinned source environment only
        raise RuntimeError(
            "abdominal enthesis registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("abdominal enthesis registration requires registration candidate v2")
    rib_receipt = registration.get("rib_source_component_registration")
    if (
        not isinstance(rib_receipt, dict)
        or rib_receipt.get("schema")
        != "numi.human.bodyparts3d-myosim-rib-source-component-registration.v1"
        or rib_receipt.get("status")
        != "candidate_passed_topology_enthesis_costovertebral_bilateral_and_order_gates"
    ):
        raise RuntimeError("abdominal enthesis registration requires the promoted rib receipt")
    if source_audit.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("abdominal enthesis registration requires source-bone audit v1")
    if worklist.get("schema") != WORKLIST_SCHEMA:
        raise RuntimeError("abdominal enthesis registration requires registration worklist v1")
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("abdominal enthesis registration requires NHTENDON2 or NHTENDON3")
    source_hashes = {
        registration.get("source", {}).get("myosim", {}).get("source", {}).get(
            "archive_sha256"
        ),
        source_audit.get("source", {}).get("archive_sha256"),
        worklist.get("source", {}).get("myosim_archive_sha256"),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("abdominal enthesis inputs do not share one MyoSim source")

    target_rows = [
        row for row in worklist.get("work_items", [])
        if row.get("disposition") == TARGET_DISPOSITION
        and row.get("source_body_name") == "torso"
    ]
    if len(target_rows) != 20:
        raise RuntimeError(
            "abdominal enthesis registration expected 20 unresolved torso endpoints"
        )
    audit_index = {
        (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in source_audit.get("endpoints", [])
    }
    tendon_index = {
        (int(endpoint["muscle_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in tendon_manifest.get("endpoints", [])
    }

    exported = export_fullbody(sources)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    torso_body = next(
        (body for body in source_bodies.values() if body.get("name") == "torso"),
        None,
    )
    if torso_body is None:
        raise RuntimeError("abdominal enthesis registration cannot resolve the torso body")
    meshes_by_body = _compiled_meshes_by_body(build_model("myofullbody"), mujoco, np)
    ribcage_meshes = [
        mesh for mesh in meshes_by_body[int(torso_body["id"])]
        if "ribcage" in str(mesh.get("mesh_name"))
    ]
    if len(ribcage_meshes) != 1:
        raise RuntimeError("abdominal enthesis registration cannot resolve one thorax mesh")
    ribcage = ribcage_meshes[0]
    source_vertices = _body_frame_to_core(
        np.asarray(ribcage["vertices"], dtype=float), torso_body, np,
    )
    source_faces = np.asarray(ribcage["faces"], dtype=int)
    all_components, components_by_side_level = _rib_components_by_side_level(
        source_vertices, source_faces, np,
    )
    component_index_by_vertex = {
        int(vertex): component_index
        for component_index, component in enumerate(all_components)
        for vertex in component
    }
    rib_identity_by_component = {
        component_index_by_vertex[int(component[0])]: (side, level)
        for (side, level), component in components_by_side_level.items()
    }

    records: list[dict[str, Any]] = []
    for row in sorted(
        target_rows,
        key=lambda item: (int(item["source_actuator_index"]), str(item["endpoint"])),
    ):
        key = (int(row["source_actuator_index"]), str(row["endpoint"]))
        audit = audit_index.get(key)
        tendon = tendon_index.get(key)
        if audit is None or tendon is None or audit.get("muscle") != row.get("muscle"):
            raise RuntimeError(f"abdominal enthesis cannot join endpoint {key}")
        muscle = str(row["muscle"])
        endpoint = str(row["endpoint"])
        ordinal = 0 if endpoint == "origin" else 1
        record: dict[str, Any] = {
            "source_actuator_index": key[0],
            "muscle": muscle,
            "endpoint": endpoint,
            "endpoint_ordinal": ordinal,
            "source_site_id": int(audit["source_site_id"]),
            "source_body_id": int(audit["source_body_id"]),
            "source_body_name": str(audit["source_body_name"]),
            "source_model_classification": str(audit["classification"]),
            "endpoint_migration_m": 0.0,
            "bone_member_ids": [],
        }
        if audit.get("classification") != "source_model_bone_adjacent":
            record["disposition"] = NON_BONE_DISPOSITION
            records.append(record)
            continue
        nearest = audit.get("nearest_source_bone_mesh")
        if (
            not isinstance(nearest, dict)
            or nearest.get("source_mesh_name") != ribcage.get("mesh_name")
        ):
            raise RuntimeError(
                f"abdominal enthesis {muscle}:{endpoint} lacks exact thorax topology"
            )
        triangle_index = int(nearest["nearest_triangle_index"])
        if not 0 <= triangle_index < len(source_faces):
            raise RuntimeError("abdominal enthesis source triangle is out of range")
        triangle = [int(value) for value in source_faces[triangle_index]]
        component_indices = {component_index_by_vertex[index] for index in triangle}
        if len(component_indices) != 1:
            raise RuntimeError("abdominal enthesis triangle crosses connected components")
        component_index = next(iter(component_indices))
        component = all_components[component_index]
        record.update({
            "source_mesh_name": str(ribcage["mesh_name"]),
            "source_triangle_index": triangle_index,
            "source_component_index": component_index,
            "source_component_vertex_count": len(component),
            "source_component_vertex_index_sha256": _component_signature(component),
            "source_surface_distance_m": float(nearest["distance_m"]),
        })
        identity = rib_identity_by_component.get(component_index)
        if identity is None:
            record["disposition"] = NON_RIB_DISPOSITION
        else:
            side, level = identity
            match = re.search(r"_([rl])$", muscle)
            if match is None or match.group(1) != side:
                raise RuntimeError(
                    f"abdominal enthesis {muscle}:{endpoint} crosses source sides"
                )
            member_id = human_model._NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS[side][level]
            record.update({
                "disposition": RIB_DISPOSITION,
                "side": "right" if side == "r" else "left",
                "thoracic_level": level,
                "bone_member_ids": [member_id],
            })
        records.append(record)

    pair_records: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        match = re.fullmatch(r"(.+)_([rl])", record["muscle"])
        if match is None:
            raise RuntimeError("abdominal enthesis muscle identity is not bilateral")
        pair_records.setdefault((match.group(1), record["endpoint"]), {})[
            match.group(2)
        ] = record
    if len(pair_records) != 10 or any(set(pair) != {"r", "l"} for pair in pair_records.values()):
        raise RuntimeError("abdominal enthesis registration lacks ten bilateral pairs")
    # One EO4 source pair lands on different connected-component classes by a
    # sub-millimetre nearest-surface difference.  That atlas asymmetry is not
    # authority to attach only one side to bone.  Keep both source sites as
    # point laws until a bilateral cartilage/rib correspondence is available.
    for pair in pair_records.values():
        dispositions = {record["disposition"] for record in pair.values()}
        if dispositions == {RIB_DISPOSITION, NON_RIB_DISPOSITION}:
            for record in pair.values():
                record["bilateral_source_component_dispositions"] = sorted(dispositions)
                record["bilateral_resolution"] = (
                    "point_owned_due_source_component_class_asymmetry"
                )
                record["disposition"] = NON_RIB_DISPOSITION
                record["bone_member_ids"] = []
                record.pop("side", None)
                record.pop("thoracic_level", None)
        elif len(dispositions) != 1:
            raise RuntimeError(
                "abdominal enthesis registration has incompatible bilateral classes"
            )
    bilateral_pairs = []
    for (base, endpoint), pair in sorted(pair_records.items()):
        same_disposition = pair["r"]["disposition"] == pair["l"]["disposition"]
        same_level = (
            pair["r"].get("thoracic_level") == pair["l"].get("thoracic_level")
        )
        distance_parity = abs(
            float(pair["r"].get("source_surface_distance_m", 0.0))
            - float(pair["l"].get("source_surface_distance_m", 0.0))
        )
        passed = same_disposition and same_level and distance_parity <= 0.0005
        bilateral_pairs.append({
            "muscle_base": base,
            "endpoint": endpoint,
            "disposition": pair["r"]["disposition"],
            "thoracic_level": pair["r"].get("thoracic_level"),
            "source_surface_distance_parity_m": distance_parity,
            "passed": passed,
        })
    if any(not pair["passed"] for pair in bilateral_pairs):
        raise RuntimeError("abdominal enthesis registration violates bilateral parity")

    counts: dict[str, int] = {}
    for disposition in (RIB_DISPOSITION, NON_RIB_DISPOSITION, NON_BONE_DISPOSITION):
        counts[disposition] = sum(
            record["disposition"] == disposition for record in records
        )
    if counts != {
        RIB_DISPOSITION: 10,
        NON_RIB_DISPOSITION: 8,
        NON_BONE_DISPOSITION: 2,
    }:
        raise RuntimeError(
            f"abdominal enthesis source topology classification drifted: {counts}"
        )

    output = json.loads(json.dumps(registration))
    output["abdominal_source_component_enthesis_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_exact_component_identity_and_bilateral_gates",
        "inputs": {
            "registration": {
                "file": registration_path.name,
                "sha256": _sha256(registration_path),
            },
            "source_bone_audit": {
                "file": source_audit_path.name,
                "sha256": _sha256(source_audit_path),
            },
            "registration_worklist": {
                "file": worklist_path.name,
                "sha256": _sha256(worklist_path),
            },
            "tendon_manifest": {
                "file": tendon_manifest_path.name,
                "sha256": _sha256(tendon_manifest_path),
            },
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
        "source_mesh_name": str(ribcage["mesh_name"]),
        "source_connected_component_count": len(all_components),
        "source_rib_component_count": len(components_by_side_level),
        "endpoint_count": len(records),
        "disposition_counts": counts,
        "bilateral_pair_count": len(bilateral_pairs),
        "bilateral_pairs": bilateral_pairs,
        "endpoint_records": records,
        "endpoint_migration_m": 0.0,
        "new_joint_count": 0,
        "evidence_boundary": (
            "This receipt resolves exact endpoint ownership from the pinned MyoSim "
            "thorax triangle and connected component. Rib components may select an "
            "existing BodyParts3D rib member; anterior non-rib and source-non-bone "
            "components remain point-owned pending cartilage/fascia mechanics. It "
            "moves no source site and adds no articulation."
        ),
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--tendon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = propose_abdominal_enthesis_registration(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            source_audit_path=arguments.source_audit.resolve(),
            worklist_path=arguments.worklist.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human abdominal enthesis registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
