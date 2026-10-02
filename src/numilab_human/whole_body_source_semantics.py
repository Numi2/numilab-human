"""Crosswalk the selected whole-body atlas to its pinned source ontology.

This records source-member identity and source taxonomy for every visible
surface. Ontology membership does not prove clinical anatomy, physical volume,
or tissue mechanics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

from . import model as human
from . import whole_body_organ_overlap as overlap
from . import whole_body_surface_gate as selected_gate


SCHEMA = "numi.human.whole-body-source-semantics.v1"
ROOT = human.REPOSITORY_ROOT
ONTOLOGY_FACETS = {
    "organ": ("FMA67498", "organ"),
    "organ_region": ("FMA67619", "organ region"),
    "organ_component": ("FMA14065", "organ component"),
    "cardinal_organ_part": ("FMA82472", "cardinal organ part"),
}
PRIORITY_FACETS = ("organ", "organ_region", "organ_component", "cardinal_organ_part")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("whole-body source semantics: " + message)


def _concept_rows(hierarchy: overlap.SourceHierarchy, relation: str,
                  member_id: str | None, *, include_ancestors: bool) -> list[dict]:
    if member_id is None:
        return []
    direct = hierarchy.mappings[relation].get(member_id, {})
    concept_ids = set(direct)
    if include_ancestors:
        for concept_id in direct:
            concept_ids.update(hierarchy.ancestors(relation, concept_id))
    return [
        {"concept_id": concept_id,
         "name": hierarchy.concept_names[relation][concept_id],
         "direct": concept_id in direct}
        for concept_id in sorted(concept_ids)
    ]


def _gzip_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == encoded,
                "receipt is immutable; choose a new output path")
        return
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    try:
        with pending.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def audit(selected_report: Path, indexed_census: Path, quotient_census: Path,
          candidate_payload: Path, candidate_audit: Path, selection: Path,
          sources: Path, source_lock: Path, body_manifest: Path) -> dict:
    paths = [Path(path).resolve() for path in (
        selected_report, indexed_census, quotient_census, candidate_payload,
        candidate_audit, selection, sources, source_lock, body_manifest,
    )]
    (selected_report, indexed_census, quotient_census, candidate_payload,
     candidate_audit, selection, sources, source_lock, body_manifest) = paths

    gate = human.read_json(selected_report)
    require(gate["schema"] == selected_gate.SCHEMA
            and gate["baseline_payload_sha256"] == overlap.repair.BASE_SHA
            and gate["source_surface_count"] == 579
            and len(gate["rows"]) == 579,
            "selected surface gate schema or completeness")
    require(human.sha256(candidate_payload) == gate["candidate_payload_sha256"],
            "selected compiled payload identity")
    proof = human.read_json(candidate_audit)
    choice = human.read_json(selection)
    require(proof["schema"] == "numi.human.source-topology-repair-audit.v1"
            and proof["passed"] is True
            and proof["raw_579_geometry_byte_identical"] is True
            and proof["payload_sha256"] == gate["candidate_payload_sha256"]
            and proof["candidate_count"] == 19
            and choice["source_payload_sha256"] == gate["candidate_payload_sha256"]
            and choice["source_audit_sha256"] == human.sha256(candidate_audit),
            "source-preserving candidate and inspection selection identity")

    indexed_summary, indexed_identity, indexed_rows, indexed_summary_sha = (
        selected_gate.census_rows(indexed_census, overlap.repair.BASE_SHA)
    )
    quotient_summary, quotient_identity, quotient_rows, quotient_summary_sha = (
        selected_gate.census_rows(quotient_census, overlap.repair.BASE_SHA)
    )
    require(indexed_summary_sha == gate["indexed_census_sha256"]
            and quotient_summary_sha == gate["quotient_census_sha256"]
            and indexed_identity["source_manifest_hashes"]
                == quotient_identity["source_manifest_hashes"]
            and all(a["geometry_sha256"] == b["geometry_sha256"]
                    and a["source_body_index"] == b["source_body_index"]
                    and a["source_layer_code"] == b["source_layer_code"]
                    and a["source_member_id"] == b["source_member_id"]
                    for a, b in zip(indexed_rows, quotient_rows, strict=True)),
            "indexed/quotient census identity")
    replayed = selected_gate.selected_surface_rows(quotient_rows, proof["rows"], choice)
    require(replayed == gate["rows"], "selected source representation does not replay")

    owner_names = human.read_json(body_manifest)["core_tree"]["body_order"]
    hierarchy = overlap.SourceHierarchy.load(sources, source_lock)
    zanatomy_config_path = (ROOT / "config/zanatomy-thorax-source.v1.json").resolve()
    zanatomy_config = human.read_json(zanatomy_config_path)
    require(zanatomy_config["schema"] == "numi.human.zanatomy-thorax-source.v1"
            and zanatomy_config["base_surface_count"] == 304,
            "pinned Z-Anatomy thorax configuration")
    export_relative = Path(zanatomy_config["export"]["file"])
    export_path = (ROOT / export_relative).resolve()
    require(ROOT.resolve() in export_path.parents and export_path.is_file()
            and human.sha256(export_path) == zanatomy_config["export"]["compressed_sha256"]
            and _gzip_payload_sha256(export_path) == zanatomy_config["export"]["sha256"],
            "pinned Z-Anatomy source export hashes")
    zanatomy_objects = zanatomy_config["objects"]
    require(len(zanatomy_objects) == 6
            and [row["stable_id"] for row in zanatomy_objects] == list(range(305, 311))
            and len({row["object_name"] for row in zanatomy_objects}) == 6
            and sum(row["concept_id"] is not None for row in zanatomy_objects) == 5,
            "complete Z-Anatomy stable surface map")
    zanatomy_by_stable_id = {row["stable_id"]: row for row in zanatomy_objects}
    concept_labels = hierarchy.concept_names["part_of"]
    concept_parents = hierarchy.parents["part_of"]
    rows = []
    for selected, census in zip(gate["rows"], quotient_rows, strict=True):
        member_id = selected["source_member_id"]
        stable_id = selected["source_stable_id"]
        layer_code = selected["source_layer_code"]
        owner = census["source_body_index"]
        require(member_id == census["source_member_id"]
                and layer_code == census["source_layer_code"]
                and stable_id == census["stable_id"]
                and (member_id is None or
                     (isinstance(member_id, str) and member_id.startswith("FJ")))
                and layer_code in selected_gate.LAYERS
                and 0 <= owner < len(owner_names),
                f"source/member/layer/owner identity at stable ID {stable_id}")
        mapped_member = member_id is not None and any(
            member_id in hierarchy.mappings[relation] for relation in ("part_of", "is_a")
        )
        require(member_id is None or mapped_member,
                f"source member absent from pinned hierarchy at stable ID {stable_id}")
        source_object = zanatomy_by_stable_id.get(stable_id)
        if source_object is not None:
            require(member_id is None
                    and selected["source_object_name"] == source_object["object_name"]
                    and layer_code == source_object["layer_code"]
                    and census.get("provider") == "zanatomy"
                    and source_object["myosim_body"] == owner_names[owner],
                    f"Z-Anatomy source object lineage at stable ID {stable_id}")
            source_concept_id = source_object["concept_id"]
            if source_concept_id is not None:
                parent_id = source_object["parent_lung_concept_id"]
                require(source_concept_id in concept_labels
                        and parent_id in concept_labels
                        and parent_id in concept_parents.get(source_concept_id, set()),
                        f"Z-Anatomy lung-lobe concept/parent identity at stable ID {stable_id}")
                source_concept_name = concept_labels[source_concept_id]
                concept_mapping_status = "matched_to_pinned_part_of_parent_edge"
            else:
                source_concept_name = None
                concept_mapping_status = "source_config_has_no_fma_concept_id"
        else:
            source_concept_id = None
            source_concept_name = None
            concept_mapping_status = (
                "bodyparts3d_member_hierarchy_mapped"
                if member_id is not None else "source_member_and_fma_concept_unmapped"
            )

        is_a_concepts = _concept_rows(hierarchy, "is_a", member_id,
                                      include_ancestors=True)
        is_a_ids = {item["concept_id"] for item in is_a_concepts}
        facets = {
            key: {"concept_id": concept_id, "name": name,
                  "present_in_is_a_closure": (concept_id in is_a_ids
                                               if member_id is not None else None)}
            for key, (concept_id, name) in ONTOLOGY_FACETS.items()
        }
        priority_class = (
            next((key for key in PRIORITY_FACETS
                  if facets[key]["present_in_is_a_closure"] is True), "other")
            if member_id is not None else "ontology_unmapped"
        )
        rows.append({
            "source_stable_id": stable_id,
            "visible_stable_id": selected["visible_stable_id"],
            "source_member_id": member_id,
            "bodyparts3d_member_mapping_status": (
                "mapped_to_pinned_bodyparts3d_member"
                if member_id is not None else "z_anatomy_object_identity_used"
            ),
            "source_object_name": selected["source_object_name"],
            "source_provider": census.get("provider", "unknown"),
            "source_object_reference_id": source_object["id"] if source_object else None,
            "source_specific_semantic_class": source_object["layer"] if source_object else None,
            "source_anatomy_concept_id": source_concept_id,
            "source_anatomy_concept_name": source_concept_name,
            "source_anatomy_concept_mapping_status": concept_mapping_status,
            "source_geometry_sha256": census["geometry_sha256"],
            "body_index": owner,
            "body_name": owner_names[owner],
            "source_layer_code": layer_code,
            "source_layer_name": selected_gate.LAYERS[layer_code],
            "source_original_topology_status": selected["source_original_status"],
            "selected_surface_status": selected["selected_status"],
            "selected_closed_embedded_surface_candidate": selected[
                "selected_closed_embedded_surface_candidate"
            ],
            "part_of_direct_concepts": _concept_rows(
                hierarchy, "part_of", member_id, include_ancestors=False,
            ),
            "is_a_direct_concepts": _concept_rows(
                hierarchy, "is_a", member_id, include_ancestors=False,
            ),
            "is_a_concept_ancestry_closure": is_a_concepts,
            "bodyparts3d_ontology_facets": facets,
            "bodyparts3d_ontology_priority_class": priority_class,
        })

    require(len(rows) == 579 and len({row["source_stable_id"] for row in rows}) == 579,
            "complete one-to-one selected surface crosswalk")
    require(len({row["source_member_id"] for row in rows
                 if row["source_member_id"] is not None}) == 573
            and sum(row["source_member_id"] is None for row in rows) == 6,
            "pinned unique source member count")
    source_identity_count = sum(
        row["source_member_id"] is not None
        or row["source_object_reference_id"] is not None for row in rows
    )
    require(source_identity_count == len(rows),
            "source identity absent from both source namespaces")
    layer_counts: dict[str, dict] = {}
    for code, layer in sorted(selected_gate.LAYERS.items()):
        group = [row for row in rows if row["source_layer_code"] == code]
        if group:
            layer_counts[layer] = {
                "layer_code": code,
                "surface_count": len(group),
                "unique_source_members": len({row["source_member_id"] for row in group
                                               if row["source_member_id"] is not None}),
                "selected_topology_status_counts": dict(sorted(Counter(
                    row["selected_surface_status"] for row in group
                ).items())),
                "bodyparts3d_ontology_facet_surface_counts": {
                    key: sum(row["bodyparts3d_ontology_facets"][key]["present_in_is_a_closure"] is True
                             for row in group)
                    for key in ONTOLOGY_FACETS
                },
                "bodyparts3d_member_id_missing_surfaces": sum(
                    row["source_member_id"] is None for row in group
                ),
                "source_anatomy_concept_id_unmapped_surfaces": sum(
                    row["source_anatomy_concept_mapping_status"]
                    == "source_config_has_no_fma_concept_id" for row in group
                ),
                "source_specific_semantic_class_counts": dict(sorted(Counter(
                    row["source_specific_semantic_class"] for row in group
                    if row["source_specific_semantic_class"] is not None
                ).items())),
                "bodyparts3d_ontology_priority_class_counts": dict(sorted(Counter(
                    row["bodyparts3d_ontology_priority_class"] for row in group
                ).items())),
            }

    return {
        "schema": SCHEMA,
        "status": "completed_source_identity_and_ontology_crosswalk",
        "selected_report_sha256": human.sha256(selected_report),
        "indexed_census_summary_sha256": indexed_summary_sha,
        "indexed_census_identity_sha256": indexed_summary["identity_sha256"],
        "quotient_census_summary_sha256": quotient_summary_sha,
        "quotient_census_identity_sha256": quotient_summary["identity_sha256"],
        "candidate_payload_sha256": human.sha256(candidate_payload),
        "candidate_audit_sha256": human.sha256(candidate_audit),
        "inspection_selection_sha256": human.sha256(selection),
        "source_lock_sha256": human.sha256(source_lock),
        "source_hierarchy_sha256": hierarchy.hashes,
        "native_body_manifest_sha256": human.sha256(body_manifest),
        "source_surface_count": len(rows),
        "unique_source_member_count": len({row["source_member_id"] for row in rows
                                            if row["source_member_id"] is not None}),
        "source_configured_object_count": sum(
            row["source_object_reference_id"] is not None for row in rows
        ),
        "source_configured_fma_concept_count": sum(
            row["source_anatomy_concept_id"] is not None for row in rows
        ),
        "source_configured_fma_concept_id_unmapped_surface_count": sum(
            row["source_anatomy_concept_mapping_status"]
            == "source_config_has_no_fma_concept_id" for row in rows
        ),
        "zanatomy_source_config_sha256": human.sha256(zanatomy_config_path),
        "zanatomy_source_export_compressed_sha256": human.sha256(export_path),
        "zanatomy_source_export_uncompressed_sha256": zanatomy_config["export"]["sha256"],
        "zanatomy_source_blend_sha256": zanatomy_config["source"]["blend"]["sha256"],
        "source_identity_complete_surface_count": source_identity_count,
        "bodyparts3d_member_id_missing_surface_count": sum(
            row["source_member_id"] is None for row in rows
        ),
        "bodyparts3d_ontology_facets_are_nonexclusive_membership_tests": True,
        "bodyparts3d_ontology_priority_class_order": list(PRIORITY_FACETS),
        "bodyparts3d_ontology_facets": {
            key: {"concept_id": concept_id, "name": name,
                  "relation": "is_a direct concepts and transitive ancestors"}
            for key, (concept_id, name) in ONTOLOGY_FACETS.items()
        },
        "layer_counts": layer_counts,
        "rows": rows,
        "clinical_anatomy": False,
        "physical_volume": False,
        "organ_mechanics": False,
        "source_taxonomy_authority": "Pinned BodyParts3D 4.0 part_of/is_a source tables only",
        "boundary": (
            "This crosswalk binds visible compiled surfaces to pinned BodyParts3D member IDs, "
            "layer labels, native rigid owners, exact source geometry hashes, and source ontology "
            "terms. Ontology ancestry does not establish tissue boundaries, component-versus-whole "
            "volume ownership, clinical placement, mechanics, or physiological function. Selected "
            "surface topology status remains an independent field; unresolved surfaces are not "
            "promoted by semantic classification."
        ),
        "executed_source_sha256": {
            "whole_body_source_semantics.py": human.sha256(Path(__file__)),
            "whole_body_organ_overlap.py": human.sha256(Path(overlap.__file__)),
            "whole_body_surface_gate.py": human.sha256(Path(selected_gate.__file__)),
            "cli.py": human.sha256(Path(__file__).with_name("cli.py")),
        },
    }


def command(arguments: argparse.Namespace) -> int:
    result = audit(
        arguments.selected_report, arguments.indexed_census, arguments.quotient_census,
        arguments.candidate_payload, arguments.candidate_audit, arguments.selection,
        arguments.sources, arguments.source_lock, arguments.body_manifest,
    )
    _write_immutable(arguments.output, result)
    print(json.dumps({
        "status": result["status"],
            "source_surface_count": result["source_surface_count"],
            "unique_source_member_count": result["unique_source_member_count"],
            "bodyparts3d_member_id_missing_surface_count": result[
                "bodyparts3d_member_id_missing_surface_count"
            ],
        "layer_counts": result["layer_counts"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True), flush=True)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    media = ROOT / "Docs/media"
    whole_body = media / "whole-body-embeddedness-20260929"
    topology = media / "source-topology-repair-20260929"
    parser.add_argument("--selected-report", type=Path,
                        default=whole_body / "selected-surface-gate.json")
    parser.add_argument("--indexed-census", type=Path,
                        default=whole_body / "indexed-census.tar.gz")
    parser.add_argument("--quotient-census", type=Path,
                        default=whole_body / "quotient-census.tar.gz")
    parser.add_argument("--candidate-payload", type=Path,
                        default=topology / "payload/source-topology-repair-candidates.nhanatomy")
    parser.add_argument("--candidate-audit", type=Path,
                        default=topology / "source-audit.json")
    parser.add_argument("--selection", type=Path,
                        default=topology / "inspection-selection.final.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--body-manifest", type=Path,
                        default=ROOT / "Build/myosim-fullbody/myosim-fullbody-reference.manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new immutable source semantics receipt")
    parser.set_defaults(handler=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except (human.ImportError, OSError, KeyError, ValueError) as error:
        parser.exit(2, f"whole-body-source-semantics: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
