"""Join exact organ-surface crossings to pinned source-family semantics.

This is a source-evidence triage only. Family membership and ontology
relationships do not determine whether a geometric overlap is intentional.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from . import model as human


SCHEMA = "numi.human.whole-body-overlap-semantics.v1"
ROOT = human.REPOSITORY_ROOT
GENERIC_CONCEPT_IDS = {
    "FMA20394",  # human body
    "FMA61775",  # physical anatomical entity
    "FMA62955",  # anatomical entity
    "FMA67135",  # anatomical structure
    "FMA67165",  # material anatomical entity
    "FMA67498",  # organ
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("whole-body overlap semantics: " + message)


def _read_receipt(path: Path) -> dict:
    path = path.resolve()
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    else:
        value = human.read_json(path)
    require(isinstance(value, dict), f"receipt is not an object: {path.name}")
    return value


def _surface_key(row: dict) -> tuple[int, int, str]:
    return (row["source_stable_id"], row["visible_stable_id"], row["source_member_id"])


def _classify_pair(pair: dict, semantics_by_key: dict[tuple[int, int, str], dict]) -> dict:
    members = []
    for side in ("first", "second"):
        surface = pair[side]
        key = (surface["source_stable_id"], surface["visible_stable_id"],
               surface["source_member_id"])
        semantic = semantics_by_key.get(key)
        require(semantic is not None,
                f"overlap surface missing from semantics crosswalk: {key}")
        require(semantic["body_index"] == surface["body_index"]
                and semantic["source_layer_code"] == 1
                and semantic["selected_closed_embedded_surface_candidate"] is True,
                f"crosswalk owner/layer/topology differs for source surface {key}")
        members.append((surface, semantic))

    first, second = members
    shared_families = sorted(
        set(first[1]["source_family_names"]) & set(second[1]["source_family_names"])
    )
    annotation = pair["source_hierarchy_annotation"]
    shared_direct = annotation["shared_direct_concepts"]
    specific_shared = {
        relation: [term for term in terms if term["concept_id"] not in GENERIC_CONCEPT_IDS]
        for relation, terms in shared_direct.items()
    }
    has_specific_shared = any(specific_shared.values())
    ancestry_count = len(annotation["direct_concept_ancestry"])
    if shared_families:
        relation_class = "shared_declared_source_family"
    elif has_specific_shared:
        relation_class = "shared_specific_fma_concept"
    elif ancestry_count:
        relation_class = "directional_fma_ancestry_only"
    elif any(shared_direct.values()):
        relation_class = "shared_generic_fma_concept_only"
    else:
        relation_class = "no_registered_family_or_fma_relation"

    classes = sorted((first[1]["bodyparts3d_ontology_priority_class"],
                      second[1]["bodyparts3d_ontology_priority_class"]))
    return {
        "source_stable_ids": [first[0]["source_stable_id"], second[0]["source_stable_id"]],
        "visible_stable_ids": [first[0]["visible_stable_id"], second[0]["visible_stable_id"]],
        "source_member_ids": [first[0]["source_member_id"], second[0]["source_member_id"]],
        "body_index": first[0]["body_index"],
        "body_name": first[0]["body_name"],
        "ontology_priority_classes": classes,
        "shared_source_families": shared_families,
        "relation_class": relation_class,
        "shared_specific_fma_concepts": {
            relation: [{"concept_id": term["concept_id"], "name": term["name"]}
                       for term in terms]
            for relation, terms in specific_shared.items() if terms
        },
        "direct_fma_ancestry_relation_count": ancestry_count,
        "exact_intersecting_triangle_pairs": pair["exact_intersecting_triangle_pairs"],
        "maximum_intersection_segment_m": pair["maximum_intersection_segment_m"],
    }


def audit(overlap_path: Path, semantics_path: Path) -> dict:
    overlap_path, semantics_path = overlap_path.resolve(), semantics_path.resolve()
    overlap = _read_receipt(overlap_path)
    semantics = _read_receipt(semantics_path)
    require(overlap.get("schema") == "numi.human.whole-body-organ-overlap-diagnostic.v1"
            and semantics.get("schema") == "numi.human.whole-body-source-semantics.v1",
            "input receipt schema differs")
    provenance_pairs = (
        ("selection_report_sha256", "selected_report_sha256"),
        ("inspection_selection_sha256", "inspection_selection_sha256"),
        ("source_payload_sha256", "candidate_payload_sha256"),
        ("candidate_audit_sha256", "candidate_audit_sha256"),
        ("indexed_census_summary_sha256", "indexed_census_summary_sha256"),
        ("indexed_census_identity_sha256", "indexed_census_identity_sha256"),
        ("quotient_census_summary_sha256", "quotient_census_summary_sha256"),
        ("quotient_census_identity_sha256", "quotient_census_identity_sha256"),
        ("native_body_manifest_sha256", "native_body_manifest_sha256"),
        ("source_lock_sha256", "source_lock_sha256"),
        ("source_hierarchy_sha256", "source_hierarchy_sha256"),
    )
    for overlap_key, semantics_key in provenance_pairs:
        require(overlap.get(overlap_key) == semantics.get(semantics_key),
                f"input provenance differs: {overlap_key}/{semantics_key}")
    semantic_rows = semantics.get("rows")
    exact_pairs = overlap.get("exact_pairs")
    require(isinstance(semantic_rows, list)
            and len(semantic_rows) == semantics.get("source_surface_count"),
            "source semantics crosswalk is incomplete")
    require(isinstance(exact_pairs, list)
            and len(exact_pairs) == overlap.get("exact_pair_tests"),
            "overlap receipt does not contain the complete exact-pair set")
    semantics_by_key = {_surface_key(row): row for row in semantic_rows}
    require(len(semantics_by_key) == len(semantic_rows),
            "duplicate selected-surface identity in source semantics")
    selected_surfaces = overlap.get("selected_organ_surfaces")
    require(isinstance(selected_surfaces, list)
            and len(selected_surfaces) == overlap.get("selected_organ_surface_count"),
            "selected organ surface identity inventory is incomplete")
    inventory_by_key = {}
    for surface in selected_surfaces:
        key = _surface_key(surface)
        semantic = semantics_by_key.get(key)
        require(semantic is not None
                and semantic["body_index"] == surface["body_index"]
                and semantic["body_name"] == surface["body_name"]
                and semantic["source_layer_code"] == 1
                and semantic["selected_closed_embedded_surface_candidate"] is True,
                f"selected surface missing or differs in crosswalk: {key}")
        require(key not in inventory_by_key,
                f"duplicate selected organ surface identity: {key}")
        inventory_by_key[key] = surface
    exact_pair_rows = [(_classify_pair(pair, semantics_by_key), pair)
                       for pair in exact_pairs]
    referenced_surfaces = set()
    for classified_pair, pair in exact_pair_rows:
        for index in (0, 1):
            key = (classified_pair["source_stable_ids"][index],
                   classified_pair["visible_stable_ids"][index],
                   classified_pair["source_member_ids"][index])
            surface = inventory_by_key.get(key)
            pair_surface = pair[("first", "second")[index]]
            require(surface is not None
                    and surface["compiled_geometry_sha256"]
                        == pair_surface["compiled_geometry_sha256"]
                    and surface["triangle_count"] == pair_surface["triangle_count"],
                    f"exact-pair geometry identity differs from surface inventory: {key}")
            referenced_surfaces.add(key)
        require((pair["status"] == "surface_crossing")
                == (pair["exact_intersecting_triangle_pairs"] > 0),
                "exact pair status and crossing count differ")

    disjoint_pairs = overlap.get("aabb_disjoint_pair_rows")
    require(isinstance(disjoint_pairs, list)
            and len(disjoint_pairs) == overlap.get("aabb_disjoint_pairs"),
            "AABB-separated pair inventory is incomplete")
    for pair in disjoint_pairs:
        for side in ("first", "second"):
            prefix = f"{side}_"
            key = (pair[prefix + "stable_id"], pair[prefix + "visible_stable_id"],
                   pair[prefix + "member_id"])
            surface = inventory_by_key.get(key)
            require(surface is not None
                    and surface["body_index"] == pair["body_index"],
                    f"AABB surface missing or differs from inventory: {key}")
            referenced_surfaces.add(key)

    crossings = [classified_pair for classified_pair, pair in exact_pair_rows
                 if pair["status"] == "surface_crossing"]
    require(len(crossings) == overlap.get("surface_crossing_pair_count"),
            "crossing-pair count differs from overlap receipt")

    classified = crossings
    classified.sort(key=lambda row: (row["source_stable_ids"], row["source_member_ids"]))
    relation_counts = Counter(row["relation_class"] for row in classified)
    class_pair_counts = Counter("/".join(row["ontology_priority_classes"])
                                for row in classified)
    family_pair_counts = Counter(
        family for row in classified for family in row["shared_source_families"]
    )
    direct_specific_count = sum(bool(row["shared_specific_fma_concepts"])
                                for row in classified)
    ancestry_count = sum(row["direct_fma_ancestry_relation_count"] > 0
                         for row in classified)
    top_pairs = sorted(classified, key=lambda row: (
        -row["maximum_intersection_segment_m"], row["source_stable_ids"]
    ))[:20]
    return {
        "schema": SCHEMA,
        "status": "source_semantic_overlap_triage_only",
        "overlap_receipt_sha256": human.sha256(overlap_path),
        "source_semantics_receipt_sha256": human.sha256(semantics_path),
        "overlap_receipt_content_sha256": hashlib.sha256(
            json.dumps(overlap, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_semantics_receipt_content_sha256": hashlib.sha256(
            json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "exact_pairs_examined": len(exact_pairs),
        "aabb_disjoint_pairs_examined": len(disjoint_pairs),
        "selected_organ_surfaces_identity_joined": len(inventory_by_key),
        "selected_organ_surfaces_with_same_owner_pairs": len(referenced_surfaces),
        "selected_organ_surfaces_without_same_owner_pairs": len(
            inventory_by_key.keys() - referenced_surfaces
        ),
        "surface_crossing_pairs": len(classified),
        "relation_class_counts": dict(sorted(relation_counts.items())),
        "crossings_with_specific_shared_fma_concepts": direct_specific_count,
        "crossings_with_directional_fma_ancestry": ancestry_count,
        "ontology_priority_class_pair_counts": dict(sorted(class_pair_counts.items())),
        "shared_source_family_pair_counts": dict(sorted(family_pair_counts.items())),
        "top_twenty_by_maximum_intersection_segment": top_pairs,
        "crossings": classified,
        "clinical_anatomy": False,
        "overlap_intent_resolved": False,
        "physical_volume": False,
        "organ_mechanics": False,
        "boundary": (
            "This deterministic join classifies exact source-surface crossing pairs by declared "
            "source-family overlap, pinned BodyParts3D ontology relationships, and the existing "
            "coarse ontology priority classes. These relationships do not determine whether a "
            "crossing is an intentional nested representation, a duplicate, a segmentation seam, "
            "or an anatomical placement error. Intersection-segment length is not penetration "
            "depth, overlap volume, contact force, or clinical severity. No geometry or ownership "
            "was changed by this audit."
        ),
        "executed_source_sha256": {
            "whole_body_overlap_semantics.py": human.sha256(Path(__file__)),
        },
    }


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


def command(arguments: argparse.Namespace) -> int:
    result = audit(arguments.overlap_receipt, arguments.source_semantics_receipt)
    _write_immutable(arguments.output, result)
    print(json.dumps({
        "status": result["status"],
        "surface_crossing_pairs": result["surface_crossing_pairs"],
        "relation_class_counts": result["relation_class_counts"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True), flush=True)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    default = ROOT / "Docs/media"
    parser.add_argument("--overlap-receipt", type=Path,
                        default=default / "whole-body-organ-overlap-20261002/receipt.json.gz")
    parser.add_argument("--source-semantics-receipt", type=Path,
                        default=default / "whole-body-source-semantics-20261002/receipt.json.gz")
    parser.add_argument("--output", type=Path, required=True,
                        help="new immutable overlap-semantics receipt")
    parser.set_defaults(handler=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except (human.ImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"whole-body-overlap-semantics: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
