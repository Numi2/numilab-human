"""Compute source-frame surface moments for overlap-census organs outside the 18-region mass inventory.

The result extends per-surface geometry evidence only. It does not sum across
surfaces or assign physical volumes, density, body registration, or mass.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

from . import cvsim21_anatomy
from .cardiac_cavity_geometry import exact_coordinate_quotient, parse_obj
from .model import ImportError as HumanImportError
from .organ_geometry import ROOT, _topology_summary
from .organ_geometry_component_moments import _aabbs_disjoint, _component_details
from .physiology import canonical, read_json

SCHEMA = "HumanPack.whole-body-overlap-organ-surface-moments.v1"
CENSUS = ROOT / "Docs/media/whole-body-source-overlap-census-20261002/receipt-v4.json"
ORGAN_MASS = ROOT / "Docs/media/tissue-mass-candidate-20260914/receipt-v2.json"
SEMANTICS = ROOT / "Docs/media/whole-body-source-semantics-20261002/receipt.json.gz"
SOURCE_LOCK = ROOT / "sources.lock.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("whole-body overlap organ moments: " + message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError(f"{label} is not valid JSON") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value, raw


def _read_semantics(path: Path) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "source semantics is not a regular file")
    compressed = path.read_bytes()
    try:
        value = json.loads(gzip.decompress(compressed))
    except (gzip.BadGzipFile, EOFError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HumanImportError("source semantics is not a valid gzip JSON receipt") from error
    require(isinstance(value, dict), "source semantics is not an object")
    return value, compressed


def _relative(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _archive_roots(source_rows: list[dict[str, Any]]) -> dict[str, str]:
    roots = {}
    for row in source_rows:
        archive_name = row.get("source_archive")
        member_path = row.get("source_member")
        member_id = row.get("source_member_id")
        require(isinstance(archive_name, str) and archive_name.endswith(".zip") and
                isinstance(member_id, str) and member_id and isinstance(member_path, str),
                "census source archive identity is malformed")
        root = archive_name[:-4]
        require(member_path == f"{root}/{member_id}.obj",
                f"source archive member path differs for {member_id}")
        roots[archive_name] = root
    return roots


def compile_candidates(
    *,
    census: Path = CENSUS,
    organ_mass: Path = ORGAN_MASS,
    semantics: Path = SEMANTICS,
    sources: Path = ROOT / "Sources",
    source_lock: Path = SOURCE_LOCK,
) -> dict[str, Any]:
    census = Path(census)
    organ_mass = Path(organ_mass)
    semantics = Path(semantics)
    sources = Path(sources)
    source_lock = Path(source_lock)
    census_doc, census_bytes = _read_json(census, "source overlap census")
    organ_doc, organ_bytes = _read_json(organ_mass, "organ mass candidate")
    semantics_doc, semantics_bytes = _read_semantics(semantics)
    lock_doc, lock_bytes = _read_json(source_lock, "source lock")

    require(census_doc.get("schema") == "numi.human.whole-body-source-overlap-census.v2" and
            census_doc.get("status") == "source_frame_overlap_census_complete",
            "unsupported or incomplete source overlap census")
    require(organ_doc.get("schema") == "HumanPack.tissue-mass-composition-candidate.v1",
            "unsupported organ mass candidate")
    require(semantics_doc.get("schema") == "numi.human.whole-body-source-semantics.v1" and
            semantics_doc.get("status") == "completed_source_identity_and_ontology_crosswalk",
            "unsupported or incomplete source semantics")
    source_semantics_ref = census_doc.get("source_semantics", {})
    require(source_semantics_ref.get("sha256") == _sha256(semantics_bytes) and
            source_semantics_ref.get("path") == _relative(semantics),
            "source semantics does not match the overlap census binding")

    census_rows = census_doc.get("source_members")
    organ_rows = organ_doc.get("candidates")
    semantic_rows = semantics_doc.get("rows")
    require(isinstance(census_rows, list) and len(census_rows) == 104 and
            isinstance(organ_rows, list) and len(organ_rows) == 378 and
            isinstance(semantic_rows, list) and len(semantic_rows) == 579,
            "source inventories are incomplete")
    overlap_by_id = {row.get("source_member_id"): row for row in census_rows
                     if isinstance(row, dict)}
    organ_ids = {row.get("member_id") for row in organ_rows if isinstance(row, dict)}
    semantic_by_id = {row.get("source_member_id"): row for row in semantic_rows
                      if isinstance(row, dict) and row.get("source_member_id") is not None}
    require(len(overlap_by_id) == 104 and len(organ_ids) == 378 and
            all(isinstance(member_id, str) and member_id for member_id in organ_ids),
            "source inventory member identities are invalid")
    selected_ids = set(overlap_by_id) - organ_ids
    require(len(selected_ids) == 74 and selected_ids <= set(semantic_by_id),
            "overlap-census-only organ identities differ from source semantics")
    selected_rows = [semantic_by_id[member_id] for member_id in sorted(selected_ids)]
    class_counts: dict[str, int] = {}
    for row in selected_rows:
        class_id = row.get("bodyparts3d_ontology_priority_class")
        require(class_id in {"organ", "organ_region", "organ_component"} and
                row.get("source_layer_name") == "organ" and
                row.get("selected_closed_embedded_surface_candidate") is True and
                row.get("selected_surface_status") == "closed_embedded_source_candidate",
                f"source semantic admission changed for {row.get('source_member_id')}")
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    require(class_counts == {"organ": 11, "organ_component": 4, "organ_region": 59},
            "overlap-census-only organ taxonomy counts changed")

    _archive_roots(census_rows)
    lock_sources = lock_doc.get("sources", {}).get("bodyparts3d_4", {}).get("files", {})
    archive_handles: dict[str, zipfile.ZipFile] = {}
    archive_records: dict[str, dict[str, Any]] = {}
    try:
        for archive_name in sorted({overlap_by_id[member_id]["source_archive"]
                                    for member_id in selected_ids}):
            lock_record = lock_sources.get(archive_name)
            archive_path = sources / archive_name
            require(isinstance(lock_record, dict) and archive_path.is_file() and
                    not archive_path.is_symlink(), f"source archive is missing or unlocked: {archive_name}")
            raw_archive = archive_path.read_bytes()
            archive_sha = _sha256(raw_archive)
            require(archive_sha == lock_record.get("sha256") and
                    len(raw_archive) == lock_record.get("bytes"),
                    f"source archive hash or size differs: {archive_name}")
            archive_handles[archive_name] = zipfile.ZipFile(archive_path)
            archive_records[archive_name] = {
                "path": _relative(archive_path),
                "sha256": archive_sha,
                "bytes": len(raw_archive),
            }

        records: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        for semantic in selected_rows:
            member_id = semantic["source_member_id"]
            source = overlap_by_id[member_id]
            archive_name = source["source_archive"]
            archive_member = source["source_member"]
            require(archive_name in archive_handles and
                    archive_member in archive_handles[archive_name].namelist(),
                    f"source archive member is missing: {archive_member}")
            raw_member = archive_handles[archive_name].read(archive_member)
            member_sha = _sha256(raw_member)
            require(member_sha == source.get("source_member_sha256"),
                    f"source member hash differs: {member_id}")
            parsed = parse_obj(raw_member, archive_member)
            require(len(parsed["vertices_mm"]) == source.get("source_vertex_count") and
                    len(parsed["triangles"]) == source.get("source_triangle_count"),
                    f"source member topology counts differ: {member_id}")
            quotient = exact_coordinate_quotient(parsed)
            raw_topology = cvsim21_anatomy.geometry.analyze_topology(
                parsed["vertices_mm"], parsed["triangles"]
            )
            topology_summary = _topology_summary(raw_topology, quotient)
            topology = quotient["topology"]
            require(topology_summary["closed_oriented_manifold_candidate"] and
                    not topology["unused_vertex_ids"],
                    f"source topology is not a closed surface: {member_id}")

            moments: dict[str, Any] | None = None
            component_records = None
            component_disjoint = None
            if topology["face_component_count"] == 1:
                moments = cvsim21_anatomy.geometry_moments(
                    quotient["vertices_m"], quotient["triangles"]
                )
                status = "computed_single_closed_component"
            else:
                details, component_moments, all_closed = _component_details(quotient)
                require(all_closed and len(component_moments) == topology["face_component_count"],
                        f"disconnected source has an open component: {member_id}")
                component_disjoint = _aabbs_disjoint([item["aabb_m"] for item in details])
                component_records = [
                    {
                        "component_index": detail["component_index"],
                        "vertex_count": detail["vertex_count"],
                        "face_count": detail["face_count"],
                        "aabb_m": detail["aabb_m"],
                        "surface_moments": moment,
                    }
                    for detail, moment in zip(details, component_moments, strict=True)
                ]
                status = "unaggregated_multi_component_member"
                moments = None

            require(moments is None or
                    (moments.get("physical_volume_m3") is None and
                     moments.get("mechanical_mass_kg") is None and
                     math.isfinite(moments.get("absolute_signed_volume_m3", float("nan"))) and
                     moments["absolute_signed_volume_m3"] > 0),
                    f"surface moment candidate has an invalid owner or integral: {member_id}")
            status_counts[status] = status_counts.get(status, 0) + 1
            records.append({
                "member_id": member_id,
                "source_archive": archive_name,
                "source_member": archive_member,
                "source_member_sha256": member_sha,
                "selected_compiled_geometry_sha256": semantic["source_geometry_sha256"],
                "source_layer_name": semantic["source_layer_name"],
                "ontology_priority_class": semantic["bodyparts3d_ontology_priority_class"],
                "source_family_names": semantic["source_family_names"],
                "moment_status": status,
                "source_topology": topology_summary,
                "source_surface_moments": moments,
                "source_component_moments": component_records,
                "component_aabb_pairwise_disjoint": component_disjoint,
                "physical_volume_m3": None,
                "density_kg_per_m3": None,
                "mechanical_mass_kg": None,
                "physical_volume_owner": False,
                "mechanical_mass_owner": False,
                "body_registration": False,
                "additive_with_other_surface_candidates": False,
            })
    finally:
        for archive in archive_handles.values():
            archive.close()

    require(status_counts == {
        "computed_single_closed_component": 73,
        "unaggregated_multi_component_member": 1,
    }, f"source moment candidate status counts changed: {status_counts}")
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.whole-body-overlap-organ-moments.1",
        "status": "partial",
        "source": {
            "census": {"path": _relative(census), "sha256": _sha256(census_bytes)},
            "organ_mass_candidate": {"path": _relative(organ_mass), "sha256": _sha256(organ_bytes)},
            "semantics": {"path": _relative(semantics), "sha256": _sha256(semantics_bytes)},
            "source_lock": {"path": _relative(source_lock), "sha256": _sha256(lock_bytes)},
            "archives": archive_records,
        },
        "counts": {
            "overlap_census_member_count": len(overlap_by_id),
            "organ_mass_inventory_member_count": len(organ_ids),
            "overlap_census_only_organ_member_count": len(selected_ids),
            "computed_single_closed_surface_moment_count": status_counts["computed_single_closed_component"],
            "unaggregated_multi_component_member_count": status_counts["unaggregated_multi_component_member"],
            "ontology_priority_class_counts": dict(sorted(class_counts.items())),
        },
        "identity_bindings": {
            "organ_candidate_only_source_member_ids": sorted(selected_ids),
            "source_member_ids_sha256": _sha256(canonical(sorted(selected_ids))),
        },
        "members": records,
        "qualification": {
            "source_archives_hash_and_size_bound": True,
            "source_member_hashes_bound": True,
            "census_and_semantics_identities_bound": True,
            "closed_surface_topology_recomputed": True,
            "single_component_surface_moments_computed": True,
            "disconnected_components_summed": False,
            "cross_surface_moments_summed": False,
            "self_intersection_recomputed": False,
            "interdomain_overlap_checked": False,
            "clinical_registration": False,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
        },
        "boundary": (
            "These are source-frame surface integral candidates for the 74 overlap-census "
            "organ identities outside the 18-region organ-mass inventory. 73 single closed "
            "components have per-surface moments. FJ3150 has two closed components whose "
            "source AABBs overlap; their moments remain separate and unaggregated. No "
            "candidate is a clinical volume, a disjoint tissue partition, a body-registered "
            "organ, a density, or a mechanical mass owner."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def run(arguments: argparse.Namespace) -> int:
    try:
        result = compile_candidates(
            census=arguments.census,
            organ_mass=arguments.organ_mass,
            semantics=arguments.semantics,
            sources=arguments.sources,
            source_lock=arguments.source_lock,
        )
        output = arguments.output.resolve()
        digest = _immutable_write(output, result)
    except (HumanImportError, OSError, KeyError, TypeError, ValueError,
            zipfile.BadZipFile, OverflowError) as error:
        print(error)
        return 2
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                      "status": result["status"], "counts": result["counts"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--organ-mass", type=Path, default=ORGAN_MASS)
    parser.add_argument("--semantics", type=Path, default=SEMANTICS)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
