"""Compare source organ crossings with the compiled same-owner census."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import abdominal_organ_separation as exact
from . import model as human
from .whole_body_source_overlap_pair import read_obj_triangles


SCHEMA = "numi.human.whole-body-source-overlap-census.v2"
ROOT = human.REPOSITORY_ROOT
DEFAULT_TOPOLOGY_MANIFEST = (
    ROOT / "Docs/media/source-topology-repair-20260929/payload/"
    "source-topology-repair-candidates.manifest.json"
)
DEFAULT_MANIFESTS = (
    ROOT / "Docs/media/whole-visceral-coverage-20260929/payload-manifest.json",
    ROOT / "Docs/media/organ-family-coverage-20260929/payload-manifest.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("whole-body source overlap census: " + message)


def _read_json(path: Path) -> dict:
    path = path.resolve()
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return human.read_json(path)


def _archive_member(hierarchy: str, member_id: str) -> str:
    require(hierarchy in ("part_of", "is_a"),
            f"unsupported BodyParts3D hierarchy: {hierarchy}")
    prefix = "partof" if hierarchy == "part_of" else "isa"
    return f"{prefix}_BP3D_4.0_obj_99/{member_id}.obj"


def source_frame_rows(manifests: list[dict], body_map: dict,
                      source_semantics: dict) -> dict[str, dict]:
    """Join source-family and baseline maps to selected BodyParts3D members."""
    family_rows: dict[str, dict] = {}
    for manifest in manifests:
        require(manifest["schema"] ==
                "numi.human.source-organ-family-composite-payload.v1",
                "source family manifest schema differs")
        for row in manifest["surfaces"]:
            member_id = row["member_id"]
            previous = family_rows.get(member_id)
            if previous is not None:
                require(all(previous[key] == row[key] for key in (
                    "source_member", "source_member_sha256", "source_body_id",
                    "core_body_index", "myosim_body", "hierarchy")),
                    f"source-family manifests disagree for {member_id}")
            family_rows[member_id] = row

    require(body_map["schema"] ==
            "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1",
            "BodyParts3D-to-MyoSim map schema differs")
    map_rows: dict[str, dict] = {}
    for row in body_map["entries"]:
        previous = map_rows.get(row["member_id"])
        if previous is not None:
            require(previous["hierarchy"] == row["hierarchy"] and
                    previous["myosim_body"] == row["myosim_body"],
                    f"baseline source map disagrees for {row['member_id']}")
        map_rows[row["member_id"]] = row

    require(source_semantics["schema"] == "numi.human.whole-body-source-semantics.v1",
            "whole-body source-semantics schema differs")
    semantic_rows = {row["source_member_id"]: row
                     for row in source_semantics["rows"]}
    result = {}
    for member_id, semantic in semantic_rows.items():
        if semantic["source_provider"] != "bodyparts3d":
            continue
        family = family_rows.get(member_id)
        mapped = map_rows.get(member_id)
        require(family is not None or mapped is not None,
                f"BodyParts3D selected member has no source-frame map: {member_id}")
        if family is not None and mapped is not None:
            require(family["myosim_body"] == mapped["myosim_body"] and
                    family["hierarchy"] == mapped["hierarchy"],
                    f"family and baseline maps disagree for {member_id}")
        if family is not None:
            hierarchy = family["hierarchy"]
            source_member = family["source_member"]
            myosim_body = family["myosim_body"]
            archive_sha256 = family["source_member_sha256"]
            manifest_vertex_count = family["vertex_count"]
            manifest_triangle_count = family["triangle_count"]
            source_body_id = family["source_body_id"]
            core_body_index = family["core_body_index"]
            frame_source = "source_family_manifest"
        else:
            hierarchy = mapped["hierarchy"]
            source_member = _archive_member(hierarchy, member_id)
            myosim_body = mapped["myosim_body"]
            archive_sha256 = None
            manifest_vertex_count = None
            manifest_triangle_count = None
            source_body_id = None
            core_body_index = None
            frame_source = "baseline_body_map"
        require(semantic["body_name"] == myosim_body,
                f"source-map owner differs from selected semantic row: {member_id}")
        result[member_id] = {
            "source_member_id": member_id,
            "source_member": source_member,
            "hierarchy": hierarchy,
            "myosim_body": myosim_body,
            "semantic_body_index": semantic["body_index"],
            "semantic_body_name": semantic["body_name"],
            "source_geometry_sha256": semantic["source_geometry_sha256"],
            "source_stable_id": semantic["source_stable_id"],
            "visible_stable_id": semantic["visible_stable_id"],
            "source_member_sha256_from_manifest": archive_sha256,
            "manifest_vertex_count": manifest_vertex_count,
            "manifest_triangle_count": manifest_triangle_count,
            "source_body_id": source_body_id,
            "core_body_index": core_body_index,
            "frame_source": frame_source,
        }
    return result


def eligible_pairs(overlap: dict, source_rows: dict[str, dict]) -> list[dict]:
    """Return crossings whose raw meshes share one declared MyoSim owner."""
    result = []
    for pair in overlap["exact_pairs"]:
        if pair["status"] != "surface_crossing":
            continue
        first = pair["first"]["source_member_id"]
        second = pair["second"]["source_member_id"]
        a, b = source_rows.get(first), source_rows.get(second)
        if a is None or b is None:
            continue
        if (a["myosim_body"] != b["myosim_body"] or
                pair["first"]["body_name"] != a["myosim_body"] or
                pair["second"]["body_name"] != b["myosim_body"] or
                pair["first"]["body_index"] != a["semantic_body_index"] or
                pair["second"]["body_index"] != b["semantic_body_index"]):
            continue
        result.append(pair)
    return result


def scope_summary(overlap: dict, candidates: list[dict], source_members: int) -> dict:
    tested = overlap["exact_pairs"]
    crossings = [row for row in tested if row["status"] == "surface_crossing"]
    return {
        "compiled_pairs_tested": len(tested),
        "compiled_crossing_pairs": len(crossings),
        "eligible_same_declared_owner_pairs": len(candidates),
        "source_members_read": source_members,
        "excluded_crossing_pairs_without_same_declared_owner_map":
            len(crossings) - len(candidates),
    }


def map_compiled_sample_to_source(
    sample: list[list[int]], first_face_map: list[int], second_face_map: list[int]
) -> list[list[int]]:
    result = []
    for first, second in sample:
        require(0 <= first < len(first_face_map) and
                0 <= second < len(second_face_map),
                "compiled triangle sample exceeds source-face map")
        result.append([first_face_map[first], second_face_map[second]])
    return result


def exact_intersection_pair_indices(first: dict, second: dict) -> list[list[int]]:
    """Retain all exact source triangle witnesses for repair attribution."""
    a_min, a_max = first["triangle_min"], first["triangle_max"]
    b_min, b_max = second["triangle_min"], second["triangle_max"]
    result = []
    if (np.any(a_max.max(axis=0) < b_min.min(axis=0)) or
            np.any(b_max.max(axis=0) < a_min.min(axis=0))):
        return result
    for i, (low, high) in enumerate(zip(a_min, a_max, strict=True)):
        candidates = np.flatnonzero(np.all((b_min <= high) & (b_max >= low), axis=1))
        for j in candidates:
            if exact.triangle_intersection_points(
                    first["records"][i][0], second["records"][int(j)][0]):
                result.append([i, int(j)])
    return result


def audit(source_dir: Path, source_manifest_paths: list[Path], body_map_path: Path,
          source_semantics_path: Path, topology_manifest_path: Path,
          overlap_path: Path) -> dict:
    source_dir, body_map_path, source_semantics_path, topology_manifest_path, overlap_path = (
        Path(path).resolve() for path in
        (source_dir, body_map_path, source_semantics_path,
         topology_manifest_path, overlap_path)
    )
    source_manifest_paths = [Path(path).resolve() for path in source_manifest_paths]
    lock_path = ROOT / "sources.lock.json"
    lock = human.read_json(lock_path)["sources"]["bodyparts3d_4"]["files"]
    manifests = [_read_json(path) for path in source_manifest_paths]
    body_map = _read_json(body_map_path)
    source_semantics = _read_json(source_semantics_path)
    topology_manifest = _read_json(topology_manifest_path)
    require(topology_manifest["schema"] ==
            "numi.human.source-topology-repair-candidates.v1",
            "source topology-repair manifest schema differs")
    repairs_by_visible_id = {row["candidate_stable_id"]: row
                             for row in topology_manifest["candidates"]}
    member_rows = source_frame_rows(manifests, body_map, source_semantics)
    overlap = _read_json(overlap_path)
    require(overlap.get("schema") == "numi.human.whole-body-organ-overlap-diagnostic.v1",
            "corrected overlap receipt/schema is missing")
    candidates = eligible_pairs(overlap, member_rows)
    require(bool(candidates), "no crossings share a declared raw source-owner frame")

    member_ids = sorted({pair[key]["source_member_id"]
                         for pair in candidates for key in ("first", "second")})
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_rows = []
    overlap_surfaces = {row["source_member_id"]: row
                        for row in overlap["selected_organ_surfaces"]}
    member_face_maps: dict[str, dict] = {}
    used_archives = sorted({Path(member_rows[member_id]["source_member"]).parts[0] + ".zip"
                            for member_id in member_ids})
    archive_paths = {name: source_dir / name for name in used_archives}
    for name, archive_path in archive_paths.items():
        archive_lock = lock.get(name)
        require(archive_lock is not None,
                f"source archive is not present in the BodyParts3D lock: {name}")
        require(archive_path.stat().st_size == archive_lock["bytes"] and
                human.sha256(archive_path) == archive_lock["sha256"],
                f"BodyParts3D source archive differs from lock: {name}")

    with ExitStack() as stack:
        archives = {name: stack.enter_context(ZipFile(path))
                    for name, path in archive_paths.items()}
        for member_id in member_ids:
            row = member_rows[member_id]
            archive_name = Path(row["source_member"]).parts[0] + ".zip"
            raw = archives[archive_name].read(row["source_member"])
            raw_sha = hashlib.sha256(raw).hexdigest()
            expected_sha = row["source_member_sha256_from_manifest"]
            require(expected_sha is None or raw_sha == expected_sha,
                    f"raw OBJ hash differs from source-family manifest: {member_id}")
            points_mm, faces = read_obj_triangles(raw)
            if row["manifest_vertex_count"] is not None:
                require(len(points_mm) == row["manifest_vertex_count"] and
                        len(faces) == row["manifest_triangle_count"],
                        f"raw OBJ topology differs from source-family manifest: {member_id}")
            points_m = (points_mm.astype(np.float32) / np.float32(1000.0)).astype(np.float64)
            meshes[member_id] = (points_m, faces)
            selected = overlap_surfaces.get(member_id)
            require(selected is not None and
                    selected["source_stable_id"] == row["source_stable_id"] and
                    selected["visible_stable_id"] == row["visible_stable_id"],
                    f"compiled overlap surface identity differs from semantic source: {member_id}")
            repair_row = repairs_by_visible_id.get(row["visible_stable_id"])
            if row["visible_stable_id"] != row["source_stable_id"]:
                require(repair_row is not None and
                        repair_row["member_id"] == member_id and
                        repair_row["stable_id"] == row["source_stable_id"] and
                        repair_row["candidate_stable_id"] == row["visible_stable_id"],
                        f"replacement surface lacks source-face provenance: {member_id}")
                face_map = repair_row["source_derivation"]["source_face_ids"]
                require(len(face_map) == selected["triangle_count"] and
                        repair_row["source_face_count"] == len(faces),
                        f"source-face map count differs from compiled mesh: {member_id}")
                map_kind = "exact_source_topology_repair_face_map"
            else:
                require(selected["triangle_count"] == len(faces),
                        f"unreplaced surface triangle count differs from source: {member_id}")
                face_map = list(range(len(faces)))
                map_kind = "identity_source_triangle_order"
            require(all(0 <= int(index) < len(faces) for index in face_map),
                    f"source-face map index escapes raw source mesh: {member_id}")
            member_face_maps[member_id] = {
                "source_face_ids": [int(index) for index in face_map],
                "source_faces_removed": sorted(set(range(len(faces))) -
                                                {int(index) for index in face_map}),
                "map_kind": map_kind,
                "compiled_triangle_count": selected["triangle_count"],
                "compiled_geometry_sha256": selected["compiled_geometry_sha256"],
            }
            source_rows.append({
                "source_member_id": member_id,
                "source_archive": archive_name,
                "source_member": row["source_member"],
                "source_member_sha256": raw_sha,
                "source_frame_source": row["frame_source"],
                "source_body_id": row["source_body_id"],
                "core_body_index": row["core_body_index"],
                "myosim_body": row["myosim_body"],
                "source_vertex_count": len(points_mm),
                "source_triangle_count": len(faces),
            })

    pair_rows = []
    for compiled in candidates:
        source_ids = (compiled["first"]["source_member_id"],
                      compiled["second"]["source_member_id"])
        denominator, prepared = exact.exact_integer_meshes(
            {member_id: meshes[member_id] for member_id in source_ids})
        source = exact.audit_pair(prepared[source_ids[0]], prepared[source_ids[1]], denominator)
        source_sample = source["first_intersecting_triangle_pairs"]
        compiled_count = compiled["exact_intersecting_triangle_pairs"]
        compiled_sample = compiled["first_intersecting_triangle_pairs"]
        mapped_compiled_sample = map_compiled_sample_to_source(
            compiled_sample,
            member_face_maps[source_ids[0]]["source_face_ids"],
            member_face_maps[source_ids[1]]["source_face_ids"],
        )
        sample_match = mapped_compiled_sample == source_sample
        crossing_count_match = source["exact_intersecting_triangle_pairs"] == compiled_count
        status_match = source["status"] == compiled["status"]
        source_triangle_counts = [len(meshes[source_ids[0]][1]),
                                  len(meshes[source_ids[1]][1])]
        compiled_triangle_counts = [compiled["first"]["triangle_count"],
                                    compiled["second"]["triangle_count"]]
        mesh_triangle_counts_match = source_triangle_counts == compiled_triangle_counts
        delta = (source["exact_intersecting_triangle_pairs"] - compiled_count)
        removed_face_hits = []
        removed_face_delta_explained = None
        if delta != 0:
            all_source_hits = exact_intersection_pair_indices(
                prepared[source_ids[0]], prepared[source_ids[1]])
            require(len(all_source_hits) == source["exact_intersecting_triangle_pairs"],
                    f"full source triangle witnesses differ from exact summary: {source_ids}")
            removed_first = set(member_face_maps[source_ids[0]]["source_faces_removed"])
            removed_second = set(member_face_maps[source_ids[1]]["source_faces_removed"])
            removed_face_hits = [pair for pair in all_source_hits
                                  if pair[0] in removed_first or pair[1] in removed_second]
            removed_face_delta_explained = delta == len(removed_face_hits)
        if sample_match and crossing_count_match and status_match:
            if mesh_triangle_counts_match:
                disposition = "raw_source_and_compiled_crossing_witnesses_match"
            elif any(member_face_maps[mid]["map_kind"] ==
                     "exact_source_topology_repair_face_map" for mid in source_ids):
                disposition = "crossings_match_after_exact_source_face_mapping"
            else:
                disposition = "crossings_match_with_surface_topology_difference"
        elif (sample_match and status_match and delta > 0 and
              removed_face_delta_explained is True):
            disposition = "topology_repair_removed_mapped_source_crossings"
        elif source["status"] != "surface_crossing":
            disposition = "compiled_only_crossing"
        elif compiled["status"] != "surface_crossing":
            disposition = "raw_source_only_crossing"
        elif not sample_match:
            disposition = "source_triangle_witnesses_differ_after_face_mapping"
        else:
            disposition = "source_compiled_crossing_count_differs"
        a, b = member_rows[source_ids[0]], member_rows[source_ids[1]]
        pair_rows.append({
            "source_member_ids": list(source_ids),
            "myosim_body": a["myosim_body"],
            "source_body_ids": [a["source_body_id"], b["source_body_id"]],
            "core_body_indices": [a["core_body_index"], b["core_body_index"]],
            "compiled_geometry_sha256": [compiled["first"]["compiled_geometry_sha256"],
                                          compiled["second"]["compiled_geometry_sha256"]],
            "source_triangle_counts": source_triangle_counts,
            "compiled_triangle_counts": compiled_triangle_counts,
            "compiled_source_face_map_kinds": [member_face_maps[mid]["map_kind"]
                                                for mid in source_ids],
            "source_status": source["status"],
            "source_intersecting_triangle_pairs":
                source["exact_intersecting_triangle_pairs"],
            "source_first_intersecting_triangle_pairs": source_sample,
            "source_maximum_intersection_segment_m":
                source["maximum_intersection_segment_m"],
            "compiled_status": compiled["status"],
            "compiled_intersecting_triangle_pairs": compiled_count,
            "compiled_first_intersecting_triangle_pairs": compiled_sample,
            "compiled_first_intersecting_source_face_pairs": mapped_compiled_sample,
            "compiled_maximum_intersection_segment_m":
                compiled["maximum_intersection_segment_m"],
            "maximum_segment_absolute_difference_m": abs(
                source["maximum_intersection_segment_m"] -
                compiled["maximum_intersection_segment_m"]),
            "crossing_status_match": status_match,
            "crossing_pair_count_match": crossing_count_match,
            "source_face_mapped_first_12_sample_match": sample_match,
            "source_compiled_mesh_triangle_counts_match": mesh_triangle_counts_match,
            "source_intersection_count_delta": delta,
            "source_intersections_on_removed_faces": removed_face_hits,
            "removed_face_intersections_explain_count_delta":
                removed_face_delta_explained,
            "disposition": disposition,
        })

    disposition_counts: dict[str, int] = {}
    for row in pair_rows:
        disposition_counts[row["disposition"]] = (
            disposition_counts.get(row["disposition"], 0) + 1)
    scope = scope_summary(overlap, candidates, len(source_rows))
    source_manifest_inputs = [{
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": human.sha256(path),
    } for path in source_manifest_paths]
    return {
        "schema": SCHEMA,
        "status": "source_frame_overlap_census_complete",
        "source_archives": [{
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "bytes": path.stat().st_size,
            "sha256": human.sha256(path),
        } for path in archive_paths.values()],
        "source_lock_sha256": human.sha256(lock_path),
        "source_family_manifests": source_manifest_inputs,
        "body_map": {
            "path": str(body_map_path.relative_to(ROOT))
                    if body_map_path.is_relative_to(ROOT) else str(body_map_path),
            "sha256": human.sha256(body_map_path),
        },
        "source_semantics": {
            "path": str(source_semantics_path.relative_to(ROOT))
                    if source_semantics_path.is_relative_to(ROOT)
                    else str(source_semantics_path),
            "sha256": human.sha256(source_semantics_path),
        },
        "topology_repair_manifest": {
            "path": str(topology_manifest_path.relative_to(ROOT))
                    if topology_manifest_path.is_relative_to(ROOT)
                    else str(topology_manifest_path),
            "sha256": human.sha256(topology_manifest_path),
        },
        "compiled_overlap_receipt_sha256": human.sha256(overlap_path),
        "scope": scope,
        "disposition_counts": disposition_counts,
        "source_members": source_rows,
        "pairs": pair_rows,
        "source_geometry_modified": False,
        "registration_repair_adopted": False,
        "boundary": (
            "The census compares raw BodyParts3D OBJ meshes after their pinned hierarchy and "
            "baseline/family maps assign them to the same MyoSim owner as the compiled overlap "
            "pair. Matching crossing counts and triangle samples localize the measured witness "
            "to source geometry; differing results flag source/compiled changes for review. "
            "Neither outcome determines intended seams, clinical anatomy, penetration volume, "
            "tissue ownership, or mechanics."
        ),
        "executed_source_sha256": human.sha256(Path(__file__)),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-manifest", type=Path, action="append", default=None)
    parser.add_argument("--body-map", type=Path,
                        default=ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json")
    parser.add_argument("--source-semantics", type=Path,
                        default=ROOT / "Docs/media/whole-body-source-semantics-20261002/receipt.json.gz")
    parser.add_argument("--topology-repair-manifest", type=Path,
                        default=DEFAULT_TOPOLOGY_MANIFEST)
    parser.add_argument("--overlap-receipt", type=Path,
                        default=ROOT / "Docs/media/whole-body-organ-overlap-20261002/receipt-v4.json.gz")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifests = arguments.source_manifest or list(DEFAULT_MANIFESTS)
    try:
        result = audit(arguments.source_dir, manifests, arguments.body_map,
                       arguments.source_semantics, arguments.topology_repair_manifest,
                       arguments.overlap_receipt)
        _write_immutable(arguments.output, result)
        print(json.dumps({"status": result["status"], **result["scope"],
                          "disposition_counts": result["disposition_counts"],
                          "output": str(arguments.output.resolve())}, sort_keys=True), flush=True)
        return 0
    except (human.ImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"whole-body-source-overlap-census: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
