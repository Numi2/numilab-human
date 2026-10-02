"""Compare one compiled whole-body crossing with its pinned raw atlas meshes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from . import abdominal_organ_separation as exact
from . import model as human


SCHEMA = "numi.human.whole-body-source-overlap-pair.v1"
ROOT = human.REPOSITORY_ROOT
MEMBER_IDS = ("FJ2577", "FJ2605")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("whole-body source overlap pair: " + message)


def read_obj_triangles(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse OBJ vertices and source-ordered triangular face fans."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise human.ImportError("whole-body source overlap pair: OBJ is not ASCII") from error
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            require(len(fields) >= 4, f"malformed OBJ vertex at line {line_number}")
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            require(len(fields) >= 4, f"malformed OBJ face at line {line_number}")
            face = []
            for value in fields[1:]:
                index = int(value.split("/", 1)[0])
                require(index > 0, f"unsupported non-positive OBJ index at line {line_number}")
                face.append(index - 1)
            for offset in range(1, len(face) - 1):
                triangles.append([face[0], face[offset], face[offset + 1]])
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) >= 4
            and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) >= 1
            and np.isfinite(points).all() and (faces >= 0).all()
            and (faces < len(points)).all(), "invalid source OBJ geometry")
    return points, faces


def _read_json(path: Path) -> dict:
    path = path.resolve()
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return human.read_json(path)


def audit(source_archive: Path, source_manifest_path: Path, overlap_path: Path) -> dict:
    source_archive, source_manifest_path, overlap_path = (
        Path(path).resolve() for path in (source_archive, source_manifest_path, overlap_path)
    )
    lock_path = ROOT / "sources.lock.json"
    lock = human.read_json(lock_path)["sources"]["bodyparts3d_4"]["files"]
    archive_lock = lock["partof_BP3D_4.0_obj_99.zip"]
    require(source_archive.stat().st_size == archive_lock["bytes"]
            and human.sha256(source_archive) == archive_lock["sha256"],
            "BodyParts3D part_of archive differs from source lock")
    source_manifest = _read_json(source_manifest_path)
    require(source_manifest["schema"]
            == "numi.human.source-organ-family-composite-payload.v1",
            "source family payload manifest schema differs")
    manifest_rows = {row["member_id"]: row for row in source_manifest["surfaces"]
                     if row.get("member_id") in MEMBER_IDS}
    require(set(manifest_rows) == set(MEMBER_IDS),
            "one or both source members are missing from the payload manifest")
    require(len({manifest_rows[member]["source_body_id"] for member in MEMBER_IDS}) == 1
            and len({manifest_rows[member]["core_body_index"] for member in MEMBER_IDS}) == 1,
            "source pair does not share its declared source/core body")

    meshes = {}
    source_rows = []
    with ZipFile(source_archive) as archive:
        for member_id in MEMBER_IDS:
            manifest_row = manifest_rows[member_id]
            archive_member = manifest_row["source_member"]
            raw = archive.read(archive_member)
            raw_sha = hashlib.sha256(raw).hexdigest()
            require(raw_sha == manifest_row["source_member_sha256"],
                    f"raw OBJ hash differs from payload manifest: {member_id}")
            points_mm, faces = read_obj_triangles(raw)
            require(len(points_mm) == manifest_row["vertex_count"]
                    and len(faces) == manifest_row["triangle_count"],
                    f"raw OBJ topology differs from source manifest: {member_id}")
            # BodyParts3D OBJ coordinates are millimetres. Cook them to binary32
            # metres before applying exact rational intersection predicates.
            points_m = (points_mm.astype(np.float32) / np.float32(1000.0)).astype(np.float64)
            meshes[member_id] = (points_m, faces)
            source_rows.append({
                "source_member_id": member_id,
                "source_member": archive_member,
                "source_member_sha256": raw_sha,
                "source_body_id": manifest_row["source_body_id"],
                "core_body_index": manifest_row["core_body_index"],
                "source_vertex_count": len(points_mm),
                "source_triangle_count": len(faces),
            })

    denominator, prepared = exact.exact_integer_meshes(meshes)
    source_result = exact.audit_pair(prepared[MEMBER_IDS[0]], prepared[MEMBER_IDS[1]], denominator)
    overlap = _read_json(overlap_path)
    require(overlap.get("schema") == "numi.human.whole-body-organ-overlap-diagnostic.v1"
            and overlap.get("selected_organ_surfaces"),
            "corrected overlap receipt/schema or inventory is missing")
    matches = [row for row in overlap["exact_pairs"]
               if {row["first"]["source_member_id"], row["second"]["source_member_id"]}
               == set(MEMBER_IDS)]
    require(len(matches) == 1, "compiled exact-pair row is missing or ambiguous")
    compiled = matches[0]
    compiled_order = [compiled["first"]["source_member_id"],
                      compiled["second"]["source_member_id"]]
    if compiled_order != list(MEMBER_IDS):
        source_indices = {member_id: i for i, member_id in enumerate(MEMBER_IDS)}
        source_pairs = source_result["first_intersecting_triangle_pairs"]
        if source_indices[compiled_order[0]] == 1:
            source_result["first_intersecting_triangle_pairs"] = [
                [right, left] for left, right in source_pairs
            ]
    same_triangle_sample = (
        source_result["first_intersecting_triangle_pairs"]
        == compiled["first_intersecting_triangle_pairs"]
    )
    require(source_result["exact_intersecting_triangle_pairs"]
            == compiled["exact_intersecting_triangle_pairs"]
            and source_result["status"] == compiled["status"]
            and same_triangle_sample,
            "raw atlas and compiled pair do not share exact crossing topology")

    return {
        "schema": SCHEMA,
        "status": "raw_source_crossings_match_compiled_triangle_pair_sample",
        "source_archive_sha256": human.sha256(source_archive),
        "source_lock_sha256": human.sha256(lock_path),
        "source_family_manifest_sha256": human.sha256(source_manifest_path),
        "compiled_overlap_receipt_sha256": human.sha256(overlap_path),
        "source_members": source_rows,
        "source_pair": {
            "status": source_result["status"],
            "exact_intersecting_triangle_pairs": source_result[
                "exact_intersecting_triangle_pairs"],
            "maximum_intersection_segment_m": source_result[
                "maximum_intersection_segment_m"],
            "first_intersecting_triangle_pairs": source_result[
                "first_intersecting_triangle_pairs"],
        },
        "compiled_pair": {
            "source_member_ids": compiled_order,
            "source_stable_ids": [compiled["first"]["source_stable_id"],
                                  compiled["second"]["source_stable_id"]],
            "visible_stable_ids": [compiled["first"]["visible_stable_id"],
                                   compiled["second"]["visible_stable_id"]],
            "status": compiled["status"],
            "exact_intersecting_triangle_pairs": compiled[
                "exact_intersecting_triangle_pairs"],
            "maximum_intersection_segment_m": compiled[
                "maximum_intersection_segment_m"],
            "first_intersecting_triangle_pairs": compiled[
                "first_intersecting_triangle_pairs"],
        },
        "same_exact_intersecting_triangle_count": True,
        "same_first_intersecting_triangle_pair_sample": same_triangle_sample,
        "maximum_segment_absolute_difference_m": abs(
            source_result["maximum_intersection_segment_m"]
            - compiled["maximum_intersection_segment_m"]
        ),
        "source_geometry_modified": False,
        "registration_repair_adopted": False,
        "boundary": (
            "The pinned raw BodyParts3D meshes already contain the measured crossing triangle "
            "pairs. This localizes the crossing to source geometry rather than a new rendering "
            "artifact; it does not determine intended regional seam ownership, clinical anatomy, "
            "penetration depth, volume, or mechanics. A registration-only transform is not a "
            "source-backed correction."
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
    parser.add_argument("--source-archive", type=Path,
                        default=ROOT / "Sources/partof_BP3D_4.0_obj_99.zip")
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / "Docs/media/whole-visceral-coverage-20260929/payload-manifest.json")
    parser.add_argument("--overlap-receipt", type=Path,
                        default=ROOT / "Docs/media/whole-body-organ-overlap-20261002/receipt-v4.json.gz")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = audit(arguments.source_archive, arguments.source_manifest,
                       arguments.overlap_receipt)
        _write_immutable(arguments.output, result)
        print(json.dumps({"status": result["status"],
                          "source_pair": result["source_pair"]["exact_intersecting_triangle_pairs"],
                          "compiled_pair": result["compiled_pair"]["exact_intersecting_triangle_pairs"],
                          "output": str(arguments.output.resolve())}, sort_keys=True), flush=True)
        return 0
    except (human.ImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"whole-body-source-overlap-pair: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
