#!/usr/bin/env python3
"""Independent replay of CT vessel-plane mask profiles and face connectivity."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
LABELS = {2: "Aorta", 11: "VCI"}
STRUCTURE_4 = ndimage.generate_binary_structure(2, 1)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def vector_area(affine: list[list[float]]) -> float:
    a = np.asarray([row[0] for row in affine[:3]], dtype=np.float64)
    b = np.asarray([row[1] for row in affine[:3]], dtype=np.float64)
    return float(np.linalg.norm(np.cross(a, b)))


def union(parents: list[int], weights: list[int], left: int, right: int) -> None:
    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    a, b = find(left), find(right)
    if a == b:
        return
    if weights[a] < weights[b]:
        a, b = b, a
    parents[b] = a
    weights[a] += weights[b]


def face_components(sizes: list[int], parents: list[int]) -> list[int]:
    totals: dict[int, int] = {}
    for node in range(1, len(parents)):
        root = node
        while parents[root] != root:
            parents[root] = parents[parents[root]]
            root = parents[root]
        totals[root] = sizes[root]
    return sorted(totals.values(), reverse=True)


def check_close(actual: float, expected: float, name: str) -> None:
    require(abs(float(actual) - float(expected)) <= 1e-8 * max(1.0, abs(float(expected))),
            f"{name} differs: actual={actual}, expected={expected}")


def verify(candidate_path: Path, output_path: Path) -> dict:
    candidate_path = candidate_path.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(candidate.get("schema") == "numi.healthy-total-body-ct-vessel-plane-profile-candidate.v1",
            "unsupported candidate schema")
    source = candidate["source"]
    archive_path = Path(source["archive_path"])
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    intake_path = Path(source["intake_receipt_path"])
    if not intake_path.is_absolute():
        intake_path = ROOT / intake_path
    config_path = ROOT / "config/healthy-total-body-cts-source.v1.json"
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(file_sha(archive_path) == source["archive_sha256"] == config["archive_sha256"],
            "source archive hash mismatch")
    require(file_sha(intake_path) == source["intake_receipt_sha256"], "intake receipt hash mismatch")
    require(file_sha(config_path) == source["source_config_sha256"], "source config hash mismatch")
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode()
    require(sha(canonical_config) == intake["source"]["source_config_sha256"],
            "intake source-config identity mismatch")
    require(intake["source"]["archive_sha256"] == source["archive_sha256"],
            "intake source-archive identity mismatch")
    require(file_sha(ROOT / source["compiler_source_path"]) == source["compiler_source_sha256"],
            "candidate compiler source hash mismatch")
    intake_by_id = {row["scan_id"]: row for row in intake["scans"]}
    checked = []
    total_voxels = 0

    with zipfile.ZipFile(archive_path) as archive:
        dictionary = archive.read(config["label_dictionary_member"])
        require(sha(dictionary) == intake["source"]["label_dictionary_sha256"],
                "registered label dictionary hash mismatch")
        names_by_id = {row["label_id"]: row["name"] for row in intake["labels"]}
        require(all(names_by_id.get(label_id) == name for label_id, name in LABELS.items()),
                "intake vessel label names mismatch")
        for scan_result in candidate["scans"]:
            scan_id = scan_result["scan_id"]
            scan = intake_by_id[scan_id]
            member_name = scan["nifti_member"]
            affine = scan["voxel_to_world_affine"]
            area_per_voxel = vector_area(affine)
            expected_affine = np.asarray(affine, dtype=np.float64)
            require(scan_result["source_nifti_member"] == member_name, "candidate NIfTI member mismatch")
            require(scan_result["shape_ijk"] == config["expected_shape_ijk"], "candidate shape mismatch")
            require(np.allclose(scan_result["voxel_to_world_affine_ras_mm"], expected_affine,
                                rtol=0.0, atol=1e-6), "candidate source affine mismatch")
            require(np.allclose(scan_result["voxel_spacing_mm"], scan["voxel_spacing_mm"],
                                rtol=0.0, atol=1e-6), "candidate voxel spacing mismatch")
            check_close(scan_result["voxel_volume_mm3"], scan["voxel_volume_mm3"],
                        "candidate voxel volume")
            expected = {label_id: int(scan["label_voxel_counts"][str(label_id)]) for label_id in LABELS}
            rows_by_label = {
                row["label_id"]: {profile["k"]: profile for profile in row["mask_centroid_profile"]}
                for row in scan_result["vessels"]
            }
            count_by_label = {label_id: 0 for label_id in LABELS}
            areas_by_label = {label_id: [] for label_id in LABELS}
            previous_global = {label_id: None for label_id in LABELS}
            parents = {label_id: [0] for label_id in LABELS}
            weights = {label_id: [0] for label_id in LABELS}
            profile_count = {label_id: 0 for label_id in LABELS}

            with archive.open(member_name) as compressed, gzip.GzipFile(fileobj=compressed) as image:
                header = image.read(352)
                require(len(header) == 352, "NIfTI header truncated")
                endian = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
                require(struct.unpack_from(endian + "i", header)[0] == 348, "NIfTI byte order unsupported")
                dimensions = struct.unpack_from(endian + "8h", header, 40)
                nx, ny, nz = (int(dimensions[index]) for index in (1, 2, 3))
                require([nx, ny, nz] == config["expected_shape_ijk"], "NIfTI dimensions mismatch")
                datatype = struct.unpack_from(endian + "h", header, 70)[0]
                bitpix = struct.unpack_from(endian + "h", header, 72)[0]
                require((datatype, bitpix) == (16, 32), "NIfTI label encoding mismatch")
                slope, intercept = struct.unpack_from(endian + "2f", header, 112)
                if slope == 0:
                    slope, intercept = 1.0, 0.0
                digest = hashlib.sha256(header)
                for k in range(nz):
                    raw = image.read(nx * ny * 4)
                    require(len(raw) == nx * ny * 4, f"NIfTI slice {k} truncated")
                    digest.update(raw)
                    plane = np.frombuffer(raw, dtype=endian + "f4").reshape(ny, nx)
                    if slope != 1.0 or intercept != 0.0:
                        plane = plane * slope + intercept
                    rounded = np.rint(plane)
                    require(np.isfinite(rounded).all() and np.equal(plane, rounded).all(),
                            "non-finite or non-integer label plane")
                    for label_id in LABELS:
                        mask = rounded == label_id
                        yy, xx = np.nonzero(mask)
                        count = int(xx.size)
                        count_by_label[label_id] += count
                        if count:
                            profile_count[label_id] += 1
                            areas_by_label[label_id].append(count * area_per_voxel)
                            profile = rows_by_label[label_id].get(k)
                            require(profile is not None, f"scan {scan_id} {LABELS[label_id]} lacks k={k} row")
                            require(profile["voxel_count"] == count, "per-plane voxel count mismatch")
                            check_close(profile["plane_occupancy_area_mm2"], count * area_per_voxel,
                                        "per-plane occupancy area")
                            center = np.asarray([xx.mean(), yy.mean(), k, 1.0], dtype=np.float64)
                            center_ras = expected_affine[:3] @ center
                            require(np.allclose(profile["mask_centroid_ras_mm"], center_ras,
                                                rtol=0.0, atol=1e-8), "per-plane RAS centroid mismatch")

                        # Independent 2-D 4-connectivity labels are joined only
                        # where adjacent k planes share the same occupied voxel.
                        labels, component_count = ndimage.label(mask, structure=STRUCTURE_4)
                        prior = previous_global[label_id]
                        current = np.zeros((ny, nx), dtype=np.int32)
                        if component_count:
                            voxel_counts = np.bincount(labels.ravel(), minlength=component_count + 1)
                            nodes = [0]
                            for component_id in range(1, component_count + 1):
                                node = len(parents[label_id])
                                parents[label_id].append(node)
                                weights[label_id].append(int(voxel_counts[component_id]))
                                nodes.append(node)
                            current = np.asarray(nodes, dtype=np.int32)[labels]
                        if prior is not None:
                            overlapping = np.flatnonzero((current > 0) & (prior > 0))
                            for flat_index in overlapping.tolist():
                                union(parents[label_id], weights[label_id],
                                      int(current.reshape(-1)[flat_index]),
                                      int(prior.reshape(-1)[flat_index]))
                        previous_global[label_id] = current
                require(image.read(1) == b"", "unexpected NIfTI trailing data")
                require(digest.hexdigest() == scan["nifti_uncompressed_sha256"] ==
                        scan_result["source_nifti_uncompressed_sha256"], "source NIfTI hash mismatch")

            for label_id, name in LABELS.items():
                vessel = next(row for row in scan_result["vessels"] if row["label_id"] == label_id)
                require(count_by_label[label_id] == expected[label_id] == vessel["source_voxel_count"],
                        f"scan {scan_id} {name} total voxel count mismatch")
                require(len(rows_by_label[label_id]) == profile_count[label_id], "profile has missing or extra plane")
                require(vessel["occupied_plane_count"] == profile_count[label_id], "occupied plane count mismatch")
                sizes = face_components(weights[label_id], parents[label_id])
                reported = vessel["source_mask_face_connectivity_6"]
                require(reported["component_voxel_counts_descending"] == sizes,
                        f"scan {scan_id} {name} 6-neighbour component counts differ")
                areas = sorted(areas_by_label[label_id])
                middle = len(areas) // 2
                median = areas[middle] if len(areas) % 2 else 0.5 * (areas[middle - 1] + areas[middle])
                summary = vessel["plane_occupancy_area_summary_mm2"]
                check_close(summary["minimum"], areas[0], "minimum profile area")
                check_close(summary["median"], median, "median profile area")
                check_close(summary["maximum"], areas[-1], "maximum profile area")
                total_voxels += count_by_label[label_id]
            checked.append({
                "scan_id": scan_id,
                "source_nifti_sha256": scan_result["source_nifti_uncompressed_sha256"],
                "vessel_labels": len(LABELS),
                "source_voxels": sum(count_by_label.values()),
                "six_neighbour_component_counts": {
                    name: next(row for row in scan_result["vessels"] if row["label_id"] == label_id)
                    ["source_mask_face_connectivity_6"]["component_count"]
                    for label_id, name in LABELS.items()
                },
            })

    candidate_sha = sha(candidate_path.read_bytes())
    result = {
        "schema": "numi.healthy-total-body-ct-vessel-plane-profile-independent-verification.v1",
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "candidate_sha256": candidate_sha,
        "verifier": "independent NumPy/SciPy source-mask reread and per-plane connectivity reconstruction",
        "scans": checked,
        "source_voxel_count": total_voxels,
        "verification": {
            "archive_and_intake_hashes_match": True,
            "raw_nifti_hashes_and_geometry_match": True,
            "all_aorta_vci_plane_areas_and_centroids_match": True,
            "all_6_neighbour_source_mask_component_counts_match": True,
            "lumens_or_flow_admitted": False,
        },
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        require(output_path.read_bytes() == payload, "verification receipt is immutable")
    else:
        with output_path.open("xb") as stream:
            stream.write(payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "Docs/media/healthy-total-body-ct-vessel-plane-profiles-20261005/candidate.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Docs/media/healthy-total-body-ct-vessel-plane-profiles-20261005/independent-verification.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.candidate, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
