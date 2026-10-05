"""Compile CT vessel-mask slice profiles without admitting lumens or flow.

The outputs are exact per-voxel-area and centroid measurements in planes
normal to the source scan's third voxel axis. They are not a medial-axis
centreline, a vessel-wall/lumen distinction, or a perfusion field.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, BinaryIO
import zipfile

from . import healthy_total_body_ct_source as source
from .model import ImportError as HumanImportError
from .physiology import canonical

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "numi.healthy-total-body-ct-vessel-plane-profile-candidate.v1"
DEFAULT_ARCHIVE = source.ROOT / "Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip"
DEFAULT_SOURCE_CONFIG = source.SOURCE_CONFIG
DEFAULT_INTAKE = ROOT / "Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json"
VESSEL_LABELS = {2: "Aorta", 11: "VCI"}
DEFAULT_SCAN_IDS = ("001", "002")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("CT vessel-plane profiles: " + message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HumanImportError(f"CT vessel-plane profiles: cannot read {label}") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _cross_area_and_normal(affine: list[list[float]]) -> tuple[float, list[float]]:
    a = [affine[row][0] for row in range(3)]
    b = [affine[row][1] for row in range(3)]
    cross = [a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]]
    magnitude = math.sqrt(sum(value * value for value in cross))
    require(math.isfinite(magnitude) and magnitude > 0.0,
            "source voxel plane has zero physical area")
    return magnitude, [value / magnitude for value in cross]


def _world_point(affine: list[list[float]], i: float, j: float, k: int) -> list[float]:
    return [sum(affine[row][axis] * (i, j, k)[axis] for axis in range(3)) + affine[row][3]
            for row in range(3)]


def _plane_four_connected_components(mask: Any, np: Any) -> list[list[int]]:
    """Return 4-neighbour component voxel indices for one binary plane."""
    ny, nx = mask.shape
    active = mask.reshape(-1)
    visited = bytearray(active.size)
    components: list[list[int]] = []
    for start in np.flatnonzero(active).tolist():
        if visited[start]:
            continue
        visited[start] = 1
        pending = [start]
        cells: list[int] = []
        while pending:
            index = pending.pop()
            cells.append(index)
            row, column = divmod(index, nx)
            neighbors = []
            if column > 0:
                neighbors.append(index - 1)
            if column + 1 < nx:
                neighbors.append(index + 1)
            if row > 0:
                neighbors.append(index - nx)
            if row + 1 < ny:
                neighbors.append(index + nx)
            for neighbor in neighbors:
                if active[neighbor] and not visited[neighbor]:
                    visited[neighbor] = 1
                    pending.append(neighbor)
        components.append(cells)
    return components


def _root(parents: list[int], value: int) -> int:
    while parents[value] != value:
        parents[value] = parents[parents[value]]
        value = parents[value]
    return value


def _union(parents: list[int], sizes: list[int], left: int, right: int) -> None:
    a = _root(parents, left)
    b = _root(parents, right)
    if a == b:
        return
    if sizes[a] < sizes[b]:
        a, b = b, a
    parents[b] = a
    sizes[a] += sizes[b]


def _connectivity_result(parents: list[int], sizes: list[int]) -> dict[str, Any]:
    components: dict[int, int] = {}
    for node in range(1, len(parents)):
        root = _root(parents, node)
        components[root] = sizes[root]
    component_sizes = sorted(components.values(), reverse=True)
    total = sum(component_sizes)
    largest = component_sizes[0] if component_sizes else 0
    return {
        "connectivity_definition": "6-neighbour source voxel face adjacency",
        "component_count": len(component_sizes),
        "component_voxel_counts_descending": component_sizes,
        "largest_component_voxel_count": largest,
        "largest_component_fraction": largest / total if total else 0.0,
    }


def _scan_profiles(
    stream: BinaryIO,
    *,
    member_name: str,
    expected_nifti_sha256: str,
    expected_counts: dict[str, Any],
    expected_scan: dict[str, Any],
    expected_shape_ijk: list[int],
    label_names: dict[int, str],
    scan_id: str,
    np: Any,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with gzip.GzipFile(fileobj=stream, mode="rb") as image:
        header = image.read(source.NIFTI_HEADER_BYTES)
        require(len(header) == source.NIFTI_HEADER_BYTES, f"{member_name} header is truncated")
        digest.update(header)
        info = source.parse_nifti_header(header)
        voxel_volume_mm3 = float(info["voxel_volume_in_source_units_cubed"])
        require(info["spatial_units"] == "mm", f"{member_name} is not in millimetres")
        require(info["shape_ijk"] == expected_shape_ijk,
                f"{member_name} shape differs from the pinned intake")
        require(all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
                    for actual, expected in zip(info["voxel_spacing"],
                                                expected_scan.get("voxel_spacing_mm", []), strict=True)),
                f"{member_name} voxel spacing differs from the pinned intake")
        expected_affine = expected_scan.get("voxel_to_world_affine")
        require(isinstance(expected_affine, list) and len(expected_affine) == 4 and
                all(isinstance(row, list) and len(row) == 4 for row in expected_affine) and
                all(math.isclose(info["voxel_to_world_affine"][row][column],
                                 float(expected_affine[row][column]), rel_tol=0.0, abs_tol=1e-6)
                    for row in range(4) for column in range(4)),
                f"{member_name} voxel-to-world affine differs from the pinned intake")
        require(math.isclose(info["label_scaling_slope"],
                             float(expected_scan.get("label_scaling_slope", 1.0)), rel_tol=0.0, abs_tol=1e-7) and
                math.isclose(info["label_scaling_intercept"],
                             float(expected_scan.get("label_scaling_intercept", 0.0)), rel_tol=0.0, abs_tol=1e-7),
                f"{member_name} label scaling differs from the pinned intake")
        require(math.isclose(voxel_volume_mm3,
                             float(expected_scan.get("voxel_volume_mm3", -1.0)),
                             rel_tol=0.0, abs_tol=1e-6),
                f"{member_name} voxel volume differs from the pinned intake")
        nx, ny, nz = info["shape_ijk"]
        affine = info["voxel_to_world_affine"]
        plane_area_mm2, plane_normal_ras = _cross_area_and_normal(affine)
        require(math.isclose(plane_area_mm2 * math.sqrt(sum(affine[row][2] ** 2 for row in range(3))),
                             voxel_volume_mm3, rel_tol=1e-6, abs_tol=1e-9),
                "oblique or sheared NIfTI slice axis is not orthogonal to the image plane")
        endian = "<f4" if info["byte_order"] == "little" else ">f4"
        row_bytes = nx * ny * 4
        slope = info["label_scaling_slope"]
        intercept = info["label_scaling_intercept"]
        counts = {label_id: 0 for label_id in VESSEL_LABELS}
        profiles = {label_id: [] for label_id in VESSEL_LABELS}
        connectivity = {
            label_id: {"previous_plane": np.zeros((ny, nx), dtype=np.int32),
                       "parents": [0], "sizes": [0]}
            for label_id in VESSEL_LABELS
        }

        for k in range(nz):
            raw = image.read(row_bytes)
            require(len(raw) == row_bytes, f"{member_name} ends inside voxel plane {k}")
            digest.update(raw)
            plane = np.frombuffer(raw, dtype=endian).reshape((ny, nx))
            if slope != 1.0 or intercept != 0.0:
                plane = plane * slope + intercept
            require(bool(np.isfinite(plane).all()), f"{member_name} has non-finite label values")
            rounded = np.rint(plane)
            require(bool(np.equal(plane, rounded).all()), f"{member_name} has non-integer label values")

            for label_id in VESSEL_LABELS:
                yy, xx = np.nonzero(rounded == label_id)
                count = int(xx.size)
                counts[label_id] += count
                mask = rounded == label_id
                plane_components = _plane_four_connected_components(mask, np)
                connection = connectivity[label_id]
                current_plane = np.zeros((ny, nx), dtype=np.int32)
                current_flat = current_plane.reshape(-1)
                for cells in plane_components:
                    node = len(connection["parents"])
                    connection["parents"].append(node)
                    connection["sizes"].append(len(cells))
                    current_flat[cells] = node
                current_flat = current_plane.reshape(-1)
                previous_flat = connection["previous_plane"].reshape(-1)
                overlaps = np.flatnonzero((current_flat > 0) & (previous_flat > 0))
                for index in overlaps.tolist():
                    _union(connection["parents"], connection["sizes"],
                           int(current_flat[index]), int(previous_flat[index]))
                connection["previous_plane"] = current_plane
                if not count:
                    continue
                centroid_i = float(xx.mean(dtype=np.float64))
                centroid_j = float(yy.mean(dtype=np.float64))
                profiles[label_id].append({
                    "k": k,
                    "voxel_count": count,
                    "plane_occupancy_area_mm2": count * plane_area_mm2,
                    "mask_centroid_ras_mm": _world_point(affine, centroid_i, centroid_j, k),
                })
        require(image.read(1) == b"", f"{member_name} has trailing data after its declared NIfTI dimensions")

    require(digest.hexdigest() == expected_nifti_sha256,
            f"{member_name} uncompressed NIfTI SHA-256 differs from the pinned intake")
    np_digest = digest.hexdigest()
    vessel_rows = []
    for label_id in VESSEL_LABELS:
        expected_count = expected_counts.get(str(label_id))
        require(type(expected_count) is int and counts[label_id] == expected_count,
                f"scan {scan_id} {VESSEL_LABELS[label_id]} voxel count differs from the pinned intake")
        rows = profiles[label_id]
        require(bool(rows), f"scan {scan_id} {VESSEL_LABELS[label_id]} mask is empty")
        k_values = [row["k"] for row in rows]
        k_gaps = [right - left - 1 for left, right in zip(k_values, k_values[1:])]
        areas = [row["plane_occupancy_area_mm2"] for row in rows]
        sorted_areas = sorted(areas)
        middle = len(sorted_areas) // 2
        median_area = (sorted_areas[middle] if len(sorted_areas) % 2
                       else 0.5 * (sorted_areas[middle - 1] + sorted_areas[middle]))
        vessel_rows.append({
            "label_id": label_id,
            "label_name": label_names[label_id],
            "source_voxel_count": counts[label_id],
            "voxel_occupancy_volume_ml": counts[label_id] * voxel_volume_mm3 / 1000.0,
            "plane_occupancy_area_summary_mm2": {
                "minimum": min(areas), "median": median_area, "maximum": max(areas),
            },
            "occupied_plane_count": len(rows),
            "k_range_inclusive": [k_values[0], k_values[-1]],
            "internal_empty_plane_count": sum(k_gaps),
            "maximum_internal_empty_plane_run": max(k_gaps, default=0),
            "source_mask_face_connectivity_6": _connectivity_result(
                connectivity[label_id]["parents"], connectivity[label_id]["sizes"]),
            "mask_centroid_profile": rows,
            "boundary": (
                "The profile is the label-mask area and centroid in each source k-axis plane. "
                "It is not a vessel-wall inner surface, an orthogonal-to-path lumen area, "
                "a 3-D medial-axis centreline, or a connected-flow claim."
            ),
        })
    return {
        "scan_id": scan_id,
        "source_nifti_member": member_name,
        "source_nifti_uncompressed_sha256": np_digest,
        "shape_ijk": info["shape_ijk"],
        "voxel_spacing_mm": info["voxel_spacing"],
        "voxel_to_world_affine_ras_mm": affine,
        "slice_plane_normal_ras": plane_normal_ras,
        "slice_plane_voxel_area_mm2": plane_area_mm2,
        "voxel_volume_mm3": voxel_volume_mm3,
        "vessels": vessel_rows,
    }


def compile_profiles(
    *,
    archive_path: Path = DEFAULT_ARCHIVE,
    intake_receipt: Path = DEFAULT_INTAKE,
    source_config: Path = DEFAULT_SOURCE_CONFIG,
    scan_ids: tuple[str, ...] = DEFAULT_SCAN_IDS,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    intake_receipt = Path(intake_receipt)
    source_config = Path(source_config)
    require(archive_path.is_file() and not archive_path.is_symlink(), "source archive is not a regular file")
    require(intake_receipt.is_file() and not intake_receipt.is_symlink(), "intake receipt is not a regular file")
    require(source_config.is_file() and not source_config.is_symlink(), "source config is not a regular file")
    require(bool(scan_ids) and len(set(scan_ids)) == len(scan_ids), "scan IDs are empty or repeated")

    np = source._load_numpy()
    config = _read_json(source_config, "source config")
    intake = _read_json(intake_receipt, "source intake receipt")
    require(config.get("schema") == "numi.human.external-segmentation-source.v1" and
            config.get("source_id") == "healthy_total_body_cts_v3",
            "unsupported CT source config")
    require(intake.get("schema") == source.SCHEMA and intake.get("status") == "partial_source_inventory",
            "unsupported CT source intake receipt")
    archive_sha256 = _sha_file(archive_path)
    require(archive_path.stat().st_size == config.get("archive_bytes") and
            archive_sha256 == config.get("archive_sha256") == intake.get("source", {}).get("archive_sha256"),
            "source archive differs from the registered TCIA release or intake")
    require(intake.get("source", {}).get("source_config_sha256") == _sha(canonical(config)),
            "source intake was compiled from a different canonical source config")

    label_dictionary_member = config.get("label_dictionary_member")
    scan_directory = config.get("nifti_member_directory")
    require(isinstance(label_dictionary_member, str) and isinstance(scan_directory, str),
            "source config omits the label dictionary or NIfTI directory")
    expected_scans = {row.get("scan_id"): row for row in intake.get("scans", [])
                      if isinstance(row, dict) and isinstance(row.get("scan_id"), str)}
    require(all(scan_id in expected_scans for scan_id in scan_ids), "requested scan is absent from intake")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            require(label_dictionary_member in names, "registered label dictionary is absent from archive")
            labels = source.read_label_dictionary(archive.read(label_dictionary_member))
            label_names = {row["label_id"]: row["name"] for row in labels}
            require(all(label_names.get(label_id) == expected_name
                        for label_id, expected_name in VESSEL_LABELS.items()),
                    "registered label dictionary vessel names changed")
            np_scans = []
            for scan_id in scan_ids:
                scan = expected_scans[scan_id]
                member_name = scan.get("nifti_member")
                require(isinstance(member_name, str) and member_name.startswith(scan_directory) and
                        member_name in names, f"scan {scan_id} NIfTI member is outside the registered archive path")
                with archive.open(member_name) as member_stream:
                    np_scans.append(_scan_profiles(
                        member_stream,
                        member_name=member_name,
                        expected_nifti_sha256=scan.get("nifti_uncompressed_sha256"),
                        expected_counts=scan.get("label_voxel_counts", {}),
                        expected_scan=scan,
                        expected_shape_ijk=config.get("expected_shape_ijk"),
                        label_names=label_names,
                        scan_id=scan_id,
                        np=np,
                    ))
    except (zipfile.BadZipFile, KeyError) as error:
        raise HumanImportError("CT vessel-plane profiles: source archive or scan member is malformed") from error

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.healthy-total-body-ct-vessel-plane-profiles.2",
        "compiler_runtime": {
            "python": sys.version.split()[0],
            "numpy": str(np.__version__),
        },
        "source": {
            "source_id": config["source_id"],
            "archive_path": str(archive_path.relative_to(ROOT) if archive_path.is_relative_to(ROOT) else archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "source_config_sha256": _sha(source_config.read_bytes()),
            "compiler_source_path": str(Path(__file__).relative_to(ROOT)),
            "compiler_source_sha256": _sha(Path(__file__).read_bytes()),
            "nifti_header_parser_source_sha256": _sha(Path(source.__file__).read_bytes()),
            "intake_receipt_path": str(intake_receipt.relative_to(ROOT) if intake_receipt.is_relative_to(ROOT) else intake_receipt),
            "intake_receipt_sha256": _sha(intake_receipt.read_bytes()),
            "registered_label_dictionary_member": label_dictionary_member,
        },
        "section_definition": {
            "kind": "source-voxel-k-axis planes",
            "area": "occupied source voxel count multiplied by the exact NIfTI i-j basis parallelogram area",
            "centroid": "arithmetic mean of occupied voxel-centre RAS coordinates within each plane",
            "limitations": [
                "no 3-D medial-axis or path-normal cross-section is computed",
                "automatic CT segmentation does not distinguish vessel lumen from vessel wall",
                "mask voxel face-connectivity does not prove a biological lumen or pressure-flow continuity",
            ],
        },
        "scans": np_scans,
        "qualification": {
            "registered_archive_hash_matches": True,
            "source_intake_config_hash_matches": True,
            "selected_uncompressed_nifti_hashes_match": True,
            "source_label_dictionary_names_match": True,
            "selected_voxel_counts_match_intake": True,
            "exact_slice_plane_occupancy_areas_and_mask_centroids": True,
            "source_mask_6_connected_components_recomputed": True,
            "single_face_connected_source_mask_per_label": all(
                row["source_mask_face_connectivity_6"]["component_count"] == 1
                for scan in np_scans for row in scan["vessels"]
            ),
            "anatomical_lumen_connectivity_or_branch_flow": False,
            "vessel_lumen_or_wall_anatomy_verified": False,
            "path_normal_lumen_cross_sections": False,
            "medial_axis_centreline": False,
            "subject_binding_or_calibration": False,
            "pressure_flow_or_tissue_exchange": False,
        },
        "boundary": (
            "These are source-bound measurements of the automatic Aorta and VCI label masks "
            "in CT slice planes. They are geometry diagnostics only and do not establish a "
            "connected lumen, anatomical centreline, blood ownership, perfusion, mechanics, "
            "subject identity, or physiological qualification."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--intake-receipt", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument("--scan-id", action="append", dest="scan_ids",
                        help="registered CT scan identifier; may be repeated (default: 001 and 002)")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    try:
        result = compile_profiles(
            archive_path=arguments.archive,
            intake_receipt=arguments.intake_receipt,
            source_config=arguments.source_config,
            scan_ids=tuple(arguments.scan_ids or DEFAULT_SCAN_IDS),
        )
        output = arguments.output.resolve()
        receipt_sha = immutable_write(output, result)
        print(json.dumps({
            "schema": SCHEMA,
            "output": str(output),
            "sha256": receipt_sha,
            "scan_count": len(result["scans"]),
            "vessel_mask_count": sum(len(scan["vessels"]) for scan in result["scans"]),
            "source_voxel_count": sum(row["source_voxel_count"]
                                       for scan in result["scans"] for row in scan["vessels"]),
            "exact_slice_profiles": result["qualification"]["exact_slice_plane_occupancy_areas_and_mask_centroids"],
            "source_mask_single_component": result["qualification"]["single_face_connected_source_mask_per_label"],
            "lumen_or_branch_flow_admitted": False,
            "pressure_flow_admitted": False,
        }, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        print(f"CT vessel-plane profiles: {error}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
