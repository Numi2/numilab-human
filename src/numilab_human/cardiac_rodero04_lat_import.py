"""Import the published Rodero-04 whole-heart LAT field by exact cell identity.

This source-data importer transfers the Healthy case's cell-centred LAT values
from the published VTK mesh to the paired TetGen mechanics mesh. It does not
run electrophysiology or qualify a Numi heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError


DATA_ARCHIVE = {
    "md5": "e173cde6c133a90ed429716a01b7e2fa",
    "sha256": "24bf1a9161c397dfabe171d9544dde1a02aadf6d9c3c607434bd5e5b453fbbc1",
}
CARDIOMECHANICS_ARCHIVE = {
    "md5": "ca0402d280edb7904c520ff9b1cdcfd5",
    "sha256": "bff98692ba404983f15faa839365490961c896f93998b7ba7174dfbdf8468e06",
}
MATLAB_TOOLS_ARCHIVE = {
    "md5": "65839742488f67c89cbc9e770d251b97",
    "sha256": "353c8aaf27ab1b077ed5a7fbc3c7a7742a9245e61b130e62fe98d21397a9b669",
}
SOURCE_MEMBERS = {
    "vtk": "data/04_M_Robin.vtk",
    "nodes": "CardioMechanics/tetgen/04_WH_T4.node",
    "elements": "CardioMechanics/tetgen/04_WH_T4.ele",
    "lat": "CardioMechanics/LATFiles/LAT_Healthy_04.txt",
    "settings": "CardioMechanics/settings/WH_Healthy_04.xml",
    "lat_mapping_script": "MatlabTools/LAT_openCARPToCardioMechanics.m",
}
SOURCE_RECORD = "https://zenodo.org/records/21934382"
MATLAB_SCRIPT_SHA256 = (
    "8f82b6b395ddfaa2ddd3a6ebfc16a2827460f8ad38f195d4b447a56cfd309066"
)
VTK_EXPECTED_COUNTS = {
    "points": 103509,
    "cells": 543571,
    "tetrahedra": 487583,
    "triangles": 55988,
}
TETGEN_EXPECTED_COUNTS = {"nodes": 103509, "elements": 487583}
COORDINATE_TOLERANCE_MM = 1.0e-5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("Rodero-04 LAT import: " + message)


def _hash_file(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def _archive_identity(
    path: Path, expected: dict[str, str], label: str
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    size, md5, sha256 = _hash_file(path)
    _require(
        md5 == expected["md5"] and sha256 == expected["sha256"],
        f"{label} does not match the pinned Zenodo archive",
    )
    return {"file": path.name, "bytes": size, "md5": md5, "sha256": sha256}


def _member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as error:
        raise HumanImportError(
            f"Rodero-04 LAT import: missing source member {name}"
        ) from error
    _require(not info.is_dir() and info.file_size > 0, f"empty source member {name}")
    try:
        return archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise HumanImportError(
            f"Rodero-04 LAT import: invalid source member {name}"
        ) from error


def _numeric_block(
    lines: list[str], start: int, stop: int, count: int, dtype: Any, label: str
):
    import numpy as np

    values = np.fromstring("\n".join(lines[start:stop]), sep=" ", dtype=dtype)
    _require(values.size == count, f"{label} count differs from its VTK header")
    return values


def _parse_vtk(raw: bytes):
    import numpy as np

    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise HumanImportError(
            "Rodero-04 LAT import: VTK member is not ASCII"
        ) from error
    point_index = next(
        (i for i, line in enumerate(lines) if line.startswith("POINTS ")), None
    )
    cells_index = next(
        (i for i, line in enumerate(lines) if line.startswith("CELLS ")), None
    )
    _require(
        point_index is not None and cells_index is not None,
        "VTK points or unstructured cells are missing",
    )
    point_header = lines[point_index].split()
    _require(
        len(point_header) == 3 and point_header[2] == "float",
        "unsupported VTK point declaration",
    )
    point_count = int(point_header[1])
    points = _numeric_block(
        lines, point_index + 1, cells_index, point_count * 3, np.float64, "VTK points"
    ).reshape(-1, 3)

    cell_header = lines[cells_index].split()
    _require(len(cell_header) == 3, "malformed VTK CELLS declaration")
    offset_count, connectivity_count = int(cell_header[1]), int(cell_header[2])
    cell_count = offset_count - 1
    _require(
        cell_count == VTK_EXPECTED_COUNTS["cells"],
        "VTK offset count differs from Rodero-04 source",
    )
    offsets_index = cells_index + 1
    _require(
        lines[offsets_index].startswith("OFFSETS "),
        "VTK cells do not use the pinned offsets/connectivity encoding",
    )
    connectivity_index = next(
        (
            i
            for i in range(offsets_index + 1, len(lines))
            if lines[i].startswith("CONNECTIVITY ")
        ),
        None,
    )
    _require(connectivity_index is not None, "VTK connectivity declaration is missing")
    offsets = _numeric_block(
        lines,
        offsets_index + 1,
        connectivity_index,
        cell_count + 1,
        np.int64,
        "VTK offsets",
    )
    types_index = next(
        (
            i
            for i in range(connectivity_index + 1, len(lines))
            if lines[i].startswith("CELL_TYPES ")
        ),
        None,
    )
    _require(types_index is not None, "VTK cell type declaration is missing")
    connectivity = _numeric_block(
        lines,
        connectivity_index + 1,
        types_index,
        connectivity_count,
        np.int64,
        "VTK connectivity",
    )
    type_header = lines[types_index].split()
    _require(
        len(type_header) == 2 and int(type_header[1]) == cell_count,
        "VTK cell type count differs from CELLS",
    )
    types = _numeric_block(
        lines,
        types_index + 1,
        types_index + 1 + cell_count,
        cell_count,
        np.int32,
        "VTK cell types",
    )
    _require(
        bool(np.isfinite(points).all())
        and offsets[0] == 0
        and offsets[-1] == connectivity_count
        and bool((np.diff(offsets) > 0).all())
        and bool(((connectivity >= 0) & (connectivity < point_count)).all()),
        "VTK point or cell topology is malformed",
    )
    sizes = np.diff(offsets)
    tetrahedron_cells = np.flatnonzero(types == 10)
    triangle_cells = np.flatnonzero(types == 5)
    _require(
        len(tetrahedron_cells) == VTK_EXPECTED_COUNTS["tetrahedra"]
        and len(triangle_cells) == VTK_EXPECTED_COUNTS["triangles"]
        and len(tetrahedron_cells) + len(triangle_cells) == cell_count
        and bool((sizes[tetrahedron_cells] == 4).all())
        and bool((sizes[triangle_cells] == 3).all()),
        "VTK cell type/topology inventory differs from Rodero-04 source",
    )
    tetrahedra = np.stack(
        [connectivity[offsets[tetrahedron_cells] + corner] for corner in range(4)],
        axis=1,
    ).astype("<u4", copy=False)
    _require(
        point_count == VTK_EXPECTED_COUNTS["points"],
        "VTK point count differs from Rodero-04 source",
    )
    return points, tetrahedron_cells, tetrahedra, cell_count


def map_cell_connectivity(vtk_tetrahedra, tetgen_tetrahedra):
    """Return each TetGen element's exact source VTK cell index."""
    import numpy as np

    vtk_tetrahedra = np.asarray(vtk_tetrahedra, dtype="<u4")
    tetgen_tetrahedra = np.asarray(tetgen_tetrahedra, dtype="<u4")
    _require(
        vtk_tetrahedra.ndim == 2
        and vtk_tetrahedra.shape[1] == 4
        and tetgen_tetrahedra.ndim == 2
        and tetgen_tetrahedra.shape[1] == 4,
        "tetrahedron arrays must have four node IDs per row",
    )

    def keys(cells):
        ordered = np.ascontiguousarray(np.sort(cells, axis=1), dtype="<u4")
        return ordered.view("V16").reshape(-1)

    vtk_keys, tetgen_keys = keys(vtk_tetrahedra), keys(tetgen_tetrahedra)
    _require(
        len(np.unique(vtk_keys)) == len(vtk_keys)
        and len(np.unique(tetgen_keys)) == len(tetgen_keys),
        "duplicate tetrahedron connectivity prevents a unique source map",
    )
    source_order = np.argsort(vtk_keys)
    sorted_source = vtk_keys[source_order]
    positions = np.searchsorted(sorted_source, tetgen_keys)
    in_bounds = positions < len(sorted_source)
    exact = np.zeros(len(positions), dtype=bool)
    exact[in_bounds] = sorted_source[positions[in_bounds]] == tetgen_keys[in_bounds]
    _require(bool(exact.all()), "TetGen contains cells absent from source VTK topology")
    mapping = source_order[positions].astype("<u4")
    _require(
        len(np.unique(mapping)) == len(mapping),
        "TetGen-to-VTK source cell map is not one-to-one",
    )
    return mapping


def _parse_tetgen(nodes_raw: bytes, elements_raw: bytes):
    import numpy as np

    try:
        node_lines = nodes_raw.decode("ascii").splitlines()
        element_lines = elements_raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise HumanImportError(
            "Rodero-04 LAT import: TetGen members are not ASCII"
        ) from error
    node_header = [int(value) for value in node_lines[0].split()]
    element_header = [int(value) for value in element_lines[0].split()]
    _require(
        node_header == [TETGEN_EXPECTED_COUNTS["nodes"], 3, 1, 0],
        "TetGen node header differs from Rodero-04 source",
    )
    _require(
        element_header == [TETGEN_EXPECTED_COUNTS["elements"], 4, 1],
        "TetGen element header differs from Rodero-04 source",
    )
    nodes = np.loadtxt(io.StringIO("\n".join(node_lines[1:])), dtype=np.float64)
    elements = np.loadtxt(io.StringIO("\n".join(element_lines[1:])), dtype=np.int64)
    _require(
        nodes.shape == (node_header[0], 5) and elements.shape == (element_header[0], 6),
        "TetGen row count or column count differs from its header",
    )
    node_ids = nodes[:, 0].astype(np.int64)
    element_ids = elements[:, 0]
    _require(
        np.array_equal(node_ids, np.arange(1, len(nodes) + 1))
        and np.array_equal(element_ids, np.arange(1, len(elements) + 1)),
        "TetGen IDs are not contiguous and one-based",
    )
    _require(
        bool(np.isfinite(nodes[:, 1:4]).all())
        and bool(((elements[:, 1:5] >= 1) & (elements[:, 1:5] <= len(nodes))).all()),
        "TetGen coordinates or element node IDs are invalid",
    )
    return nodes[:, 1:4], (elements[:, 1:5] - 1).astype("<u4"), elements[:, 5]


def _parse_lat(raw: bytes, expected_cell_count: int):
    import numpy as np

    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise HumanImportError(
            "Rodero-04 LAT import: LAT member is not ASCII"
        ) from error
    header = [int(value) for value in lines[0].split()]
    _require(
        header == [expected_cell_count, 1],
        "LAT header is not one scalar per source VTK cell",
    )
    rows = np.loadtxt(io.StringIO("\n".join(lines[1:])), dtype=np.float64)
    _require(
        rows.shape == (expected_cell_count, 2)
        and np.array_equal(
            rows[:, 0].astype(np.int64), np.arange(1, expected_cell_count + 1)
        )
        and bool(np.isfinite(rows).all())
        and bool((rows[:, 1] >= 0).all()),
        "LAT rows are not finite, nonnegative, contiguous one-based cell records",
    )
    return rows[:, 1]


def _member_digest(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_immutable(path: Path, data: bytes) -> None:
    _require(not path.is_symlink(), f"output symlink {path.name}")
    if path.exists():
        _require(path.read_bytes() == data, f"changed immutable output {path.name}")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def import_source(
    geometry_zip: Path,
    cardiomechanics_zip: Path,
    matlab_tools_zip: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np

    geometry_zip, cardiomechanics_zip, matlab_tools_zip = map(
        lambda path: Path(path).resolve(),
        (geometry_zip, cardiomechanics_zip, matlab_tools_zip),
    )
    output_dir = Path(output_dir).resolve()
    archives = {
        "geometry": _archive_identity(geometry_zip, DATA_ARCHIVE, "data.zip"),
        "cardiomechanics": _archive_identity(
            cardiomechanics_zip, CARDIOMECHANICS_ARCHIVE, "CardioMechanics.zip"
        ),
        "matlab_tools": _archive_identity(
            matlab_tools_zip, MATLAB_TOOLS_ARCHIVE, "MatlabTools.zip"
        ),
    }
    try:
        with zipfile.ZipFile(geometry_zip) as archive:
            vtk_raw = _member(archive, SOURCE_MEMBERS["vtk"])
        with zipfile.ZipFile(cardiomechanics_zip) as archive:
            nodes_raw = _member(archive, SOURCE_MEMBERS["nodes"])
            elements_raw = _member(archive, SOURCE_MEMBERS["elements"])
            lat_raw = _member(archive, SOURCE_MEMBERS["lat"])
            settings_raw = _member(archive, SOURCE_MEMBERS["settings"])
        with zipfile.ZipFile(matlab_tools_zip) as archive:
            script_raw = _member(archive, SOURCE_MEMBERS["lat_mapping_script"])
    except (OSError, zipfile.BadZipFile) as error:
        raise HumanImportError(
            "Rodero-04 LAT import: cannot read a pinned source archive"
        ) from error

    _require(
        hashlib.sha256(script_raw).hexdigest() == MATLAB_SCRIPT_SHA256,
        "LAT conversion source code differs from the reviewed Zenodo script",
    )
    script = script_raw.decode("utf-8")
    _require(
        "lat_V/1000" in script
        and "EleIDs = [1:length(target.cells)]" in script
        and "LATHeader = [size(target.cells,1); 1]" in script,
        "LAT source-unit or cell-index semantics differ from the reviewed script",
    )
    settings = settings_raw.decode("utf-8")
    _require(
        "<FilePath>../LATFiles/LAT_Healthy_04.txt</FilePath>" in settings
        and "<Nodes>../tetgen/04_WH_T4.node</Nodes>" in settings
        and "<Elements>../tetgen/04_WH_T4.ele</Elements>" in settings,
        "Healthy case 04 settings do not bind the pinned LAT and TetGen members",
    )

    points, vtk_tetra_cells, vtk_tetrahedra, vtk_cell_count = _parse_vtk(vtk_raw)
    tetgen_points, tetgen_tetrahedra, tetgen_regions = _parse_tetgen(
        nodes_raw, elements_raw
    )
    maximum_coordinate_delta_mm = float(np.max(np.abs(points - tetgen_points)))
    _require(
        maximum_coordinate_delta_mm <= COORDINATE_TOLERANCE_MM,
        "source VTK and TetGen point IDs do not share one coordinate ordering",
    )
    vtk_cell_for_element = map_cell_connectivity(vtk_tetrahedra, tetgen_tetrahedra)
    source_lat = _parse_lat(lat_raw, vtk_cell_count)
    activation_by_element = source_lat[vtk_cell_for_element]
    source_vtk_cell_ids_1based = (vtk_cell_for_element + 1).astype("<u4")
    _require(
        bool(np.isfinite(activation_by_element).all()),
        "mapped activation field contains nonfinite values",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    lat_path = output_dir / "activation-time-tetgen-elements.f64le"
    map_path = output_dir / "tetgen-element-to-source-vtk-cell.u32le"
    _write_immutable(
        lat_path, activation_by_element.astype("<f8", copy=False).tobytes()
    )
    _write_immutable(map_path, source_vtk_cell_ids_1based.tobytes())
    members = {
        name: {"path": path, **_member_digest(data)}
        for name, path, data in (
            ("vtk", SOURCE_MEMBERS["vtk"], vtk_raw),
            ("nodes", SOURCE_MEMBERS["nodes"], nodes_raw),
            ("elements", SOURCE_MEMBERS["elements"], elements_raw),
            ("lat", SOURCE_MEMBERS["lat"], lat_raw),
            ("settings", SOURCE_MEMBERS["settings"], settings_raw),
            ("lat_mapping_script", SOURCE_MEMBERS["lat_mapping_script"], script_raw),
        )
    }
    report: dict[str, Any] = {
        "schema": "HumanPack.rodero04-source-lat-import.v1",
        "status": "source_cell_activation_mapped_by_exact_tetrahedron_connectivity",
        "source_record": SOURCE_RECORD,
        "source_geometry_id": "Rodero-04",
        "activation_case": "Healthy",
        "source_archives": archives,
        "source_members": members,
        "mesh": {
            "vtk_point_count": len(points),
            "tetgen_node_count": len(tetgen_points),
            "maximum_coordinate_delta_mm": maximum_coordinate_delta_mm,
            "coordinate_tolerance_mm": COORDINATE_TOLERANCE_MM,
            "vtk_cell_count": vtk_cell_count,
            "vtk_tetrahedron_count": len(vtk_tetrahedra),
            "vtk_triangle_count": vtk_cell_count - len(vtk_tetrahedra),
            "tetgen_element_count": len(tetgen_tetrahedra),
            "exact_tetrahedron_connectivity_matches": len(vtk_cell_for_element),
            "unique_source_cells": len(np.unique(vtk_cell_for_element)),
            "tetgen_region_values": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(tetgen_regions, return_counts=True))
            },
        },
        "activation": {
            "source_record_rows": len(source_lat),
            "source_records_per_vtk_cell": 1,
            "source_vtk_cell_ids_are_one_based_contiguous": True,
            "mapped_tetgen_element_count": len(activation_by_element),
            "mapped_time_unit": "s",
            "unit_basis": "pinned LAT_openCARPToCardioMechanics.m divides the openCARP LAT by 1000",
            "minimum_s": float(activation_by_element.min()),
            "maximum_s": float(activation_by_element.max()),
            "zero_value_count": int(np.count_nonzero(activation_by_element == 0)),
            "field_sha256": hashlib.sha256(lat_path.read_bytes()).hexdigest(),
            "element_map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        },
        "outputs": {
            lat_path.name: {
                "bytes": lat_path.stat().st_size,
                "sha256": hashlib.sha256(lat_path.read_bytes()).hexdigest(),
                "order": "rows follow 1-based TetGen element order",
            },
            map_path.name: {
                "bytes": map_path.stat().st_size,
                "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "values": "1-based row IDs into source LAT text file",
            },
        },
        "qualification": {
            "case18_source_match": False,
            "native_accepted_electrical_steps": 0,
            "voltage_or_ionic_model": False,
            "electromechanical_coupling": False,
            "heartbeat_qualified": False,
            "clinical_electrophysiology": False,
            "source_license_status": "not stated in the Zenodo record snapshot reviewed",
        },
        "boundary": (
            "This is an exact, source-bound transfer of one published Healthy Rodero-04 "
            "LAT field from VTK tetrahedra to the paired TetGen element order. It is a "
            "different heart source from the current case18 candidate. It does not run "
            "electrophysiology or establish voltage, ionic dynamics, a native accepted "
            "electrical step, electromechanical coupling, a qualified heartbeat, or "
            "clinical electrophysiology."
        ),
    }
    report_body = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(report_body).hexdigest()
    receipt_path = output_dir / "source-lat-import.json"
    receipt_data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    _write_immutable(receipt_path, receipt_data)
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geometry-zip",
        type=Path,
        required=True,
        help="pinned Zenodo data.zip containing the source VTK mesh",
    )
    parser.add_argument(
        "--cardiomechanics-zip",
        type=Path,
        required=True,
        help="pinned Zenodo CardioMechanics.zip containing TetGen and LAT files",
    )
    parser.add_argument(
        "--matlab-tools-zip",
        type=Path,
        required=True,
        help="pinned Zenodo MatlabTools.zip proving LAT conversion and units",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=command)


def command(arguments: argparse.Namespace) -> int:
    report = import_source(
        arguments.geometry_zip,
        arguments.cardiomechanics_zip,
        arguments.matlab_tools_zip,
        arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "tetgen_elements": report["mesh"]["tetgen_element_count"],
                "exact_connectivity_matches": report["mesh"][
                    "exact_tetrahedron_connectivity_matches"
                ],
                "activation_range_s": [
                    report["activation"]["minimum_s"],
                    report["activation"]["maximum_s"],
                ],
                "case18_source_match": False,
                "heartbeat_qualified": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except HumanImportError as error:
        parser.exit(2, f"rodero-case04-source-lat: {error}\n")
    except OSError as error:
        parser.exit(3, f"rodero-case04-source-lat: I/O failure: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
