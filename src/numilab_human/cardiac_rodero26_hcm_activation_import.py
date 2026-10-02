"""Import one published Rodero-2026 HCM activation map by VTK point identity.

This transfers an upstream activation-time vector to a sidecar field ordered
exactly like the pinned HCM1 VTK points. It does not import a healthy case18
field, execute electrophysiology, or qualify a Numi-native heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError


MESH_ARCHIVE = {
    "record": "https://zenodo.org/records/21282274",
    "filename": "HCM1.vtk",
    "bytes": 165_694_916,
    "md5": "1539b10494086f27da5cc1a3f90bed04",
    "sha256": "44e350b1df2eeff0633e2ab9683d1781490997b472bdb40c26f1bc01a0ec3fc1",
}
EP_ARCHIVE = {
    "record": "https://zenodo.org/records/21720235",
    "filename": "HCM1_EP_light.tar.zst",
    "bytes": 378_052_908,
    "md5": "4c6c6e4e2037182f2f0bc9ff8b15bd4c",
    "sha256": "0858f24eae8754e3727abc04863cfcedeb234d998e344fa6fcb6039d55913c85",
}
UPSTREAM_CODE = {
    "repository": "https://github.com/CEMRG-publications/Rodero_2026_JMCC",
    "revision": "0e13e424ceb76967ae8341a9419cf0cf3525600b",
    "visualization_path": "simulation_toolbox/visualise_EP.py",
    "visualization_sha256": "79492f9e7db194b6761779d48aa537bd12fcbd0d1315de699415fdd11b806215",
}
POINT_COUNT = 749_238
SAMPLE_ID = 53
ACTIVATION_MEMBER = "HCM1_EP/activation_maps/53.dat"
PARAMETERS_MEMBER = "HCM1_EP/inputs/json_files/53.json"
TAGS_MEMBER = "HCM1_EP/inputs/json_files/tags_EP.json"
PARAMETER_KEYS = {
    "CV_f_v",
    "ani_ratio_v",
    "k_FEC",
    "CV_f_a",
    "ani_ratio_a",
    "k_BB",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("Rodero-26 HCM activation import: " + message)


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


def _file_identity(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    size, md5, sha256 = _hash_file(path)
    _require(
        size == expected["bytes"]
        and md5 == expected["md5"]
        and sha256 == expected["sha256"],
        f"{label} differs from the pinned Zenodo file",
    )
    return {"file": path.name, "bytes": size, "md5": md5, "sha256": sha256}


def _vtk_point_count(path: Path) -> int:
    """Read only the ASCII header of a legacy binary VTK unstructured grid."""
    try:
        with path.open("rb") as stream:
            header = [stream.readline(512) for _ in range(6)]
    except OSError as error:
        raise HumanImportError("Rodero-26 HCM activation import: cannot read VTK") from error
    _require(
        len(header) == 6
        and header[0].startswith(b"# vtk DataFile Version ")
        and header[2].strip().lower() == b"binary"
        and header[3].strip() == b"DATASET UNSTRUCTURED_GRID",
        "mesh is not the pinned legacy binary unstructured-grid format",
    )
    fields = header[5].decode("ascii", errors="strict").split()
    _require(
        len(fields) == 3 and fields[0] == "POINTS" and fields[2] == "float",
        "mesh POINTS declaration is malformed or unsupported",
    )
    try:
        count = int(fields[1])
    except ValueError as error:
        raise HumanImportError("Rodero-26 HCM activation import: invalid point count") from error
    _require(count == POINT_COUNT, "mesh point count differs from HCM1 source")
    return count


def _extract_members(archive_path: Path, destination: Path) -> None:
    """Stream the pinned zstd tar and extract only the three reviewed members."""
    members = (ACTIVATION_MEMBER, PARAMETERS_MEMBER, TAGS_MEMBER)
    try:
        with subprocess.Popen(
            ["zstd", "-d", "--stdout", str(archive_path)], stdout=subprocess.PIPE
        ) as decoder:
            assert decoder.stdout is not None
            extraction = subprocess.run(
                ["tar", "-xf", "-", "-C", str(destination), *members],
                stdin=decoder.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            decoder.stdout.close()
            decoder_status = decoder.wait()
    except OSError as error:
        raise HumanImportError(
            "Rodero-26 HCM activation import: zstd and tar are required to read the source archive"
        ) from error
    _require(
        decoder_status == 0 and extraction.returncode == 0,
        "cannot extract the pinned HCM1 EP archive members",
    )
    for member in members:
        path = destination / member
        _require(path.is_file() and not path.is_symlink(), f"missing regular source member {member}")


def _read_parameters(path: Path) -> dict[str, float]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        ep = content["EP"]
        values = {key: float(ep[key]) for key in PARAMETER_KEYS}
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise HumanImportError(
            "Rodero-26 HCM activation import: sample 53 EP parameters are malformed"
        ) from error
    import numpy as np

    _require(
        set(ep) == PARAMETER_KEYS
        and bool(np.isfinite(list(values.values())).all())
        and all(value > 0 for value in values.values()),
        "sample 53 EP parameters do not match the reviewed field schema",
    )
    return values


def _read_activation(path: Path, expected_points: int):
    import numpy as np

    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise HumanImportError(
            "Rodero-26 HCM activation import: sample 53 activation map is malformed"
        ) from error
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(
        values.size == expected_points
        and bool(np.isfinite(values).all())
        and bool((values >= -1).all())
        and bool(np.all(values[values < 0] == -1)),
        "sample 53 map must contain one finite time or the publisher's -1 inactive sentinel per mesh point",
    )
    return values


def _member_digest(path: Path) -> dict[str, Any]:
    size, _, sha256 = _hash_file(path)
    return {"bytes": size, "sha256": sha256}


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


def import_source(mesh_path: Path, ep_archive_path: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    mesh_path, ep_archive_path = Path(mesh_path).resolve(), Path(ep_archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    mesh_identity = _file_identity(mesh_path, MESH_ARCHIVE, "HCM1.vtk")
    ep_identity = _file_identity(ep_archive_path, EP_ARCHIVE, "HCM1_EP_light.tar.zst")
    point_count = _vtk_point_count(mesh_path)

    with tempfile.TemporaryDirectory(prefix="numi-hcm1-ep-") as temporary_dir:
        temporary = Path(temporary_dir)
        _extract_members(ep_archive_path, temporary)
        activation_path = temporary / ACTIVATION_MEMBER
        parameters_path = temporary / PARAMETERS_MEMBER
        tags_path = temporary / TAGS_MEMBER
        parameters = _read_parameters(parameters_path)
        try:
            tags = json.loads(tags_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise HumanImportError("Rodero-26 HCM activation import: EP tags are malformed") from error
        _require(
            tags.get("LV") == 1
            and tags.get("RV") == 2
            and tags.get("LA") == 3
            and tags.get("RA") == 4
            and tags.get("FEC_LV") == 25
            and tags.get("FEC_RV") == 28
            and tags.get("FEC_SV") == 29
            and tags.get("BB") == 26,
            "HCM1 EP tag definitions differ from the pinned source layout",
        )
        values = _read_activation(activation_path, point_count)
        activation_raw = values.astype("<f8", copy=False).tobytes()
        active_values = values[values >= 0]
        activation = {
            "sample_id": SAMPLE_ID,
            "value_count": int(values.size),
            "values_per_mesh_point": 1,
            "unit": "ms",
            "unit_basis": (
                "publisher visualization labels the activation-time progression in ms"
            ),
            "minimum_active_ms": float(active_values.min()),
            "maximum_active_ms": float(active_values.max()),
            "inactive_sentinel": -1.0,
            "inactive_sentinel_count": int(np.count_nonzero(values == -1)),
            "active_point_count": int(active_values.size),
            "parameters": parameters,
            "source_member": ACTIVATION_MEMBER,
            "source_member_sha256": _member_digest(activation_path)["sha256"],
            "parameter_member": PARAMETERS_MEMBER,
            "parameter_member_sha256": _member_digest(parameters_path)["sha256"],
            "field_sha256": hashlib.sha256(activation_raw).hexdigest(),
        }
        tags_identity = {
            "path": TAGS_MEMBER,
            **_member_digest(tags_path),
            "definitions": tags,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "hcm1-sample53-activation-time-ms.f64le"
    _write_immutable(field_path, activation_raw)
    report: dict[str, Any] = {
        "schema": "HumanPack.rodero26-hcm-source-activation-import.v1",
        "status": "source_bound_point_activation_imported",
        "source_records": {
            "mesh": MESH_ARCHIVE["record"],
            "electrophysiology": EP_ARCHIVE["record"],
        },
        "source_files": {"mesh": mesh_identity, "electrophysiology": ep_identity},
        "source_members": {
            "mesh": {
                "file": MESH_ARCHIVE["filename"],
                "sha256": mesh_identity["sha256"],
                "vtk_dataset": "legacy binary UNSTRUCTURED_GRID",
                "point_count": point_count,
            },
            "activation": activation,
            "tags": tags_identity,
        },
        "point_order": {
            "mapping": "direct source vector index to same-index HCM1.vtk point",
            "basis": (
                "pinned publisher visualization loads the .dat vector and assigns it "
                "directly to the matching VTK mesh point_data without reordering"
            ),
            "upstream_code": UPSTREAM_CODE,
        },
        "output": {
            field_path.name: {
                "bytes": len(activation_raw),
                "sha256": hashlib.sha256(activation_raw).hexdigest(),
                "dtype": "little-endian float64",
                "order": "HCM1.vtk POINTS order; one value per point",
            }
        },
        "qualification": {
            "patient_variant": "HCM1 hypertrophic cardiomyopathy",
            "case18_source_match": False,
            "healthy_reference": False,
            "native_numi_electrical_steps": 0,
            "voltage_or_ionic_state_imported": False,
            "electromechanical_coupling": False,
            "whole_body_circulation": False,
            "heartbeat_qualified": False,
            "clinical_prediction": False,
        },
        "boundary": (
            "This is a pinned source transfer of one published HCM1 reaction-eikonal "
            "activation-time map to the exact VTK point order used by its publisher. "
            "It is a patient-specific HCM variant, not the Healthy case18 reference. "
            "It does not run electrophysiology in Numi, import voltage or ionic state, "
            "couple electrical and mechanical dynamics, establish whole-body circulation, "
            "or qualify a heartbeat or clinical prediction."
        ),
    }
    body = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(body).hexdigest()
    receipt_path = output_dir / "source-activation-import.json"
    _write_immutable(
        receipt_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mesh-vtk", type=Path, required=True, help="pinned HCM1.vtk mesh")
    parser.add_argument(
        "--ep-archive", type=Path, required=True, help="pinned HCM1_EP_light.tar.zst"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=command)


def command(arguments: argparse.Namespace) -> int:
    report = import_source(arguments.mesh_vtk, arguments.ep_archive, arguments.output_dir)
    activation = report["source_members"]["activation"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "patient_variant": report["qualification"]["patient_variant"],
                "sample_id": activation["sample_id"],
                "mesh_points": activation["value_count"],
                "activation_range_ms": [
                    activation["minimum_active_ms"], activation["maximum_active_ms"]
                ],
                "case18_source_match": False,
                "heartbeat_qualified": False,
            },
            sort_keys=True,
        )
    )
    return 0
