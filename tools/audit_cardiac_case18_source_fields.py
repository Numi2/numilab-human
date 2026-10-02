"""Audit the exact field inventory in the pinned Rodero case18 source archive.

The audit verifies archive/member identity before inspecting VTK declarations.
It distinguishes source-provided anatomy from absent electrical activation and
fast-endocardial cell tags; it does not reconstruct either missing source field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/cardiac-wall-rodero18.v1.json"
DECLARATION_TAGS = {
    b"SCALARS", b"VECTORS", b"TENSORS", b"NORMALS", b"FIELD",
    b"TEXTURE_COORDINATES", b"COLOR_SCALARS",
}
STRUCTURE_TAGS = {
    b"DATASET", b"POINTS", b"CELLS", b"CELL_TYPES", b"CELL_DATA",
    b"POINT_DATA",
}
MARKER = re.compile(
    rb"^\s*(DATASET|POINTS|CELLS|CELL_TYPES|CELL_DATA|POINT_DATA|"
    rb"SCALARS|VECTORS|TENSORS|NORMALS|FIELD|TEXTURE_COORDINATES|"
    rb"COLOR_SCALARS)\b"
)
EXPECTED_STRUCTURE = [
    ["DATASET", "UNSTRUCTURED_GRID"],
    ["POINTS", "300965", "float"],
    ["CELL_TYPES", "1470083"],
    ["CELLS", "1470083", "7350415"],
    ["CELL_DATA", "1470083"],
    ["POINT_DATA", "300965"],
]
EXPECTED_DECLARATIONS = [
    ["SCALARS", "ID", "int", "1"],
    ["VECTORS", "fibres", "float"],
    ["VECTORS", "sheets", "float"],
    ["SCALARS", "RHO.dat", "float", "1"],
    ["SCALARS", "PHI.dat", "float", "1"],
    ["SCALARS", "Z.dat", "float", "1"],
    ["SCALARS", "V.dat", "float", "1"],
]


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("case18 source field audit: " + reason)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_vtk(stream) -> tuple[list[list[str]], list[list[str]], str, int]:
    """Read declarations and hash the uncompressed source member in one pass."""
    digest = hashlib.sha256()
    structure: list[list[str]] = []
    declarations: list[list[str]] = []
    byte_count = 0
    for line in stream:
        digest.update(line)
        byte_count += len(line)
        match = MARKER.match(line)
        if not match:
            continue
        tag = match.group(1)
        tokens = line.decode("ascii", "strict").split()
        if tag in STRUCTURE_TAGS:
            structure.append(tokens)
        elif tag in DECLARATION_TAGS:
            declarations.append(tokens)
    return structure, declarations, digest.hexdigest(), byte_count


def verify_declarations(structure: list[list[str]],
                       declarations: list[list[str]]) -> None:
    require(structure == EXPECTED_STRUCTURE,
            "source VTK geometry/data section declarations")
    require(declarations == EXPECTED_DECLARATIONS,
            "source VTK field declarations; unexpected or missing source field")


def audit_archive(archive_path: Path) -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    expected_archive = source["archive"]
    require(archive_path.is_file() and not archive_path.is_symlink(),
            "archive must be a regular, non-symlink file")
    require(archive_path.stat().st_size == expected_archive["bytes"]
            and sha256_file(archive_path) == expected_archive["sha256"],
            "pinned archive byte count or SHA-256")

    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            members = tar.getmembers()
            require(len(members) == 1 and members[0].isfile()
                    and members[0].name == source["member"]["path"],
                    "archive must contain only the pinned regular VTK member")
            member = members[0]
            require(member.size == source["member"]["bytes"],
                    "pinned VTK member byte count")
            member_stream = tar.extractfile(member)
            require(member_stream is not None, "readable pinned VTK member")
            with member_stream:
                structure, declarations, member_sha, member_bytes = (
                    inspect_vtk(member_stream))
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("case18 source field audit: invalid tar/gzip archive") from exc

    require(member_bytes == source["member"]["bytes"]
            and member_sha == source["member"]["sha256"],
            "pinned VTK member SHA-256")
    verify_declarations(structure, declarations)

    return {
        "schema": "HumanPack.cardiac-case18-source-field-audit.v1",
        "source_config_sha256": sha256_file(CONFIG_PATH),
        "source_record": source["record_url"],
        "archive": {
            "name": expected_archive["name"],
            "bytes": expected_archive["bytes"],
            "sha256": expected_archive["sha256"],
            "member_count": 1,
        },
        "member": {
            "path": source["member"]["path"],
            "bytes": member_bytes,
            "sha256": member_sha,
        },
        "vtk_structure": structure,
        "vtk_field_declarations": declarations,
        "field_presence": {
            "cell_material_labels": True,
            "fibres": True,
            "sheets": True,
            "ventricular_coordinates_rho_phi_z_v": True,
            "case18_activation_time_field": False,
            "transmembrane_voltage_or_ionic_state": False,
            "fast_endocardial_cell_tag": False,
        },
        "boundary": (
            "The pinned archive contains geometry and the listed source fields. "
            "Electrical activation and fast-endocardial membership remain "
            "reconstruction inputs; this audit performs no electrophysiology "
            "or mechanics step and qualifies no heartbeat."
        ),
    }


def write_immutable(path: Path, result: dict) -> None:
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), "receipt destination must not be a symlink")
    if path.exists():
        require(path.read_bytes() == payload, "existing receipt is immutable")
        return
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_archive(args.archive)
    if args.output:
        write_immutable(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
