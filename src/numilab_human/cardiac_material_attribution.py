"""Attribute sourced passive classes and LV geometric mass; never cook or step.

This partial source sidecar leaves artificial inlet closures unresolved and all
native inertial densities absent. Class identifiers are not Matter indices.
Original source files, material-frame producers and historical receipts remain
unchanged. LV volume is an exact sum of source-coordinate cell determinants;
without global embedding it is not asserted to be a geometric union volume.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from . import cardiac_material_frames as frames
from . import cardiac_wall_source as wall
from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/cardiac-material-rodero18.v1.json"
CONFIG_SHA256 = "2091ac78031e5751f4062639b9408542765b5368e04d909420a2633d7242a8b1"
SCHEMA = "HumanPack.cardiac-material-attribution.v1"
BUFFER = "passive-material-classes.u32le"
UNRESOLVED = 0xffffffff


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac material attribution: " + message)


def exact_record(value: Fraction) -> dict:
    return {"numerator_hex": hex(value.numerator), "denominator_hex": hex(value.denominator)}


def load_config(path: Path = CONFIG) -> tuple[dict, bytes]:
    with frames._open_source(Path(path)) as stream:
        raw = stream.read()
    require(frames.sha256(raw) == CONFIG_SHA256, "attribution config hash mismatch")
    result = frames._read_json(raw)
    require(result["parent_source_config"]["sha256"] == wall.CONFIG_SHA256,
            "parent source configuration mismatch")
    return result, raw


def source_class_map(configuration: dict) -> dict[int, int]:
    """Build an explicit complete label lookup, retaining unresolved sentinels."""
    require(configuration.get("schema") == "HumanPack.cardiac-material-rodero18-attribution.v1",
            "unsupported configuration schema")
    policy = configuration["assignment_policy"]
    require(policy["unresolved_class_identifier"] == UNRESOLVED and
            policy["class_id_is_native_material_index"] is False, "invalid class encoding")
    result = {}; class_ids = set()
    for material in configuration["passive_classes"]:
        identifier = material["class_id"]
        require(type(identifier) is int and 0 <= identifier < UNRESOLVED and identifier not in class_ids,
                "duplicate or invalid material class")
        class_ids.add(identifier)
        require(material["inertial_density_kg_per_m3"] is None, "source inertial density remains unresolved")
        for label in material["source_labels"]:
            require(type(label) is int and 1 <= label <= 24 and label not in result,
                    "duplicate or invalid source label")
            result[label] = identifier
    for label in policy["unresolved_source_labels"]:
        require(type(label) is int and 1 <= label <= 24 and label not in result,
                "duplicate or invalid unresolved label")
        result[label] = UNRESOLVED
    require(set(result) == set(range(1, 25)), "source label coverage is incomplete")
    return result


def require_native_ready(attribution: dict) -> None:
    """Fail closed: this partial format has no native material/density owner.

    A future native bridge needs a separately versioned, complete assignment.
    Editing a qualification flag cannot turn class IDs into Matter indices.
    """
    require(attribution.get("schema") == SCHEMA, "unsupported attribution schema")
    raise HumanImportError("cardiac material attribution: native cooking denied; partial source classes "
                           "are not Matter indices, closure classes and inertial densities are unresolved")


def _owners() -> dict:
    paths = {"cardiac_material_attribution.py": Path(__file__),
             "cardiac_material_frames.py": Path(frames.__file__),
             "cardiac_wall_source.py": Path(wall.__file__)}
    return {name: frames.sha256(path.read_bytes()) for name, path in paths.items()}


def _opened_bytes(path: Path, pin: tuple) -> bytes:
    with frames._open_source(path) as stream:
        raw = stream.read()
    require(len(raw) == pin[2] and frames.sha256(raw) == pin[0], "consumed source changed: " + path.name)
    return raw


def _write_or_compare(asset: Path, target: Path, existing: bool, configuration: dict,
                      parent: dict, pins: dict) -> tuple[dict, dict]:
    """Bounded streaming conversion; exact geometry uses existing source values."""
    class_map = source_class_map(configuration)
    count = parent["mesh"]["cells"]; node_count = parent["mesh"]["points"]
    raw_nodes = _opened_bytes(asset / "nodes.f64le", pins["nodes.f64le"])
    require(len(raw_nodes) == node_count * 24, "node count mismatch")
    coordinates = array("d"); coordinates.frombytes(raw_nodes)
    require(coordinates.itemsize == 8, "unsupported Float64 host width")
    if sys.byteorder != "little":
        coordinates.byteswap()
    require(all(math.isfinite(value) for value in coordinates), "nonfinite source coordinates")
    points, denominator = wall.integer_points(coordinates)
    del raw_nodes, coordinates
    counts = Counter(); class_counts = Counter(); exact_volumes = Counter()
    hashes = {name: hashlib.sha256() for name in ("tetrahedra.u32le", "labels.u32le")}
    output_hash = hashlib.sha256()
    with ExitStack() as stack:
        sources = {name: stack.enter_context(frames._open_source(asset/name)) for name in hashes}
        output = stack.enter_context(frames._open_source(target) if existing else target.open("xb"))
        for offset in range(0, count, 4096):
            size = min(4096, count-offset)
            raw = {name: sources[name].read(size * (16 if name.startswith("tetrahedra") else 4)) for name in hashes}
            for name, data in raw.items():
                require(len(data) == size * (16 if name.startswith("tetrahedra") else 4), "truncated consumed source: " + name)
                hashes[name].update(data)
            encoded = bytearray(size*4)
            for local, (tet, (label,)) in enumerate(zip(struct.iter_unpack("<4I", raw["tetrahedra.u32le"]),
                                                       struct.iter_unpack("<I", raw["labels.u32le"]))):
                require(label in class_map, f"unknown source label at cell {offset+local}")
                require(len(set(tet)) == 4 and all(node < node_count for node in tet),
                        f"invalid source tetrahedron at cell {offset+local}")
                determinant = wall.determinant(*(points[node] for node in tet))
                require(determinant > 0, f"nonpositive source tetrahedron at cell {offset+local}")
                identifier = class_map[label]
                struct.pack_into("<I", encoded, local*4, identifier)
                counts[str(label)] += 1; class_counts[str(identifier)] += 1
                exact_volumes[str(label)] += determinant
            if existing:
                require(output.read(len(encoded)) == encoded, "existing class buffer changed")
            else:
                output.write(encoded)
            output_hash.update(encoded)
        for name, stream in sources.items():
            require(not stream.read(1) and hashes[name].hexdigest() == pins[name][0], "consumed source changed: " + name)
        if existing:
            require(not output.read(1), "existing class buffer has trailing bytes")
        else:
            output.flush(); os.fsync(output.fileno())
    divisor = 6 * denominator ** 3
    lv_volume = Fraction(exact_volumes["1"], divisor)
    require(counts["1"] > 0, "source contains no LV tissue cells")
    density = configuration["lv_geometric_mass_attribution"]["density_kg_per_m3"]
    require(type(density) is int and density == 1050, "invalid LV mass-estimation convention")
    mass = lv_volume * density
    report = {"counts_by_source_label": dict(sorted(counts.items())),
              "counts_by_class": dict(sorted(class_counts.items())),
              "unresolved_cells": class_counts[str(UNRESOLVED)],
              "regional_geometric_volume_m3": {label: float(Fraction(det, divisor)) for label, det in sorted(exact_volumes.items())},
              "lv_geometric_mass": {"source_label": 1, "cell_count": counts["1"],
                  "volume_m3": float(lv_volume), "volume_exact": exact_record(lv_volume),
                  "volume_semantics": "sum of source tetrahedron volumes; global geometric union is not established",
                  "density_kg_per_m3": density, "mass_kg": float(mass), "mass_exact": exact_record(mass),
                  "mass_is_native_inertia": False, "blood_mass_partitioned": False},
              "consumed_source_sha256": {"nodes.f64le": pins["nodes.f64le"][0],
                                         **{name: h.hexdigest() for name, h in hashes.items()}},
              "geometry_predicate": "exact integer determinants of unchanged binary64 metre coordinates"}
    buffer = {"path": BUFFER, "bytes": count*4, "shape": [count], "sha256": output_hash.hexdigest(),
              "encoding": "little-endian uint32 source passive class; 0xffffffff unresolved; not native material indices"}
    return buffer, report


def _prepare(asset: Path, output: Path, configuration: dict, config_raw: bytes,
             parent: dict, parent_sha: str, pins: dict, *, source_kind: str,
             config_path: Path | None = None) -> dict:
    """Private dependency-injected core permits explicitly synthetic controls."""
    require(frames.canonical(configuration) == frames.canonical(frames._read_json(config_raw)), "config bytes mismatch")
    source_class_map(configuration)
    require(asset.is_dir() and not asset.is_symlink(), "asset must be a real directory")
    require(not output.is_symlink() and not output.resolve().is_relative_to(asset.resolve()), "output must be separate from source")
    existing = output.exists()
    if existing:
        require(output.is_dir() and {p.name for p in output.iterdir()} == {BUFFER, "manifest.json"}, "changed or unrelated output")
    with frames._open_source(asset/"manifest.json") as stream:
        source_manifest = stream.read()
    manifest = frames._validate_asset(asset, source_manifest, configuration["asset_manifest_sha256"], parent, parent_sha, pins)
    owners = _owners(); output.parent.mkdir(parents=True, exist_ok=True)
    staging = None if existing else Path(tempfile.mkdtemp(prefix=".cardiac-material-attribution-", dir=output.parent))
    target = output if existing else staging
    try:
        buffer, report = _write_or_compare(asset, target/BUFFER, existing, configuration, parent, pins)
        # Numeric summaries cannot silently disagree with the validated source
        # importer, nor can metadata replacement rebind consumed opened inputs.
        topology = manifest["topology"]
        for key, actual in (("regional_cell_counts", report["counts_by_source_label"]),
                            ("regional_geometric_volume_m3", report["regional_geometric_volume_m3"])):
            require(topology.get(key) == actual, "source topology summary mismatch: " + key)
        for name, pin in pins.items():
            frames._digest_source(asset/name, pin)
        with frames._open_source(asset/"manifest.json") as stream:
            require(stream.read() == source_manifest, "source manifest changed during attribution")
        require(_owners() == owners, "implementation changed during attribution")
        if config_path is not None:
            with frames._open_source(config_path) as stream:
                require(stream.read() == config_raw, "configuration changed during attribution")
        record = {"schema": "HumanPack.cardiac-material-attribution-identity.v1", "source_kind": source_kind,
            "attribution_config_sha256": frames.sha256(config_raw), "parent_source_config_sha256": parent_sha,
            "source_asset_manifest_sha256": frames.sha256(source_manifest),
            "source_buffers": {name: {"sha256": pin[0], "shape": pin[1], "bytes": pin[2]} for name, pin in pins.items()},
            "buffer": buffer, "lv_geometric_mass": report["lv_geometric_mass"]}
        result = {"schema": SCHEMA, "source_identity_sha256": frames.sha256(frames.canonical(record)),
            "source_identity_record": record, "configuration": configuration, "buffer": buffer, **report,
            "implementation_sha256": owners,
            "qualification": {**configuration["qualification"], "physical_steps": 0,
                "source_fields_modified": False, "source_geometry_modified": False,
                "native_cooking_ready": False, "complete_passive_material_map": report["unresolved_cells"] == 0,
                "native_inertial_density_map": False, "lv_geometric_mass_attributed": True}}
        expected = frames.canonical(result)
        if existing:
            require(output.is_dir() and not output.is_symlink() and {p.name for p in output.iterdir()} == {BUFFER, "manifest.json"}, "output changed during verification")
            with frames._open_source(output/"manifest.json") as stream:
                require(stream.read() == expected, "existing attribution manifest changed")
            frames._digest_source(output/BUFFER, (buffer["sha256"], buffer["shape"], buffer["bytes"]))
        else:
            (staging/"manifest.json").write_bytes(expected)
            require(not output.exists() and not output.is_symlink(), "output appeared during attribution")
            os.rename(staging, output)
        return result
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def prepare_material_attribution(asset: Path, output: Path, *, config_path: Path = CONFIG) -> dict:
    configuration, raw = load_config(config_path)
    parent, parent_sha = wall.load_config(wall.CONFIG)
    return _prepare(Path(asset), Path(output), configuration, raw, parent, parent_sha, frames.SOURCE_BUFFERS,
                    source_kind="pinned_rodero_2021_ct_case18", config_path=Path(config_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    try:
        result = prepare_material_attribution(args.asset, args.output, config_path=args.config)
        print(json.dumps({"status": result["qualification"]["status"], "output": str(args.output),
            "source_identity_sha256": result["source_identity_sha256"], "unresolved_cells": result["unresolved_cells"],
            "lv_geometric_mass_kg": result["lv_geometric_mass"]["mass_kg"], "native_cooking_ready": False,
            "physical_steps": 0}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
