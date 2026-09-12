#!/usr/bin/env python3
"""Verify a source asset's identities and run the independent offline C++ check."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numilab_human.cardiac_wall_source import (  # noqa: E402
    CONFIG, CONFIG_SHA256, SCHEMA, canonical, digest, load_config, require, write_immutable,
)


def qualify(asset: Path, checker: Path, evidence: Path) -> dict:
    evidence.mkdir(parents=True, exist_ok=True)
    configuration, config_sha = load_config(CONFIG)
    manifest_data = (asset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    require(manifest["schema"] == SCHEMA and manifest["source_config"] == configuration
            and manifest["source_config_sha256"] == config_sha == CONFIG_SHA256,
            "manifest source identity mismatch")
    counts = {"nodes": 300965, "tetrahedra": 1470083, "boundary_faces": 230364}
    expected = {
        "nodes.f64le": ([counts["nodes"], 3], 8),
        "tetrahedra.u32le": ([counts["tetrahedra"], 4], 4),
        "source_reversed_cells.u32le": ([0], 4),
        "labels.u32le": ([counts["tetrahedra"]], 4),
        "fibres.f64le": ([counts["tetrahedra"], 3], 8),
        "sheets.f64le": ([counts["tetrahedra"], 3], 8),
        "boundary.u32le": ([counts["boundary_faces"], 3], 4),
        "boundary_owners.u32le": ([counts["boundary_faces"]], 4),
        "boundary_components.u32le": ([counts["boundary_faces"]], 4),
    }
    for field in ("rho", "phi", "z", "v"):
        expected[f"uvc_{field}.f64le"] = ([counts["nodes"]], 8)
    require(set(expected) == set(manifest["buffers"]), "unexpected/missing asset buffer")
    verified = {}
    for name, (shape, width) in expected.items():
        record, path = manifest["buffers"][name], asset / name
        require(not path.is_symlink() and path.is_file(), f"nonregular asset buffer {name}")
        require(record["shape"] == shape and record["bytes"] == math.prod(shape) * width
                and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                f"asset buffer mismatch {name}")
        verified[name] = record["sha256"]
    qualification = manifest["qualification"]
    require(qualification["status"] == "source_imported_runtime_unqualified"
            and qualification["physical_steps"] == 0
            and all(qualification[key] is False for key in
                    ("native_matter_package", "material_fit", "body_registration", "mechanical_blood_mass", "anatomical_wall_simulation")),
            "unexpected source promotion")
    checker = checker.resolve(strict=True)
    checker_sha = digest(checker)
    command = [str(checker), str(asset.resolve(strict=True))]
    process = subprocess.run(command, capture_output=True, check=False)
    write_immutable(evidence / "checker.stdout", process.stdout)
    write_immutable(evidence / "checker.stderr", process.stderr)
    require(process.returncode == 0, f"independent checker failed: {process.returncode}")
    result = json.loads(process.stdout)
    require(result["status"] == "pass" and result["physical_steps"] == 0
            and result["native_matter_runtime_qualified"] is False, "invalid independent checker result")
    require(all(result[key] == value for key, value in counts.items()), "independent count mismatch")
    topology, boundary = manifest["topology"], manifest["boundary"]
    require(result["boundary_edge_manifold_orientation_defects"] == boundary["edge_manifold_orientation_defect_count"] == 31
            and result["boundary_components"] == len(boundary["components"]) == 6,
            "source boundary defects or components differ")
    for region in result["regions"]:
        label = str(region["label"])
        volume = topology["regional_geometric_volume_m3"][label]
        require(region["cells"] == topology["regional_cell_counts"][label]
                and abs(region["volume_m3"] - volume) <= 1e-12 * volume,
                f"independent exact/FP64 volume mismatch in region {label}")
    require(digest(checker) == checker_sha, "checker changed during run")
    require((asset / "manifest.json").read_bytes() == manifest_data, "manifest changed during run")
    for name, sha in verified.items():
        require(digest(asset / name) == sha, f"asset changed during check {name}")
    report = {
        "schema": "HumanPack.cardiac-wall-source-qualification.v1",
        "source_import_and_independent_geometry_check": "pass",
        "native_cardiac_mechanics_admission": "not_qualified",
        "boundary_manifold_gate": "failed", "boundary_edge_defects": 31,
        "physical_steps": 0, "actual_execution": "offline Python authoring and native C++ CPU geometry checker",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "source_config_sha256": config_sha,
        "asset_manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "buffers": verified, "checker": {"command": command, "sha256": checker_sha, "exit_code": process.returncode},
        "independent_result": result,
        "owner_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / "src/numilab_human/cardiac_wall_source.py", ROOT / "src/numilab_human/model.py",
            ROOT / "tools/cardiac_wall_asset_check.cpp", Path(__file__).resolve(), CONFIG)},
    }
    write_immutable(evidence / "qualification.json", canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--checker", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = qualify(args.asset, args.checker, args.evidence)
        print(json.dumps({key: report[key] for key in ("source_import_and_independent_geometry_check", "native_cardiac_mechanics_admission", "boundary_manifold_gate")}))
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
