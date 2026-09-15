"""Bind the explicit absence of an adipose source layer.

The current one-male package contains a visual full-skin shell and sparse
muscle/tendon surfaces, but neither source is adipose geometry.  This receipt
keeps that fact machine-checkable and prevents a skin shell or an unresolved
organ surface from being promoted to fat volume or mechanical mass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config/fat-source-absence-candidate.v1.json"
SOFT_TISSUE = ROOT / "Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json"
SKIN_SHELL = ROOT / "Docs/media/skin-shell-candidate-native-v2-20260914/receipt-v2.json"
TISSUE_MASS = ROOT / "Docs/media/tissue-mass-candidate-20260914/receipt-v2.json"
COMPOSITION = ROOT / "Docs/media/body-composition-integration-20260914/receipt-v14.json"
SCHEMA = "HumanPack.fat-source-absence-candidate.v1"


class FatSourceAbsenceError(HumanImportError):
    """The source absence evidence is inconsistent or was promoted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FatSourceAbsenceError("fat source absence: " + message)


def _read(path: Path, label: str, *, canonical_required: bool = False) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise FatSourceAbsenceError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    if canonical_required:
        _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _input(path: Path, value: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(value.get("schema")),
            "file_sha256": digest}


def _profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, digest = _read(path, "fat source profile", canonical_required=True)
    required = {
        "schema", "id", "soft_tissue_surface_receipt", "skin_shell_receipt",
        "tissue_mass_receipt", "composition_receipt", "expected_muscle_surface_count",
        "expected_fat_surface_count", "expected_skin_surface_count", "boundary",
    }
    _require(set(profile) == required, "fat source profile fields differ")
    _require(profile["schema"] == "numi.human.fat-source-absence-candidate.v1",
             "unsupported fat source profile schema")
    _require(profile["id"] == "one_adult_male_no_adipose_source_layer",
             "unsupported fat source profile")
    for key in ("soft_tissue_surface_receipt", "skin_shell_receipt",
                "tissue_mass_receipt", "composition_receipt"):
        value = profile[key]
        _require(isinstance(value, str) and value.strip() and not Path(value).is_absolute()
                 and ".." not in Path(value).parts and "\\" not in value,
                 f"{key} path is unsafe")
    _require(profile["expected_muscle_surface_count"] == 148,
             "muscle surface source count differs")
    _require(profile["expected_fat_surface_count"] == 0,
             "fat surface count is not the no-source contract")
    _require(profile["expected_skin_surface_count"] == 0,
             "skin surface count is not the sparse-surface contract")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "fat source boundary is missing")
    return profile, digest


def compile_candidate(*, profile: Path = PROFILE,
                      soft_tissue: Path = SOFT_TISSUE,
                      skin_shell: Path = SKIN_SHELL,
                      tissue_mass: Path = TISSUE_MASS,
                      composition: Path = COMPOSITION) -> dict[str, Any]:
    profile_doc, profile_sha = _profile(Path(profile))
    paths = {
        "soft_tissue_surface": Path(soft_tissue),
        "skin_shell": Path(skin_shell),
        "tissue_mass": Path(tissue_mass),
        "composition": Path(composition),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    labels = {
        "soft_tissue_surface": "soft-tissue surface receipt",
        "skin_shell": "skin-shell receipt",
        "tissue_mass": "tissue-mass receipt",
        "composition": "composition receipt",
    }
    for key, path in paths.items():
        documents[key], hashes[key] = _read(path, labels[key])

    soft = documents["soft_tissue_surface"]
    _require(soft.get("schema") == "HumanPack.soft-tissue-surface-candidate.v1"
             and soft.get("status") == "partial", "soft-tissue surface receipt changed")
    soft_counts = soft.get("counts", {})
    soft_qualification = soft.get("qualification", {})
    _require(soft_counts.get("muscle_surface_count") == profile_doc["expected_muscle_surface_count"]
             and soft_counts.get("fat_surface_count") == profile_doc["expected_fat_surface_count"]
             and soft_counts.get("skin_surface_count") == profile_doc["expected_skin_surface_count"],
             "soft-tissue surface counts changed")
    _require(soft_qualification.get("fat_geometry_present") is False
             and soft_qualification.get("skin_geometry_present") is False,
             "sparse surface receipt promotes fat or skin geometry")
    _require(soft_counts.get("physical_volume_owner_count") == 0
             and soft_counts.get("mechanical_mass_owner_count") == 0,
             "soft-tissue surface receipt contains an owner")

    skin = documents["skin_shell"]
    _require(skin.get("schema") == "HumanPack.skin-shell-candidate.v2"
             and skin.get("status") == "partial", "skin-shell receipt changed")
    skin_qualification = skin.get("qualification", {})
    skin_ownership = skin.get("ownership", {})
    _require(skin_qualification.get("source_skin_member_bound") is True
             and skin_qualification.get("fat_geometry") is False
             and skin_qualification.get("fat_mass") is False,
             "skin-shell receipt changes the fat boundary")
    _require(skin_ownership.get("fat_geometry_owner") is False
             and skin_ownership.get("fat_mass_owner") is False,
             "skin-shell receipt contains a fat owner")

    tissue = documents["tissue_mass"]
    _require(tissue.get("schema") == "HumanPack.tissue-mass-composition-candidate.v1"
             and tissue.get("status") == "partial", "tissue-mass receipt changed")
    fat_class = tissue.get("classes", {}).get("fat")
    _require(isinstance(fat_class, dict)
             and fat_class.get("density_kg_per_m3") is None
             and fat_class.get("physical_volume_owner") is None
             and fat_class.get("mechanical_mass_owner") is None
             and fat_class.get("density_provenance") == "unresolved_no_source_fat_partition",
             "tissue-mass receipt supplies an untracked fat assignment")
    _require(tissue.get("qualification", {}).get("fat_geometry_and_mass") is False,
             "tissue-mass receipt promotes fat geometry or mass")

    composition = documents["composition"]
    _require(composition.get("schema") == "HumanPack.body-composition-integration-candidate.v1"
             and composition.get("status") == "partial", "composition receipt changed")
    composition_qualification = composition.get("qualification", {})
    ownership = composition.get("ownership", {})
    _require(composition_qualification.get("fat_source_absence_bound") is True
             and composition_qualification.get("fat_geometry_and_mass") is False,
             "composition receipt changes the fat source boundary")
    _require(ownership.get("fat_volume_and_mass_owner_count") == 0,
             "composition receipt contains a fat physical owner")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.fat-source-absence-candidate.1",
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            "profile": _relative(Path(profile)),
            "profile_sha256": profile_sha,
            "soft_tissue_surface_receipt_sha256": hashes["soft_tissue_surface"],
            "skin_shell_receipt_sha256": hashes["skin_shell"],
            "tissue_mass_receipt_sha256": hashes["tissue_mass"],
            "composition_receipt_sha256": hashes["composition"],
        },
        "inputs": {
            key: _input(paths[key], documents[key], hashes[key]) for key in paths
        } | {"profile": {"path": _relative(Path(profile)),
                         "schema": profile_doc["schema"], "file_sha256": profile_sha}},
        "counts": {
            "fat_surface_count": 0,
            "fat_volume_candidate_count": 0,
            "fat_mass_candidate_count": 0,
            "fat_physical_volume_owner_count": 0,
            "fat_mechanical_mass_owner_count": 0,
            "skin_visual_shell_count": 1,
            "sparse_muscle_surface_count": profile_doc["expected_muscle_surface_count"],
        },
        "qualification": {
            "fat_source_absence_bound": True,
            "fat_geometry_present": False,
            "fat_volume_candidate": False,
            "fat_mass_candidate": False,
            "fat_material_calibration": False,
            "fat_physical_volume_owner": False,
            "fat_mechanical_mass_owner": False,
            "subject_calibration": False,
            "integrated_human_qualification": False,
        },
        "boundary": profile_doc["boundary"],
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(profile=arguments.profile,
                               soft_tissue=arguments.soft_tissue,
                               skin_shell=arguments.skin_shell,
                               tissue_mass=arguments.tissue_mass,
                               composition=arguments.composition)
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": digest, "output": str(output), **result["counts"]},
                     sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--soft-tissue", type=Path, default=SOFT_TISSUE)
    parser.add_argument("--skin-shell", type=Path, default=SKIN_SHELL)
    parser.add_argument("--tissue-mass", type=Path, default=TISSUE_MASS)
    parser.add_argument("--composition", type=Path, default=COMPOSITION)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (FatSourceAbsenceError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"fat-source-absence-candidate: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
