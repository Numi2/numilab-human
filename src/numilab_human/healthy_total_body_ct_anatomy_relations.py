"""Scan-local gross organ centroid-order checks for external CT candidates.

These checks validate only source-centroid ordering in each participant's own
NIfTI RAS+ frame. They do not register the cohort to Numi Human or qualify
clinical anatomy, tissue geometry, mechanics, or physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import model as human


SCHEMA = "numi.healthy-total-body-ct-anatomy-relations.v1"
INTAKE_SCHEMA = "HumanPack.external-segmentation-source-ingest.v2"
CODE_FILES = ("healthy_total_body_ct_anatomy_relations.py",
              "healthy_total_body_ct_source.py")
RELATIONS = (
    {"id": "brain_superior_to_heart", "superior": "Brain", "inferior": "Heart"},
    {"id": "heart_superior_to_liver", "superior": "Heart", "inferior": "Liver"},
    {"id": "lung_superior_to_liver", "superior": "Lung", "inferior": "Liver"},
    {"id": "kidneys_superior_to_bladder", "superior": "Kidneys", "inferior": "Bladder"},
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("healthy CT anatomy relations: " + message)


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain {length} values")
    require(all(type(item) in (int, float) and math.isfinite(item) for item in value),
            f"{label} contains a non-finite or non-numeric value")
    return [float(item) for item in value]


def _affine_point(affine: list[list[float]], ijk: list[float]) -> list[float]:
    return [sum(affine[row][column] * point
                for column, point in enumerate((*ijk, 1.0)))
            for row in range(3)]


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compile_relations(intake_path: Path) -> dict[str, Any]:
    intake_path = Path(intake_path).resolve()
    require(intake_path.is_file() and not intake_path.is_symlink(),
            "source intake is not a retained regular file")
    intake = human.read_json(intake_path)
    require(intake.get("schema") == INTAKE_SCHEMA,
            "unsupported source intake schema")
    scans = intake.get("scans")
    labels = intake.get("labels")
    require(isinstance(scans, list) and scans,
            "source intake has no scans")
    require(isinstance(labels, list) and labels,
            "source intake has no label dictionary")

    label_names: dict[str, int] = {}
    label_ids: set[int] = set()
    for label in labels:
        require(isinstance(label, dict), "malformed source label row")
        label_id, name = label.get("label_id"), label.get("name")
        require(type(label_id) is int and label_id > 0
                and isinstance(name, str) and name
                and label_id not in label_ids and name not in label_names,
                "source label IDs and names must be unique")
        label_ids.add(label_id)
        label_names[name] = label_id
    for relation in RELATIONS:
        require(relation["superior"] in label_names
                and relation["inferior"] in label_names,
                f"source dictionary omits relation labels for {relation['id']}")

    scan_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    missing = 0
    passed = 0
    failed = 0
    for scan in scans:
        require(isinstance(scan, dict), "malformed source scan row")
        scan_id = scan.get("scan_id")
        require(isinstance(scan_id, str) and scan_id and scan_id not in scan_ids,
                "source scan IDs must be unique strings")
        scan_ids.add(scan_id)
        affine_raw = scan.get("voxel_to_world_affine")
        require(isinstance(affine_raw, list) and len(affine_raw) == 4,
                f"scan {scan_id} has a malformed voxel-to-world affine")
        affine = [_finite_vector(row, 4, f"scan {scan_id} affine row")
                  for row in affine_raw]
        require(all(abs(a - b) <= 1e-10 for a, b in
                    zip(affine[3], [0.0, 0.0, 0.0, 1.0])),
                f"scan {scan_id} affine has an invalid homogeneous row")
        affine_sha = _canonical_sha256(affine)
        geometry = scan.get("label_spatial_geometry_candidates")
        require(isinstance(geometry, dict),
                f"scan {scan_id} has no label spatial geometry")
        by_name: dict[str, tuple[int, list[float], str]] = {}
        for label_id_text, candidate in geometry.items():
            require(isinstance(label_id_text, str) and label_id_text.isdigit()
                    and int(label_id_text) in label_ids,
                    f"scan {scan_id} uses an unknown spatial label ID")
            require(isinstance(candidate, dict),
                    f"scan {scan_id} has a malformed spatial label candidate")
            name = next(name for name, label_id in label_names.items()
                        if label_id == int(label_id_text))
            centroid_ijk = _finite_vector(
                candidate.get("centroid_voxel_center_ijk"), 3,
                f"scan {scan_id}/{name} centroid IJK")
            centroid_ras = _finite_vector(
                candidate.get("centroid_ras_mm"), 3,
                f"scan {scan_id}/{name} centroid RAS")
            voxel_count = candidate.get("voxel_count")
            require(type(voxel_count) is int and voxel_count > 0,
                    f"scan {scan_id}/{name} has an invalid voxel count")
            projected = _affine_point(affine, centroid_ijk)
            require(max(abs(projected[i] - centroid_ras[i]) for i in range(3))
                    <= 1e-7,
                    f"scan {scan_id}/{name} centroid does not reproduce through its affine")
            by_name[name] = (voxel_count, centroid_ras,
                             _canonical_sha256({
                                 "label_id": int(label_id_text),
                                 "voxel_count": voxel_count,
                                 "centroid_voxel_center_ijk": centroid_ijk,
                                 "centroid_ras_mm": centroid_ras,
                             }))

        for relation in RELATIONS:
            superior, inferior = relation["superior"], relation["inferior"]
            if superior not in by_name or inferior not in by_name:
                missing += 1
                rows.append({
                    "scan_id": scan_id,
                    "affine_source": scan.get("affine_source"),
                    "affine_sha256": affine_sha,
                    "relation": relation["id"],
                    "superior_label": superior,
                    "inferior_label": inferior,
                    "status": "missing_label_geometry",
                })
                continue
            sup_count, sup_xyz, sup_sha = by_name[superior]
            inf_count, inf_xyz, inf_sha = by_name[inferior]
            delta = sup_xyz[2] - inf_xyz[2]
            satisfied = delta > 0.0
            passed += int(satisfied)
            failed += int(not satisfied)
            rows.append({
                "scan_id": scan_id,
                "affine_source": scan.get("affine_source"),
                "affine_sha256": affine_sha,
                "relation": relation["id"],
                "superior_label": superior,
                "superior_label_id": label_names[superior],
                "superior_voxel_count": sup_count,
                "superior_centroid_ras_mm": sup_xyz,
                "superior_label_geometry_sha256": sup_sha,
                "inferior_label": inferior,
                "inferior_label_id": label_names[inferior],
                "inferior_voxel_count": inf_count,
                "inferior_centroid_ras_mm": inf_xyz,
                "inferior_label_geometry_sha256": inf_sha,
                "ras_superior_delta_mm": delta,
                "satisfied": satisfied,
                "status": "passed" if satisfied else "failed_ordering",
            })

    total = len(rows)
    complete = missing == 0 and failed == 0 and total == len(scans) * len(RELATIONS)
    source_intake_sha = human.sha256(intake_path)
    code_hashes = {name: human.sha256(Path(__file__).with_name(name))
                   for name in CODE_FILES}
    result = {
        "schema": SCHEMA,
        "status": "passed_external_source_centroid_ordering" if complete
                  else "partial_external_source_centroid_ordering",
        "source": {
            "intake_path": intake_path.name,
            "intake_schema": intake["schema"],
            "intake_sha256": source_intake_sha,
            "archive_sha256": intake.get("source", {}).get("archive_sha256"),
            "coordinate_convention": intake.get("source", {}).get("coordinate_convention"),
            "scan_count": len(scans),
            "source_scan_frames_kept_separate": True,
        },
        "method": {
            "relations": list(RELATIONS),
            "coordinate": "NIfTI RAS+ millimetres; Z increases superiorly",
            "predicate": "centroid(superior_label).RasZ > centroid(inferior_label).RasZ",
            "no_cross_participant_coordinate_pooling": True,
        },
        "counts": {
            "scan_count": len(scans),
            "relation_count": len(RELATIONS),
            "total_relation_rows": total,
            "passed_relation_rows": passed,
            "failed_relation_rows": failed,
            "missing_relation_rows": missing,
        },
        "qualification": {
            "external_source_centroid_ordering_consistency": complete,
            "current_numi_subject_binding": False,
            "clinical_anatomy": False,
            "organ_mechanics": False,
            "physiology": False,
        },
        "predicate_source_sha256": code_hashes,
        "rows": rows,
        "boundary": (
            "External automatic-segmentation centroid ordering only. This checks four gross "
            "superior/inferior relationships independently in each scan's own RAS+ frame. "
            "It does not establish segmentation accuracy, cross-subject registration, Numi "
            "subject binding, organ surface quality, physical volume, mechanics, physiology, "
            "or clinical anatomy."
        ),
    }
    require({name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES}
            == code_hashes, "predicate source changed during compilation")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compile_relations(args.intake)
    output = args.output.resolve()
    require(not output.exists() and not output.is_symlink(),
            "refusing to replace an existing output")
    from .whole_body_embeddedness import atomic_json
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "output": str(output)}, sort_keys=True))
    return 0 if result["qualification"]["external_source_centroid_ordering_consistency"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
