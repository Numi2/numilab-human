"""Classify retained exact full-skin intersection pairs by source sheet."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from numilab_human import model as human
from numilab_human.skin_full_solid_motion import (
    HELDOUT_DIR,
    PLAN_PATH,
    SOLID_DIR,
    _load_skin,
    _load_solid,
    _outer_to_full,
)
from numilab_human.whole_body_embeddedness import atomic_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError("skin crossing class audit: " + message)


def classify_pairs(pairs: list[list[int]], faces, outer_face_keys: set[tuple[int, ...]]) -> Counter:
    classes: Counter = Counter()
    for pair in pairs:
        require(len(pair) == 2 and pair[0] != pair[1], "triangle-pair layout")
        face_classes = []
        for face_id in pair:
            require(0 <= face_id < len(faces), "triangle id range")
            key = tuple(sorted(map(int, faces[face_id])))
            face_classes.append("outer" if key in outer_face_keys else "inner_or_connector")
        classes["_to_".join(sorted(face_classes))] += 1
    return classes


def audit(results_dir: Path, output: Path) -> dict[str, Any]:
    results_dir = Path(results_dir).resolve()
    plan = human.read_json(PLAN_PATH)
    summary_path = results_dir / "summary.json"
    summary = human.read_json(summary_path)
    solid_path = SOLID_DIR / "bodyparts3d-full-source-skin-solid.nhsolid"
    solid_manifest_path = SOLID_DIR / "manifest.json"
    solid_audit_path = SOLID_DIR / "audit.json"
    payload_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.nhskin"
    manifest_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json"
    require(summary.get("plan_sha256") == human.sha256(PLAN_PATH)
            and summary.get("status") == "failed_sampled_full_solid_embeddedness"
            and summary.get("case_count") == len(plan["conditions"])
            and {name.removesuffix(".json") for name in summary["rows_sha256"]}
            == {condition["id"] for condition in plan["conditions"]}
            and plan["model"]["full_solid_sha256"] == human.sha256(solid_path)
            and summary.get("full_solid_sha256") == human.sha256(solid_path)
            and summary.get("full_solid_manifest_sha256") == human.sha256(solid_manifest_path)
            and summary.get("base_skin_payload_sha256") == human.sha256(payload_path)
            and summary.get("base_skin_manifest_sha256") == human.sha256(manifest_path),
            "completed screen and pinned source identities")
    _, _, _, _, _, _, outer, outer_faces, _, outer_weights = _load_skin(
        payload_path, manifest_path,
    )
    _, _, full_vertices, full_faces = _load_solid(
        solid_path, solid_manifest_path, solid_audit_path,
    )
    outer_to_full, _, _ = _outer_to_full(outer, full_vertices, outer_weights)
    outer_face_keys = {tuple(sorted(map(int, outer_to_full[face]))) for face in outer_faces}
    require(len(outer_face_keys) == len(outer_faces)
            and len(outer_face_keys) == 109183,
            "exact outer triangle source subset")
    rows = []
    for case, record_sha in summary["rows_sha256"].items():
        row_path = results_dir / case
        row = human.read_json(row_path)
        pairs_path = results_dir / row["full_solid_intersection_pairs_file"]
        require(human.sha256(row_path) == record_sha
                and human.sha256(pairs_path) == row["full_solid_intersection_pairs_sha256"],
                f"{case} exact-pair witness identity")
        with gzip.open(pairs_path, "rt", encoding="utf-8") as stream:
            pairs = json.load(stream)
        require(len(pairs) == row["full_solid_exact_intersection_pair_count"],
                f"{case} retained witness count")
        classes = classify_pairs(pairs, full_faces, outer_face_keys)
        counts = dict(sorted(classes.items()))
        rows.append({
            "case": row["case"],
            "exact_pair_count": len(pairs),
            "outer_triangle_count": len(outer_face_keys),
            "classification_counts": counts,
            "pair_file_sha256": human.sha256(pairs_path),
            "row_sha256": human.sha256(row_path),
        })
        require(sum(counts.values()) == len(pairs), f"{case} classification closure")
    result = {
        "schema": "numi.human.skin-full-solid-crossing-classes.v1",
        "status": "retained_exact_pair_witnesses_classified_by_source_sheet",
        "plan_sha256": human.sha256(PLAN_PATH),
        "screen_summary_sha256": human.sha256(summary_path),
        "source_solid_sha256": human.sha256(solid_path),
        "source_solid_manifest_sha256": human.sha256(solid_manifest_path),
        "base_skin_payload_sha256": human.sha256(payload_path),
        "base_skin_manifest_sha256": human.sha256(manifest_path),
        "classifier_source_sha256": human.sha256(Path(__file__)),
        "cases": rows,
        "new_predicate_run": False,
        "clinical_anatomy": False,
        "physical_skin_mechanics": False,
        "boundary": ("Post hoc classification of the retained exact triangle-pair lists. "
                     "A vertex set distinguishes outer visual triangles from all "
                     "inner-sheet and connector triangles; it does not diagnose "
                     "material, contact, or clinical anatomy."),
    }
    atomic_json(Path(output).resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.results, args.output)
    print(json.dumps({
        "status": result["status"],
        "cases": [{"case": row["case"], "pairs": row["exact_pair_count"],
                   "classes": row["classification_counts"]}
                  for row in result["cases"]],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
