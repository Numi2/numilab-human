#!/usr/bin/env python3
"""Build and hash exact-support repairs for open/defective organ source surfaces."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.organ_surface_topology_candidates import SCHEMA, build_candidates  # noqa: E402
from numilab_human.organ_geometry import TEMPLATE  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload, encoded = build_candidates(
        sources=args.sources, source_lock=args.source_lock, template=args.template,
    )
    candidate_path = output / "organ-surface-topology-candidates.json.gz"
    with candidate_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as archive:
            archive.write(encoded)

    source = ROOT / "src/numilab_human/organ_surface_topology_candidates.py"
    repair = ROOT / "src/numilab_human/surface_topology_repair.py"
    receipt = {
        "schema": "numi.human.organ-surface-topology-candidate-receipt.v1",
        "compiler": "numilab-human.organ-surface-topology-candidates.1",
        "inputs": {
            "source_lock": str(args.source_lock.resolve().relative_to(ROOT)),
            "source_lock_sha256": sha(args.source_lock.resolve().read_bytes()),
            "template": str(args.template.resolve().relative_to(ROOT)),
            "template_sha256": sha(args.template.resolve().read_bytes()),
            "builder_sha256": sha(source.read_bytes()),
            "repair_algorithm_sha256": sha(repair.read_bytes()),
            "source_inventory_sha256": payload["source_inventory_sha256"],
            "source_archive": payload["source_archive"],
        },
        "candidate": {
            "file": candidate_path.name,
            "schema": SCHEMA,
            "sha256_gzip": sha(candidate_path.read_bytes()),
            "sha256_uncompressed": sha(encoded),
            "compressed_bytes": candidate_path.stat().st_size,
            "member_count": len(payload["source_members"]),
            "members": [
                {
                    "member_id": row["member_id"],
                    "regions": row["regions"],
                    "source_member_sha256": row["source_member_sha256"],
                    "source_faces": row["source_face_count"],
                    "candidate_faces": row["candidate_face_count"],
                    "passes": row["passes"],
                    "candidate_topology": row["candidate_topology"],
                    "component_geometry": row["component_geometry"],
                    "source_support_preserved": row["source_support_preserved"],
                    "exact_signed_integral_preserved": row["exact_signed_integral_preserved"],
                    "vertex_coordinates_modified": row["vertex_coordinates_modified"],
                    "self_intersections": row["self_intersections"],
                }
                for row in payload["source_members"]
            ],
            "source_support_preserved": payload["source_support_preserved"],
            "exact_signed_integral_preserved": payload["exact_signed_integral_preserved"],
            "raw_source_archives_modified": payload["raw_source_archives_modified"],
            "new_coordinates_added": payload["new_coordinates_added"],
            "physical_volume": payload["physical_volume"],
            "mechanics": payload["mechanics"],
        },
        "evidence_boundary": payload["evidence_boundary"],
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": receipt["schema"],
        "candidate_members": receipt["candidate"]["member_count"],
        "closed_candidates": sum(
            row["candidate_topology"]["closed_oriented_manifold_candidate"]
            for row in receipt["candidate"]["members"]
        ),
        "candidate_sha256": receipt["candidate"]["sha256_gzip"],
        "receipt": str(receipt_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
