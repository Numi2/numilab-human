#!/usr/bin/env python3
"""Exactly test embeddedness and containment of a retained organ candidate bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numilab_human.organ_surface_embeddedness_audit import audit_payload  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="topology-candidate bundle directory")
    parser.add_argument("--output", type=Path, help="new immutable audit JSON; defaults inside bundle")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    receipt_path = bundle / "receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    candidate_path = bundle / receipt["candidate"]["file"]
    candidate_raw = candidate_path.read_bytes()
    if sha(candidate_raw) != receipt["candidate"]["sha256_gzip"]:
        raise SystemExit("candidate gzip SHA-256 differs from topology receipt")
    candidate_bytes = gzip.decompress(candidate_raw)
    if sha(candidate_bytes) != receipt["candidate"]["sha256_uncompressed"]:
        raise SystemExit("candidate payload SHA-256 differs from topology receipt")
    payload = json.loads(candidate_bytes)
    result = audit_payload(payload)

    output = (args.output or (bundle / "embeddedness-audit.json")).resolve()
    if output.exists():
        raise SystemExit(f"embeddedness audit is immutable; choose a new output path: {output}")
    intersection_source = ROOT / "src/numilab_human/cardiac_cavity_intersections.py"
    audit_source = ROOT / "src/numilab_human/organ_surface_embeddedness_audit.py"
    document = {
        **result,
        "inputs": {
            "topology_receipt_sha256": sha(receipt_raw),
            "candidate_gzip_sha256": sha(candidate_raw),
            "candidate_payload_sha256": sha(candidate_bytes),
            "exact_intersection_kernel_sha256": sha(intersection_source.read_bytes()),
            "audit_implementation_sha256": sha(audit_source.read_bytes()),
        },
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "members": result["candidate_count"],
        "embedded_disjoint_candidates": sum(
            row["status"] == "embedded_disjoint_component_union_candidate" for row in result["results"]
        ),
        "self_intersecting_candidates": sum(
            row["status"] == "self_intersecting_source_candidate" for row in result["results"]
        ),
        "output": str(output),
        "sha256": sha(output.read_bytes()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
