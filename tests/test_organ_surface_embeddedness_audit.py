from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from numilab_human.organ_surface_embeddedness_audit import audit_payload

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "Docs/media/organ-surface-topology-candidates-20261004"


def test_exact_embeddedness_audit_separates_five_candidates_from_two_open_gates() -> None:
    receipt = json.loads((BUNDLE / "receipt.json").read_text())
    candidate_path = BUNDLE / receipt["candidate"]["file"]
    compressed = candidate_path.read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == receipt["candidate"]["sha256_gzip"]
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == receipt["candidate"]["sha256_uncompressed"]

    result = audit_payload(json.loads(raw))
    by_id = {row["member_id"]: row for row in result["results"]}
    assert len(by_id) == 7
    for member_id in {"FJ2404", "FJ2409", "FJ2434", "FJ2821", "FJ2928"}:
        assert by_id[member_id]["status"] == "embedded_disjoint_component_union_candidate"
        assert by_id[member_id]["component_self_intersection_count"] == 0
    assert by_id["FJ2405"]["status"] == "self_intersecting_source_candidate"
    assert by_id["FJ2405"]["component_self_intersection_count"] == 10
    assert by_id["FJ2820"]["status"] == "self_intersecting_source_candidate"
    assert by_id["FJ2820"]["component_self_intersection_count"] == 33
    assert len(by_id["FJ2820"]["nested_component_pairs"]) == 16
    assert all(row["physical_volume"] is False and row["mechanics"] is False
               for row in by_id.values())
