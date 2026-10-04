from __future__ import annotations

import json
from pathlib import Path

from numilab_human.organ_geometry import TEMPLATE
from numilab_human.organ_surface_topology_candidates import SCHEMA, build_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_all_pinned_open_organ_members_have_exact_support_closed_candidates() -> None:
    payload, encoded = build_candidates(
        sources=ROOT / "Sources",
        source_lock=ROOT / "sources.lock.json",
        template=TEMPLATE,
    )
    assert payload["schema"] == SCHEMA
    assert len(encoded) > 0
    assert payload["raw_source_archives_modified"] is False
    assert payload["new_coordinates_added"] is False
    assert payload["source_support_preserved"] is True
    assert payload["exact_signed_integral_preserved"] is True

    candidates = {row["member_id"]: row for row in payload["source_members"]}
    assert set(candidates) == {"FJ2404", "FJ2405", "FJ2409", "FJ2434", "FJ2820", "FJ2821", "FJ2928"}
    assert all(row["candidate_topology"]["closed_oriented_manifold_candidate"] for row in candidates.values())
    assert all(row["source_support_preserved"] for row in candidates.values())
    assert all(row["exact_signed_integral_preserved"] for row in candidates.values())
    assert all(not row["vertex_coordinates_modified"] for row in candidates.values())
    assert all(row["self_intersections"] == "not_checked" for row in candidates.values())
    assert all(row["physical_volume"] is False and row["mechanics"] is False for row in candidates.values())
    assert all(row["component_geometry"]["all_components_closed_oriented_manifold_candidates"]
               for row in candidates.values())
    assert all(row["component_geometry"]["component_aabb_pairwise_disjoint"]
               for member_id, row in candidates.items() if member_id not in {"FJ2405", "FJ2820"})
    assert not candidates["FJ2405"]["component_geometry"]["component_aabb_pairwise_disjoint"]
    assert not candidates["FJ2820"]["component_geometry"]["component_aabb_pairwise_disjoint"]

    liver = candidates["FJ2820"]
    assert len(liver["passes"]) == 2
    assert liver["passes"][0]["after"]["closed_oriented_manifold_candidate"] is False
    assert liver["passes"][1]["after"]["closed_oriented_manifold_candidate"] is True
    assert liver["passes"][1]["cancelled_opposite_face_pair_count"] == 2
    decoded = json.loads(encoded)
    assert decoded["source_members"] == payload["source_members"]
