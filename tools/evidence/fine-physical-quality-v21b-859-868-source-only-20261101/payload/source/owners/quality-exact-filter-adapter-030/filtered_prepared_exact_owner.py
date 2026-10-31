"""Task-local DP adapter. Preserve frozen exact predicates and one complete AABB index."""
from __future__ import annotations

class FilteredPreparedExactOwner:
    """Mask explicitly retired faces, without a face-ID ordering early-out.

    Indices stay immutable. Replacement records must retain real vertex IDs and
    use distinct face IDs for deterministic witnesses. This adapter never drops
    a surviving AABB pair because a replacement's face ID is smaller.
    """
    def __init__(self, owner, allowed_retirable_face_ids, excluded_face_ids=()):
        self._owner = owner
        self._allowed = frozenset(int(x) for x in allowed_retirable_face_ids)
        if any(x < 0 for x in self._allowed):
            raise ValueError("negative bone barrier IDs cannot be retired")
        self.set_excluded(excluded_face_ids)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def set_excluded(self, face_ids):
        selected = frozenset(int(x) for x in face_ids)
        if not selected <= self._allowed:
            raise ValueError("retirement mask is outside the authenticated role domain")
        self.excluded_face_ids = selected

    def _audit_pair_prepared_first(self, first_index, second, *, same_surface=False):
        if first_index is None or second is None:
            raise ValueError("missing prepared index or complete query records")
        queries = tuple(second)
        ids = [int(r[3]) for r in queries]
        if len(ids) != len(set(ids)):
            raise ValueError("replacement face IDs must be unique")
        excluded = self.excluded_face_ids
        counts = {"retired": 0, "surviving": 0}
        def pairs():
            for query, baseline in self._owner._query_surface_aabb(queries, first_index):
                if int(baseline[3]) in excluded:
                    counts["retired"] += 1
                    continue
                counts["surviving"] += 1
                yield baseline, query
        result = self._owner._audit_record_pairs(pairs(), same_surface=same_surface)
        if result["aabb_candidate_pairs"] != counts["surviving"]:
            raise AssertionError("exact owner did not consume the complete surviving domain")
        result["filtered_retired_aabb_pairs"] = counts["retired"]
        result["complete_query_records"] = len(queries)
        result["face_id_ordering_filter"] = False
        return result
