"""Compare exact shared-surface witness lengths for the unchanged and fitted banks."""
import json

import numpy as np

from check_native import HERE, KEEP, MUSCLE, REFERENCE, SKIN, captured, install_for_float32
from classify_native_crossing_geometry import layer, stats


def selected(report, semantic, stable_id):
    matches = [row for row in report["surfaces"]
               if row["semantic"] == semantic and row["stable_id"] == stable_id]
    assert len(matches) == 1
    return frozenset(map(tuple, matches[0]["triangle_pairs"]))


def measure(skin, target, pairs):
    skin_rows, muscle_rows = map(sorted, map(set, zip(*pairs)))
    kinds, length = stats(layer(*skin, skin_rows), layer(*target, muscle_rows), pairs)
    return dict(count=len(pairs), kinds=dict(kinds), pair_segment_sum_mm=float(sum(length)),
                max_mm=max(length), median_mm=float(np.median(length)))


def main():
    check = json.loads((HERE / "native-comparison-extent.json").read_text())
    baseline_exact = json.loads((REFERENCE / "exact-skin-surface.json").read_text())
    install_for_float32()
    old_receipt, old = captured(REFERENCE, 16)
    current_receipt, current = captured(HERE / "native-verified", 16)
    assert baseline_exact["accepted_pack_sha256"] == old_receipt["pack_file_sha256"]
    assert check["candidate_pack_sha256"] == current_receipt["pack_file_sha256"]
    results = {}
    for bank in check["banks"]:
        stable_id = bank["stable_id"]
        key = MUSCLE, stable_id
        assert stable_id in KEEP
        before = selected(baseline_exact, *key)
        after = frozenset(map(tuple, bank["skin"]["after_pairs"]))
        assert len(before) == bank["skin"]["before"] and len(after) == bank["skin"]["after"]
        added = after - before
        assert added == frozenset(map(tuple, bank["skin"]["new_pairs"]))
        baseline_ring = set(old[SKIN][1][[a for a, _ in before]].ravel().tolist())
        outside = sum(baseline_ring.isdisjoint(current[SKIN][1][a]) for a, _ in added)
        row = {"exact_new_pairs_outside_previous_skin_vertex_ring": outside,
               "reference": measure(old[SKIN], old[key], sorted(before)),
               "candidate": measure(current[SKIN], current[key], sorted(after))}
        results[stable_id] = row
        print(json.dumps({"stable_id": stable_id, **row}), flush=True)

    output = {"reference_pack_sha256": old_receipt["pack_file_sha256"],
              "candidate_pack_sha256": current_receipt["pack_file_sha256"],
              "banks": results, "comment": "Pairwise segment lengths measure surface-crossing extent only. They are not penetration depths or signed volumes."}
    with (HERE / "intersection-extent.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
