"""Count terminal witnesses a position-only repair cannot move under the cook pins."""
from collections import Counter
from fit_recooked_skin import SKIN, dependencies
from fit_clearance_patches import decode
from run_support_dependency_preflight import HERE, digest, read, write

_, faces, _, legacy, pins, proof = dependencies(decode(SKIN.read_bytes()))
case = HERE / "recomputed-support-native"
report = read(case / "step-8000-skin-exact.json")
assert report["accepted_pack_sha256"] == read(case/"summary.json")["terminal_pack_sha256"]
banks = []
by_kind = {}
for surface in report["surfaces"]:
    if not surface["count"]:
        continue
    counts = Counter(int(pins[faces[skin_face]].sum()) for skin_face, _ in surface["triangle_pairs"])
    assert sum(counts.values()) == surface["count"]
    by_kind.setdefault(str(surface["semantic"]), Counter()).update(counts)
    banks.append({"semantic":surface["semantic"], "source_id":surface["stable_id"],
                  "pairs":surface["count"], "pinned_skin_vertices_per_pair":dict(sorted(counts.items()))})
self_histogram = Counter(
    int(pins[faces[left]].all()) + int(pins[faces[right]].all())
    for left, right in report["self_intersections"]["triangle_pairs"]
)
write(HERE / "terminal-fixed-face-dependencies.json", {
    "terminal_pack_sha256":report["accepted_pack_sha256"],
    "counts_by_fixed_vertices_for_tissue_pairs":{key:dict(sorted(hist.items())) for key,hist in by_kind.items()},
    "counts_by_fully_fixed_skin_faces_for_self_pairs":dict(sorted(self_histogram.items())),
    "top_whole_fixed_banks":[row for row in sorted(banks,key=lambda row: -row["pinned_skin_vertices_per_pair"].get(3,0))
                             if row["pinned_skin_vertices_per_pair"].get(3,0)],
})
