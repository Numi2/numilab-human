# Whole-body same-owner organ overlap diagnostic

This is the first exact cross-surface census over the selected, individually
embedded organ atlas surfaces. It compares meshes only when they share a native
rigid owner, using exact rational predicates over the compiled Float32
coordinates in that owner-local frame.

The selected view contains 112 individual surfaces admitted as
closed-embedded source candidates and 240,008 triangles across five owners.
Eight source organ surfaces with individual topology defects are excluded:
stable IDs 16/FJ2819, 17/FJ2820,
23/FJ2428, 496/FJ2568, 497/FJ2569, 498/FJ2570, 564/FJ3132, and 566/FJ3134.

| Native owner | Surfaces | Possible pairs | AABB-separated | Exact tests | Crossings | Separate closed domains |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Abdomen | 77 | 2,926 | 2,532 | 394 | 165 | 229 |
| Torso | 21 | 210 | 138 | 72 | 24 | 48 |
| Pelvis | 9 | 36 | 21 | 15 | 9 | 6 |
| Head | 4 | 6 | 6 | 0 | 0 | 0 |
| Neck | 1 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **112** | **3,178** | **2,697** | **481** | **198** | **283** |

The 198 crossing surface pairs contain 34,804 exact triangle-pair
intersections. The largest reported intersection segment is 13.401 mm. The
receipt keeps the pair-level outcomes, bounded pair inventory, exact predicate
counts, source IDs, source hierarchy annotations, and the eight-organ control
replay. All 28 pairs in the earlier abdominal receipt matched, including the
seven previously reported crossing pairs.

The source hierarchy annotations record direct FMA concept memberships and
part-of/is-a ancestry. For example, FJ1895 and FJ2629 both map to the FMA
pancreas concept; that labels a source relationship but does not decide whether
their intersecting meshes are an intended aggregate/component representation.
Pairs without a shared specific source term also remain unresolved by hierarchy.

This is a source-geometry diagnostic. A surface crossing is not by itself a
clinical anatomy finding, and a disjoint pair is not a tissue or mechanics
qualification. The eight excluded surfaces, clinical registration, tissue
ownership, physical organ volume, and organ mechanics remain open.

The compressed receipt is deterministic gzip. Its uncompressed JSON SHA-256 is
d57bcdeee80f7f058f4904219b5b3fe7a4d48b9ac380382a539f0d628a20e07f;
the compressed file SHA-256 is
e1c1e22e9b0fa85b67da784896aa980626929bd052213080705eb466c6834b1d.

Reproduce from the repository root with:

~~~sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  whole-body-organ-overlap --output /tmp/numi-whole-body-organ-overlap.json
gzip -n -9 -c /tmp/numi-whole-body-organ-overlap.json \
  > Docs/media/whole-body-organ-overlap-20261002/receipt.json.gz
~~~

The command verifies the candidate payload, selected gate, both source
embeddedness censuses, hierarchy source hashes, and abdominal control receipt
before publishing a new immutable JSON report. The retained compressed file is
the byte-preserving gzip form of that report.
