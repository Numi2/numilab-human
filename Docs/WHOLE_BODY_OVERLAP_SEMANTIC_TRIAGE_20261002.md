# Whole-body source-overlap identity and semantic triage — 2 October 2026

The corrected [exact-overlap receipt](media/whole-body-organ-overlap-20261002/receipt-v4.json.gz)
now carries the final visible stable ID for each selected mesh and a complete
inventory of all 112 individually admitted organ-surface candidates. This fixes
a receipt-reference defect: the earlier overlap receipt recorded the source
stable ID in the visible-ID field for topology-reviewed replacement meshes.
The exact-pair set, per-pair crossing outcomes, intersection measures, and
compiled geometry hashes are identical to that earlier receipt. Thirty-four
of the 481 exact-pair records now carry corrected visible IDs; all 2,697
AABB-separated pair records also carry both visible IDs. One selected surface
has no same-owner pair, so the explicit inventory is needed to prove complete
surface identity coverage.

The [semantic triage receipt](media/whole-body-overlap-semantics-20261002/receipt-v2.json)
joins the corrected inventory and pair rows to the pinned source-family and
BodyParts3D crosswalk. The identity join covers all 112 surfaces, including the
singleton, with exact selection, payload, source-lock, hierarchy, and native
body-manifest provenance matching across receipts.

The geometry results remain 198 crossing pairs and 283 separate closed-domain
pairs among 481 exact tests; 2,697 of 3,178 possible same-owner pairs were
AABB-separated. The 198 crossing pairs contain 34,804 intersecting triangle
pairs. Source semantics divide them into 156 pairs sharing a declared source
family, 28 without a shared declared family but with a specific shared FMA
concept, and 14 with only a directional FMA ancestry relation. The pairs are
mostly region-to-region: 137 of 198 have that priority-class combination.
Small-intestine family members account for 120 crossings.

The largest reported intersection segments occur between FJ2577/FJ2605
(middle/proximal ileum, 13.401 mm), FJ2567/FJ2572 (descending/transverse colon,
10.910 mm), FJ2821/FJ2822 (liver segments V/VI, 10.781 mm), and FJ1895/FJ2629
(pancreas/parenchyma, 10.484 mm). These are geometry-review priorities only.
Segment length is not penetration depth or overlap volume. Family membership
and ontology relationships do not determine whether a crossing is intentional,
duplicated, a segmentation seam, or a registration error. No surface, source
registration, volume, or mechanics owner was changed in this increment.

The highest-ranked pair has a direct [raw-source comparison receipt](media/whole-body-source-overlap-pair-20261002/receipt-v2.json).
It verifies both OBJ member hashes against the source-family manifest and the
BodyParts3D archive against `sources.lock.json`. FJ2577 and FJ2605 share source
body 4 and compiled owner 7. Exact predicates find 21 crossing triangle pairs
in the raw OBJ meshes and the same 21 in the compiled surfaces; the first 12
triangle-pair identities also match. The maximum segment is 13.300 mm raw and
13.401 mm compiled. This demonstrates that the measured pair already exists in
the pinned source geometry. It does not say whether the adjacent ileum regions
should share a seam or how that seam should be partitioned. A translation-only
repair would discard source registration without supplying that missing
anatomical boundary evidence.

The prior `receipt.json.gz` remains as the original immutable run. The corrected
overlap receipt has compressed SHA-256
`0f26dfb0dcff5a436e5c374aad8ff57259a0b382b19ecbcf70d154ae66722ee5`; the
semantic triage JSON has SHA-256
`34ce905c0935b58616c472a6fdcf42d5dc62cd7ce365342365fef40e549a8a9c`.

Reproduce the corrected overlap receipt from the repository root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cli whole-body-organ-overlap \
  --output /private/tmp/whole-body-organ-overlap.json
gzip -n -9 -c /private/tmp/whole-body-organ-overlap.json \
  > Docs/media/whole-body-organ-overlap-20261002/receipt-v4.json.gz
```

Then reproduce the semantic join:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.whole_body_overlap_semantics \
  --overlap-receipt Docs/media/whole-body-organ-overlap-20261002/receipt-v4.json.gz \
  --source-semantics-receipt Docs/media/whole-body-source-semantics-20261002/receipt.json.gz \
  --output /private/tmp/whole-body-overlap-semantics.json
```

Reproduce the source comparison with:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.whole_body_source_overlap_pair \
  --overlap-receipt Docs/media/whole-body-organ-overlap-20261002/receipt-v4.json.gz \
  --output /private/tmp/whole-body-source-overlap-pair.json
```

This source-data triage does not establish clinical anatomy, tissue boundaries,
physical organ volume, mechanics, calibration, or physiology. The eight
individually unqualified organ surfaces remain excluded.
