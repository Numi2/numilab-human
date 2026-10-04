# Full Source Skin: Registered-Bone-Seeded Motion Screen (2026-10-04)

## Result

The preregistered source-bone-seeded extension failed its three-pose full-solid
embeddedness screen and was not adopted. It improves the exact intersection
count at 0.4 rad, but increases crossings beyond the retained ABI 5 outer-sheet
baseline at 0.8 and 1.2 rad. This rejects this fixed candidate and prediction;
it does not establish that no other weight field can work.

| Bilateral knee angle | Retained ABI 5 outer baseline | Full source candidate | Hidden/connector to outer | Outer to outer | Hidden to hidden |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 rad | 0 | 1,145 | 1,145 | 0 | 0 |
| 0.8 rad | 0 | 2,770 | 2,770 | 0 | 0 |
| 1.2 rad | 75 | 2,985 | 2,910 | 75 | 0 |

For comparison, the earlier full-solid harmonic-motion control retained
2,760, 2,403, and 2,098 pairs at those poses respectively. The source-seeded
candidate reduces q0.4 intersections by 1,615, adds 367 at q0.8, and adds 887
at q1.2. At q1.2 its 75 outer-to-outer pairs exactly match the retained outer
baseline; the additional 2,910 are hidden/connector-to-outer intersections.
Neither candidate has hidden-to-hidden pairs in these three snapshots.

## Source and geometry checks

The input was the pinned 101,691-vertex, 203,382-triangle closed FJ2810 source
solid. The screened positive graph solve seeded hidden/connector vertices from
registered bone-to-skin projections while fixing ABI 5 outer weights. All 86
bindings received seeds on unknown vertices. The source-rest reconstruction
maximum error was 0.000153 mm; the maximum outer-to-native-pack discrepancy
over the three poses was 0.001434 mm. Every candidate snapshot retained closed
oriented manifold topology. These checks do not override the failed exact
intersection result.

All source identities, the reconstructed candidate weight field, rest gate,
per-pose result files, compressed exact pair lists, pair classes, and native
outer-pack parity were revalidated from retained artifacts. The summary says
`exact_predicates_reexecuted: false`; recovery did not rerun the costly exact
surface predicates. Source hashes and full per-pose evidence are in
[`summary.json`](media/skin-full-solid-motion-20261004/source-seeded-results/summary.json).

An initial projection attempt mixed registered-world bone coordinates with
local source vertices. Its invalid counts are preserved in
[`attempt-001.json`](media/skin-full-solid-motion-20261004/source-seeded-attempts/attempt-001.json);
no candidate weights or pose predicates were run from that attempt. The frame
was corrected before the three preregistered pose predicates, and the original
prediction and conditions were retained.

The candidate is not adopted. This is a deterministic three-snapshot visual
geometry screen, not continuous-motion, native-renderer, contact, skin-material,
physical-mechanics, clinical-anatomy, or biological qualification.

## Reproduction

Run the complete source-seeded extension and three exact CPU pose predicates
with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.skin_full_solid_source_seeded_motion \
  --output Docs/media/skin-full-solid-motion-20261004/source-seeded-results
```

Validate the retained field and rows and rebuild the summary without rerunning
the exact predicates with:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.skin_full_solid_source_seeded_motion \
  --output Docs/media/skin-full-solid-motion-20261004/source-seeded-results \
  --summarize-existing
```
