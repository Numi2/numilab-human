# Exact patellofemoral contact-surface intersections - 30 September 2026

The earlier [surface-orientation gate](PATELLOFEMORAL_SURFACE_ORIENTATION_20260930.md)
proves exterior, outward winding and the facing of a nearby cartilage patch.
It does not prove that the two cartilage surfaces are free of intersections.
An [exact source and bilateral compiled audit](media/patellofemoral-intersections-20260930/receipt.json)
now checks the full authored `PTC_@_FMC_ContactFaces` and
`FMC_@_PTC_ContactFaces` triangle sets. A Float64 AABB pass narrows candidate
pairs; rational predicates classify intersections of the actual source and
compiled binary coordinates.

| Representation | AABB candidates | Exact crossing triangle pairs | Longest intersection segment |
| --- | ---: | ---: | ---: |
| Pinned `oks003` source | 2,043 | 18 | 0.423 mm |
| Compiled left knee | 1,800 | 18 | 0.416 mm |
| Compiled mirrored right knee | 1,800 | 18 | 0.416 mm |

All 18 are segments or polygons; there are no isolated point-contact pairs.
The **same 18 source triangle-index pairs** cross in both compiled placements.
In the left payload, all intersection points lie within an approximately
0.823 × 0.233 × 0.571 mm world-axis bounding box. Thus the compiler did not
introduce this local condition, and sagittal mirroring did not remove it.
Neither the triangle-pair count nor the bounding box measures penetration
depth, tissue overlap volume, clinical severity, or contact force.

**Static noninterpenetration fails.** This is a source-authored initial
cartilage-contact condition, not evidence that the kneecap sits behind the
knee. A contact solver might resolve initial overlap, but no loaded contact
transaction, pressure field, force transfer, or sustained flexion is proved by
this audit. The source coordinates, registration, faces and payloads were not
moved or clipped to produce a passing geometry result. The next admission gate
needs native contact execution with explicit initial-penetration handling,
load/energy accounting and deterministic accepted-state evidence. Clinical
placement remains unqualified.

Reproduce with the retained pinned source and bilateral payloads. Exit 2 is
the expected failed static-noninterpenetration result; the receipt is written
before exit:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m tools.verify_patellofemoral_intersections \
  --output Docs/media/patellofemoral-intersections-20260930/receipt.json
```
