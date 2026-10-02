# Compiled muscle and tendon surface embeddedness

The earlier muscle geometry audit classified the original BodyParts3D OBJ
members. It counted 60 single-closed muscle source surfaces, but did not test
self-intersections or the geometry actually emitted to NHTISS4. The new
[baseline receipt](media/muscle-surface-embeddedness-20260929/receipt-baseline.json)
tests the exact-coordinate quotient of the compiled Float32 payload: only 34
of those 60 are single, closed, and free of exact self-intersections. The
60-member algebraic-volume and route-incidence receipts remain historical
source-geometry bookkeeping, not an admission of 60 embedded compiled volumes.

The importer now cancels only exact coincident opposite-winding face pairs
before computing normals, bindings, and indices. The pinned 150-surface build
cancelled 462 pairs on 80 surfaces. This removes 462 unique visual triangles,
with 2.7145019056868547 mm² total source-space area; it is a measured loss
of rendered support, not a claim that the visual mesh stayed identical. No
points were moved and no replacement faces or caps were added. The
[repaired-payload receipt](media/muscle-surface-embeddedness-20260929/receipt-v1.json)
verifies that every retained compiled Float32 triangle has exactly the same
three positions as the baseline, in the same order. All 434,917 compiled
vertex records, including normals and sparse motion weights, are byte-identical
to the baseline. Both payloads have the same NHTISS4 ABI, source identity,
registration fingerprint, 150 surface IDs, and 512 body bindings.

The repaired build admits 52 of 148 muscle surfaces as *single embedded
surface candidates*: the original 34 plus 18 whose source topology was
defective. No tendon surface passes that gate. Of all 150 emitted surfaces,
82 are closed and individually embedded but 30 of those have multiple
components; 62 are closed but self-intersecting, three remain open or
nonmanifold and intersecting, two remain open or nonmanifold without detected
intersections, and one has an exact degenerate triangle. Thus the existing
60-source-volume candidate cannot be promoted wholesale to an embedded
compiled-tissue claim. The 18 newly admitted surfaces have no new volume
moment, density, mass, material, or active-force owner.

These are rest-configuration, per-surface geometric checks. Cross-surface
overlap and containment, anatomical placement under motion, clinical anatomy,
skin and tendon mechanics, organ interfaces, physical volume/mass ownership,
and standing qualification remain open. In particular, this receipt does not
establish that the repaired muscles attach to the clinically correct sites.

The later [right EHL Float32 visual-face repair](EHL_FP32_VISUAL_PROJECTION_20260929.md)
removes that one exact-degenerate status without changing the 52 single
embedded muscle candidates. Its [v2 census](media/muscle-surface-embeddedness-20260929/receipt-v2.json)
exact-checks all 150 surfaces: 82 closed embedded (including 30 with multiple
components), 62 closed self-intersecting, four open/nonmanifold intersecting,
and two open/nonmanifold without detected intersections. The right EHL remains
open/intersecting after the quantization repair.

The subsequent [calcaneal tendon boundary repair](TENDON_HARMONIC_BOUNDARY_20260929.md)
removes 145 right and 163 left compiled exact intersection pairs. Its
[v3 census](media/muscle-surface-embeddedness-20260929/receipt-v3.json)
still admits 52 single embedded muscle candidates and no tendon volume:
82 closed embedded, 62 closed self-intersecting, two open/nonmanifold
intersecting and four open/nonmanifold without detected intersections. Both
tendons remain open and retain one or two local source-face normal reversals.

The later [bilateral short-head visual tip repair](BICEPS_SHORT_HEAD_VISUAL_UNTANGLE_20260929.md)
keeps the original BodyParts3D OBJ members unchanged but emits two explicitly
derived muscle visual candidates. The [v4 full census](media/muscle-surface-embeddedness-20260929/receipt-v4.json)
raises single embedded muscle visual candidates to **54 of 148**; both biceps
short-head members change from three exact self-intersections to zero. This
does not add physical muscle volumes or change MyoSim force paths.

## Pairwise candidate-domain audit

The [2026-10-02 exact pairwise receipt](media/muscle-volume-disjointness-20261002/receipt-v1.json)
checks every pair among the 54 single, closed, self-embedded compiled muscle
candidates from a freshly rebuilt NHTISS4 payload. It classifies 1,237 pairs
as strictly separated by exact-coordinate bounds, 144 as separate closed
domains after exact triangle and containment checks, and 50 as surface
intersections. Those 50 rows form 25 mirrored right/left member-pair
relationships. The audit finds no nested or indeterminate pair.

“Surface intersection” is deliberately conservative: it includes boundary
contact and does not by itself prove positive-volume penetration. Those pairs
cannot yet be certified as separate closed domains. The other 94 of 148 muscle
surfaces fail the single-embedded-candidate input gate; both tendon surfaces
and skin, bone, organs, and other tissue layers are outside this pair set.
Cross-layer placement, physical volume and mass ownership, material and force
transfer, and mechanics remain unqualified. The result therefore admits no
physical volume owner and does not qualify whole-body disjointness.

This receipt binds payload SHA-256
`7cefa97bf65aa75edddbb7ac4c0a56d5d5f41c8b0aeca4c6146e867cc45d4bd8`, manifest
SHA-256 `bb4e9c63dd5e526fc28f26141cf53208cb4fce92228091a5f36d9a07e795f64c`,
and self-embeddedness receipt SHA-256
`240fe0db336b4ec38bf5450cd708558333f3ff639a9c3bd62fc90378dd767018`. Its
[payload manifest](media/muscle-volume-disjointness-20261002/payload-manifest.json)
and [matching self-embeddedness receipt](media/muscle-volume-disjointness-20261002/self-embeddedness.json)
are retained with the pairwise receipt. The manifest binds BodyParts3D 4.0,
MyoSim, the surface map, and the rebuilt registration candidate; registration
SHA-256 is
`a241f5d368b686b31890256eadd552e7547e72646f009369f91eb0d6ac4df84d`, and the
MyoSim reference manifest SHA-256 is
`5d1d749632b521bc84bbd1d08b86044740635724c062667d6a4710e7c9735b07`. The
full import route is documented in [IMPORT.md](IMPORT.md). To repeat the pair
audit with this retained manifest and embeddedness receipt:

```sh
NUMI_HUMAN_PYTHON=.venv-mujoco312/bin/python \
  .numi/commands/human muscle-surface-volume-disjointness \
  --payload Build/muscle-geometry-current-20261002/surface-payload/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  --manifest Docs/media/muscle-volume-disjointness-20261002/payload-manifest.json \
  --embeddedness Docs/media/muscle-volume-disjointness-20261002/self-embeddedness.json \
  --output Build/muscle-geometry-current-20261002/muscle-volume-disjointness-reproduced.json
```
