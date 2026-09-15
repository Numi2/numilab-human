# Conservative disconnected organ source moments — 2026-09-15

The immutable [component moment receipt](media/organ-geometry-component-moments-20260915/receipt-v1.json)
recomputes the pinned BodyParts3D 4.0 inventory and extends the earlier
single-surface integral with a narrowly bounded component sum.  It covers all
18 authored regions and 378 unique members.  The result has 357 single closed
members plus seven members whose disconnected components are each closed and
whose source-frame axis-aligned bounds are pairwise disjoint:

`FJ1893`, `FJ3090`, `FJ3093`, `FJ3110`, `FJ3113`, `FJ3115`, and `FJ3116`.

The seven remaining multi-component members have overlapping component bounds
and remain explicitly unresolved: `FJ1913`, `FJ2416`, `FJ2418`, `FJ2436`,
`FJ3072`, `FJ3074`, and `FJ3075`.  The seven source-topology-defective
members remain unresolved as well.  The receipt therefore contains 364
source-frame algebraic moment rows and retains the exact source hash and
component topology for every member.

For an admitted disconnected member, each component is compacted without
moving a vertex, integrated with the existing oriented surface tetrahedral
method, and combined by zeroth, first, and second raw moments.  The component
sum is an audit-friendly source geometry quantity.  It is not a watertight
union, a body-frame registration, a lumen or capillary volume, a density, a
material, a blood owner, or a mechanical mass owner.  No self-intersection or
inter-domain overlap claim is made, and no physical stepping is performed.

The receipt was generated with:

```sh
PYTHONPATH=src python -m numilab_human.organ_geometry_component_moments \
  --output Docs/media/organ-geometry-component-moments-20260915/receipt-v1.json
```

SHA-256: `e86bd94e3924a0cc1cb95b4936d22793467d8cbaeac898e1dea721a86ef5c236`.
