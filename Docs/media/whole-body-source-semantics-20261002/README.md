# Whole-body source semantics crosswalk

This immutable receipt joins all 579 selected compiled atlas surfaces to their
source IDs (where supplied), BodyParts3D 4.0 `part_of` and `is_a` terms, native
rigid owner, source geometry hash, topology-selection status, and exact input
hashes. It complements the [same-owner surface-overlap diagnostic](../whole-body-organ-overlap-20261002/README.md).

The atlas layer code `organ` contains 120 surfaces. A priority grouping over
the pinned `is_a` ancestry finds 21 organ terms, 70 organ-region terms, 21
organ-component terms, and 8 other cardinal-organ-part terms. The receipt also
retains the nonexclusive ontology closure: 99 of those surfaces inherit the
cardinal-organ-part concept, so these priority counts are a reporting
convention, not exclusive ontology truth.

There are 573 unique mapped BodyParts3D member IDs. Six Z-Anatomy surfaces use
their pinned source object IDs at stable IDs 305–310 instead of BodyParts3D
member IDs; all six match the source configuration and its hash-locked export.
The five lung lobes have FMA concept IDs that match the pinned parent-lung
edges. The Pleura object has no FMA concept ID in the source configuration and
remains taxonomy-unmapped.

Eight surfaces in the `organ` layer remain individually unqualified by the
existing selected-topology gate (one self-intersection, three exact
degenerate-face cases, and four open or nonmanifold intersecting surfaces).
Their semantic annotations do not change their topology status.

To rebuild into a new immutable output path from the repository root:

```sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  whole-body-source-semantics --output /private/tmp/whole-body-source-semantics.json
```

The deterministic gzip receipt is `receipt.json.gz` (compressed SHA-256
`a5dd478ede383e35e8d20eda601742fe8f3f83cef4bb8f2d577c6e29f11ff98f`); its
uncompressed JSON SHA-256 is
`d254bf04a618da6b7732fbea5ccbb7a75c133cb750a1ba80b87e9f1a78b1c36b`.

This is a source-identity and taxonomy crosswalk. It does not establish
clinical anatomy, resolve tissue boundaries or component ownership, construct
physical volumes or lumens, or qualify mechanics, calibration, or physiology.
