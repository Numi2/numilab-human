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

The 573 BodyParts3D member IDs partition into 571 members of the 46 declared
source families and two baseline members outside those families (`FJ1737` and
`FJ2428`). Six Z-Anatomy surfaces use pinned source object IDs at stable IDs
305–310 instead of BodyParts3D member IDs. Together these groups account for
all 579 surfaces. The six Z-Anatomy objects match the source configuration and
its hash-locked export. The five lung lobes have FMA concept IDs that match the
pinned parent-lung edges. The Pleura object has no FMA concept ID in the source
configuration and remains taxonomy-unmapped.

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
`b3282a2341791c9edca182a0e324bacd16b980a52d409e9e1b22b60f759bca6d`); its
uncompressed JSON SHA-256 is
`aa51a4a536e1d0d0bdb7f738d6f5d586ad615d33996722e8982890f57a65a226`.

This is a source-identity and taxonomy crosswalk. It does not establish
clinical anatomy, resolve tissue boundaries or component ownership, construct
physical volumes or lumens, or qualify mechanics, calibration, or physiology.
