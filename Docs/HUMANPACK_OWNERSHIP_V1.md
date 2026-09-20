# HumanPack ownership v1

`HumanPack.ownership.v1` is the stable authoring boundary between cumulative
Human source coverage and future physical mechanics owners. It materializes one
record per semantic target/source identity, retains every current muscle-route
action without imposing an action-count ceiling, and exposes explicit slots for:

- topology, field, and motor-compartment IDs;
- source composition and unresolved conflicts;
- physical-volume, mechanical-mass, material, active-force, and state owners;
- volume plus zeroth, first, and second mass moments; and
- additive versus replacement force semantics.

The compiler validates `HumanPack.target-coverage.v1`, the current body-
composition release join, and the route mass-partition candidate. Current route
volume and mass are retained as candidate moments. First and second moments,
motor compartments, and all five physical owner roles remain explicitly
unresolved unless a future source-bound authoring declaration supplies them.
The current body-composition join reports zero production physical owners, so
v1 admits only source and candidate declarations and rejects every production
binding, owner, moment, force rule, action, closure, or qualification label.
Blocked source coverage and unresolved current registers make the ownership
manifest blocked; authoring declarations cannot erase or downgrade retained
source identities, conflicts, or qualification facts.

The route partition is checked against the exact, hash-bound route-volume
identity receipt named by that partition. Source action indices, equal-incidence
policy, disjoint candidate partition, non-mechanical status, source totals, and
mass/volume residuals must all close before actions are admitted.

`second_mass_moment_kg_m2` means the raw tensor `integral(x x^T dm)` in the
declared frame, not an inertia tensor. Candidate tensors must be symmetric and
positive semidefinite; when first and zeroth moments are present, the centered
tensor must also be positive semidefinite. Zero mass cannot carry nonzero
spatial moments.

The route force rule is recorded as a candidate replacement of the existing
source `J^T` contribution under the exact reserved
`<route-semantic-id>/owner/source-jt` identity. Other replacement targets must
be declared candidate owner IDs. These rules are non-executable authoring
records: they are not active continuum-force owners and cannot be interpreted
as additive force permission.

Compile a previously materialized target-coverage manifest with:

```sh
.numi/commands/human-ownership-compile \
  --coverage /path/to/current-target-coverage.json \
  --output /path/to/humanpack-ownership.json
```

The output is canonical and immutable. Reusing an output path with different
bytes fails closed. Source declarations, candidate geometry/moments, semantic
actions, and successful authoring validation do not qualify physical mechanics,
materials, calibration, standing, gait, physiology, or an integrated Human.
