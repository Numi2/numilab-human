# Paired source-route geometry

This increment extends the [accepted-coordinate owner](ACCEPTED_STATE_PRECISION_20260913.md)
through muscle routes, route suffixes, terminal transfer and extensor-hood
geometry. It keeps the source wrap switches, constitutive laws and force-map
equations unchanged. The source-route acceptance gate still requires the
full source-force convergence, timestep refinement and held-out mechanical
evidence. Native owner `53670294dd229e5a0d876a472964130742c89e44` is qualified
with Brain `38c4b5c72be88dc83c9c2364b5da47fb10239d71`. Exact source and artifact
identities are in [publication.json](media/source-route-precision-20260913/publication.json).

## Geometry ownership

The existing paired body positions now supply the production MyoSim route,
suffix and tendon-transfer entry points. Attachment rotations, wrap-center
subtraction, tangent reconstruction and Jacobian lever arms retain the low
coordinates until the local geometric difference is formed. Legacy entry
points and their buffer slots remain available.

The hood solves its local unknowns in a ray frame anchored to the first source
node's paired world position. A private relative-node buffer connects solve and
assembly in the same borrowed command buffer. Published world positions remain
rounded diagnostics; paired assembly does not read those rounded positions.
The buffer is derived work, with no second state owner, queue or physical clock.

## Retained physical discriminator

The first direct GPU run failed a sphere path: paired length 0.511700988 m
versus FP64 0.503479396 m. A diagnostic rerun showed the legacy GPU length
0.503479421 m agreed with the oracle. Independent body-position checks ruled
out a fixture-pose error.

The shared source helper had global inline functions with identical parameter
signatures but different return types in the legacy and paired shader units.
The AIR files contained the same external symbol. Giving the conditional
helpers internal linkage removed this collision. Two suffix tangent rotations
also now use the same paired rotation helper as the full route. Original
failures and their source identities remain in the
[evidence directory](media/source-route-precision-20260913/).

With that repair, `direct-controls-003` passes 89 CPU assertions, 2,972 Metal
checks and the existing independent FP64 hood reference. No numerical
threshold was relaxed. The physical M4 Pro driver evaluates straight, sphere
and cylinder routes, full and suffix derivatives, nonzero source forces,
distributed terminal conservation and a prestrained hood with a closed-form
three-bar/foundation reference. Each runs at world origins 0, 64 and 4,096 m.

Maximum route-length error is 3.92e-8 m, route-Jacobian error 6.97e-8 and
source-force error 3.67e-7 N in these fixtures. Hood position and force errors
are 1.73e-9 m and 1.73e-7 N. Measured origin-dependent differences are zero for
the paired outputs. The direct route fixtures use identity orientations;
their results do not establish full-body source-path accuracy.

## Authored model and admission

An explicit read-only paired mode now admits canonical root coordinates to
the existing articulated FK and MyoSim owner without enabling Stand. It
allocates the required coordinate companions, checks scratch capacity and
selects the production paired pipelines. A live published resident context
rejects this query before mutation. Empty root input retains legacy geometry.
Dependent C++ consumers must be rebuilt; the C ABI and cooked formats are
unchanged.

`--prepared-compensated-paths` evaluates all 416 authored routes against the
existing independent native FP64 reference at the same admitted FP32 q/v and
initial muscle state. Six invalid-input controls pass, as do bitwise repeated
evaluation and paired → legacy → paired reuse of the same context. Input
state remains unchanged and no physical root or accepted timestamp is
published. A positive timestep evaluates only the private source constitutive
map.

| Authored constitutive timestep | Maximum path error | Full source-force difference | Same-path force difference |
| --- | ---: | ---: | ---: |
| 100 µs | 0.128102 µm | 0.057521 N | 0.002789 N |
| 1 µs | 0.128102 µm | 0.378799 N | 0.002760 N |

Both runs pass the existing 2 µm path, 1e-5 normalized same-path force and
0.501-ULP fibre-publication gates. The full source-force differences are
retained observations; those same-path gates do not establish full source
convergence or empirical force calibration.

Hood admission now checks exact consumed GPU-address intervals, including
overflow-safe extents, device identity and the shader's index capacity. It
rejects distinct heap buffers whose low-coordinate region overlaps source,
result, generalized-force, body or Jacobian data. The public-callback probe
passes 41 controls, including 16 distinct heap overlaps and valid disjoint,
adjacent and legacy-null cases. It encodes admission controls without
committing any GPU command. The final `core-controls-001` run passes all 23
selected native checks on the physical Mac mini.

## Coupled execution and replay

The final library and shaders pass the original 16-iteration nonlinear budget
and 0.005 tolerance at all three supported timestep levels. Each scenario
covers the same 1.6 ms interval. The independent verifier joins all ten state
trace kinds and the 48-byte root coordinate to each actual committed
transaction, generation and timestamp, then compares every recorded replay
byte. Source, runtime and fixture identities remain unchanged during each run.

| Timestep | Accepted roots including replay | Wall time | Result |
| --- | ---: | ---: | --- |
| 100 µs | 32 | 40.867 s | complete, bitwise replay |
| 50 µs | 64 | 79.412 s | complete, bitwise replay |
| 25 µs | 128 | 157.621 s | complete, bitwise replay |

The final production Brain–Human–Matter metric run passes in 8.056 s with
four accepted roots, four samples and an identical repeated final collection.
Its unavailable contact/audit/reset/TaskPack/proof fields remain unavailable,
and `full_behavior_qualified=false` remains mandatory.

The [refinement comparison](media/source-route-precision-20260913/refinement-observations.json)
uses sixteen common observation times and admits only byte-identical authored
state and solver inputs apart from the exact timestep and dependent identities.
Maximum root-axis differences are 6.017e-7 m (100 versus 50 µs) and 3.907e-7 m
(50 versus 25 µs). Maximum applied-force differences are 0.31317 N and
0.38963 N respectively. Force differences still do not decrease with refinement;
the 12.5 µs clock path and the complete convergence gate remain open. Twenty-one
trajectory and comparison negative tests pass against the retained records.

## Qualification boundary

FP32 quaternion composition, wrapping trigonometry, local hood Newton unknowns
and force-vector storage remain. Millard's separate route owner is outside
this increment. Private constitutive evaluation at an authored timestep does
not advance an accepted physical root or qualify loaded equilibrium.

Full source-force consistency, the 12.5 µs transaction clock, sustained loaded
anatomy, calibration and standing/walking retain their existing gates. The
execution registry continues to distinguish engineering checks from source
data, calibration and biological evidence.
