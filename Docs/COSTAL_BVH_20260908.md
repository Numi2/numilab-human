# Costal contact search: measured runtime repair

The unchanged costal transaction test now completes in **29.047 seconds instead
of 277.393 seconds**, a **9.55× improvement** in runner time on the Mac mini.
XCTest itself takes 26.693 seconds. Both runs enable Metal API validation, use
the same input artifacts and Brain test binary, and retain the 60-second
physical-callback deadline.

The published native revision is
`2ca77bbcb0b7976b1706f3a9c8e5b0274fc08a75` on `origin/coupled`;
Brain remains `30f138fa17378fce788701edb5c7d834ee3c7c69`. The final run records
clean source trees. It covers the source-default NHCNT1/NHEQ2/NHLIM1 costal
fixture: 46,278 tetrahedra, 2,871 attachments, eight accepted 10-microsecond
roots, and the rejected/retried candidate. This is 80 microseconds of physical
time, not sustained standing or prepared anatomical equilibrium.

## What changed

Direct GPU timestamp sampling isolated the exhaustive deformable surface-pair
scan at approximately 3.44 seconds per assembly; sorting took approximately
25 milliseconds. The earlier system trace had localized the cost to combined
coupled GPU stages. The counter run is retained as an intentionally interrupted
profiling experiment; it is not a successful behavior trial.

Matter now builds a balanced bounds hierarchy over its existing stable Morton
order. Independent left-surface traversals reject disjoint subtrees. Their
ordered count/scan/scatter keeps candidate membership, source ordering, swept
bounds and eligibility rules intact. The count phase also repairs a capacity
bug: filling a prefix exactly no longer hides additional eligible pairs.
Overflow rejects the environment before scattering candidates.

The hierarchy is regenerated scratch state. It adds 16 MiB for this costal
world, with checked device index bounds and normal resident-memory accounting.
Sort scratch is reused for counts and offsets. Zero contact capacity allocates
no hierarchy. This is not a full-run peak-memory qualification.

Validation also exposed missing specialization bindings in MetalWorld's
borrowed articulated queries. A shared function now selects the operator and
binds its required source-program or task-parameter inputs together. The
active coupled-contact regression now enables Metal API validation by default.

## Evidence and limits

Seven native tests pass: the new surface hierarchy probe, CCD, mixed MPM/FEM,
cohesive mutation, transaction rollback, active shared rigid contact, and Human
tissue attachment. The new probe checks 13 fixtures across three environments,
including exhaustive pair/order comparison, touching bounds, object masks,
shared nodes, cohesive lineage, exact-prefix overflow, changed geometry and
restored replay. Large separated geometry has an independent separation witness.

The final costal XCTest passes. All nine Matter and Human status records,
nine risk-diagnostic records and 28 policy-head records match the prior run
exactly. This is exact diagnostic equality, not a complete cross-revision
q/v/FEM state comparison. Existing native replay and transaction tests remain
separate evidence.

The prior timeout attempts and interrupted contention run remain in the
[earlier qualification](COSTAL_IDLE_REQUALIFICATION_20260908.md). This repair
removes a measured dominant cost; it does not establish the sole cause of those
historical timeouts or qualify the five production performance workloads.

The [loaded-support diagnostic](media/costal-bvh-20260908/support-history-current.json)
still fails five of seven physical cases on this new native build. In the
97 kg cold-history case, the production support kernel supplies only 1.222% of
the required weight impulse. The harness exits 2 deliberately. Contact
complementarity, loaded source-compliant equilibrium, registered prepared
tissue, sustained standing/recovery/walking and independent calibration remain
open. Faster replay must not be substituted for those outcomes.

Run `python3 Docs/media/costal-bvh-20260908/compare_runs.py` to reproduce the
bounded timing and diagnostic comparison. The
[evidence bundle](media/costal-bvh-20260908/) retains native test output,
the exact costal runner, launch metadata, workload observations, profiling
patch/output and the failing native/Metal support diagnostic sources.
