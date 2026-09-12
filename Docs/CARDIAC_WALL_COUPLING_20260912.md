# Native pressure–volume wall coupling — 12 September 2026

Matter now couples an absolute-volume hydraulic compartment to a deforming
closed FEM cavity. Pressure deforms the wall, and wall geometry constrains the
conserved hydraulic volume in the same Newton–FGMRES solve. The physical M4 Pro
qualification passes a synthetic hollow-wall case with 32 accepted 1 ms steps
in each of two environments, a fixed-wall control, conservation, exact replay,
rollback, reset, and invalid-state rejection.

This closes the native pressure–volume interface and its bounded numerical
qualification. Anatomical heart-wall integration, calibrated material and
loading parameters, blood mechanical mass and momentum ownership, and
whole-Human standing/walking remain open.

## Permanent owner and compatibility

The native implementation is published as
[`f73b071`](https://github.com/Numi2/numi-lab/commit/f73b07133d09ceb9ed7ff94d48d5863157f2527b)
on `coupled`. All 14 qualified source hashes match that commit; the runtime
binaries remained unchanged during the final fixture-only recheck. The
public interface is `WorldSource::vascular.cavities` with
`VascularPressureLaw::deformingCavity`. Existing Human physiology JSON v1–v3
retain their lumped-network meanings. The hollow-shell fixture demonstrates
native C++ authoring; it is not an anatomical payload or a second simulator.

A bound compartment retains its conservative volume equation and gains an
independent pressure unknown plus `V_hydraulic - V_wall(x) = 0`. Its old lumped
compliance and elastance must be absent. Connections consume the independent
absolute pressure; the wall receives transmural pressure work. The wall force
uses a Simpson discrete gradient of the closed surface's cubic volume, and the
operator differentiates both pressure and wall geometry. Geometry participates
in line search and final certification. Bound walls use every hydraulic
microtick, including after reset and restore.

Real material boundary faces, positive enclosed volume, closed oriented
vertex-manifold topology, exact object ownership, matching cooked initial
volume, and provenance identities are required. Artificial partition faces and
filled solids presented as lumens are rejected. Source hashes and topological
admission do not establish embedded anatomical myocardium or calibrated wall
mechanics.

Current compatibility is **Matter ABI 29, package 14, snapshot archive 7, and
accepted-proof manifest 7**. Pressure, volume, flow, species and exact clock
share Matter's existing accepted-state transaction with the FEM wall. Older
payloads require recooking; historical receipts keep their original revisions.
Existing Human attachment lifting and transpose transfer are reused, while
this increment's executed mechanical qualification is the standalone hollow
FEM shell.

## Physical Mac mini results

The fixture has 64 nodes and 156 tetrahedra surrounding an absent central cell.
Its 12 oriented inner triangles enclose 1 mL. Eight inner nodes move; 56 outer
nodes are fixed. The tissue mass is 26 g, with explicitly synthetic density,
Lamé parameters. No mechanical blood mass is added.

| Measured quantity | Result |
| --- | ---: |
| Accepted trajectory | 32 steps × 2 environments, approximately 32 ms each |
| Maximum inner-wall displacement | 17.1811 µm |
| Maximum flow, moving wall | 9.74518e-8 m³/s |
| Maximum flow, fixed wall | 5.36442e-14 m³/s |
| Maximum normalized geometry residual | 1.21471e-7 |
| Maximum normalized volume-balance residual | 5.83304e-8 |
| Maximum normalized flow residual | 3.21675e-7 |
| Relative total hydraulic-volume error | 1.43051e-7 |
| Relative total tracer-amount error | 1.07510e-7 |
| Production force versus independent FP64 oracle, relative error | 8.23427e-8 |
| Production force-work identity, relative error | 4.89198e-7 |
| Production force Jv versus finite differences, relative error | 1.88243e-5 |
| Production geometry Jv versus independent FP64 derivative, absolute normalized error | 1.73580e-10 |
| Production flow-pressure Jv versus independent derivative, absolute normalized error | 1.05836e-8 |

The direct kernel tests also check rowwise finite differences of all hydraulic
and transport equations, independent signed pressure, translated geometry,
common absolute/external pressure shifts, preservation of borrowed external
loads, and static/converged rows. Accepted-step tests verify unchanged nodal
mass bits, fixed exterior nodes, paired environments, exact clock increments,
bitwise replay, isolated rollback and reset. Finite negative pressure restores;
NaN pressure, inconsistent positive volume versus geometry, invalid scheduler
cadence, and infinite scheduler enablement are rejected without publishing
state.

The FP64 calculations are geometry, force, derivative and work oracles. Matter
on Metal performs every physical step. The interface work identity does not
establish zero integration error or conservation of the entire model's energy.

## Failures retained and repaired

- The first integrated build rejected a deprecated test-only Metal library
  loader. The test now uses the supported URL API; warning checks remain on.
- The first GPU direct-equation check caught a missing reservoir elastance in
  its manufactured input. The test now runs the production coefficient-preparation
  kernel before residual and Jv evaluation. The independent analytic check and
  all original assertions remain.
- The first accepted trajectory exposed an unconditional topology rebuild in
  an immutable world. Fixed node 1's mass changed from float bits `39aec33d`
  to `39aec33e`, a one-ULP recomputation unrelated to pressure. Matter now skips
  complete topology transactions and mass rebuilding when no mutable FEM or
  mutation command can require them. Mutable worlds and explicit commands keep
  their existing validation and transaction path. The exact mass check remains.
- The broad compatibility run passed 18 of 19 checks and caught an old direct
  cardiac-kernel fixture missing the ABI 29 pressure-base field and new empty
  boundary bindings. That fixture is migrated to the new ABI without changing
  pressure-law expectations; its targeted recheck is recorded separately.

**All 19 selected checks pass**: 18 passed in the broad run and the migrated
cardiac fixture passed its targeted recheck. The final selected checks cover
vascular/compiler/JSON admission, Shi–Hose and
CVSim source runs, cardiac clock/valve transactions, snapshot archive,
immutable stateful FEM, mutable topology/cohesive/puncture/conservation and
rollback, production rollback, and Human tendon/FEM transactions. This is a
focused current-stack regression set, not a requalification of all previous
Human receipts or long-horizon source-model results.

## Evidence and remaining anatomy

The [qualification record](media/cardiac-wall-coupling-20260912/qualification.json),
[publication record](media/cardiac-wall-coupling-20260912/publication.json), and
[evidence directory](media/cardiac-wall-coupling-20260912/) retain command
arguments, exit codes, elapsed times, complete output logs, native source and
binary SHA256 identities, the publication record, and the failed attempts.
Native testing ran through `ssh macmini` in the isolated Human completion
checkout. The unrelated omics CPU job was left running; GPU work was serialized.

The next anatomical dependency is a valid material wall with a matching lumen,
registration, physical port/valve ownership, and calibrated loading. Current
BodyParts3D ventricular-wall topology requires repair and independent admission;
the existing torso visual mapping selects a right-atrial wall member rather
than a whole mechanical heart. The earlier
[RA/RV partition alternatives](CARDIAC_CAVITY_PARTITION_20260912.md) remain
geometric alternatives, with no selected biological interface.

Mechanical blood ownership additionally needs a sourced density and explicit
wet/gross donor inclusion, spatial mass and inertia, and conservative momentum
transfer. The existing organ visual map does not supply that accounting. The
other aggregate circulation regions, organ perfusion, reflex/tilt response,
biological calibration, active organ mechanics, standing and walking remain
unqualified.
