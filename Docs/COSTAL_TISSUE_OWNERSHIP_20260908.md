# Registered costal tissue ownership, 8 September 2026

The new `costal-binding` compiler and native runtime v5 connect fourteen source
costal-cartilage regions to the current Human without adding duplicate tissue
mass. The binding preserves the current rigid bytes and the historical source
registration through explicit source-body pose checks. This advances the
[neuromusculoskeletal release](NEUROMUSCULOSKELETAL_RELEASE_MATRIX.md); the
complete Human release remains unfinished.

## Implemented path

`NHTBIND1` binds actual NHCART1, NHRIGID2 and registration bytes. It retains the
named sternum/manubrium/rib mesh identities and admits one common proper
similarity, rather than independently moving attachment endpoints. Missing
manubrium, source/body/pose drift, reflection, shear and split-thorax ownership
without an explicit volumetric map fail admission.

The common atlas-metres to world-metres transform has scale
`1.0076111869732587` and translation
`[-0.02486208348842422, 0.24627952500546432, 0.12388234159326472] m`.
It is a source-bound reference candidate, not a validated subject registration.
All current rib/sternal anchors and the mass donor resolve to torso body 20.
The explicit assumption is that the source gross torso inertia includes this
costal volume. Subject-specific evidence is still needed for that assumption.

The native compiler transforms nodes before final FP32 cooking, then partitions
mass from the actual Matter nodal masses. It conserves zeroth, first and second
mass moments, rejects impossible residual inertia, and keeps the full residual
inertia tensor. It rebases joint anchors, shapes and the floating-root reference
when applicable. It recooks tissue attachment locals in the new COM frames and
requires bit-identical world-space positions and cooked masses.

Runtime v5 consumes the same partition and rebases muscle sites/wraps, support
contacts, physical point queries, camera origin and visual body bounds before
entering the existing Human–Matter transaction. NHEQ2 source constraints remain
active. The binding and cooked world identities participate in the runtime
model fingerprint. Older v4 admission rejects the rebased package against its
original body frames. Swift transports the immutable v5 descriptor; it does not
step physical state.

## Measured geometry and mass

| Quantity | Result |
|---|---:|
| Regions / nodes / tetrahedra / attachments | 14 / 13,516 / 46,278 / 2,871 |
| Cooked tissue mass | 0.11369939548001184 kg |
| Original torso mass | 18.618999481201172 kg |
| Residual torso mass, packed | 18.505300521850586 kg |
| Residual COM offset in old frame | [-0.000905929949, 0.000246364545, -0.000001278821] m |
| Maximum packed mass-moment relative error | 4.04762051e-8 |
| Maximum original-frame attachment reconstruction error | 6.98491970e-9 m |
| Eight-pose CPU point / Jacobian error | 1.49011612e-8 |
| Eight-pose CPU velocity error | 2.51266619e-9 m/s |
| Eight-pose M4 Pro point / Jacobian error | 8.34605958e-7 m / 8.57668762e-7 |
| M4 Pro isolated donor mass-matrix scaled error | 1.49011612e-8 |

The GPU checks ran with Metal API validation and eight bit-identical Jacobian replays.
The complete 157-body dense Metal mass-matrix diagnostic exceeds its supported
bucket. Full-body kinematics and the isolated donor tensor are tested;
whole-body dynamic mass-matrix qualification remains false.

The CPU mass oracle passes 64 rigid-twist momentum/energy cases and 11 negative
cases, including impossible distributions and repeated application. The native
binding audit passes seven negative cases. The runtime rejects four distinct
v5/legacy admission mismatches. Human's Python suite passes 169 tests with
seven skips and 111 passing subtests. The Swift v5 size/offset/dependency test
and five selected native compatibility regressions pass. The final published
Brain source also passes both legacy joint-publication tests and three
connectome audit regressions.

## Joint transaction evidence

On the initial Brain base, the full release-mode test passes eight accepted
10-microsecond roots (80 microseconds of committed physical time), stale-root
rejection, quarantine/restore and bit-identical accepted-gate and sensor retry.
The developmental Brain seed emits zero muscle excitation in this fixture;
active motor control is not qualified. All seven sensory modalities enter
committed learning records. Deterministic
MLX update, successor materialization and 28 policy-head interventions pass.
The invocation takes 281.739 seconds including initialization and learning;
this is a bounded transaction test, not sustained behaviour or a throughput
benchmark. Its original failed attempt reached four roots before a missing
release MLX metallib stopped the learning stage. Both records are retained.

The integrated Brain base adds the upstream connectome work. Its first build
exposed a missing audit export in the SwiftPM umbrella header; a narrow include
fix restores that public C interface. The final costal test also passes on the integrated source in 283.364 seconds,
with all eight roots and no XCTest failures. The publication merge adds only
an upstream workflow and an inert patch artifact; executable sources, tests,
package and dependencies are byte-identical to the tested revision. Exact
revision and artifact identities are in [the receipt](media/tissue-ownership-20260908/receipt.json).

## Reproduce

From Human with Python 3.11 or newer:

```sh
PYTHONPATH=src python3 -m numilab_human.cli costal-binding \
  --cartilage-manifest Build/costal-cartilage-v1/bodyparts3d-costal-cartilage.manifest.json \
  --human-manifest Build/flex-eq-current/myosim-fullbody-reference.manifest.json \
  --registration Build/fullbody-articular-v3.registration.json \
  --output Build/tissue-ownership-20260908/binding
```

The native `numi_human_tissue_binding_check` takes the binding, cartilage,
rigid payload and registration paths, followed by `--metal` and
`--output-world=costal-rebased.nmatterpack`. Its detailed mass/frame and ABI
contract is `numi-lab/docs/HUMAN_TISSUE_MASS_OWNERSHIP.md`.

Build NumiBrain's release tests, then run
`tools/build_swiftpm_mlx_metallib.sh release` before the joint-publication test.
The evidence directory retains the exact host invocation and artifact hashes.

## Admission boundary and next work

The costal material is still the corrected population prior described in the
[tissue calibration update](TISSUE_INTEGRATION_CALIBRATION_20260908.md):
`E=22 MPa`, assumed `nu=0.45`, density `1100 kg/m^3`. No experimental costal fit
or sealed specimen validation is supplied by this integration. The existing
patellar candidate uses previously inspected same-plug repeats; its held-out
test day is not blinded or an independent specimen.

Independent rib/sternal mechanics, source prestress equilibrium, tissue contact,
loaded deformation and timestep/mesh convergence remain required. Deformable
contact is explicitly disabled in this package. The experimental passive tissue solve is active; no production force-ownership
promotion is made. The binding manifest describes compiler authority only;
accepted-root evidence belongs to the separate runtime receipt. The current source law, experimental calibration, physical
validation, behaviour qualification and performance rows remain distinct.
