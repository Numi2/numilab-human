# Local source-route geometry stage

Status: source staged locally only; no remote edits and no GPU execution. This
stage follows the independently qualified compensated root/FK/support cohort.
The exact 11 proposed native paths, byte sizes, proposed hashes, and required
base hashes are in `source-sha256.json`; `source-route.patch` is their unified
diff. Apply only after the base hashes match the newly published native owner.

The three paired wrappers use the same production source routines as the legacy
entries. They keep world points in high/low form through attachment rotation,
wrap-center subtraction, endpoint/tangent differences, and Jacobian moment
arms. The source wrap switches, scalar wrapping/trigonometry, muscle activation,
force/length/velocity, compliant-fiber, and tendon force-map equations are
unchanged. Full-route, cut/suffix, distributed terminal transfer, and hood
assembly are migrated together. Legacy GPU entry points and buffer slots are
preserved.

The optional C++ `MetalNumiHumanTendonLoadPass` companion has `bodyPositionLow`,
`bodyPositionLowGPUAddress`, and `bodyPositionLowElementCount`. The live Stand
owner supplies its already admitted paired body array; a null pointer with zero
address/count preserves legacy use. Hood validation checks exact address/count,
length, device, and direct aliases. The hood keeps its Newton unknowns in a
bounded ray-local FP32 chart rooted at the first source node's paired world
position. Its private `relativeNodePositionBuffer` is recomputed each solve and
consumed by assembly in the same borrowed command buffer. It does not add a
second state or queue. Existing node result positions remain a rounded world
diagnostic; neither assembly nor force computation reads that rounded diagnostic
in paired mode. The ray result's moment closure retains its original world-origin
meaning. Published result, serialized source, and GPU descriptor ABIs are unchanged.

Local checks retained in `compile-003`:

- Both host Objective-C++ source files passed syntax compilation (`compile-002`).
- All six legacy/paired shader units compiled at Metal 4.0, `-O3 -fno-fast-math`,
  with no warnings; their combined metallib linked without duplicate entries.
- The driver compiled with `-Wall -Wextra -Werror`; `--cpu` passed 59 assertions.
  The existing FP64 Core source oracle checks straight, sphere, cylinder,
  nonzero prismatic route derivatives, actual full/suffix difference, nonzero
  source force, and rigid-motion invariance. Its copied library SHA is recorded.
- The Metal checks are **not run**. CPU-only mode never creates a Metal device
  and never advances physics.

Parent integration requires CMake additions for these precise shader units:

```
MujocoMuscleReferenceCompensated.metal
NumiHumanTendonTransferCompensated.metal
NumiHumanExtensorHoodCompensated.metal
```

Each requires `-fno-fast-math` and dependency edges to its corresponding base
source, `PairedSourceGeometry.metalinc`, `compensated_geometry_gpu.h`, and
`compensated_translation_gpu.h`. Retain the old shader compile flags for the
legacy units. Add `apps/source_route_precision_check.mm` linked to `metalrobo`,
Foundation, and Metal, with C++20, `-fno-fast-math -ffp-contract=off`.

Invocations, with the parent's chosen executable name:

```
source-route-precision-check --cpu
source-route-precision-check /path/to/actual/production/Numi.metallib
```

The second invocation must run on the physical Mac mini under the parent's sole
GPU-runner coordination. It dispatches actual production paired full-route,
suffix, terminal, hood solve, and hood assembly entry points. The predeclared
numerical controls are in the driver: origins 0/64/4096 m, independent FP64 path
and attachment finite differences, Jv and muscle-force checks, large-origin
legacy negative control, force/moment conservation, and a closed-form prestrained
three-bar/foundation hood equilibrium. Its free node belongs to a separate
prismatic body, producing a nonzero independently known source replacement
correction. No thresholds have been tuned against a GPU result.

Limits: this does not change FP32 joint/quaternion/trigonometric constitutive
inputs, local hood Newton arithmetic, or force-vector storage. The direct route
fixtures use identity orientations; nonidentity paired rotation itself has
separate existing CPU/Metal coverage. Millard's separate route owner is outside
this patch. Local source/compiler checks do not qualify the actual Human source
case, biological calibration, standing/walking, or long-horizon behavior. After
direct production tests pass, rerun the exact authored 25/50/100 microsecond
accepted-root cohorts, source musculotendon regressions, and replay with original
gates; retain any failures.
