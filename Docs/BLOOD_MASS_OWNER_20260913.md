# Registered blood mass owner: bounded native increment

This receipt records the next addressable blood-mechanics slice for the single
source-bound adult-male release. It registers blood mechanical mass against a
real FEM tissue region while preserving hydraulic volume as the only volume
authority. It does not claim full blood momentum, organ calibration, or Human
completion.

## Source boundary

The native implementation is the isolated Mac mini worktree
`human-blood-mass-20260913` at commit
`cd2456739721be31eacbbb120638c3175b9f5f5b`, based on
`53670294dd229e5a0d876a472964130742c89e44`. The qualified native checkout was
not modified. The commit is published on
`origin/human-blood-mass-20260913`. The implementation raises
`NM_MATTER_ABI_VERSION` to 36 and requires recooking affected Matter packages.

The authoring contract adds `bloodCompartment` and `bloodDensity` to a vascular
tissue. A nonzero compartment must resolve to one hydraulic compartment, use a
positive density, name a real FEM object, and provide a normalized nonempty FEM
region. The compiler rejects missing identities, nonpositive density, absent
regions, and duplicate mechanical owners. Owner indices are stored as
compartment-index plus one; zero remains the explicit no-owner value.

ABI36 also cooks the region's normalized first spatial moment and symmetric raw
second spatial moments in the authored FEM frame. Package validation recomputes
those moments from the actual cooked node positions and rejects stale or
nonfinite metadata. These are reference geometry moments; they are not yet
dynamic mass-moment or fluid-inertia closure.

At runtime the accepted hydraulic compartment volume is multiplied by the
registered density and distributed over the owner's normalized FEM bindings as
a gravity body force. The per-environment hydraulic state is offset separately
from the object-local FEM binding index. The merged force buffer is admitted
only when a cavity or explicit blood owner exists, and the owner/binding buffers
are carried through the protected arena and transactional encode path.

## Mac mini evidence

Host: physical Apple M4 Pro, `ssh macmini`, Release build, ABI 36. The complete
build reached 100%. The focused checks were run from
`/Users/n/MetalRobo-blood-mass-build-20260913`:

```text
numi-matter-vascular-compiler-check: pass
numi-matter-vascular-check: pass
numi-matter-vascular-cavity-check: pass
```

The new direct Metal gate reports:

```text
blood_mass_owner=pass compartment_stable_id=12 density_kg_m3=1060
accepted_volume_m3=3.9999999899009708e-06 total_force_N=-0.083188809454441071
normalized_region_weight=0.125 replay=bitwise
```

That gate checks both environments, every owned and unowned FEM node, normalized
regional weights, density-times-accepted-volume force, total-force conservation,
finite status, and repeated-kernel bitwise equality. The existing coupled
cavity equations, independent FP64 wall-work/Jacobian checks, production
operators, pressure-gauge control, 32 accepted moving-wall steps, exact clock,
snapshot replay, rollback, reset, and invalid-restore rejection continue to
pass. The moving-wall receipt still reports
`blood_mechanical_mass=absent` because that synthetic path has no registered
owner; the separate owner gate is the evidence for this increment.

Raw logs and hashes are retained in
`Docs/media/blood-mass-owner-20260913/native/`:

- `full-build.log`
- `vascular-compiler.log`
- `vascular-runtime.log`
- `vascular-cavity.log`
- `native-revision.txt`
- `native-status.txt`
- `SHA256SUMS`

## Boundary that remains open

This increment closes explicit zeroth-order ownership and reference first/second
spatial moments, plus a one-way gravity body-force scatter. It does not provide
dynamic mass-moment evolution, fluid inertia, pressure-driven momentum transfer,
reaction-force closure, wet/dry mass partition for anatomical organs, gas or
metabolic state, thermal/fluid balance, source activation, or subject-specific
density and perfusion calibration. The synthetic density and volume are fixture
values, not anatomical measurements. The completion acceptance for
`blood.spatial_owner` therefore remains open until dynamic moments, momentum,
atomic restore, anatomical registration, and held-out calibration evidence pass
together.
