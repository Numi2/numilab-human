# Registered blood mass owner: bounded native increment

This receipt records the next addressable blood-mechanics slice for the single
source-bound adult-male release. It registers blood mechanical mass against a
real FEM tissue region while preserving hydraulic volume as the only volume
authority. It does not claim full blood momentum, organ calibration, or Human
completion.

## Source boundary

The native implementation is the isolated Mac mini worktree
`human-blood-mass-20260913` at commit
`e6a695eb4a947bcd91c277c1adc119f764138b12`, based on
`53670294dd229e5a0d876a472964130742c89e44`. The qualified native checkout was
not modified. The commit is published on
`origin/human-blood-mass-20260913`. The implementation raises
`NM_MATTER_ABI_VERSION` to 38 and requires recooking affected Matter packages.

The authoring contract adds `bloodCompartment` and `bloodDensity` to a vascular
tissue. A nonzero compartment must resolve to one hydraulic compartment, use a
positive density, name a real FEM object, and provide a normalized nonempty FEM
region. The compiler rejects missing identities, nonpositive density, absent
regions, and duplicate mechanical owners. Owner indices are stored as
compartment-index plus one; zero remains the explicit no-owner value.

ABI38 also cooks the region's normalized first spatial moment and symmetric raw
second spatial moments in the authored FEM frame. Package validation recomputes
those moments from the actual cooked node positions and rejects stale or
nonfinite metadata. The initial registered blood mass is now partitioned into
the owner's real FEM nodal inertia using the normalized bindings. Runtime adds
only the current-volume correction, including co-moving inertial transfer, so
initial gravity and inertia are not double-counted. A production Metal audit now
recomputes current mass first/second moments and co-moving linear momentum from
the accepted deforming FEM nodes, with a bitwise replay gate.

At runtime the compiler-owned initial blood mass is already present in the
FEM nodal inertia. The accepted hydraulic compartment volume is multiplied by
the registered density; only the difference from that initial mass is
distributed over the owner's normalized FEM bindings as gravity and co-moving
inertial correction. The per-environment hydraulic state is offset separately
from the object-local FEM binding index. The merged force buffer is admitted
only when a cavity or explicit blood owner exists, and the owner/binding buffers
are carried through the protected arena and transactional encode path.

## Mac mini evidence

Host: physical Apple M4 Pro, `ssh macmini`, Release build, ABI 38. The complete
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
initial_mass_kg=0.0042400001548230648 dynamic_volume_delta_m3=4.9999999873762135e-07
dynamic_force_total_N=-0.010398597456514835 partitioned_inertia=pass
dynamic_gravity_correction=pass dynamic_spatial_moments=pass
co_moving_inertia=pass replay=bitwise
```

That gate checks both environments, every owned and unowned FEM node, normalized
regional weights, exact initial mass partition, current-volume correction,
co-moving acceleration transfer, total-force conservation, finite status, and
repeated-kernel bitwise equality. The existing coupled
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

This increment closes explicit zeroth-order ownership, initial FEM mass partition,
reference first/second spatial moments, dynamic-volume gravity correction,
co-moving inertial transfer, and a production audit of current dynamic spatial
moments and co-moving linear momentum. It does not provide time-integrated
dynamic-moment closure, independent fluid inertia, pressure-driven momentum transfer,
reaction-force closure, wet/dry mass partition for anatomical organs, gas or
metabolic state, thermal/fluid balance, source activation, or subject-specific
density and perfusion calibration. The synthetic density and volume are fixture
values, not anatomical measurements. The completion acceptance for
`blood.spatial_owner` therefore remains open until time-integrated moments,
two-way momentum, atomic restore, anatomical registration, and held-out
calibration evidence pass together.
