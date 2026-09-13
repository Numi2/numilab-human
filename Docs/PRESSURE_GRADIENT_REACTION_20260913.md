# Native pressure-gradient wall reaction — 2026-09-13

The ABI38 vascular owner now accepts an explicit pressure-gradient wall
reaction on a registered FEM tissue region. The source contract names two
stable pressure compartments, a finite unit path direction, and a positive
cross-sectional area. The compiler resolves the endpoints canonically,
rejects partial or non-unit contracts, and preserves the fields through the
package round trip. The native Metal force path evaluates each compartment's
actual pressure law, including deforming-cavity pressure rows and prepared
cardiac elastance, then applies `-A (p_from - p_to) direction` to the registered
region. The same derivative is included in the FEM operator so the hydraulic
pressure direction and mechanical tangent share one source of truth.

This is a mechanical wall-reaction path. It does not invent blood density,
perfusion, an anatomical vessel tube, or a separate fluid momentum state.
Therefore it reduces the engineering gap while keeping biological and
anatomical qualification fail-closed.

Native evidence:

- Mac mini: physical Apple M4 Pro, Matter ABI38.
- Branch and commit: `human-blood-mass-20260913`, `2e51092b070260962b3ab343c3048ba222a0b3b0`.
- Source/binary hashes and exact output: [pressure-reaction log](media/organ-blood-cavity-bridge-20260913/native/pressure-reaction-20260913.log).
- Log SHA-256: `4d5f31bcedcc26d21330ae1e585dcf117d3494af49a4d333e8b8676af3c21f45`.

The physical run passed:

```text
pressure_gradient_reaction=pass from_compartment=12 to_compartment=11
area_m2=0.00019999999494757503 total_force_per_environment_N=-0.019999999494757503
replay=bitwise jacobian=pass
vascular_compiler=pass ... semantic_rejection=true provenance_bound=true
human_organ_blood_native_binding=pass
```

The synthetic pressure is 100 Pa across a 2e-4 m² section and is distributed
across eight registered shell nodes. Existing cavity equations, moving-wall,
clock, rollback, package, organ identity, and ABI38 owner checks also pass in
the same build. The Human four-chamber source remains `pressure_momentum =
unqualified` because it has no registered vessel direction/area contract or
body-frame vessel geometry. Two-way blood/tissue transfer, calibrated density,
anatomical organ mechanics, systemic perfusion, force convergence, sustained
standing/walking, and the 420-trial behavior gate remain open.
