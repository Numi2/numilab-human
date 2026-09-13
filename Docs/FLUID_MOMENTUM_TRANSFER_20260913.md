# Explicit fluid-momentum transfer owner — 2026-09-13

The native vascular contract now has an explicit `bloodMomentumTransfer` owner
flag. A source tissue can opt into finite-volume blood-momentum reaction only
when it already owns a registered blood compartment, has an authored pressure
reaction, and names exactly one matching inertial vascular edge. The compiler
rejects an unowned or ambiguous request. The high identity bit is masked when
reading the blood owner, so existing mass and moment ownership remains stable.

For the opted-in owner, the Metal wall force adds the finite-volume reaction

```text
-rho * V / A * dQ/dt * direction
```

to the authored pressure reaction `-A * (p_from - p_to) * direction`. The
production cavity force and coupled operator receive the accepted vascular
state and registered connection table, so the force and Jacobian use the same
flow perturbation. No vessel, density, direction, or area is inferred from
mesh geometry.

Native evidence was captured on the physical Apple M4 Pro after the ABI39
build and published from branch `human-blood-mass-20260913` at commit
`2c9ea8d09fb149f0ceff147b69fa97cff2e73199`. The [full native log](media/organ-blood-cavity-bridge-20260913/native/fluid-momentum-20260913.log)
and [machine-readable receipt](media/organ-blood-cavity-bridge-20260913/native/fluid-momentum-receipt.json)
contain the compiler, Human binding, cavity, legacy vascular, and cardiac
checks. The log SHA-256 is
`888ce9b1cdeb7e4825b37c0eba9381aef89b5dbe1e1c37db955c1e321a6eac95`.

The new synthetic owner check passes with an authored section of
`0.00019999999494757503 m²`, a pressure reaction of
`-0.019999999494757503 N` per environment, and a fluid-momentum reaction of
`-0.0021199999722831524 N` per environment. It also passes bitwise replay,
finite-difference force conservation, and the coupled Jacobian. Existing
cavity geometry, moving-wall, exact-clock, rollback, legacy vascular, and
cardiac transaction checks pass in the same ABI39 build.

This receipt is an engineering subgate. The current Human source bridge still
has no body-frame vessel direction/area registration, subject-specific density
or material calibration, or tissue-side exchange law. The binding output
therefore remains `pressure_momentum=unqualified`, `two_way_transfer=unqualified`,
`anatomical_registration=unqualified`, `subject_calibration=unqualified`, and
`standing_walking=unqualified`. It does not close anatomical organ mechanics,
full two-way blood/tissue transfer, force convergence, activation calibration,
material closure, or sustained standing/walking.

The same owner and cavity path was requalified again at native commit
`e07026ab3ad497a869ee247cdfd1aa23ecc16ebb` on the physical Apple M4 Pro as
part of a focused 17-test native selection (17/17 passed). The direct output
retains `pressure_gradient_reaction=pass` and
`pressure_driven_fluid_momentum=pass` while preserving the explicit
`anatomical_registration=unqualified` and `subject_calibration=unqualified`
outputs. The current receipt and hash-bound logs are under
[`Docs/media/organ-blood-cavity-bridge-20260913/native-current/`](media/organ-blood-cavity-bridge-20260913/native-current/).
