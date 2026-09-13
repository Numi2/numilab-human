# Blood moment and atomic restore closure — 13 September 2026

The native ABI39 blood owner now has a direct checkpoint gate for the coupled
FEM and vascular momentum state. On the physical Apple M4 Pro, the owner check
restores a snapshot containing the co-moving FEM velocity state and accepted
vascular state, recomputes the production blood moments, rewinds and restores
the same checkpoint, and rejects a NaN vascular mutation without changing the
accepted state. The restore and replay comparisons cover the accepted FEM
nodes, vascular values, exact vascular clock, and the production moment output.

The current native source is commit
`171b74062a283b04aec59f949552c38ed5cafef4` on the isolated
`human-blood-mass-20260913` branch. The parent is the declared stand-horizon
increment `a134ad9d0e6ca3416ec3e98e66c5fcb84dc05b0b`; the focused run was
rebuilt against ABI 39 on `ssh macmini` (Apple M4 Pro). All four selected CTest
cases passed: compiler vascular, Metal vascular, Human binding, and the
vascular cavity owner suite.

The same run retains the prior synthetic engineering gates:

```text
blood_mass_owner=pass ... dynamic_spatial_moments=pass co_moving_inertia=pass
... time_integrated_moment_closure=pass pressure_impulse_closure=pass
atomic_full_momentum_restore=pass pressure_driven_momentum=unqualified
pressure_gradient_reaction=pass ... pressure_driven_fluid_momentum=pass
```

The machine-readable [receipt](media/blood-momentum-restore-20260913/receipt.json)
and its build, native output, CTest output, identity, and SHA-256 files are
retained together. This closes the atomic-restore subgate for the synthetic
registered owner. It does not close the Human organ/vessel gap: the six source
vessel surfaces still have no admitted lumen/tube geometry, cross-sectional
area, calibrated material or density, body-link mechanical owner, tissue-side
exchange law, subject calibration, or held-out physiological validation.
