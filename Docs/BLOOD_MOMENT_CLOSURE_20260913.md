# Blood moment and pressure-impulse closure, 13 September 2026

The native blood-owner branch `f89991dc55c591fb6dc5ca5388de4b9034592c5a`
adds a bounded transfer audit to the ABI38 synthetic registered-cavity check.
On the physical Apple M4 Pro, the production `nm_vascular_blood_moments`
kernel was evaluated on consecutive co-moving states. The finite difference of
the first mass moment matches the reported co-moving linear momentum, and the
current mass remains unchanged when volume is unchanged. The same run removes
the separately accounted current-volume mass correction and verifies that a
uniform pressure impulse on the closed cavity surface cancels.

The retained [native log](media/blood-moment-closure-20260913/native.log),
[build log](media/blood-moment-closure-20260913/build.log),
[identity](media/blood-moment-closure-20260913/identity.txt), and
[receipt](media/blood-moment-closure-20260913/receipt.json) bind the source
commit, Apple M4 Pro device, ABI38, executable, Metal library, and source hash.
The output includes:

```text
blood_mass_owner=pass ... dynamic_spatial_moments=pass co_moving_inertia=pass
... time_integrated_moment_closure=pass pressure_impulse_closure=pass
pressure_driven_momentum=unqualified subject_calibration=unqualified
```

The existing 32-step moving-wall, pressure-work/Jacobian, conservation,
bitwise replay, rejected-environment rollback, reset, and invalid-restore
checks also pass in the same executable. The fixture is synthetic: its uniform
pressure has no spatial pressure-gradient state. Consequently this increment
does not qualify pressure-driven fluid momentum, anatomical body-frame
registration, organ mechanics, absolute density/perfusion, or subject
calibration. It closes only the time-integrated co-moving moment and
closed-surface impulse subgates for the registered synthetic owner.
