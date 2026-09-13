# Native source-limit projection — 2026-09-13

The physical Apple M4 Pro native Human owner now projects source-authored scalar joint limits after the authored support-contact and joint-equality passes. The stand kernel retains every active limit, factors its generalized response against the same mass matrix, and uses a reserved `nv` response-vector slab so a large active limit set cannot be silently dropped or rejected by the contact/equality arena. The static recruitment builder also receives the requested runtime timestep instead of a hard-coded 100 µs value.

The clean native checkout at revision `aeca9c737c706a13ac1a3719b0c1afe92796e3bd` was built as `/Users/n/MetalRobo-blood-mass-build-20260913/bin/metalrobo_numilab_human_myosim_visual_probe` on an Apple M4 Pro. The focused native regression selection passed 3/3:

```
numanx.integration.fullbody_bridge_prepared_root
numanx.integration.fullbody_authored_world
numanx.integration.fullbody_vascular_admission
```

The exact source-bound fixture uses one adult full-body source, all 416 source muscle routes at activation `1.0`, NHTENDON3 transfer, 51 joint equalities, six authored support contacts, no root assistance, and 64 steps at `12.5 µs` for `0.8 ms`. The native transaction passes one-step FP64 parity, completes all 64 steps, leaves stderr empty, and reproduces bitwise on the deterministic replay. The receipt, immutable native logs, and source patch are in [`Docs/media/exact-stand-limits-20260913/`](media/exact-stand-limits-20260913/).

The corrected horizon reports:

- `persistent_max_acceleration=1715.86254883 m/s²`;
- `muscle_step_max_velocity_delta=0.120149672031 rad/s`;
- `muscle_step_max_configuration_delta=4.84237643832e-05`;
- `persistent_max_penetration_m=2.93606206014e-07 m`;
- `compiled_stand_max_root_force_residual=1.87160674159e-06 N`;
- `compiled_stand_active_limits=28` and `compiled_stand_active_support_contacts=6`;
- `stand_deterministic_replay=bitwise`.

This is a bounded native mechanics improvement. `compiled_stand_balanced=false` remains, the normalized static residual remains `0.86077456182`, and the `0.8 ms` horizon is not sustained standing, recovery, walking, anatomical tissue loading, activation calibration, material calibration, or blood/tissue qualification. Those gates remain open in the completion ledger.
