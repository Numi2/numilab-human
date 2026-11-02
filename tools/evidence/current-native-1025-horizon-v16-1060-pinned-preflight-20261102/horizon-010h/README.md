# 010h: sparse accepted-geometry motion sampling

This immutable successor to 010g uses the existing `NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS` selection and the existing `.mrvpack` plus per-frame receipt format. It adds `sampled-three` (default), which captures accepted FEM surface state at step 0, the horizon midpoint, and the terminal step. `quartiles-six` retains the 010g roster for higher spatial-temporal sampling.

Each captured frame is checked by the pinned v17c geometry verifier for exact role-row mapping, closed Skin/Fat shell geometry, paired-I interface winding, inter-role and self-intersections, and intersections with accepted bone geometry. These are checks at the sampled accepted frames; the receipt reports omitted step IDs and does not claim a continuous swept-volume test. The native compact terminal FEM audit still runs over the requested horizon. Full FEM endpoint text dumping remains disabled.

The native pack writer's storage preflight automatically budgets one initial pack plus one copy per selected frame, along with its 32 MiB per-pack margin and 2 GiB reserve. At 1024 steps, `sampled-three` requires four pack copies total versus seven under `quartiles-six`; it reduces duplicate MRV storage while keeping a midpoint geometry check.

No simulation was run while adding this capture option. `test_current_native_1025_horizon_010h.py` exercises profile rosters, exact profile-to-plan/observer/verifier matching, horizon progression, and shell argument wiring without launching native physics.
