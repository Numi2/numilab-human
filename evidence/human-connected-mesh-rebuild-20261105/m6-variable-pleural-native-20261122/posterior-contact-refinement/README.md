# Posterior support on the registered M6 body

The experiment tests whether the bed reaction can be reproduced when its contact surface is refined, and whether a better starting traction distribution improves released motion. It uses the dense skin/fat material and the same gravity, rigid plane, finite elastic attachments and variable pleural pressure as the [conserved material control](../dense-supported-equilibrium/bonded-report.md).

## Matched physical configurations

The reconstruction covers every registered outer triangle with a corner in the plane barrier at either of the two measured configurations. The 366 selected triangles have 1,515 incident volume cells in their complete vertex neighborhoods. Each bisection splits the entire incident skin and volume edge star. Children inherit the parent material and source face or tetrahedron. Their reference volumes partition the parent mass. Midpoint coordinates use the native Float32 emission policy. A bisection that fails cell quality in any retained configuration is skipped.

| Independent nodes added | Bed reaction at fixed starting coordinates | Bed reaction at fixed, previously captured two-step coordinates |
|---:|---:|---:|
| 0 | 953.18 N | 2,245.76 N |
| 582 | 672.65 N | 1,598.45 N |
| 2,842 | 524.79 N | 1,292.11 N |
| 11,746 | 461.24 N | 1,152.49 N |

At the same 952.86 N applied body weight, the last refinement still changes the initial center of pressure by 8.09 mm and the initial source-node force distribution by 11.28%. At the previously captured coordinates the changes are 4.21 mm and 6.98%. The current vertex contact rule fails this spatial convergence test. The guarded fine mesh serves as a construction target with its measured uncertainty. No time integration used the refined mesh.

## Installed physical experiment

The candidate changes the initial normal coordinates of 115 bed-facing nodes on the existing dense material by at most 1.095 µm. The fit jointly constrains the initial bed wrench, donor mass moments, local material forces and reach of the stretch bounds under Float32 emission. An unconstrained fit exceeded the stretch bound in one cell. The final emitted candidate passed full material admission on 1,448,799 cells. All attachments and the reference, connectivity and mass data have their original identities. No extra force is applied during the run.

The fitting code extracts the normal-normal block of the material tangent directly. Its size at these measurements is 136,160 bytes, compared with 4,744,440 bytes for the complete array. Against the existing material tangent at both recorded configurations, relative error is below 2.0e-16. This reduction applies to preparing the starting pose.

The exact motion results, timing and failure receipts are reported alongside this experiment. Promotion requires a sustained supported load, anatomical clearance, physiological qualification and end-to-end timing on the same native configuration.

## Native release decision

Keep the conserved control as the dense baseline. Both the candidate and the control advanced two 15.625 µs steps and rejected the third. The accepted bed reaction moved from 2,252.30 N to 2,252.14 N against 952.86 N of body weight. Vertical FEM force on the articulated body moved from 17.854 N to 17.862 N against an articulated gravitational load of 811.45 N. All physical contact and the coupled acceptance checks ran.

An independent replay of the two accepted steps reproduced the full trial's accepted force samples. The endpoint has zero inverted or low-quality skin/fat cells, zero skin self-intersections, zero inter-layer crossing pairs and zero skin/bone pairs. It has 10,895 nonocular skin/tissue crossing pairs across the audited anatomy. Thirty-seven dermal cells and sixteen fat cells exceed the registration stretch envelope. Three dermal cells exceed the severe stretch threshold. The recorded pleural pressure is −490.33 Pa throughout this short horizon.

The dense candidate achieved 4.87e-7× for the body horizon (64.14 s wall time) and 1.49e-7× end to end (210.43 s wall time). The third-step residual remained 0.03632484, and the accepted common coordinates and respiratory status survived rollback. Two adipose tetrahedra limited the initial trial correction. A subsequent deformable-contact limit reduced the final admissible fraction from 7.63e-6 to 3.77e-11. These are different stages of the same Newton iteration. Reporting only the material bound would miss most of the restriction. Further submicron skin preload tuning has no demonstrated benefit for this failure.

[Matched physical measurements](motion-comparison.json) · [Native bed loads](native-bed.json) · [Matched-weight convergence](matched-weight.json) · [FEM admission](emitted-material.json)

The sparse Metal observer reproduced every per-iteration residual and line-search fraction, all accepted force samples and the same failure. On root 2, Newton iteration 15, contact slot 1,223,979 in skin/fat object 15 produced the final bound through its barrier trial. Incoming material fraction was 7.62939453e-6. The contact limited it to 3.7724622e-11. The independent vascular residual was 2.94e-8. Full source, compiler and replay identities, material and mass data, captures and rejected fits are recoverable from the [independently restored 96-file archive](recovery.json). The compact contact witness and per-stage fractions are recorded in [the observer comparison](contact-decision.json).

The subsequent [M6 displacement-update experiment](reference-update.md) identifies the limiting registered contact, tests paired displacement on the GPU and checks the independently replayed physical endpoint. It also resolves the 10,895 physical crossing pairs by tissue and source muscle identity.

The [M6 reference-area bed operator](face-contact/README.md) is the tested replacement candidate for the mesh-dependent stationary normal response. Its matched-weight reaction distribution and consistent tangent converge on the registered posterior surface. Coupled integration and loaded tissue-to-skeleton equilibrium remain open.
