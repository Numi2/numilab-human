# Physical skin and solver decisions

The completed regional motion case is a functional test of supported cardiopulmonary motion. Whole-body anatomical release failed in both captured poses. The dense scene independently shows that outer-skin crossings exist in the physical material geometry. This evidence uses SSH Numi’s native Metal states and the exact retained triangle predicates.

## Skin and material registration

At the audited two-step dense endpoint (15.625 µs), we tested each of the 878 emitted anatomical meshes against each representation of the external skin. The first audit used the viewer’s native replay geometry. The second substituted the actual accepted outward skin/fat material boundary and retained the same target geometry, poses and predicates. All 76,153 Float32 world-space vertex positions and all 152,261 geometric triangles matched, independent of face order and winding. The exact 11,716 target intersection witnesses also matched. None of these external skin faces intersected itself.

| Target class | Dense physical skin triangle pairs |
| --- | ---: |
| Skeletal surfaces (51004) | 212 |
| Muscle surfaces (51005) | 9,846 |
| 51006 | 76 |
| 51010 | 624 |
| 51011 | 958 |
| Remaining audited classes | 0 |

The auditors flagged 350 of the 624 class-51010 pairs as known ocular-source interfaces. Pair counts describe surface events and do not measure penetration depth or imply 11,716 disjoint lesions. The earlier zero count for crossings among the outer skin, inner skin, deep fat and selected cortical surface has a smaller target set. It did not certify the individual muscle surfaces against the skin.

At the 16-second regional endpoint, 8,869 of 11,531 observed skin-to-structure pairs involved muscle surfaces. That model and the dense scene have different registrations, tissue owners and supported time horizons. The two independent tests identify a common registration problem class. See the [regional evidence and launch command](../m6-supported-motion-traction-20261124/README.md).

## Dense mechanics

The instrumented control reproduced two accepted steps and all recorded residuals, iteration counts, backtracking fractions and failure witnesses. Inside the GPU sparse solver, the exact limiting cell 140978 was selected and its individual projection returned success (code 6). At the end of that same sweep, a read-only probe reconstructed the cell using the written nodal directions. Both the trial determinant and swept material control were already outside the allowed interval. This occurs before the bed-direction kernel or final certificate. The instrumentation altered only the diagnostic suffix and preserved the physical solution.

A separate counterfactual restricted later constraints to the remaining free nodes where earlier rows had passed. The existing mass-orthogonal projection, signed material bounds and full nonlinear acceptance were retained. That variant accepted one 7.8125 µs step and rejected its second with nonvascular residual 0.0355385 after 560 Krylov columns and zero exhausted linear solves. The control accepted two steps and rejected its third at 0.0317014 after 551 columns. The direction restriction regressed accepted time and was excluded. See the [underlying control and witness](../m6-registered-volume-search-order-20261124/README.md).

## Engineering decision

Keep the load-tested shoulder force registration and the fully reproducible regional motion benchmark. Restore anatomical nesting in the mechanical registration that supplies the actual skin, fat and muscle boundaries. Rank connected defects by signed penetration and affected area in addition to exact crossing witnesses. At the dense step failure, the needed numerical property is that the *combined* correction remains admissible to every coupled material constraint and reduces the original nonlinear residual. More individual row projections that can invalidate each other and an increased Krylov budget have both failed their native tests. The fresh model forces, conserved mass mappings and independent physical acceptance remain the qualification authority.

The [face-level evidence archive receipt](material-recovery.json) and [Metal trial archive receipt](direction-recovery.json) include independently verified recovery hashes. Neither experimental solver was promoted.
