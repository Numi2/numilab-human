# Use actual skull crossings in the simultaneous layer fit

The native reference has 702 outer, 2,311 interface and 2,685 deep skull
pairs. The geometry-identical parent split preserves all of them. The
previous interior-edge fit conserved mass but increased the deep count.

At each fit linearization, generate finite-triangle separating planes for
the exact intersecting Float32 shell triangles and closed skull banks.
Give the deep layer higher weight and require measured deep and total
crossing counts to be nonincreasing during the discrete backtrack. Retain
the existing determinants, shape floors, source mass, thickness minimum
and untouched bed constraints. The source-plane residual guides the step.
Exact predicates determine the observed crossing response.

Prediction. A bounded eight-iteration run from the identical interior-edge
control will reduce deep crossings below 2,685 without raising aggregate
crossings or adding external-bank or self intersections. Compare against
the saved untargeted variant. The target gate is still zero, including
full-solid containment and all native checks before installation.

## Guarded variant and continuation

The solid guard classifies against individual closed bones using their exact
indexed point predicates. A trial cannot create a newly contained shell
vertex or an unresolved boundary witness. The layer guard uses the nominal
2.5 mm and 10 mm bounds from the nominal reference, or the initial column
width where it already exceeds that value. The exact width check allows
0.2 μm for Float32 coordinate emission. The adjusted active search restores
affected columns to their previously accepted authoring position, then
rechecks the complete volume, width, mass capacity and contact constraints.
No runtime holding force is added by this authoring process.

Resume from the verified guarded candidate for one more bounded iteration
sequence. Reconstruct the same authoring displacements and preserve the
original source volumes, density limits, paired-cell topology, width
limits, checks and bank identities. The measured test compares exact deep
and total pairs and the number of newly inside pairs for both fits. Reuse the
original certificates while keeping the prior zero-crossing target.
