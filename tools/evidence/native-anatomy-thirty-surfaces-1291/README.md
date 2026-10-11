# Thirty repaired surfaces in the native resting human

Both fibularis brevis surfaces join the [previous 28 repairs](../native-anatomy-twenty-eight-surfaces-1290/README.md) in the existing full-body Metal scene. **Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.**

## Source repair and attachment limits

Each original fibularis brevis has nine exact self-intersection pairs, including a long narrow fold. Existing exact positive-winding boundary extraction removes inverted exterior folds but leaves two point junctions. Those rejected results are retained. The existing vertex-fan splitter separates their indices; four junction vertices then retreat toward their own fan-neighbor centroids by one micrometre, resolving the point contacts without changing their source-derived binding maps. This is an explicit reference correction, not measured anatomy or an accuracy claim.

The completed repair changes source surface area by approximately -0.199% and signed volume by -0.00223%. These figures include boundary extraction; the one-micrometre bound applies only to the subsequent junction separation. All eight exactly one-hot attachment proxies per side are preserved. This is not measured attachment fidelity. Physical routes, forces, mass and physiological equations are unchanged.

Source Float32 closure and exact self-intersection checks pass. Sensitivity checks at 0.1, 1 and 10 micrometres pass all 24 retained early and late poses on both sides; the right side also passes six additional scales through 100 micrometres. One micrometre was selected for native verification.

## Actual native checks

All eight native captures show closed, self-intersection-free repaired surfaces. Changed regions add no crossings against 859 surrounding anatomical structures. Existing crossings elsewhere are not classified or waived by this test. All 858 other surfaces and all 157 accepted body poses match the previous 28-repair run exactly; all 1,000 rows and 60 columns of coupled mechanics/physiology match exactly. The native run spans 8,000 accepted steps and 16 simulated seconds, including settling and a complete breathing cycle.

The package records the launch, exact inputs and revisions, device, timing, coupled trace and continuous native framebuffer recording on the SSH Mac mini. The viewer retains whole-body layers, the existing flat contact support and synchronized measurements. No remote desktop visibility is asserted.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-thirty-surfaces-1291/reproduce.py

The pinned Mini runtime and assets are required. Use --prepare-only to verify inputs without launching. Final neck/back integration, remaining muscle defects, foot registration and unexplained interfaces remain open. The old five-minute baseline/intervention evidence uses older anatomy and does not qualify this final body. Source rights and mixed-source provenance remain in the bound receipts.
