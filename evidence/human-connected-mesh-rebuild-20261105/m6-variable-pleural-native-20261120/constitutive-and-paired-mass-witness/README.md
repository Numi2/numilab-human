# Repaired skin/fat geometry and 44 ms solver witness

The `3171dfb` geometry candidate retains 311,020 nodes and 1,448,804 tetrahedra with zero inverted or below-threshold cells and zero static skin/fat layer crossings in the retained boundary audit. The existing TetGen files round-trip to the exact audited Float32 coordinates, material IDs and ordered connectivity. The facet remap retains 833,108 source-face identities with 119 faces bound to unique oriented replacement tetrahedra. See the geometry and facet ancestry receipts for the emitted hashes and scope.

## Current-reference mass routing

The exact current-frame parent query covered both materials and all 1,448,804 cells. Its regional masks agreed with the original P1 integration and expansion on 516 controls. The retained mass flow contains 4,448 invalid expanded routes to 859 cells, totaling 0.371362 g across 5,336,517 checks. The separately verified carrier quotas and density were unchanged by the audit. Candidate rerouting with fixed element masses and extra typed P1/projection routes left 0.306214 g unassigned. Native mass admission remains open.

## Regional native reproduction

Three Apple GPU trials replayed the same regional scene as `7497c02`, with 2 ms roots, 256 Matter microsteps per root and variable pleural pressure. The accepted output and failure receipts match the control byte-for-byte. The trials accepted 42 ms and rejected the 44 ms root with byte-identical rollback. The observed run took 344.191 seconds, giving 0.0001220× end-to-end headless throughput. This scene uses the regional mechanics and registered shoulder rows. The fine skin/fat volume has yet to pass native admission.

At the rejecting root and microstep, the contact stage inherited its four reduced trial fractions. At the strongest restriction, the CCD check cleared and the first local trial passed. The upstream backtracking records show that the full first correction would invert pectoral tetrahedron 200 and fractions 1/2 through 1/32 fail material validity. The 1/64 trial passes. The final equilibrium residual witness belongs to the left latissimus fascia. These are separate witnesses. The extension is an opt-in read-only observer built from the pinned runtime. No solve or contact equation changed.

## Remaining acceptance

The repaired volume still needs typed registration, a closed native mass flow and assembly with the native scene. The supported equilibrium check, the 44 ms solve and 16 seconds of zero-crossing motion with variable pleural pressure at 0.1× remain open.

The launch, execution and comparison receipts hold the exact scene and replay configuration. The retained scripts regenerate the larger local mesh inputs and checks. SHA256SUMS covers the published files.
