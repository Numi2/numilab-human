# Conditioned lung source and native pilot

The lung source contained triangles that collapsed after native Float32 deformation. The source owner now repairs bounded slivers with reciprocal interfaces, reconstructs diaphragm patch registration from exact source coordinates, and rejects endpoint/midpoint fan ears. Each declared edge-collapse endpoint displacement and tangent relocation is bounded by 1 micrometer; unrelated anatomy is byte-identical. The respiratory effective area is bit-identical as Float32, and the aggregate lobe volume change is -6.12084e-14 m3. This is inferred source conditioning, not new measured anatomy.

Candidate 897 is bound by retained asset/lineage identities and source-module pins. Its final source has no zero-area lung faces and no lung altitude below 128 nm; 2,870 lung faces remain below 512 nm. The source threshold does not prove validity at every deformed pose. Independent exact geometry checks passed for lungs 305-309, pleura 310, and diaphragm 311 at accepted steps 0, 6111, and 9999. The diaphragm retains 3.04-6.20 nm minimum altitudes in these samples, an explicit residual precision risk.

Native run 900 on the SSH Mac mini completed 10,000 accepted 2 ms steps (20 seconds), 3 complete breaths, and 23 complete filling/ejection cycles. All 10,000 coupled trace rows are byte-identical to the preceding 894-source run 896. End-to-end wall time was 193.546785 s (0.103334x including initialization). This is a correctness pilot using the existing frozen runtime, not a new speed result or the required final endurance study.

The native recording has 315 decoded frames: one black startup frame and 314 nonblank frames matching the surface trace. Timestamps are strictly increasing. The seven inspection samples and full movie remain on the Mini under native-final-lung-source-review-901 and native-final-lung-source-cycle-900.

The canonical shared-atlas skin still requires lower-limb and ocular clearance corrections. This run does not qualify the complete anatomy. An exact captured-geometry audit also identified an older 894-source collapsed face at step 5375 despite that run's native area counter reporting zero; retain both observations. Runtime counter correctness is being investigated separately.

Validation: 58 tests and 22 subtests passed. The repository contains source owners and focused regressions. Full source meshes, receipts, failed attempts, compact traces, and the native movie remain in /Users/n/numi-human-resting-evidence-20261005 with identities retained here. No source asset rights are changed.

Reproduce source conditioning with the existing module numilab_human.resting_lung_edge_repair using --precision-flips, the pinned input_payload_path and input_receipt_path from the full receipt, --respiratory-owner /Users/n/numi-human-resting-resp-source-20261006/src/numilab_human/resting_respiratory_conforming_field.py, and a fresh --output-dir. The exact command is retained in reproduce-source.json.

Native launch argv, environment, device, binaries, shaders, and assets are recorded in invocation.json, run-metadata.json, and run-declaration.json. Choose a fresh output directory when replaying.
